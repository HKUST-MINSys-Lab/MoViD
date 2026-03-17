# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar
from collections import defaultdict
import gc

from configs import constants as _C
from lib.utils import transforms
from lib.utils.utils import AverageMeter, prepare_batch
from lib.eval.eval_utils import (
    compute_accel,
    compute_error_accel,
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
)
from lib.models import build_body_model

logger = logging.getLogger(__name__)


class StreamingTrainer:
    """
    Streaming training class that processes sequences frame by frame while maintaining RNN state.
    """
    def __init__(self, network, device, max_history_frames=10):
        self.network = network
        self.device = device
        self.max_history_frames = max_history_frames
        self.reset_state()
        
    def reset_state(self):
        """Reset all states for a new sequence"""
        self.hidden_states = None
        self.prev_context = None
        self.prev_kp3d = None
        self.prev_output = None
        
    def process_frame(self, x, gt, inits,window_size=50, features=None, **kwargs):
        """
        Process a single frame maintaining state for the next frame.
        """
        # Run streaming inference
        if x.shape[1] == 11:
            print(x.shape)
        output, self.hidden_states, self.prev_context, self.prev_kp3d = self.network.stream_inference(
            x, inits, img_features=features, window_size=window_size,
            hidden_states=self.hidden_states, prev_context=self.prev_context, 
            prev_kp3d=self.prev_kp3d, prev_output=self.prev_output, **kwargs
        )
        
        # Store output for next frame
        self.prev_output = output
        
        # Limit history length to prevent memory growth
        if self.prev_context is not None and self.prev_context.shape[1] > self.max_history_frames:
            self.prev_context = self.prev_context[:, -self.max_history_frames:]
        if self.prev_kp3d is not None and self.prev_kp3d.shape[1] > self.max_history_frames:
            self.prev_kp3d = self.prev_kp3d[:, -self.max_history_frames:]
        
        return output
    
    def clear_cache(self):
        """Clear GPU cache and limit history to free memory"""
        # Clear stored contexts to reduce memory
        if self.prev_context is not None:
            self.prev_context = self.prev_context[:, -5:]  # Keep only last 5 frames
        if self.prev_kp3d is not None:
            self.prev_kp3d = self.prev_kp3d[:, -5:]  # Keep only last 5 frames
        
        torch.cuda.empty_cache()
        gc.collect()


class FrameByFrameTrainer:
    def __init__(self, 
                 data_loaders,
                 network,
                 optimizer,
                 criterion=None,
                 train_stage='syn',
                 start_epoch=0,
                 checkpoint=None,
                 end_epoch=999,
                 lr_scheduler=None,
                 device=None,
                 writer=None,
                 debug=False,
                 resume=False,
                 logdir='output',
                 performance_type='min',
                 summary_iter=1,
                 accumulate_grads=4,  # Number of frames to accumulate gradients
                 ):
        
        self.train_loader, self.valid_loader = data_loaders
        
        # Model and optimizer
        self.network = network
        self.optimizer = optimizer
        
        # Training parameters
        self.train_stage = train_stage
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.resume = resume
        self.logdir = logdir
        self.summary_iter = summary_iter
        self.accumulate_grads = accumulate_grads
        
        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')
        self.summary_loss_keys = ['pose']

        self.evaluation_accumulators = dict.fromkeys(
            ['pred_j3d', 'target_j3d', 'pve'])
        
        self.J_regressor_eval = torch.from_numpy(
            np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
        )[_C.KEYPOINTS.H36M_TO_J14, :].unsqueeze(0).float().to(device)
        
        # Initialize streaming trainer
        self.stream_trainer = StreamingTrainer(network, device, max_history_frames=10)
        
        # Timing statistics
        self.timing_stats = {
            'data_preparation': [],
            'forward_pass': [],
            'loss_computation': [],
            'backward_pass': [],
            'memory_cleanup': []
        }
        
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        if checkpoint is not None:
            self.load_pretrained(checkpoint)

    def _extract_frame_data(self, batch, frame_idx):
        """Extract single frame data from batch"""
        frame_batch = {}
        
        # Extract frame-specific data
        for key, value in batch.items():
            if key == 'res':
                frame_batch[key] = value
            elif key == 'pose_root':
                frame_batch[key] = value[:, frame_idx:frame_idx+2]
            elif isinstance(value, torch.Tensor) and len(value.shape) > 1:
                if value.shape[1] > frame_idx:  # Check if frame exists
                    frame_batch[key] = value[:, frame_idx:frame_idx+1]  # Keep batch dimension
                else:
                    frame_batch[key] = value[:, -1:]  # Use last frame if index out of bounds
            else:
                frame_batch[key] = value
                
        return frame_batch

    def _memory_cleanup(self, frame_idx, cleanup_interval=10):
        """Periodic memory cleanup"""
        if frame_idx % cleanup_interval == 0:
            self.stream_trainer.clear_cache()
            gc.collect()
            torch.cuda.empty_cache()

    def _train_sequence_frame_by_frame(self, batch_idx, batch):
        """Train on a single sequence frame by frame"""
        
        # Reset state for new sequence
        self.stream_trainer.reset_state()
        
        sequence_length = batch['kp2d'].shape[1]  # Assuming shape is [batch, seq_len, ...]
        
        # Initialize sequence losses
        sequence_losses = AverageMeter()
        sequence_kp_2d_loss = AverageMeter()
        sequence_kp_3d_loss = AverageMeter()
        
        accumulated_loss = 0.0
        loss_dict_accumulated = defaultdict(float)
        
        # Store all frame losses for batch backward pass
        frame_losses = []
        
        # Process each frame in the sequence  
        for frame_idx in range(sequence_length):
            # 1. Data preparation
            data_start = time.time()
            frame_batch = self._extract_frame_data(batch, frame_idx)
            x, inits, features, kwargs, gt = prepare_batch(
                frame_batch, self.device, self.train_stage=='stage2'
            )
            data_time = time.time() - data_start
            self.timing_stats['data_preparation'].append(data_time)
            
            # 2. Forward pass
            forward_start = time.time()
            window_size=1
            pred = self.stream_trainer.process_frame(x, gt, inits,window_size, features, **kwargs)
            
            forward_time = time.time() - forward_start
            self.timing_stats['forward_pass'].append(forward_time)
            
            # 3. Loss computation
            loss_start = time.time()
            loss, loss_dict = self.criterion(pred, gt)
            #loss += pred['contrastive_loss'] + pred['ortho_loss']  # + pred['imu_loss']
            
            # Scale loss by total sequence length for proper averaging
            loss = loss / sequence_length
            loss_time = time.time() - loss_start
            self.timing_stats['loss_computation'].append(loss_time)
            
            # Store loss for later backward pass
            frame_losses.append(loss)
            
            # Accumulate losses for logging
            accumulated_loss += loss.item() * sequence_length  # Unscale for logging
            for k, v in loss_dict.items():
                loss_dict_accumulated[k] += v.item()
            
            # 4. Memory cleanup (but keep essential variables)
            cleanup_start = time.time()
            self._memory_cleanup(frame_idx)
            
            # Clean up frame-level variables (but keep loss)
            del x, inits, features, kwargs, gt, pred
            torch.cuda.empty_cache()
            
            cleanup_time = time.time() - cleanup_start
            self.timing_stats['memory_cleanup'].append(cleanup_time)
        
        # 5. Backward pass on all accumulated losses
        backward_start = time.time()
        
        # Sum all frame losses and perform backward pass
        total_loss = frame_losses[-1]
        total_loss.backward()
        
        backward_time = time.time() - backward_start
        self.timing_stats['backward_pass'].append(backward_time)
        
        # 6. Update weights after processing entire sequence
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Average accumulated losses for logging
        avg_loss = accumulated_loss / sequence_length
        avg_loss_dict = {k: v / sequence_length for k, v in loss_dict_accumulated.items()}
        
        # Update meters
        if not torch.isnan(torch.tensor(avg_loss)):
            sequence_losses.update(avg_loss, sequence_length)
            sequence_kp_2d_loss.update(avg_loss_dict['2d'], sequence_length)
            sequence_kp_3d_loss.update(avg_loss_dict['3d'], sequence_length)
        
        # Clean up stored losses
        del frame_losses, total_loss
        torch.cuda.empty_cache()
        
        return sequence_losses, sequence_kp_2d_loss, sequence_kp_3d_loss

    def train(self):
        """Single epoch training routine with frame-by-frame processing"""
        
        losses = AverageMeter()
        kp_2d_loss = AverageMeter()
        kp_3d_loss = AverageMeter()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }
        
        self.network.train()
        start = time.time()
        summary_string = ''
        
        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=len(self.train_loader))
        
        for i, batch in enumerate(self.train_loader):
            if batch is None:
                continue
                
            batch_start = time.time()
            
            # Process sequence frame by frame
            seq_losses, seq_kp_2d_loss, seq_kp_3d_loss = self._train_sequence_frame_by_frame(i, batch)
            
            # Update global meters
            losses.update(seq_losses.avg, seq_losses.count)
            kp_2d_loss.update(seq_kp_2d_loss.avg, seq_kp_2d_loss.count)
            kp_3d_loss.update(seq_kp_3d_loss.avg, seq_kp_3d_loss.count)
            
            batch_time = time.time() - batch_start
            timer['batch'] = batch_time
            
            # Logging
            summary_string = f'({i + 1}/{len(self.train_loader)}) | Total: {bar.elapsed_td} ' \
                            f'| loss: {losses.avg:.2f} | 2d: {kp_2d_loss.avg:.2f} ' \
                            f'| 3d: {kp_3d_loss.avg:.2f} | Time: {batch_time:.2f}s'

            if (i + 1) % self.summary_iter == 0:
                self.writer.add_scalar('train_loss/loss', losses.avg, global_step=self.train_global_step)
                self.writer.add_scalar('train_loss/2d', kp_2d_loss.avg, global_step=self.train_global_step)
                self.writer.add_scalar('train_loss/3d', kp_3d_loss.avg, global_step=self.train_global_step)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next(1)
            
            # Print timing statistics periodically
            if i % 10 == 0 and len(self.timing_stats['forward_pass']) > 0:
                self._print_timing_stats()
                
        logger.info(summary_string)
        bar.finish()

    def _print_timing_stats(self):
        """Print timing statistics"""
        if len(self.timing_stats['forward_pass']) > 10:  # Only print if we have enough samples
            avg_times = {}
            for key in self.timing_stats:
                if len(self.timing_stats[key]) > 0:
                    avg_times[key] = np.mean(self.timing_stats[key][-10:])  # Average of last 10
            
            logger.info("Frame-by-frame Timing Statistics:")
            for key, val in avg_times.items():
                logger.info(f"{key:20s}: {val*1000:.1f}ms")

    def validate(self):
        """Validation with frame-by-frame processing"""
        self.network.eval()
        
        start = time.time()
        summary_string = ''
        bar = Bar('Validation', fill='#', max=len(self.valid_loader))

        if self.evaluation_accumulators is not None:
            for k, v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.valid_loader):
                if batch is None:
                    continue
                    
                # Reset state for new sequence
                self.stream_trainer.reset_state()
                
                sequence_length = batch['kp2d'].shape[1]
                
                # Process sequence frame by frame for validation
                for frame_idx in range(sequence_length):
                    frame_batch = self._extract_frame_data(batch, frame_idx)
                    x, inits, features, kwargs, gt = prepare_batch(
                        frame_batch, self.device, self.train_stage=='stage2'
                    )
                    
                    if x[0].shape[0] != features[0].shape[0]:
                        continue
                    window_size = 1
                    # Forward pass
                    pred = self.stream_trainer.process_frame(x, gt, inits, window_size, features, **kwargs)
                    
                    # Build SMPL for evaluation (only for last frame or periodically)
                    if frame_idx == sequence_length - 1:  # Only evaluate on last frame
                        smpl = build_body_model(self.device, batch_size=len(pred['verts_cam']), gender=batch['gender'][0])
                        gt_output = smpl.get_output(
                            body_pose=transforms.rotation_6d_to_matrix(gt['pose'][0, :, 1:]),
                            global_orient=transforms.rotation_6d_to_matrix(gt['pose'][0, :, :1]),
                            betas=gt['betas'][0],
                            pose2rot=False
                        )
                        
                        pred_j3d = torch.matmul(self.J_regressor_eval, pred['verts_cam']).cpu()
                        target_j3d = torch.matmul(self.J_regressor_eval, gt_output.vertices).cpu()
                        pred_verts = pred['verts_cam'].cpu()
                        target_verts = gt_output.vertices.cpu()
                        
                        pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
                            [pred_j3d, target_j3d, pred_verts, target_verts], [2, 3]
                        )
                        
                        self.evaluation_accumulators['pred_j3d'].append(pred_j3d.numpy())
                        self.evaluation_accumulators['target_j3d'].append(target_j3d.numpy())
                        pve = np.sqrt(np.sum((target_verts.numpy() - pred_verts.numpy()) ** 2, axis=-1)).mean(-1) * 1e3
                        self.evaluation_accumulators['pve'].append(pve[:, None])
                    
                    # Clean up
                    del x, inits, features, kwargs, gt, pred
                    if frame_idx == sequence_length - 1:
                        del smpl, gt_output
                    torch.cuda.empty_cache()
            
                batch_time = time.time() - start
                summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                                f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

                self.valid_global_step += 1
                bar.suffix = summary_string
                bar.next()

        logger.info(summary_string)
        bar.finish()
    
    def evaluate(self):
        """Evaluation method (unchanged)"""
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        m2mm = 1000
        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm
        
        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'accel': accel,
            'accel_err': accel_err
        }
        
        if 'pred_verts' in self.evaluation_accumulators.keys():
            eval_dict.update({'pve': self.evaluation_accumulators['pve'].mean()})

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        logger.info(log_str)

        for k, v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return pa_mpjpe
    
    def save_model(self, performance, epoch):
        """Save model method (unchanged)"""
        save_dict = {
            'epoch': epoch,
            'model': self.network.state_dict(),
            'performance': performance,
            'optimizer': self.optimizer.state_dict(),
        }

        filename = osp.join(self.logdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best:
            logger.info('Best performance achieved, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename, osp.join(self.logdir, 'model_best.pth.tar'))

            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))

    def fit(self):
        """Main training loop"""
        self.validate()
        performance = self.evaluate()
        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train()
            self.validate()
            performance = self.evaluate()

            self.criterion.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # log the learning rate
            for param_group in self.optimizer.param_groups[:4]:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr', param_group['lr'], global_step=self.epoch)

            logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')

            self.save_model(performance, epoch)
            self.train_loader.dataset.prepare_video_batch()

        self.writer.close()
        
    def load_pretrained(self, model_path):
        """Load pretrained model method (unchanged)"""
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path)

            # network
            ignore_keys = ['smpl.body_pose', 'smpl.betas', 'smpl.global_orient', 'smpl.J_regressor_extra', 'smpl.J_regressor_eval']
            model_state_dict = {k: v for k, v in checkpoint['model'].items() if k not in ignore_keys}
            model_state_dict = {k: v for k, v in model_state_dict.items() if k in self.network.state_dict().keys()}

            self.network.load_state_dict(model_state_dict, strict=False)
            
            if self.resume:
                self.start_epoch = checkpoint['epoch']
                self.best_performance = checkpoint['performance']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            logger.info(f"=> loaded checkpoint '{model_path}' "
                  f"(epoch {self.start_epoch}, performance {self.best_performance})")
        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")