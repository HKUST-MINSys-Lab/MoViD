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
import os
import os.path as osp
from progress.bar import Bar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def visualize_alignment(pred_verts, gt_verts, pred_joints, gt_joints, 
                       frame_idx=0, pelvis_idxs=[2, 3], output_path='alignment_check.png',
                       vid_name='sequence', metrics=None):
    """
    Visualize predicted and GT alignment in 3D
    Shows: Original, Pelvis-aligned, and Procrustes-aligned views
    
    Args:
        pred_verts: (N, V, 3) predicted vertices (numpy array)
        gt_verts: (N, V, 3) ground truth vertices (numpy array)
        pred_joints: (N, J, 3) predicted joints (numpy array)
        gt_joints: (N, J, 3) ground truth joints (numpy array)
        frame_idx: frame index to visualize
        pelvis_idxs: pelvis joint indices
        output_path: path to save the visualization
        vid_name: video/sequence name for title
        metrics: dict with mpjpe, pa_mpjpe, pve values
    """
    fig = plt.figure(figsize=(18, 5))
    
    # Sample vertices for faster visualization (every 10th vertex)
    sample_rate = 10
    
    # Extract single frame
    pred_v = pred_verts[frame_idx]
    gt_v = gt_verts[frame_idx]
    pred_j = pred_joints[frame_idx]
    gt_j = gt_joints[frame_idx]
    
    # 1. Original coordinates
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(pred_v[::sample_rate, 0], pred_v[::sample_rate, 1], pred_v[::sample_rate, 2], 
                c='red', marker='o', s=1, alpha=0.6, label='Pred')
    ax1.scatter(gt_v[::sample_rate, 0], gt_v[::sample_rate, 1], gt_v[::sample_rate, 2], 
                c='blue', marker='^', s=1, alpha=0.6, label='GT')
    
    # Plot joints
    ax1.scatter(pred_j[:, 0], pred_j[:, 1], pred_j[:, 2], 
                c='darkred', marker='o', s=30, alpha=0.8, label='Pred Joints')
    ax1.scatter(gt_j[:, 0], gt_j[:, 1], gt_j[:, 2], 
                c='darkblue', marker='^', s=30, alpha=0.8, label='GT Joints')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Coordinates')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.view_init(elev=20, azim=45)
    
    # 2. Pelvis-aligned
    pred_pelvis = pred_j[pelvis_idxs, :].mean(axis=0, keepdims=True)
    gt_pelvis = gt_j[pelvis_idxs, :].mean(axis=0, keepdims=True)
    
    pred_v_aligned = pred_v - pred_pelvis
    gt_v_aligned = gt_v - gt_pelvis
    pred_j_aligned = pred_j - pred_pelvis
    gt_j_aligned = gt_j - gt_pelvis
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pred_v_aligned[::sample_rate, 0], pred_v_aligned[::sample_rate, 1], pred_v_aligned[::sample_rate, 2], 
                c='red', marker='o', s=1, alpha=0.6, label='Pred')
    ax2.scatter(gt_v_aligned[::sample_rate, 0], gt_v_aligned[::sample_rate, 1], gt_v_aligned[::sample_rate, 2], 
                c='blue', marker='^', s=1, alpha=0.6, label='GT')
    
    ax2.scatter(pred_j_aligned[:, 0], pred_j_aligned[:, 1], pred_j_aligned[:, 2], 
                c='darkred', marker='o', s=30, alpha=0.8)
    ax2.scatter(gt_j_aligned[:, 0], gt_j_aligned[:, 1], gt_j_aligned[:, 2], 
                c='darkblue', marker='^', s=30, alpha=0.8)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Pelvis-Aligned')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.view_init(elev=20, azim=45)
    
    # 3. Procrustes-aligned
    # Compute Procrustes transform using joints
    pred_j_aligned_T = pred_j_aligned.T
    gt_j_aligned_T = gt_j_aligned.T
    
    # Center
    mu1 = pred_j_aligned_T.mean(axis=1, keepdims=True)
    mu2 = gt_j_aligned_T.mean(axis=1, keepdims=True)
    X1 = pred_j_aligned_T - mu1
    X2 = gt_j_aligned_T - mu2
    
    # Compute scale and rotation
    var1 = np.sum(X1**2)
    K = X1.dot(X2.T)
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R_mat = V.dot(Z.dot(U.T))
    scale = np.trace(R_mat.dot(K)) / var1
    t = mu2 - scale * (R_mat.dot(mu1))
    
    # Apply transform to vertices
    pred_v_centered = pred_v_aligned.T - mu1
    pred_v_proc = scale * R_mat.dot(pred_v_centered) + t
    pred_v_proc = pred_v_proc.T
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(pred_v_proc[::sample_rate, 0], pred_v_proc[::sample_rate, 1], pred_v_proc[::sample_rate, 2], 
                c='red', marker='o', s=1, alpha=0.6, label='Pred (aligned)')
    ax3.scatter(gt_v_aligned[::sample_rate, 0], gt_v_aligned[::sample_rate, 1], gt_v_aligned[::sample_rate, 2], 
                c='blue', marker='^', s=1, alpha=0.6, label='GT')
    
    # Apply transform to joints for visualization
    pred_j_centered = pred_j_aligned.T - mu1
    pred_j_proc = scale * R_mat.dot(pred_j_centered) + t
    pred_j_proc = pred_j_proc.T
    
    ax3.scatter(pred_j_proc[:, 0], pred_j_proc[:, 1], pred_j_proc[:, 2], 
                c='darkred', marker='o', s=30, alpha=0.8)
    ax3.scatter(gt_j_aligned[:, 0], gt_j_aligned[:, 1], gt_j_aligned[:, 2], 
                c='darkblue', marker='^', s=30, alpha=0.8)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Procrustes-Aligned')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.view_init(elev=20, azim=45)
    
    # Add overall title with metrics if provided
    if metrics is not None:
        title = f'{vid_name} - Frame {frame_idx} | '
        title += f'MPJPE: {metrics.get("mpjpe", 0):.1f}mm | '
        title += f'PA-MPJPE: {metrics.get("pa_mpjpe", 0):.1f}mm | '
        title += f'PVE: {metrics.get("pve", 0):.1f}mm'
    else:
        title = f'{vid_name} - Frame {frame_idx}'
    
    fig.suptitle(title, fontsize=14, y=1.00)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  📊 Saved visualization to {output_path}")


class Trainer():
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
                 viz_enabled=False,
                 viz_num_samples=3,
                 viz_frames_per_sample=3,
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
        
        # Visualization parameters
        self.viz_enabled = viz_enabled
        self.viz_num_samples = viz_num_samples
        self.viz_frames_per_sample = viz_frames_per_sample
        
        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')
        self.summary_loss_keys = ['pose']

        self.evaluation_accumulators = dict.fromkeys(
            ['pred_j3d', 'target_j3d', 'pve'])
        
        # For visualization
        self.viz_accumulators = dict.fromkeys(
            ['pred_verts', 'target_verts', 'pred_j3d', 'target_j3d', 
             'mpjpe', 'pa_mpjpe', 'pve', 'batch_idx']
        )
        
        self.J_regressor_eval = torch.from_numpy(
            np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
        )[_C.KEYPOINTS.H36M_TO_J14, :].unsqueeze(0).float().to(device)
        
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create visualization directory
        if self.viz_enabled:
            self.viz_dir = osp.join(self.logdir, 'validation_viz')
            os.makedirs(self.viz_dir, exist_ok=True)
            logger.info(f"Validation visualizations will be saved to: {self.viz_dir}")
            
        if checkpoint is not None:
            self.load_pretrained(checkpoint)
        
    def train(self, ):
        # Single epoch training routine

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
            # <======= Feedforward 
            x, inits, features, kwargs, gt = prepare_batch(batch, self.device, self.train_stage=='stage2')
            timer['data'] = time.time() - start
            start = time.time()
            
            pred = self.network(x, gt, inits, features, **kwargs)
            timer['forward'] = time.time() - start
            start = time.time()
            # =======>

            # <======= Backprop            
            loss, loss_dict = self.criterion(pred, gt)
            if 'contrastive_loss' in pred:
                loss += pred['contrastive_loss'] + pred['ortho_loss']
            timer['loss'] = time.time() - start
            start = time.time()
            
            # Clip gradients
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            # =======>
            
            # <======= Log training info
            total_loss = loss
            if torch.isnan(total_loss):
                continue

            losses.update(total_loss.item(), x.size(0))
            kp_2d_loss.update(loss_dict['2d'].item(), x.size(0))
            kp_3d_loss.update(loss_dict['3d'].item(), x.size(0))
            
            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{len(self.train_loader)}) | Total: {bar.elapsed_td} ' \
                            f'| loss: {losses.avg:.2f} | 2d: {kp_2d_loss.avg:.2f} ' \
                            f'| 3d: {kp_3d_loss.avg:.2f} '

            for k, v in loss_dict.items():
                if k in self.summary_loss_keys: 
                    summary_string += f' | {k}: {v:.2f}'
                if (i + 1) % self.summary_iter == 0:
                    self.writer.add_scalar('train_loss/'+k, v, global_step=self.train_global_step)

            if (i + 1) % self.summary_iter == 0:
                self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next(1)
                
            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>

        logger.info(summary_string)
        bar.finish()

    def validate(self):
        self.network.eval()
        
        start = time.time()
        summary_string = ''
        bar = Bar('Validation', fill='#', max=len(self.valid_loader))

        # Reset evaluation accumulators
        if self.evaluation_accumulators is not None:
            for k, v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []
        
        # Reset visualization accumulators
        if self.viz_enabled:
            for k in self.viz_accumulators.keys():
                self.viz_accumulators[k] = []
       
        with torch.no_grad():
            for i, batch in enumerate(self.valid_loader):
                if batch is None:
                    continue
                x, inits, features, kwargs, gt = prepare_batch(batch, self.device, self.train_stage=='stage2')
                
                # <======= Feedforward 
                pred = self.network(x, gt, inits, features, **kwargs)
                
                # 3DPW dataset has groundtruth vertices
                # NOTE: Following SPIN, we compute PVE against ground truth from Gendered SMPL mesh
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
                
                # Store for visualization before alignment (if needed for first N samples)
                if self.viz_enabled and i < self.viz_num_samples:
                    # Compute metrics for this batch
                    pred_j3d_aligned, target_j3d_aligned, pred_verts_aligned, target_verts_aligned = batch_align_by_pelvis(
                        [pred_j3d.clone(), target_j3d.clone(), pred_verts.clone(), target_verts.clone()], [2, 3]
                    )
                    
                    # MPJPE
                    mpjpe = torch.sqrt(((pred_j3d_aligned - target_j3d_aligned) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * 1000
                    
                    # PA-MPJPE
                    S1_hat = batch_compute_similarity_transform_torch(pred_j3d_aligned, target_j3d_aligned)
                    pa_mpjpe = torch.sqrt(((S1_hat - target_j3d_aligned) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * 1000
                    
                    # PVE
                    pve = np.sqrt(np.sum((target_verts_aligned.numpy() - pred_verts_aligned.numpy()) ** 2, axis=-1)).mean(-1) * 1000
                    
                    # Store original (non-aligned) data for visualization
                    self.viz_accumulators['pred_verts'].append(pred_verts.numpy())
                    self.viz_accumulators['target_verts'].append(target_verts.numpy())
                    self.viz_accumulators['pred_j3d'].append(pred_j3d.numpy())
                    self.viz_accumulators['target_j3d'].append(target_j3d.numpy())
                    self.viz_accumulators['mpjpe'].append(mpjpe)
                    self.viz_accumulators['pa_mpjpe'].append(pa_mpjpe)
                    self.viz_accumulators['pve'].append(pve)
                    self.viz_accumulators['batch_idx'].append(i)
                
                # Standard alignment for evaluation
                pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
                    [pred_j3d, target_j3d, pred_verts, target_verts], [2, 3]
                )
                
                self.evaluation_accumulators['pred_j3d'].append(pred_j3d.numpy())
                self.evaluation_accumulators['target_j3d'].append(target_j3d.numpy())
                pve = np.sqrt(np.sum((target_verts.numpy() - pred_verts.numpy()) ** 2, axis=-1)).mean(-1) * 1e3
                self.evaluation_accumulators['pve'].append(pve[:, None])
                # =======>
            
                batch_time = time.time() - start

                summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                                f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

                self.valid_global_step += 1
                bar.suffix = summary_string
                bar.next()

        logger.info(summary_string)
        bar.finish()
        
        # Generate visualizations after validation
        if self.viz_enabled:
            self._generate_visualizations()
    
    def _generate_visualizations(self):
        """Generate visualizations from accumulated validation data"""
        logger.info("Generating validation visualizations...")
        
        epoch_viz_dir = osp.join(self.viz_dir, f'epoch_{self.epoch:04d}')
        os.makedirs(epoch_viz_dir, exist_ok=True)
        
        pelvis_idxs = [2, 3]
        
        for sample_idx in range(len(self.viz_accumulators['pred_verts'])):
            pred_verts = self.viz_accumulators['pred_verts'][sample_idx]
            target_verts = self.viz_accumulators['target_verts'][sample_idx]
            pred_j3d = self.viz_accumulators['pred_j3d'][sample_idx]
            target_j3d = self.viz_accumulators['target_j3d'][sample_idx]
            
            mpjpe = self.viz_accumulators['mpjpe'][sample_idx]
            pa_mpjpe = self.viz_accumulators['pa_mpjpe'][sample_idx]
            pve = self.viz_accumulators['pve'][sample_idx]
            batch_idx = self.viz_accumulators['batch_idx'][sample_idx]
            
            num_frames = pred_verts.shape[0]
            
            # Select frames to visualize
            if self.viz_frames_per_sample == 1:
                frame_indices = [0]
            elif self.viz_frames_per_sample == 2:
                frame_indices = [0, num_frames - 1]
            elif self.viz_frames_per_sample >= 3:
                # First, middle, last
                frame_indices = [0, num_frames // 2, num_frames - 1]
            else:
                frame_indices = [0]
            
            # Limit to available frames
            frame_indices = [f for f in frame_indices if f < num_frames]
            
            # Generate visualization for each selected frame
            for frame_idx in frame_indices:
                output_path = osp.join(
                    epoch_viz_dir,
                    f'sample{sample_idx:03d}_batch{batch_idx:03d}_frame{frame_idx:04d}.png'
                )
                
                # Prepare metrics for this specific frame
                frame_metrics = {
                    'mpjpe': mpjpe[frame_idx] if frame_idx < len(mpjpe) else 0,
                    'pa_mpjpe': pa_mpjpe[frame_idx] if frame_idx < len(pa_mpjpe) else 0,
                    'pve': pve[frame_idx] if frame_idx < len(pve) else 0,
                }
                
                visualize_alignment(
                    pred_verts, target_verts,
                    pred_j3d, target_j3d,
                    frame_idx=frame_idx,
                    pelvis_idxs=pelvis_idxs,
                    output_path=output_path,
                    vid_name=f'Epoch{self.epoch}_Sample{sample_idx}_Batch{batch_idx}',
                    metrics=frame_metrics
                )
        
        logger.info(f"✅ Saved {len(self.viz_accumulators['pred_verts']) * len(frame_indices)} visualizations to {epoch_viz_dir}")
    
    def evaluate(self, ):
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
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Extract motion_encoder parameters
            model_dict = checkpoint['model']
            motion_encoder_state = {
                k.replace('motion_encoder.', ''): v
                for k, v in model_dict.items()
                if k.startswith('motion_encoder.')
            }

            # Check whether network has motion_encoder
            if hasattr(self.network, 'motion_encoder'):
                missing_keys, unexpected_keys = self.network.motion_encoder.load_state_dict(
                    motion_encoder_state, strict=False
                )
                logger.info(f"=> Loaded motion_encoder from '{model_path}' "
                            f"(missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)})")
            else:
                logger.warning("=> self.network has no attribute 'motion_encoder'")

        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")