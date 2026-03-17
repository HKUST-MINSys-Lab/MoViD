import os
import time
import gc
import os.path as osp
from glob import glob
from collections import defaultdict

import torch
import imageio
import numpy as np
from smplx import SMPL
from loguru import logger
from progress.bar import Bar

from configs import constants as _C
from configs.config import parse_args
from lib.data.dataloader import setup_eval_dataloader
from lib.models import build_network, build_body_model
from lib.eval.eval_utils import (
    compute_error_accel,
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
)
from lib.utils import transforms
from lib.utils.utils import prepare_output_dir
from lib.utils.utils import prepare_batch
from lib.utils.imutils import avg_preds

try:
    from lib.vis.renderer import Renderer
    _render = True
except:
    print("PyTorch3D is not properly installed! Cannot render the SMPL mesh")
    _render = False


class StreamingEvaluator:
    """
    Streaming evaluation class that processes sequences frame by frame while maintaining RNN state.
    """
    def __init__(self, network, device, max_history_frames=10):
        self.network = network
        self.device = device
        self.max_history_frames = max_history_frames
        self.hidden_states = None
        self.prev_context = None
        self.prev_kp3d = None
        self.prev_output = None
        
    def process_frame(self, x, inits, features=None, **kwargs):
        """
        Process a single frame maintaining state for the next frame.
        """
        # Run streaming inference
        output, self.hidden_states, self.prev_context, self.prev_kp3d = self.network.stream_inference(
            x, inits, img_features=features, 
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
    
    def reset_state(self):
        """Reset all states for a new sequence"""
        self.hidden_states = None
        self.prev_context = None
        self.prev_kp3d = None
        self.prev_output = None


class SequentialEvaluationProcessor:
    """Process evaluation sequences in a sequential manner, one frame at a time"""
    def __init__(self, cfg, eval_loader, network, smpl_models, J_regressor_eval, args):
        self.cfg = cfg
        self.eval_loader = eval_loader
        self.network = network
        self.smpl_models = smpl_models
        self.J_regressor_eval = J_regressor_eval
        self.args = args
        self.device = cfg.DEVICE
        self.pelvis_idxs = [2, 3]
        
        # Initialize streaming evaluator
        self.stream_evaluator = StreamingEvaluator(network, cfg.DEVICE, max_history_frames=50)
        
        # Initialize accumulator for results
        self.accumulator = defaultdict(list)
        
        # Add timing statistics
        self.timing_stats = {
            'total': [],
            'data_preparation': [],
            'inference': [],
            'smpl_forward': [],
            'metrics_computation': [],
            'visualization': [],
            'memory_cleanup': []
        }
    
    def _print_timing_stats(self, seq_idx):
        """Print timing statistics"""
        if seq_idx % 5 == 0 and len(self.timing_stats['total']) > 0:  # Every 5 sequences
            avg_times = {}
            for key in self.timing_stats:
                if len(self.timing_stats[key]) > 0:
                    avg_times[key] = self.timing_stats[key][-1]
            
            logger.info(f"\nSequence {seq_idx} Timing Statistics:")
            for key, val in avg_times.items():
                logger.info(f"{key:20s}: {val*1000:.1f}ms")
            logger.info("")
    
    def _memory_cleanup(self, frame_idx, cleanup_interval=10):
        """Periodic memory cleanup"""
        if frame_idx % cleanup_interval == 0:
            self.stream_evaluator.clear_cache()
            gc.collect()
            torch.cuda.empty_cache()
    
    def _process_sequence_frame_by_frame(self, seq_idx):
        """Process a single sequence frame by frame"""
        sequence_start_time = time.time()
        
        # Reset state for new sequence
        self.stream_evaluator.reset_state()
        
        # Get batch data
        batch = self.eval_loader.dataset.load_data(seq_idx, False)
        sequence_length = batch['kp2d'].shape[1]  # Assuming shape is [batch, seq_len, ...]
        
        # Initialize sequence-level storage
        pred_poses_all = []
        pred_betas_all = []
        pred_verts_all = []
        pred_j3d_all = []
        target_j3d_all = []
        target_verts_all = []
        
        # Process each frame in the sequence
        for frame_idx in range(sequence_length):
            frame_start_time = time.time()
            
            # 1. Data preparation
            data_start = time.time()
            
            # Extract single frame data
            frame_batch = self._extract_frame_data(batch, frame_idx)
            x, inits, features, kwargs, gt = prepare_batch(
                frame_batch, self.cfg.DEVICE, 
                self.cfg.TRAIN.STAGE=='stage2' or self.cfg.TRAIN.STAGE=='stage3'
            )
            
            data_time = time.time() - data_start
            self.timing_stats['data_preparation'].append(data_time)
            
            # 2. Inference
            infer_start = time.time()
            with torch.no_grad():
                if self.cfg.FLIP_EVAL:
                    flipped_frame_batch = self._extract_frame_data(batch, frame_idx, flipped=True)
                    f_x, f_inits, f_features, f_kwargs, _ = prepare_batch(
                        flipped_frame_batch, self.cfg.DEVICE, 
                        self.cfg.TRAIN.STAGE=='stage2' or self.cfg.TRAIN.STAGE=='stage3'
                    )
                    
                    # Forward pass with flipped input
                    flipped_pred = self.stream_evaluator.process_frame(f_x, f_inits, f_features, **f_kwargs)
                
                # Forward pass with normal input
                pred = self.stream_evaluator.process_frame(x, inits, features, **kwargs)
                
                if self.cfg.FLIP_EVAL:
                    # Merge two predictions
                    flipped_pose = flipped_pred['pose'].squeeze(0)
                    pose = pred['pose'].squeeze(0)
                    flipped_shape = flipped_pred['betas'].squeeze(0)
                    shape = pred['betas'].squeeze(0)
                    
                    flipped_pose = flipped_pose.reshape(-1, 24, 6)
                    pose = pose.reshape(-1, 24, 6)
                    avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                    
                    # Update prediction with averaged results
                    pred['pose'] = avg_pose.reshape(-1, 144)
                    pred['betas'] = avg_shape
            
            infer_time = time.time() - infer_start
            self.timing_stats['inference'].append(infer_time)
            
            # 3. SMPL forward pass
            smpl_start = time.time()
            with torch.no_grad():
                # Build predicted SMPL
                pred_output = self.smpl_models['neutral'](
                    body_pose=pred['poses_body'][[-1]], 
                    global_orient=pred['poses_root_cam'][[-1]], 
                    betas=pred['betas'].squeeze(0)[[-1]], 
                    pose2rot=False
                )
                pred_verts = pred_output.vertices.cpu()
                pred_j3d = torch.matmul(self.J_regressor_eval, pred_output.vertices).cpu()
                
                # Build groundtruth SMPL
                target_output = self.smpl_models[batch['gender']](
                    body_pose=transforms.rotation_6d_to_matrix(gt['pose'][0, :, 1:]),
                    global_orient=transforms.rotation_6d_to_matrix(gt['pose'][0, :, :1]),
                    betas=gt['betas'][0],
                    pose2rot=False
                )
                target_verts = target_output.vertices.cpu()
                target_j3d = torch.matmul(self.J_regressor_eval, target_output.vertices).cpu()
            
            smpl_time = time.time() - smpl_start  
            self.timing_stats['smpl_forward'].append(smpl_time)
            
            # Store frame results
            pred_poses_all.append(pred['pose'])
            pred_betas_all.append(pred['betas'])
            pred_verts_all.append(pred_verts)
            pred_j3d_all.append(pred_j3d)
            target_j3d_all.append(target_j3d)
            target_verts_all.append(target_verts)
            
            # Memory cleanup
            cleanup_start = time.time()
            self._memory_cleanup(frame_idx)
            cleanup_time = time.time() - cleanup_start
            self.timing_stats['memory_cleanup'].append(cleanup_time)
            
            # Clean up frame-level variables
            del x, inits, features, kwargs, gt, pred, pred_output, target_output
            if self.cfg.FLIP_EVAL:
                del f_x, f_inits, f_features, f_kwargs, flipped_pred
            torch.cuda.empty_cache()
            
            frame_time = time.time() - frame_start_time
            self.timing_stats['total'].append(frame_time)
        
        # 4. Compute sequence-level metrics
        metrics_start = time.time()
        
        # Concatenate all frames
        pred_j3d_seq = torch.cat(pred_j3d_all, dim=0)
        target_j3d_seq = torch.cat(target_j3d_all, dim=0)
        pred_verts_seq = torch.cat(pred_verts_all, dim=0)
        target_verts_seq = torch.cat(target_verts_all, dim=0)
        
        # Compute metrics
        pred_j3d_seq, target_j3d_seq, pred_verts_seq, target_verts_seq = batch_align_by_pelvis(
            [pred_j3d_seq, target_j3d_seq, pred_verts_seq, target_verts_seq], self.pelvis_idxs
        )
        
        S1_hat = batch_compute_similarity_transform_torch(pred_j3d_seq, target_j3d_seq)
        pa_mpjpe = torch.sqrt(((S1_hat - target_j3d_seq) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * 1e3
        mpjpe = torch.sqrt(((pred_j3d_seq - target_j3d_seq) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * 1e3
        pve = torch.sqrt(((pred_verts_seq - target_verts_seq) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * 1e3
        accel = compute_error_accel(joints_pred=pred_j3d_seq, joints_gt=target_j3d_seq)[1:-1]
        accel = accel * (30 ** 2)  # per frame^s to per s^2
        
        metrics_time = time.time() - metrics_start
        self.timing_stats['metrics_computation'].append(metrics_time)
        
        # 5. Optional visualization
        viz_start = time.time()
        if _render and self.args.render:
            self._render_sequence(batch, pred_verts_all)
        viz_time = time.time() - viz_start
        self.timing_stats['visualization'].append(viz_time)
        
        # Accumulate results
        self.accumulator['pa_mpjpe'].append(pa_mpjpe)
        self.accumulator['mpjpe'].append(mpjpe)
        self.accumulator['pve'].append(pve)
        self.accumulator['accel'].append(accel)
        
        sequence_time = time.time() - sequence_start_time
        
        # Print sequence summary
        summary_string = f'{batch["vid"]} | PA-MPJPE: {pa_mpjpe.mean():.1f}   MPJPE: {mpjpe.mean():.1f}   PVE: {pve.mean():.1f} | Time: {sequence_time:.2f}s'
        
        return summary_string
    
    def _extract_frame_data(self, batch, frame_idx, flipped=False):
        """Extract single frame data from batch"""
        frame_batch = {}
        
        # Extract frame-specific data
        for key, value in batch.items():
            if key == 'res':
                frame_batch[key] = value
                continue
            if isinstance(value, torch.Tensor) and len(value.shape) > 1:
                if value.shape[1] > frame_idx:  # Check if frame exists
                    frame_batch[key] = value[:, frame_idx:frame_idx+1]  # Keep batch dimension
                else:
                    frame_batch[key] = value[:, -1:]  # Use last frame if index out of bounds
            else:
                frame_batch[key] = value
        
        if flipped:
            # Apply flipping logic here if needed
            # This depends on your specific flipping implementation
            pass
            
        return frame_batch
    
    def _render_sequence(self, batch, pred_verts_all):
        """Render the sequence (optional visualization)"""
        if not (_render and self.args.render):
            return
            
        # Save path
        viz_pth = osp.join('output', 'visualization')
        os.makedirs(viz_pth, exist_ok=True)
        
        # Build Renderer
        width, height = batch['cam_intrinsics'][0][0, :2, -1].numpy() * 2
        focal_length = batch['cam_intrinsics'][0][0, 0, 0].item()
        renderer = Renderer(width, height, focal_length, self.cfg.DEVICE, self.smpl_models['neutral'].faces)
        
        # Get images and writer
        frame_list = batch['frame_id'][0].numpy()
        imname_list = sorted(glob(osp.join(_C.PATHS.THREEDPW_PTH, 'imageFiles', batch['vid'][:-2], '*.jpg')))
        writer = imageio.get_writer(osp.join(viz_pth, batch['vid'] + '.mp4'), 
                                    mode='I', format='FFMPEG', fps=30, macro_block_size=1)
        
        # Render each frame
        for i, frame in enumerate(frame_list):
            if i < len(pred_verts_all):
                image = imageio.imread(imname_list[frame])
                vertices = pred_verts_all[i]
                image = renderer.render_mesh(vertices, image)
                writer.append_data(image)
        
        writer.close()
    
    def run(self):
        """Run the sequential evaluation"""
        start_total = time.time()
        
        bar = Bar('Sequential Evaluation', fill='#', max=len(self.eval_loader))
        
        for seq_idx in range(len(self.eval_loader)):
            summary_string = self._process_sequence_frame_by_frame(seq_idx)
            
            bar.suffix = summary_string
            bar.next()
            
            # Print timing statistics
            self._print_timing_stats(seq_idx)
        
        bar.finish()
        
        # Compute final results
        for k, v in self.accumulator.items():
            self.accumulator[k] = np.concatenate(v).mean()

        print('')
        log_str = 'Sequential Evaluation on 3DPW, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in self.accumulator.items()])
        logger.info(log_str)
        logger.info(f"Total evaluation time: {time.time() - start_total:.2f}s")
        
        return self.accumulator


m2mm = 1e3

@torch.no_grad()
def main(cfg, args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Dataloaders ========= #
    eval_loader = setup_eval_dataloader(cfg, '3dpw', 'test', cfg.MODEL.BACKBONE)
    logger.info(f'Dataset loaded')
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Build SMPL models with each gender
    smpl_models = {k: SMPL(_C.BMODEL.FLDR, gender=k).to(cfg.DEVICE) for k in ['male', 'female', 'neutral']}
    
    # Load vertices -> joints regression matrix to evaluate
    J_regressor_eval = torch.from_numpy(
        np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
    )[_C.KEYPOINTS.H36M_TO_J14, :].unsqueeze(0).float().to(cfg.DEVICE)
    
    # Create sequential processor
    processor = SequentialEvaluationProcessor(
        cfg, eval_loader, network, smpl_models, J_regressor_eval, args
    )
    
    # Run sequential evaluation
    results = processor.run()
    
    return results


if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)
    cfg = prepare_output_dir(cfg, cfg_file)
    
    main(cfg, args)