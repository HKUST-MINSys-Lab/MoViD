#optimized_stream.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class RingBuffer:
    """Efficient ring buffer that avoids frequent copying"""
    def __init__(self, max_size: int, feat_dim: int, device: str):
        self.buffer = torch.zeros(1, max_size, feat_dim, device=device)
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        self.device = device
    
    def push(self, x: torch.Tensor):
        """Add a new element in O(1)"""
        self.buffer[:, self.ptr] = x.squeeze(1)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def get_sequence(self) -> torch.Tensor:
        """Get the current sequence without unnecessary copying"""
        if self.size < self.max_size:
            return self.buffer[:, :self.size]
        
        if self.ptr == 0:
            return self.buffer  # already aligned; return directly
        
        # Reorder only when necessary
        return torch.cat([
            self.buffer[:, self.ptr:],
            self.buffer[:, :self.ptr]
        ], dim=1)
    
    def clear(self):
        self.ptr = 0
        self.size = 0


class StreamStateManager:
    """Manage streaming-inference state"""
    def __init__(self, window_size: int, device: str, d_embed: int = 128,
                 n_joints: int = 17, view_change_thresh: float = 0.5):
        self.window_size = window_size
        self.device = device
        self.view_change_thresh = view_change_thresh
        self.last_view_img = None
        self.last_view_feat = None
        self.d_embed = d_embed
        self.n_joints = n_joints
        self.kp3d_dim = n_joints * 3  # 17*3 = 51
        self.motion_with_kp_dim = d_embed + self.kp3d_dim  # 128 + 51 = 179

        # Ring buffer
        self.motion_buffer = RingBuffer(window_size, d_embed, device)  # motion_context dimension
        self.kp3d_buffer = RingBuffer(window_size, self.kp3d_dim, device)  # 17*3 keypoints

        # State flags
        self.is_initialized = False
        self.frame_count = 0

        # Preallocated tensor pool
        self.tensor_pool = {
            'motion_with_kp': torch.empty(1, 1, self.motion_with_kp_dim, device=device),
            'integrated_feat': None,
        }
    


    @staticmethod
    def compute_view_diff(x1, x2):
        # Can be replaced with a more complex metric such as histogram or feature differences
        return torch.mean(torch.abs(x1 - x2))

    def get_view_feat(self, x, view_encoder):
        if self.last_view_img is None:
            # Must compute on the first call
            self.last_view_feat = view_encoder(x)
            self.last_view_img = x.clone()
        else:
            diff = self.compute_view_diff(x, self.last_view_img)
            if diff > self.view_change_thresh:
                # View changes are large -> recompute
                self.last_view_feat = view_encoder(x)
                self.last_view_img = x.clone()
            # Otherwise reuse the cache directly
        return self.last_view_feat
    
    def update_buffers(self, motion_context: torch.Tensor, kp3d: torch.Tensor):
        """Update the buffer"""
        self.motion_buffer.push(motion_context)
        self.kp3d_buffer.push(kp3d.reshape(1, 1, -1))
        self.frame_count += 1
    
    def get_windowed_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get window features without copying"""
        motion_seq = self.motion_buffer.get_sequence()
        kp3d_seq = self.kp3d_buffer.get_sequence()
        return motion_seq, kp3d_seq
    
    def reset(self):
        """Reset state"""
        self.motion_buffer.clear()
        self.kp3d_buffer.clear()
        self.is_initialized = False
        self.frame_count = 0


class StreamInference(nn.Module):
    """Optimized streaming inference module"""
    def __init__(self, network, window_size: int = 10, device: str = 'cuda',
                 d_embed: int = 128, n_joints: int = 17):
        super().__init__()
        self.network = network
        self.window_size = window_size
        self.device = device
        self.d_embed = d_embed
        self.n_joints = n_joints
        self.kp3d_dim = n_joints * 3

        # State manager
        self.state_manager = StreamStateManager(
            window_size, device, d_embed=d_embed, n_joints=n_joints
        )
        
        # Performance statistics
        self.stats = {
            'buffer_ops': 0,
            'concat_ops': 0,
            'smpl_calls': 0,
        }
    
    @torch.no_grad()
    def process_frame(self,
                     x: torch.Tensor,
                     inits: Tuple[torch.Tensor, torch.Tensor],
                     img_features: Optional[torch.Tensor] = None,
                     mask: Optional[torch.Tensor] = None,
                     init_root: Optional[torch.Tensor] = None,
                     cam_angvel: Optional[torch.Tensor] = None,
                     hidden_states: Optional[Dict] = None,
                     prev_output: Optional[Dict] = None,
                     **kwargs) -> Tuple[Dict, Dict]:
        """
        Edge single-frame processing - follow the forward pipeline strictly, without using feature/SLAM

        forward pipeline:
        1. motion_encoder -> pred_kp3d, motion_context
        2. cat(motion_context, kp3d) -> trajectory_decoder -> pred_root, pred_vel
        3. clip_proj + clip_gated_fusion -> motion_context (CLIP fusion)
        4. view_encoder(pred_kp3d) -> view_feat
        5. gated_fusion(motion_context, view_feat) -> motion_context
        6. cat(motion_context, kp3d) -> view_decoder -> pred_global_orient, pred_cam
        7. dynamic_projection(motion_context, view_feat) -> motion_context
        8. cat(motion_context, kp3d) -> motion_decoder -> pred_body_pose, pred_shape, pred_contact
        9. cat(pred_global_orient, pred_body_pose) -> pred_pose
        """

        # ===== initialization =====
        if hidden_states is None:
            hidden_states = self._init_hidden_states()
        # Ensure view_decoder key exists
        if 'view_decoder' not in hidden_states:
            hidden_states['view_decoder'] = None

        b = x.shape[0]

        # ===== Step 1: preprocessing =====
        x_current = x[:, -1:] if x.shape[1] > 1 else x
        mask_current = mask[:, -1:] if mask is not None and mask.shape[1] > 1 else mask
        x_processed = self.network.preprocess(x_current, mask_current)

        init_kp, init_smpl = inits

        # ===== Step 2: get previous-frame 3D keypoints =====
        prev_kp3d = self._get_prev_kp3d(prev_output, init_kp, b)

        # ===== Step 3: Motion Encoder =====
        pred_kp3d, motion_context, hidden_states['motion_encoder'] = \
            self.network.motion_encoder.forward_step(
                x_processed,
                prev_kp3d.reshape(b, 1, -1),
                hidden_states['motion_encoder']
            )

        # ===== Step 4: cat(motion_context, kp3d) for trajectory decoder =====
        # Key point: trajectory_decoder uses the original motion_context before CLIP fusion
        motion_with_kp_original = torch.cat([
            motion_context,
            pred_kp3d.reshape(b, 1, -1)
        ], dim=-1)

        # ===== Step 5: Trajectory Decoder =====
        prev_root = self._get_prev_root(prev_output, init_root, b)
        pred_root, pred_vel, hidden_states['trajectory_decoder'] = \
            self.network.trajectory_decoder.forward_step(
                motion_with_kp_original,
                prev_root,
                cam_angvel,
                hidden_states['trajectory_decoder']
            )

        # ===== Step 6: CLIP feature fusion (after the trajectory decoder) =====
        if img_features is not None:
            clip_feat = self.network.clip_proj(img_features[:, -1:])
            motion_context = self.network.clip_gated_fusion(motion_context, clip_feat)

        # ===== Step 7: View encoding =====
        view_feat = self.network.view_encoder(pred_kp3d)

        # ===== Step 8: Gated fusion =====
        motion_context_fused = self.network.gated_fusion(motion_context, view_feat)

        # ===== Step 9: cat(motion_context_fused, kp3d) for view_decoder =====
        motion_with_kp_for_view = torch.cat([
            motion_context_fused,
            pred_kp3d.reshape(b, 1, -1)
        ], dim=-1)

        # ===== Step 10: Prepare init_smpl_view [B, 1, 24, 6] =====
        prev_smpl = self._get_prev_smpl(prev_output, init_smpl, b)
        init_smpl_view = self._prepare_init_smpl_view(prev_smpl, prev_output, b)

        # ===== Step 11: View Decoder - pred_global_orient, pred_cam =====
        pred_global_orient, pred_cam, hidden_states['view_decoder'] = \
            self.network.view_decoder.forward_step(
                motion_with_kp_for_view,
                init_smpl_view,
                hidden_states['view_decoder']
            )

        # ===== Step 12: Dynamic projection =====
        motion_context_projected = self.network.dynamic_projection(motion_context_fused, view_feat)

        # ===== Step 13: cat(motion_context_projected, kp3d) for motion_decoder =====
        motion_with_kp_for_pose = torch.cat([
            motion_context_projected,
            pred_kp3d.reshape(b, 1, -1)
        ], dim=-1)

        # ===== Step 14: Update the buffer =====
        self.state_manager.update_buffers(motion_context_projected, pred_kp3d)
        self.stats['buffer_ops'] += 1

        # ===== Step 15: Motion Decoder - pred_body_pose, pred_shape, pred_contact =====
        pred_body_pose, pred_shape, pred_contact, hidden_states['motion_decoder'] = \
            self.network.motion_decoder.forward_step(
                motion_with_kp_for_pose,
                init_smpl_view,
                hidden_states['motion_decoder']
            )

        # ===== Step 16: Combine pose = [global_orient(6) + body_pose(138)] =====
        pred_pose = torch.cat([pred_global_orient, pred_body_pose], dim=-1)

        # ===== Step 17: SMPL forward =====
        output = self._forward_smpl_optimized(
            pred_pose, pred_shape, pred_cam,
            pred_contact, pred_root, pred_vel,
            pred_kp3d, **kwargs
        )
        self.stats['smpl_calls'] += 1

        return output, hidden_states
    
    def _concat_features(self, motion: torch.Tensor, kp3d: torch.Tensor, b: int) -> torch.Tensor:
        """Concatenate motion and kp3d features"""
        return torch.cat([
            motion,
            kp3d.reshape(b, 1, -1)
        ], dim=-1)
    
    def _get_prev_kp3d(self, prev_output: Optional[Dict], 
                       init_kp: Optional[torch.Tensor], b: int) -> torch.Tensor:
        """Get previous-frame 3D keypoints (avoid conditional branches)"""
        if prev_output is not None and 'kp3d_nn' in prev_output:
            return prev_output['kp3d_nn'][:, -1:].clone()
        if init_kp is None:
            return torch.zeros(b, 1, self.kp3d_dim, device=self.device)
        if init_kp.dim() == 2:
            return init_kp.unsqueeze(1)
        return init_kp[:, -1:] if init_kp.shape[1] > 0 else init_kp
    
    def _get_prev_root(self, prev_output: Optional[Dict],
                       init_root: Optional[torch.Tensor], b: int) -> torch.Tensor:
        """Get the previous root and ensure the return shape is (b, 1, 6) to match trajectory_decoder"""
        if prev_output is not None and 'poses_root_r6d' in prev_output:
            return prev_output['poses_root_r6d'][:, -1:].clone()
        if init_root is None:
            return torch.zeros(b, 1, 6, device=self.device)
        if init_root.dim() == 2:
            return init_root.unsqueeze(1)
        return init_root[:, -1:] if init_root.shape[1] > 0 else init_root
    
    def _get_prev_smpl(self, prev_output: Optional[Dict],
                       init_smpl: Optional[torch.Tensor], b: int) -> torch.Tensor:
        """Get the previous-frame SMPL parameters"""
        if prev_output is not None and 'pose' in prev_output:
            return prev_output['pose'][:, -1:].clone()
        if init_smpl is None:
            return torch.zeros(b, 1, 144, device=self.device)
        if init_smpl.dim() == 2:
            return init_smpl.unsqueeze(1)
        return init_smpl[:, -1:] if init_smpl.shape[1] > 0 else init_smpl

    def _prepare_init_smpl_view(self, prev_smpl: torch.Tensor,
                                prev_output: Optional[Dict], b: int) -> torch.Tensor:
        """Prepare the initialization input for ViewDecoder (requires [B, 1, 24, 6])"""
        if prev_smpl.shape[-1] == 144:
            # If prev_smpl is flattened as [B, 1, 144], reshape it
            return prev_smpl.reshape(b, 1, 24, 6)
        elif prev_smpl.dim() >= 3 and prev_smpl.shape[-1] == 6 and prev_smpl.shape[-2] == 24:
            # If it is already in [B, 1, 24, 6] format
            return prev_smpl
        else:
            # Default: create a zero-initialized [B, 1, 24, 6] tensor
            init_smpl_view = torch.zeros(b, 1, 24, 6, device=self.device)
            if prev_output is not None and 'pose' in prev_output:
                # Try to extract global_orient from prev_output
                prev_pose = prev_output['pose'][:, -1:].clone()
                if prev_pose.shape[-1] >= 6:
                    init_smpl_view[:, :, 0, :] = prev_pose[:, :, :6]
            return init_smpl_view
    
    def _forward_smpl_optimized(self, pred_pose, pred_shape, pred_cam,
                               pred_contact, pred_root, pred_vel, pred_kp3d,
                               **kwargs) -> Dict:
        """Optimized SMPL forward pass (current frame only)"""
        self.network.pred_pose = pred_pose
        self.network.pred_shape = pred_shape
        self.network.pred_cam = pred_cam
        self.network.pred_contact = pred_contact
        self.network.pred_root = pred_root
        self.network.pred_vel = pred_vel
        self.network.pred_kp3d = pred_kp3d
        
        # For single-frame prediction, pred_cam has shape (1,1,3). Passing a full-window bbox causes incorrect SMPL projection
        # Use only the last frame's bbox/cam_intrinsics to match pred_cam
        smpl_kwargs = dict(kwargs)
        if pred_cam.shape[1] == 1 and 'bbox' in smpl_kwargs and smpl_kwargs['bbox'] is not None:
            bbox = smpl_kwargs['bbox']
            if bbox.shape[1] > 1:
                smpl_kwargs['bbox'] = bbox[:, -1:, :]
        if pred_cam.shape[1] == 1 and 'cam_intrinsics' in smpl_kwargs and smpl_kwargs['cam_intrinsics'] is not None:
            K = smpl_kwargs['cam_intrinsics']
            if K.dim() == 4 and K.shape[1] > 1:
                smpl_kwargs['cam_intrinsics'] = K[:, -1:, :, :]
        
        output = self.network.forward_smpl(**smpl_kwargs)
        
        # Add the required keys to preserve continuity
        if 'poses_root_r6d' not in output:
            output['poses_root_r6d'] = pred_root
        if 'vel' not in output:
            output['vel'] = pred_vel
        if 'contact' not in output:
            output['contact'] = pred_contact
        
        return output
    
    def _init_hidden_states(self) -> Dict:
        """Initialize hidden states"""
        return {
            'motion_encoder': None,
            'trajectory_decoder': None,
            'motion_decoder': None,
            'view_decoder': None,
            'trajectory_refiner': None
        }
    
    def reset(self):
        """Reset inferencer state"""
        self.state_manager.reset()
        self.stats = {k: 0 for k in self.stats}
    
    def print_stats(self):
        """Print performance statistics"""
        print(f"\n=== Optimized Stream Inference Stats ===")
        print(f"Buffer Operations: {self.stats['buffer_ops']}")
        print(f"Concat Operations: {self.stats['concat_ops']}")
        print(f"SMPL Calls: {self.stats['smpl_calls']}")
        print(f"Frames Processed: {self.state_manager.frame_count}")
        if self.state_manager.frame_count > 0:
            print(f"Avg Concat/Frame: {self.stats['concat_ops']/self.state_manager.frame_count:.2f}")

