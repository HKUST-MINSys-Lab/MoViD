# optimized_streaming.py

import os
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from loguru import logger


class AdaptiveWindowPredictor(nn.Module):
    """Adaptive window-size predictor"""
    def __init__(self, d_context=256, min_window=5, max_window=15):
        super().__init__()
        self.min_window = min_window
        self.max_window = max_window
        
        self.complexity_net = nn.Sequential(
            nn.Linear(d_context, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, motion_context):
        """Predict the optimal window size"""
        if motion_context.shape[1] > 1:
            motion_std = torch.std(motion_context, dim=1).mean(dim=1, keepdim=True)
        else:
            motion_std = torch.zeros(motion_context.shape[0], 1, device=motion_context.device)
        
        avg_context = motion_context.mean(dim=1)
        learned_weight = self.complexity_net(avg_context)
        
        # Combine statistical and learned features
        complexity = 0.6 * torch.tanh(motion_std * 10) + 0.4 * learned_weight
        window_size = self.min_window + (self.max_window - self.min_window) * complexity
        
        return window_size.squeeze().int()


class OptimizedStreamingInference:
    """
    Optimized streaming inference class - replacement for the original StreamingInference
    
    Main improvements:
    1. adaptive window sizing (statistics issue fixed)
    2. smart memory management
    3. performance monitoring
    4. (cache logic moved into the main processor)
    """
    def __init__(self, network, device, max_history_frames=10,
                 enable_adaptive_window=True,
                 min_window=5,
                 max_window=15):
        self.network = network
        self.device = device
        self.max_history_frames = max_history_frames
        
        # RNN state
        self.hidden_states = None
        self.prev_context = None
        self.prev_kp3d = None
        self.prev_output = None
        self.subject_id = 0
        
        # Optimization feature toggles
        self.enable_adaptive_window = enable_adaptive_window
        
        # Adaptive window predictor
        self.min_window = min_window
        self.max_window = max_window
        self.default_window = (min_window + max_window) // 2

        if enable_adaptive_window:
            try:
                # Dimension calculation may need adjustment for the actual network
                d_context = network.motion_encoder.d_embed * 2 + 17 * 3
                self.window_predictor = AdaptiveWindowPredictor(
                    d_context=d_context,
                    min_window=min_window,
                    max_window=max_window
                ).to(device)
                logger.info(f"Adaptive window predictor initialized (range: {min_window}-{max_window})")
            except Exception as e:
                logger.warning(f"Failed to init adaptive window: {e}, using fixed window.")
                self.enable_adaptive_window = False
        
        # Performance statistics
        self.stats = {
            'window_sizes': [],
            'cache_hits': 0,  # cache statistics are managed by the external processor
            'cache_misses': 0,
            'inference_times': [],
            'memory_usage': []
        }
        
        # Frame counter
        self.frame_count = 0

    def _predict_window_size(self):
        """
        Predict the current optimal window size
        (Fixed: record the window size in every case)
        """
        if not self.enable_adaptive_window or self.prev_context is None:
            self.stats['window_sizes'].append(self.default_window)
            return self.default_window
        
        try:
            with torch.no_grad():
                window_size = self.window_predictor(self.prev_context)
                if torch.is_tensor(window_size):
                    window_size = window_size.item()
                # Clamp the window size to a reasonable range
                window_size = int(max(self.min_window, min(self.max_window, window_size)))
                self.stats['window_sizes'].append(window_size)
                return window_size
        except Exception as e:
            # Use warning level so it is easier to notice
            logger.warning(f"Window prediction error: {e}, using default window.")
            self.stats['window_sizes'].append(self.default_window)
            return self.default_window

    def process_frame(self, x, inits, window_size=None, img_features=None, 
                    mask=None, init_root=None, cam_angvel=None,
                    cam_intrinsics=None, bbox=None, res=None, 
                    return_y_up=False, subject_id=0):
        """
        Optimized single-frame processing method
        """
        start_time = time.time()
        
        # Detect a new subject and reset the state
        if self.subject_id != subject_id:
            self.reset(subject_id)
        
        # 1. Adaptive window size (if not specified)
        if window_size is None or window_size == 'auto':
            window_size = self._predict_window_size()
        
        # 2. Prepare kwargs
        kwargs = {}
        if cam_intrinsics is not None: kwargs['cam_intrinsics'] = cam_intrinsics
        if bbox is not None: kwargs['bbox'] = bbox
        if res is not None: kwargs['res'] = res
        
        # 3. Run inference
        # Note: prev_kp3d is set to None because it is already included in prev_context
        with torch.no_grad():
            output, self.hidden_states, curr_context, curr_kp3d, avg_output = \
                self.network.stream_inference(
                    x, inits,
                    img_features=img_features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=return_y_up,
                    window_size=window_size,
                    hidden_states=self.hidden_states,
                    prev_context=self.prev_context,  # Contains motion_context + kp3d
                    prev_kp3d=None,  # Set to None - not used anymore
                    prev_output=self.prev_output,
                    **kwargs
                )
        
        # 4. Update state
        self.prev_output = output
        
        # Accumulate context (curr_context already contains motion_context + kp3d)
        if self.prev_context is None:
            self.prev_context = curr_context
        else:
            self.prev_context = torch.cat([self.prev_context, curr_context], dim=1)
        
        # prev_kp3d is no longer maintained separately because it is already stored in prev_context
        # For compatibility, it can still be extracted from prev_context elsewhere if needed
        # for example: context_dim = 512  # motion context dimension
        #       self.prev_kp3d = self.prev_context[..., context_dim:]
        
        self.frame_count += 1
        
        # 5. Limit history length
        if self.prev_context is not None and self.prev_context.shape[1] > self.max_history_frames:
            self.prev_context = self.prev_context[:, -self.max_history_frames:]
        
        # 6. Record performance statistics
        inference_time = time.time() - start_time
        self.stats['inference_times'].append(inference_time)
        
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            self.stats['memory_usage'].append(mem_used)
        
        return output


    def copy_state_from(self, other, flip=False):
        """Copy state from another stream. When flip=True, flip pose/kp3d/root left-right so flip mode can reuse the previous normal-frame state"""
        if other.hidden_states is not None:
            if isinstance(other.hidden_states, dict):
                self.hidden_states = {}
                for k, v in other.hidden_states.items():
                    if v is None:
                        self.hidden_states[k] = None
                    elif isinstance(v, tuple):
                        self.hidden_states[k] = tuple(h.clone() if h is not None else None for h in v)
                    else:
                        self.hidden_states[k] = v.clone()
            elif isinstance(other.hidden_states, tuple):
                self.hidden_states = tuple(h.clone() if h is not None else None for h in other.hidden_states)
            else:
                self.hidden_states = other.hidden_states.clone()
        else:
            self.hidden_states = None

        if other.prev_context is not None:
            ctx = other.prev_context.clone()
            if flip:
                ctx = self._flip_context(ctx)
            self.prev_context = ctx
        else:
            self.prev_context = None

        if other.prev_output is not None:
            self.prev_output = self._flip_output(other.prev_output) if flip else {k: v.clone() for k, v in other.prev_output.items()}
        else:
            self.prev_output = None

    def _flip_context(self, ctx):
        """Flip the kp3d section inside context (the last 51 dimensions correspond to COCO 17 joints)"""
        kp3d_dim = 51
        if ctx.shape[-1] <= kp3d_dim:
            return ctx
        motion_part = ctx[..., :-kp3d_dim]
        kp3d_part = ctx[..., -kp3d_dim:].reshape(*ctx.shape[:-1], 17, 3)
        kp3d_part = kp3d_part[..., [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], :].clone()
        kp3d_part[..., 0] = -kp3d_part[..., 0]
        return torch.cat([motion_part, kp3d_part.reshape(*ctx.shape[:-1], kp3d_dim)], dim=-1)

    def _flip_output(self, out):
        """Flip pose, kp3d_nn, and poses_root_r6d inside prev_output"""
        from lib.utils.imutils import flip_pose
        from lib.utils import transforms
        flipped = {}
        for k, v in out.items():
            v = v.clone()
            if k == 'pose':
                sh = v.shape
                v = flip_pose(v.reshape(-1, sh[-1]), representation='rotation_6d').reshape(sh)
            elif k == 'poses_root_r6d':
                # root uses a single joint and must be flipped separately by negating axis-angle y and z
                sh = v.shape
                aa = transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(v.reshape(-1, 6)))
                aa = aa.reshape(*sh[:-1], 3)
                aa[..., 1] = -aa[..., 1]
                aa[..., 2] = -aa[..., 2]
                v = transforms.matrix_to_rotation_6d(transforms.axis_angle_to_matrix(aa)).reshape(sh)
            elif k == 'kp3d_nn':
                v = v.reshape(*v.shape[:-1], 17, 3)
                v = v[..., [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], :].clone()
                v[..., 0] = -v[..., 0]
                v = v.reshape(*v.shape[:-2], 51)
            # Copy the other keys directly
            flipped[k] = v
        return flipped

    def fuse_view_independent_states(self, other, alpha=0.5):
        """Blend another stream's view-independent hidden states into the current one.

        Blend only the hidden states from motion_encoder and motion_decoder,
        because they are view-independent and encode motion without depending on viewpoint.
        trajectory_decoder and view_decoder are view-dependent and therefore remain unchanged.

        Args:
            other: another OptimizedStreamingInference instance (flip stream)
            alpha: blend weight: self uses alpha and other uses (1-alpha)
        """
        if self.hidden_states is None or other.hidden_states is None:
            return
        if not isinstance(self.hidden_states, dict) or not isinstance(other.hidden_states, dict):
            return

        view_independent_keys = ['motion_encoder', 'motion_decoder']
        for key in view_independent_keys:
            self_h = self.hidden_states.get(key)
            other_h = other.hidden_states.get(key)
            if self_h is None or other_h is None:
                continue
            if isinstance(self_h, tuple) and isinstance(other_h, tuple):
                # LSTM: (h, c), each shaped [num_layers, B, d_embed]
                fused = []
                for s, o in zip(self_h, other_h):
                    if s is not None and o is not None and s.shape == o.shape:
                        fused.append(alpha * s + (1 - alpha) * o)
                    else:
                        fused.append(s)
                self.hidden_states[key] = tuple(fused)
            elif isinstance(self_h, torch.Tensor) and isinstance(other_h, torch.Tensor):
                # GRU: single tensor
                if self_h.shape == other_h.shape:
                    self.hidden_states[key] = alpha * self_h + (1 - alpha) * other_h

    def reset(self, subject_id=None):
        """
        Reset streaming-inference state
        """
        self.subject_id = subject_id
        self.hidden_states = None
        self.prev_context = None  # Contains motion_context + kp3d
        self.prev_kp3d = None  # Not used anymore, kept for compatibility
        self.prev_output = None
        self.frame_count = 0
        # FIX: Include ALL keys that exist in __init__
        self.stats = {
            'window_sizes': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'inference_times': [],
            'memory_usage': []
        }
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_cache(self):
        """Clear caches and trim history"""
        if self.prev_context is not None: self.prev_context = self.prev_context[:, -5:]
        if self.prev_kp3d is not None: self.prev_kp3d = self.prev_kp3d[:, -5:]
        
        torch.cuda.empty_cache()
        gc.collect()


    def get_stats(self):
        """Get performance statistics"""
        stats = {}
        if self.stats['window_sizes']:
            stats['avg_window_size'] = sum(self.stats['window_sizes']) / len(self.stats['window_sizes'])
        else:
            stats['avg_window_size'] = 0.0

        
        if self.stats['memory_usage']:
            stats['avg_memory_gb'] = sum(self.stats['memory_usage']) / len(self.stats['memory_usage'])
            stats['peak_memory_gb'] = max(self.stats['memory_usage'])
        else:
            stats['avg_memory_gb'] = 0.0
            stats['peak_memory_gb'] = 0.0
            
        return stats

    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        logger.info("\n" + "="*60)
        logger.info("Optimized Streaming Inference Statistics")
        logger.info("="*60)
        logger.info(f"Total Frames Processed: {self.frame_count}")
        logger.info(f"Avg Window Size:        {stats['avg_window_size']:.1f} frames")
        logger.info(f"Avg GPU Memory:         {stats['avg_memory_gb']:.2f}GB")
        logger.info(f"Peak GPU Memory:        {stats['peak_memory_gb']:.2f}GB")
        logger.info("="*60 + "\n")