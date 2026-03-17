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
    """自适应窗口大小预测器"""
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
        """预测最优窗口大小"""
        if motion_context.shape[1] > 1:
            motion_std = torch.std(motion_context, dim=1).mean(dim=1, keepdim=True)
        else:
            motion_std = torch.zeros(motion_context.shape[0], 1, device=motion_context.device)
        
        avg_context = motion_context.mean(dim=1)
        learned_weight = self.complexity_net(avg_context)
        
        # 结合统计和学习特征
        complexity = 0.6 * torch.tanh(motion_std * 10) + 0.4 * learned_weight
        window_size = self.min_window + (self.max_window - self.min_window) * complexity
        
        return window_size.squeeze().int()


class OptimizedStreamingInference:
    """
    优化的流式推理类 - 替代原StreamingInference
    
    主要改进:
    1. 自适应窗口大小 (已修复统计问题)
    2. 智能内存管理
    3. 性能监控
    4. (缓存逻辑移至主处理器实现)
    """
    def __init__(self, network, device, max_history_frames=10,
                 enable_adaptive_window=True,
                 min_window=5,
                 max_window=15):
        self.network = network
        self.device = device
        self.max_history_frames = max_history_frames
        
        # RNN状态
        self.hidden_states = None
        self.prev_context = None
        self.prev_kp3d = None
        self.prev_output = None
        self.subject_id = 0
        
        # 优化功能开关
        self.enable_adaptive_window = enable_adaptive_window
        
        # 自适应窗口预测器
        self.min_window = min_window
        self.max_window = max_window
        self.default_window = (min_window + max_window) // 2

        if enable_adaptive_window:
            try:
                # 维度计算可能需要根据实际网络调整
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
        
        # 性能统计
        self.stats = {
            'window_sizes': [],
            'cache_hits': 0,  # 缓存统计将由外部处理器管理
            'cache_misses': 0,
            'inference_times': [],
            'memory_usage': []
        }
        
        # 帧计数器
        self.frame_count = 0

    def _predict_window_size(self):
        """
        预测当前最优窗口大小
        (已修复: 无论何种情况都记录窗口大小)
        """
        if not self.enable_adaptive_window or self.prev_context is None:
            self.stats['window_sizes'].append(self.default_window)
            return self.default_window
        
        try:
            with torch.no_grad():
                window_size = self.window_predictor(self.prev_context)
                if torch.is_tensor(window_size):
                    window_size = window_size.item()
                # 限制窗口大小在合理范围
                window_size = int(max(self.min_window, min(self.max_window, window_size)))
                self.stats['window_sizes'].append(window_size)
                return window_size
        except Exception as e:
            # 使用 warning 级别，更容易被注意到
            logger.warning(f"Window prediction error: {e}, using default window.")
            self.stats['window_sizes'].append(self.default_window)
            return self.default_window

    def process_frame(self, x, inits, window_size=None, img_features=None, 
                    mask=None, init_root=None, cam_angvel=None,
                    cam_intrinsics=None, bbox=None, res=None, 
                    return_y_up=False, subject_id=0):
        """
        优化的单帧处理方法
        """
        start_time = time.time()
        
        # 检测新subject，重置状态
        if self.subject_id != subject_id:
            self.reset(subject_id)
        
        # 1. 自适应窗口大小（如果未指定）
        if window_size is None or window_size == 'auto':
            window_size = self._predict_window_size()
        
        # 2. 准备kwargs
        kwargs = {}
        if cam_intrinsics is not None: kwargs['cam_intrinsics'] = cam_intrinsics
        if bbox is not None: kwargs['bbox'] = bbox
        if res is not None: kwargs['res'] = res
        
        # 3. 运行推理
        # 注意：prev_kp3d 设置为 None，因为它已经包含在 prev_context 中
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
        
        # 4. 更新状态
        self.prev_output = output
        
        # 累积 context (curr_context 已经包含了 motion_context + kp3d)
        if self.prev_context is None:
            self.prev_context = curr_context
        else:
            self.prev_context = torch.cat([self.prev_context, curr_context], dim=1)
        
        # prev_kp3d 不再单独维护，因为它已经在 prev_context 中了
        # 但为了兼容性，如果其他地方需要用到，可以从 prev_context 中提取
        # 例如: context_dim = 512  # motion context dimension
        #       self.prev_kp3d = self.prev_context[..., context_dim:]
        
        self.frame_count += 1
        
        # 5. 限制历史长度
        if self.prev_context is not None and self.prev_context.shape[1] > self.max_history_frames:
            self.prev_context = self.prev_context[:, -self.max_history_frames:]
        
        # 6. 记录性能统计
        inference_time = time.time() - start_time
        self.stats['inference_times'].append(inference_time)
        
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            self.stats['memory_usage'].append(mem_used)
        
        return output


    def copy_state_from(self, other, flip=False):
        """从另一 stream 复制状态。flip=True 时对 pose/kp3d/root 做左右翻转（用于 flip 时利用 normal 的上一帧信息）"""
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
        """翻转 context 中的 kp3d 部分（最后 51 维为 COCO 17 joints）"""
        kp3d_dim = 51
        if ctx.shape[-1] <= kp3d_dim:
            return ctx
        motion_part = ctx[..., :-kp3d_dim]
        kp3d_part = ctx[..., -kp3d_dim:].reshape(*ctx.shape[:-1], 17, 3)
        kp3d_part = kp3d_part[..., [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], :].clone()
        kp3d_part[..., 0] = -kp3d_part[..., 0]
        return torch.cat([motion_part, kp3d_part.reshape(*ctx.shape[:-1], kp3d_dim)], dim=-1)

    def _flip_output(self, out):
        """翻转 prev_output 中的 pose、kp3d_nn、poses_root_r6d"""
        from lib.utils.imutils import flip_pose
        from lib.utils import transforms
        flipped = {}
        for k, v in out.items():
            v = v.clone()
            if k == 'pose':
                sh = v.shape
                v = flip_pose(v.reshape(-1, sh[-1]), representation='rotation_6d').reshape(sh)
            elif k == 'poses_root_r6d':
                # root 仅 1 关节，需单独翻转：negate axis-angle y,z
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
            # 其他 key 直接复制
            flipped[k] = v
        return flipped

    def fuse_view_independent_states(self, other, alpha=0.5):
        """将另一个 stream 的 view-independent hidden states 融合到自身。

        只融合 motion_encoder 和 motion_decoder 的 hidden states，
        因为它们是 view-independent 的（提取运动信息，不依赖视角）。
        trajectory_decoder 和 view_decoder 是 view-dependent，保持不变。

        Args:
            other: 另一个 OptimizedStreamingInference 实例（flip stream）
            alpha: 融合权重，self 的权重为 alpha，other 的权重为 (1-alpha)
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
                # LSTM: (h, c) 每个都是 [num_layers, B, d_embed]
                fused = []
                for s, o in zip(self_h, other_h):
                    if s is not None and o is not None and s.shape == o.shape:
                        fused.append(alpha * s + (1 - alpha) * o)
                    else:
                        fused.append(s)
                self.hidden_states[key] = tuple(fused)
            elif isinstance(self_h, torch.Tensor) and isinstance(other_h, torch.Tensor):
                # GRU: 单个 tensor
                if self_h.shape == other_h.shape:
                    self.hidden_states[key] = alpha * self_h + (1 - alpha) * other_h

    def reset(self, subject_id=None):
        """
        重置流式推理状态
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
        """清理缓存和限制历史"""
        if self.prev_context is not None: self.prev_context = self.prev_context[:, -5:]
        if self.prev_kp3d is not None: self.prev_kp3d = self.prev_kp3d[:, -5:]
        
        torch.cuda.empty_cache()
        gc.collect()


    def get_stats(self):
        """获取性能统计"""
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
        """打印性能统计"""
        stats = self.get_stats()
        logger.info("\n" + "="*60)
        logger.info("Optimized Streaming Inference Statistics")
        logger.info("="*60)
        logger.info(f"Total Frames Processed: {self.frame_count}")
        logger.info(f"Avg Window Size:        {stats['avg_window_size']:.1f} frames")
        logger.info(f"Avg GPU Memory:         {stats['avg_memory_gb']:.2f}GB")
        logger.info(f"Peak GPU Memory:        {stats['peak_memory_gb']:.2f}GB")
        logger.info("="*60 + "\n")