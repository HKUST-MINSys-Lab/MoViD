"""
稳定的 Action Recognition 包装器

在现有模型基础上添加简单的稳定化机制：
1. 滑动平均平滑
2. 置信度过滤
3. 动作切换延迟
"""
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class StableActionRecognizer:
    """
    稳定的动作识别包装器
    
    在现有 ActionRecognizer 基础上添加简单的稳定化机制
    """
    
    def __init__(self, base_recognizer, 
                 smoothing_window: int = 5,
                 confidence_threshold: float = 0.15,
                 min_switch_frames: int = 8):
        """
        Args:
            base_recognizer: 基础的 ActionRecognizer 或 ActionRecognizerTRT 实例
            smoothing_window: 滑动平均窗口大小（帧数）
            confidence_threshold: 置信度阈值
            min_switch_frames: 最小切换帧数（新动作需要持续多少帧才切换）
        """
        self.base_recognizer = base_recognizer
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.min_switch_frames = min_switch_frames
        
        # 预测历史（用于滑动平均）
        self.prediction_buffer: list = []  # [(class_idx, confidence, label), ...]
        self.current_stable_action: str = "waiting..."
        self.current_stable_confidence: float = 0.0
        self.current_action_frames: int = 0
        self.pending_action: Optional[Tuple[str, float]] = None
        self.pending_frames: int = 0
        
        logger.info(f"Stable wrapper initialized:")
        logger.info(f"  Smoothing window: {smoothing_window} frames")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  Min switch frames: {min_switch_frames}")
    
    def predict_action(self, joints3d: Optional[np.ndarray] = None) -> Tuple[int, float, str]:
        """
        稳定的动作预测
        
        Args:
            joints3d: 3D 关键点
            
        Returns:
            (class_idx, confidence, label)
        """
        # 调用基础模型进行预测
        class_idx, confidence, label = self.base_recognizer.predict_action(joints3d)
        
        # 如果基础模型返回错误，直接返回
        if class_idx < 0 or "buffering" in label or "error" in label:
            return class_idx, confidence, label
        
        # 添加到预测缓冲区
        self.prediction_buffer.append((class_idx, confidence, label))
        if len(self.prediction_buffer) > self.smoothing_window:
            self.prediction_buffer.pop(0)
        
        # 应用稳定化
        stable_pred = self._get_stable_prediction()
        if stable_pred:
            return stable_pred
        
        # 如果稳定化失败，返回当前稳定动作
        if self.current_stable_action != "waiting...":
            return -1, self.current_stable_confidence, self.current_stable_action
        
        return class_idx, confidence, label
    
    def _get_stable_prediction(self) -> Optional[Tuple[int, float, str]]:
        """使用滑动平均和延迟切换获取稳定预测"""
        if len(self.prediction_buffer) < 3:
            return None
        
        # 统计最近窗口内的动作
        action_scores = {}  # {label: [confidences]}
        action_counts = {}  # {label: count}
        
        for class_idx, conf, label in self.prediction_buffer:
            if label not in action_scores:
                action_scores[label] = []
                action_counts[label] = 0
            action_scores[label].append(conf)
            action_counts[label] += 1
        
        # 找出出现次数最多的动作
        best_label = max(action_counts.keys(), key=lambda x: action_counts[x])
        best_count = action_counts[best_label]
        avg_confidence = np.mean(action_scores[best_label])
        
        # 置信度过滤
        if avg_confidence < self.confidence_threshold:
            return None
        
        # 获取对应的 class_idx
        best_class_idx = -1
        for class_idx, _, label in self.prediction_buffer:
            if label == best_label:
                best_class_idx = class_idx
                break
        
        # 动作切换延迟机制
        if best_label != self.current_stable_action:
            # 新动作需要确认
            if self.pending_action is None or self.pending_action[0] != best_label:
                # 开始新的待确认动作
                self.pending_action = (best_label, avg_confidence)
                self.pending_frames = 1
                # 继续使用当前动作
                return None
            else:
                # 待确认动作持续中
                self.pending_frames += 1
                if self.pending_frames >= self.min_switch_frames:
                    # 确认切换
                    self.current_stable_action = best_label
                    self.current_stable_confidence = avg_confidence
                    self.current_action_frames = 0
                    self.pending_action = None
                    self.pending_frames = 0
                    logger.debug(f"Action switched: {best_label} (conf: {avg_confidence:.3f}, count: {best_count}/{len(self.prediction_buffer)})")
                    return best_class_idx, avg_confidence, best_label
                else:
                    # 继续使用当前动作
                    return None
        else:
            # 动作未变化
            self.current_action_frames += 1
            self.pending_action = None
            self.pending_frames = 0
            self.current_stable_confidence = avg_confidence
            return best_class_idx, avg_confidence, best_label
    
    def get_buffer_size(self) -> int:
        """获取 buffer 大小"""
        return self.base_recognizer.get_buffer_size()
    
    def reset_buffer(self):
        """重置 buffer"""
        self.base_recognizer.reset_buffer()
        self.prediction_buffer = []
        self.current_stable_action = "waiting..."
        self.current_stable_confidence = 0.0
        self.current_action_frames = 0
        self.pending_action = None
        self.pending_frames = 0
    
    def get_buffer_info(self) -> dict:
        """获取 buffer 信息"""
        info = self.base_recognizer.get_buffer_info()
        info['stable_action'] = self.current_stable_action
        info['stable_confidence'] = self.current_stable_confidence
        info['prediction_buffer_size'] = len(self.prediction_buffer)
        return info
    
    def __getattr__(self, name):
        """代理其他属性到基础模型"""
        return getattr(self.base_recognizer, name)
