"""
Stable Action Recognition wrapper

Adds a lightweight stabilization layer on top of the existing model:
1. Moving-average smoothing
2. Confidence filtering
3. Delayed action switching
"""
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class StableActionRecognizer:
    """
    Stable action-recognition wrapper
    
    Adds a lightweight stabilization layer on top of the current ActionRecognizer
    """
    
    def __init__(self, base_recognizer, 
                 smoothing_window: int = 5,
                 confidence_threshold: float = 0.15,
                 min_switch_frames: int = 8):
        """
        Args:
            base_recognizer: base ActionRecognizer or ActionRecognizerTRT instance
            smoothing_window: moving-average window size (in frames)
            confidence_threshold: confidence threshold
            min_switch_frames: minimum switch frames (how long a new action must persist before switching)
        """
        self.base_recognizer = base_recognizer
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.min_switch_frames = min_switch_frames
        
        # prediction history used for moving-average smoothing
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
        stable action prediction
        
        Args:
            joints3d: 3D keypoints
            
        Returns:
            (class_idx, confidence, label)
        """
        # Call the base model for prediction
        class_idx, confidence, label = self.base_recognizer.predict_action(joints3d)
        
        # Return immediately if the base model reports an error
        if class_idx < 0 or "buffering" in label or "error" in label:
            return class_idx, confidence, label
        
        # Append to the prediction buffer
        self.prediction_buffer.append((class_idx, confidence, label))
        if len(self.prediction_buffer) > self.smoothing_window:
            self.prediction_buffer.pop(0)
        
        # Apply stabilization
        stable_pred = self._get_stable_prediction()
        if stable_pred:
            return stable_pred
        
        # If stabilization fails, return the current stable action
        if self.current_stable_action != "waiting...":
            return -1, self.current_stable_confidence, self.current_stable_action
        
        return class_idx, confidence, label
    
    def _get_stable_prediction(self) -> Optional[Tuple[int, float, str]]:
        """Use moving-average smoothing and delayed switching to produce a stable prediction"""
        if len(self.prediction_buffer) < 3:
            return None
        
        # Count actions within the recent window
        action_scores = {}  # {label: [confidences]}
        action_counts = {}  # {label: count}
        
        for class_idx, conf, label in self.prediction_buffer:
            if label not in action_scores:
                action_scores[label] = []
                action_counts[label] = 0
            action_scores[label].append(conf)
            action_counts[label] += 1
        
        # Find the most frequent action
        best_label = max(action_counts.keys(), key=lambda x: action_counts[x])
        best_count = action_counts[best_label]
        avg_confidence = np.mean(action_scores[best_label])
        
        # Confidence filtering
        if avg_confidence < self.confidence_threshold:
            return None
        
        # Get the corresponding class_idx
        best_class_idx = -1
        for class_idx, _, label in self.prediction_buffer:
            if label == best_label:
                best_class_idx = class_idx
                break
        
        # Action-switch delay mechanism
        if best_label != self.current_stable_action:
            # A new action needs confirmation
            if self.pending_action is None or self.pending_action[0] != best_label:
                # Start tracking a new pending action
                self.pending_action = (best_label, avg_confidence)
                self.pending_frames = 1
                # Keep using the current action
                return None
            else:
                # The pending action is still being confirmed
                self.pending_frames += 1
                if self.pending_frames >= self.min_switch_frames:
                    # Confirm the switch
                    self.current_stable_action = best_label
                    self.current_stable_confidence = avg_confidence
                    self.current_action_frames = 0
                    self.pending_action = None
                    self.pending_frames = 0
                    logger.debug(f"Action switched: {best_label} (conf: {avg_confidence:.3f}, count: {best_count}/{len(self.prediction_buffer)})")
                    return best_class_idx, avg_confidence, best_label
                else:
                    # Keep using the current action
                    return None
        else:
            # The action has not changed
            self.current_action_frames += 1
            self.pending_action = None
            self.pending_frames = 0
            self.current_stable_confidence = avg_confidence
            return best_class_idx, avg_confidence, best_label
    
    def get_buffer_size(self) -> int:
        """Get buffer size"""
        return self.base_recognizer.get_buffer_size()
    
    def reset_buffer(self):
        """Reset the buffer"""
        self.base_recognizer.reset_buffer()
        self.prediction_buffer = []
        self.current_stable_action = "waiting..."
        self.current_stable_confidence = 0.0
        self.current_action_frames = 0
        self.pending_action = None
        self.pending_frames = 0
    
    def get_buffer_info(self) -> dict:
        """Get buffer information"""
        info = self.base_recognizer.get_buffer_info()
        info['stable_action'] = self.current_stable_action
        info['stable_confidence'] = self.current_stable_confidence
        info['prediction_buffer_size'] = len(self.prediction_buffer)
        return info
    
    def __getattr__(self, name):
        """Proxy all other attributes to the base model"""
        return getattr(self.base_recognizer, name)
