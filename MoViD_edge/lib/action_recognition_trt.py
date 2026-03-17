"""
TensorRT-based Action Recognition Module

Use a TensorRT engine for fast action-recognition inference
Supports the TensorRT format for STGCN models
"""
import os
import numpy as np
from typing import List, Optional, Tuple
from loguru import logger

# Check TensorRT availability
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TensorRT not available: {e}")
    logger.warning("Falling back to PyTorch inference")


class ActionRecognizerTRT:
    """
    TensorRT-based Action Recognizer for skeleton-based action recognition
    
    Features:
    - Faster inference than PyTorch
    - Support for FP16/INT8 precision
    - Optimized for real-time applications
    """
    
    def __init__(self,
                 engine_path: str,
                 label_map_path: Optional[str] = None,
                 window_size: int = 100,
                 num_keypoints: int = 25,
                 num_persons: int = 2):
        """
        Initialize TensorRT action recognizer
        
        Args:
            engine_path: Path to TensorRT engine file (.engine)
            label_map_path: Path to label map file
            window_size: Number of frames for temporal window
            num_keypoints: Number of skeleton keypoints (25 for NTU)
            num_persons: Maximum number of persons
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT or PyCUDA not available")
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        self.window_size = window_size
        self.num_keypoints = num_keypoints
        self.num_persons = num_persons
        
        # Load TensorRT engine
        logger.info(f"Loading TensorRT engine from {engine_path}")
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
        
        # Set input shape for dynamic batch (required for TensorRT with optimization profiles)
        self.context.set_input_shape('input', self.input_shape)
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        # Load label map
        self.label_map = None
        if label_map_path and os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = [x.strip() for x in f.readlines()]
            logger.info(f"Loaded {len(self.label_map)} action labels")
        
        # Skeleton buffer
        self.skeleton_buffer: List[np.ndarray] = []
        
        # Smoothing parameters
        self.prediction_history: List[Tuple[int, float, str]] = []
        self.history_size = 5
        self.confidence_threshold = 0.1  # Lower the threshold because confidence may be lower after softmax
        self.current_action = "waiting..."
        self.current_confidence = 0.0
        self.frames_since_prediction = 0
        self.prediction_interval = 5
        
        logger.info(f"TensorRT action recognizer initialized: "
                   f"window_size={window_size}, num_keypoints={num_keypoints}")
    
    def _allocate_buffers(self):
        """Allocate GPU memory for input/output buffers"""
        # Input shape: (1, num_persons, window_size, num_keypoints, 3)
        self.input_shape = (1, self.num_persons, self.window_size, 
                           self.num_keypoints, 3)
        input_size = int(np.prod(self.input_shape) * np.float32().nbytes)
        
        # Output shape: (1, num_classes) - assume 60 classes for NTU60
        self.num_classes = 60
        self.output_shape = (1, self.num_classes)
        output_size = int(np.prod(self.output_shape) * np.float32().nbytes)
        
        # Allocate device memory
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        
        # Allocate host memory
        self.h_input = np.zeros(self.input_shape, dtype=np.float32)
        self.h_output = np.zeros(self.output_shape, dtype=np.float32)
        
        logger.debug(f"Allocated buffers: input={self.input_shape}, output={self.output_shape}")
    
    def add_skeleton_frame(self, joints3d: np.ndarray):
        """
        Add a skeleton frame to the buffer (real-time updates)
        
        Args:
            joints3d: 3D joints, shape (num_keypoints, 3) or (batch, num_keypoints, 3)
        """
        # Handle different input shapes
        if len(joints3d.shape) == 3:
            # (batch, num_joints, 3) -> take last frame
            joints3d = joints3d[-1]
        
        # Handle keypoint count mismatch
        if joints3d.shape[0] != self.num_keypoints:
            if joints3d.shape[0] > self.num_keypoints:
                joints3d = joints3d[:self.num_keypoints]
            else:
                # Pad with zeros
                padded = np.zeros((self.num_keypoints, 3), dtype=joints3d.dtype)
                padded[:joints3d.shape[0]] = joints3d
                joints3d = padded
        
        # Real-time updates: append every frame to the buffer
        self.skeleton_buffer.append(joints3d.astype(np.float32))
        
        # Keep only window_size frames (sliding window)
        # This keeps the buffer aligned to the most recent window_size frames
        if len(self.skeleton_buffer) > self.window_size:
            self.skeleton_buffer.pop(0)
    
    def _prepare_input(self) -> np.ndarray:
        """
        Prepare input tensor for TensorRT inference
        Use the latest buffer contents to preserve real-time behavior
        """
        num_frames = len(self.skeleton_buffer)
        
        # Use the latest buffer contents (real-time updates)
        # Stack frames: (T, V, C) - newest frames at the end
        keypoints = np.stack(self.skeleton_buffer, axis=0)
        
        # Pad or sample to window_size
        if num_frames < self.window_size:
            # Pad with zeros (place the newest frames at the end and pad the front)
            padded = np.zeros((self.window_size, self.num_keypoints, 3), dtype=np.float32)
            padded[-num_frames:] = keypoints  # place the newest frames at the end
            keypoints = padded
        elif num_frames > self.window_size:
            # Uniform sample - ensure the newest frames are included
            indices = np.linspace(0, num_frames - 1, self.window_size, dtype=int)
            # Ensure the last index is num_frames - 1 (the newest frame)
            indices[-1] = num_frames - 1
            keypoints = keypoints[indices]
        
        # Reshape to (1, M, T, V, C)
        # Person 0 is the main skeleton, Person 1 is zeros (single person)
        input_tensor = np.zeros(self.input_shape, dtype=np.float32)
        input_tensor[0, 0] = keypoints  # First person
        
        # Pre-normalize (similar to PreNormalize3D in pyskl)
        # Center the skeleton at spine/hip
        spine_idx = 1  # Spine base in NTU format
        if self.num_keypoints >= 25:
            # Center at spine
            center = input_tensor[0, 0, :, spine_idx:spine_idx+1, :].mean(axis=0, keepdims=True)
            input_tensor[0, 0] = input_tensor[0, 0] - center
        
        return input_tensor
    
    def predict_action(self, joints3d: Optional[np.ndarray] = None) -> Tuple[int, float, str]:
        """
        Predict action using TensorRT engine
        
        Args:
            joints3d: Optional new frame to add before prediction
            
        Returns:
            Tuple of (class_idx, confidence, label_name)
        """
        # Add new frame if provided
        if joints3d is not None:
            self.add_skeleton_frame(joints3d)
        
        # Need at least 15 frames for reliable prediction (for RNN/temporal models)
        min_frames_required = 15
        buffer_size = len(self.skeleton_buffer)
        
        if buffer_size < min_frames_required:
            logger.debug(f"Buffer has {buffer_size}/{min_frames_required} frames, waiting...")
            return -1, 0.0, f"buffering_{buffer_size}/{min_frames_required}"
        
        # Real-time prediction: run inference on every frame to preserve accuracy
        # The buffer is updated in real time because add_skeleton_frame is called on every frame
        # Now predict on every frame using the latest buffer contents
        logger.debug(f"Performing real-time prediction: buffer={buffer_size} frames")
        
        try:
            # Prepare input
            self.h_input = self._prepare_input()
            
            # Copy to device
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            
            # Run inference
            self.context.execute_async_v2(
                [int(self.d_input), int(self.d_output)], 
                self.stream.handle
            )
            
            # Copy output to host
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            
            # Get prediction
            scores = self.h_output[0]
            class_idx = int(np.argmax(scores))
            raw_score = float(scores[class_idx])
            
            # Apply softmax for proper confidence
            exp_scores = np.exp(scores - np.max(scores))
            softmax_scores = exp_scores / exp_scores.sum()
            confidence = float(softmax_scores[class_idx])
            
            # Get label
            label_name = self.label_map[class_idx] if self.label_map else f"class_{class_idx}"
            
            logger.debug(f"TRT prediction: {label_name}, raw={raw_score:.4f}, softmax={confidence:.4f}")
            
            # Update current_action for display regardless of confidence
            self.current_action = label_name
            self.current_confidence = confidence
            
            # Update history
            self.prediction_history.append((class_idx, confidence, label_name))
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
            
            # Get stable prediction
            stable = self._get_stable_prediction()
            if stable:
                self.current_action = stable[2]
                self.current_confidence = stable[1]
                return stable
            
            self.current_action = label_name
            self.current_confidence = confidence
            return class_idx, confidence, label_name
            
        except Exception as e:
            logger.error(f"TensorRT inference error: {e}")
            return -1, 0.0, f"error: {e}"
    
    def _get_stable_prediction(self) -> Optional[Tuple[int, float, str]]:
        """Get stable prediction using voting mechanism"""
        if len(self.prediction_history) < 3:
            return None
        
        # Count occurrences
        action_counts = {}
        action_confidences = {}
        
        for class_idx, conf, label in self.prediction_history:
            if label not in action_counts:
                action_counts[label] = 0
                action_confidences[label] = []
            action_counts[label] += 1
            action_confidences[label].append(conf)
        
        # Find most frequent action
        best_label = max(action_counts, key=action_counts.get)
        avg_conf = np.mean(action_confidences[best_label])
        
        # Find class_idx for best_label
        for class_idx, _, label in self.prediction_history:
            if label == best_label:
                return class_idx, avg_conf, best_label
        
        return None
    
    def reset_buffer(self):
        """Reset skeleton buffer and prediction state"""
        self.skeleton_buffer = []
        self.prediction_history = []
        self.current_action = "waiting..."
        self.current_confidence = 0.0
        self.frames_since_prediction = 0
        logger.info("Action recognition buffer reset - ready for real-time updates")
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        return len(self.skeleton_buffer)
    
    def get_buffer_info(self) -> dict:
        """Get buffer information for monitoring"""
        return {
            'size': len(self.skeleton_buffer),
            'window_size': self.window_size,
            'num_keypoints': self.num_keypoints,
            'is_ready': len(self.skeleton_buffer) >= 15,
            'prediction_interval': self.prediction_interval
        }
    
    def __del__(self):
        """Cleanup CUDA resources"""
        try:
            if hasattr(self, 'd_input'):
                self.d_input.free()
            if hasattr(self, 'd_output'):
                self.d_output.free()
        except Exception:
            pass


def create_action_recognizer(config_path: str = None,
                            checkpoint_path: str = None,
                            engine_path: str = None,
                            label_map_path: str = None,
                            device: str = 'cuda:0',
                            window_size: int = 100,
                            num_keypoints: int = 25) -> object:
    """
    Factory function to create the best available action recognizer
    
    Prefers TensorRT if engine is available, otherwise falls back to PyTorch
    
    Args:
        config_path: PyTorch config path (for fallback)
        checkpoint_path: PyTorch checkpoint path (for fallback)
        engine_path: TensorRT engine path (preferred)
        label_map_path: Label map path
        device: Device for PyTorch inference
        window_size: Temporal window size
        num_keypoints: Number of skeleton keypoints
        
    Returns:
        ActionRecognizerTRT or ActionRecognizer instance
    """
    # Try TensorRT first
    if engine_path and os.path.exists(engine_path) and TRT_AVAILABLE:
        try:
            return ActionRecognizerTRT(
                engine_path=engine_path,
                label_map_path=label_map_path,
                window_size=window_size,
                num_keypoints=num_keypoints
            )
        except Exception as e:
            logger.warning(f"Failed to load TensorRT engine: {e}")
    
    # Fallback to PyTorch
    if config_path and checkpoint_path:
        from lib.action_recognition import ActionRecognizer
        return ActionRecognizer(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            label_map_path=label_map_path,
            device=device,
            window_size=window_size,
            num_keypoints=num_keypoints
        )
    
    raise ValueError("Either engine_path or (config_path, checkpoint_path) must be provided")
