"""
Action Recognition Module using pyskl
Converts MoViD 3D skeleton output to pyskl format and performs action recognition
"""
import os
import sys
import numpy as np
import torch
from loguru import logger
from typing import Dict, List, Optional, Tuple

# Try to add pyskl to path if not installed
_pyskl_paths = [
    os.environ.get('PYSKL_PATH', ''),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'third-party', 'pyskl'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyskl'),
    os.path.expanduser('~/pyskl'),
    '/opt/pyskl'
]
for _pyskl_path in _pyskl_paths:
    if _pyskl_path and os.path.exists(_pyskl_path) and _pyskl_path not in sys.path:
        sys.path.insert(0, _pyskl_path)
        logger.info(f"Added pyskl path to sys.path: {_pyskl_path}")
        break

try:
    import mmcv
    from pyskl.apis import inference_recognizer, init_recognizer
    PYSKL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"pyskl not available: {e}. Action recognition will be disabled.")
    PYSKL_AVAILABLE = False


class ActionRecognizer:
    """
    Action Recognition using pyskl models
    Converts MoViD 3D skeleton (joints3d) to pyskl format and performs inference
    """
    
    def __init__(self, 
                 config_path: str,
                 checkpoint_path: str,
                 label_map_path: Optional[str] = None,
                 device: str = 'cuda:0',
                 window_size: int = 48,
                 num_keypoints: int = 17):
        """
        Initialize action recognizer
        
        Args:
            config_path: Path to pyskl config file
            checkpoint_path: Path to model checkpoint
            label_map_path: Path to label map file (optional)
            device: Device to run inference on
            window_size: Number of frames for action recognition window
            num_keypoints: Number of keypoints (17 for COCO, 25 for NTU)
        """
        if not PYSKL_AVAILABLE:
            raise ImportError("pyskl is not available. Please install it first.")
        
        self.device = device
        self.window_size = window_size
        
        # Load model
        logger.info(f"Loading action recognition model from {checkpoint_path}")
        self.config = mmcv.Config.fromfile(config_path)
        
        # Detect required number of keypoints and window size from config
        # Check if it's a GCN model (uses NTU format with 25 keypoints)
        if hasattr(self.config, 'model') and 'GCN' in self.config.model.get('type', ''):
            # GCN models typically use NTU format (25 keypoints)
            self.num_keypoints = 25
            # GCN models typically use 100 frames
            if window_size == 48:  # Default, adjust for GCN
                self.window_size = 100
            logger.info("Detected GCN model, using 25 keypoints (NTU format), window_size={}".format(self.window_size))
        else:
            # PoseC3D models typically use COCO format (17 keypoints)
            self.num_keypoints = num_keypoints
            # PoseC3D models typically use 48 frames
            if window_size == 100:  # If set to GCN default, adjust for PoseC3D
                self.window_size = 48
            logger.info(f"Using {self.num_keypoints} keypoints (COCO format), window_size={self.window_size}")
        
        # Remove DecompressPose from pipeline if present (for real-time inference)
        self.config.data.test.pipeline = [
            x for x in self.config.data.test.pipeline 
            if x.get('type') != 'DecompressPose'
        ]
        
        self.model = init_recognizer(self.config, checkpoint_path, device)
        logger.info("Action recognition model loaded successfully")
        
        # Load label map if provided
        self.label_map = None
        if label_map_path and os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = [x.strip() for x in f.readlines()]
            logger.info(f"Loaded {len(self.label_map)} action labels from {label_map_path}")
        else:
            logger.warning(f"Label map not found at {label_map_path}, will return class indices")
        
        # Buffer to store skeleton sequences
        self.skeleton_buffer: List[np.ndarray] = []
        
        # Keep the previous prediction for stable display
        self.current_action: str = "waiting..."
        self.current_confidence: float = 0.0
        
    def convert_movid_to_pyskl_format(self, joints3d: np.ndarray) -> np.ndarray:
        """
        Convert MoViD joints3d to pyskl format
        
        MoViD joints3d shape: (batch, num_joints, 3) or (num_joints, 3)
        pyskl keypoint format: (M, T, V, C) where:
            M = number of persons (usually 1)
            T = number of frames
            V = number of keypoints
            C = coordinates (3 for 3D)
        
        Args:
            joints3d: 3D joints from MoViD output, shape (num_joints, 3) or (batch, num_joints, 3)
            
        Returns:
            keypoint array in pyskl format: (1, 1, V, 3)
        """
        # Handle different input shapes
        if len(joints3d.shape) == 2:
            # (num_joints, 3)
            joints3d = joints3d[np.newaxis, :, :]  # (1, num_joints, 3)
        elif len(joints3d.shape) == 3:
            # (batch, num_joints, 3) - take the last frame
            joints3d = joints3d[-1:, :, :]  # (1, num_joints, 3)
        
        # Handle keypoint count mismatch
        input_num_keypoints = joints3d.shape[1]
        if input_num_keypoints != self.num_keypoints:
            if input_num_keypoints == 25 and self.num_keypoints == 17:
                # NTU 25 to COCO 17 - use subset of joints
                logger.debug(f"Mapping NTU 25 keypoints to COCO 17 format")
                # Select relevant joints from NTU 25 to form COCO 17
                coco_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Approximate mapping
                joints3d = joints3d[:, coco_indices[:min(17, input_num_keypoints)], :]
            elif input_num_keypoints == 17 and self.num_keypoints == 25:
                # COCO 17 to NTU 25 - this shouldn't happen if we use get_ntu_joints
                logger.warning(f"Received 17 keypoints but model expects 25. "
                             f"Consider using get_ntu_joints() from SMPL model.")
                # Pad with zeros
                ntu_joints = np.zeros((joints3d.shape[0], 25, 3), dtype=joints3d.dtype)
                ntu_joints[:, :17, :] = joints3d[:, :17, :]
                joints3d = ntu_joints
            elif input_num_keypoints > self.num_keypoints:
                logger.warning(f"Expected {self.num_keypoints} keypoints, got {input_num_keypoints}. "
                             f"Will use first {self.num_keypoints} keypoints.")
                joints3d = joints3d[:, :self.num_keypoints, :]
            else:
                # Pad with zeros if we have fewer keypoints
                logger.warning(f"Expected {self.num_keypoints} keypoints, got {input_num_keypoints}. "
                             f"Will pad with zeros.")
                padded = np.zeros((joints3d.shape[0], self.num_keypoints, 3), dtype=joints3d.dtype)
                padded[:, :input_num_keypoints, :] = joints3d
                joints3d = padded
        
        # Reshape to pyskl format: (M=1, T=1, V, C=3)
        keypoint = joints3d.reshape(1, 1, -1, 3)
        
        return keypoint
    
    def add_skeleton_frame(self, joints3d: np.ndarray):
        """
        Add a skeleton frame to the buffer (real-time updates)
        
        Args:
            joints3d: 3D joints from MoViD, shape (num_joints, 3) or (batch, num_joints, 3)
        """
        keypoint = self.convert_movid_to_pyskl_format(joints3d)
        # Extract single frame: (1, V, 3)
        frame_keypoint = keypoint[0, 0, :, :]  # (V, 3)
        
        # Real-time updates: append every frame to the buffer
        self.skeleton_buffer.append(frame_keypoint)
        
        # Keep only the last window_size frames (sliding window)
        # This keeps the buffer aligned to the most recent window_size frames
        if len(self.skeleton_buffer) > self.window_size:
            self.skeleton_buffer.pop(0)
    
    def predict_action(self, joints3d: Optional[np.ndarray] = None) -> Tuple[int, float, str]:
        """
        Predict action from skeleton sequence
        
        Args:
            joints3d: Optional new frame to add before prediction. If None, uses buffer.
            
        Returns:
            Tuple of (class_idx, confidence, label_name)
        """
        if not PYSKL_AVAILABLE:
            return -1, 0.0, "pyskl_not_available"
        
        # Add new frame if provided
        if joints3d is not None:
            self.add_skeleton_frame(joints3d)
        
        # Need at least 15 frames for reliable prediction (for RNN/temporal models)
        min_frames_required = 30
        buffer_size = len(self.skeleton_buffer)
        
        if buffer_size < min_frames_required:
            logger.debug(f"Buffer has {buffer_size}/{min_frames_required} frames, waiting...")
            return -1, 0.0, f"buffering_{buffer_size}/{min_frames_required}"
        
        # Real-time prediction: use the latest buffer contents
        # The buffer is updated in real time because add_skeleton_frame is called on every frame
        # skeleton_buffer is a list with the newest frame at the end
        # Predict using the latest buffer contents
        
        # Convert buffer to pyskl format: (M=1, T, V, C=3)
        # Use the latest buffer contents (newest frame at the end of the list)
        num_frames = buffer_size
        keypoint = np.stack(self.skeleton_buffer, axis=0)  # (T, V, 3) - newest frames at the end
        keypoint = keypoint[np.newaxis, :, :, :]  # (1, T, V, 3)
        
        # Create fake annotation dict for pyskl
        fake_anno = dict(
            frame_dir='',
            label=-1,
            total_frames=num_frames,
            modality='Pose',
            start_index=0,
            keypoint=keypoint.astype(np.float32),
            # For NTU 3D skeletons, do NOT include keypoint_score (it causes assertion error)
            # keypoint_score is only for 2D skeletons
        )
        
        try:
            # Perform inference
            results = inference_recognizer(self.model, fake_anno)
            
            # results is a list of (class_idx, score) tuples, sorted by score
            if len(results) > 0:
                class_idx, confidence = results[0]
                label_name = self.label_map[class_idx] if self.label_map else f"class_{class_idx}"
                # Update the current prediction result
                self.current_action = label_name
                self.current_confidence = float(confidence)
                return class_idx, float(confidence), label_name
            else:
                # No prediction result; return the previous prediction
                return -1, self.current_confidence, self.current_action
        except Exception as e:
            import traceback
            error_msg = str(e) if e else "Unknown error"
            logger.error(f"Error during action recognition: {error_msg}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            logger.debug(f"Keypoint shape: {keypoint.shape}, num_frames: {num_frames}, num_keypoints: {self.num_keypoints}")
            return -1, 0.0, f"error: {error_msg}"
    
    def reset_buffer(self):
        """Reset the skeleton buffer and prediction state"""
        self.skeleton_buffer = []
        self.current_action = "waiting..."
        self.current_confidence = 0.0
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        return len(self.skeleton_buffer)
    
    def get_buffer_info(self) -> dict:
        """Get buffer information for monitoring"""
        return {
            'size': len(self.skeleton_buffer),
            'window_size': self.window_size,
            'num_keypoints': self.num_keypoints,
            'is_ready': len(self.skeleton_buffer) >= 15
        }
