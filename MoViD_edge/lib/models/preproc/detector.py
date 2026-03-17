import torch
import cv2
import numpy as np
import json
import time
import gc
from collections import defaultdict
import scipy.signal as signal
np.bool = bool
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import concurrent.futures

from ...utils.top_down_eval import keypoints_from_heatmaps

TRT_PATH = "/home/dlq/easy_ViTPose/ckpts/vitpose-b-coco.trt"

VIS_THRESH = 0.3
BBOX_CONF = 0.3
TRACKING_THR = 0.1
MINIMUM_FRMAES = 0
MINIMUM_JOINTS = 10

# Low confidence threshold, below which a full image redetection is used
LOW_CONFIDENCE_THRESH = 0.4
MIN_VALID_KEYPOINTS = 15  # At least 15 valid keypoints are needed

def load_engine(trt_engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    try:
        with open(trt_engine_path,'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        print(f"load: {trt_engine_path} successfully")
        return engine
    except Exception as e:
        print(f"Failed to deserialize the engine: {e}")
        return None
    
def allocate_buffers(engine):
    """Optimized buffer allocation to use less memory"""
    inputs = []
    outputs = []
    bindings = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        mode = engine.get_tensor_mode(name)
        is_input = (mode == trt.TensorIOMode.INPUT)

        shape = engine.get_tensor_shape(name)
        print(f"Binding {i}: Name={name}, Shape={shape}, Dtype={dtype}, Input={is_input}")
        size = trt.volume(shape)
        
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if is_input:
            inputs.append({'name':name,'host':host_mem,'device':device_mem,'shape':shape})
        else:
            outputs.append({'name':name,'host':host_mem,'device':device_mem,'shape':shape})
    
    return inputs, outputs, bindings

class DetectionModel(object):
    def __init__(self, device='cuda'):
        print("Initializing Memory Optimized DetectionModel...")
        self.device = device
        self.WIDTH = 192
        self.HEIGHT = 256
        self.next_id = 0
        self.frame_id = 0
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': []
        }
        # Accumulate all results for batch processing (demo.py)
        self.all_tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': []
        }
        
        # Record the number of consecutive low-confidence frames
        self.low_confidence_count = 0
        self.max_low_confidence_frames = 3  # Use full image after 3 consecutive low-confidence frames
        
        self.trt_initialized = False
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def lazy_init_trt(self):
        """Lazy initialization of TensorRT components"""
        if self.trt_initialized:
            return
            
        try:
            self.context, self.inputs, self.outputs, self.bindings = self.setup_trt(TRT_PATH)
            self.stream = cuda.Stream()
            
            if len(self.outputs) > 0:
                self.output_shape = self.outputs[0]['shape']
            else:
                self.output_shape = (1, 17, 64, 48)
            
            self.output_size = np.prod(self.output_shape)
            self.trt_initialized = True
                
        except Exception as e:
            print(f"TensorRT initialization failed: {e}")
            raise

    def setup_trt(self, trt_path):
        """Set up TensorRT engine and buffers"""
        logger = trt.Logger(trt.Logger.ERROR)
        trt_runtime = trt.Runtime(logger)
        trt_engine = load_engine(trt_path)
        
        if trt_engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        inputs, outputs, bindings = allocate_buffers(trt_engine)
        context = trt_engine.create_execution_context()
        
        return context, inputs, outputs, bindings
    
    def inference_stream(self, resized_img):
        """
        Asynchronous inference using CUDA stream.
        This function now assumes the input 'resized_img' is already at the model's
        required dimensions (WIDTH, HEIGHT) and returns keypoints relative to that size.
        """
        if not self.trt_initialized:
            self.lazy_init_trt()
        try:
            # The input image is already the correct size, no need to resize again.
            img_input = resized_img.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
            img_input_contiguous = np.ascontiguousarray(img_input)

            if len(self.inputs) > 0:
                cuda.memcpy_htod_async(self.inputs[0]['device'], img_input_contiguous, self.stream)

            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not success:
                return None

            if len(self.outputs) > 0:
                cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            if len(self.outputs) > 0:
                output_host = self.outputs[0]['host'].reshape(self.output_shape)
                heatmaps = output_host
                
                # ALWAYS use the model's dimensions for center and scale transformation
                center = np.array([[self.WIDTH / 2, self.HEIGHT / 2]])
                scale = np.array([[self.WIDTH, self.HEIGHT]])

                points, prob = keypoints_from_heatmaps(
                    heatmaps=heatmaps,
                    center=center,
                    scale=scale,
                    unbiased=True,
                    use_udp=True
                )
                points = np.concatenate([points, prob], axis=2)
                return points
            else:
                return None
        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def check_keypoints_quality(self, kpts):
        """
        Check keypoint quality
        Returns: (is_good_quality, avg_confidence, num_valid)
        """
        if kpts is None or len(kpts) == 0:
            return False, 0.0, 0
        
        # Calculate number of valid keypoints and average confidence
        valid_mask = kpts[:, 2] > VIS_THRESH
        num_valid = valid_mask.sum()
        
        if num_valid > 0:
            avg_confidence = kpts[valid_mask, 2].mean()
        else:
            avg_confidence = 0.0
        
        # Judge quality: needs enough valid points and average confidence not too low
        is_good = (num_valid >= MIN_VALID_KEYPOINTS and 
                   avg_confidence > LOW_CONFIDENCE_THRESH)
        
        return is_good, avg_confidence, num_valid

    def compute_bboxes_from_keypoints(self, s_factor=1.25):
        X = np.asarray(self.tracking_results['keypoints'])
        if X.shape[0] == 0:
            return
        mask = X[..., -1] > VIS_THRESH
        valid = mask.sum(axis=1) > 0
        bbox = np.zeros((X.shape[0], 3))
        if valid.any():
            x_min = np.where(mask, X[..., 0], np.inf).min(axis=1)
            y_min = np.where(mask, X[..., 1], np.inf).min(axis=1)
            x_max = np.where(mask, X[..., 0], -np.inf).max(axis=1)
            y_max = np.where(mask, X[..., 1], -np.inf).max(axis=1)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            s = np.maximum(x_max - x_min, y_max - y_min)
            bbox[valid, 0] = cx[valid]
            bbox[valid, 1] = cy[valid]
            bbox[valid, 2] = s[valid]
        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results['bbox'] = bbox

    def match_id_with_kp3d(self, kpts3d, thr=150):
        """
        Use 3D keypoints to determine if it is the same person
        """
        if len(self.tracking_results['id']) == 0:
            new_id = self.next_id
            self.next_id += 1
            return new_id

        last_kpts = self.tracking_results['keypoints'][-1]
        if last_kpts.shape[1] == 3:
            mask = (kpts3d[:,2] > VIS_THRESH) & (last_kpts[:,2] > VIS_THRESH)
            if mask.sum() > 0:
                dist = np.linalg.norm(kpts3d[mask,:2] - last_kpts[mask,:2], axis=1).mean()
            else:
                dist = 1e9
        else:
            dist = 1e9

        if dist < thr:
            return self.tracking_results['id'][-1]
        else:
            new_id = self.next_id
            self.next_id += 1
            return new_id

    def track(self, img, fps, length, use_full_frame_fallback=True):
        """
        Tracking using stream, with support for falling back to full-frame detection on low confidence.
        
        Args:
            img: Input image
            fps: Frame rate
            length: Total frames
            use_full_frame_fallback: Whether to use full-frame redetection on low confidence
        """
        org_h, org_w = img.shape[:2]
        
        # Keep results of the last 10 frames
        for key in ['id', 'frame_id', 'keypoints','bbox']:
            self.tracking_results[key] = list(self.tracking_results[key][-10:]) if len(self.tracking_results[key]) > 0 else []
        
        bbox = self.tracking_results['bbox'][-1] if len(self.tracking_results['bbox']) > 0 else None
        
        keypoints_result = None
        used_full_frame = False
        
        # Attempt 1: Use bbox crop (if available)
        if bbox is not None:
            if bbox.shape == (1, 3):
                cx, cy, s = bbox[0]
            else:
                cx, cy, s = bbox

            w = h = s * 200 / 1.05 * 1.5  # Enlarge slightly
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2) # <-- BUG FIX: This line was missing
            
            # Clamp coordinates to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(org_w, x2)
            y2 = min(org_h, y2)
            
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size > 0:
                crop_img_resized = cv2.resize(crop_img, (self.WIDTH, self.HEIGHT))
                keypoints_result = self.inference_stream(crop_img_resized)
                
                # Check keypoint quality
                if keypoints_result is not None and len(keypoints_result) > 0:
                    kpts = keypoints_result[0]
                    is_good, avg_conf, num_valid = self.check_keypoints_quality(kpts)
                    
                    if not is_good:
                        self.low_confidence_count += 1
                        print(f"Frame {self.frame_id}: Low quality detection - "
                              f"Valid points: {num_valid}/{len(kpts)}, "
                              f"Avg confidence: {avg_conf:.3f}, "
                              f"Count: {self.low_confidence_count}/{self.max_low_confidence_frames}")
                        
                        # If low confidence for several consecutive frames, trigger full-frame redetection
                        if (use_full_frame_fallback and 
                            self.low_confidence_count >= self.max_low_confidence_frames):
                            print(f"Frame {self.frame_id}: Falling back to full frame detection")
                            keypoints_result = None # Reset to trigger full-frame logic
                            bbox = None # Force entry into the next block
                    else:
                        # Good quality, reset counter
                        self.low_confidence_count = 0
                        print(f"Frame {self.frame_id}: Good quality - "
                              f"Valid points: {num_valid}/{len(kpts)}, "
                              f"Avg confidence: {avg_conf:.3f}")
        
        # Full Frame Detection Logic (if no bbox or fallback is triggered)
        if bbox is None:
            full_frame_resized = cv2.resize(img, (self.WIDTH, self.HEIGHT))
            keypoints_result = self.inference_stream(full_frame_resized)
            used_full_frame = True
            if keypoints_result is not None:
                self.low_confidence_count = 0 # Reset counter on successful full detection
        
        # Process detection results
        if keypoints_result is not None and len(keypoints_result) > 0:
            kpts = keypoints_result[0]
            
            if kpts.shape[0] > 17:
                kpts = kpts[:17]
            
            # Map keypoints back to original image coordinates
            if not used_full_frame:
                # Mapped from the cropped region
                scale_x = (x2 - x1) / self.WIDTH
                scale_y = (y2 - y1) / self.HEIGHT
                kpts[:, 0] = kpts[:, 0] * scale_x + x1
                kpts[:, 1] = kpts[:, 1] * scale_y + y1
            else:
                # Mapped from the resized full frame
                scale_x = org_w / self.WIDTH
                scale_y = org_h / self.HEIGHT
                kpts[:, 0] = kpts[:, 0] * scale_x
                kpts[:, 1] = kpts[:, 1] * scale_y

            # Final check on valid keypoints
            valid = kpts[:, 2] > VIS_THRESH
            if valid.sum() < MINIMUM_JOINTS:
                print(f"Frame {self.frame_id}: Too few valid keypoints ({valid.sum()}), skipping")
                self.frame_id += 1
                return

            subject_id = self.match_id_with_kp3d(kpts)
            self.tracking_results['id'].append(subject_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['keypoints'].append(kpts)

            # Accumulate for batch processing
            self.all_tracking_results['id'].append(subject_id)
            self.all_tracking_results['frame_id'].append(self.frame_id)
            self.all_tracking_results['keypoints'].append(kpts.copy())

            for key in ['id', 'frame_id', 'keypoints']:
                if len(self.tracking_results[key]) > 0:
                    self.tracking_results[key] = np.array(self.tracking_results[key])
            self.compute_bboxes_from_keypoints()

            # Also accumulate bbox
            if len(self.tracking_results['bbox']) > 0:
                bbox_arr = np.array(self.tracking_results['bbox'])
                self.all_tracking_results['bbox'].append(bbox_arr[-1].copy())
        else:
            # If still no detection, increment low confidence counter
            if use_full_frame_fallback:
                self.low_confidence_count += 1
                print(f"Frame {self.frame_id}: No detection, count: {self.low_confidence_count}")
        
        self.frame_id += 1

    def process(self, fps):
        """Process tracking results using all accumulated data"""
        # Use all_tracking_results (full history) instead of tracking_results (last 10)
        all_results = {k: np.array(v) for k, v in self.all_tracking_results.items() if len(v) > 0}

        if len(all_results.get('id', [])) == 0:
            return {}

        output = defaultdict(lambda: defaultdict(list))

        ids = np.unique(all_results['id'])
        for _id in ids:
            idxs = np.where(all_results['id'] == _id)[0]
            for key, val in all_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]

            if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                del output[_id]
                continue

            kernel = min(5, len(output[_id]['bbox']))
            if kernel >= 3 and kernel < len(output[_id]['bbox']):
                smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
                output[_id]['bbox'] = smoothed_bbox

        print(f"Process: {len(output)} subjects, total frames: {sum(len(v['frame_id']) for v in output.values())}")
        return output
    
    def cleanup(self):
        """Actively clean up resources"""
        if not getattr(self, 'trt_initialized', False):
            return
        try:
            if self.stream:
                self.stream.synchronize()
            if self.inputs:
                for inp in self.inputs:
                    if inp and isinstance(inp, dict) and 'device' in inp and inp['device'] is not None:
                        try:
                            inp['device'].free()
                        except Exception:
                            pass
            if self.outputs:
                for out in self.outputs:
                    if out and isinstance(out, dict) and 'device' in out and out['device'] is not None:
                        try:
                            out['device'].free()
                        except Exception:
                            pass
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.trt_initialized = False

    def __del__(self):
        """Destructor - avoid a segfault when the interpreter exits"""
        try:
            self.cleanup()
        except Exception:
            pass