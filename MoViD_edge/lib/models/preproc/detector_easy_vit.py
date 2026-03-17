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

from ...utils.top_down_eval import keypoints_from_heatmaps

TRT_PATH = "/home/dlq/easy_ViTPose/ckpts/vitpose-l-coco.trt"

VIS_THRESH = 0.3
BBOX_CONF = 0.3
TRACKING_THR = 0.1
MINIMUM_FRMAES = 0
MINIMUM_JOINTS = 6

def load_engine(trt_engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    try:
        with open(trt_engine_path,'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        print(f"Successfully loaded engine: {trt_engine_path}")
        return engine
    except Exception as e:
        print(f"Failed to deserialize the engine: {e}")
        return None
    
def allocate_buffers(engine):
    """Optimized buffer allocation using less memory"""
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
        
        # Use page-locked memory while allocating only the minimum required size
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
        
        # Lazily initialize TensorRT components after other models finish loading
        self.trt_initialized = False
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        
        print("Base DetectionModel initialization finished; TensorRT will be initialized when needed")

    def lazy_init_trt(self):
        """Lazily initialize TensorRT components"""
        if self.trt_initialized:
            return
            
        print("Starting TensorRT component initialization...")
        
        # Clear the GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        try:
            # Initialize TensorRT-related components
            self.context, self.inputs, self.outputs, self.bindings = self.setup_trt(TRT_PATH)
            
            # Create a CUDA stream for asynchronous processing
            self.stream = cuda.Stream()
            
            # Get the output shape
            if len(self.outputs) > 0:
                self.output_shape = self.outputs[0]['shape']
            else:
                self.output_shape = (1, 17, 64, 48)
            
            self.output_size = np.prod(self.output_shape)
            self.trt_initialized = True
            
            print("TensorRT component initialization complete!")
            if torch.cuda.is_available():
                print(f"GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                
        except Exception as e:
            print(f"TensorRT initialization failed: {e}")
            raise

    def setup_trt(self, trt_path):
        """Set up the TensorRT engine and buffers"""
        logger = trt.Logger(trt.Logger.ERROR)
        trt_runtime = trt.Runtime(logger)
        trt_engine = load_engine(trt_path)
        
        if trt_engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        # Allocate buffers
        inputs, outputs, bindings = allocate_buffers(trt_engine)
        
        # Create the execution context
        context = trt_engine.create_execution_context()
        
        return context, inputs, outputs, bindings
    
    def inference_stream(self, img):
        """Run asynchronous inference with a CUDA stream"""
        # Ensure TensorRT has been initialized
        if not self.trt_initialized:
            self.lazy_init_trt()
            
        try:
            org_h, org_w = img.shape[:2]
            
            img_input = cv2.resize(img, (self.WIDTH,self.HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            # Convert to the model input format: (batch, channels, height, width)
            img_input = img_input.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
            img_input_contiguous = np.ascontiguousarray(img_input)

            
            # Asynchronously copy input data to the GPU
            if len(self.inputs) > 0:
                np.copyto(self.inputs[0]['host'], img_input_contiguous.ravel())
                cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # Launch inference asynchronously
            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not success:
                return None
            
            # Asynchronously copy output data back to the CPU
            if len(self.outputs) > 0:
                cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            
            # Synchronize and wait for all asynchronous work to finish
            self.stream.synchronize()
            
            # Fetch the output results
            if len(self.outputs) > 0:
                output_host = self.outputs[0]['host'].reshape(self.output_shape)
                heatmaps = output_host
                
                # Correct the center and scale computation
                center = np.array([[org_w//2, org_h//2]])
                scale = np.array([[org_w, org_h]])
                
                # Extract keypoints from heatmaps
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
            print(f"Inference failed: {e}")
            return None
        
    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
        return np.array([[cx, cy, scale]])

    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        if len(self.tracking_results['keypoints']) == 0:
            return
            
        X = self.tracking_results['keypoints'].copy()
        mask = X[..., -1] > VIS_THRESH

        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            if m.sum() == 0:
                continue
            bb = [kp[m, 0].min(), kp[m, 1].min(),
                  kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = [(bb[2]+bb[0])/2, (bb[3]+bb[1])/2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            s = np.stack((bb_w, bb_h)).max()
            bb = np.array((cx, cy, s))
            bbox[i] = bb
        
        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results['bbox'] = bbox

    def track(self, img, fps, length):
        """Track with the streaming path"""
        # Keep only the most recent 3 frames of results to reduce memory usage
        for key in ['id', 'frame_id', 'keypoints','bbox']:
            self.tracking_results[key] = list(self.tracking_results[key][-10:]) if len(self.tracking_results[key]) > 0 else []

        # Use the streaming path for inference
        keypoints_result = self.inference_stream(img)
        
        if keypoints_result is not None and len(keypoints_result) > 0:
            # Take the keypoints for the first person
            kpts = keypoints_result[0]
            
            if kpts.shape[0] > 17:
                kpts = kpts[:17]  # Keep only the first 17 keypoints
            
            # Check the number of valid keypoints
            valid = kpts[:, 2] > VIS_THRESH
            if valid.sum() < MINIMUM_JOINTS:
                self.frame_id += 1
                return  # Skip this frame
            
            # Compute the bounding box
            x1, y1 = kpts[valid, 0].min(), kpts[valid, 1].min()
            x2, y2 = kpts[valid, 0].max(), kpts[valid, 1].max()
            bbox = np.array([x1, y1, x2, y2])
            bbox = self.xyxy_to_cxcys(bbox, s_factor=1.05)

            # Store tracking results
            self.tracking_results['id'].append(0)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(kpts)
        
        self.frame_id += 1
        
        # Periodically free memory
        if self.frame_id % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process(self, fps):
        """Process tracking results"""
        if len(self.tracking_results['id']) == 0:
            return {}
            
        for key in ['id', 'frame_id', 'keypoints']:
            if len(self.tracking_results[key]) > 0:
                self.tracking_results[key] = np.array(self.tracking_results[key])
                
        self.compute_bboxes_from_keypoints()     

        output = defaultdict(lambda: defaultdict(list))

        if len(self.tracking_results['id']) > 0:
            ids = np.unique(self.tracking_results['id'])
            for _id in ids:
                idxs = np.where(self.tracking_results['id'] == _id)[0]
                for key, val in self.tracking_results.items():
                    if key == 'id': continue
                    output[_id][key] = val[idxs]
            
            # Use simplified smoothing to reduce computation
            ids = list(output.keys())
            for _id in ids:
                if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                    del output[_id]
                    continue
                
                # Use a smaller kernel to reduce computation
                kernel = min(5, len(output[_id]['bbox']))
                if kernel >= 3 and kernel < len(output[_id]['bbox']):
                    smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
                    output[_id]['bbox'] = smoothed_bbox

        return output
    
    def cleanup(self):
        """Actively release resources"""
        if self.trt_initialized:
            try:
                if self.stream:
                    self.stream.synchronize()
                    
                # Free CUDA memory
                if self.inputs:
                    for inp in self.inputs:
                        if 'device' in inp:
                            inp['device'].free()
                            
                if self.outputs:
                    for out in self.outputs:
                        if 'device' in out:
                            out['device'].free()
                            
                print("TensorRT resources have been released")
            except:
                pass
                
        # Free Python memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Destructor"""
        self.cleanup()


# Memory-monitoring utility
def print_memory_usage(prefix=""):
    """Print the current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"{prefix}GPU memory - allocated: {allocated:.1f}MB, cached: {cached:.1f}MB")
    
    import psutil
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024**2
    print(f"{prefix}CPU memory: {cpu_mem:.1f}MB")

