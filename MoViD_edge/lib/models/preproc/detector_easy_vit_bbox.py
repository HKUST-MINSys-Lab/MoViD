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
        print(f"成功加载引擎: {trt_engine_path}")
        return engine
    except Exception as e:
        print(f"Failed to deserialize the engine: {e}")
        return None
    
def allocate_buffers(engine):
    """优化的缓冲区分配，使用更少的内存"""
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
        
        # 使用页锁定内存，但分配最小必要大小
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
        
        # 延迟初始化TensorRT组件，在其他模型加载后再初始化
        self.trt_initialized = False
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        
        print("DetectionModel基础初始化完成，TensorRT将在需要时初始化")

    def lazy_init_trt(self):
        """延迟初始化TensorRT组件"""
        if self.trt_initialized:
            return
            
        print("开始初始化TensorRT组件...")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU内存清理前: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        try:
            # 初始化TensorRT相关组件
            self.context, self.inputs, self.outputs, self.bindings = self.setup_trt(TRT_PATH)
            
            # 创建CUDA stream用于异步处理
            self.stream = cuda.Stream()
            
            # 获取输出形状
            if len(self.outputs) > 0:
                self.output_shape = self.outputs[0]['shape']
            else:
                self.output_shape = (1, 17, 64, 48)
            
            self.output_size = np.prod(self.output_shape)
            self.trt_initialized = True
            
            print("TensorRT组件初始化完成!")
            if torch.cuda.is_available():
                print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                
        except Exception as e:
            print(f"TensorRT初始化失败: {e}")
            raise

    def setup_trt(self, trt_path):
        """设置TensorRT引擎和缓冲区"""
        logger = trt.Logger(trt.Logger.ERROR)
        trt_runtime = trt.Runtime(logger)
        trt_engine = load_engine(trt_path)
        
        if trt_engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        # 分配缓冲区
        inputs, outputs, bindings = allocate_buffers(trt_engine)
        
        # 创建执行上下文
        context = trt_engine.create_execution_context()
        
        return context, inputs, outputs, bindings
    
    def inference_stream(self, img):
        """使用CUDA stream进行异步推理"""
        # 确保TensorRT已初始化
        if not self.trt_initialized:
            self.lazy_init_trt()
            
        try:
            org_h, org_w = img.shape[:2]
            if org_h == self.WIDTH or org_w == self.HEIGHT:
                # 如果图像已经是目标大小，直接使用
                img_input = img
            else:
                img_input = cv2.resize(img, (self.WIDTH,self.HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            # 转换为模型输入格式: (batch, channels, height, width)
            img_input = img_input.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
            img_input_contiguous = np.ascontiguousarray(img_input)

            
            # 异步拷贝输入数据到GPU
            if len(self.inputs) > 0:
                np.copyto(self.inputs[0]['host'], img_input_contiguous.ravel())
                cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # 异步执行推理
            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not success:
                return None
            
            # 异步拷贝输出数据到CPU
            if len(self.outputs) > 0:
                cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            
            # 同步等待所有异步操作完成
            self.stream.synchronize()
            
            # 获取输出结果
            if len(self.outputs) > 0:
                output_host = self.outputs[0]['host'].reshape(self.output_shape)
                heatmaps = output_host
                
                # 修正center和scale的计算
                center = np.array([[org_w//2, org_h//2]])
                scale = np.array([[org_w, org_h]])
                
                # 从热图中提取关键点
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
            print(f"推理出错: {e}")
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
        """使用stream方式进行追踪"""
        # 保持最近3帧的结果（减少内存使用）
        for key in ['id', 'frame_id', 'keypoints','bbox']:
            self.tracking_results[key] = list(self.tracking_results[key][-10:]) if len(self.tracking_results[key]) > 0 else []
        bbox = self.tracking_results['bbox'][-1] if len(self.tracking_results['bbox']) > 0 else None

        # 如果有bbox，使用bbox裁剪后送入推理
        if bbox is not None:
            # bbox 形状: (1, 3) -> (cx, cy, s)
            if bbox.shape == (1, 3):
                cx, cy, s = bbox[0]
            else:
                cx, cy, s = bbox
            # 还原为xyxy
            w = h = s * 200 / 1.05  # 反推回原始宽高
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            crop_img = img[y1:y2, x1:x2]
            crop_img_resized = cv2.resize(crop_img, (self.WIDTH, self.HEIGHT))
            keypoints_result = self.inference_stream(crop_img_resized)
        else:
            keypoints_result = self.inference_stream(img)
        
        if keypoints_result is not None and len(keypoints_result) > 0:
            # 取第一个人的关键点
            kpts = keypoints_result[0]
            
            if kpts.shape[0] > 17:
                kpts = kpts[:17]  # 只保留前17个关键点
            
            # 如果用bbox裁剪过，需要把关键点映射回原图
            if bbox is not None:
                scale_x = (x2 - x1) / self.WIDTH
                scale_y = (y2 - y1) / self.HEIGHT
                kpts[:, 0] = kpts[:, 0] * scale_x + x1
                kpts[:, 1] = kpts[:, 1] * scale_y + y1

            # 检查有效关键点数量
            valid = kpts[:, 2] > VIS_THRESH
            if valid.sum() < MINIMUM_JOINTS:
                self.frame_id += 1
                return  # 跳过该帧

            # 保存追踪结果
            self.tracking_results['id'].append(0)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['keypoints'].append(kpts)

            for key in ['id', 'frame_id', 'keypoints']:
                if len(self.tracking_results[key]) > 0:
                    self.tracking_results[key] = np.array(self.tracking_results[key])
            self.compute_bboxes_from_keypoints()    
        self.frame_id += 1
    

    def process(self, fps):
        """处理追踪结果"""
        if len(self.tracking_results['id']) == 0:
            return {} 

        output = defaultdict(lambda: defaultdict(list))

        if len(self.tracking_results['id']) > 0:
            ids = np.unique(self.tracking_results['id'])
            for _id in ids:
                idxs = np.where(self.tracking_results['id'] == _id)[0]
                for key, val in self.tracking_results.items():
                    if key == 'id': continue
                    output[_id][key] = val[idxs]
            
            # 简化的平滑处理，减少计算量
            ids = list(output.keys())
            for _id in ids:
                if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                    del output[_id]
                    continue
                
                # 使用较小的kernel减少计算量
                kernel = min(5, len(output[_id]['bbox']))
                if kernel >= 3 and kernel < len(output[_id]['bbox']):
                    smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
                    output[_id]['bbox'] = smoothed_bbox

        return output
    
    def cleanup(self):
        """主动清理资源"""
        if self.trt_initialized:
            try:
                if self.stream:
                    self.stream.synchronize()
                    
                # 清理CUDA内存
                if self.inputs:
                    for inp in self.inputs:
                        if 'device' in inp:
                            inp['device'].free()
                            
                if self.outputs:
                    for out in self.outputs:
                        if 'device' in out:
                            out['device'].free()
                            
                print("TensorRT资源已清理")
            except:
                pass
                
        # 清理Python内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """析构函数"""
        self.cleanup()


# 内存监控工具
def print_memory_usage(prefix=""):
    """打印当前内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"{prefix}GPU内存 - 已分配: {allocated:.1f}MB, 缓存: {cached:.1f}MB")
    
    import psutil
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024**2
    print(f"{prefix}CPU内存: {cpu_mem:.1f}MB")

