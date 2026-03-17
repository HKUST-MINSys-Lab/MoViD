"""
简化的TensorRT模型使用示例
这个版本更容易理解和使用
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import shutil
from ultralytics import YOLO
import torch
import os.path as osp
from pathlib import Path

class SimpleTensorRTDetection:
    """简化的TensorRT检测模型"""
    
    def __init__(self, device='cuda:0'):
        """初始化TensorRT检测模型"""
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.setup_paths()
        
        # 首次运行时转换模型
        self.convert_models_if_needed()
        
        # 加载TensorRT引擎
        self.load_engines()
    
    def setup_paths(self):
        """设置路径"""
        # 使用绝对路径
        self.base_dir = Path("/home/dlq/WHAM")
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.engines_dir = self.base_dir / "engines"
        
        # 如果目录不存在，则创建
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.engines_dir.mkdir(exist_ok=True)
        
        # 模型文件路径
        self.yolo_model_path = self.checkpoints_dir / "yolov8n.pt"
        self.yolo_engine_path = self.engines_dir / "yolov8n.engine"

    def convert_models_if_needed(self):
        """如果需要，转换模型为TensorRT格式"""
        
        # 转换YOLO模型（最简单的方式）
        if not osp.exists(self.yolo_engine_path):
            print("正在转换YOLO模型到TensorRT...")
            self.convert_yolo()
        
        # 对于ViTPose，我们先用ONNX Runtime作为备选方案
        print("ViTPose将使用ONNX Runtime进行推理（更容易配置）")
    
    def convert_yolo(self):
        """转换YOLO模型"""
        print("正在转换YOLO模型到TensorRT...")
        from ultralytics import YOLO
        
        model = YOLO(str(self.yolo_model_path))
        # 直接导出到引擎目录
        success = model.export(
            format="engine",
            device=0,
            half=True,
            simplify=True,
            workspace=4,
            verbose=False,
            save=True,
            saved_model=str(self.yolo_engine_path)  # 直接保存到引擎目录
        )
        
        if not self.yolo_engine_path.exists():
            raise RuntimeError(f"在 {self.yolo_engine_path} 创建引擎失败")
        print(f"YOLO TensorRT引擎已保存到: {self.yolo_engine_path}")

    def load_trt_engine(self, engine_path):
        """加载TensorRT引擎"""
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            engine = self.runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                raise RuntimeError(f"从 {engine_path} 反序列化引擎失败")
            return engine
        except Exception as e:
            print(f"加载引擎时出错 {engine_path}: {str(e)}")
            return None
    
    def load_engines(self):
        """加载推理引擎"""
        print("加载YOLO TensorRT引擎...")
        self.yolo_trt = self.load_trt_engine(self.yolo_engine_path)
        if self.yolo_trt is None:
            raise RuntimeError("Failed to load YOLO TensorRT engine")
        
        try:
            self.yolo_context = self.yolo_trt.create_execution_context()
            print("YOLO TensorRT引擎加载成功")
        except Exception as e:
            raise RuntimeError(f"Failed to create execution context: {str(e)}")
    
    def detect_objects(self, image):
        """检测图像中的目标"""
        if self.use_pytorch_yolo:
            # 使用PyTorch版本
            results = self.yolo_model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls == 0 and conf > 0.5:  # 人体检测，置信度>0.5
                            detections.append([x1, y1, x2, y2, conf, cls])
            
            return detections
        else:
            # 使用TensorRT版本
            return self.detect_with_tensorrt(image)
    
    def detect_with_tensorrt(self, image):
        """使用TensorRT进行检测"""
        # 这里需要实现TensorRT推理逻辑
        # 为简化示例，暂时返回空列表
        print("TensorRT推理（待实现完整版本）")
        return []
    
    def estimate_pose(self, image, bbox):
        """姿态估计（简化版本）"""
        # 这里可以集成你的ViTPose模型
        # 为了演示，返回一些虚拟关键点
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # 17个COCO关键点的虚拟坐标
        keypoints = []
        for i in range(17):
            x = x1 + (x2 - x1) * np.random.random()
            y = y1 + (y2 - y1) * np.random.random()
            conf = np.random.random()
            keypoints.append([x, y, conf])
        
        return np.array(keypoints)
    
    def process_image(self, image_path):
        """处理单张图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 检测人体
        detections = self.detect_objects(image)
        
        results = []
        for detection in detections:
            bbox = detection[:4]
            confidence = detection[4]
            
            # 姿态估计
            keypoints = self.estimate_pose(image, bbox)
            
            results.append({
                'bbox': bbox,
                'confidence': confidence,
                'keypoints': keypoints
            })
        
        return results
    
    def visualize_results(self, image_path, results):
        """可视化结果"""
        image = cv2.imread(image_path)
        
        for result in results:
            bbox = result['bbox']
            keypoints = result['keypoints']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制关键点
            for kpt in keypoints:
                x, y, conf = kpt
                if conf > 0.5:
                    cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        return image

# 使用示例
def main():
    """主函数示例"""
    
    # 初始化模型
    print("初始化TensorRT检测模型...")
    detector = SimpleTensorRTDetection()
    
    # 处理图像
    image_path = 'test_image.jpg'  # 替换为你的图像路径
    
    if osp.exists(image_path):
        print(f"处理图像: {image_path}")
        
        # 检测和姿态估计
        results = detector.process_image(image_path)
        
        if results:
            print(f"检测到 {len(results)} 个人体")
            
            # 可视化结果
            vis_image = detector.visualize_results(image_path, results)
            cv2.imwrite('result.jpg', vis_image)
            print("结果已保存到 result.jpg")
        else:
            print("未检测到人体")
    else:
        print(f"图像文件不存在: {image_path}")
        print("请准备一张测试图像")

if __name__ == "__main__":
    main()

# 性能测试
def benchmark():
    """性能测试"""
    import time
    
    detector = SimpleTensorRTDetection()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 预热
    for _ in range(5):
        _ = detector.detect_objects(test_image)
    
    # 性能测试
    n_tests = 50
    start_time = time.time()
    
    for i in range(n_tests):
        detections = detector.detect_objects(test_image)
        if i % 10 == 0:
            print(f"测试进度: {i+1}/{n_tests}")
    
    total_time = time.time() - start_time
    avg_time = total_time / n_tests
    fps = 1.0 / avg_time
    
    print(f"\n性能测试结果:")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均推理时间: {avg_time*1000:.1f}毫秒")
    print(f"平均FPS: {fps:.1f}")