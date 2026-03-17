"""
Simplified TensorRT model usage example
This version is easier to understand and use
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
    """Simplified TensorRT detection model"""
    
    def __init__(self, device='cuda:0'):
        """Initialize the TensorRT detection model"""
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.setup_paths()
        
        # Convert the model on first use
        self.convert_models_if_needed()
        
        # Load the TensorRT engine
        self.load_engines()
    
    def setup_paths(self):
        """Configure paths"""
        # Use absolute paths
        self.base_dir = Path("/home/dlq/MoViD")
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.engines_dir = self.base_dir / "engines"
        
        # Create the directory if it does not exist
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.engines_dir.mkdir(exist_ok=True)
        
        # Model file paths
        self.yolo_model_path = self.checkpoints_dir / "yolov8n.pt"
        self.yolo_engine_path = self.engines_dir / "yolov8n.engine"

    def convert_models_if_needed(self):
        """Convert the model to TensorRT format when needed"""
        
        # Convert the YOLO model using the simplest path
        if not osp.exists(self.yolo_engine_path):
            print("Converting the YOLO model to TensorRT...")
            self.convert_yolo()
        
        # For ViTPose, start with ONNX Runtime as a fallback option
        print("ViTPose will use ONNX Runtime for inference because it is easier to configure")
    
    def convert_yolo(self):
        """Convert the YOLO model"""
        print("Converting the YOLO model to TensorRT...")
        from ultralytics import YOLO
        
        model = YOLO(str(self.yolo_model_path))
        # Export directly into the engine directory
        success = model.export(
            format="engine",
            device=0,
            half=True,
            simplify=True,
            workspace=4,
            verbose=False,
            save=True,
            saved_model=str(self.yolo_engine_path)  # save directly into the engine directory
        )
        
        if not self.yolo_engine_path.exists():
            raise RuntimeError(f"at  {self.yolo_engine_path} failed to create the engine")
        print(f"YOLO TensorRT engine saved to: {self.yolo_engine_path}")

    def load_trt_engine(self, engine_path):
        """Load the TensorRT engine"""
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            engine = self.runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                raise RuntimeError(f"from  {engine_path} failed to deserialize the engine")
            return engine
        except Exception as e:
            print(f"Error while loading the engine {engine_path}: {str(e)}")
            return None
    
    def load_engines(self):
        """Load the inference engine"""
        print("Loading the YOLO TensorRT engine...")
        self.yolo_trt = self.load_trt_engine(self.yolo_engine_path)
        if self.yolo_trt is None:
            raise RuntimeError("Failed to load YOLO TensorRT engine")
        
        try:
            self.yolo_context = self.yolo_trt.create_execution_context()
            print("YOLO TensorRT engine loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to create execution context: {str(e)}")
    
    def detect_objects(self, image):
        """Detect objects in the image"""
        if self.use_pytorch_yolo:
            # Use the PyTorch version
            results = self.yolo_model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls == 0 and conf > 0.5:  # person detection with confidence > 0.5
                            detections.append([x1, y1, x2, y2, conf, cls])
            
            return detections
        else:
            # Use the TensorRT version
            return self.detect_with_tensorrt(image)
    
    def detect_with_tensorrt(self, image):
        """Run detection with TensorRT"""
        # TensorRT inference logic still needs to be implemented here
        # Return an empty list for now to keep the example simple
        print("TensorRT inference (full implementation pending)")
        return []
    
    def estimate_pose(self, image, bbox):
        """Pose estimation (simplified version)"""
        # Integrate your ViTPose model here
        # Return a few dummy keypoints for demonstration
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # dummy coordinates for 17 COCO keypoints
        keypoints = []
        for i in range(17):
            x = x1 + (x2 - x1) * np.random.random()
            y = y1 + (y2 - y1) * np.random.random()
            conf = np.random.random()
            keypoints.append([x, y, conf])
        
        return np.array(keypoints)
    
    def process_image(self, image_path):
        """Process a single image"""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Unable to read the image: {image_path}")
            return None
        
        # Detect people
        detections = self.detect_objects(image)
        
        results = []
        for detection in detections:
            bbox = detection[:4]
            confidence = detection[4]
            
            # Pose estimation
            keypoints = self.estimate_pose(image, bbox)
            
            results.append({
                'bbox': bbox,
                'confidence': confidence,
                'keypoints': keypoints
            })
        
        return results
    
    def visualize_results(self, image_path, results):
        """Visualize the results"""
        image = cv2.imread(image_path)
        
        for result in results:
            bbox = result['bbox']
            keypoints = result['keypoints']
            
            # Draw bounding boxes
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw keypoints
            for kpt in keypoints:
                x, y, conf = kpt
                if conf > 0.5:
                    cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        return image

# Usage example
def main():
    """Main-function example"""
    
    # Initialize the model
    print("Initialize the TensorRT detection model...")
    detector = SimpleTensorRTDetection()
    
    # Process the image
    image_path = 'test_image.jpg'  # replace with your image path
    
    if osp.exists(image_path):
        print(f"Process the image: {image_path}")
        
        # Detection and pose estimation
        results = detector.process_image(image_path)
        
        if results:
            print(f"Detected  {len(results)}  people")
            
            # Visualize the results
            vis_image = detector.visualize_results(image_path, results)
            cv2.imwrite('result.jpg', vis_image)
            print("Results saved to result.jpg")
        else:
            print("No people were detected")
    else:
        print(f"Image file not found: {image_path}")
        print("Please prepare a test image")

if __name__ == "__main__":
    main()

# Performance test
def benchmark():
    """Performance test"""
    import time
    
    detector = SimpleTensorRTDetection()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warm-up
    for _ in range(5):
        _ = detector.detect_objects(test_image)
    
    # Performance test
    n_tests = 50
    start_time = time.time()
    
    for i in range(n_tests):
        detections = detector.detect_objects(test_image)
        if i % 10 == 0:
            print(f"Test progress: {i+1}/{n_tests}")
    
    total_time = time.time() - start_time
    avg_time = total_time / n_tests
    fps = 1.0 / avg_time
    
    print(f"\nPerformance test results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average inference time: {avg_time*1000:.1f}ms")
    print(f"Average FPS: {fps:.1f}")