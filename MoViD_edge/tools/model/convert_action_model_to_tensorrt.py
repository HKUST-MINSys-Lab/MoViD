#!/usr/bin/env python3
"""
Convert a pyskl STGCN model to TensorRT format for faster inference

Usage:
    python convert_action_model_to_tensorrt.py \
        --config models/action_recognition/stgcn_ntu60_xsub_3d_config.py \
        --checkpoint models/action_recognition/stgcn_ntu60_xsub_3d.pth \
        --output models/action_recognition/stgcn_ntu60_xsub_3d.engine \
        --fp16

Dependencies:
    pip install torch2trt  # or use tensorrt + onnx
    pip install onnx onnxruntime tensorrt
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add pyskl to path
for _pyskl_path in (
    os.environ.get('PYSKL_PATH', ''),
    str(REPO_ROOT / 'third-party' / 'pyskl'),
    str(REPO_ROOT / 'pyskl'),
    os.path.expanduser('~/pyskl'),
):
    if _pyskl_path and os.path.exists(_pyskl_path) and _pyskl_path not in sys.path:
        sys.path.insert(0, _pyskl_path)
        break

def parse_args():
    parser = argparse.ArgumentParser(description='Convert pyskl model to TensorRT')
    parser.add_argument('--config', type=str, 
                        default='models/action_recognition/stgcn_ntu60_xsub_3d_config.py',
                        help='Config file path')
    parser.add_argument('--checkpoint', type=str,
                        default='models/action_recognition/stgcn_ntu60_xsub_3d.pth',
                        help='Checkpoint file path')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Output TensorRT engine path (default: same as checkpoint with .engine extension)')
    parser.add_argument('--onnx_output', type=str,
                        default=None,
                        help='ONNX output path (intermediate file)')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision')
    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 precision (requires calibration data)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for TensorRT engine')
    parser.add_argument('--num_frames', type=int, default=100,
                        help='Number of frames (temporal dimension)')
    parser.add_argument('--num_keypoints', type=int, default=25,
                        help='Number of keypoints (NTU: 25, COCO: 17)')
    parser.add_argument('--num_persons', type=int, default=2,
                        help='Number of persons')
    parser.add_argument('--workspace_size', type=int, default=1,
                        help='TensorRT workspace size in GB')
    return parser.parse_args()


class STGCNWrapper(nn.Module):
    """
    Wrapper for STGCN model to make it compatible with ONNX/TensorRT export
    Handles the full forward pass including backbone + head
    """
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.cls_head = model.cls_head
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, M, T, V, C)
               N = batch size
               M = number of persons
               T = number of frames
               V = number of keypoints
               C = coordinates (3 for 3D)
        Returns:
            cls_score: Classification scores of shape (N, num_classes)
        """
        # Backbone forward
        feat = self.backbone(x)
        # Head forward (global pooling + classification)
        cls_score = self.cls_head(feat)
        return cls_score


def load_pyskl_model(config_path, checkpoint_path, device='cuda'):
    """Load pyskl model from config and checkpoint"""
    import mmcv
    from pyskl.apis import init_recognizer
    
    print(f"Loading model from {config_path} and {checkpoint_path}")
    config = mmcv.Config.fromfile(config_path)
    model = init_recognizer(config, checkpoint_path, device)
    model.eval()
    return model


def export_to_onnx(model, onnx_path, input_shape, device='cuda'):
    """Export PyTorch model to ONNX format"""
    print(f"\n=== Exporting to ONNX ===")
    print(f"Input shape: {input_shape}")
    print(f"Output path: {onnx_path}")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Wrap model for export
    wrapped_model = STGCNWrapper(model)
    wrapped_model.eval()
    
    # Export to ONNX (opset 14 to support einsum and other ops)
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,  # Use opset 14 for einsum support
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX model saved to {onnx_path}")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")
    
    return onnx_path


def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16=False, int8=False, 
                              workspace_size_gb=1, input_shape=None):
    """Convert ONNX model to TensorRT engine"""
    print(f"\n=== Converting to TensorRT ===")
    print(f"Input ONNX: {onnx_path}")
    print(f"Output engine: {engine_path}")
    print(f"FP16: {fp16}, INT8: {int8}")
    
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: tensorrt not installed. Please install TensorRT first.")
        print("On Jetson: sudo apt-get install tensorrt")
        print("On x86: pip install tensorrt")
        return None
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parsing error: {parser.get_error(error)}")
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1 << 30))
    
    # Add optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    if input_shape:
        # Set min, optimal, and max shapes
        min_shape = (1,) + input_shape[1:]  # batch_size = 1
        opt_shape = input_shape  # optimal = specified
        max_shape = (4,) + input_shape[1:]  # max batch = 4
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        print(f"Optimization profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
    
    config.add_optimization_profile(profile)
    
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        else:
            print("WARNING: FP16 not supported on this platform, using FP32")
    
    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("INT8 mode enabled (requires calibration data)")
        else:
            print("WARNING: INT8 not supported on this platform")
    
    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return None
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved to {engine_path}")
    return engine_path


def test_tensorrt_engine(engine_path, input_shape):
    """Test TensorRT engine inference"""
    print(f"\n=== Testing TensorRT Engine ===")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError as e:
        print(f"WARNING: Cannot test engine - missing dependency: {e}")
        return
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Load engine
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    
    # Set binding shapes for dynamic batch
    context.set_binding_shape(0, input_shape)
    
    # Allocate buffers - cast to int to avoid numpy.int64 issue
    input_size = int(np.prod(input_shape) * np.float32().nbytes)
    output_shape = (input_shape[0], 60)  # NTU60 has 60 classes
    output_size = int(np.prod(output_shape) * np.float32().nbytes)
    
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    
    # Create test input
    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    
    # Copy input to device
    cuda.memcpy_htod(d_input, h_input)
    
    # Run inference
    import time
    stream = cuda.Stream()
    
    # Warmup
    for _ in range(10):
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
    stream.synchronize()
    
    # Benchmark
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
    stream.synchronize()
    elapsed_time = time.time() - start_time
    
    # Copy output to host
    cuda.memcpy_dtoh(h_output, d_output)
    
    avg_time = elapsed_time / num_iterations * 1000  # ms
    fps = num_iterations / elapsed_time
    
    print(f"Test passed!")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {h_output.shape}")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {fps:.1f} FPS")
    print(f"Top-5 predictions: {np.argsort(h_output[0])[-5:][::-1]}")


def main():
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available. TensorRT requires CUDA.")
        return
    
    # Set output paths
    if args.output is None:
        args.output = args.checkpoint.replace('.pth', '.engine')
    
    if args.onnx_output is None:
        args.onnx_output = args.checkpoint.replace('.pth', '.onnx')
    
    # Define input shape: (batch_size, num_persons, num_frames, num_keypoints, 3)
    input_shape = (args.batch_size, args.num_persons, args.num_frames, 
                   args.num_keypoints, 3)
    
    print("=" * 60)
    print("STGCN to TensorRT Conversion")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ONNX output: {args.onnx_output}")
    print(f"TensorRT output: {args.output}")
    print(f"Input shape: {input_shape}")
    print(f"Precision: {'FP16' if args.fp16 else 'FP32'}{' + INT8' if args.int8 else ''}")
    print("=" * 60)
    
    # Load model
    model = load_pyskl_model(args.config, args.checkpoint, device)
    
    # Export to ONNX
    export_to_onnx(model, args.onnx_output, input_shape, device)
    
    # Convert to TensorRT
    engine_path = convert_onnx_to_tensorrt(
        args.onnx_output, 
        args.output,
        fp16=args.fp16,
        int8=args.int8,
        workspace_size_gb=args.workspace_size,
        input_shape=input_shape
    )
    
    if engine_path:
        # Test the engine
        test_tensorrt_engine(engine_path, input_shape)
        
        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print(f"TensorRT engine: {args.output}")
        print("=" * 60)


if __name__ == '__main__':
    main()
