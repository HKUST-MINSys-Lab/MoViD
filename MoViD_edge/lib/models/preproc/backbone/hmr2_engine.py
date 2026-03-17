import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch

class HMR2Engine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 分配内存
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for binding in self.engine:
            binding_shape = self.engine.get_binding_shape(binding)
            size = trt.volume(binding_shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # 分配 host 和 device 内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def infer(self, input_tensor: torch.Tensor):
        # 转 numpy (NCHW float32)
        np_input = input_tensor.detach().cpu().numpy().astype(np.float32).ravel()

        # 拷贝输入
        np.copyto(self.inputs[0]["host"], np_input)

        # H2D
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # 执行推理
        self.context.execute_async_v2(self.bindings, self.stream.handle)

        # D2H
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        # 整理输出
        results = [out["host"] for out in self.outputs]
        torch_outputs = [torch.tensor(r).reshape(-1) for r in results]
        return torch_outputs
