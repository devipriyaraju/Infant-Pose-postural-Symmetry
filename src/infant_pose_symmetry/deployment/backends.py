from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


class ORTRunner:
    """ONNX Runtime runner with CUDA I/O binding when available."""

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.ort = ort
        options = ort.SessionOptions()
        try:
            options.intra_op_num_threads = len(os.sched_getaffinity(0))
        except AttributeError:
            options.intra_op_num_threads = max(1, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        options.log_severity_level = 3

        available = ort.get_available_providers()
        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ]
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=providers,
        )
        self.provider = self.session.get_providers()[0]

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.provider == "CUDAExecutionProvider":
            binding = self.session.io_binding()
            binding.bind_input(
                "input",
                "cuda",
                0,
                np.float32,
                list(x.shape),
                x.data_ptr(),
            )
            for output in self.session.get_outputs():
                binding.bind_output(output.name, "cuda")
            self.session.run_with_iobinding(binding)
            return tuple(
                torch.from_numpy(array)
                for array in binding.copy_outputs_to_cpu()
            )

        arrays = self.session.run(None, {"input": x.cpu().numpy()})
        return tuple(torch.from_numpy(array) for array in arrays)


class TRTRunner:
    """TensorRT runner for dynamic batch ONNX pose networks.

    The notebook found that pure FP16 RTMPose can overflow in normalization
    and prediction head layers. The default filters keep sensitive head layers
    in FP32 while allowing the remaining network to use FP16.
    """

    DEFAULT_FP32_FILTERS = (
        "/final_layer",
        "/mlp",
        "/gau",
        "/cls_x",
        "/cls_y",
    )

    def __init__(
        self,
        onnx_path: str | Path,
        engine_path: str | Path,
        height: int,
        width: int,
        fp16: bool = True,
        max_batch: int = 32,
        fp32_name_filter=DEFAULT_FP32_FILTERS,
        workspace_bytes: int = 2 << 30,
    ):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        onnx_path = Path(onnx_path)
        engine_path = Path(engine_path)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        if engine_path.exists():
            serialized = engine_path.read_bytes()
        else:
            builder = trt.Builder(self.logger)
            flags = 0
            if int(trt.__version__.split(".")[0]) < 10:
                flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flags)
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse(onnx_path.read_bytes()):
                errors = "\n".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)
                )
                raise RuntimeError(errors)

            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                workspace_bytes,
            )

            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                if fp32_name_filter:
                    filters = (
                        (fp32_name_filter,)
                        if isinstance(fp32_name_filter, str)
                        else tuple(fp32_name_filter)
                    )
                    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                    for i in range(network.num_layers):
                        layer = network.get_layer(i)
                        if not any(f in layer.name for f in filters):
                            continue
                        if not layer.num_outputs:
                            continue
                        if layer.get_output(0).dtype != trt.float32:
                            continue
                        layer.precision = trt.float32
                        for j in range(layer.num_outputs):
                            layer.set_output_type(j, trt.float32)

            profile = builder.create_optimization_profile()
            profile.set_shape(
                "input",
                (1, 3, height, width),
                (max_batch, 3, height, width),
                (max_batch, 3, height, width),
            )
            config.add_optimization_profile(profile)

            built = builder.build_serialized_network(network, config)
            if built is None:
                raise RuntimeError("TensorRT engine build failed")
            serialized = bytes(built)
            engine_path.write_bytes(serialized)

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]
        self.outputs = sorted(
            (
                name
                for name in names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            ),
            key=lambda value: int(value[3:]) if value.startswith("out") and value[3:].isdigit() else value,
        )
        self.stream = torch.cuda.Stream()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        trt = self.trt
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.context.set_input_shape("input", tuple(x.shape))
            self.context.set_tensor_address("input", x.data_ptr())
            outputs = []
            for name in self.outputs:
                dtype = torch.from_numpy(
                    np.empty(0, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
                ).dtype
                tensor = torch.empty(
                    tuple(self.context.get_tensor_shape(name)),
                    dtype=dtype,
                    device="cuda",
                )
                self.context.set_tensor_address(name, tensor.data_ptr())
                outputs.append(tensor)
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return tuple(output.float().cpu() for output in outputs)
