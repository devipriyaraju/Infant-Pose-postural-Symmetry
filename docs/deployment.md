# Deployment

The deployment stage in the updated experiment notebook converts the fine tuned pose networks to ONNX and benchmarks three execution paths.

1. PyTorch CUDA
2. ONNX Runtime CUDA
3. TensorRT

The preprocessing and keypoint decoding remain in MMPose. Only the neural network backend changes during the benchmark.

## ONNX export

The export wrapper contains the MMPose backbone, neck and pose head. The exported graph uses opset 17 and a dynamic batch dimension.

```bash
python scripts/export_onnx.py \
  --project-dir . \
  --model RTMPose-S \
  --fold 0
```

Generated ONNX files are written to `deploy/`.

## ONNX Runtime

`ORTRunner` uses CUDA I/O binding when the CUDA execution provider is available. The input tensor stays on the GPU and ONNX Runtime writes outputs on the GPU before they are copied back for the common MMPose decode step.

## TensorRT

`TRTRunner` parses the ONNX graph and creates a dynamic batch optimization profile from batch 1 to batch 32 by default.

The updated notebook found a precision issue with pure FP16 RTMPose-S. The two SimCC output tensors became non finite when the entire network was forced to FP16. An FP32 TensorRT engine was stable.

The final engine therefore runs most layers in FP16 and keeps sensitive prediction head layers in FP32. The default layer name filters are:

```text
/final_layer
/mlp
/gau
/cls_x
/cls_y
```

This behavior is implemented in `src/infant_pose_symmetry/deployment/backends.py`.

## Benchmark

```bash
python scripts/benchmark_deployment.py \
  --project-dir . \
  --fold 0
```

The benchmark reports:

- batch 1 latency
- batch 32 throughput
- PCK@0.05
- mean keypoint deviation from PyTorch FP32
- maximum keypoint deviation from PyTorch FP32
- speedup relative to PyTorch FP32

The measured values from the uploaded notebook are stored in:

```text
results/metrics/deployment_benchmark.csv
```

## TensorRT diagnostic

Use this script if a TensorRT engine produces unstable outputs:

```bash
python scripts/diagnose_tensorrt.py \
  --project-dir . \
  --model RTMPose-S \
  --fold 0
```

It compares PyTorch outputs with TensorRT FP32 and pure TensorRT FP16 outputs and reports maximum absolute difference, non finite values and argmax agreement.

## Generated model artifacts

ONNX and TensorRT engine files are not committed to Git. They depend on the trained checkpoints, CUDA environment and TensorRT version.

```text
deploy/
├── RTMPose-S_fold0.onnx
├── RTMPose-S_fold0_fp16_headfp32.engine
├── HRNet-W32_fold0.onnx
└── HRNet-W32_fold0_fp16_headfp32.engine
```

Recreate them from the trained checkpoints using the scripts above.
