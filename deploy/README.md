# Generated deployment artifacts

This directory is used for exported ONNX graphs and TensorRT engine files.

The generated binaries are intentionally not committed because TensorRT engines are hardware and software environment dependent, and the model exports are reproducible from the trained checkpoints.

Typical generated files:

```text
RTMPose-S_fold0.onnx
RTMPose-S_fold0_fp16_headfp32.engine
HRNet-W32_fold0.onnx
HRNet-W32_fold0_fp16_headfp32.engine
```

See `docs/deployment.md` for export and benchmark commands.
