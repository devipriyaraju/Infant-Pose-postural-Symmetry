#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from infant_pose_symmetry.deployment.backends import ORTRunner, TRTRunner
from infant_pose_symmetry.deployment.benchmark import benchmark_callable
from infant_pose_symmetry.deployment.export import PoseNet, export_onnx
from infant_pose_symmetry.deployment.mmpose_io import decode_outputs, preprocess_batch
from infant_pose_symmetry.evaluation.metrics import pck
from infant_pose_symmetry.pose.mmpose_pipeline import POSE_MODELS


def parse_bbox(value):
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=float)
    return np.asarray(value, dtype=float)


def load_model(project: Path, model_name: str, fold: int):
    from mmengine.config import Config
    from mmpose.apis import init_model

    config = Config.fromfile(str(project / "configs" / f"{model_name}_fold{fold}.py"))
    candidates = sorted(Path(config.work_dir).glob("best_coco_AP*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found in {config.work_dir}")
    model = init_model(config, str(candidates[-1]), device="cuda").eval()
    model.test_cfg["flip_test"] = False
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--skip-tensorrt", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(POSE_MODELS), default=list(POSE_MODELS))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Deployment benchmarking requires a CUDA GPU")

    project = Path(args.project_dir).resolve()
    output_dir = project / "deploy"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = project / "results" / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(project / "data" / "processed" / "metadata.csv")
    metadata["bbox"] = metadata["bbox"].apply(parse_bbox)
    ground_truth = np.load(project / "data" / "processed" / "gt_keypoints.npy")
    folds = np.load(project / "data" / "splits" / "folds.npy")
    test_index = np.where(folds == args.fold)[0]

    rows = []
    for model_name in args.models:
        model = load_model(project, model_name, args.fold)
        inputs, samples = preprocess_batch(model, metadata, test_index)
        height, width = inputs.shape[2:]
        onnx_path = export_onnx(
            model,
            output_dir / f"{model_name}_fold{args.fold}.onnx",
            height,
            width,
            device="cuda",
            opset=17,
        )
        network = PoseNet(model).eval()

        def torch_fp32(x):
            with torch.no_grad():
                return tuple(output.float().cpu() for output in network(x))

        def torch_fp16(x):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return tuple(output.float().cpu() for output in network(x))

        runners = {
            "PyTorch FP32": torch_fp32,
            "PyTorch FP16 (autocast)": torch_fp16,
        }

        ort_runner = ORTRunner(onnx_path)
        runners[f"ONNX Runtime ({ort_runner.provider.replace('ExecutionProvider', '')})"] = ort_runner

        if not args.skip_tensorrt:
            try:
                runners["TensorRT FP16"] = TRTRunner(
                    onnx_path,
                    output_dir / f"{model_name}_fold{args.fold}_fp16_headfp32.engine",
                    height,
                    width,
                    fp16=True,
                    max_batch=args.max_batch,
                )
            except ImportError:
                print("TensorRT is not installed. TensorRT rows will be skipped.")
            except Exception as exc:
                print(f"TensorRT build failed for {model_name}: {exc}")

        reference_keypoints = None
        for backend_name, runner in runners.items():
            output_chunks = [
                runner(inputs[start:start + args.max_batch])
                for start in range(0, len(inputs), args.max_batch)
            ]
            outputs = tuple(
                torch.cat([chunk[j] for chunk in output_chunks])
                for j in range(len(output_chunks[0]))
            )
            keypoints, _ = decode_outputs(model, outputs, samples)
            if reference_keypoints is None:
                reference_keypoints = keypoints
            deviation = np.linalg.norm(keypoints - reference_keypoints, axis=2)

            bs1 = inputs[:1].clone()
            if len(inputs) >= args.batch_size:
                batch = inputs[:args.batch_size].clone()
            else:
                batch = inputs.repeat(
                    math.ceil(args.batch_size / len(inputs)),
                    1,
                    1,
                    1,
                )[:args.batch_size]

            pose_pck = pck(
                keypoints,
                ground_truth[test_index],
                np.stack(metadata.bbox.iloc[test_index].to_numpy()),
                alpha=0.05,
            )
            rows.append(
                {
                    "model": model_name,
                    "backend": backend_name,
                    "latency bs1 (ms)": 1000 * benchmark_callable(runner, bs1),
                    f"throughput bs{args.batch_size} (img/s)": args.batch_size
                    / benchmark_callable(runner, batch, warmup=5, iterations=30),
                    "PCK@0.05": float(np.nanmean(pose_pck)),
                    "mean |Δkp| vs FP32 (px)": float(deviation.mean()),
                    "max |Δkp| (px)": float(deviation.max()),
                }
            )

        del model, network, runners
        gc.collect()
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    baseline = (
        frame[frame.backend == "PyTorch FP32"]
        .set_index("model")["latency bs1 (ms)"]
    )
    frame["speed-up vs FP32"] = frame.apply(
        lambda row: baseline[row["model"]] / row["latency bs1 (ms)"],
        axis=1,
    )
    output_csv = results_dir / "deployment_benchmark.csv"
    frame.to_csv(output_csv, index=False)
    print(frame.round(3).to_string(index=False))
    print(output_csv)


if __name__ == "__main__":
    main()
