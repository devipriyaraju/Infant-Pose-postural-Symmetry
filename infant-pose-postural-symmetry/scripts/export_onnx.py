#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from infant_pose_symmetry.deployment.export import export_onnx
from infant_pose_symmetry.deployment.mmpose_io import preprocess_batch
from infant_pose_symmetry.pose.mmpose_pipeline import POSE_MODELS


def parse_bbox(value):
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=float)
    return np.asarray(value, dtype=float)


def load_model(project: Path, model_name: str, fold: int, device: str):
    from mmengine.config import Config
    from mmpose.apis import init_model

    config = Config.fromfile(str(project / "configs" / f"{model_name}_fold{fold}.py"))
    candidates = sorted(Path(config.work_dir).glob("best_coco_AP*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found in {config.work_dir}")
    model = init_model(config, str(candidates[-1]), device=device).eval()
    model.test_cfg["flip_test"] = False
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--model", choices=list(POSE_MODELS), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-dir", default="deploy")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ONNX export is configured for the CUDA MMPose model used in this project")

    project = Path(args.project_dir).resolve()
    metadata = pd.read_csv(project / "data" / "processed" / "metadata.csv")
    metadata["bbox"] = metadata["bbox"].apply(parse_bbox)
    folds = np.load(project / "data" / "splits" / "folds.npy")
    fold_indices = np.where(folds == args.fold)[0]
    if len(fold_indices) == 0:
        raise RuntimeError(f"No samples found for fold {args.fold}")

    model = load_model(project, args.model, args.fold, "cuda")
    inputs, _ = preprocess_batch(model, metadata, fold_indices[:1])
    height, width = inputs.shape[2:]

    output = project / args.output_dir / f"{args.model}_fold{args.fold}.onnx"
    export_onnx(
        model,
        output,
        height=height,
        width=width,
        device="cuda",
        opset=args.opset,
    )
    print(output)
    print(f"size_mb={output.stat().st_size / 1e6:.1f}")


if __name__ == "__main__":
    main()
