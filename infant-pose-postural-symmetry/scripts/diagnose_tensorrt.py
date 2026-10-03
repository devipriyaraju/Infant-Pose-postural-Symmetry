#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from infant_pose_symmetry.deployment.backends import TRTRunner
from infant_pose_symmetry.deployment.export import PoseNet
from infant_pose_symmetry.deployment.mmpose_io import preprocess_batch


def parse_bbox(value):
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=float)
    return np.asarray(value, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--model", default="RTMPose-S")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    import tensorrt as trt
    from mmengine.config import Config
    from mmpose.apis import init_model

    project = Path(args.project_dir).resolve()
    metadata = pd.read_csv(project / "data" / "processed" / "metadata.csv")
    metadata["bbox"] = metadata["bbox"].apply(parse_bbox)
    folds = np.load(project / "data" / "splits" / "folds.npy")
    index = np.where(folds == args.fold)[0][:args.samples]

    config = Config.fromfile(str(project / "configs" / f"{args.model}_fold{args.fold}.py"))
    checkpoint = sorted(Path(config.work_dir).glob("best_coco_AP*.pth"))[-1]
    model = init_model(config, str(checkpoint), device="cuda").eval()
    inputs, _ = preprocess_batch(model, metadata, index)
    height, width = inputs.shape[2:]
    onnx_path = project / "deploy" / f"{args.model}_fold{args.fold}.onnx"

    with torch.no_grad():
        reference = [tensor.float().cpu() for tensor in PoseNet(model).eval()(inputs)]

    for tag, kwargs in (
        ("trt_fp32", {"fp16": False, "fp32_name_filter": None}),
        ("trt_fp16", {"fp16": True, "fp32_name_filter": None}),
    ):
        engine = project / "deploy" / f"diag_{args.model}_{tag}.engine"
        if engine.exists():
            engine.unlink()
        outputs = TRTRunner(
            onnx_path,
            engine,
            height,
            width,
            **kwargs,
        )(inputs)
        for j, (ref, out) in enumerate(zip(reference, outputs)):
            finite = torch.isfinite(out)
            print(
                tag,
                f"out{j}",
                f"max_abs_diff={(ref - out).abs().max().item():.6g}",
                f"non_finite={(~finite).sum().item()}",
                f"argmax_agreement={(ref.argmax(-1) == out.argmax(-1)).float().mean().item():.3f}",
            )

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    parser.parse(onnx_path.read_bytes())
    names = [network.get_layer(i).name for i in range(network.num_layers)]
    print(f"TensorRT layers: {len(names)}")
    print("Last 15 layers:")
    for name in names[-15:]:
        print(name)


if __name__ == "__main__":
    main()
