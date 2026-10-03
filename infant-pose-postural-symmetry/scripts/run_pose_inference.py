#!/usr/bin/env python
import argparse
import json
import os
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from infant_pose_symmetry.constants import FLIP_PERM
from infant_pose_symmetry.pose.mmpose_pipeline import (
    POSE_MODELS,
    find_checkpoint,
    predict_one,
)
from infant_pose_symmetry.pose.robustness import CORRUPTIONS, corrupt


def parse_bbox(value):
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=float)
    return np.asarray(value, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--checkpoint-dir", default="checkpoints/mmpose")
    parser.add_argument(
        "--model",
        choices=list(POSE_MODELS),
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=["zeroshot", "finetuned"],
        required=True,
    )
    parser.add_argument(
        "--robustness-subset",
        choices=["test100", "all", "none"],
        default="test100",
    )
    args = parser.parse_args()

    from mmengine.config import Config
    from mmpose.apis import init_model

    project = Path(args.project_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    metadata = pd.read_csv(project / "data" / "processed" / "metadata.csv")
    metadata["bbox"] = metadata["bbox"].apply(parse_bbox)
    gt_keypoints = np.load(project / "data" / "processed" / "gt_keypoints.npy")
    folds = np.load(project / "data" / "splits" / "folds.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_images = len(metadata)
    result = {
        "kp": np.zeros((n_images, 17, 2)),
        "sc": np.zeros((n_images, 17)),
        "kp_raw": np.zeros((n_images, 17, 2)),
        "kp_flip": np.zeros((n_images, 17, 2)),
        "sc_flip": np.zeros((n_images, 17)),
        "rob": {name: {} for name in CORRUPTIONS},
    }

    if args.robustness_subset == "test100":
        robustness_index = set(np.where(metadata.test100.values)[0])
    elif args.robustness_subset == "all":
        robustness_index = set(range(n_images))
    else:
        robustness_index = set()

    rng = np.random.default_rng(0)

    passes = range(5) if args.mode == "finetuned" else range(1)
    for fold in passes:
        if args.mode == "finetuned":
            config = Config.fromfile(
                str(project / "configs" / f"{args.model}_fold{fold}.py")
            )
            work_dir = Path(config.work_dir)
            candidates = sorted(work_dir.glob("best_coco_AP*.pth"))
            if not candidates:
                raise FileNotFoundError(
                    f"No best checkpoint in {work_dir}"
                )
            model = init_model(
                config,
                str(candidates[-1]),
                device=device,
            )
            indices = np.where(folds == fold)[0]
        else:
            config_name = POSE_MODELS[args.model]["config"]
            config_path, checkpoint = find_checkpoint(
                checkpoint_dir,
                config_name,
            )
            config = Config.fromfile(str(config_path))
            if config.model.backbone.get("init_cfg"):
                config.model.backbone.init_cfg = None
            model = init_model(
                config,
                str(checkpoint),
                device=device,
            )
            indices = np.arange(n_images)

        for index in indices:
            image = cv2.imread(metadata.img_path.iloc[index])
            if image is None:
                raise FileNotFoundError(metadata.img_path.iloc[index])

            height, width = image.shape[:2]
            x0, y0, box_width, box_height = metadata.bbox.iloc[index]
            box = [
                x0,
                y0,
                x0 + box_width,
                y0 + box_height,
            ]

            result["kp"][index], result["sc"][index] = predict_one(
                model,
                image,
                box,
                flip_test=True,
            )
            result["kp_raw"][index], _ = predict_one(
                model,
                image,
                box,
                flip_test=False,
            )

            flipped_box = [
                width - 1 - box[2],
                box[1],
                width - 1 - box[0],
                box[3],
            ]
            keypoints_flip, scores_flip = predict_one(
                model,
                image[:, ::-1].copy(),
                flipped_box,
                flip_test=False,
            )
            keypoints_flip[:, 0] = width - 1 - keypoints_flip[:, 0]
            result["kp_flip"][index] = keypoints_flip[FLIP_PERM]
            result["sc_flip"][index] = scores_flip[FLIP_PERM]

            if index in robustness_index:
                for corruption in CORRUPTIONS:
                    transformed, gt, transformed_box, target = corrupt(
                        image,
                        corruption,
                        gt_keypoints[index],
                        metadata.bbox.iloc[index],
                        rng,
                    )
                    predicted, predicted_score = predict_one(
                        model,
                        transformed,
                        transformed_box,
                        flip_test=True,
                    )
                    result["rob"][corruption][index] = {
                        "kp": predicted,
                        "sc": predicted_score,
                        "gt": gt,
                        "box": transformed_box,
                        "target": target,
                    }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir = project / "results" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.model}_{args.mode}.pkl"
    with open(output, "wb") as handle:
        pickle.dump(result, handle)
    print(output)


if __name__ == "__main__":
    main()
