#!/usr/bin/env python
import argparse
from pathlib import Path

from infant_pose_symmetry.pose.mmpose_pipeline import (
    POSE_MODELS,
    build_finetune_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    annotation_dir = project / "data" / "annotations"
    image_dir = project / "data" / "processed" / "images"

    for model_name in POSE_MODELS:
        for fold in range(5):
            output = build_finetune_config(
                model_name=model_name,
                fold=fold,
                project_dir=project,
                checkpoint_dir=checkpoint_dir,
                train_json=annotation_dir / f"fold{fold}_train.json",
                val_json=annotation_dir / f"fold{fold}_val.json",
                image_dir=image_dir,
            )
            print(output)


if __name__ == "__main__":
    main()
