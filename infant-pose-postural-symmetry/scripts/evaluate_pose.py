#!/usr/bin/env python
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from infant_pose_symmetry.constants import KP_NAMES
from infant_pose_symmetry.evaluation.metrics import pck


def parse_bbox(value):
    return np.asarray(json.loads(value), dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    metadata = pd.read_csv(project / "data" / "processed" / "metadata.csv")
    boxes = np.stack(metadata.bbox.apply(parse_bbox))
    ground_truth = np.load(
        project / "data" / "processed" / "gt_keypoints.npy"
    )

    subsets = {
        "all700": np.arange(len(metadata)),
        "test500": np.where(metadata.syrip_test500.values)[0],
        "test100": np.where(metadata.test100.values)[0],
    }

    rows = []
    per_joint = {}
    for prediction_file in sorted(
        (project / "results" / "predictions").glob("*.pkl")
    ):
        with open(prediction_file, "rb") as handle:
            prediction = pickle.load(handle)

        source = prediction_file.stem.replace("_finetuned", " FT")
        source = source.replace("_zeroshot", " zero-shot")

        for subset_name, index in subsets.items():
            hits05 = pck(
                prediction["kp"][index],
                ground_truth[index],
                boxes[index],
                alpha=0.05,
            )
            hits10 = pck(
                prediction["kp"][index],
                ground_truth[index],
                boxes[index],
                alpha=0.10,
            )
            rows.append(
                {
                    "source": source,
                    "subset": subset_name,
                    "PCK@0.05": float(np.nanmean(hits05)),
                    "PCK@0.1": float(np.nanmean(hits10)),
                }
            )

        per_joint[source] = np.nanmean(
            pck(
                prediction["kp"],
                ground_truth,
                boxes,
                alpha=0.05,
            ),
            axis=0,
        )

    output_dir = project / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "pose_pck.csv", index=False)
    pd.DataFrame(per_joint, index=KP_NAMES).to_csv(
        output_dir / "per_joint_pck.csv"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
