#!/usr/bin/env python
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from infant_pose_symmetry.symmetry.geometry import angle_class, limb_angles
from infant_pose_symmetry.symmetry.training import evaluate_pose_source


def load_prediction(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--gnn-epochs", type=int, default=150)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    processed = project / "data" / "processed"
    predictions_dir = project / "results" / "predictions"

    metadata = pd.read_csv(processed / "metadata.csv")
    ground_truth = np.load(processed / "gt_keypoints.npy")
    folds = np.load(project / "data" / "splits" / "folds.npy")

    gt_angles = limb_angles(ground_truth[:, :, :2]).reshape(-1)
    y_sym = (gt_angles < 39.1).astype(int)
    y_angle = angle_class(gt_angles)

    rows = []
    cache = {}

    for prediction_file in sorted(predictions_dir.glob("*.pkl")):
        prediction = load_prediction(prediction_file)
        source = prediction_file.stem.replace("_finetuned", " FT")
        source = source.replace("_zeroshot", " zero-shot")

        results = evaluate_pose_source(
            prediction["kp"],
            prediction["sc"],
            y_sym,
            y_angle,
            folds,
            gnn_epochs=args.gnn_epochs,
            device=args.device,
        )
        cache[source] = results

        for assessor, result in results.items():
            rows.append(
                {
                    "pose": source,
                    "assessor": assessor,
                    "AUROC sym": roc_auc_score(
                        y_sym,
                        result["score_sym"],
                    ),
                    "kappa sym": cohen_kappa_score(
                        y_sym,
                        result["pred_sym"],
                    ),
                    "weighted kappa angle": cohen_kappa_score(
                        y_angle,
                        result["pred_ang"],
                        weights="quadratic",
                    ),
                    "macro-F1 angle": f1_score(
                        y_angle,
                        result["pred_ang"],
                        average="macro",
                    ),
                }
            )

    output_dir = project / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(
        "AUROC sym",
        ascending=False,
    )
    table.to_csv(
        output_dir / "symmetry_results_recomputed.csv",
        index=False,
    )

    with open(
        project / "results" / "metrics" / "symmetry_cache.pkl",
        "wb",
    ) as handle:
        pickle.dump(cache, handle)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
