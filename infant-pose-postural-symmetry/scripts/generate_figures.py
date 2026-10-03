#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    metrics = project / "results" / "metrics"
    output = project / "results" / "plots"
    output.mkdir(parents=True, exist_ok=True)

    per_joint = metrics / "per_joint_pck.csv"
    if per_joint.exists():
        frame = pd.read_csv(per_joint, index_col=0)
        ax = frame.plot.bar(figsize=(12, 4), width=0.8)
        ax.set_ylabel("PCK@0.05")
        ax.set_xlabel("Joint")
        ax.set_title("Per-joint pose accuracy")
        plt.tight_layout()
        plt.savefig(output / "per_joint_pck.png", dpi=150)
        plt.close()

    print(f"Figures written to {output}")


if __name__ == "__main__":
    main()
