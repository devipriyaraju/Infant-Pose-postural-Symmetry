#!/usr/bin/env python
import argparse
from pathlib import Path

from infant_pose_symmetry.data.syrip import (
    dataset_inventory,
    discover_keypoint_records,
    discover_optional_wacv_files,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    print(f"Dataset root: {root}")
    print(dataset_inventory(root).to_string(index=False))

    real, synthetic, _ = discover_keypoint_records(root)
    print(f"Real annotated images: {len(real)}")
    print(f"Synthetic annotated images: {len(synthetic)}")

    print("\nOptional WACV files:")
    for name, path in discover_optional_wacv_files(root).items():
        print(f"{name:12s} {path if path else 'not found'}")


if __name__ == "__main__":
    main()
