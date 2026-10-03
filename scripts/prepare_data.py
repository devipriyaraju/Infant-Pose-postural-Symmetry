#!/usr/bin/env python
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from infant_pose_symmetry.data.splits import grouped_pose_folds
from infant_pose_symmetry.data.syrip import (
    build_master_table,
    discover_keypoint_records,
    discover_optional_wacv_files,
    keypoint_bbox,
)


def link_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def coco_record(image_id, file_name, width, height, keypoints, bbox):
    keypoints = np.asarray(keypoints, dtype=float).copy()
    keypoints[keypoints[:, 2] == 0, :2] = 0
    image = {
        "id": int(image_id),
        "file_name": file_name,
        "width": int(width),
        "height": int(height),
    }
    annotation = {
        "id": int(image_id),
        "image_id": int(image_id),
        "category_id": 1,
        "iscrowd": 0,
        "keypoints": keypoints.reshape(-1).tolist(),
        "num_keypoints": int((keypoints[:, 2] > 0).sum()),
        "bbox": [float(value) for value in bbox],
        "area": float(bbox[2] * bbox[3]),
    }
    return image, annotation


def write_coco(records, categories, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [record[0] for record in records],
        "annotations": [record[1] for record in records],
        "categories": categories,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--no-extra-real", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_dir).expanduser().resolve()
    project = Path(args.project_dir).expanduser().resolve()

    real, synthetic, categories = discover_keypoint_records(data_root)
    optional = discover_optional_wacv_files(data_root)

    meta, gt_keypoints, survey_order = build_master_table(
        real,
        optional["name_map"],
    )
    folds = grouped_pose_folds(
        meta,
        n_splits=args.folds,
        seed=args.seed,
    )
    meta["fold"] = folds

    prepared = project / "data" / "processed"
    annotations = project / "data" / "annotations"
    splits = project / "data" / "splits"
    image_links = prepared / "images"
    for folder in [prepared, annotations, splits, image_links]:
        folder.mkdir(parents=True, exist_ok=True)

    meta_for_csv = meta.copy()
    meta_for_csv["bbox"] = meta_for_csv["bbox"].apply(
        lambda value: json.dumps([float(item) for item in value])
    )
    meta_for_csv.to_csv(prepared / "metadata.csv", index=False)
    np.save(prepared / "gt_keypoints.npy", gt_keypoints)
    np.save(splits / "folds.npy", folds)

    real_records = []
    original_names = list(meta.orig_name)
    for index, row in meta.iterrows():
        source = Path(row.img_path)
        target_name = f"real_{index:04d}{source.suffix}"
        link_or_copy(source, image_links / target_name)
        real_records.append(
            coco_record(
                index,
                target_name,
                row.width,
                row.height,
                gt_keypoints[index],
                row.bbox,
            )
        )

    extra_records = []
    extra = []
    if not args.no_synthetic:
        extra.extend(
            (record, "synthetic")
            for record in synthetic.values()
        )
    if not args.no_extra_real:
        original_set = set(original_names)
        extra.extend(
            (record, "real")
            for name, record in real.items()
            if name not in original_set
        )

    for extra_index, (record, kind) in enumerate(extra):
        annotation = record["ann"]
        image_record = record["im"]
        keypoints = np.asarray(
            annotation["keypoints"],
            dtype=float,
        ).reshape(17, 3)

        if (keypoints[:, 2] > 0).sum() < 4:
            continue

        source = Path(record["path"])
        target_name = f"{kind}_{extra_index:05d}{source.suffix}"
        link_or_copy(source, image_links / target_name)
        bbox = keypoint_bbox(
            keypoints,
            image_record["width"],
            image_record["height"],
        )
        extra_records.append(
            coco_record(
                10000 + extra_index,
                target_name,
                image_record["width"],
                image_record["height"],
                keypoints,
                bbox,
            )
        )

    write_coco(
        real_records,
        categories,
        annotations / "real700.json",
    )

    for fold in range(args.folds):
        train_records = [
            real_records[index]
            for index in range(len(real_records))
            if folds[index] != fold
        ] + extra_records
        validation_records = [
            real_records[index]
            for index in range(len(real_records))
            if folds[index] == fold
        ]
        write_coco(
            train_records,
            categories,
            annotations / f"fold{fold}_train.json",
        )
        write_coco(
            validation_records,
            categories,
            annotations / f"fold{fold}_val.json",
        )

    manifest = {
        "data_root": str(data_root),
        "n_images": int(len(meta)),
        "n_synthetic": int(len(synthetic)),
        "n_extra_training": int(len(extra_records)),
        "survey_order": bool(survey_order),
        "optional_files": {
            key: str(value) if value else None
            for key, value in optional.items()
        },
    }
    with open(prepared / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
