from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_coco(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_files(root: Path):
    return [
        Path(directory) / name
        for directory, _, names in os.walk(root)
        for name in names
        if not name.startswith("._")
    ]


def dataset_inventory(root: Path) -> pd.DataFrame:
    rows = []
    counts = Counter()
    for path in list_files(root):
        relative = path.relative_to(root).parts
        folder = "/".join(relative[: min(4, len(relative) - 1)]) or "."
        counts[(folder, path.suffix.lower())] += 1
    for (folder, extension), count in counts.items():
        rows.append({"dir": folder, "ext": extension, "n": count})
    return pd.DataFrame(rows).sort_values(["dir", "ext"]).reset_index(drop=True)


def find_optional_file(files, pattern: str):
    regex = re.compile(pattern, re.I)
    matches = sorted(
        [path for path in files if regex.search(path.name)],
        key=lambda path: (len(path.parts), str(path)),
    )
    return matches[0] if matches else None


def discover_optional_wacv_files(root: Path):
    files = list_files(root)
    return {
        "survey": find_optional_file(files, r"^asymmetryanalysis.*\.xlsx$"),
        "name_map": find_optional_file(files, r"^img_name700_map\.npy$"),
        "fidip": find_optional_file(files, r"^keypoints_validate_infant_results.*\.json$"),
        "p3d": find_optional_file(files, r"^output_pose_3D\.npy$"),
        "p3d_names": find_optional_file(files, r"^output_imgnames\.npy$"),
        "c3d": find_optional_file(files, r"^correct_3D\.npy$"),
    }


def _is_synthetic(image_record):
    return bool(image_record.get("is_synthetic", False)) or Path(
        image_record["file_name"]
    ).name.lower().startswith("syn")


def discover_keypoint_records(root: Path):
    files = list_files(root)
    by_name = defaultdict(list)
    for path in files:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            by_name[path.name].append(path)

    def resolve_image(image_record, json_path):
        candidates = by_name.get(Path(image_record["file_name"]).name, [])
        if len(candidates) > 1:
            same_size = []
            for candidate in candidates:
                try:
                    if Image.open(candidate).size == (
                        image_record.get("width"),
                        image_record.get("height"),
                    ):
                        same_size.append(candidate)
                except OSError:
                    pass
            candidates = same_size or candidates
            parent_parts = json_path.parent.parts
            candidates = sorted(
                candidates,
                key=lambda candidate: -sum(
                    1 for a, b in zip(candidate.parts, parent_parts) if a == b
                ),
            )
        return candidates[0] if candidates else None

    real = {}
    synthetic = {}
    categories = None

    json_files = [
        path
        for path in files
        if path.suffix.lower() == ".json" and "result" not in path.name.lower()
    ]
    for json_path in sorted(json_files):
        try:
            graph = load_coco(json_path)
        except Exception:
            continue

        annotations = graph.get("annotations") if isinstance(graph, dict) else None
        if not annotations or "keypoints" not in annotations[0]:
            continue

        categories = categories or graph.get("categories")
        annotation_by_image = defaultdict(list)
        for annotation in annotations:
            annotation_by_image[annotation["image_id"]].append(annotation)

        for image_record in graph.get("images", []):
            image_annotations = annotation_by_image.get(image_record["id"], [])
            if len(image_annotations) != 1:
                continue
            if len(image_annotations[0].get("keypoints", [])) != 51:
                continue

            key = image_record.get("original_file_name") or Path(
                image_record["file_name"]
            ).name
            pool = synthetic if _is_synthetic(image_record) else real
            if key in pool:
                continue

            image_path = resolve_image(image_record, json_path)
            if image_path is None:
                continue

            pool[key] = {
                "im": image_record,
                "ann": image_annotations[0],
                "path": image_path,
                "json": json_path,
            }

    if categories is None or not real:
        raise RuntimeError(
            "No single infant COCO keypoint annotations with resolvable images were found"
        )
    return real, synthetic, categories


def natural_sort_key(text: str):
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", text)]


def keypoint_bbox(keypoints, width, height, pad=0.15):
    keypoints = np.asarray(keypoints, dtype=float)
    visible = keypoints[:, 2] > 0
    xs = keypoints[visible, 0]
    ys = keypoints[visible, 1]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_w = (x1 - x0) * pad + 5
    pad_h = (y1 - y0) * pad + 5
    x0 = max(0, x0 - pad_w)
    y0 = max(0, y0 - pad_h)
    x1 = min(width - 1, x1 + pad_w)
    y1 = min(height - 1, y1 + pad_h)
    return np.array([x0, y0, x1 - x0, y1 - y0], dtype=float)


def build_master_table(real_records, name_map=None):
    survey_order = False
    original_names = None

    if name_map is not None:
        mapping = np.load(name_map).astype(str)
        if set(mapping[:, 1]) <= set(real_records):
            original_names = list(mapping[:, 1])
            survey_order = True

    if original_names is None:
        original_names = sorted(real_records, key=natural_sort_key)

    records = [real_records[name] for name in original_names]
    keypoints = np.array(
        [record["ann"]["keypoints"] for record in records],
        dtype=float,
    ).reshape(len(records), 17, 3)

    table = pd.DataFrame(
        {
            "idx": np.arange(len(records)),
            "orig_name": original_names,
            "img_path": [str(record["path"]) for record in records],
            "width": [record["im"]["width"] for record in records],
            "height": [record["im"]["height"] for record in records],
            "posture": [
                record["im"].get("posture", "NA")
                for record in records
            ],
            "source": [
                record["im"].get("image_source", "") or ""
                for record in records
            ],
        }
    )

    table["bbox"] = [
        keypoint_bbox(keypoints[i], table.width[i], table.height[i])
        for i in range(len(table))
    ]
    table["test100"] = table.orig_name.isin(
        [f"test{i}.jpg" for i in range(100)]
    )
    table["syrip_test500"] = table.orig_name.str.startswith("test")
    table["group"] = [
        source if source else f"solo_{index}"
        for index, source in enumerate(table.source)
    ]

    return table, keypoints, survey_order
