from __future__ import annotations

import copy

import numpy as np
import torch


@torch.no_grad()
def preprocess_batch(model, metadata, indices):
    """Apply the model's MMPose test pipeline and data preprocessor."""
    from mmengine.dataset import Compose, pseudo_collate

    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    items = []
    for index in indices:
        x0, y0, width, height = metadata.bbox.iloc[index]
        item = {
            "img_path": metadata.img_path.iloc[index],
            "bbox": np.array(
                [[x0, y0, x0 + width, y0 + height]],
                dtype=np.float32,
            ),
            "bbox_score": np.ones(1, dtype=np.float32),
        }
        item.update(model.dataset_meta)
        items.append(pipeline(item))

    batch = model.data_preprocessor(pseudo_collate(items), False)
    return batch["inputs"].float().contiguous(), batch["data_samples"]


@torch.no_grad()
def decode_outputs(model, outputs, data_samples):
    """Decode network tensors to image-coordinate keypoints with MMPose."""
    head_outputs = outputs[0] if len(outputs) == 1 else tuple(outputs)
    predictions = model.head.decode(head_outputs)
    results = model.add_pred_to_datasample(
        predictions,
        None,
        copy.deepcopy(data_samples),
    )
    keypoints = np.stack(
        [result.pred_instances.keypoints[0] for result in results]
    )
    scores = np.stack(
        [result.pred_instances.keypoint_scores[0] for result in results]
    )
    return keypoints, scores
