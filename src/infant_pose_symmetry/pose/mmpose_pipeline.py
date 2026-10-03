from __future__ import annotations

import copy
import os
import re
from pathlib import Path

from infant_pose_symmetry.constants import FLIP_PERM


POSE_MODELS = {
    "RTMPose-S": {
        "config": "rtmpose-s_8xb256-420e_coco-256x192",
        "epochs": 120,
        "lr": 5e-4,
        "batch": 64,
    },
    "HRNet-W32": {
        "config": "td-hm_hrnet-w32_8xb64-210e_coco-256x192",
        "epochs": 80,
        "lr": 2e-4,
        "batch": 32,
    },
}


def find_checkpoint(checkpoint_dir: Path, config_name: str):
    config_files = sorted(
        checkpoint_dir.glob(f"{config_name}*.py")
    )
    size = re.search(r"\d+x\d+", config_name).group(0)
    stem = config_name.split("_")[0]
    checkpoint_files = [
        path
        for path in checkpoint_dir.glob("*.pth")
        if path.name.startswith(stem) and size in path.name
    ]

    if not config_files or not checkpoint_files:
        raise FileNotFoundError(
            "Missing MMPose config or checkpoint for "
            f"{config_name}"
        )
    return config_files[0], checkpoint_files[0]


def build_finetune_config(
    model_name,
    fold,
    project_dir,
    checkpoint_dir,
    train_json,
    val_json,
    image_dir,
    seed=0,
):
    from mmengine.config import Config

    spec = POSE_MODELS[model_name]
    config_path, checkpoint = find_checkpoint(
        Path(checkpoint_dir),
        spec["config"],
    )
    config = Config.fromfile(str(config_path))

    epochs = spec["epochs"]
    config.load_from = str(checkpoint)
    config.work_dir = str(
        Path(project_dir) / "work_dirs" / f"{model_name}_fold{fold}"
    )

    if config.model.backbone.get("init_cfg"):
        config.model.backbone.init_cfg = None

    config.randomness = {"seed": seed}
    config.auto_scale_lr = {
        "enable": False,
        "base_batch_size": spec["batch"],
    }

    config.train_cfg.max_epochs = epochs
    config.train_cfg.val_interval = max(1, epochs // 10)
    config.optim_wrapper.optimizer.lr = spec["lr"]
    config.param_scheduler = [
        {
            "type": "LinearLR",
            "start_factor": 1e-3,
            "by_epoch": False,
            "begin": 0,
            "end": 200,
        },
        {
            "type": "CosineAnnealingLR",
            "eta_min": spec["lr"] * 0.05,
            "begin": epochs // 3,
            "end": epochs,
            "T_max": epochs - epochs // 3,
            "by_epoch": True,
            "convert_to_iter_based": True,
        },
    ]

    for hook in config.get("custom_hooks", []):
        if "PipelineSwitchHook" in hook["type"]:
            hook["switch_epoch"] = epochs - max(1, epochs // 7)

    def set_dataset(dataloader, annotation_file, test_mode):
        dataloader.dataset.data_root = ""
        dataloader.dataset.ann_file = str(annotation_file)
        dataloader.dataset.data_prefix = {
            "img": str(image_dir) + "/"
        }
        dataloader.dataset.pop("bbox_file", None)
        dataloader.dataset.data_mode = "topdown"
        dataloader.num_workers = min(
            8,
            os.cpu_count() or 2,
        )
        dataloader.persistent_workers = True
        if test_mode:
            dataloader.dataset.test_mode = True

    set_dataset(
        config.train_dataloader,
        train_json,
        False,
    )
    config.train_dataloader.batch_size = spec["batch"]

    set_dataset(
        config.val_dataloader,
        val_json,
        True,
    )
    config.val_dataloader.batch_size = 32
    config.test_dataloader = copy.deepcopy(config.val_dataloader)

    evaluator = {
        "type": "CocoMetric",
        "ann_file": str(val_json),
    }
    config.val_evaluator = evaluator
    config.test_evaluator = evaluator

    config.default_hooks.checkpoint = {
        "type": "CheckpointHook",
        "interval": config.train_cfg.val_interval,
        "save_best": "coco/AP",
        "rule": "greater",
        "max_keep_ckpts": 1,
    }
    config.default_hooks.logger = {
        "type": "LoggerHook",
        "interval": 20,
    }

    output = (
        Path(project_dir)
        / "configs"
        / f"{model_name}_fold{fold}.py"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    config.dump(str(output))
    return output


def predict_one(model, image, box_xyxy, flip_test=True):
    import numpy as np
    from mmpose.apis import inference_topdown

    model.test_cfg["flip_test"] = flip_test
    result = inference_topdown(
        model,
        image,
        bboxes=np.asarray(box_xyxy, dtype=float)[None],
        bbox_format="xyxy",
    )[0]

    keypoints = result.pred_instances.keypoints[0].astype(float)
    scores = result.pred_instances.keypoint_scores[0].astype(float)
    return keypoints, scores


def unflip_keypoints(keypoints, scores, image_width):
    keypoints = keypoints.copy()
    keypoints[:, 0] = image_width - 1 - keypoints[:, 0]
    return keypoints[FLIP_PERM], scores[FLIP_PERM]
