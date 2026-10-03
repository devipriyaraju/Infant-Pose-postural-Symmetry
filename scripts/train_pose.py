#!/usr/bin/env python
import argparse
import gc
import time
from pathlib import Path


def strip_albumentation(pipeline):
    return [
        transform
        for transform in pipeline
        if "Albumentation" not in str(transform.get("type", ""))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument(
        "--model",
        choices=["HRNet-W32", "RTMPose-S"],
        required=True,
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner

    project = Path(args.project_dir).resolve()
    config_path = project / "configs" / f"{args.model}_fold{args.fold}.py"
    config = Config.fromfile(str(config_path))
    work_dir = Path(config.work_dir)
    done = work_dir / "DONE"

    if done.exists():
        print(f"Already complete: {work_dir}")
        return

    last_checkpoint = work_dir / "last_checkpoint"
    if last_checkpoint.exists():
        checkpoint_path = Path(
            last_checkpoint.read_text(encoding="utf-8").strip()
        )
        if checkpoint_path.exists():
            config.resume = True
            config.load_from = None

    try:
        import albumentations
        _ = albumentations
    except ImportError:
        config.train_dataloader.dataset.pipeline = strip_albumentation(
            config.train_dataloader.dataset.pipeline
        )
        for hook in config.get("custom_hooks", []):
            if "PipelineSwitchHook" in hook["type"]:
                hook["switch_pipeline"] = strip_albumentation(
                    hook["switch_pipeline"]
                )

    start = time.time()
    runner = Runner.from_cfg(config)
    runner.train()
    done.touch()
    print(f"Finished in {(time.time() - start) / 60:.1f} minutes")

    del runner
    gc.collect()


if __name__ == "__main__":
    main()
