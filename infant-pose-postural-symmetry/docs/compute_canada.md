# Compute Canada workflow

The original notebook used persistent paths under `$SCRATCH`.

Recommended environment variables:

```bash
export SYRIP_DIR=$SCRATCH/SyRIP
export CKPT_DIR=$SCRATCH/mmpose_ckpts
export PROJECT=$SCRATCH/infant_symmetry
```

## Artifacts to copy from the completed run

For exact evaluation without retraining, copy:

```text
$PROJECT/outputs/pose_predictions/
$PROJECT/outputs/pose_metrics.csv
$PROJECT/outputs/flip_consistency.csv
$PROJECT/outputs/robustness_pck.csv
$PROJECT/outputs/gat_ablation.csv
$PROJECT/outputs/symmetry_results_full.csv
$PROJECT/outputs/summary.json
```

For exact model reproduction, also copy:

```text
$PROJECT/work_dirs/HRNet-W32_fold0/
$PROJECT/work_dirs/HRNet-W32_fold1/
$PROJECT/work_dirs/HRNet-W32_fold2/
$PROJECT/work_dirs/HRNet-W32_fold3/
$PROJECT/work_dirs/HRNet-W32_fold4/

$PROJECT/work_dirs/RTMPose-S_fold0/
$PROJECT/work_dirs/RTMPose-S_fold1/
$PROJECT/work_dirs/RTMPose-S_fold2/
$PROJECT/work_dirs/RTMPose-S_fold3/
$PROJECT/work_dirs/RTMPose-S_fold4/
```

Each work directory only needs:

```text
best_coco_AP*.pth
DONE
*.py
*.json
*.log
```

You can omit intermediate epoch checkpoints.

## Commands to inventory the run

```bash
tree -L 3 $PROJECT > project_tree.txt
find $PROJECT/work_dirs -type f | sort > work_dirs_files.txt
find $PROJECT/outputs -type f | sort > output_files.txt
```

Upload those text files if the directory layout differs from the structure above.
