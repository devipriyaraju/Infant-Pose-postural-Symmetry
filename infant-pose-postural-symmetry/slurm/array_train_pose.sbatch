#!/bin/bash
#SBATCH --job-name=infant-pose-array
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$PWD}
MODELS=("HRNet-W32" "RTMPose-S")

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
MODEL=${MODELS[$MODEL_INDEX]}

source ~/.bashrc
conda activate infant-symmetry

mkdir -p "$PROJECT_DIR/logs"

python scripts/train_pose.py   --project-dir "$PROJECT_DIR"   --model "$MODEL"   --fold "$FOLD"
