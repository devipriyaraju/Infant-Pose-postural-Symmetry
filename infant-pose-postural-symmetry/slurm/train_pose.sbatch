#!/bin/bash
#SBATCH --job-name=infant-pose
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MODEL=${MODEL:-HRNet-W32}
FOLD=${FOLD:-0}
PROJECT_DIR=${PROJECT_DIR:-$PWD}

source ~/.bashrc
conda activate infant-symmetry

mkdir -p "$PROJECT_DIR/logs"

python scripts/train_pose.py   --project-dir "$PROJECT_DIR"   --model "$MODEL"   --fold "$FOLD"
