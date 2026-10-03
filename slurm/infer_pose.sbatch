#!/bin/bash
#SBATCH --job-name=infant-infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MODEL=${MODEL:-HRNet-W32}
MODE=${MODE:-finetuned}
PROJECT_DIR=${PROJECT_DIR:-$PWD}
CKPT_DIR=${CKPT_DIR:-$PROJECT_DIR/checkpoints/mmpose}

source ~/.bashrc
conda activate infant-symmetry

mkdir -p "$PROJECT_DIR/logs"

python scripts/run_pose_inference.py   --project-dir "$PROJECT_DIR"   --checkpoint-dir "$CKPT_DIR"   --model "$MODEL"   --mode "$MODE"
