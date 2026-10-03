#!/bin/bash
#SBATCH --job-name=infant-deploy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/deployment_%j.out

set -euo pipefail

PROJECT=${PROJECT:-$PWD}
cd "$PROJECT"
mkdir -p logs

source "${INFANT_SYMMETRY_ENV:-$HOME/envs/infsym/bin/activate}"

python scripts/benchmark_deployment.py \
  --project-dir "$PROJECT" \
  --fold 0
