#!/bin/bash
#SBATCH --job-name=muo-overfit-ext
#SBATCH --partition=gpu_titan
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

# Stage 14.5 follow-up: the 500-iteration overfit run reached train SSIM
# 0.9669 (< 0.98 criterion). Extend the iteration budget 4x under the adopted
# ref_equivalent contract to settle whether capacity or budget was the limit.
set -euo pipefail
mkdir -p logs
source scripts/cluster/env_gpu.sh
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} runs=$MUOGRAPHY_RUNS"

echo "=== Stage 14.5 overfit-10 extended (<= 2000 iterations)"
python scripts/training/train_cwgan_gp.py --config scripts/training/configs/e15_reference.json \
  --overfit-n 10 --max-iters 2000 --epochs 2000 \
  --input-scaling ref_equivalent --hflip 0 \
  --run-id e14-overfit10-ext --seed 20260805140001 --device cuda

echo "NIGHT JOB muo-overfit-ext: COMPLETE"
