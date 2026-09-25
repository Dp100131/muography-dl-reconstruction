#!/bin/bash
#SBATCH --job-name=muo-unetl1c
#SBATCH --partition=gpu_titan
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

# Stage 13.3 follow-up: U-Net L1 baselines re-trained under the adopted
# ref_equivalent input contract (E13.4 A/B winner). Submit from the code dir.
set -euo pipefail
mkdir -p logs
source scripts/cluster/env_gpu.sh
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} runs=$MUOGRAPHY_RUNS"

run() {
  local frac="$1" tag="$2" seed="$3"
  echo "=== U-Net L1 corrected f=$frac"
  python scripts/training/train_unet_l1.py --fraction "$frac" --epochs 150 \
    --input-scaling ref_equivalent --hflip 1 \
    --run-id "e13-unetl1c-f$tag" --seed "$seed" --device cuda
}

run 0.50 50 20260805131000
run 0.10 10 20260805131100
run 0.05 5  20260805131200
run 0.01 1  20260805131300

echo "NIGHT JOB muo-unetl1c: COMPLETE"
