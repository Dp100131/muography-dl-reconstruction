#!/bin/bash
#SBATCH --job-name=muo-e19-retrain
#SBATCH --partition=gpu_titan
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

# Stage 19.5: retrain the U-Net L1 baselines whose e13 checkpoints were
# pruned by the quota policy. Archived configs and seeds verbatim; the
# acceptance check (val SSIM within +-0.005 of the archived value) runs in
# the Stage 19 dry run. GROUP=a runs {f50, f10, f5-long}; GROUP=b runs
# {f1, f1-shared} plus the lambda_adv=0 final-architecture probes.
set -euo pipefail
mkdir -p logs
source scripts/cluster/env_gpu.sh
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} group=${GROUP:-a}"

run_l1() {  # fraction, run_id, seed, extra args...
  local frac="$1" rid="$2" seed="$3"; shift 3
  echo "=== U-Net L1 retrain $rid (f=$frac)"
  python scripts/training/train_unet_l1.py --fraction "$frac" \
    --input-scaling ref_equivalent --hflip 1 \
    --run-id "$rid" --seed "$seed" --device cuda "$@" &
  local child=$!
  while true; do
    set +e; wait "$child"; status=$?; set -e
    if [[ $status -gt 128 ]] && kill -0 "$child" 2>/dev/null; then continue; fi
    break
  done
  [[ $status -eq 0 ]] || exit "$status"
}

if [[ "${GROUP:-a}" == "a" ]]; then
  run_l1 0.50 e19-unetl1-f50 20260805131000 --epochs 150
  run_l1 0.10 e19-unetl1-f10 20260805131100 --epochs 150
  run_l1 0.05 e19-unetl1-f5  20260805131500 --epochs 300 --patience 50
else
  run_l1 0.01 e19-unetl1-f1 20260805131300 --epochs 150
  echo "=== U-Net L1 retrain e19-unetl1-f1-shared (continuity, shared contract)"
  python scripts/training/train_unet_l1.py --fraction 0.01 --epochs 150 \
    --input-scaling shared --hflip 1 \
    --run-id e19-unetl1-f1-shared --seed 20260805130300 --device cuda
  echo "=== final-architecture L1-only probes (unet_dc_5, lambda_adv=0)"
  python scripts/training/train_cwgan_gp.py --config scripts/training/configs/e17_final.json \
    --lambda-adv 0 --fraction 0.05 --epochs 150 --patience 50 \
    --run-id e19-dc5-l1only-f5 --seed 20260805191000 --device cuda
  python scripts/training/train_cwgan_gp.py --config scripts/training/configs/e17_final.json \
    --lambda-adv 0 --fraction 0.01 --epochs 150 --patience 50 \
    --run-id e19-dc5-l1only-f1 --seed 20260805191100 --device cuda
fi

echo "NIGHT JOB muo-e19-retrain-${GROUP:-a}: COMPLETE"
