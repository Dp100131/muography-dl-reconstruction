#!/bin/bash
#SBATCH --job-name=muo-e15-a1
#SBATCH --partition=gpu_titan
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

# Stage 15 initial training, attempt 1: reference config C0 at the 5% level.
# Pre-registered acceptance: no divergence, best val SSIM > 0.7777 (best
# classical), no checkerboard signature, resume reproducibility. Submit from
# the code directory.
set -euo pipefail
mkdir -p logs
source scripts/cluster/env_gpu.sh
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} runs=$MUOGRAPHY_RUNS"

# RUN_ID/SEED overridable via sbatch --export for attempts 2 and 3.
RUN_ID="${RUN_ID:-e15-c0-attempt1}"
SEED="${SEED:-20260805150000}"
# 1200 s of margin: the trainer only checks its STOP file between epochs and an
# epoch takes ~300 s at the 5% level, so a shorter signal would let SIGKILL
# land in the middle of one and throw the epoch away.
trap 'mkdir -p "$MUOGRAPHY_RUNS/$RUN_ID"; touch "$MUOGRAPHY_RUNS/$RUN_ID/STOP"; \
      echo "SIGUSR1: STOP requested for $RUN_ID"' USR1

echo "=== Stage 15 C0 $RUN_ID (5%, ref_equivalent, patience 50, seed $SEED)"
set +e
python scripts/training/train_cwgan_gp.py \
  --config scripts/training/configs/e15_reference.json \
  --run-id "$RUN_ID" --seed "$SEED" --device cuda &
child=$!
# A bare `wait $!` under set -e dies with status 138 the moment USR1 arrives,
# killing python mid-epoch — exactly what the signal exists to prevent. Status
# above 128 means the wait was interrupted, not that the child failed.
while true; do
  wait "$child"
  status=$?
  if [ "$status" -gt 128 ] && kill -0 "$child" 2>/dev/null; then
    continue
  fi
  break
done
set -e

# The job name is set at submit time (muo-e15-a1/a2/a3), so the closing line
# reports the real one instead of the default in the header.
echo "JOB ${SLURM_JOB_NAME:-muo-e15-a1} ($RUN_ID): exit $status"
exit "$status"
