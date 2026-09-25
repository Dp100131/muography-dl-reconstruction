#!/bin/bash
#SBATCH --job-name=muo-classical
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

# Stage 13.1 + 13.2 on CPU: the classical denoising grid and the no-model
# equivalence grid over the validation split. No GPU is needed, so this takes
# the third of the four QoS submit slots and leaves one deliberately free.
# Submit from the code directory:  sbatch scripts/cluster/classical_cpu.sbatch

set -euo pipefail
mkdir -p logs

# env_gpu.sh brings the venv (numpy, scikit-image, matplotlib). Its cuda/11.8
# load is guarded there, so a CPU partition without the module only warns.
source "$SLURM_SUBMIT_DIR/scripts/cluster/env_gpu.sh"
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-8}"

# The two CPU-only test suites gate the grid: the metric wrappers must still
# reproduce the frozen Stage 11 baseline and the loader must still hold its
# throughput floor before four CPU-hours are spent on top of them. The first
# run also writes results/evaluation/golden_metrics.json — commit it afterwards.
echo "=== 1/3 frozen metric contract"
python scripts/evaluation/test_metrics.py

echo "=== 2/3 dataset contract"
python scripts/training/test_dataset.py

echo "=== 3/3 Stage 13.1 + 13.2 classical grid"
python scripts/training/classical_baselines.py \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --out results/dataset/stats/stage13_baselines.json

echo "NIGHT JOB muo-classical: COMPLETE"
