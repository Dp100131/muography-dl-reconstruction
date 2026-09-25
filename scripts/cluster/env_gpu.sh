# SC3 environment for TRAINING jobs (partition gpu_titan).
# Source it from every training job:  source scripts/cluster/env_gpu.sh
#
# Do NOT load gnu14/geant4 here: that toolchain belongs to the simulator
# (scripts/cluster/env.sh) and its libstdc++ collides with the CUDA wheels.
#
# This file deliberately does NOT run `set -euo pipefail`: it is sourced, so it
# would change the caller's shell options (an interactive shell included). The
# job scripts set the flags themselves before sourcing.

# `module` is a shell function that trips over `set -u`; disable it only while
# the module commands run, then restore whatever the caller had.
case "$-" in
  *u*) _muo_nounset=1 ;;
  *)   _muo_nounset=0 ;;
esac
set +u

module purge || true
# Tolerated, not silenced: the CPU partitions (classical_cpu.sbatch) may not
# expose the CUDA module and do not need it.
module load cuda/11.8 || echo "warning: cuda/11.8 not loaded (CPU node?)" >&2
module load conda/machine_learning 2>/dev/null || true

export MUOGRAPHY_TRAIN_HOME="${MUOGRAPHY_TRAIN_HOME:-$HOME/muography-train}"
# shellcheck disable=SC1091
source "$MUOGRAPHY_TRAIN_HOME/venv/bin/activate"

if [ "$_muo_nounset" = "1" ]; then
  set -u
fi
unset _muo_nounset

# Runs live on the NFS /home so they survive the death of a compute node;
# --resume auto then costs at most one epoch.
export MUOGRAPHY_RUNS="${MUOGRAPHY_RUNS:-$MUOGRAPHY_TRAIN_HOME/runs}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p "$MUOGRAPHY_RUNS"

# The code directory mirrors the repository layout: scripts/training/, scripts/evaluation/,
# dataset/shards (raw, gitignored), results/dataset/{splits.json,stats}, scripts/cluster/.
cd "$MUOGRAPHY_TRAIN_HOME/code"
