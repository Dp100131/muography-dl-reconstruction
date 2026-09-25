#!/bin/bash
#SBATCH --job-name=muo-chain
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

# Phase C chain driver: one pass of the Stage 15 -> 18 state machine, then it
# re-submits itself 45 minutes out. That is what makes the chain autonomous for
# the four to five days the campaign takes -- a job that sleeps for days would
# burn a QoS slot and die at the two-day wall.
#
# It takes the CPU slot of the budget: driver + 2 GPU workers = 3 of the 4
# submitted jobs the QoS allows, and the fourth stays free so sbatch is never
# refused (the failure mode that silently stalled the Phase B campaign).
#
# Start the chain from the code directory:  sbatch scripts/cluster/chain_driver.sbatch
# Stop it: scancel the pending muo-chain job (the GPU workers finish on their own).

set -euo pipefail
mkdir -p logs

# env_gpu.sh gives the venv python; its cuda/11.8 load is guarded, so a CPU
# node without the module only warns.
source "${SLURM_SUBMIT_DIR:-.}/scripts/cluster/env_gpu.sh"
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} runs=$MUOGRAPHY_RUNS"

STATE="results/training/phase_c_chain_state.json"
SELF="scripts/cluster/chain_driver.sbatch"
DELAY="${CHAIN_DELAY:-45}"

set +e
python scripts/training/chain_driver.py
rc=$?
set -e
echo "chain_driver.py exit $rc"

# Terminal states stop the chain: COMPLETE (Stage 18 closed) and HALTED (a
# human decision is required). Any other state means another pass is due.
if [ -f "$STATE" ] && grep -Eq '"status": "(COMPLETE|HALTED)"' "$STATE"; then
  echo "chain state is terminal — not resubmitting:"
  grep -E '"(phase|status|detail)"' "$STATE" || true
  exit 0
fi

# Two guards before the self-resubmit: never queue a second driver, and never
# fill the last QoS slot.
pending_chain="$(squeue -h -n muo-chain -t PD -u "$USER" 2>/dev/null | wc -l | tr -d ' ')"
own_jobs="$(squeue -h -u "$USER" 2>/dev/null | wc -l | tr -d ' ')"
pending_chain="${pending_chain:-0}"
own_jobs="${own_jobs:-0}"
echo "pending muo-chain=$pending_chain own jobs=$own_jobs"

if [ "$pending_chain" != "0" ]; then
  echo "a muo-chain job is already pending — not queueing a second driver"
  exit 0
fi

# A transient full queue must not end the chain permanently: the driver is the
# only thing that keeps it moving, so it retries for ten minutes before giving
# up and asking for a hand. (The wall is 30 min, so ten 60 s retries fit.)
for attempt in $(seq 1 10); do
  own_jobs="$(squeue -h -u "$USER" 2>/dev/null | wc -l | tr -d ' ')"
  own_jobs="${own_jobs:-0}"
  if [ "$own_jobs" -lt 4 ] && sbatch --begin="now+${DELAY}minutes" "$SELF"; then
    echo "next pass queued (attempt $attempt, own jobs=$own_jobs)"
    exit 0
  fi
  echo "resubmit attempt $attempt failed (own jobs=$own_jobs) — retrying in 60 s"
  sleep 60
done

echo "ERROR: could not queue the next chain pass after 10 attempts."
echo "The chain is STOPPED. Restart it by hand with: sbatch $SELF"
exit 1
