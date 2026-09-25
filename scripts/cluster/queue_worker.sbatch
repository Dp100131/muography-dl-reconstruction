#!/bin/bash
#SBATCH --job-name=muo-qw
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

# Phase C queue worker: pulls configurations out of a grid until none is left.
# One of these per GPU of felix (job names muo-qw-a and muo-qw-b, set at submit
# time), which is how 12+ configurations fit under a QoS that allows 4 jobs.
#
# Submitted by scripts/training/chain_driver.py, or by hand from the code directory:
#   sbatch --job-name=muo-qw-a \
#     --export=ALL,PHASE=e16,GRID=scripts/training/configs/e16_grid.json,WORKER=a \
#     scripts/cluster/queue_worker.sbatch
#
# SIGUSR1 fires 1200 s before the wall clock -- FOUR times the ~300 s an epoch
# takes at the 5% level, because the trainer only checks its STOP file between
# epochs: at @300 the epoch in flight would still be running when SIGKILL
# arrived and the job would lose it mid-checkpoint. The trap stops ONLY the run
# this worker is training (run_queue publishes it in .worker-$WORKER-current),
# so it never kills the other worker's run, and drops .worker-$WORKER-stop so
# the loop does not claim a configuration it has no time to finish.
#
# When the job does end at the wall clock, the run continues in the NEXT worker
# the chain driver submits: --requeue only covers node failure and preemption,
# not a job that reaches its time limit.

set -euo pipefail
mkdir -p logs

source "${SLURM_SUBMIT_DIR:-.}/scripts/cluster/env_gpu.sh"

PHASE="${PHASE:?PHASE must be exported (e16 | e17 | e18)}"
GRID="${GRID:?GRID must be exported (path to the grid JSON)}"
WORKER="${WORKER:-a}"
RECLAIM_STALE_MINUTES="${RECLAIM_STALE_MINUTES:-30}"
CURRENT_FILE="$MUOGRAPHY_RUNS/.worker-$WORKER-current"
STOP_FILE="$MUOGRAPHY_RUNS/.worker-$WORKER-stop"

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} runs=$MUOGRAPHY_RUNS"
echo "=== queue worker $WORKER, phase $PHASE, grid $GRID"

on_usr1() {
  : > "$STOP_FILE"
  if [ -f "$CURRENT_FILE" ]; then
    read -r rid < "$CURRENT_FILE" || rid=""
    if [ -n "$rid" ]; then
      mkdir -p "$MUOGRAPHY_RUNS/$rid"
      touch "$MUOGRAPHY_RUNS/$rid/STOP"
      echo "SIGUSR1: STOP requested for $rid (worker $WORKER)"
      return
    fi
  fi
  echo "SIGUSR1: no run in flight for worker $WORKER, stopping the loop"
}
trap on_usr1 USR1

# The architecture tests gate the round: every generator and critic the grids
# can name is built and shape-checked here, so a typo in a model variant costs
# two minutes now instead of failing 120 epochs into an ablation.
echo "=== architecture unit tests"
python scripts/training/test_model_units.py

# --signal=B:USR1@1200 signals only the batch shell, and bash runs traps between
# commands: a foreground python would never see it. The worker therefore runs in
# the background and is waited on explicitly.
set +e
python scripts/training/run_queue.py \
  --grid "$GRID" --phase "$PHASE" --worker "$WORKER" \
  --reclaim-stale-minutes "$RECLAIM_STALE_MINUTES" &
child=$!
while true; do
  wait "$child"
  status=$?
  # status > 128 means wait was interrupted by the trapped signal, not that the
  # worker died: keep waiting while the process is still alive.
  if [ "$status" -gt 128 ] && kill -0 "$child" 2>/dev/null; then
    continue
  fi
  break
done
set -e

rm -f "$CURRENT_FILE" "$STOP_FILE"

# 3 is informational (this worker ran at least one configuration and the rest
# are held by the other worker), so it must not mark the job FAILED.
if [ "$status" = "3" ]; then
  echo "queue worker $WORKER: partial round, other configurations claimed elsewhere"
  status=0
fi

echo "QUEUE WORKER $WORKER ($PHASE): exit $status"
exit "$status"
