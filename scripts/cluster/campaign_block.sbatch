#!/bin/bash
#SBATCH --job-name=muo-block
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:55:00
#SBATCH --output=logs/%x-%j.out
# Optional (fill before submitting):
# #SBATCH --mail-type=FAIL
# #SBATCH --mail-user=your_email@correo.uis.edu.co

# One campaign block = (geometry, view, block_index) with its own derived
# seed and event budget sized to finish well under the 1-hour QoS wall.
# Blocks are the checkpoint unit: submit_missing.py re-submits only gaps.
#
# Required exported variables (submit_missing.py sets them):
#   GEO VIEW BLOCK SEED and either EVENTS (fixed count) or DAYS (calibrated
#   exposure -> per-view Poisson N, same statistical contract as the
#   surrogate campaign)

set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/cluster/env.sh"

: "${GEO:?}" "${VIEW:?}" "${BLOCK:?}" "${SEED:?}"
BUDGET_ARGS=()
[[ -n "${EVENTS:-}" ]] && BUDGET_ARGS+=(--events "$EVENTS")
[[ -n "${DAYS:-}" ]] && BUDGET_ARGS+=(--days "$DAYS")
[[ ${#BUDGET_ARGS[@]} -gt 0 ]] || { echo "need EVENTS or DAYS"; exit 2; }

BLOCK_DIR="$MUOGRAPHY_SCRATCH/campaign/$GEO/view$VIEW"
FINAL="$BLOCK_DIR/block$(printf '%04d' "$BLOCK")"
# Idempotent short-circuit: another job already published this block
# (in-flight resubmission race) — do NOT waste a full view re-simulation.
if [[ -f "$FINAL/block.sha256" ]]; then
  echo "block already published: $FINAL — skipping"
  exit 0
fi
mkdir -p "$BLOCK_DIR"
WORK="$BLOCK_DIR/tmp-block$BLOCK-$SLURM_JOB_ID"
mkdir -p "$WORK"
cd "$WORK"

BIN="$SLURM_SUBMIT_DIR/build_sc3/muography_sim"
CATALOG="$SLURM_SUBMIT_DIR/scripts/simulation/geometry_catalog/geometries"
TABLES="$SLURM_SUBMIT_DIR/scripts/simulation/geant4_app/tables"

"$BIN" --geometry "$CATALOG/$GEO.json" --view "$VIEW" \
       --tables "$TABLES" "${BUDGET_ARGS[@]}" --seed "$SEED" --out "$WORK" \
       --threads "$SLURM_CPUS_PER_TASK"

# Atomic publish: only completed blocks become visible to the merger.
# If a racing duplicate published first, discard our copy (mv onto an
# existing dir would NEST it and corrupt the layout).
STEM="${GEO}_view${VIEW}"
sha256sum "$WORK/${STEM}_hist.csv" > "$WORK/block.sha256"
if [[ -e "$FINAL" ]]; then
  echo "lost publish race for $FINAL — discarding duplicate result"
  rm -rf "$WORK"
else
  mv "$WORK" "$FINAL"
  echo "block done: $FINAL"
fi

# Self-perpetuating chain: refill the queue with pending blocks so the
# campaign completes unattended (QoS MaxSubmitPU=4; concurrent refills at
# the limit fail harmlessly and the next finishing job retries).
if [[ "${CHAIN:-0}" == "1" ]]; then
  cd "$SLURM_SUBMIT_DIR"
  python3 scripts/cluster/submit_missing.py --days "${DAYS:-30}" \
      --partition "${SLURM_JOB_PARTITION}" --max-submit 4 || true
fi
