#!/bin/bash
#SBATCH --job-name=muo-e20-test
#SBATCH --partition=gpu_titan
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

# Stage 20 — THE SINGLE TEST PASS. Submit this job ONCE, EVER.
#
# ##########################################################################
# #  DO NOT SUBMIT BEFORE THE PHASE D GATE.                                #
# #                                                                        #
# #  Required, all four, per the execution plan section 2.6:               #
# #    1. explicit GO from the user                                        #
# #    2. protocol-v1.1.md frozen to ZERO TBDs and tagged (non-negotiable)  #
# #    3. the external backup of the six best.pt checkpoints done and       #
# #       verified against models/weights-manifest.json                      #
# #    4. the Stage 19 dry run green (scripts/cluster/e19_dryrun.sbatch), with  #
# #       z_thr, X, the S2 floors and the power table written INTO v1.1     #
# #                                                                        #
# #  Strongly recommended first: the OE2 signatures, the framework          #
# #  ratification and the E14/E19 code review. The test set is              #
# #  irrepeatable -- running this job early trades an irreversible asset    #
# #  for schedule.                                                         #
# #                                                                        #
# #  The test split is opened here and nowhere else. The held-out guard    #
# #  variable is set INSIDE run_evaluation.py, by its resolve_geometries()  #
# #  and by nothing else -- do NOT export it from this script. Naming it    #
# #  here would already fail scripts/evaluation/test_single_callsite.py, which is   #
# #  what keeps the single-use claim of the thesis verifiable.              #
# ##########################################################################
set -euo pipefail

# ==========================================================================
# MECHANICAL GATE. The four conditions above are prose until something
# refuses to run without them. Everything below aborts the job BEFORE the
# split is opened, because the access log is append-only: a job that dies
# after opening leaves an entry that can never be taken back, and "we opened
# the test set by accident" is not a sentence this thesis can afford.
#
# The job will not start unless it is submitted deliberately:
#     sbatch --export=ALL,PHASE_D_GATE=protocol-v1.1 scripts/cluster/e20_test_pass.sbatch
# There is no default and no interactive prompt: a variable that has to be
# typed on the submission line cannot be supplied by muscle memory.
# ==========================================================================
: "${PHASE_D_GATE:?GATE: submit with --export=ALL,PHASE_D_GATE=protocol-v1.1 after the explicit user GO}"
if [ "$PHASE_D_GATE" != "protocol-v1.1" ]; then
  echo "FATAL: PHASE_D_GATE is '${PHASE_D_GATE}', expected 'protocol-v1.1'" >&2
  exit 1
fi

# 1. the split has never been opened. A resume after a crash is legitimate and
#    is declared with PHASE_D_RESUME_REASON, which the driver records as a
#    `restart` event instead of a second opening.
ACCESS_LOG="results/evaluation/test_access_log.json"
RESUME_ARGS=()
if [ -e "$ACCESS_LOG" ]; then
  if [ -n "${PHASE_D_RESUME_REASON:-}" ]; then
    echo "RESUMING a previous attempt: ${PHASE_D_RESUME_REASON}"
    RESUME_ARGS=(--resume --resume-reason "${PHASE_D_RESUME_REASON}")
  else
    echo "FATAL: test already opened once — ${ACCESS_LOG} exists." >&2
    echo "       Nothing here may run a second time. If a previous attempt" >&2
    echo "       crashed, resubmit with --export=ALL,PHASE_D_GATE=protocol-v1.1," >&2
    echo "       PHASE_D_RESUME_REASON='<what happened>'." >&2
    exit 1
  fi
fi

# 2. protocol v1.1 exists and is frozen to zero TBDs
# NOTE: the protocol document lives in the thesis write-up, not in this code
# repository; point PROTOCOL at your local copy before running this gate.
PROTOCOL="docs/protocol/protocol-v1.1.md"
if [ ! -f "$PROTOCOL" ]; then
  echo "FATAL: ${PROTOCOL} does not exist — the protocol is not frozen" >&2
  exit 1
fi
if grep -q "TBD" "$PROTOCOL"; then
  echo "FATAL: ${PROTOCOL} still contains TBD:" >&2
  grep -n "TBD" "$PROTOCOL" >&2
  echo "       Its own rule is that it is approved only at zero TBDs." >&2
  exit 1
fi

# 3. the numeric thresholds are frozen, not provisional. Scoring the test
#    split against provisional numbers is a post-hoc verdict in a
#    pre-registration's clothes, and the split cannot be re-read to fix it.
THRESHOLDS="results/evaluation/protocol_thresholds.json"
if [ ! -f "$THRESHOLDS" ]; then
  echo "FATAL: ${THRESHOLDS} does not exist — z_thr, X and the S2 floors" >&2
  echo "       must be frozen from the dry run before the test is read" >&2
  exit 1
fi
if grep -Eq '"provisional"[[:space:]]*:[[:space:]]*true' "$THRESHOLDS"; then
  echo "FATAL: ${THRESHOLDS} still declares \"provisional\": true" >&2
  exit 1
fi

echo "GATE PASSED: protocol frozen, thresholds frozen, split never opened"

mkdir -p logs
source scripts/cluster/env_gpu.sh
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} date=$(date -u +%FT%TZ)"
echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

CACHE="${MUOGRAPHY_RUNS}/e20-cache"
mkdir -p "$CACHE"

echo "=== pre-flight: the frozen metric contract and the single-call-site guard"
python scripts/evaluation/test_metrics.py
python scripts/evaluation/test_single_callsite.py

echo "=== pre-flight: ROI checks V1-V5 on the TEST and GENERALIZATION geometries"
# These read the geometry catalog and the stored surrogate opacity maps, not
# the count shards, and they run inside this authorized job.
python scripts/evaluation/test_roi_cluster.py \
  --geometries geo-test-01 geo-test-02 geo-test-03 geo-gen-01 \
  --overlay-dir results/evaluation/roi-overlays-test

echo "=== THE SINGLE PASS over the test split and geo-gen-01"
python scripts/evaluation/run_evaluation.py --split test --out-root "$CACHE" \
  --runs-root "$MUOGRAPHY_RUNS" --device cuda \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}   # set -u safe on an empty array

echo "=== master tables, factor analysis and figures (post-processing only)"
python scripts/evaluation/report_tables.py --cache "$CACHE" --out results/evaluation/fragments
python scripts/evaluation/factor_analysis.py --cache "$CACHE" \
  --out results/evaluation/fragments --bootstrap 10000
python scripts/evaluation/report_figures.py --cache "$CACHE" \
  --out figures/results \
  --factors results/evaluation/fragments/factor_tables.json

echo "=== access log (transcribe this into the research log)"
cat results/evaluation/test_access_log.json

echo "=== cache inventory"
du -sh "$CACHE"
find "$CACHE" -name '*.npz' -printf '%s\t%p\n' | sort -rn | head -20
sha256sum "$CACHE"/metrics.csv "$CACHE"/octiles.csv "$CACHE"/summary.json

echo "NIGHT JOB muo-e20-test: COMPLETE — Stages 21 and 22 read this cache only"
