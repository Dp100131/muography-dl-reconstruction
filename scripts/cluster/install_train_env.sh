#!/bin/bash
# Build the pinned training venv. Run this on the guane FRONTEND — it is the
# only host with egress to PyPI — and only AFTER confirming the driver version
# and that the wheels carry sm_50/sm_52 cubins. The driver check always
# precedes any pip install.
#
# Fallback ladder (first one that installs wins):
#   1. torch 2.6.0+cu118 on the conda/machine_learning python (parity with the
#      local Windows executor)
#   2. torch 2.7.1+cu118 on the same python
#   3. torch 2.4.1+cu118 on the system python3.9
#
# Run:  bash scripts/cluster/install_train_env.sh

set -euo pipefail

TRAIN_HOME="${MUOGRAPHY_TRAIN_HOME:-$HOME/muography-train}"
VENV="$TRAIN_HOME/venv"
FREEZE="$TRAIN_HOME/env-freeze.txt"
INDEX_URL="https://download.pytorch.org/whl/cu118"

mkdir -p "$TRAIN_HOME"

make_venv() {
  local python_bin="$1"
  echo "--- creating venv with $python_bin"
  rm -rf "$VENV"
  "$python_bin" -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install -U pip
}

try_torch() {
  local spec="$1"
  echo "--- trying $spec from $INDEX_URL"
  pip install "$spec" --index-url "$INDEX_URL"
}

# module is a shell function that does not survive `set -u`
set +u
module purge
module load conda/machine_learning 2>/dev/null || true
set -u

CHOSEN=""
if make_venv python3; then
  for SPEC in "torch==2.6.0+cu118" "torch==2.7.1+cu118"; do
    if try_torch "$SPEC"; then
      CHOSEN="$SPEC on $(python -V 2>&1) (module conda/machine_learning)"
      break
    fi
  done
fi

if [ -z "$CHOSEN" ]; then
  echo "--- conda python failed; falling back to the system python3.9"
  set +u
  module purge
  set -u
  make_venv python3.9
  if try_torch "torch==2.4.1+cu118"; then
    CHOSEN="torch==2.4.1+cu118 on $(python -V 2>&1) (system python3.9)"
  else
    echo "ERROR: no CUDA 11.8 torch wheel could be installed. Report this in"
    echo "the research log — risk R1 of the Phase C plan (plan B: run the whole"
    echo "phase on the local GTX 1070)."
    exit 1
  fi
fi

pip install numpy scikit-image matplotlib

python -c "import torch; print(torch.__version__, torch.version.cuda)"

{
  echo "# chosen combination: $CHOSEN"
  echo "# recorded $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
  pip freeze
} > "$FREEZE"

echo "--- chosen: $CHOSEN"
echo "--- freeze written to $FREEZE"
echo "TRAIN ENV INSTALL: COMPLETE"
