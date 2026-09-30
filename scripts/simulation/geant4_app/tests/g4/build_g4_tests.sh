#!/usr/bin/env bash
# OE1 stage 1 (P2): builds and runs the Geant4-linked synthetic list-mode
# harness (tests/g4/test_list_mode_g4.cc). Needs Geant4 11.4.2 (geant4-config
# on PATH), e.g. on SC3:
#   source /etc/profile.d/lmod.sh; module purge; module load gnu14/14.2.0 geant4/11.4.2
# and should run inside an allocation (srun), not on the login node.
#
# No run manager is created and BeamOn is never called: hand-made G4Event /
# G4Step objects are fed to EventAction and RunAction (see the harness header).
#
# Usage: tests/g4/build_g4_tests.sh OUT_DIR      (OUT_DIR must not exist)
set -u -o pipefail
# Nothing of this script may leave __pycache__ folders behind.
export PYTHONDONTWRITEBYTECODE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$(cd "$HERE/../.." && pwd)"
OUT="${1:?usage: build_g4_tests.sh OUT_DIR}"
CXX="${CXX:-g++}"
mkdir "$OUT" || { echo "OUT_DIR exists or cannot be created: $OUT"; exit 2; }
command -v geant4-config >/dev/null || { echo "geant4-config not found (load the geant4 module)"; exit 2; }

echo "compiler: $("$CXX" --version | head -n 1); geant4 $(geant4-config --version)"
# shellcheck disable=SC2046
"$CXX" -O2 $(geant4-config --cflags) -I"$APP/include" -I"$APP/tests" -I"$APP/../third_party/json" \
  "$HERE/test_list_mode_g4.cc" \
  "$APP/src/EventAction.cc" "$APP/src/RunAction.cc" "$APP/src/PrimaryGeneratorAction.cc" \
  "$APP/src/PrimaryList.cc" "$APP/src/Oe1Cli.cc" "$APP/src/FluxTables.cc" "$APP/src/GeometryConfig.cc" \
  $(geant4-config --libs) -Wl,-rpath,"$(geant4-config --prefix)/lib64" \
  -o "$OUT/test_list_mode_g4" >"$OUT/build.log" 2>&1
rc=$?
warns=$(grep -c 'warning:' "$OUT/build.log" || true)
if [[ $rc -ne 0 ]]; then
  echo "BUILD FAILED (see $OUT/build.log)"
  head -n 60 "$OUT/build.log"
  exit 1
fi
echo "build ok (warnings: $warns)"

fail=0
# Strict-warning pass over the Geant4-linked sources of P2 (-Wall -Wextra -Wshadow -pedantic;
# Geant4, CLHEP and the vendored headers are system headers, so only P2's own code is judged).
strict=(-fsyntax-only -Wall -Wextra -Wshadow -pedantic)
for flag in $(geant4-config --cflags); do
  case "$flag" in
    -I*) strict+=(-isystem "${flag#-I}") ;;
    -std=*|-D*) strict+=("$flag") ;;
  esac
done
strict+=(-isystem "$APP/../third_party/json" -isystem "$APP/../third_party/ecomug" -I"$APP/include" -I"$APP/tests")
strict_warns=0
for src in "$APP/muography_sim.cc" "$APP/src/EventAction.cc" "$APP/src/RunAction.cc" \
           "$APP/src/Oe1Cli.cc" "$APP/src/PrimaryList.cc" "$APP/src/ActionInitialization.cc" \
           "$APP/src/PrimaryGeneratorAction.cc" "$APP/src/SteppingAction.cc" "$HERE/test_list_mode_g4.cc"; do
  log="$OUT/strict-$(basename "$src").log"
  "$CXX" "${strict[@]}" "$src" >"$log" 2>&1
  src_rc=$?
  n=$(grep -c 'warning:' "$log" || true)
  # muography_sim.cc keeps three warnings in P0 lines that P2 must not touch (they exist at 9520170:
  # the "\" that ends the usage comment on line 4, and the loop variable `s` of the two --slice-y /
  # --slice-z loops, which shadows the CLHEP unit s). Anything else counts.
  if [[ "$(basename "$src")" == muography_sim.cc ]]; then
    legacy=$(grep 'warning:' "$log" | grep -c -E 'multi-line comment|declaration of .s. shadows a global declaration' || true)
    n=$((n - legacy))
    echo "strict pass muography_sim.cc: $legacy known warnings in P0 lines (expected 3), $n others"
    [[ $legacy -le 3 ]] || n=$((n + legacy - 3))
  fi
  strict_warns=$((strict_warns + n))
  if [[ $src_rc -ne 0 || $n -ne 0 ]]; then
    echo "*** strict pass $(basename "$src"): exit $src_rc, $n warnings (see $log)"
    head -n 20 "$log"
    fail=1
  fi
done
echo "strict-warning pass (-Wall -Wextra -Wshadow -pedantic) over P2's Geant4-linked sources: $strict_warns warnings"
for scenario in ok-full ok-min ok-noregion aborted hist-mismatch missing-event dup-id two-workers bad-engine bad-outdir bad-variant bad-run-manager bad-both; do
  mkdir "$OUT/$scenario" || exit 2
  echo "=== $scenario"
  if ! "$OUT/test_list_mode_g4" "$scenario" "$OUT/$scenario" >"$OUT/$scenario.log" 2>&1; then
    fail=1
    echo "*** scenario $scenario FAILED"
    grep -E 'FAIL' "$OUT/$scenario.log" | head -n 40
  fi
  grep -E '^(PASS|FAIL) \[|pre-BeamOn checks|errors after finalisation|OE1-ETAPA1' "$OUT/$scenario.log" | cut -c1-400
done

# A/A identity: the same scenario twice gives a byte-identical muon log, and
# a meta that differs only in the timing keys.
mkdir "$OUT/aa" "$OUT/aa/1" "$OUT/aa/2" || exit 2
"$OUT/test_list_mode_g4" ok-full "$OUT/aa/1" >"$OUT/aa/1.log" 2>&1 || fail=1
"$OUT/test_list_mode_g4" ok-full "$OUT/aa/2" >"$OUT/aa/2.log" 2>&1 || fail=1
if cmp -s "$OUT/aa/1/run/geo-train-06_view0_muones.csv" "$OUT/aa/2/run/geo-train-06_view0_muones.csv" &&
   cmp -s "$OUT/aa/1/run/geo-train-06_view0_hist.csv" "$OUT/aa/2/run/geo-train-06_view0_hist.csv"; then
  echo "A/A: _muones.csv and _hist.csv byte-identical"
else
  echo "*** A/A: outputs differ"; fail=1
fi
python3 - "$OUT/aa/1/run/geo-train-06_view0.meta.json" "$OUT/aa/2/run/geo-train-06_view0.meta.json" <<'PY' || fail=1
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
# durations and resource use (plan W3: the getrusage keys are listed, with wall_seconds and
# seconds_per_event, in oe1_stage1_common.TIMING_META_KEYS); they differ between two identical runs
timing = ["wall_seconds", "seconds_per_event"] + sorted(k for k in a if k.startswith("rusage_"))
assert sorted(k for k in a if k.startswith("rusage_")) == ["rusage_max_rss_kb", "rusage_system_cpu_seconds", "rusage_user_cpu_seconds"], sorted(a)
for k in timing:
    a.pop(k), b.pop(k)
# the input paths differ between the two folders only through the temp dirs;
# the oe1_etapa1 block must not carry any of them
assert a == b, "meta differs beyond the timing keys"
print("A/A: meta.json equal apart from", timing)
PY

# ok-min vs ok-full: the reduced log carries the same identity/state columns.
python3 - "$OUT/ok-full/run/geo-train-06_view0_muones.csv" "$OUT/ok-min/run/geo-train-06_view0_muones.csv" <<'PY' || fail=1
import csv, sys
full = list(csv.DictReader(open(sys.argv[1])))
mini = list(csv.DictReader(open(sys.argv[2])))
assert len(full) == len(mini) == 19
same = ["caso", "geometry_id", "view", "variant", "replica", "list_sha256", "id", "prim_fnv", "rng0",
        "rng1", "detected", "iel", "iaz", "in_rint", "ekin_plane_gev", "el_plane_deg",
        "relaz_plane_deg", "status", "end_process", "n_steps"]
for f, m in zip(full, mini):
    for c in same:
        assert f[c] == m[c], (f["id"], c, f[c], m[c])
print("min vs full: identical on", len(same), "identity/state columns for", len(full), "muons")
PY

if [[ $fail -eq 0 ]]; then echo "ALL SYNTHETIC LIST-MODE SCENARIOS PASSED"; exit 0; fi
echo "SYNTHETIC LIST-MODE SCENARIOS FAILED"
exit 1
