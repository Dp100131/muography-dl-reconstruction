#!/usr/bin/env bash
# OE1 stage 1 (P2): legacy-inertness check WITHOUT transport. Builds the probe
# tests/g4/legacy_inertness.cc twice, against the app at the pre-P2 commit
# (ORIG_APP) and against this P2 tree, and compares
#   1. the primaries of the legacy sampler (same private engine, same seed);
#   2. hist.csv (byte for byte) and meta.json (equal but for the two timing
#      keys) and the "RUN DONE" line written by the legacy RunAction.
# Needs Geant4 (geant4-config on PATH); run inside an allocation.
#
# Usage: tests/g4/legacy_inertness.sh ORIG_APP_DIR OUT_DIR
#   ORIG_APP_DIR : scripts/simulation/geant4_app extracted from the pre-P2 commit
#                  (e.g. git archive 9520170 scripts/simulation | tar -x -C DIR)
#   OUT_DIR      : must not exist
set -u -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW="$(cd "$HERE/../.." && pwd)"
ORIG="$(cd "${1:?usage: legacy_inertness.sh ORIG_APP_DIR OUT_DIR}" && pwd)"
OUT="${2:?usage: legacy_inertness.sh ORIG_APP_DIR OUT_DIR}"
mkdir "$OUT" || exit 2
OUT="$(cd "$OUT" && pwd)"
command -v geant4-config >/dev/null || { echo "geant4-config not found"; exit 2; }
CXX="${CXX:-g++}"
GEOM="${GEOMETRY_JSON:-$NEW/../geometry_catalog/geometries/geo-train-06.json}"

build() { # NAME APP_DIR EXTRA_SOURCES...
  local name="$1" app="$2"
  shift 2
  # shellcheck disable=SC2046
  "$CXX" -O3 -DNDEBUG -std=gnu++17 -pthread $(geant4-config --cflags | tr ' ' '\n' | grep -E '^-I|^-D' | tr '\n' ' ') \
    -I"$app/include" -I"$app/../third_party/json" "$NEW/tests/g4/legacy_inertness.cc" \
    "$app/src/PrimaryGeneratorAction.cc" "$app/src/RunAction.cc" "$app/src/FluxTables.cc" \
    "$app/src/GeometryConfig.cc" "$@" $(geant4-config --libs) \
    -Wl,-rpath,"$(geant4-config --prefix)/lib64" -o "$OUT/$name" >"$OUT/$name.build.log" 2>&1 ||
    { echo "BUILD FAILED: $name"; head -n 30 "$OUT/$name.build.log"; exit 1; }
  echo "built $name ($(grep -c 'warning:' "$OUT/$name.build.log") warnings)"
}
build probe_orig "$ORIG"
build probe_new "$NEW" "$NEW/src/PrimaryList.cc"

fail=0
for seed in 2026092901 2026092902 7; do
  a=$("$OUT/probe_orig" pga "$ORIG/tables" "$GEOM" 20000 $seed)
  b=$("$OUT/probe_new" pga "$NEW/tables" "$GEOM" 20000 $seed)
  if [[ "$a" == "$b" ]]; then echo "legacy sampler identical: $a"; else echo "*** legacy sampler DIFFERS"; echo "  orig: $a"; echo "  new:  $b"; fail=1; fi
done

mkdir "$OUT/run-orig" "$OUT/run-new"
"$OUT/probe_orig" run "$OUT/run-orig" >"$OUT/run-orig.stdout" 2>&1 || fail=1
"$OUT/probe_new" run "$OUT/run-new" >"$OUT/run-new.stdout" 2>&1 || fail=1
if cmp -s "$OUT/run-orig/geo-train-06_view0_hist.csv" "$OUT/run-new/geo-train-06_view0_hist.csv"; then
  echo "legacy hist.csv byte-identical ($(sha256sum "$OUT/run-new/geo-train-06_view0_hist.csv" | cut -c1-16)...)"
else
  echo "*** legacy hist.csv DIFFERS"; fail=1
fi
norm() { sed -E 's/("wall_seconds": )[^,]*/\1X/; s/("seconds_per_event": )[^,]*/\1X/' "$1"; }
if diff <(norm "$OUT/run-orig/geo-train-06_view0.meta.json") <(norm "$OUT/run-new/geo-train-06_view0.meta.json") >/dev/null; then
  echo "legacy meta.json byte-identical apart from wall_seconds / seconds_per_event"
else
  echo "*** legacy meta.json DIFFERS"; diff <(norm "$OUT/run-orig/geo-train-06_view0.meta.json") <(norm "$OUT/run-new/geo-train-06_view0.meta.json") | head; fail=1
fi
if diff <(sed -E 's/wall_s=[^ ]*/wall_s=X/' "$OUT/run-orig.stdout") <(sed -E 's/wall_s=[^ ]*/wall_s=X/' "$OUT/run-new.stdout") >/dev/null; then
  echo "legacy 'RUN DONE' line identical apart from wall_s: $(sed -E 's/wall_s=[^ ]*/wall_s=X/' "$OUT/run-new.stdout" | head -1)"
else
  echo "*** RUN DONE line DIFFERS"; fail=1
fi
grep -c oe1 "$OUT/run-new/geo-train-06_view0.meta.json" | sed 's/^/meta.json occurrences of "oe1" in a legacy run: /'
if [[ $fail -eq 0 ]]; then echo "LEGACY INERTNESS: PASS"; exit 0; fi
echo "LEGACY INERTNESS: FAIL"
exit 1
