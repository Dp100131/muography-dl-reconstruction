#!/usr/bin/env bash
# OE1 stage 1 (P2): build and run the standalone tests of the list mode.
# No Geant4 is needed: tests/ lives outside src/ because CMakeLists.txt globs
# src/*.cc, so nothing here is ever linked into muography_sim.
#
# Usage: tests/build_tests.sh [OUT_DIR]
#   CXX=g++            compiler (default g++)
#   NO_R1=1            skip test R1 (needs python3 + numpy)
#   R1_N=100000        number of random segments of R1 (per seed)
#   R1_SEEDS="..."     seeds of R1 (default: six, among them 12 and 13, which failed
#                      with the first, size-blind tolerance; plan correction W1)
#   R1_DENSE=2000      segments checked with the dense-sampling oracle
#   R1_EXACT=600       segments checked with the 60-digit reference (+ doubtful ones)
#   SKIP_SAN=1         skip the sanitizer build
#   CATALOG=DIR        catalog folder (default ../geometry_catalog/geometries)
#
# Each test is built twice: production flags (-O3 -DNDEBUG) and
# -fsanitize=address,undefined (coarser volume grids, --fast). Exit status is
# 0 only if every build succeeds with 0 warnings and every test passes. A
# summary is written to OUT_DIR/pruebas-cruces.json (compiler, flags, source
# SHA-256, the R1 seeds and each result, and the deviations from the plan's
# literal numbers).
set -u -o pipefail
# Nothing of this script may leave __pycache__ folders in the source tree.
export PYTHONDONTWRITEBYTECODE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$(cd "$HERE/.." && pwd)"
CATALOG="${CATALOG:-$APP/../geometry_catalog/geometries}"  # directory with geo-train-06.json and geo-train-01.json
CXX="${CXX:-g++}"
OUT="${1:-${TMPDIR:-/tmp}/oe1-tests-$$}"
mkdir -p "$OUT" || exit 2
OUT="$(cd "$OUT" && pwd)"  # absolute: the tests run from the app directory

INC=(-I"$APP/include" -I"$HERE" -I"$APP/../third_party/json")
COMMON=(-std=gnu++17 -Wall -Wextra -Wshadow -pedantic)
REL=(-O3 -DNDEBUG)
SAN=(-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined)

fail=0
results=()
warn_total=0

note() { printf '%s\n' "$*"; }

# build MODE NAME SOURCES... : returns 0 on success, prints compiler output
build() {
  local mode="$1" name="$2"
  shift 2
  local flags=("${REL[@]}")
  [[ "$mode" == "sanitize" ]] && flags=("${SAN[@]}")
  mkdir -p "$OUT/$mode"
  local log="$OUT/$mode/$name.build.log"
  "$CXX" "${COMMON[@]}" "${flags[@]}" -pthread "${INC[@]}" "$@" -o "$OUT/$mode/$name" >"$log" 2>&1
  local rc=$?
  local warns
  warns=$(grep -c 'warning:' "$log" || true)
  warn_total=$((warn_total + warns))
  if [[ $rc -ne 0 ]]; then
    note "BUILD FAIL [$mode] $name (see $log)"
    sed -n '1,40p' "$log"
    return 1
  fi
  note "build ok   [$mode] $name (warnings: $warns)"
  if [[ $warns -ne 0 ]]; then
    note "WARNINGS [$mode] $name: 0 are allowed (see $log)"
    sed -n '1,40p' "$log"
  fi
  return 0
}

run() { # MODE NAME ARGS...
  local mode="$1" name="$2"
  shift 2
  note "--- run [$mode] $name $*"
  if (cd "$APP" && "$OUT/$mode/$name" "$@") 2>&1 | tee "$OUT/$mode/$name.out"; then
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
      results+=("\"$mode/$name\": \"pass\"")
      return 0
    fi
  fi
  results+=("\"$mode/$name\": \"FAIL\"")
  fail=1
  return 1
}

note "compiler: $("$CXX" --version | head -n 1)"
note "output:   $OUT"

modes=(release)
[[ "${SKIP_SAN:-0}" == "1" ]] || modes+=(sanitize)

for mode in "${modes[@]}"; do
  ok=1
  build "$mode" test_region_crossing "$HERE/test_region_crossing.cc" "$APP/src/GeometryConfig.cc" || ok=0
  build "$mode" crossing_eval "$HERE/crossing_eval.cc" "$APP/src/GeometryConfig.cc" || ok=0
  build "$mode" test_primary_list "$HERE/test_primary_list.cc" "$APP/src/PrimaryList.cc" || ok=0
  build "$mode" test_oe1_support "$HERE/test_oe1_support.cc" "$APP/src/PrimaryList.cc" \
    "$APP/src/Oe1Cli.cc" || ok=0
  if [[ $ok -ne 1 ]]; then
    fail=1
    continue
  fi
  fast=()
  [[ "$mode" == "sanitize" ]] && fast=(--fast)
  run "$mode" test_region_crossing ${fast[@]+"${fast[@]}"} --catalog "$CATALOG"
  run "$mode" test_primary_list
  run "$mode" test_oe1_support
done

# R1: 1e5 random segments per seed against an independent numpy implementation,
# a 60-digit reference and a dense-sampling oracle. Several seeds, because the
# first (size-blind) tolerance passed the default seed and failed seeds 12 and 13
# (plan correction W1); every seed and its result go to the summary.
R1_SEEDS="${R1_SEEDS:-20260930 11 12 13 14 15}"
r1="skipped"
r1_seeds_run=()
if [[ "${NO_R1:-0}" != "1" && -x "$OUT/release/crossing_eval" ]]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import numpy' 2>/dev/null; then
    mkdir -p "$OUT/r1"
    note "--- R1 (numpy $(python3 -c 'import numpy; print(numpy.__version__)'), seeds: $R1_SEEDS, ${R1_N:-100000} segments each)"
    r1="pass"
    for seed in $R1_SEEDS; do
      d="$OUT/r1/seed-$seed"
      mkdir -p "$d"
      note "--- R1 seed $seed"
      seed_ok=1
      python3 "$HERE/crossing_reference.py" gen --out "$d/seg.csv" --n "${R1_N:-100000}" --seed "$seed" || seed_ok=0
      if [[ $seed_ok -eq 1 ]]; then
        "$OUT/release/crossing_eval" --geometry "$CATALOG/geo-train-06.json" --pair 0,1 \
          "$d/seg.csv" >"$d/eval.csv" || seed_ok=0
      fi
      if [[ $seed_ok -eq 1 ]]; then
        python3 "$HERE/crossing_reference.py" compare "$d/seg.csv" "$d/eval.csv" \
          --dense "${R1_DENSE:-2000}" --exact "${R1_EXACT:-600}" 2>&1 | tee "$d/compare.out"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || seed_ok=0
      fi
      r1_seeds_run+=("$seed")
      if [[ $seed_ok -eq 1 ]]; then
        results+=("\"R1_seed_$seed\": \"pass\"")
      else
        results+=("\"R1_seed_$seed\": \"FAIL\"")
        r1="FAIL"
        fail=1
      fi
    done
  else
    note "R1 SKIPPED: python3 with numpy is not available (run it where numpy exists)"
  fi
fi
results+=("\"R1\": \"$r1\"")

# Zero warnings are allowed (-Wall -Wextra -Wshadow -pedantic).
if [[ $warn_total -ne 0 ]]; then
  note "compiler warnings: $warn_total (0 are allowed)"
  fail=1
fi

json_str() {   # JSON string literal of $1 (backslash and double quote escaped)
  local s="${1//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf '"%s"' "$s"
}

# Summary.
{
  printf '{\n'
  printf '  "date_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "host": "%s",\n' "$(hostname)"
  printf '  "compiler": "%s",\n' "$("$CXX" --version | head -n 1)"
  printf '  "flags_release": "%s",\n' "${COMMON[*]} ${REL[*]} -pthread"
  printf '  "flags_sanitize": "%s",\n' "${COMMON[*]} ${SAN[*]} -pthread"
  printf '  "python": "%s",\n' "$(python3 --version 2>&1 || true)"
  printf '  "numpy": "%s",\n' "$(python3 -c 'import numpy; print(numpy.__version__)' 2>&1 || true)"
  printf '  "compiler_warnings": %d,\n' "$warn_total"
  printf '  "r1_segments_per_seed": %s,\n' "${R1_N:-100000}"
  printf '  "r1_seeds": ['
  for i in "${!r1_seeds_run[@]}"; do
    [[ $i -eq 0 ]] || printf ', '
    printf '%s' "${r1_seeds_run[$i]}"
  done
  printf '],\n'
  # Where the tests differ from the plan's literal numbers (plan correction W1).
  printf '  "deviations_from_plan": [\n'
  printf '    %s,\n' "$(json_str "R1 tolerance: the plan asks +-1e-9 m. The acceptance is tol(len, X) = 1e-9 m + 8 u S X / max(len, 1e-12 m), capped at len + 1e-9 m, with u = 2^-53, S = 72.9 m and X = max(S, largest |endpoint coordinate| of the segment). The extra term is the unavoidable double-precision error of a millimetre-long, near-tangent chord (rounding an endpoint moves the line by u X): it is 4.7e-9 m at len = 1 mm for X = S, grows in proportion to X, and is at most 2.9e-10 m for len >= 5 cm and X <= 220 m. The first version used X = S for every segment and failed seeds 12 and 13 at N = 1e5 (errors of 5e-9 to 8e-9 m at len = 1 mm and X = 135 to 220 m).")"
  printf '    %s,\n' "$(json_str "C9 (reversal): the plan asks 1e-12 relative. The test uses 1e-12 * max(len) + 128 u (|a| + |b|): a length built from coordinates of size |a|, |b| cannot be reproduced better than the rounding of those coordinates, and near-tangent chords amplify it. The worst difference against a long-double evaluation is 1.3e-11 m over 1e5 segments up to 5 km long, far inside the 1e-9 m acceptance.")"
  printf '    %s,\n' "$(json_str "V (volume by parallel-line grids), release build: 1 m^3 as in the plan, with h = 0.01 (vertical lines), 0.02 (lines along x and along detector-centroid) and 0.05 (dense body alone).")"
  printf '    %s\n' "$(json_str "V, sanitizer build (--fast): coarser grids (h = 0.02, 0.04, 0.04 and 0.1) and a tolerance of 2 m^3 instead of 1 m^3; that build exercises the code for memory and undefined behaviour, and the 1 m^3 acceptance is the release build's.")"
  printf '  ],\n'
  printf '  "sources_sha256": {\n'
  first=1
  for f in "$HERE"/*.cc "$HERE"/*.hh "$HERE"/*.py "$APP/include/RegionCrossing.hh" \
           "$APP/include/PrimaryList.hh" "$APP/include/MuonRecord.hh" "$APP/include/RoiMask.hh" \
           "$APP/include/Sha256.hh" "$APP/include/Oe1Hash.hh" "$APP/include/Oe1ListMode.hh" \
           "$APP/include/Oe1Cli.hh" "$APP/include/RunData.hh" "$APP/include/GeometryConfig.hh" \
           "$APP/src/PrimaryList.cc" "$APP/src/Oe1Cli.cc" "$APP/src/GeometryConfig.cc"; do
    [[ -f "$f" ]] || continue
    [[ $first -eq 1 ]] || printf ',\n'
    first=0
    printf '    "%s": "%s"' "${f#"$APP"/}" "$(sha256sum "$f" | cut -d' ' -f1)"
  done
  printf '\n  },\n'
  printf '  "results": {\n'
  for i in "${!results[@]}"; do
    printf '    %s' "${results[$i]}"
    [[ $i -lt $((${#results[@]} - 1)) ]] && printf ','
    printf '\n'
  done
  printf '  },\n'
  printf '  "status": "%s"\n' "$([[ $fail -eq 0 ]] && echo PASS || echo FAIL)"
  printf '}\n'
} >"$OUT/pruebas-cruces.json"

if [[ $fail -eq 0 ]]; then
  note "ALL TESTS PASSED (summary: $OUT/pruebas-cruces.json)"
  exit 0
fi
note "TESTS FAILED (summary: $OUT/pruebas-cruces.json)"
exit 1
