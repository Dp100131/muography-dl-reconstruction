#!/usr/bin/env bash
# OE1 stage 1 (P2, plan S3): proves on the REAL binaries that the run checks that need no transport
# (the run manager class, and --variant against the geometry that was built) end a list-mode run with
# exit code 4 right after runManager->Initialize(), BEFORE BeamOn: no transport runs and no output
# file is written. The synthetic harness (test_list_mode_g4.cc) evaluates the same functions but has
# no run manager, so it cannot show the exit itself.
#
#   bash tests/g4/early_exit.sh BIN_A2 BIN_B2 GEO-TRAIN-06.json TABLES_DIR OUT_DIR   (OUT_DIR must not exist)
#
# Needs the two built binaries (Geant4 environment loaded) and the catalog file of geo-train-06 (any
# geometry whose anomalies 0 and 1 are a conduit and a dense body makes the two variants differ).
# Every failing case stops before BeamOn, so the whole script costs seconds and makes NO transport: it
# can run on SC3 inside an srun allocation. It writes a 20-muon technical list (source seed
# 2026092990, never a production seed) with --write-primaries, which is generation only.
#
# Cases (each must give exit 4, an EMPTY --out, an OE1-PRERUN-CHECK status=FAIL line with the expected
# error codes on stdout, no "RUN DONE" or OE1-ETAPA1 line, and the reason on stderr):
#   1  A2 --variant B                             variant_geometry_mismatch
#   2  B2 --variant A                             variant_geometry_mismatch
#   3  A2 --variant A, G4RUN_MANAGER_TYPE=Serial  run_manager_not_task (the sequential G4RunManager)
#   4  B2 --variant A, G4RUN_MANAGER_TYPE=Serial  run_manager_not_task,variant_geometry_mismatch
# Order of the exit codes (the errors that come first win, whatever the variant or run manager):
#   5  wrong --primaries-sha256                   exit 1 (input), not 4
#   6  no --region-pair on geo-train-06           exit 2 (argument), not 4
#   7  output folder that already holds a result  exit 1 (preflight), not 4
# OE1_EARLY_EXIT_POSITIVE=1 adds the two controls that pass the checks and then RUN (20 muons of the
# technical list): exit 0, OE1-PRERUN-CHECK status=OK before RUN DONE and the three outputs written.
# That is a transport: local development only, never on SC3.
set -u -o pipefail

BIN_A="${1:?usage: early_exit.sh BIN_A2 BIN_B2 GEO-TRAIN-06.json TABLES_DIR OUT_DIR}"
BIN_B="${2:?usage: early_exit.sh BIN_A2 BIN_B2 GEO-TRAIN-06.json TABLES_DIR OUT_DIR}"
GEOM="${3:?usage: early_exit.sh BIN_A2 BIN_B2 GEO-TRAIN-06.json TABLES_DIR OUT_DIR}"
TABLES="${4:?usage: early_exit.sh BIN_A2 BIN_B2 GEO-TRAIN-06.json TABLES_DIR OUT_DIR}"
OUT="${5:?usage: early_exit.sh BIN_A2 BIN_B2 GEO-TRAIN-06.json TABLES_DIR OUT_DIR}"
S_SRC=2026092990
S_TR=$((S_SRC - 1000000000))
N=20

for f in "$BIN_A" "$BIN_B" "$GEOM" "$TABLES/row_weights_30d.txt"; do
  [[ -e "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir "$OUT" || { echo "OUT_DIR exists or cannot be created: $OUT" >&2; exit 2; }
mkdir "$OUT/cases" "$OUT/inputs"

# --- inputs: the 256 x 256 mask (rows 17-50, columns 131-138 set) and the technical list ------------------
awk 'BEGIN { for (r = 0; r < 256; r++) { line = ""; for (c = 0; c < 256; c++) line = line ((r >= 17 && r <= 50 && c >= 131 && c <= 138) ? "1" : "0"); print line } }' \
  > "$OUT/inputs/mask.txt"
MASK_SHA="$(sha256sum "$OUT/inputs/mask.txt" | cut -d' ' -f1)"
LIST="$OUT/inputs/geo-train-06_v0_r0_s${S_SRC}.primarios.csv"
if ! "$BIN_A" --write-primaries "$LIST" --geometry "$GEOM" --view 0 --tables "$TABLES" --seed "$S_SRC" \
     --events "$N" --case-code 6 --replica 0 > "$OUT/inputs/write.log" 2>&1; then
  echo "cannot write the technical list (see $OUT/inputs/write.log)" >&2
  exit 2
fi
LIST_SHA="$(sha256sum "$LIST" | cut -d' ' -f1)"

fails=0
checks=0
ok() { checks=$((checks + 1)); }
bad() { checks=$((checks + 1)); fails=$((fails + 1)); echo "  *** $1"; }

# case NAME WANT_EXIT WANT_CODES BIN ENV_ASSIGNMENT|- [--flag value ...]  (overrides of the default command line)
# The default command line is a complete, valid list-mode call; the trailing arguments replace flags of it.
run_case() {
  local name="$1" want_rc="$2" want_codes="$3" bin="$4" envs="$5"; shift 5
  local dir="$OUT/cases/$name"
  mkdir "$dir"
  # default arguments as name/value pairs so that an override replaces the value
  local -A a=([--primaries]="$LIST" [--primaries-sha256]="$LIST_SHA" [--roi-mask]="$OUT/inputs/mask.txt"
              [--roi-mask-sha256]="$MASK_SHA" [--transport-seed]="$S_TR" [--variant]=A [--case-code]=6
              [--replica]=0 [--region-pair]=0,1 [--muon-log]=full [--geometry]="$GEOM" [--view]=0
              [--tables]="$TABLES" [--seed]="$S_TR" [--threads]=1 [--out]="$dir")
  local drop=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--drop" ]]; then drop="$2"; shift 2; continue; fi
    a["$1"]="$2"; shift 2
  done
  local -a cmd=()
  local k
  for k in "${!a[@]}"; do
    [[ "$k" == "$drop" ]] && continue
    cmd+=("$k" "${a[$k]}")
  done
  local rc
  if [[ "$envs" == "-" ]]; then
    "$bin" "${cmd[@]}" > "$dir.out" 2> "$dir.err"; rc=$?
  else
    env "$envs" "$bin" "${cmd[@]}" > "$dir.out" 2> "$dir.err"; rc=$?
  fi
  echo "case $name: exit $rc (want $want_rc)"
  if [[ $rc -eq $want_rc ]]; then ok; else bad "$name: exit code $rc, expected $want_rc"; fi
  if [[ $want_rc -ne 0 ]]; then
    # nothing may have been written into --out (a preflight probe file is removed again)
    if [[ -z "$(ls -A "$dir")" ]]; then ok; else bad "$name: --out is not empty: $(ls -A "$dir" | tr '\n' ' ')"; fi
    if grep -q -e '^RUN DONE' -e '^OE1-ETAPA1' "$dir.out"; then bad "$name: the run started (RUN DONE / OE1-ETAPA1 on stdout)"; else ok; fi
  fi
  if [[ -n "$want_codes" ]]; then
    if grep -q "^OE1-PRERUN-CHECK .* errors=$want_codes status=FAIL\$" "$dir.out"; then ok
    else bad "$name: no 'OE1-PRERUN-CHECK ... errors=$want_codes status=FAIL' line on stdout"; fi
    if grep -q 'a check failed before BeamOn: no transport was run and no output was written (exit 4)' "$dir.err"; then ok
    else bad "$name: stderr does not say that the check failed before BeamOn"; fi
  fi
  echo "    $(grep -m1 -e '^OE1-PRERUN-CHECK' "$dir.out" | cut -c1-200)"
  echo "    $(head -n 2 "$dir.err" | cut -c1-200 | tr '\n' '|')"
}

echo "=== checks that end the run before BeamOn (exit 4)"
run_case a2-as-b   4 variant_geometry_mismatch                          "$BIN_A" -                             --variant B
run_case b2-as-a   4 variant_geometry_mismatch                          "$BIN_B" -                             --variant A
run_case a2-serial 4 run_manager_not_task                               "$BIN_A" G4RUN_MANAGER_TYPE=Serial     --variant A
run_case b2-both   4 run_manager_not_task,variant_geometry_mismatch     "$BIN_B" G4RUN_MANAGER_TYPE=Serial     --variant A

# the reasons are on stderr
grep -q -e "--variant B" -e "G4SubtractionSolid" "$OUT/cases/a2-as-b.err" && ok || bad "a2-as-b: the message does not name --variant B / G4SubtractionSolid"
grep -q -e "--variant A" -e "G4Tubs" "$OUT/cases/b2-as-a.err" && ok || bad "b2-as-a: the message does not name --variant A / G4Tubs"
grep -q "expected 'G4TaskRunManager'" "$OUT/cases/a2-serial.err" && ok || bad "a2-serial: the message does not name the expected run manager"
# G4RUN_MANAGER_TYPE=Serial makes G4RunManagerFactory build the plain sequential G4RunManager
grep -q 'run_manager=G4RunManager ' "$OUT/cases/a2-serial.out" && ok || bad "a2-serial: the run manager was not the sequential G4RunManager (G4RUN_MANAGER_TYPE ignored?)"

echo "=== order of the exit codes: earlier errors win over the pre-BeamOn checks"
# a wrong list SHA-256 (1) and a missing --region-pair (2) are found long before the checks could run
run_case order-sha  1 "" "$BIN_A" - --variant B --primaries-sha256 0000000000000000000000000000000000000000000000000000000000000000
run_case order-pair 2 "" "$BIN_A" - --variant B --drop --region-pair
# an output folder that already holds a result is refused before anything starts (preflight)
mkdir "$OUT/cases/order-exists"
: > "$OUT/cases/order-exists/geo-train-06_view0_hist.csv"
"$BIN_A" --primaries "$LIST" --primaries-sha256 "$LIST_SHA" --roi-mask "$OUT/inputs/mask.txt" \
  --roi-mask-sha256 "$MASK_SHA" --transport-seed "$S_TR" --variant B --case-code 6 --replica 0 \
  --region-pair 0,1 --geometry "$GEOM" --view 0 --tables "$TABLES" --seed "$S_TR" --threads 1 \
  --out "$OUT/cases/order-exists" > "$OUT/cases/order-exists.out" 2> "$OUT/cases/order-exists.err"
rc=$?
echo "case order-exists: exit $rc (want 1)"
if [[ $rc -eq 1 ]]; then ok; else bad "order-exists: exit code $rc, expected 1"; fi
if grep -q 'output already exists' "$OUT/cases/order-exists.err"; then ok; else bad "order-exists: no 'output already exists' message"; fi

if [[ "${OE1_EARLY_EXIT_POSITIVE:-0}" == 1 ]]; then
  echo "=== positive controls (a real transport of $N muons: local development only)"
  for spec in "a2-ok:$BIN_A:A" "b2-ok:$BIN_B:B"; do
    name="${spec%%:*}"; rest="${spec#*:}"; bin="${rest%:*}"; var="${rest##*:}"
    run_case "$name" 0 "" "$bin" - --variant "$var"
    d="$OUT/cases/$name"
    if grep -q "^OE1-PRERUN-CHECK .* variant_check=ok errors=none status=OK\$" "$d.out"; then ok; else bad "$name: no OE1-PRERUN-CHECK status=OK line"; fi
    if grep -q '^RUN DONE ' "$d.out"; then ok; else bad "$name: no RUN DONE line"; fi
    n_files="$(ls "$d" | grep -c -e '_hist.csv$' -e '.meta.json$' -e '_muones.csv$')"
    if [[ "$n_files" == 3 ]]; then ok; else bad "$name: expected 3 output files, found $n_files"; fi
    # the pre-BeamOn line comes before the run
    first="$(grep -n -m1 -e '^OE1-PRERUN-CHECK' "$d.out" | cut -d: -f1)"
    done_line="$(grep -n -m1 -e '^RUN DONE ' "$d.out" | cut -d: -f1)"
    if [[ -n "$first" && -n "$done_line" && "$first" -lt "$done_line" ]]; then ok; else bad "$name: OE1-PRERUN-CHECK is not before RUN DONE"; fi
  done
else
  echo "(positive controls skipped: set OE1_EARLY_EXIT_POSITIVE=1 for a local run that transports $N muons)"
fi

echo "early_exit: $checks checks, $fails failed"
if [[ $fails -eq 0 ]]; then echo "PASS early_exit"; exit 0; fi
echo "FAIL early_exit"
exit 1
