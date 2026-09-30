#!/bin/bash
#SBATCH --job-name=oe1-e1-tecnicas
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/%x-%j.out
#
# ============================================================================================
#  DV1 GATE. This script REFUSES TO START (exit code 4, nothing created) unless
#      ~/oe1-trabajo/etapa1/herramientas/DV1-aprobado.txt
#  exists, is not empty and contains the token DV1-APROBADO, which the operator writes once DV1 is decided
#  (README.md, step 9). These technical tests are transports OUTSIDE the 13 runs of the protocol:
#  24 120 muons in all, 1.86 % of the stage's 1.3e6 (13 runs x 1e5). Without them PROT:17 (logging consumes
#  no random numbers) is only verified statically. The gate file and its SHA-256 are copied into the output
#  folder, and every output is labelled "prueba tecnica".
#
#  Transports, per test (1 thread each):
#      T-L   8 000   4 runs x 2 000: pilot A, A2, pilot B, B2, legacy mode
#      T-R   2 000   B2 list mode, --muon-log min (the `full` side is the g06-B run of T-F)
#      T-M   2 000   B2 list mode, another master seed
#      T-P   2 000   B2 list mode, list with its first 100 rows replaced
#      T-S     120   60 synthetic mu- x 2 binaries (A2, B2)
#      T-F  10 000   5 runs x 2 000 through the production run_one
#      total 24 120  (about 1.86 % of 1.3e6)
# ============================================================================================
#
# One process = one thread (`--threads 1`), six at a time (`xargs -P 6`). Technical lists have 2000
# muons, source seed 2026092990 (2026092991 for the auxiliary list), replica 0, so they never coincide
# with the production lists.
#
# Disk: it refuses to start with less than 400 MB free in /home (exit code 5). Inside a SLURM job
# DRYRUN and STAGE1_SKIP_MODULES are ignored (env_stage1.sh). There is no dry run here: started outside a SLURM
# job (SLURM_JOB_ID empty) and outside a rehearsal tree (marker .arbol-de-prueba) it refuses to start, exit code 2,
# before anything else, whatever DRYRUN says.
#
#   T-L  legacy inertness: pilot binaries ~/oe1-trabajo/build-A and build-B (executed read-only) against
#        A2 and B2 in legacy mode: geo-train-06 view 0, `--events 2000 --seed 2026092900 --threads 1`.
#        hist.csv byte-identical; meta identical apart from wall_seconds / seconds_per_event.
#   T-R  B2 list mode on the technical geo-train-06 list, `--muon-log full` against `min`: identical
#        hist.csv and identical rng1 for every id (PROT:17).
#   T-M  as T-R full but with another MASTER `--seed` and the SAME `--transport-seed`: rng0, rng1, the
#        histogram and the whole log identical (no dependence on the master engine).
#   T-S  the 60 synthetic mu- of 10 TeV in A2 and B2 (classification, length within 1 %, lz_a0 and
#        lz_rock <= 1 um in B2, lz_a0 + lz_a1 = lz in A2).
#   T-F  functional end-to-end rehearsal: the technical lists of both cases in A2 and B2 plus an A/A
#        repeat, launched with the SAME run_one as principal.sbatch, then the SAME analysis script
#        (--replicas 0). STOPPED = OTHER = ABORTED = 0 is asserted.
#   T-P  independence from the previous muons: the first 100 rows of the geo-train-06 list are replaced
#        by rows of the auxiliary list; ids >= 100 must keep rng1 and the whole outcome.
#
#     cd ~/oe1-trabajo/etapa1 && sbatch --parsable herramientas/pruebas_tecnicas.sbatch
#
# Exit codes: 0 only if every test passes; 1 a test failed; 2 a required file is missing or it was started outside
# a SLURM job; 3 the TEC_NN folder exists; 4 the DV1 gate is closed; 5 less than 400 MB free in /home.
# TEC_NN: edit in a COPY per attempt (folders are never reused).
TEC_NN="01"
K_REPLACED=100
MIN_FREE_TEC_MB=400      # free space in /home below which this job refuses to start (exit code 5)

E1="$HOME/oe1-trabajo/etapa1"
H="$E1/herramientas"
# run_one and its helpers are those of principal.sbatch (sourcing it does not run its main)
# shellcheck disable=SC1091
source "$H/principal.sbatch"
# shellcheck disable=SC1091
source "$H/env_stage1.sh" || exit 2
set -u -o pipefail
# 24 120 transports belong in a SLURM job: refuse before the gate and before anything is created
stage1_require_job_or_rehearsal pruebas_tecnicas.sbatch || exit 2

# --- the DV1 gate: the gate file must be on disk before anything else happens ---------------------------------
DV1_FILE="$H/DV1-aprobado.txt"
if [[ ! -s "$DV1_FILE" ]] || ! grep -q 'DV1-APROBADO' "$DV1_FILE"; then
  {
    echo "DV1: $DV1_FILE is missing, empty or does not contain the token DV1-APROBADO."
    echo "These technical tests are 24 120 transports outside the 13 runs of the protocol (DV1)."
    echo "Once DV1 is decided, write the token, the date and a reference:"
    echo "  printf 'DV1-APROBADO %s %s\\n' \"<YYYY-MM-DD>\" \"<reference>\" > $DV1_FILE"
    echo "Nothing was started or created."
  } >&2
  exit 4
fi
FREE_MB="$(stage1_free_mb_home)" || FREE_MB=""
if [[ -z "$FREE_MB" ]] || (( FREE_MB < MIN_FREE_TEC_MB )); then
  echo "only ${FREE_MB:-an unreadable amount of} MB free in /home (quota -s); need >= $MIN_FREE_TEC_MB MB. Nothing was started." >&2
  exit 5
fi

TABLES="$HOME/oe1-trabajo/tablas"
CAT="$HOME/oe1-trabajo/catalogo"
BIN_A="$E1/build/A2/muography_sim"
BIN_B="$E1/build/B2/muography_sim"
PILOT_A="$HOME/oe1-trabajo/build-A/muography_sim"
PILOT_B="$HOME/oe1-trabajo/build-B/muography_sim"
ROI="$E1/analisis/roi"
TECH="$E1/pruebas/tecnicas-$TEC_NN"
LIS="$TECH/listas"
RUNS="$TECH/corridas"
REP="$TECH/resultados"
TF="$TECH/tf"
export OE1_CROSSING_REFERENCE="$E1/fuentes/A2/simulation/geant4_app/tests/crossing_reference.py"
S_TEC=2026092990
S_AUX=2026092991
S_TR=$((S_TEC - 1000000000))
S_TR_SHIFTED=$((S_TR + 5))

FAILS=()
fail() { FAILS+=("$1"); echo "FALLO: $1" >&2; }

for f in "$BIN_A" "$BIN_B" "$PILOT_A" "$PILOT_B" "$H/oe1_stage1_tecnicas.py" "$H/oe1_stage1_lists.py" \
         "$H/oe1_stage1_synthetic.py" "$H/oe1_stage1_analysis.py" "$H/make_manifest.py" "$H/run_timed.py" \
         "$OE1_CROSSING_REFERENCE" "$ROI/R_int.mask.txt" "$ROI/R_int.mask.txt.sha256" \
         "$ROI/R_int-congelada.json" "$ROI/roi-mascaras.npz"; do
  [[ -e "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir "$TECH" || { echo "$TECH already exists: use a new TEC_NN" >&2; exit 3; }
mkdir "$LIS" "$RUNS" "$REP" "$TECH/jobs" "$TECH/casos"
for d in tl tr tm tp ts; do mkdir "$RUNS/$d"; done
stage1_write_entorno "$TECH/entorno.txt"
# the gate file travels with the evidence
cp -p "$DV1_FILE" "$TECH/DV1-aprobado.txt"
( cd "$TECH" && sha256sum DV1-aprobado.txt > DV1-aprobado.txt.sha256 )
{
  echo "prueba tecnica (DV1), fuera de las 13 corridas; trabajo ${SLURM_JOB_ID:-manual}"
  echo "transportes: T-L 8000, T-R 2000, T-M 2000, T-P 2000, T-S 120, T-F 10000 = 24120 (1,86 % de 1,3e6)"
  echo "DV1-aprobado.txt sha256=$(cut -d' ' -f1 "$TECH/DV1-aprobado.txt.sha256")"
  echo "libre_en_home_mb=$FREE_MB (minimo exigido $MIN_FREE_TEC_MB)"
  echo "interruptores_ignorados_en_trabajo=${STAGE1_IGNORED_SWITCHES:-ninguno}"
} > "$TECH/ETIQUETA.txt"

# ------------------------------------------------------------------------------------------------
# Stage 0: the technical lists (serial, seconds)
# ------------------------------------------------------------------------------------------------
echo "=== technical lists"
mkdir "$LIS/B2-control"
: > "$LIS/generacion.txt"
gen() {   # gen BIN OUTDIR CASO CODE SEED
  local bin="$1" outdir="$2" caso="$3" code="$4" seed="$5"
  local name="${caso}_v0_r0_s${seed}.primarios.csv"
  local -a cmd=("$bin" --write-primaries "$outdir/$name" --geometry "$CAT/$caso.json" --view 0 --tables "$TABLES"
                --seed "$seed" --events 2000 --case-code "$code" --replica 0)
  { printf '$ '; printf '%q ' "${cmd[@]}"; printf '\n'; } >> "$LIS/generacion.txt"
  "${cmd[@]}" >> "$LIS/generacion.txt" 2>&1 || fail "write-primaries $name (exit $?)"
}
L06="$LIS/geo-train-06_v0_r0_s${S_TEC}.primarios.csv"
L01="$LIS/geo-train-01_v0_r0_s${S_TEC}.primarios.csv"
LAUX="$LIS/geo-train-06_v0_r0_s${S_AUX}.primarios.csv"
gen "$BIN_A" "$LIS" geo-train-06 6 "$S_TEC"
gen "$BIN_A" "$LIS" geo-train-01 1 "$S_TEC"
gen "$BIN_A" "$LIS" geo-train-06 6 "$S_AUX"
gen "$BIN_B" "$LIS/B2-control" geo-train-06 6 "$S_TEC"
cmp -s "$L06" "$LIS/B2-control/$(basename "$L06")" || fail "the technical list written by B2 differs from A2's"
for l in "$L06" "$L01" "$LAUX"; do
  "$BIN_A" --check-primaries "$l" >> "$LIS/check-A2.txt" 2>&1 || fail "check-primaries A2 $l"
  "$BIN_B" --check-primaries "$l" >> "$LIS/check-B2.txt" 2>&1 || fail "check-primaries B2 $l"
done
cmp -s "$LIS/check-A2.txt" "$LIS/check-B2.txt" || fail "--check-primaries differs between A2 and B2"
python3 "$H/oe1_stage1_lists.py" derive --base "$L06" --other "$LAUX" --replace-first "$K_REPLACED" \
  --out "$LIS/geo-train-06_v0_r0_s${S_TEC}-derivada.primarios.csv" > "$LIS/derivada.txt" 2>&1 || fail "cannot derive the T-P list"
LDER="$LIS/geo-train-06_v0_r0_s${S_TEC}-derivada.primarios.csv"
python3 "$H/oe1_stage1_synthetic.py" generate --out-dir "$LIS/sintetico" --catalog "$CAT/geo-train-06.json" > "$LIS/sintetico.txt" 2>&1 || fail "cannot generate the synthetic list"
LSYN="$LIS/sintetico/geo-train-06_v0_r9_sintetico.primarios.csv"
for l in "$LDER" "$LSYN"; do
  "$BIN_A" --check-primaries "$l" > "$l.check-A2.txt" 2>&1 || fail "check-primaries A2 $l"
  "$BIN_B" --check-primaries "$l" > "$l.check-B2.txt" 2>&1 || fail "check-primaries B2 $l"
  cmp -s "$l.check-A2.txt" "$l.check-B2.txt" || fail "--check-primaries differs between A2 and B2 for $l"
done
python3 "$H/oe1_stage1_lists.py" validate "$L06" "$L01" "$LAUX" --tables "$TABLES" --catalog-dir "$CAT" --no-stats \
  --out "$REP/validacion-listas-tecnicas.json" > "$REP/validacion-listas-tecnicas.txt" 2>&1 || fail "the technical lists do not validate"
[[ ${#FAILS[@]} -eq 0 ]] || { echo "stopping: the lists are not usable" >&2; printf '%s\n' "${FAILS[@]}" > "$REP/FALLOS.txt"; exit 1; }

# ------------------------------------------------------------------------------------------------
# Stage 1: launch (T-F through the production run_one; the rest through run_case)
# ------------------------------------------------------------------------------------------------
MASK="$ROI/R_int.mask.txt"
MASK_SHA="$(cut -d' ' -f1 "$ROI/R_int.mask.txt.sha256")"
sha_of() { sha256sum "$1" | cut -d' ' -f1; }

mkdir "$TF" "$TF/estado"
python3 "$H/make_manifest.py" --technical --lists-dir "$LIS" --catalog-dir "$CAT" --tables-dir "$TABLES" \
  --bin-a "$BIN_A" --bin-b "$BIN_B" --roi-mask "$MASK" --out "$TF/manifiesto-corridas.tsv" \
  --inputs-sha256 "$TF/entradas-esperadas.sha256" > "$TECH/manifiesto-tecnico.txt" 2>&1 || { echo "cannot build the technical manifest" >&2; exit 1; }
ATT="$TF"; STATUSDIR="$TF/estado"
export ATT STATUSDIR TABLES MASK MASK_SHA RUN_TIMED="$H/run_timed.py" DRYRUN=0
export MANIFEST="$TF/manifiesto-corridas.tsv" EXPECTED_SHA="$TF/entradas-esperadas.sha256"
while IFS= read -r out; do mkdir -p "$TF/$(dirname "$out")"; done < <(tail -n +2 "$MANIFEST" | cut -f12)

run_case() {   # run_case NAME : executes $TECH/jobs/NAME.cmd (a printf %q line) with --out $TECH/casos/NAME-dir
  set -u -o pipefail
  : "${TECH:?}" "${RUN_TIMED:?}"
  local name="$1" dir rc
  dir="$(cut -d' ' -f1 "$TECH/jobs/$name.dir")"
  mkdir "$dir" 2>/dev/null || { printf '%s\t-1\tno se pudo crear la carpeta\n' "$name" > "$TECH/casos/$name.rc"; return 0; }
  local -a args
  eval "args=($(cat "$TECH/jobs/$name.cmd"))"
  cp "$TECH/jobs/$name.cmd" "$dir/comando.txt"
  python3 "$RUN_TIMED" "$dir/recursos.json" -- "${args[@]}" --out "$dir" > "$dir/run.log" 2>&1
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(hostname)" > "$TECH/casos/$name.rc"
  return 0
}
export -f run_case
export TECH

add_case() {   # add_case NAME DIR CMD... (the command WITHOUT --out)
  local name="$1" dir="$2"; shift 2
  printf '%s\n' "$dir" > "$TECH/jobs/$name.dir"
  printf '%q ' "$@" > "$TECH/jobs/$name.cmd"
  echo "C:$name" >> "$TECH/jobs/lista.txt"
}
: > "$TECH/jobs/lista.txt"
legacy=(--geometry "$CAT/geo-train-06.json" --view 0 --tables "$TABLES" --events 2000 --seed 2026092900 --threads 1)
add_case tl-pilot-A "$RUNS/tl/pilot-A" "$PILOT_A" "${legacy[@]}"
add_case tl-A2      "$RUNS/tl/A2"      "$BIN_A"   "${legacy[@]}"
add_case tl-pilot-B "$RUNS/tl/pilot-B" "$PILOT_B" "${legacy[@]}"
add_case tl-B2      "$RUNS/tl/B2"      "$BIN_B"   "${legacy[@]}"
list_args() {   # list_args LIST VARIANT CODE REPLICA LOGLEVEL MASTER_SEED TRANSPORT_SEED
  # Every list handled here is a geo-train-06 list, so the region pair is explicit: 0,1 (conduit, dense
  # body); without --region-pair the binary exits 2 for a geometry with two anomalies (plan W2). The
  # geo-train-01 runs of T-F go through run_one, which passes `--region-pair none`.
  printf '%s\n' --primaries "$1" --primaries-sha256 "$(sha_of "$1")" --roi-mask "$MASK" --roi-mask-sha256 "$MASK_SHA" \
    --transport-seed "$7" --variant "$2" --case-code "$3" --replica "$4" --region-pair 0,1 --muon-log "$5" \
    --geometry "$CAT/geo-train-06.json" --view 0 --tables "$TABLES" --seed "$6" --threads 1
}
mapfile -t a_min < <(list_args "$L06" B 6 0 min "$S_TR" "$S_TR")
add_case tr-min "$RUNS/tr/min" "$BIN_B" "${a_min[@]}"
mapfile -t a_shift < <(list_args "$L06" B 6 0 full "$S_TR_SHIFTED" "$S_TR")
add_case tm-shifted "$RUNS/tm/shifted" "$BIN_B" "${a_shift[@]}"
mapfile -t a_der < <(list_args "$LDER" B 6 0 full "$S_TR" "$S_TR")
add_case tp-derivada "$RUNS/tp/derivada" "$BIN_B" "${a_der[@]}"
mapfile -t a_sa < <(list_args "$LSYN" A 6 9 full "$S_TR" "$S_TR")
add_case ts-A2 "$RUNS/ts/A2" "$BIN_A" "${a_sa[@]}"
mapfile -t a_sb < <(list_args "$LSYN" B 6 9 full "$S_TR" "$S_TR")
add_case ts-B2 "$RUNS/ts/B2" "$BIN_B" "${a_sb[@]}"
for id in $(tail -n +2 "$MANIFEST" | cut -f1); do echo "F:$id" >> "$TECH/jobs/lista.txt"; done

echo "=== launching $(wc -l < "$TECH/jobs/lista.txt") technical runs, 6 at a time"
export -f run_one warning_codes check_meta
xargs -P 6 -I{} bash -c 'case "$1" in F:*) run_one "${1#F:}" ;; C:*) run_case "${1#C:}" ;; esac' _ {} < "$TECH/jobs/lista.txt"
xrc=$?
[[ $xrc -eq 0 ]] || fail "xargs exited with $xrc"

# ------------------------------------------------------------------------------------------------
# Stage 2: evaluation
# ------------------------------------------------------------------------------------------------
echo "=== evaluation"
T="$H/oe1_stage1_tecnicas.py"
for name in tl-pilot-A tl-A2 tl-pilot-B tl-B2 tr-min tm-shifted tp-derivada ts-A2 ts-B2; do
  [[ -f "$TECH/casos/$name.rc" && "$(cut -f2 "$TECH/casos/$name.rc")" == 0 ]] || fail "the run $name did not finish with exit code 0"
done
for f in "$STATUSDIR"/*.status; do
  [[ "$(cut -f2 "$f")" == OK ]] || fail "T-F run $(basename "$f" .status): $(cut -f2,4 "$f")"
done
python3 "$T" t-l --pilot "$RUNS/tl/pilot-A" --new "$RUNS/tl/A2" --label A --out "$REP/T-L-A.json" || fail "T-L (A)"
python3 "$T" t-l --pilot "$RUNS/tl/pilot-B" --new "$RUNS/tl/B2" --label B --out "$REP/T-L-B.json" || fail "T-L (B)"
FULL="$TF/geo-train-06/B/$S_TEC"
python3 "$T" t-r --full "$FULL" --min "$RUNS/tr/min" --out "$REP/T-R.json" || fail "T-R"
python3 "$T" t-m --ref "$FULL" --shifted "$RUNS/tm/shifted" --out "$REP/T-M.json" || fail "T-M"
python3 "$T" t-p --base "$FULL" --derived "$RUNS/tp/derivada" --k "$K_REPLACED" --out "$REP/T-P.json" || fail "T-P"
python3 "$H/oe1_stage1_synthetic.py" check --expected "$LIS/sintetico/sintetico-esperado.json" \
  --muones "$RUNS/ts/A2/geo-train-06_view0_muones.csv" --variant A --out "$REP/T-S-A.json" || fail "T-S (A2)"
python3 "$H/oe1_stage1_synthetic.py" check --expected "$LIS/sintetico/sintetico-esperado.json" \
  --muones "$RUNS/ts/B2/geo-train-06_view0_muones.csv" --variant B --out "$REP/T-S-B.json" || fail "T-S (B2)"
python3 "$H/oe1_stage1_analysis.py" --attempt "$TF" --roi-mask "$MASK" --roi-frozen "$ROI/R_int-congelada.json" \
  --roi-npz "$ROI/roi-mascaras.npz" --lists-dir "$LIS" --replicas 0 --no-figures --out "$REP/analisis-TF" \
  > "$REP/analisis-TF.txt" 2>&1
rc_tf=$?
[[ $rc_tf -eq 0 ]] || fail "T-F: the end-to-end analysis failed with exit code $rc_tf (3 = blocking contract, 2 = unusable input, other = crash); see analisis-TF.txt"
python3 "$T" t-f --analysis-dir "$REP/analisis-TF" --expected-runs 5 --out "$REP/T-F.json" || fail "T-F"
python3 "$T" resumen "$REP"/T-*.json --out "$REP/resumen-pruebas-tecnicas.json" || fail "the summary reports failures"
{
  echo "fecha_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "trabajo=${SLURM_JOB_ID:-manual}"
  echo "nodo=$(hostname)"
  echo "TEC_NN=$TEC_NN"
  echo "fallos=${#FAILS[@]}"
  for m in "${FAILS[@]}"; do echo "FALLO: $m"; done
  echo "resultado=$([[ ${#FAILS[@]} -eq 0 ]] && echo OK || echo FALLO)"
} | tee "$REP/estado-pruebas-tecnicas.txt"
[[ ${#FAILS[@]} -eq 0 ]]
