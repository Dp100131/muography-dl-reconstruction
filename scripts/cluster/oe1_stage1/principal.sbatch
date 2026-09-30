#!/bin/bash
#SBATCH --job-name=oe1-etapa1
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
#
# OE1 stage 1, PRODUCTION: the 13 single-thread runs in ONE job on ONE node.
#
#   g06-{A,B}-r{1,2,3}   geo-train-06 view 0, variants A (pilot geometry) and B (corrected "B1")
#   g01-{A,B}-r{1,2,3}   geo-train-01 view 0 (control)
#   g01-A-r1-rep         exact A/A repeat of g01-A-r1 (same list, same s_tr; excluded from Delta)
#
# One job because QoS `normal` allows 4 submitted jobs (MaxSubmitPU=4; each array task counts as a job)
# and because A/A and every A/B pair must share node, CPU and libm. `xargs -P 13` starts the 13
# `run_one` processes at once (--cpus-per-task=13); each is a single-thread `muography_sim` in list mode
# with an explicit region pair: `--region-pair 0,1` on geo-train-06, `--region-pair none` on geo-train-01.
#
# HOW TO SUBMIT (only after the gates of README.md: K0-K2, K-ROI, the DV1 technical job when it runs,
# the freeze commit, and `DRYRUN=1` printing the 13 commands):
#     cd ~/oe1-trabajo/etapa1 && sbatch --parsable herramientas/principal.sbatch
# (the relative --output needs ~/oe1-trabajo/etapa1/slurm to exist and the job to be submitted from there)
#
# ATTEMPT NUMBER: edit INTENTO below in a COPY of this file per attempt; the copy that runs is archived
# next to the runs. Attempt folders are created with a plain mkdir and are never reused or overwritten.
#
# DRYRUN=1 bash principal.sbatch    prints the 13 commands, checks every fixed input and prints the free space in
#                                   /home; writes nothing. Only outside SLURM: inside a job (SLURM_JOB_ID set)
#                                   DRYRUN and STAGE1_SKIP_MODULES are IGNORED and said so (env_stage1.sh), because
#                                   sbatch exports the submitting shell's environment.
# bash principal.sbatch             without DRYRUN, outside a SLURM job (SLURM_JOB_ID empty) and outside a rehearsal
#                                   tree (env_stage1.sh: marker .arbol-de-prueba) it REFUSES to start, exit code 2,
#                                   before any folder is created: the real run is submitted with sbatch.
#
# Disk: the real run needs >= 800 MB free in /home (`quota -s`; the 13 runs peak at about 0.5 GB and post_run.sh
# adds the compressed copies, about 0.95 GB in all) and stops with exit code 2 below that. See README.md: with
# about 1.5 GB free only ONE production attempt fits; free etapa1-dev before submitting.
#
# Lessons of job 30465 (the STATUSDIR bug) built in:
#   * EVERY variable run_one needs is exported, and run_one starts with `: "${VAR:?}"` guards;
#   * `set -u -o pipefail` inside the child (it does not inherit them through bash -c);
#   * the script exits non-zero if xargs fails, if there are fewer than 13 status files or if any
#     status is not OK (SLURM alone would report COMPLETED 0:0).
#
# Warnings (`*** G4Exception` codes): B must have none (a FAIL otherwise). For A the expected code
# (GeomVol1002=1 on geo-train-06) is a historical diagnostic: a different count is recorded as evidence
# (avisos.txt of the run, a note in the status file, K6 non-blocking) but is NOT a failure of the run.
#
# Exit codes: 0 all 13 runs OK; 1 a run failed; 2 pre-flight failure (missing input, fixed input changed, less
# than 800 MB free, a real run started outside a SLURM job); 3 the attempt folder exists.

INTENTO="01"

E1="$HOME/oe1-trabajo/etapa1"
H="$E1/herramientas"
ATT="$E1/corridas/intento-$INTENTO"
STATUSDIR="$ATT/estado"
EXPECTED_RUNS=13
CATALOG_GEOMETRIES="geo-train-06 geo-train-01"
MIN_FREE_MB=800          # free space in /home below which the real run refuses to start (exit code 2)

# ----------------------------------------------------------------------------------------------
# helpers (defined before anything runs so that pruebas_tecnicas.sbatch can `source` this file and
# reuse the very same run_one for its end-to-end rehearsal T-F)
# ----------------------------------------------------------------------------------------------
# "GeomVol1002=1,GeomNav0003=2" or "ninguno" from the *** G4Exception : <code> lines of a log.
warning_codes() {
  local codes
  codes="$(grep -o '^\*\*\* G4Exception : [^ ]*' "$1" 2>/dev/null | awk '{c[$4]++} END {for (k in c) printf "%s=%d\n", k, c[k]}' | LC_ALL=C sort | paste -sd, -)"
  if [[ -z "$codes" ]]; then echo ninguno; else echo "$codes"; fi
}

# Structural checks of the meta written by the binary (prints "ok" or the failing fields).
#   check_meta META WANT_REGION N BIN_SHA GEOM_SHA WEIGHTS_SHA CDF_SHA TABLES_META_SHA
# The five SHA-256 values are those that run_one computed from the files just before the launch; the binary
# records its own (/proc/self/exe) and those of the geometry JSON and of the three tables in oe1_etapa1
# (plan S2), so a swap or a stale copy between the hashing and the execution shows here.
check_meta() {
  python3 - "$@" <<'PY'
import json, sys
meta_path, want_region, n = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3])
bin_sha, geom_sha, weights_sha, cdf_sha, tmeta_sha = sys.argv[4:9]
with open(meta_path) as fh:
    m = json.load(fh)
oe = m.get("oe1_etapa1")
if not isinstance(oe, dict):
    print("sin clave oe1_etapa1")
    sys.exit(1)
bad = []
if bool(oe.get("region", {}).get("applicable")) != want_region:
    bad.append("region.applicable")
if oe.get("errors") != []:
    bad.append("errors")
chk = oe.get("checks", {})
for k in ("ids_complete_unique", "hist_reconstructed_equal", "detected_equal", "events_equal",
          "generated_equal", "writes_ok"):
    if chk.get(k) is not True:
        bad.append("checks." + k)
# --variant against the geometry that was built: checked (true) whenever a region pair was given
if want_region and chk.get("variant_matches_geometry") is not True:
    bad.append("checks.variant_matches_geometry")
if oe.get("muon_log", {}).get("rows") != n:
    bad.append("muon_log.rows")
if m.get("events_transported") != n:
    bad.append("events_transported")
if oe.get("counts", {}).get("status", {}).get("ABORTED") != 0:
    bad.append("ABORTED")
# provenance recorded by the binary against the hashes of the launch
if oe.get("binary_sha256") != bin_sha:
    bad.append("binary_sha256")
inp = oe.get("inputs", {})
if inp.get("geometry_json", {}).get("sha256") != geom_sha:
    bad.append("inputs.geometry_json")
tables = inp.get("tables", {})
for name, sha in (("row_weights_30d.txt", weights_sha), ("p_cdf.txt", cdf_sha), ("meta.json", tmeta_sha)):
    if tables.get(name) != sha:
        bad.append("inputs.tables." + name)
# the getrusage figures of the process (plan W3) are written by the corrected binary
for k in ("rusage_user_cpu_seconds", "rusage_system_cpu_seconds", "rusage_max_rss_kb"):
    if not isinstance(m.get(k), (int, float)) or isinstance(m.get(k), bool):
        bad.append(k)
print(",".join(bad) if bad else "ok")
sys.exit(1 if bad else 0)
PY
}

# One run of the manifest. Never called with an id that is not in $MANIFEST.
# Writes $STATUSDIR/<id>.status: id, verdict, rc, reason, host, job, start_utc, end_utc.
run_one() {
  set -u -o pipefail
  : "${ATT:?ATT not exported}" "${MANIFEST:?MANIFEST not exported}" "${STATUSDIR:?STATUSDIR not exported}" \
    "${TABLES:?TABLES not exported}" "${MASK:?MASK not exported}" "${MASK_SHA:?MASK_SHA not exported}" \
    "${EXPECTED_SHA:?EXPECTED_SHA not exported}" "${RUN_TIMED:?RUN_TIMED not exported}" "${DRYRUN:?DRYRUN not exported}"
  local id="$1"
  local st="$STATUSDIR/$id.status"
  local line run_id caso geom var bin ssrc str ccode list lsha n out rol avisos
  line="$(awk -F'\t' -v k="$id" 'NR > 1 && $1 == k { print; exit }' "$MANIFEST")"
  if [[ -z "$line" ]]; then
    if [[ "$DRYRUN" == 1 ]]; then echo "DRYRUN $id: NOT IN THE MANIFEST"; return 0; fi
    printf '%s\tFAIL\t-\tno esta en el manifiesto\t%s\t%s\t-\t-\n' "$id" "$(hostname)" "${SLURM_JOB_ID:-manual}" > "$st"
    return 0
  fi
  IFS=$'\t' read -r run_id caso geom var bin ssrc str ccode list lsha n out rol avisos <<< "$line"
  local rep=""
  if [[ "$run_id" =~ -r([0-9]+) ]]; then rep="${BASH_REMATCH[1]}"; fi
  local dir="$ATT/$out"
  # The region pair is always explicit (plan W2): the binary exits 2 when a geometry with two or more
  # anomalies gets no --region-pair. geo-train-06: conduit 0 and dense body 1; geo-train-01 (no anomalies,
  # so no region): `none`, on purpose.
  local -a region=(--region-pair none)
  if [[ "$caso" == geo-train-06 ]]; then region=(--region-pair 0,1); fi
  # list mode: --seed is the master seed (= s_tr), --transport-seed = s_tr; never --events or --days
  local -a cmd=("$bin" --primaries "$list" --primaries-sha256 "$lsha" --roi-mask "$MASK" --roi-mask-sha256 "$MASK_SHA"
    --transport-seed "$str" --variant "$var" --case-code "$ccode" --replica "$rep"
    ${region[@]+"${region[@]}"} --muon-log full --geometry "$geom" --view 0 --tables "$TABLES" --seed "$str"
    --threads 1 --out "$dir")
  # inputs of THIS run, verified against the fixed list and against the manifest (also in DRYRUN)
  local inputs in_problem=""
  inputs="$(sha256sum "$bin" "$geom" "$list" "$list.meta.json" "$MASK" "$TABLES/row_weights_30d.txt" \
    "$TABLES/p_cdf.txt" "$TABLES/meta.json" 2>&1)" || in_problem+="sha256sum de las entradas;"
  if [[ -n "$(grep -Fxv -f "$EXPECTED_SHA" <<< "$inputs")" ]]; then
    in_problem+="entradas distintas de las congeladas;"
  fi
  if [[ "$(awk -v p="$list" '$2 == p { print $1 }' <<< "$inputs")" != "$lsha" ]]; then
    in_problem+="sha256 de la lista distinto del manifiesto;"
  fi
  if [[ "$DRYRUN" == 1 ]]; then
    printf 'DRYRUN %s [inputs=%s]: ' "$id" "${in_problem:-ok}"; printf '%q ' "${cmd[@]}"; printf '\n'
    return 0
  fi

  local t0 t1 rc=0 verdict=OK reason=""
  t0="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if ! mkdir "$dir" 2>/dev/null; then    # leaf folder: plain mkdir, the parents were created by main
    printf '%s\tFAIL\t-\tno se pudo crear %s (ya existe?)\t%s\t%s\t%s\t-\n' "$id" "$out" "$(hostname)" "${SLURM_JOB_ID:-manual}" "$t0" > "$st"
    return 0
  fi
  printf '%s\n' "$inputs" > "$dir/entradas.sha256"
  if [[ -n "$in_problem" ]]; then verdict=FAIL; reason+="$in_problem"; fi
  { printf '%q ' "${cmd[@]}"; printf '\n'; } > "$dir/comando.txt"
  if [[ "$verdict" == OK ]]; then
    python3 "$RUN_TIMED" "$dir/recursos.json" -- "${cmd[@]}" > "$dir/run.log" 2>&1
    rc=$?
  else
    rc=-1
  fi
  t1="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  # verdict of the run
  if [[ "$verdict" == OK ]]; then
    if [[ $rc -ne 0 ]]; then verdict=FAIL; reason+="codigo de salida $rc;"; fi
    grep -q '^RUN DONE ' "$dir/run.log" || { verdict=FAIL; reason+="falta RUN DONE;"; }
    local oe
    oe="$(grep '^OE1-ETAPA1 ' "$dir/run.log" | tail -n 1)"
    if [[ -z "$oe" || "$oe" != *" status=OK"* || "$oe" != *" errors=none "* ]]; then verdict=FAIL; reason+="linea OE1-ETAPA1 ausente o no OK;"; fi
    local -a metas=("$dir"/*.meta.json) hists=("$dir"/*_hist.csv) logs=("$dir"/*_muones.csv)
    if [[ ${#metas[@]} -ne 1 || ! -s "${metas[0]}" || ${#hists[@]} -ne 1 || ! -s "${hists[0]}" || ${#logs[@]} -ne 1 || ! -s "${logs[0]}" ]]; then
      verdict=FAIL; reason+="faltan meta, hist o _muones.csv;"
    else
      local want_region=0
      if [[ "$caso" == geo-train-06 ]]; then want_region=1; fi
      local why
      sha_of_input() { awk -v p="$1" '$2 == p { print $1; exit }' <<< "$inputs"; }
      why="$(check_meta "${metas[0]}" "$want_region" "$n" "$(sha_of_input "$bin")" "$(sha_of_input "$geom")" \
        "$(sha_of_input "$TABLES/row_weights_30d.txt")" "$(sha_of_input "$TABLES/p_cdf.txt")" \
        "$(sha_of_input "$TABLES/meta.json")")" || { verdict=FAIL; reason+="meta: $why;"; }
      local rows=$(( $(wc -l < "${logs[0]}") - 1 ))
      if [[ $rows -ne $n ]]; then verdict=FAIL; reason+="filas del registro $rows != N=$n;"; fi
    fi
    if compgen -G "$dir/*.part" > /dev/null; then verdict=FAIL; reason+="archivo .part sin renombrar;"; fi
    local got want
    got="$(warning_codes "$dir/run.log")"
    want="${avisos:-ninguno}"
    # B must show no geometry warning at all: a different set is fatal. For A the expected code is a historical
    # diagnostic (the overlap of the pilot's geometry), so a different one is recorded as evidence and does not
    # fail the run (K6 of the analysis reports it, non-blocking).
    {
      printf 'variante=%s\n' "$var"
      printf 'obtenidos=%s\n' "$got"
      printf 'esperados=%s\n' "$want"
      printf 'coinciden=%s\n' "$([[ "$got" == "$want" ]] && echo si || echo no)"
    } > "$dir/avisos.txt"
    if [[ "$got" != "$want" ]]; then
      if [[ "$var" == B ]]; then
        verdict=FAIL; reason+="avisos $got != esperados $want;"
      else
        reason+="nota: avisos de A $got != esperados $want (solo evidencia, ver avisos.txt);"
      fi
    fi
  fi
  # explicit-file SHA256SUMS (also for failed runs: they are evidence)
  ( cd "$dir" && sha256sum -- entradas.sha256 comando.txt $(ls run.log recursos.json avisos.txt 2>/dev/null) \
      $(ls -- *_hist.csv *.meta.json *_muones.csv 2>/dev/null) > SHA256SUMS ) 2>/dev/null || true
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$verdict" "$rc" "$reason" "$(hostname)" \
    "${SLURM_JOB_ID:-manual}" "$t0" "$t1" > "$st"
  return 0
}
export -f run_one warning_codes check_meta

# ----------------------------------------------------------------------------------------------
main() {
  set -u -o pipefail
  # env_stage1.sh first: inside a SLURM job it unsets DRYRUN and STAGE1_SKIP_MODULES (a job always runs for real),
  # so DRYRUN must be read AFTER it. It also unsets the five G4 override variables and records their values.
  # shellcheck disable=SC1091
  source "$H/env_stage1.sh" || return 2
  set -u
  local dryrun="${DRYRUN:-0}"
  # A real run starts 13 transports: only inside a SLURM job (or a rehearsal tree). Refuse before anything is created.
  stage1_require_job_or_rehearsal principal.sbatch "$dryrun" || {
    echo "(DRYRUN=1 bash herramientas/principal.sbatch prints the 13 commands and writes nothing.)" >&2
    return 2
  }

  local MANIFEST_SRC="$H/manifiesto-corridas.tsv"
  local EXPECTED_SRC="$H/entradas-esperadas.sha256"
  local MASKDIR="$E1/analisis/roi"
  local pre=() f
  for f in "$MANIFEST_SRC" "$EXPECTED_SRC" "$MASKDIR/R_int.mask.txt" "$MASKDIR/R_int.mask.txt.sha256" "$H/run_timed.py"; do
    [[ -s "$f" ]] || pre+=("missing $f")
  done
  # disk: the real run needs >= 800 MB free in /home (about 0.5 GB peak, about 0.95 GB with post_run.sh's copies)
  local free_mb
  free_mb="$(stage1_free_mb_home)" || free_mb=""
  if [[ "$dryrun" != 1 ]]; then
    if [[ -z "$free_mb" ]]; then
      pre+=("cannot read the free space in /home from quota -s (need >= $MIN_FREE_MB MB)")
    elif (( free_mb < MIN_FREE_MB )); then
      pre+=("only $free_mb MB free in /home (need >= $MIN_FREE_MB MB; free etapa1-dev first, README.md)")
    fi
  fi
  if [[ ${#pre[@]} -gt 0 ]]; then printf 'pre-flight: %s\n' "${pre[@]}" >&2; return 2; fi
  local nruns
  nruns="$(tail -n +2 "$MANIFEST_SRC" | grep -c .)"
  if [[ "$nruns" -ne "$EXPECTED_RUNS" ]]; then echo "the manifest has $nruns rows, expected $EXPECTED_RUNS" >&2; return 2; fi
  if [[ "$(tail -n +2 "$MANIFEST_SRC" | cut -f1 | sort | uniq -d | wc -l)" -ne 0 ]]; then echo "duplicate run_id in the manifest" >&2; return 2; fi
  if tail -n +2 "$MANIFEST_SRC" | cut -f12 | grep -Eq '^/|(^|/)\.\.(/|$)'; then echo "'salida' must be a relative path without .." >&2; return 2; fi
  local chk
  if ! chk="$(sha256sum -c "$EXPECTED_SRC" 2>&1)"; then
    echo "pre-flight: fixed inputs differ from $EXPECTED_SRC:" >&2; echo "$chk" | grep -v ': OK$' >&2; return 2
  fi

  export MANIFEST="$H/manifiesto-corridas.tsv" EXPECTED_SHA="$EXPECTED_SRC"
  export TABLES="$HOME/oe1-trabajo/tablas" RUN_TIMED="$H/run_timed.py" DRYRUN="$dryrun"
  export MASK="$MASKDIR/R_int.mask.txt"
  MASK_SHA="$(cut -d' ' -f1 "$MASKDIR/R_int.mask.txt.sha256")"
  export MASK_SHA ATT STATUSDIR

  if [[ "$dryrun" == 1 ]]; then
    echo "DRYRUN=1: $nruns commands, nothing is executed or written"
    echo "DRYRUN=1: free space in /home: ${free_mb:-unknown} MB (the real run needs >= $MIN_FREE_MB MB and stops below that)"
    for id in $(tail -n +2 "$MANIFEST_SRC" | cut -f1); do run_one "$id"; done
    return 0
  fi

  # attempt folder: plain mkdir (never reused), then the parents of every run, leaves left to run_one
  mkdir "$ATT" || { echo "the attempt folder $ATT already exists: use a new INTENTO" >&2; return 3; }
  mkdir "$STATUSDIR"
  cp -p "$MANIFEST_SRC" "$ATT/manifiesto-corridas.tsv"
  cp -p "$EXPECTED_SRC" "$ATT/entradas-esperadas.sha256"
  cp -p "$H/env_stage1.sh" "$ATT/env_stage1.sh.copia"
  cp -p "$H/run_timed.py" "$ATT/run_timed.py.copia"
  cp -p "$0" "$ATT/principal.sbatch.copia" 2>/dev/null || true
  export MANIFEST="$ATT/manifiesto-corridas.tsv" EXPECTED_SHA="$ATT/entradas-esperadas.sha256"
  stage1_write_entorno "$ATT/entorno.txt"
  {
    echo "fecha_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "trabajo=${SLURM_JOB_ID:-manual}"
    echo "nodo=$(hostname)"
    echo "intento=$INTENTO"
    echo "principal_sbatch_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
    echo "manifiesto_sha256=$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
    echo "entradas_esperadas_sha256=$(sha256sum "$EXPECTED_SHA" | cut -d' ' -f1)"
    echo "xargs_paralelismo=$nruns"
    echo "libre_en_home_mb=$free_mb (minimo exigido $MIN_FREE_MB)"
    echo "interruptores_ignorados_en_trabajo=${STAGE1_IGNORED_SWITCHES:-ninguno}"
  } > "$ATT/lanzamiento.txt"
  while IFS=$'\t' read -r out; do mkdir -p "$ATT/$(dirname "$out")"; done < <(tail -n +2 "$MANIFEST" | cut -f12)

  tail -n +2 "$MANIFEST" | cut -f1 | xargs -P "$nruns" -I{} bash -c 'run_one "$1"' _ {}
  local xrc=$?
  local n bad
  n="$(find "$STATUSDIR" -name '*.status' | wc -l)"
  bad="$(cat "$STATUSDIR"/*.status 2>/dev/null | awk -F'\t' '$2 != "OK"' | wc -l)"
  { printf 'corrida\tveredicto\tcodigo\tmotivo\tnodo\ttrabajo\tinicio_utc\tfin_utc\n'; cat "$STATUSDIR"/*.status 2>/dev/null | LC_ALL=C sort; } > "$ATT/estado.tsv"
  cat "$ATT/estado.tsv"
  if [[ $xrc -ne 0 || "$n" -ne "$nruns" || "$bad" -ne 0 ]]; then
    echo "ETAPA1 FALLO xargs=$xrc estados=$n esperados=$nruns no_OK=$bad" | tee "$ATT/FALLO.txt"
    return 1
  fi
  echo "ETAPA1 OK: $nruns de $nruns corridas OK en $(hostname), intento $INTENTO" | tee "$ATT/OK.txt"
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
  exit $?
fi
