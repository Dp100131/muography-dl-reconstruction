#!/bin/bash
#SBATCH --job-name=oe1-e2
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/home/dapintoo/oe1-trabajo/etapa2/slurm/%x-%j.out
#
# OE1 stage 2 (protocol v1.1): the transports of one group of replicas in ONE job on ONE node.
#
#   GRUPO=controles DESDE=201 HASTA=203   the 7 control runs (3 A/B on geo-train-01 and the A/A repeat of r201)
#   GRUPO=m1 DESDE=101 HASTA=112          12 main blocks of geo-train-06 (24 runs)
#   GRUPO=m2 DESDE=113 HASTA=124          the other 12 main blocks
#   INTENTO=01                            attempt number; a retry uses a new number and never reuses outputs
#
#   sbatch --parsable --export=ALL,GRUPO=controles,DESDE=201,HASTA=203,INTENTO=01 corridas.sbatch
#   (sbatch --export separates variables with commas, so the replicas are given as a range)
#
# Steps, all inside the job:
#   1. regenerate the lists of the group with A2 and the frozen seed in the node-local /tmp and stop unless every
#      file and sidecar has the SHA-256 frozen by listas.sbatch (job L, before any transport);
#   2. write the stage-1 run manifest of the group (oe1_stage2_analysis.py plan) and the expected input hashes;
#   3. run every transport with stage 1's run_one (sourced from principal.sbatch, unchanged): one single-thread
#      process per run, all at once with xargs -P (A and B of a block, and the A/A repeat, share the node);
#   4. analyse in the node: `controls` for the control group, `block` per main replica;
#   5. compress the muon logs and the lists (gzip -n) and export them, with a tar of all the small files, through
#      the /home buffer $E2/salida: a file is copied only while it leaves >= 150 MB of quota free (one exporter at
#      a time, mkdir lock), with its .sha256 and a .listo marker; pull_bundles.py (on the students' machine)
#      downloads it, verifies the SHA-256 and only then removes it from SC3.
# Storage deviation (PREPARACION_ETAPA_2.md): no 30 GiB location exists in SC3 for this user.
#
# Exit codes: 0 all runs OK and the analysis OK; 1 a run failed or the analysis blocked (everything is still
# exported as evidence); 2 missing input / not in a job / not enough /tmp; 3 the attempt record already exists;
# 4 a regenerated list does not match its frozen SHA-256 (nothing is transported).

set -u -o pipefail
: "${GRUPO:?GRUPO (controles, m1, m2)}" "${DESDE:?DESDE}" "${HASTA:?HASTA}" "${INTENTO:?INTENTO}"
MY_INTENTO="$INTENTO"
E1="$HOME/oe1-trabajo/etapa1"
E2="$HOME/oe1-trabajo/etapa2"
H2="$E2/herramientas"
# shellcheck disable=SC1091
source "$H2/env_stage1.sh" || exit 2
# run_one, warning_codes and check_meta of stage 1, unchanged: principal.sbatch only defines them when sourced
# (its top-level variables, among them INTENTO, ATT and STATUSDIR, are set again below)
# shellcheck disable=SC1091
source "$H2/principal.sbatch"
set -u -o pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "corridas.sbatch runs only inside a SLURM job (sbatch)" >&2; exit 2; }

INTENTO="$MY_INTENTO"
REPLICAS="$(seq -s, "$DESDE" "$HASTA")"
export TABLES="$HOME/oe1-trabajo/tablas"
CAT="$HOME/oe1-trabajo/catalogo"
BIN_A="$E1/build/A2/muography_sim"
BIN_B="$E1/build/B2/muography_sim"
export MASK="$E1/analisis/roi/R_int.mask.txt"
CORRIDAS="$H2/corridas.csv"
FROZEN="$E2/listas-congeladas/listas.sha256"
SAL="$E2/salida"
REG="$E2/registro/$GRUPO-intento-$INTENTO"
TMPW="/tmp/oe1-etapa2-$SLURM_JOB_ID"
LIS="$TMPW/listas"
export ATT="$TMPW/intento-$INTENTO"
export STATUSDIR="$ATT/estado"
EXP="$TMPW/exportar"
TAG="i$INTENTO"
MIN_LEFT_MB=150

for f in "$BIN_A" "$BIN_B" "$MASK" "$CORRIDAS" "$FROZEN" "$H2/run_timed.py" "$H2/oe1_stage2_analysis.py" \
         "$H2/entradas-congeladas.sha256"; do
  [[ -e "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
[[ -d "$SAL" ]] || { echo "missing buffer $SAL" >&2; exit 2; }
mkdir -p "$E2/registro"
mkdir "$REG" || { echo "$REG already exists: use a new INTENTO" >&2; exit 3; }
mkdir "$TMPW" "$LIS" "$EXP" || exit 2
cleanup() { rm -rf "$TMPW"; }
trap cleanup EXIT

nrep=$(( HASTA - DESDE + 1 ))
if [[ "$GRUPO" == controles ]]; then need_gb=3; else need_gb=$(( nrep * 2 )); fi
avail_gb="$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')"
{ echo "grupo=$GRUPO replicas=$REPLICAS intento=$INTENTO trabajo=$SLURM_JOB_ID nodo=$(hostname)"
  echo "tmp_libre_GB=$avail_gb necesario_GB=$need_gb"; df -h /tmp; free -g; } > "$REG/espacio.txt"
(( avail_gb >= need_gb )) || { echo "only ${avail_gb} GB in /tmp, need ${need_gb}" >&2; exit 2; }
stage1_write_entorno "$REG/entorno.txt"
( cd "$HOME/oe1-trabajo" && sha256sum -c --strict "$H2/entradas-congeladas.sha256" ) > "$REG/entradas-congeladas.check" 2>&1 \
  || { echo "frozen inputs changed" >&2; exit 2; }
sha256sum "$0" "$H2"/*.py "$H2"/*.sh "$H2"/*.sbatch "$CORRIDAS" "$FROZEN" > "$REG/herramientas.sha256" 2>/dev/null || true

# ---------------------------------------------------------------------------- 1. lists
python3 "$H2/oe1_stage2_analysis.py" plan --corridas "$CORRIDAS" --replicas "$REPLICAS" --lists-tsv "$REG/listas.tsv" \
  > "$REG/plan.txt" 2>&1 || { echo "plan failed (plan.txt)" >&2; exit 2; }
gen_one() {
  local name="$1" caso="$2" code="$3" rep="$4" seed="$5" n="$6"
  local -a cmd=("$BIN_A" --write-primaries "$LIS/$name" --geometry "$CAT/$caso.json" --view 0 --tables "$TABLES"
                --seed "$seed" --events "$n" --case-code "$code" --replica "$rep")
  { printf '$ '; printf '%q ' "${cmd[@]}"; printf '\n'; "${cmd[@]}" 2>&1; echo "exit=$?"; } > "$LIS/$name.gen.log"
}
export -f gen_one
export BIN_A CAT LIS
tail -n +2 "$REG/listas.tsv" | tr '\t' ' ' | xargs -P 24 -L 1 bash -c 'gen_one "$@"' _
names=()
while IFS=$'\t' read -r name rest; do names+=("$name"); done < <(tail -n +2 "$REG/listas.tsv")
cat "$LIS"/*.gen.log > "$REG/generacion.log"
printf '%s\n' "${names[@]}" "${names[@]/%/.meta.json}" > "$TMPW/nombres.txt"
want="$(awk 'NR == FNR { w[$1] = 1; next } ($2 in w)' "$TMPW/nombres.txt" "$FROZEN")"
[[ "$(grep -c . <<< "$want")" -eq $(( 2 * ${#names[@]} )) ]] || { echo "frozen SHA missing for some list" >&2; exit 4; }
( cd "$LIS" && sha256sum -c --strict <<< "$want" ) > "$REG/listas-regeneradas.check" 2>&1 \
  || { echo "a regenerated list differs from its frozen SHA-256: nothing is transported" >&2; exit 4; }

# ---------------------------------------------------------------------------- 2. manifest and expected hashes
python3 "$H2/oe1_stage2_analysis.py" plan --corridas "$CORRIDAS" --replicas "$REPLICAS" --manifest-tsv "$REG/manifiesto-corridas.tsv" \
  --list-sha "$FROZEN" --lists-dir "$LIS" --catalog-dir "$CAT" --bin-a "$BIN_A" --bin-b "$BIN_B" >> "$REG/plan.txt" 2>&1 \
  || { echo "manifest failed (plan.txt)" >&2; exit 2; }
export MANIFEST="$REG/manifiesto-corridas.tsv"
export EXPECTED_SHA="$REG/entradas-esperadas.sha256"
{
  while read -r sha rel; do printf '%s  %s\n' "$sha" "$HOME/oe1-trabajo/$rel"; done < "$H2/entradas-congeladas.sha256"
  while read -r sha name; do printf '%s  %s\n' "$sha" "$LIS/$name"; done <<< "$want"
} > "$EXPECTED_SHA"
export MASK_SHA
MASK_SHA="$(sha256sum "$MASK" | cut -d' ' -f1)"
export RUN_TIMED="$H2/run_timed.py" DRYRUN=0
nruns="$(tail -n +2 "$MANIFEST" | grep -c .)"

# ---------------------------------------------------------------------------- 3. transports
mkdir "$ATT" "$STATUSDIR"
while IFS=$'\t' read -r out; do mkdir -p "$ATT/$(dirname "$out")"; done < <(tail -n +2 "$MANIFEST" | cut -f12)
{ echo "fecha_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"; echo "trabajo=$SLURM_JOB_ID"; echo "nodo=$(hostname)"
  echo "grupo=$GRUPO"; echo "intento=$INTENTO"; echo "replicas=$REPLICAS"; echo "corridas=$nruns"
  echo "xargs_paralelismo=$nruns"; echo "corridas_sbatch_sha256=$(sha256sum "$0" | cut -d' ' -f1)"; } > "$REG/lanzamiento.txt"
tail -n +2 "$MANIFEST" | cut -f1 | xargs -P "$nruns" -I{} bash -c 'run_one "$1"' _ {}
xrc=$?
{ printf 'corrida\tveredicto\tcodigo\tmotivo\tnodo\ttrabajo\tinicio_utc\tfin_utc\n'; cat "$STATUSDIR"/*.status 2>/dev/null | LC_ALL=C sort; } > "$ATT/estado.tsv"
nst="$(find "$STATUSDIR" -name '*.status' | wc -l)"
bad="$(cat "$STATUSDIR"/*.status 2>/dev/null | awk -F'\t' '$2 != "OK"' | wc -l)"
runs_ok=0
[[ $xrc -eq 0 && "$nst" -eq "$nruns" && "$bad" -eq 0 ]] && runs_ok=1
cp -p "$ATT/estado.tsv" "$REG/estado.tsv"
echo "corridas_ok=$runs_ok xargs=$xrc estados=$nst no_OK=$bad fin_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$REG/lanzamiento.txt"

# ---------------------------------------------------------------------------- 4. analysis in the node
mkdir "$ATT/analisis"
ana_ok=1
if [[ "$GRUPO" == controles ]]; then
  python3 "$H2/oe1_stage2_analysis.py" controls --manifest "$MANIFEST" --attempt "$ATT" --lists-dir "$LIS" \
    --mask "$MASK" --out "$ATT/analisis" > "$ATT/analisis/controles.txt" 2>&1 || ana_ok=0
else
  ana_one() {
    python3 "$H2/oe1_stage2_analysis.py" block --manifest "$MANIFEST" --attempt "$ATT" --lists-dir "$LIS" \
      --mask "$MASK" --replica "$1" --out "$ATT/analisis" > "$ATT/analisis/bloque-r$1.txt" 2>&1
    echo "r$1 $?" > "$ATT/analisis/bloque-r$1.rc"
  }
  export -f ana_one
  export H2 LIS
  tr ',' '\n' <<< "$REPLICAS" | xargs -P 8 -I{} bash -c 'ana_one "$1"' _ {}
  for rc in "$ATT"/analisis/*.rc; do [[ "$(cut -d' ' -f2 "$rc")" == 0 ]] || ana_ok=0; done
fi
cat "$ATT"/analisis/*.txt | tee "$REG/analisis.txt"
echo "analisis_ok=$ana_ok fin_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$REG/lanzamiento.txt"

# ---------------------------------------------------------------------------- 5. compress and export
# the big files: one .gz per muon log and per list; the raw SHA-256 of each log stays in the run's SHA256SUMS
gz_one() {
  local f="$1" dst="$2"
  gzip -n -6 -c "$f" > "$dst" && rm -f "$f"
}
export -f gz_one
tail -n +2 "$MANIFEST" | cut -f1,12 | while IFS=$'\t' read -r run_id out; do
  for f in "$ATT/$out"/*_muones.csv; do [[ -e "$f" ]] && printf '%s\t%s\n' "$f" "$EXP/$TAG-$run_id-muones.csv.gz"; done
done > "$TMPW/comprimir.tsv"
for name in "${names[@]}"; do printf '%s\t%s\n' "$LIS/$name" "$EXP/$TAG-$name.gz"; done >> "$TMPW/comprimir.tsv"
tr '\t' ' ' < "$TMPW/comprimir.tsv" | xargs -P 12 -L 1 bash -c 'gz_one "$1" "$2"' _
# the small files: every run folder (logs compressed above), status, analysis and the job record
cp -rp "$ATT" "$REG/intento"
mkdir "$REG/sidecars" && cp -p "$LIS"/*.meta.json "$REG/sidecars/"
( cd "$EXP" && sha256sum -- * ) > "$REG/exportados.sha256"
echo "compresion_fin_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$REG/lanzamiento.txt"
tar -C "$E2/registro" -cf "$EXP/$TAG-$GRUPO-pequenos.tar" "$GRUPO-intento-$INTENTO"

export_one() {
  local f="$1" base size_mb left rc
  base="$(basename "$f")"
  size_mb=$(( $(stat -c %s "$f") / 1048576 + 1 ))
  while true; do
    if mkdir "$SAL/.cerrojo" 2>/dev/null; then
      left="$(stage1_free_mb_home)" || left=0
      if (( left - size_mb >= MIN_LEFT_MB )); then
        cp "$f" "$SAL/$base.part" && mv "$SAL/$base.part" "$SAL/$base" \
          && ( cd "$SAL" && sha256sum -- "$base" > "$base.sha256" ) && : > "$SAL/$base.listo"
        rc=$?
        rmdir "$SAL/.cerrojo"
        echo "$(date -u '+%H:%M:%S') exportado $base (${size_mb} MB, rc=$rc, libre antes ${left} MB)" >> "$REG/exportacion.log"
        return $rc
      fi
      rmdir "$SAL/.cerrojo"
    fi
    sleep 15
  done
}
exp_ok=1
export_one "$EXP/$TAG-$GRUPO-pequenos.tar" || exp_ok=0
for f in "$EXP"/*.gz; do export_one "$f" || exp_ok=0; done
echo "exportacion_ok=$exp_ok fin_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$REG/exportacion.log"

echo "GRUPO $GRUPO intento $INTENTO: corridas_ok=$runs_ok analisis_ok=$ana_ok exportacion_ok=$exp_ok" | tee "$REG/RESULTADO.txt"
[[ $runs_ok -eq 1 && $ana_ok -eq 1 && $exp_ok -eq 1 ]]
