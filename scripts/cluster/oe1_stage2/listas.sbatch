#!/bin/bash
#SBATCH --job-name=oe1-e2-listas
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dapintoo/oe1-trabajo/etapa2/slurm/%x-%j.out
#
# OE1 stage 2 (protocol v1.1), job L: freeze the 27 primary lists BEFORE any transport.
#
#   * K2 geometry checks of A2 and B2 on both catalogs (as in stage 1);
#   * every list of corridas.csv generated with A2 AND with B2 (--write-primaries): the files must be identical
#     (cmp) and the sidecars equal except binary_sha256;
#   * --check-primaries of A2 and B2 on every list (identical output, status=OK, 27 distinct canonical FNV);
#   * the stage-1 list validator (structure, geometry and source-model tests; numpy only) on every list. A
#     statistical flag (exit 4) is KEPT and explained, never answered by regenerating the list (protocol 8.3).
#
# Storage deviation (PREPARACION_ETAPA_2.md): /home has 596 MB free and no 30 GiB location exists, so the lists
# are written to the node-local /tmp and discarded at the end; only their SHA-256, sidecars and evidence are kept
# in $E2/listas-congeladas. The transport jobs regenerate each list with A2 and the same seed and refuse to
# transport unless its SHA-256 equals the one frozen here.
#
# Submit from anywhere: sbatch --parsable ~/oe1-trabajo/etapa2/herramientas/listas.sbatch
# Exit codes: 0 frozen; 1 a check failed (see estado-listas.txt); 2 missing input or not in a job; 3 output exists.

E1="$HOME/oe1-trabajo/etapa1"
E2="$HOME/oe1-trabajo/etapa2"
H2="$E2/herramientas"
# shellcheck disable=SC1091
source "$H2/env_stage1.sh" || exit 2
set -u -o pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "listas.sbatch runs only inside a SLURM job (sbatch)" >&2; exit 2; }

TABLES="$HOME/oe1-trabajo/tablas"
CAT="$HOME/oe1-trabajo/catalogo"
BIN_A="$E1/build/A2/muography_sim"
BIN_B="$E1/build/B2/muography_sim"
CORRIDAS="$H2/corridas.csv"
OUT="$E2/listas-congeladas"
TMPW="/tmp/oe1-etapa2-$SLURM_JOB_ID"
K2_SEED=20260929

for f in "$BIN_A" "$BIN_B" "$CORRIDAS" "$H2/entradas-congeladas.sha256" "$H2/oe1_stage1_lists.py" \
         "$H2/oe1_stage2_analysis.py" "$CAT/geo-train-06.json" "$CAT/geo-train-01.json"; do
  [[ -e "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir "$OUT" || { echo "$OUT already exists" >&2; exit 3; }
mkdir "$OUT/K2" "$OUT/sidecars" "$OUT/validacion" "$TMPW" "$TMPW/A2" "$TMPW/B2" || exit 3
trap 'rm -rf "$TMPW"' EXIT

FAILS=()
NOTES=()
fail() { FAILS+=("$1"); echo "FALLO: $1" >&2; }
note() { NOTES+=("$1"); echo "NOTA: $1"; }

stage1_write_entorno "$OUT/entorno.txt"
{ echo "--- df -h /tmp (nodo de cálculo) ---"; df -h /tmp; echo "--- memoria ---"; free -g; } >> "$OUT/entorno.txt"
( cd "$HOME/oe1-trabajo" && sha256sum -c --strict "$H2/entradas-congeladas.sha256" ) > "$OUT/entradas-congeladas.check" 2>&1 \
  || fail "insumos congelados distintos (entradas-congeladas.check)"
sha256sum "$0" "$H2"/*.py "$H2"/*.sh "$H2"/*.sbatch "$CORRIDAS" > "$OUT/herramientas.sha256" 2>/dev/null || true

# ------------------------------------------------------------------------------------------------
# K2: geometry checks (same criteria as stage 1)
# ------------------------------------------------------------------------------------------------
printf 'binario\tcaso\tsalida\tstatus\toverlapping\tANOMALY-CSG\tclipped_no\tG4Exception\n' > "$OUT/K2/resumen-K2.tsv"
for V in A2 B2; do
  if [[ $V == A2 ]]; then bin="$BIN_A"; else bin="$BIN_B"; fi
  for caso in geo-train-06 geo-train-01; do
    base="$OUT/K2/$V-$caso"
    "$bin" --geometry "$CAT/$caso.json" --view 0 --tables "$TABLES" --seed "$K2_SEED" --check-geometry > "$base.out" 2> "$base.err"
    rc=$?
    summary="$(grep -m1 '^OVERLAP-CHECK-SUMMARY' "$base.out")"
    status="$(sed -n 's/.* status=\([A-Z]*\).*/\1/p' <<< "$summary")"
    overl="$(sed -n 's/.* overlapping=\([0-9]*\).*/\1/p' <<< "$summary")"
    csg="$(grep -c '^ANOMALY-CSG' "$base.out")"
    csg_no="$(grep '^ANOMALY-CSG' "$base.out" | grep -c ' clipped=no ')"
    nexc="$(cat "$base.out" "$base.err" | grep -c 'G4Exception')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$V" "$caso" "$rc" "${status:-AUSENTE}" "${overl:-?}" "$csg" "$csg_no" "$nexc" >> "$OUT/K2/resumen-K2.tsv"
    if [[ $V == B2 ]]; then
      [[ $rc -eq 0 && "$status" == OK && "$overl" == 0 && $nexc -eq 0 ]] || fail "K2 B2 $caso: status=$status overlapping=$overl avisos=$nexc exit=$rc"
      if [[ $caso == geo-train-06 ]]; then [[ $csg -eq 2 && $csg_no -eq 2 ]] || fail "K2 B2 $caso: ANOMALY-CSG=$csg clipped=no=$csg_no"; fi
    elif [[ $caso == geo-train-01 ]]; then
      [[ $rc -eq 0 && "$status" == OK ]] || fail "K2 A2 $caso: status=$status exit=$rc"
    else
      note "K2 A2 $caso: status=$status overlapping=$overl avisos=$nexc (solape histórico declarado)"
    fi
  done
done

# ------------------------------------------------------------------------------------------------
# lists
# ------------------------------------------------------------------------------------------------
python3 "$H2/oe1_stage2_analysis.py" plan --corridas "$CORRIDAS" --lists-tsv "$OUT/listas.tsv" > "$OUT/plan.txt" 2>&1 \
  || { fail "plan: corridas.csv no coincide con el protocolo (plan.txt)"; }
[[ -s "$OUT/listas.tsv" ]] || { echo "no list table" >&2; exit 1; }
gen_one() {
  set -u -o pipefail
  local V="$1" name="$2" caso="$3" code="$4" rep="$5" seed="$6" n="$7" bin
  if [[ $V == A2 ]]; then bin="$BIN_A"; else bin="$BIN_B"; fi
  local -a cmd=("$bin" --write-primaries "$TMPW/$V/$name" --geometry "$CAT/$caso.json" --view 0 --tables "$TABLES"
                --seed "$seed" --events "$n" --case-code "$code" --replica "$rep")
  { printf '$ '; printf '%q ' "${cmd[@]}"; printf '\n'; } > "$TMPW/$V/$name.gen.log"
  "${cmd[@]}" >> "$TMPW/$V/$name.gen.log" 2>&1
  echo "exit=$?" >> "$TMPW/$V/$name.gen.log"
}
export -f gen_one
export BIN_A BIN_B TMPW CAT TABLES
tail -n +2 "$OUT/listas.tsv" | while IFS=$'\t' read -r name caso code rep seed n; do
  for V in A2 B2; do printf '%s %s %s %s %s %s %s\n' "$V" "$name" "$caso" "$code" "$rep" "$seed" "$n"; done
done | xargs -P 24 -L 1 bash -c 'gen_one "$@"' _
names=()
while IFS=$'\t' read -r name caso code rep seed n; do names+=("$name"); done < <(tail -n +2 "$OUT/listas.tsv")
nl="${#names[@]}"
[[ $nl -eq 27 ]] || fail "se esperaban 27 listas y la tabla tiene $nl"
for name in "${names[@]}"; do
  cat "$TMPW/A2/$name.gen.log" "$TMPW/B2/$name.gen.log" >> "$OUT/generacion.log"
  grep -q '^exit=0$' "$TMPW/A2/$name.gen.log" && grep -q '^exit=0$' "$TMPW/B2/$name.gen.log" || fail "write-primaries $name"
  cmp -s "$TMPW/A2/$name" "$TMPW/B2/$name" || fail "la lista $name de A2 y B2 difiere (cmp)"
  diff <(grep -v '"binary_sha256"' "$TMPW/A2/$name.meta.json") <(grep -v '"binary_sha256"' "$TMPW/B2/$name.meta.json") \
    > /dev/null || fail "el sidecar de $name difiere entre A2 y B2 más allá de binary_sha256"
  cp -p "$TMPW/A2/$name.meta.json" "$OUT/sidecars/"
done
( cd "$TMPW/A2" && sha256sum -- "${names[@]}" $(printf '%s.meta.json ' "${names[@]}") ) > "$OUT/listas.sha256" \
  || fail "no se pudo escribir listas.sha256"
{ cd "$TMPW/A2" && for name in "${names[@]}"; do printf '%s\t%s\n' "$name" "$(stat -c %s "$name")"; done; } > "$OUT/tamanos.tsv"

chk_one() {
  local name="$1"
  "$BIN_A" --check-primaries "$TMPW/A2/$name" > "$TMPW/check-A2-$name.txt" 2>&1
  "$BIN_B" --check-primaries "$TMPW/A2/$name" > "$TMPW/check-B2-$name.txt" 2>&1
}
export -f chk_one
printf '%s\n' "${names[@]}" | xargs -P 24 -I{} bash -c 'chk_one "$1"' _ {}
: > "$OUT/check-A2.txt"; : > "$OUT/check-B2.txt"
for name in "${names[@]}"; do
  cat "$TMPW/check-A2-$name.txt" >> "$OUT/check-A2.txt"; cat "$TMPW/check-B2-$name.txt" >> "$OUT/check-B2.txt"
done
cmp -s "$OUT/check-A2.txt" "$OUT/check-B2.txt" || fail "las salidas de --check-primaries de A2 y B2 difieren"
[[ "$(grep -c ' status=OK$' "$OUT/check-A2.txt")" -eq "$nl" ]] || fail "--check-primaries no dio status=OK en las $nl listas"
[[ "$(grep -o 'canon_fnv1a64=[0-9a-f]*' "$OUT/check-A2.txt" | sort -u | wc -l)" -eq "$nl" ]] || fail "los FNV canónicos no son $nl distintos"

val_one() {
  local name="$1"
  python3 "$H2/oe1_stage1_lists.py" validate "$TMPW/A2/$name" --tables "$TABLES" --catalog-dir "$CAT" \
    --cpp-check "$TMPW/check-A2-$name.txt" --out "$OUT/validacion/$name.json" > "$OUT/validacion/$name.txt" 2>&1
  echo "$name $?" > "$OUT/validacion/$name.rc"
}
export -f val_one
export OUT H2
printf '%s\n' "${names[@]}" | xargs -P 12 -I{} bash -c 'val_one "$1"' _ {}
for name in "${names[@]}"; do
  rc="$(cut -d' ' -f2 "$OUT/validacion/$name.rc" 2>/dev/null)"
  case "$rc" in
    0) ;;
    4) note "validador: prueba estadística con p < 1e-4 en $name (se conserva la lista y se explica)";;
    *) fail "validador de $name terminó con código ${rc:-ausente} (estructura o geometría)";;
  esac
done

{
  echo "fecha_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "trabajo=$SLURM_JOB_ID"
  echo "nodo=$(hostname)"
  echo "listas=$nl"
  echo "fallos=${#FAILS[@]}"
  for m in "${FAILS[@]}"; do echo "FALLO: $m"; done
  echo "notas=${#NOTES[@]}"
  for m in "${NOTES[@]}"; do echo "NOTA: $m"; done
  echo "resultado=$([[ ${#FAILS[@]} -eq 0 ]] && echo OK || echo FALLO)"
} | tee "$OUT/estado-listas.txt"
[[ ${#FAILS[@]} -eq 0 ]]
