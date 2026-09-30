#!/bin/bash
#SBATCH --job-name=oe1-e1-prep
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/%x-%j.out
#
# OE1 stage 1, PREPARATION job (no transport): controls K2, K1 and K-ROI.
#
#   K2     `--check-geometry` of A2 and B2 on geo-train-06 and geo-train-01, view 0, seed 20260929
#          (the header must say resolution=100000 tolerance=0 maxErr=100 seed=20260929).
#          B2: OVERLAP-CHECK-SUMMARY status=OK, overlapping=0, no G4Exception of any code, and on
#          geo-train-06 exactly two ANOMALY-CSG lines, both `clipped=no` (the 1 mm clip margin is not
#          exercised). A2: geo-train-01 must be OK; geo-train-06 is expected to show OVERLAPS (the
#          historical defect, recorded; a different outcome is only reported).
#   K1     the six production lists (geo-train-06 and geo-train-01, source seeds 2026092901/02/03,
#          N = 100000, case codes 6 and 1, replicas 1..3) written by A2 and by B2 into separate
#          folders and compared with cmp; `--check-primaries` by A2 and by B2 on the same lists,
#          outputs compared with cmp; six distinct canonical FNVs; the Python validator
#          (oe1_stage1_lists.py: structure identical to the binary's check, geometry, source model).
#   K-ROI  sha256 of the exported R_int mask, verification against R_int-congelada.json and the
#          geometric proof of its orientation (oe1_stage1_roi.py kroi).
#   then   chmod a-w on the lists and sidecars.
#
# Needs (see README.md): build/A2 and build/B2 from build_stage1.sh, the tools in herramientas/, the
# exported mask and the npz in analisis/roi/, tables in ~/oe1-trabajo/tablas, catalog in
# ~/oe1-trabajo/catalogo.
#
#     cd ~/oe1-trabajo/etapa1 && sbatch --parsable herramientas/preparacion.sbatch
#
# PREP_NN: edit in a COPY per attempt; every folder below carries it and is created with a plain mkdir.
# The script exits non-zero if ANY check fails (all checks still run so the report is complete). Started outside a
# SLURM job (SLURM_JOB_ID empty) and outside a rehearsal tree (env_stage1.sh: marker .arbol-de-prueba) it refuses
# to start, exit code 2, before any folder is created; a missing input is exit code 2 too, an existing folder 3.
PREP_NN="01"

E1="$HOME/oe1-trabajo/etapa1"
H="$E1/herramientas"
# shellcheck disable=SC1091
source "$H/env_stage1.sh" || exit 2
set -u -o pipefail
stage1_require_job_or_rehearsal preparacion.sbatch || exit 2

TABLES="$HOME/oe1-trabajo/tablas"
CAT="$HOME/oe1-trabajo/catalogo"
BIN_A="$E1/build/A2/muography_sim"
BIN_B="$E1/build/B2/muography_sim"
ROI="$E1/analisis/roi"
K2="$E1/control-geometrico/prep-$PREP_NN"
LIS="$E1/listas/prep-$PREP_NN"
VER="$E1/verificacion/prep-$PREP_NN"
export OE1_CROSSING_REFERENCE="$E1/fuentes/A2/simulation/geant4_app/tests/crossing_reference.py"
CASES=("geo-train-06:6" "geo-train-01:1")
REPS=(1 2 3)
K2_SEED=20260929

FAILS=()
fail() { FAILS+=("$1"); echo "FALLO: $1" >&2; }
note() { echo "NOTA: $1"; }

for f in "$BIN_A" "$BIN_B" "$H/oe1_stage1_lists.py" "$H/oe1_stage1_roi.py" "$H/oe1_stage1_common.py" \
         "$OE1_CROSSING_REFERENCE" "$ROI/R_int.mask.txt" "$ROI/R_int.mask.txt.sha256" \
         "$ROI/R_int-congelada.json" "$ROI/roi-mascaras.npz" "$TABLES/row_weights_30d.txt" \
         "$CAT/geo-train-06.json" "$CAT/geo-train-01.json"; do
  [[ -e "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir "$K2" "$LIS" "$LIS/A2" "$LIS/B2" "$VER" || { echo "the preparation folders for PREP_NN=$PREP_NN already exist: use a new PREP_NN" >&2; exit 3; }
stage1_write_entorno "$VER/entorno.txt"
sha256sum "$BIN_A" "$BIN_B" "$0" "$H"/oe1_stage1_*.py "$H/make_manifest.py" "$H/env_stage1.sh" > "$VER/herramientas.sha256" 2>/dev/null || true

# ------------------------------------------------------------------------------------------------
# K2: geometry checks
# ------------------------------------------------------------------------------------------------
echo "=== K2 (--check-geometry, seed $K2_SEED)"
printf 'binario\tcaso\tsalida\tstatus\toverlapping\tANOMALY-CSG\tclipped_no\tG4Exception\tcabecera_ok\n' > "$K2/resumen-K2.tsv"
for V in A2 B2; do
  if [[ $V == A2 ]]; then bin="$BIN_A"; else bin="$BIN_B"; fi
  for spec in "${CASES[@]}"; do
    caso="${spec%%:*}"
    base="$K2/$V-$caso"
    "$bin" --geometry "$CAT/$caso.json" --view 0 --tables "$TABLES" --seed "$K2_SEED" --check-geometry > "$base.out" 2> "$base.err"
    rc=$?
    summary="$(grep -m1 '^OVERLAP-CHECK-SUMMARY' "$base.out")"
    status="$(sed -n 's/.* status=\([A-Z]*\).*/\1/p' <<< "$summary")"
    overl="$(sed -n 's/.* overlapping=\([0-9]*\).*/\1/p' <<< "$summary")"
    csg="$(grep -c '^ANOMALY-CSG' "$base.out")"
    csg_no="$(grep '^ANOMALY-CSG' "$base.out" | grep -c ' clipped=no ')"
    nexc="$(cat "$base.out" "$base.err" | grep -c 'G4Exception')"
    header_ok=no
    if grep -q '^OVERLAP-CHECK-HEADER resolution=100000 tolerance=0 maxErr=100 seed='"$K2_SEED"' ' "$base.out"; then header_ok=yes; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$V" "$caso" "$rc" "${status:-AUSENTE}" "${overl:-?}" "$csg" "$csg_no" "$nexc" "$header_ok" >> "$K2/resumen-K2.tsv"
    [[ $header_ok == yes ]] || fail "K2 $V $caso: la cabecera OVERLAP-CHECK-HEADER no es resolution=100000 tolerance=0 maxErr=100 seed=$K2_SEED"
    [[ -n "$summary" ]] || fail "K2 $V $caso: falta OVERLAP-CHECK-SUMMARY"
    if [[ $V == B2 ]]; then
      [[ $rc -eq 0 && "$status" == OK && "$overl" == 0 ]] || fail "K2 B2 $caso: status=$status overlapping=$overl exit=$rc (debe ser OK, 0, 0)"
      [[ $nexc -eq 0 ]] || fail "K2 B2 $caso: $nexc avisos G4Exception (un aviso nuevo en B2 bloquea)"
      if [[ $caso == geo-train-06 ]]; then
        [[ $csg -eq 2 && $csg_no -eq 2 ]] || fail "K2 B2 $caso: ANOMALY-CSG=$csg con clipped=no=$csg_no (deben ser 2 y 2: el margen de 1 mm no se ejerce)"
      else
        [[ $csg -eq 0 ]] || fail "K2 B2 $caso: aparecen lineas ANOMALY-CSG"
      fi
    else
      if [[ $caso == geo-train-01 ]]; then
        [[ $rc -eq 0 && "$status" == OK ]] || fail "K2 A2 $caso: status=$status exit=$rc (debe ser OK)"
      else
        [[ "$status" == OVERLAPS ]] && note "K2 A2 $caso: el solape historico se reproduce (overlapping=$overl, $nexc avisos G4Exception)" \
          || note "K2 A2 $caso: se esperaba OVERLAPS y se obtuvo status=$status"
      fi
    fi
  done
done

# ------------------------------------------------------------------------------------------------
# K1: lists
# ------------------------------------------------------------------------------------------------
echo "=== K1 (six production lists, A2 and B2)"
GEN="$LIS/generacion.txt"
: > "$GEN"
names=()
for V in A2 B2; do
  if [[ $V == A2 ]]; then bin="$BIN_A"; else bin="$BIN_B"; fi
  for spec in "${CASES[@]}"; do
    caso="${spec%%:*}"; code="${spec##*:}"
    for r in "${REPS[@]}"; do
      s=$((2026092900 + r))
      name="${caso}_v0_r${r}_s${s}.primarios.csv"
      [[ $V == A2 ]] && names+=("$name")
      cmd=("$bin" --write-primaries "$LIS/$V/$name" --geometry "$CAT/$caso.json" --view 0 --tables "$TABLES"
           --seed "$s" --events 100000 --case-code "$code" --replica "$r")
      { printf '$ '; printf '%q ' "${cmd[@]}"; printf '\n'; } >> "$GEN"
      "${cmd[@]}" >> "$GEN" 2>&1 || fail "K1 write-primaries $V $name (exit $?)"
    done
  done
done
for name in "${names[@]}"; do
  if [[ -s "$LIS/A2/$name" && -s "$LIS/B2/$name" ]]; then
    cmp -s "$LIS/A2/$name" "$LIS/B2/$name" || fail "K1: la lista $name de A2 y B2 difiere (cmp)"
    diff <(grep -v '"binary_sha256"' "$LIS/A2/$name.meta.json") <(grep -v '"binary_sha256"' "$LIS/B2/$name.meta.json") \
      > /dev/null || fail "K1: el sidecar de $name difiere entre A2 y B2 mas alla de binary_sha256"
  else
    fail "K1: falta la lista $name en A2 o en B2"
  fi
done
( cd "$LIS/A2" && sha256sum -- "${names[@]}" $(printf '%s.meta.json ' "${names[@]}") ) > "$LIS/listas.sha256" 2>/dev/null \
  || fail "K1: no se pudo escribir listas.sha256"
: > "$LIS/check-A2.txt"; : > "$LIS/check-B2.txt"
for name in "${names[@]}"; do
  "$BIN_A" --check-primaries "$LIS/A2/$name" >> "$LIS/check-A2.txt" 2>> "$LIS/check-A2.err" || fail "K1 check-primaries A2 $name"
  "$BIN_B" --check-primaries "$LIS/A2/$name" >> "$LIS/check-B2.txt" 2>> "$LIS/check-B2.err" || fail "K1 check-primaries B2 $name"
done
cmp -s "$LIS/check-A2.txt" "$LIS/check-B2.txt" || fail "K1: las salidas de --check-primaries de A2 y B2 difieren (cmp)"
[[ "$(grep -c ' status=OK$' "$LIS/check-A2.txt")" -eq 6 ]] || fail "K1: --check-primaries no dio status=OK en las seis listas"
[[ "$(grep -o 'canon_fnv1a64=[0-9a-f]*' "$LIS/check-A2.txt" | sort -u | wc -l)" -eq 6 ]] || fail "K1: los FNV canonicos no son seis distintos"
python3 "$H/oe1_stage1_lists.py" validate "${names[@]/#/$LIS/A2/}" --tables "$TABLES" --catalog-dir "$CAT" \
  --cpp-check "$LIS/check-A2.txt" --expect-distinct --out "$VER/validacion-listas.json" > "$VER/validacion-listas.txt" 2>&1
rcv=$?
[[ $rcv -eq 0 ]] || fail "K1: el validador de listas termino con codigo $rcv (3 = estructura o geometria; 4 = prueba estadistica con p < 1e-4; ver validacion-listas.txt)"
cat "$VER/validacion-listas.txt"

# ------------------------------------------------------------------------------------------------
# K-ROI
# ------------------------------------------------------------------------------------------------
echo "=== K-ROI"
( cd "$ROI" && sha256sum -c R_int.mask.txt.sha256 ) > "$VER/roi-sha256.txt" 2>&1 || fail "K-ROI: el sha256 del texto de la mascara no coincide con R_int.mask.txt.sha256"
python3 "$H/oe1_stage1_roi.py" verify --mask "$ROI/R_int.mask.txt" --frozen "$ROI/R_int-congelada.json" \
  --npz "$ROI/roi-mascaras.npz" > "$VER/roi-verificacion.txt" 2>&1 || fail "K-ROI: la mascara no coincide con R_int-congelada.json / el plan"
python3 "$H/oe1_stage1_roi.py" kroi --npz "$ROI/roi-mascaras.npz" --out-dir "$VER/kroi" --no-figure > "$VER/kroi.txt" 2>&1 \
  || fail "K-ROI: la prueba geometrica contradice la mascara (detener e informar)"
cat "$VER/roi-verificacion.txt" "$VER/kroi.txt"

# ------------------------------------------------------------------------------------------------
chmod a-w "$LIS"/A2/* "$LIS"/B2/* 2>/dev/null || fail "no se pudo hacer chmod a-w a las listas"
{
  echo "fecha_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "trabajo=${SLURM_JOB_ID:-manual}"
  echo "nodo=$(hostname)"
  echo "PREP_NN=$PREP_NN"
  echo "fallos=${#FAILS[@]}"
  for m in "${FAILS[@]}"; do echo "FALLO: $m"; done
  echo "resultado=$([[ ${#FAILS[@]} -eq 0 ]] && echo OK || echo FALLO)"
} | tee "$VER/estado-preparacion.txt"
[[ ${#FAILS[@]} -eq 0 ]]
