#!/bin/bash
# OE1 stage 1: Release build of A2 and B2 inside a compute allocation.
#
# Run it from the login node with EXACTLY this line (records the job in sacct and in compilacion.txt):
#
#   srun -p legacy -N1 -c8 --mem=8G -t 00:30:00 bash ~/oe1-trabajo/etapa1/herramientas/build_stage1.sh
#
# Builds ~/oe1-trabajo/etapa1/fuentes/{A2,B2}/simulation/geant4_app into ~/oe1-trabajo/etapa1/build/{A2,B2}
# (`cmake -S ... -B ... -DCMAKE_BUILD_TYPE=Release`, i.e. -O3 -DNDEBUG). It refuses to reuse a build
# directory and writes:
#   build/cmake-<V>.log  build/build-<V>.log  build/compilacion.txt
# compilacion.txt: compiler, cmake, flags, warning count, binary sha256 / size / mtime, job id, host.
# Any compiler warning or error makes the script exit non-zero (PLAN: 0 warnings expected).
set -u -o pipefail

E1="$HOME/oe1-trabajo/etapa1"
# shellcheck disable=SC1091
source "$E1/herramientas/env_stage1.sh" || exit 2
set -u

fail=0
for V in A2 B2; do
  if [[ ! -d "$E1/fuentes/$V/simulation/geant4_app" ]]; then echo "missing sources for $V" >&2; exit 2; fi
  if [[ -e "$E1/build/$V" ]]; then echo "refusing to reuse $E1/build/$V" >&2; exit 3; fi
done
if [[ -e "$E1/build/compilacion.txt" ]]; then echo "refusing to overwrite compilacion.txt" >&2; exit 3; fi

REC="$E1/build/compilacion.txt"
JOBS="${SLURM_CPUS_PER_TASK:-8}"
{
  echo "fecha_inicio_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "linea_srun=srun -p legacy -N1 -c8 --mem=8G -t 00:30:00 bash $E1/herramientas/build_stage1.sh"
  echo "modulos=$STAGE1_MODULES"
  echo "compilador_ruta=$(command -v g++)"
  echo "compilador=$(g++ --version | head -1)"
  echo "cmake_ruta=$(command -v cmake)"
  echo "cmake=$(cmake --version | head -1)"
  echo "geant4=$(geant4-config --version)"
  echo "paralelismo=-j$JOBS"
  echo "env_stage1_sh_sha256=$(sha256sum "$E1/herramientas/env_stage1.sh" | cut -d' ' -f1)"
} > "$REC"

for V in A2 B2; do
  SRC="$E1/fuentes/$V/simulation/geant4_app"
  BLD="$E1/build/$V"
  cmake -S "$SRC" -B "$BLD" -DCMAKE_BUILD_TYPE=Release > "$E1/build/cmake-$V.log" 2>&1
  crc=$?
  brc=99
  if [[ $crc -eq 0 ]]; then
    cmake --build "$BLD" -j"$JOBS" > "$E1/build/build-$V.log" 2>&1
    brc=$?
  fi
  warnings="$(grep -c 'warning:' "$E1/build/build-$V.log" 2>/dev/null || true)"
  errors="$(grep -c 'error:' "$E1/build/build-$V.log" 2>/dev/null || true)"
  bin="$BLD/muography_sim"
  {
    echo "--- $V ---"
    echo "$V.configure_exit=$crc"
    echo "$V.build_exit=$brc"
    echo "$V.advertencias=${warnings:-0}"
    echo "$V.errores=${errors:-0}"
    echo "$V.CMAKE_BUILD_TYPE=$(grep -E '^CMAKE_BUILD_TYPE:' "$BLD/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
    echo "$V.CMAKE_CXX_FLAGS=$(grep -E '^CMAKE_CXX_FLAGS:' "$BLD/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
    echo "$V.CMAKE_CXX_FLAGS_RELEASE=$(grep -E '^CMAKE_CXX_FLAGS_RELEASE:' "$BLD/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
    echo "$V.flags_de_compilacion=$(grep -E '^CXX_FLAGS' "$BLD/CMakeFiles/muography_sim.dir/flags.make" 2>/dev/null | cut -d= -f2-)"
    if [[ -x "$bin" ]]; then
      echo "$V.binario=$bin"
      echo "$V.sha256=$(sha256sum "$bin" | cut -d' ' -f1)"
      echo "$V.bytes=$(stat -c %s "$bin")"
      echo "$V.mtime=$(stat -c %y "$bin")"
    else
      echo "$V.binario=AUSENTE"
    fi
  } >> "$REC"
  if [[ $crc -ne 0 || $brc -ne 0 || "${warnings:-0}" != 0 || "${errors:-0}" != 0 || ! -x "$bin" ]]; then
    echo "BUILD $V FAILED or has warnings (see $E1/build/build-$V.log)" >&2
    fail=1
  fi
done
echo "fecha_fin_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$REC"
echo "resultado=$([[ $fail -eq 0 ]] && echo OK || echo FALLO)" >> "$REC"
sha256sum "$E1/build/A2/muography_sim" "$E1/build/B2/muography_sim" > "$E1/build/binarios.sha256" 2>/dev/null || true
cat "$REC"
exit $fail
