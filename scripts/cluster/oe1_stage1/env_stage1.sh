# shellcheck shell=bash
# OE1 stage 1 -- environment of every SC3 script (SOURCE this file, do not execute it).
#
# What it does when sourced
#   1. lmod: source /etc/profile.d/lmod.sh; module purge; module load gnu14/14.2.0 geant4/11.4.2
#      (the pilot's toolchain; a bare `module purge` without gnu14 breaks the link with a GLIBCXX error).
#   2. unset ONLY the five variables that override the run manager or the thread count. The Geant4
#      data variables (G4LEDATA, G4NEUTRONHPDATA, ...) must stay set.
#   3. defines stage1_write_entorno FILE, the writer of entorno.txt.
#   4. exports PYTHONDONTWRITEBYTECODE=1 (no __pycache__ next to the Python tools).
#   5. defines stage1_require_job_or_rehearsal SCRIPT [DRYRUN], the guard that keeps real work off the login
#      node (see the end of this file); preparacion.sbatch, pruebas_tecnicas.sbatch and principal.sbatch call it.
#
# lmod's own scripts read unset variables, so this file switches `set -u` off while it sources lmod and
# restores it afterwards; callers may enable `set -u` before or after sourcing.
#
# STAGE1_SKIP_MODULES=1 skips step 1 and is honoured ONLY together with DRYRUN=1, or when the marker file
# ~/oe1-trabajo/etapa1/.arbol-de-prueba exists (a local rehearsal tree without lmod; never create it on
# SC3). A real run on SC3 can never skip the modules. The same marker is what lets real work start outside
# a SLURM job (stage1_is_rehearsal_tree).
#
# Inside a SLURM job (SLURM_JOB_ID set: sbatch or srun) both test switches, DRYRUN and STAGE1_SKIP_MODULES,
# are IGNORED and unset: sbatch exports the submitting shell's environment, so a DRYRUN=1 left over in the
# login shell would otherwise turn the job into a silent dry run, or skip the modules. What was ignored is
# printed and recorded in entorno.txt (line interruptores_ignorados_en_trabajo).
#
# The five override variables are unset (step 2). Their values BEFORE that (as inherited, and after the modules
# were loaded) are kept in STAGE1_G4_OVERRIDES_INHERITED / STAGE1_G4_OVERRIDES_AFTER_MODULES and written to
# entorno.txt, so the record shows whether the environment tried to override the run manager or the threads.

_stage1_had_nounset=0
case "$-" in *u*) _stage1_had_nounset=1; set +u ;; esac

_stage1_env_fail() {
  echo "env_stage1.sh: $1" >&2
  if [[ $_stage1_had_nounset == 1 ]]; then set -u; fi
  unset _stage1_had_nounset
}

# "NAME=value" (or "NAME=<sin definir>") for each of the five override variables, one per line
_stage1_g4_override_values() {
  local v
  for v in G4FORCENUMBEROFTHREADS G4FORCE_EVENTS_PER_TASK G4FORCE_GRAINSIZE G4RUN_MANAGER_TYPE G4FORCE_RUN_MANAGER_TYPE; do
    if [[ -n "${!v+x}" ]]; then printf '%s=%s\n' "$v" "${!v}"; else printf '%s=<sin definir>\n' "$v"; fi
  done
}

# Exit 0 inside a local rehearsal tree: the marker file exists (never create it on SC3).
stage1_is_rehearsal_tree() { [[ -e "${HOME:-}/oe1-trabajo/etapa1/.arbol-de-prueba" ]]; }

STAGE1_IGNORED_SWITCHES=""
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  for _stage1_v in DRYRUN STAGE1_SKIP_MODULES; do
    if [[ -n "${!_stage1_v:-}" && "${!_stage1_v}" != 0 ]]; then
      STAGE1_IGNORED_SWITCHES+="$_stage1_v=${!_stage1_v} "
    fi
    unset "$_stage1_v"
  done
  unset _stage1_v
  if [[ -n "$STAGE1_IGNORED_SWITCHES" ]]; then
    echo "env_stage1.sh: SLURM_JOB_ID=$SLURM_JOB_ID: IGNORED inside a job (it always runs for real and loads the modules): $STAGE1_IGNORED_SWITCHES" >&2
  fi
fi
export STAGE1_IGNORED_SWITCHES
STAGE1_G4_OVERRIDES_INHERITED="$(_stage1_g4_override_values)"
export STAGE1_G4_OVERRIDES_INHERITED

if [[ "${STAGE1_SKIP_MODULES:-0}" == 1 ]]; then
  if [[ "${DRYRUN:-0}" != 1 ]] && ! stage1_is_rehearsal_tree; then
    _stage1_env_fail "STAGE1_SKIP_MODULES=1 is only allowed with DRYRUN=1 or in a rehearsal tree (marker file)"
    return 1 2>/dev/null || exit 1
  fi
  echo "env_stage1.sh: WARNING modules skipped (DRYRUN or rehearsal tree, no lmod)" >&2
  STAGE1_MODULES="SKIPPED (DRYRUN or rehearsal tree)"
else
  if [[ ! -r /etc/profile.d/lmod.sh ]]; then
    _stage1_env_fail "/etc/profile.d/lmod.sh not found (not on SC3?)"
    return 1 2>/dev/null || exit 1
  fi
  # shellcheck disable=SC1091
  source /etc/profile.d/lmod.sh
  module purge
  module load gnu14/14.2.0 geant4/11.4.2
  STAGE1_MODULES="gnu14/14.2.0 geant4/11.4.2"
  if ! command -v geant4-config >/dev/null 2>&1 || [[ "$(geant4-config --version)" != "11.4.2" ]]; then
    _stage1_env_fail "geant4-config does not report 11.4.2"
    return 1 2>/dev/null || exit 1
  fi
fi

STAGE1_G4_OVERRIDES_AFTER_MODULES="$(_stage1_g4_override_values)"
export STAGE1_G4_OVERRIDES_AFTER_MODULES
unset G4FORCENUMBEROFTHREADS G4FORCE_EVENTS_PER_TASK G4FORCE_GRAINSIZE G4RUN_MANAGER_TYPE G4FORCE_RUN_MANAGER_TYPE
export STAGE1_MODULES
# The Python tools of herramientas/ must not leave __pycache__ folders next to themselves.
export PYTHONDONTWRITEBYTECODE=1

if [[ $_stage1_had_nounset == 1 ]]; then set -u; fi
unset _stage1_had_nounset

# Free space in /home in whole MB from `quota -s` (SC3 format, e.g. "ioserver:/home  18954M  20480M  20480M ..."
# with the numbers on the next line when the filesystem name is long): limit minus used, negative when the quota is
# exceeded. Prints nothing and returns 1 when quota cannot be read (no /home line, no quota, or no answer in 30 s).
# quota itself exits 1 whenever a limit is exceeded (which is also when it prints the "*"), so its exit status is
# ignored: under pipefail it would hide the figure that awk printed.
stage1_free_mb_home() {
  { timeout 30 quota -s 2>/dev/null || true; } | awk '
    function mb(s,   n, u) {
      sub(/\*$/, "", s)   # quota -s flags an exceeded quota with a trailing "*"
      n = s + 0; u = substr(s, length(s))
      if (u == "K") return n / 1024
      if (u == "M") return n
      if (u == "G") return n * 1024
      if (u == "T") return n * 1024 * 1024
      return n / 1024
    }
    /\/home/ { if (NF >= 3) { used = $2; lim = $3 } else { getline; used = $1; lim = $2 }
               printf "%d\n", mb(lim) - mb(used); found = 1; exit }
    END { if (!found) exit 1 }'
}

# Real work (transports, list generation, geometry checks) belongs in a SLURM job, never on the login node.
#   stage1_require_job_or_rehearsal SCRIPT [DRYRUN]
# Returns 0 when SLURM_JOB_ID is set (sbatch or srun), when the rehearsal marker exists or when DRYRUN is 1 (the
# caller's own dry-run flag: only principal.sbatch has one, and it writes nothing). Otherwise it says why on
# stderr and returns 1. Call it in real mode BEFORE any folder is created and exit non-zero when it fails.
stage1_require_job_or_rehearsal() {
  local script="${1:?stage1_require_job_or_rehearsal SCRIPT [DRYRUN]}" dryrun="${2:-0}"
  if [[ "$dryrun" == 1 || -n "${SLURM_JOB_ID:-}" ]] || stage1_is_rehearsal_tree; then return 0; fi
  {
    echo "$script: REFUSED. This is real work: submit it, do not run it on the login node:"
    echo "  cd ~/oe1-trabajo/etapa1 && sbatch --parsable herramientas/$script"
    echo "SLURM_JOB_ID is empty and the rehearsal marker ~/oe1-trabajo/etapa1/.arbol-de-prueba does not exist"
    echo "(only local rehearsal trees have it; never create it on SC3). Nothing was started or created."
  } >&2
  return 1
}

# Writes the environment record of the node that executes this call.
#   stage1_write_entorno FILE     (FILE must not exist)
stage1_write_entorno() {
  local f="${1:?stage1_write_entorno FILE}"
  if [[ -e "$f" ]]; then echo "refusing to overwrite $f" >&2; return 1; fi
  {
    echo "fecha_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "host=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "slurm_nodelist=${SLURM_JOB_NODELIST:-none}"
    echo "usuario=${USER:-unknown}"
    echo "cwd=$(pwd)"
    echo "modulos_cargados=${STAGE1_MODULES:-unknown}"
    echo "env_stage1_sh_sha256=$(sha256sum "${BASH_SOURCE[0]}" 2>/dev/null | cut -d' ' -f1)"
    echo "interruptores_ignorados_en_trabajo=${STAGE1_IGNORED_SWITCHES:-ninguno}"
    echo "--- las cinco variables de anulacion ANTES de borrarlas: heredadas del entorno ---"
    printf '%s\n' "${STAGE1_G4_OVERRIDES_INHERITED:-<no registradas>}"
    echo "--- las cinco variables de anulacion ANTES de borrarlas: tras cargar los modulos ---"
    printf '%s\n' "${STAGE1_G4_OVERRIDES_AFTER_MODULES:-<no registradas>}"
    echo "--- lscpu (nodo de computo) ---"
    lscpu 2>&1 || true
    echo "--- module list ---"
    { module list 2>&1; } || true
    echo "--- env | grep ^G4 (las cinco variables de anulacion NO deben aparecer) ---"
    { env | grep '^G4' | sort; } || true
    echo "--- quota -s ---"
    { timeout 30 quota -s 2>&1; } || true
    echo "--- python3 / numpy (nodo de computo) ---"
    { python3 --version 2>&1; python3 -c 'import numpy; print("numpy", numpy.__version__)' 2>&1; } || true
    echo "--- toolchain ---"
    { g++ --version 2>&1 | head -1; } || true
    { cmake --version 2>&1 | head -1; } || true
    { geant4-config --version 2>&1; } || true
  } > "$f" 2>&1
}
