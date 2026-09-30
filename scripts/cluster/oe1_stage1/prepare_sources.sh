#!/bin/bash
# OE1 stage 1, step 0 on SC3: create ~/oe1-trabajo/etapa1 and the two instrumented source trees
# A2 (= pilot A + P2) and B2 (= pilot B + P2). Light work only: run it on the login node.
#
#   bash prepare_sources.sh P2-instrumentacion.diff P2-instrumentacion.sha256 [--base-a FILE --base-b FILE]
#
# P2-instrumentacion.diff  the P2 patch as `git diff 9520170 <A2 commit> -- scripts/simulation/geant4_app`
#                          (LF only). The strip level is derived from its paths: git says
#                          b/scripts/simulation/geant4_app/..., the SC3 copies are
#                          simulation/geant4_app/..., hence -p2.
# P2-instrumentacion.sha256 sha256sum lines "<sha256>  simulation/geant4_app/<path>" for exactly the
#                          files the patch touches (make_p2_hashes.sh writes it from the A2 commit).
# --base-a / --base-b      optional sha256 lists of the pilot's unpatched trees (the delivery's
#                          A-fuentes.sha256 / B-fuentes.sha256) checked BEFORE patching.
#
# It only READS ~/oe1-trabajo/A and ~/oe1-trabajo/B and only WRITES under ~/oe1-trabajo/etapa1
# (created with a plain mkdir: the script aborts if the folder already exists).
#
# Checks (any failure aborts with a non-zero code):
#   * >= 1 GB free in /home;
#   * the set of files the patch touches EXCLUDES the files the plan keeps unchanged: DetectorConstruction.*,
#     CMakeLists.txt, GeometryConfig.*, FluxTables.*, StackingAction.*, GeometryDiagnostics.* and tables/
#     (P2 must not change the geometry, the build, the generator or the kill rules of either variant);
#   * `patch --dry-run --fuzz=0` on both copies, then the real patch;
#   * ONLY the P2-touched files are verified against the shipped sha256 list (CMakeLists.txt of the
#     bundle differs from the repo and A's DetectorConstruction.cc has CRLF: neither is touched);
#   * `diff -ru --strip-trailing-cr A2 B2` lists exactly one file, src/DetectorConstruction.cc, and, when
#     ~/oe1-trabajo/registro/diff-A-B.txt (the pilot's own A/B diff, read only) exists, its hunks are IDENTICAL to
#     those of diff(A2, B2) once the header lines (paths and dates) are masked: P2 changed nothing of the A/B
#     difference.
set -u -o pipefail

SRC="$HOME/oe1-trabajo"
E1="$SRC/etapa1"

die() { echo "prepare_sources.sh: $2" >&2; exit "$1"; }

[[ $# -ge 2 ]] || die 2 "usage: prepare_sources.sh P2.diff P2.sha256 [--base-a FILE --base-b FILE]"
PATCH="$1"; HASHES="$2"; shift 2
BASE_A=""; BASE_B=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-a) BASE_A="${2:?--base-a needs a file}"; shift 2 ;;
    --base-b) BASE_B="${2:?--base-b needs a file}"; shift 2 ;;
    *) die 2 "unknown argument: $1" ;;
  esac
done
[[ -s "$PATCH" && -s "$HASHES" ]] || die 2 "patch or sha256 list missing or empty"
PATCH="$(cd "$(dirname "$PATCH")" && pwd)/$(basename "$PATCH")"
HASHES="$(cd "$(dirname "$HASHES")" && pwd)/$(basename "$HASHES")"
for f in "$BASE_A" "$BASE_B"; do
  if [[ -n "$f" ]]; then [[ -s "$f" ]] || die 2 "$f missing"; fi
done
[[ -d "$SRC/A/simulation/geant4_app" && -d "$SRC/B/simulation/geant4_app" ]] || die 2 "$SRC/{A,B}/simulation/geant4_app not found"

# --- the patch: LF only, strip level, list of touched files -------------------------------------
if LC_ALL=C grep -q $'\r' "$PATCH"; then die 2 "the patch contains CR bytes (must be LF only)"; fi
first="$(awk '/^\+\+\+ /{print $2; exit}' "$PATCH")"
[[ "$first" == *"simulation/geant4_app/"* ]] || die 2 "cannot find simulation/geant4_app in the patch paths (first file: '$first')"
prefix="${first%%simulation/*}"
STRIP="$(awk -v s="$prefix" 'BEGIN { print gsub("/", "/", s) }')"
touched="$(awk '/^\+\+\+ /{print $2}' "$PATCH" | grep -v '^/dev/null$' | awk -v p="$STRIP" '{ n = split($0, a, "/"); out = ""; for (i = p + 1; i <= n; i++) out = out (i > p + 1 ? "/" : "") a[i]; print out }' | LC_ALL=C sort)"
listed="$(awk '{print $2}' "$HASHES" | sed -e 's#^\*##' -e 's#^\./##' | LC_ALL=C sort)"
if [[ "$touched" != "$listed" ]]; then
  echo "files touched by the patch:" >&2; echo "$touched" >&2
  echo "files in the sha256 list:" >&2; echo "$listed" >&2
  die 2 "the sha256 list must cover exactly the files the patch touches"
fi
# The plan keeps these unchanged (step 2, "Sin cambios"): a patch that touches one of them is not P2.
forbidden="$(echo "$touched" | grep -E '/(CMakeLists\.txt|tables/.*|(src|include)/(DetectorConstruction|GeometryConfig|FluxTables|StackingAction|GeometryDiagnostics)\.(cc|hh))$' || true)"
if [[ -n "$forbidden" ]]; then
  echo "files the patch must NOT touch (the plan's unchanged list):" >&2; echo "$forbidden" >&2
  die 2 "the patch touches files that P2 leaves unchanged (DetectorConstruction.*, CMakeLists.txt, GeometryConfig.*, FluxTables.*, StackingAction.*, GeometryDiagnostics.*, tables/)"
fi

# --- quota ----------------------------------------------------------------------------------------
# Prints the free space in /home in whole MB (negative when the quota is exceeded) and returns 1 when quota cannot be
# read (no /home line, no quota, or no answer in 30 s). quota exits 1 whenever a limit is exceeded, so its exit status
# is ignored: under pipefail it would hide the figure that awk printed.
free_mb_home() {
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
free="$(free_mb_home)" || die 2 "cannot read quota -s; check the free space in /home by hand (need >= 1 GB)"
if (( free < 1024 )); then die 2 "only $free MB free in /home (need >= 1024 MB)"; fi

# --- workspace: plain mkdir, never -p -------------------------------------------------------------
mkdir "$E1" 2>/dev/null || die 3 "$E1 already exists: refusing to reuse it (pick a review-safe new location or remove it by hand)"
for d in fuentes build pruebas control-geometrico listas verificacion corridas slurm registro herramientas analisis; do
  mkdir "$E1/$d" || die 3 "cannot create $E1/$d"
done
REG="$E1/registro"
LOG="$REG/preparacion-fuentes.txt"
{
  echo "fecha_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "host=$(hostname)"
  echo "libre_en_home_mb=$free"
  echo "parche=$PATCH"
  echo "parche_sha256=$(sha256sum "$PATCH" | cut -d' ' -f1)"
  echo "lista_sha256_de_P2=$HASHES"
  echo "lista_sha256_de_P2_sha256=$(sha256sum "$HASHES" | cut -d' ' -f1)"
  echo "nivel_de_recorte_patch=-p$STRIP"
  echo "archivos_que_toca_P2:"; echo "$touched" | sed 's/^/  /'
} > "$LOG"
cp -p "$PATCH" "$REG/P2-instrumentacion.diff"
cp -p "$HASHES" "$REG/P2-instrumentacion.sha256"

# --- copies of the pilot trees --------------------------------------------------------------------
cp -a "$SRC/A" "$E1/fuentes/A2" || die 3 "cannot copy A"
cp -a "$SRC/B" "$E1/fuentes/B2" || die 3 "cannot copy B"
diff -rq "$SRC/A" "$E1/fuentes/A2" >/dev/null || die 3 "the copy of A differs from ~/oe1-trabajo/A"
diff -rq "$SRC/B" "$E1/fuentes/B2" >/dev/null || die 3 "the copy of B differs from ~/oe1-trabajo/B"
# The delivery's lists name the files "<letter>/simulation/..." (relative to ~/oe1-trabajo); the copies
# A2/B2 have no leading letter, so it is dropped before the check inside the copy. A line that does not
# start with "<letter>/" stays as it is and therefore fails the check (file not found).
verify_base() {   # verify_base LETTER LIST
  local v="$1" list="$2"
  sed -e "s#^\([0-9a-f]\{64\}\)  \*\{0,1\}$v/#\1  #" "$list" > "$REG/base-$v.sha256"
  [[ -s "$REG/base-$v.sha256" ]] || die 5 "$list is empty"
  (cd "$E1/fuentes/${v}2" && sha256sum -c "$REG/base-$v.sha256") > "$REG/verificacion-base-$v.txt" 2>&1 \
    || die 5 "$v does not match $list (see $REG/verificacion-base-$v.txt)"
}
if [[ -n "$BASE_A" ]]; then verify_base A "$BASE_A"; fi
if [[ -n "$BASE_B" ]]; then verify_base B "$BASE_B"; fi

# --- patch: dry run on both, then for real --------------------------------------------------------
for V in A2 B2; do
  (cd "$E1/fuentes/$V" && patch -p"$STRIP" --dry-run --fuzz=0 --no-backup-if-mismatch < "$PATCH") > "$REG/patch-dryrun-$V.txt" 2>&1 \
    || die 4 "patch --dry-run failed on $V (see $REG/patch-dryrun-$V.txt); nothing was patched"
done
for V in A2 B2; do
  (cd "$E1/fuentes/$V" && patch -p"$STRIP" --fuzz=0 --no-backup-if-mismatch < "$PATCH") > "$REG/patch-$V.txt" 2>&1 \
    || die 4 "patch failed on $V (see $REG/patch-$V.txt)"
done
if find "$E1/fuentes" \( -name '*.orig' -o -name '*.rej' \) | grep -q .; then die 4 "patch left .orig or .rej files"; fi

# --- verification: ONLY the P2-touched files --------------------------------------------------------
for V in A2 B2; do
  (cd "$E1/fuentes/$V" && sha256sum -c "$HASHES") > "$REG/verificacion-P2-$V.txt" 2>&1 \
    || die 5 "P2-touched files of $V differ from the expected sha256 (see $REG/verificacion-P2-$V.txt)"
done

# --- record --------------------------------------------------------------------------------------------
for V in A2 B2; do
  (cd "$E1/fuentes/$V" && find . -type f | LC_ALL=C sort | xargs sha256sum) > "$REG/$V-fuentes.sha256"
done
diff -ru --strip-trailing-cr "$E1/fuentes/A2" "$E1/fuentes/B2" > "$REG/diff-A2-B2.txt"
rc=$?
listed_files="$(grep -c '^diff -ru' "$REG/diff-A2-B2.txt")"
only_in="$(grep -c '^Only in' "$REG/diff-A2-B2.txt")"
if [[ $rc -ne 1 || "$listed_files" != 1 || "$only_in" != 0 ]] || ! grep '^diff -ru' "$REG/diff-A2-B2.txt" | grep -q 'src/DetectorConstruction\.cc$'; then
  die 5 "diff -ru A2 B2 must list only src/DetectorConstruction.cc (rc=$rc, files=$listed_files, only_in=$only_in; see $REG/diff-A2-B2.txt)"
fi
# The pilot's own diff of A and B (read only): after P2 the difference between the two trees must be exactly the
# same hunks. The header of each file (the `diff -ru` line and the two lines after it: paths and dates) is masked.
PILOT_DIFF="$SRC/registro/diff-A-B.txt"
mask_diff() { awk '/^diff /{ h = 2; next } h > 0 { h--; next } { print }' "$1"; }
if [[ -f "$PILOT_DIFF" ]]; then
  mask_diff "$PILOT_DIFF" > "$REG/diff-A-B-piloto.enmascarado.txt"
  mask_diff "$REG/diff-A2-B2.txt" > "$REG/diff-A2-B2.enmascarado.txt"
  if ! diff "$REG/diff-A-B-piloto.enmascarado.txt" "$REG/diff-A2-B2.enmascarado.txt" > "$REG/comparacion-diff-A-B.txt"; then
    die 5 "the hunks of diff(A2, B2) differ from those of the pilot's diff-A-B.txt (see $REG/comparacion-diff-A-B.txt)"
  fi
  DIFF_PILOTO="identico al diff-A-B.txt del piloto (encabezados enmascarados; sha256 del enmascarado $(sha256sum "$REG/diff-A2-B2.enmascarado.txt" | cut -d' ' -f1))"
else
  DIFF_PILOTO="no comparado: no existe $PILOT_DIFF"
fi
{
  echo "resultado=OK"
  echo "A2_fuentes_sha256=$(sha256sum "$REG/A2-fuentes.sha256" | cut -d' ' -f1)"
  echo "B2_fuentes_sha256=$(sha256sum "$REG/B2-fuentes.sha256" | cut -d' ' -f1)"
  echo "diff_A2_B2=solo src/DetectorConstruction.cc"
  echo "diff_A2_B2_frente_al_del_piloto=$DIFF_PILOTO"
} >> "$LOG"
echo "OK: $E1 ready (A2 and B2 patched with P2, -p$STRIP; diff A2/B2 = DetectorConstruction.cc only; $DIFF_PILOTO)"
