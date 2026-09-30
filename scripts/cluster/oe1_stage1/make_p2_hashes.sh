#!/bin/bash
# OE1 stage 1: writes the sha256 list that ships with the P2 patch (input of prepare_sources.sh).
# Run it LOCALLY, in the repository at the A2 commit, right after producing the patch:
#
#   git diff 9520170 <A2 commit> -- scripts/simulation/geant4_app > P2-instrumentacion.diff
#   bash make_p2_hashes.sh P2-instrumentacion.diff <repo root> P2-instrumentacion.sha256
#
# For every file the patch touches it hashes the file as it is in <repo root> and writes
# "<sha256>  simulation/geant4_app/<path>" (paths as they appear in the SC3 bundle: the leading
# "scripts/" of the repository is dropped). Deleted files cannot be listed and are refused.
set -u -o pipefail
PATCH="${1:?usage: make_p2_hashes.sh P2.diff REPO_ROOT OUT}"
ROOT="${2:?usage: make_p2_hashes.sh P2.diff REPO_ROOT OUT}"
OUT="${3:?usage: make_p2_hashes.sh P2.diff REPO_ROOT OUT}"
[[ -s "$PATCH" && -d "$ROOT" ]] || { echo "patch or repository root missing" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
if grep -q '^+++ /dev/null' "$PATCH"; then echo "the patch deletes a file: not supported" >&2; exit 2; fi

first="$(awk '/^\+\+\+ /{print $2; exit}' "$PATCH")"
prefix="${first%%simulation/*}"                       # b/scripts/
gitstrip="$(awk -v s="${prefix%%scripts/*}" 'BEGIN { print gsub("/", "/", s) }')"   # components of the git prefix (b/)
awk '/^\+\+\+ /{print $2}' "$PATCH" | LC_ALL=C sort | while read -r path; do
  repo_path="$(echo "$path" | awk -v p="$gitstrip" '{ n = split($0, a, "/"); out = ""; for (i = p + 1; i <= n; i++) out = out (i > p + 1 ? "/" : "") a[i]; print out }')"
  sc3_path="simulation/${path#*simulation/}"
  [[ -f "$ROOT/$repo_path" ]] || { echo "missing $ROOT/$repo_path" >&2; exit 2; }
  printf '%s  %s\n' "$(sha256sum "$ROOT/$repo_path" | cut -d' ' -f1)" "$sc3_path"
done > "$OUT"
status=${PIPESTATUS[2]}
if [[ $status -ne 0 || ! -s "$OUT" ]]; then echo "failed" >&2; rm -f "$OUT"; exit 2; fi
echo "wrote $OUT ($(wc -l < "$OUT") files)"
