#!/bin/bash
# OE1 stage 1: archiving after the main job. Nothing is deleted or overwritten.
#
#   bash post_run.sh JOBID NN        (JOBID printed by `sbatch --parsable`, NN = INTENTO of the attempt)
#
# 1. sacct -j JOBID --format=JobID,Partition,AllocCPUS,ReqMem,MaxRSS,Elapsed,TotalCPU,UserCPU,SystemCPU,
#    State,ExitCode,NodeList,Submit,Start,End -P   ->  intento-NN/sacct-JOBID.txt
#    (times are UTC-5; TotalCPU was empty for the pilot jobs, so the per-run recursos.json written by
#    run_timed.py is the authoritative source for CPU time and RSS.)
# 2. gzip -n -9 -k of every *_hist.csv and *_muones.csv (the raw files stay on SC3 until the review);
#    sha256 of the raw and of the compressed file in SHA256SUMS-crudos.txt / SHA256SUMS-gz.txt
#    (relative paths, checkable with `sha256sum -c` from the attempt folder).
# 3. the SLURM output file is copied next to the runs.
# 4. one tar per attempt (compressed files and the small files; the raw big files are left out) and its sha256.
#
# Then, from your machine:  rsync -av sc3-guane:oe1-trabajo/etapa1/corridas/intento-NN.tar* .
# and `sha256sum -c intento-NN.tar.sha256`.
set -u -o pipefail

JOBID="${1:?usage: post_run.sh JOBID NN}"
NN="${2:?usage: post_run.sh JOBID NN}"
E1="$HOME/oe1-trabajo/etapa1"
ATT="$E1/corridas/intento-$NN"
[[ -d "$ATT" ]] || { echo "no such attempt folder: $ATT" >&2; exit 2; }
[[ "$JOBID" =~ ^[0-9]+$ ]] || { echo "JOBID must be numeric" >&2; exit 2; }
for f in "$ATT/sacct-$JOBID.txt" "$ATT/SHA256SUMS-crudos.txt" "$ATT/SHA256SUMS-gz.txt" \
         "$E1/corridas/intento-$NN.tar" "$E1/corridas/intento-$NN.tar.sha256"; do
  [[ ! -e "$f" ]] || { echo "refusing to overwrite $f" >&2; exit 3; }
done

# quota: the compressed copies and the tar need roughly a third of the raw size each
raw_mb="$(find "$ATT" -type f \( -name '*_muones.csv' -o -name '*_hist.csv' \) -printf '%s\n' 2>/dev/null | awk '{ s += $1 } END { printf "%d\n", s / 1048576 }')"
free_mb="$(timeout 30 quota -s 2>/dev/null | awk '
  function mb(s,  n, u) { sub(/\*$/, "", s); n = s + 0; u = substr(s, length(s)); if (u == "K") return n / 1024; if (u == "M") return n
    if (u == "G") return n * 1024; if (u == "T") return n * 1024 * 1024; return n / 1024 }
  /\/home/ { if (NF >= 3) { used = $2; lim = $3 } else { getline; used = $1; lim = $2 }; printf "%d\n", mb(lim) - mb(used); exit }')"
echo "raw big files: ${raw_mb:-?} MB; free in /home: ${free_mb:-?} MB"
if [[ -n "$free_mb" && "$free_mb" -lt 512 ]]; then echo "less than 512 MB free: free space first (nothing is deleted here)" >&2; exit 4; fi

# 1. accounting
sacct -j "$JOBID" --format=JobID,Partition,AllocCPUS,ReqMem,MaxRSS,Elapsed,TotalCPU,UserCPU,SystemCPU,State,ExitCode,NodeList,Submit,Start,End -P > "$ATT/sacct-$JOBID.txt" 2>&1
if [[ "$(wc -l < "$ATT/sacct-$JOBID.txt")" -lt 2 ]]; then echo "sacct returned no rows for $JOBID (see $ATT/sacct-$JOBID.txt)" >&2; fi
state="$(awk -F'|' 'NR > 1 && $1 == "'"$JOBID"'" { print $10 " " $11; exit }' "$ATT/sacct-$JOBID.txt")"
echo "sacct: job $JOBID state/exit: ${state:-unknown}"

# 2. compress, keep the raw file, hash both
cd "$ATT" || exit 2
: > SHA256SUMS-crudos.txt; : > SHA256SUMS-gz.txt
rc=0
while IFS= read -r f; do
  f="${f#./}"
  if [[ -e "$f.gz" ]]; then echo "refusing to overwrite $f.gz" >&2; rc=1; continue; fi
  sha256sum -- "$f" >> SHA256SUMS-crudos.txt
  gzip -n -9 -k -- "$f" || { echo "gzip failed on $f" >&2; rc=1; continue; }
  sha256sum -- "$f.gz" >> SHA256SUMS-gz.txt
done < <(find . -type f \( -name '*_hist.csv' -o -name '*_muones.csv' \) | LC_ALL=C sort)
echo "compressed files: $(wc -l < SHA256SUMS-gz.txt), raw files hashed: $(wc -l < SHA256SUMS-crudos.txt)"
sha256sum -c SHA256SUMS-crudos.txt > /dev/null || { echo "raw files changed while archiving" >&2; rc=1; }

# 3. SLURM output
for o in "$E1"/slurm/*-"$JOBID".out; do
  if [[ -f "$o" && ! -e "$ATT/$(basename "$o")" ]]; then cp -p "$o" "$ATT/"; fi
done

# 4. tar
cd "$E1/corridas" || exit 2
tar -cf "intento-$NN.tar" --exclude='*_hist.csv' --exclude='*_muones.csv' "intento-$NN" || { echo "tar failed" >&2; rc=1; }
sha256sum "intento-$NN.tar" > "intento-$NN.tar.sha256"
ls -l "intento-$NN.tar"
cat "intento-$NN.tar.sha256"
exit $rc
