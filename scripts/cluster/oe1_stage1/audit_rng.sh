#!/usr/bin/env bash
# OE1 stage 1, step 2 closing condition (review suggestion S2): static audit of the code P2 ADDS.
#
# PROT:17 says that logging must not consume random numbers, and the transport of A and B must not
# change because of P2. So the code that P2 adds may not draw a random number, query the navigator or
# the safety helper, and may touch the random engine only to RESEED it and to READ its state.
#
#   audit_rng.sh --diff P2-instrumentacion.diff [--build-dir DIR | --objects OBJ...]
#   audit_rng.sh --base PILOT_APP_DIR [--app APP_DIR] [--build-dir DIR | --objects OBJ...]
#   audit_rng.sh --files FILE... [--objects OBJ...]
#
#   --diff FILE|-     unified diff of the P2 patch (git diff 9520170 <A2 commit> -- scripts/simulation/geant4_app,
#                     or ~/oe1-trabajo/etapa1/registro/P2-instrumentacion.diff on SC3); only its ADDED lines are
#                     audited, so the legacy sampler (PrimaryGeneratorAction.cc, 7 G4UniformRand per muon) is not.
#   --base DIR        pilot tree (the geant4_app folder of 9520170 / of the pilot bundle); the added lines are
#                     those of `diff -ruN --strip-trailing-cr DIR APP` (no git and no patch file needed).
#   --app DIR         the geant4_app folder to audit against --base (default: ../../simulation/geant4_app
#                     of this script, when it exists).
#   --files FILE...   audit EVERY line of these files (for files that P2 created).
#   --build-dir DIR   also run `nm -C --undefined-only` over the P2 objects of the CMake build DIR.
#   --objects OBJ...  also run `nm -C --undefined-only` over these object files.
#   --include-tests   also audit tests/ (default: skipped, they are not linked into muography_sim).
#
# Audited: muography_sim.cc, src/*.{cc,hh} and include/*.hh (never tests/, tools/ or CMakeLists.txt).
# In each audited line the string literals and the // comments are removed first (a comment that names
# G4UniformRand is listed as information, not as a finding); a /* */ comment fails the audit because
# the script does not understand it.
#
# REFUSED (any hit in code fails): G4UniformRand, flat( , flatArray( , shoot*( , RandFlat, RandGauss,
#   G4QuickRand, G4Navigator, SafetyHelper, and, for the same reason, fire( , RandExponential/Poisson/Bit/
#   Gamma/Binomial/General/Landau/ChiSquare/StudentT/BreitWigner/SkewNormal, ComputeSafety, ComputeStep,
#   LocateGlobalPointAndSetup, G4TransportationManager, GetNavigatorForTracking.
# ALLOWED, explicitly and nothing else:
#   * engine calls setTheSeed, setTheSeeds, getTheEngine, put, name (reseed the worker engine, hash and
#     name its state) on G4Random / HepRandom or on a variable declared as MixMaxRng / HepRandomEngine;
#   * setSeeds and setTheEngine ONLY in muography_sim.cc: the private MixMaxRng of --write-primaries;
#   * the single `fresh.flat()` line of muography_sim.cc: the proof that the legacy sampler drew exactly
#     7 uniforms per muon, on a PRIVATE engine that is discarded (exactly one such line must exist).
# Any other call on those objects, or a new engine variable name, is a finding.
#
# nm mode: the undefined symbols of the P2 objects that match Rand|flat|shoot|Navigator|SafetyHelper|
#   ComputeSafety|LocateGlobal|TransportationManager|Gauss|Exponential|Poisson|Engine must all be one of
#   CLHEP::HepRandom::{createInstance, getTheEngine, setTheSeed, setTheSeeds, setTheEngine}.
#   (`flat` is a virtual call: it never appears as an undefined symbol, so the source audit is the real
#   check; nm only proves that nothing else was linked in.)
#
# BLIND SPOTS. The audit is name-based: it matches identifiers in the added lines and does not parse C++.
# It does NOT detect
#   (a) an implicit draw through the conversion operators of the engine (operator double(), float() and
#       unsigned int() of CLHEP::HepRandomEngine), such as `double u = *engine;` or `double u = fresh;`: it escapes
#       on a DECLARED engine variable (pass 3 only inspects `var.` and `var->`) and through an alias declared
#       with `auto` (a `HepRandomEngine&` reference is caught as an unknown engine variable);
#   (b) draws from non-CLHEP generators, such as `std::mt19937` or `std::rand`.
# PASS is therefore not a proof. The run-time tests T-R, T-M and T-P of pruebas_tecnicas.sbatch cover PROT:17 at
# run time, on the real binary.
#
# Writes nothing. Exit 0 = PASS, 1 = findings, 2 = usage / missing input. Needs bash, awk (POSIX), diff
# and, for nm mode, nm; runs on the SC3 login node and in WSL.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# prints the comment block at the top of this file
usage() { awk 'NR > 1 { if ($0 ~ /^#/) { sub(/^# ?/, ""); print } else exit }' "${BASH_SOURCE[0]}"; }

DIFF=""; BASE=""; APP=""; BUILD=""; INCLUDE_TESTS=0
FILES=(); OBJS=()
mode=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --diff) DIFF="${2:?--diff needs a file}"; shift 2 ;;
    --base) BASE="${2:?--base needs a folder}"; shift 2 ;;
    --app) APP="${2:?--app needs a folder}"; shift 2 ;;
    --build-dir) BUILD="${2:?--build-dir needs a folder}"; shift 2 ;;
    --include-tests) INCLUDE_TESTS=1; shift ;;
    --files) shift; while [[ $# -gt 0 && "$1" != --* ]]; do FILES+=("$1"); shift; done ;;
    --objects) shift; while [[ $# -gt 0 && "$1" != --* ]]; do OBJS+=("$1"); shift; done ;;
    -h|--help) usage; exit 0 ;;
    *) echo "audit_rng.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
n_modes=0
[[ -n "$DIFF" ]] && n_modes=$((n_modes + 1))
[[ -n "$BASE" ]] && n_modes=$((n_modes + 1))
[[ ${#FILES[@]} -gt 0 ]] && n_modes=$((n_modes + 1))
if [[ $n_modes -ne 1 ]]; then
  echo "audit_rng.sh: give exactly one of --diff, --base or --files" >&2
  exit 2
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/audit-rng.XXXXXX")" || exit 2
trap 'rm -rf "$TMP"' EXIT

# --- the stream of lines to audit ---------------------------------------------------------------------
INPUT_KIND=diff
APPDIR=""
if [[ -n "$DIFF" ]]; then
  if [[ "$DIFF" == "-" ]]; then cat > "$TMP/p2.diff"; DIFF="$TMP/p2.diff"; fi
  [[ -f "$DIFF" && -s "$DIFF" ]] || { echo "audit_rng.sh: $DIFF is not a non-empty file" >&2; exit 2; }
  SRC="$DIFF"
elif [[ -n "$BASE" ]]; then
  [[ -z "$APP" && -d "$HERE/../../simulation/geant4_app" ]] && APP="$HERE/../../simulation/geant4_app"
  [[ -d "$BASE" && -d "$APP" ]] || { echo "audit_rng.sh: --base and --app must be folders" >&2; exit 2; }
  BASE="${BASE%/}"; APP="${APP%/}"                 # the paths in the diff must equal $APP/<file>
  APPDIR="$APP"
  # diff exits 1 when the trees differ, which is the normal case here
  diff -ruN --strip-trailing-cr "$BASE" "$APP" > "$TMP/p2.diff"
  rc=$?
  [[ $rc -le 1 ]] || { echo "audit_rng.sh: diff failed (exit $rc)" >&2; exit 2; }
  [[ -s "$TMP/p2.diff" ]] || { echo "audit_rng.sh: $BASE and $APP do not differ: nothing to audit" >&2; exit 2; }
  SRC="$TMP/p2.diff"
else
  INPUT_KIND=files
  for f in "${FILES[@]}"; do [[ -f "$f" ]] || { echo "audit_rng.sh: $f is not a file" >&2; exit 2; }; done
  SRC=""
fi

echo "audit_rng.sh: $(date -u '+%Y-%m-%dT%H:%M:%SZ') host=$(hostname)"
if [[ "$INPUT_KIND" == diff ]]; then
  echo "input: unified diff $SRC sha256=$(sha256sum "$SRC" | cut -d' ' -f1)"
else
  echo "input: whole files (${#FILES[@]})"
fi

read -r -d '' AWK_PROG <<'AWK'
# ---------------------------------------------------------------------------------------------------
function fail(msg) { nfail++; print "FAIL  " msg }
function info(msg) { print "info  " msg }
function okline(msg) { nok++; print "ok    " msg }

function wanted(rel) {
  if (rel ~ /^(muography_sim\.cc|src\/[^\/]+\.(cc|hh)|include\/[^\/]+\.hh)$/) return 1
  if (with_tests && rel ~ /^tests\//) return 1
  return 0
}

function emit(rel, n, text) {
  if (!wanted(rel)) { skipped[rel] = 1; return }
  nl++
  R[nl] = rel; N[nl] = n; T[nl] = text
  audited[rel]++
}

function relpath(p,    i) {
  sub(/\t.*$/, "", p)
  gsub(/^"|"$/, "", p)
  if (appdir != "" && index(p, appdir "/") == 1) return substr(p, length(appdir) + 2)
  i = index(p, "geant4_app/")
  if (i > 0) return substr(p, i + 11)
  return p
}

# ---- front end: unified diff (added lines only) or whole files ----------------------------------------
mode == "diff" {
  if (inhunk && (oldrem > 0 || newrem > 0)) {
    c = substr($0, 1, 1)
    if (c == "+") { if (rel != "") emit(rel, newline, substr($0, 2)); newline++; newrem-- }
    else if (c == "-") { oldrem-- }
    else if (c == "\\") { }
    else { newline++; oldrem--; newrem-- }
    if (oldrem <= 0 && newrem <= 0) inhunk = 0
    next
  }
  inhunk = 0
  if ($0 ~ /^\+\+\+ /) { p = substr($0, 5); rel = (p ~ /^\/dev\/null/) ? "" : relpath(p); next }
  if ($0 ~ /^@@ /) {
    s = $0
    match(s, /-[0-9]+(,[0-9]+)?/); o = substr(s, RSTART + 1, RLENGTH - 1)
    match(s, /\+[0-9]+(,[0-9]+)?/); w = substr(s, RSTART + 1, RLENGTH - 1)
    split(o, op, ","); split(w, wp, ",")
    oldrem = (op[2] == "") ? 1 : op[2] + 0
    newrem = (wp[2] == "") ? 1 : wp[2] + 0
    newline = wp[1] + 0
    inhunk = (oldrem > 0 || newrem > 0)
    next
  }
  next
}
mode == "files" {
  rel = relpath(FILENAME)
  emit(rel, FNR, $0)
  next
}

# ---- the audit ------------------------------------------------------------------------------------------
function build_patterns() {
  np = 0
  add("G4UniformRand", "G4UniformRand")
  add("flat(", "(^|[^A-Za-z0-9_])flat[ \t]*\\(")
  add("flatArray(", "(^|[^A-Za-z0-9_])flatArray[ \t]*\\(")
  add("shoot", "(^|[^A-Za-z0-9_])shoot[A-Za-z]*[ \t]*\\(")
  add("RandFlat", "RandFlat")
  add("RandGauss", "RandGauss")
  add("G4QuickRand", "G4QuickRand")
  add("G4Navigator", "G4Navigator")
  add("SafetyHelper", "SafetyHelper")
  add("fire(", "(^|[^A-Za-z0-9_])fire[ \t]*\\(")
  add("RandExponential|Poisson|Bit|Gamma|Binomial|General|Landau|ChiSquare|StudentT|BreitWigner|SkewNormal",
      "Rand(Exponential|Poisson|Bit|Gamma|Binomial|General|Landau|ChiSquare|StudentT|BreitWigner|SkewNormal)")
  add("ComputeSafety|ComputeStep|LocateGlobalPointAndSetup",
      "(ComputeSafety|ComputeStep|LocateGlobalPointAndSetup)")
  add("G4TransportationManager|GetNavigatorForTracking", "(G4TransportationManager|GetNavigatorForTracking)")
}
function add(label, re) { np++; plabel[np] = label; pre[np] = re }

function strip_line(text, k,    c, pos) {
  c = text
  if (index(c, "/*") > 0) fail(R[k] ":" N[k] ": block comment (/* */): the audit only understands // comments")
  gsub(/"([^"\\]|\\.)*"/, "\"\"", c)
  gsub(/'([^'\\]|\\.)'/, "''", c)
  pos = index(c, "//")
  COMMENT = ""
  if (pos > 0) { COMMENT = substr(c, pos); c = substr(c, 1, pos - 1) }
  return c
}

function count_matches(s, re,    n) {
  n = 0
  while (match(s, re)) { n++; s = substr(s, RSTART + (RLENGTH > 0 ? RLENGTH : 1)) }
  return n
}

function ident_allowed(rel, id) {
  if (id == "put" || id == "name" || id == "setTheSeed" || id == "setTheSeeds" || id == "getTheEngine") return 1
  if ((id == "setSeeds" || id == "setTheEngine") && rel == "muography_sim.cc") return 1
  return 0
}

function check_call(rel, n, tok, id) {
  if (id == "flat") return                       # reported by the refused-token pass
  if (ident_allowed(rel, id)) { okline(rel ":" n ": " tok); return }
  fail(rel ":" n ": call not in the allow-list: " tok)
}

END {
  build_patterns()
  if (nl == 0) { fail("no audited line: the input has no line of muography_sim.cc, src/ or include/"); }
  # pass 1: strip, keep the code part; collect declared engine variables
  for (k = 1; k <= nl; k++) {
    code = strip_line(T[k], k)
    CODE[k] = code; CMT[k] = COMMENT
    s = code
    while (match(s, /(MixMaxRng|HepRandomEngine)[ \t]*[*&]?[ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*[=;({,)]/)) {
      d = substr(s, RSTART, RLENGTH)
      sub(/^(MixMaxRng|HepRandomEngine)[ \t]*[*&]?[ \t]*/, "", d)
      sub(/[ \t]*[=;({,)].*$/, "", d)
      if (d != "") { if (!(d in evar)) evar[d] = R[k] ":" N[k] }
      s = substr(s, RSTART + RLENGTH)
    }
  }
  nev = 0
  for (v in evar) {
    nev++
    if (v != "engine" && v != "fresh" && v != "previous")
      fail(evar[v] ": engine variable '" v "' is not one the audit knows (engine, fresh, previous): extend the audit on purpose")
    else okline(evar[v] ": engine variable '" v "'")
  }
  # pass 2: refused tokens
  for (k = 1; k <= nl; k++) {
    code = CODE[k]; cmt = CMT[k]
    for (i = 1; i <= np; i++) {
      if (code ~ pre[i]) {
        if (plabel[i] == "flat(" && R[k] == "muography_sim.cc" && code ~ /(^|[^A-Za-z0-9_])fresh\.flat[ \t]*\([ \t]*\)/) {
          # every flat( call of the line must be the fresh.flat() one: count both and compare
          allowed_flat += count_matches(code, "(^|[^A-Za-z0-9_])fresh\\.flat[ \t]*\\([ \t]*\\)")
          if (count_matches(code, pre[i]) != count_matches(code, "(^|[^A-Za-z0-9_])fresh\\.flat[ \t]*\\([ \t]*\\)"))
            fail(R[k] ":" N[k] ": a flat( call other than fresh.flat() shares the line: " T[k])
          okline(R[k] ":" N[k] ": the fresh.flat() proof of --write-primaries (private engine, 7 draws per muon)")
        } else {
          fail(R[k] ":" N[k] ": refused token " plabel[i] " in code: " T[k])
        }
      }
      if (cmt ~ pre[i]) { ncmt++; cmtlist = cmtlist "\n        " R[k] ":" N[k] " (" plabel[i] ")" }
    }
  }
  if (audited["muography_sim.cc"] > 0 && allowed_flat != 1)
    fail("muography_sim.cc must hold exactly one fresh.flat() line (the 7-draws proof), found " allowed_flat + 0)
  # pass 3: engine-touching calls
  for (k = 1; k <= nl; k++) {
    code = CODE[k]
    s = code
    while (match(s, /(G4Random|HepRandom)[ \t]*::[ \t]*[A-Za-z_][A-Za-z0-9_]*/)) {
      tok = substr(s, RSTART, RLENGTH); id = tok; sub(/^.*::[ \t]*/, "", id)
      check_call(R[k], N[k], tok, id)
      s = substr(s, RSTART + RLENGTH)
    }
    s = code
    while (match(s, /getTheEngine[ \t]*\([ \t]*\)[ \t]*(->|\.)[ \t]*[A-Za-z_][A-Za-z0-9_]*/)) {
      tok = substr(s, RSTART, RLENGTH); id = tok; sub(/^.*(->|\.)[ \t]*/, "", id)
      check_call(R[k], N[k], tok, id)
      s = substr(s, RSTART + RLENGTH)
    }
    for (v in evar) {
      s = code
      re = "(^|[^A-Za-z0-9_])" v "[ \t]*(->|\\.)[ \t]*[A-Za-z_][A-Za-z0-9_]*"
      while (match(s, re)) {
        tok = substr(s, RSTART, RLENGTH); sub(/^[^A-Za-z0-9_]/, "", tok); id = tok; sub(/^.*(->|\.)[ \t]*/, "", id)
        check_call(R[k], N[k], tok, id)
        s = substr(s, RSTART + RLENGTH)
      }
    }
  }
  # report
  print ""
  print "audited lines per file (added lines in diff mode):"
  nf = 0
  for (f in audited) { nf++; names[nf] = f }
  for (a = 2; a <= nf; a++) {                       # insertion sort: portable, no pipe to sort
    key = names[a]; b = a - 1
    while (b >= 1 && names[b] > key) { names[b + 1] = names[b]; b-- }
    names[b + 1] = key
  }
  for (a = 1; a <= nf; a++) printf "  %5d  %s\n", audited[names[a]], names[a]
  nsk = 0
  for (f in skipped) nsk++
  print "skipped files (not part of the binary, or not C++): " nsk
  if (ncmt > 0) { print "info: refused names mentioned only in comments (" ncmt "):" cmtlist }
  print "allowed engine touches / proofs found: " nok
  print "findings: " nfail + 0
  exit (nfail > 0) ? 1 : 0
}
AWK

awk_status=0
if [[ "$INPUT_KIND" == diff ]]; then
  awk -v mode=diff -v with_tests="$INCLUDE_TESTS" -v appdir="$APPDIR" "$AWK_PROG" "$SRC" || awk_status=$?
else
  awk -v mode=files -v with_tests="$INCLUDE_TESTS" -v appdir="" "$AWK_PROG" "${FILES[@]}" || awk_status=$?
fi

# --- nm mode -----------------------------------------------------------------------------------------------
nm_status=0
if [[ -n "$BUILD" ]]; then
  d="$BUILD/CMakeFiles/muography_sim.dir"
  [[ -d "$d" ]] || { echo "audit_rng.sh: $d not found (is $BUILD a CMake build folder of muography_sim?)" >&2; exit 2; }
  for o in src/EventAction.cc.o src/RunAction.cc.o src/PrimaryList.cc.o src/Oe1Cli.cc.o src/SteppingAction.cc.o \
           src/ActionInitialization.cc.o src/PrimaryGeneratorAction.cc.o muography_sim.cc.o; do
    [[ -f "$d/$o" ]] || { echo "audit_rng.sh: missing object $d/$o" >&2; exit 2; }
    OBJS+=("$d/$o")
  done
fi
if [[ ${#OBJS[@]} -gt 0 ]]; then
  command -v nm > /dev/null || { echo "audit_rng.sh: nm not found" >&2; exit 2; }
  echo ""
  echo "nm -C --undefined-only over ${#OBJS[@]} object files (RNG / navigator related symbols):"
  ALLOWED_SYMS=$'CLHEP::HepRandom::createInstance()\nCLHEP::HepRandom::getTheEngine()\nCLHEP::HepRandom::setTheSeed(long, int)\nCLHEP::HepRandom::setTheSeeds(long const*, int)\nCLHEP::HepRandom::setTheEngine(CLHEP::HepRandomEngine*)'
  for o in "${OBJS[@]}"; do
    [[ -f "$o" ]] || { echo "audit_rng.sh: $o is not a file" >&2; exit 2; }
    syms="$(nm -C --undefined-only "$o" 2>/dev/null | grep -i -E 'Rand|flat|shoot|QuickRand|Navigator|SafetyHelper|ComputeSafety|LocateGlobal|TransportationManager|Gauss|Exponential|Poisson|Engine' | sed 's/^ *U //' | LC_ALL=C sort -u || true)"
    bad=""
    while IFS= read -r s; do
      [[ -z "$s" ]] && continue
      if ! grep -qxF -- "$s" <<< "$ALLOWED_SYMS"; then bad+="$s"$'\n'; fi
    done <<< "$syms"
    name="${o##*/CMakeFiles/muography_sim.dir/}"
    if [[ -z "$bad" ]]; then
      echo "  ok    $name: $(printf '%s\n' "$syms" | grep -c . || true) matching symbols, all allowed"
    else
      echo "  FAIL  $name: symbols outside the allow-list:"
      printf '%s' "$bad" | sed 's/^/          /'
      nm_status=1
    fi
  done
fi

echo ""
if [[ $awk_status -eq 0 && $nm_status -eq 0 ]]; then
  echo "RESULT: PASS (no random number is drawn and no navigator is queried by the code P2 adds)"
  exit 0
fi
echo "RESULT: FAIL (source audit exit $awk_status, nm audit exit $nm_status)"
exit 1
