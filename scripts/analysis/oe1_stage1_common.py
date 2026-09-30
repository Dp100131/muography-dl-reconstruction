#!/usr/bin/env python3
"""OE1 stage 1: helpers shared by the oe1_stage1_* tools.

Runs on Python >= 3.9 and numpy >= 1.23 (the SC3 login node has 3.9.25 and
numpy 1.23.5; compute nodes reported numpy 2.0.2). SciPy is never required: the
few special functions the statistics need (chi-square and Kolmogorov tail
probabilities, exact binomial sums, Tango's score interval) are implemented
here with ``math.lgamma`` and are checked against SciPy in the unit tests.

Contents
  * constants of the Geant4 application (grid, FNV, tags, CSV columns);
  * hashing (SHA-256, FNV-1a-64) and the canonical hash of a primary list;
  * strict readers for the primary list (same defects as the C++
    ``--check-primaries``), the per-muon log ``<stem>_muones.csv``,
    ``hist.csv``, ``meta.json`` and ``run.log``;
  * the pixel convention of ``SteppingAction.cc`` (``binIndex``, ``relAz``);
  * the R_int mask text format read by ``RoiMask.hh``;
  * statistics: chi-square, KS, binomial, exact McNemar, Tango score interval;
  * a loader for ``crossing_reference.py`` (reused, never duplicated).
"""
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys

import numpy as np

# --------------------------------------------------------------------------
# Constants (mirror include/GeometryConfig.hh, Oe1Hash.hh, Oe1ListMode.hh)
# --------------------------------------------------------------------------
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = 0xFFFFFFFFFFFFFFFF

TAG_TRANSPORT = 0x4F455431
TAG_SOURCE = 0x4F455331

LIST_FORMAT = "oe1-primarios/1"
LIST_HEADER = "id,x_m,y_m,z_m,dx,dy,dz,ekin_gev,pdg,charge_e"
NORM_TOLERANCE = 1.0e-12

N_EL = 256
N_AZ = 256
AZ_HALF_DEG = 30.0
EL_MIN_DEG = 2.0
EL_MAX_DEG = 47.0
DET_SIDE_M = 1.2
DET_DIST_M = 1000.0
DET_HEIGHT_M = 1.5
VIEW_STEP_DEG = 60.0
MUON_MASS_GEV = 0.1056583745

# Frozen R_int (pilot delivery 712c786, see PLAN step 0/6).
FROZEN_NPZ_SHA256 = "7546773bc4a5e4e13c8a1ada843c0b6f9559d238c5b6f1a6bcbbbac6d0907d8c"
FROZEN_RINT_CONTENT_SHA256 = "3029f266419a6a156ecf809a2967bf1b9b72349c6a95e4697115a740140cce63"
# SHA-256 of the exported text mask (256 lines of 256 '0'/'1' characters, LF): the file the binary loads
# (--roi-mask-sha256) and the one the analysis reads. Hard-coded so that an export that drifted from the plan
# cannot pass just because its own JSON says so.
FROZEN_RINT_TEXT_SHA256 = "e76dcbe5926f9f4aeb61c556729d15c64124cd50a8577dca6beee9ffa317c5fd"
FROZEN_RINT_SHAPE = (256, 256)
FROZEN_RINT_PIXELS = 195
FROZEN_RINT_ROWS = (17, 50)
FROZEN_RINT_COLS = (131, 138)
FROZEN_RINT_PROVENANCE_COMMIT = "712c786"
FROZEN_RINT_PROVENANCE_PATH = (
    "objetivo-1/entregas/2026-09-29-geo-train-06/analisis/roi-mascaras.npz")

STATUS_NAMES = ("DET", "PLANE_OUT", "WORLD_EXIT", "DECAY", "MU_CAPTURE",
                "STOPPED", "OTHER", "ABORTED")

MUON_COLUMNS = (
    "caso", "geometry_id", "view", "variant", "replica", "list_sha256", "id",
    "prim_fnv", "rng0", "rng1", "detected", "iel", "iaz", "in_rint",
    "ekin_plane_gev", "el_plane_deg", "relaz_plane_deg", "status",
    "end_process", "n_steps", "end_x_m", "end_y_m", "end_z_m", "end_vol",
    "crossed", "lz_m", "lz_true_m", "lz_rock_m", "lz_a0_m", "lz_a1_m",
    "lz_other_m", "lc_m", "le_m", "rho_lz_gcm2", "n_steps_z",
    "n_steps_z_both_out", "n_entries_z", "n_graze_z", "mat_in", "mat_out",
    "e_in_gev", "e_in_lo_gev", "e_in_hi_gev", "e_out_gev", "e_out_lo_gev",
    "e_out_hi_gev", "max_lat_mm")
MUON_HEADER = ",".join(MUON_COLUMNS)

# Columns of the corrected binary (PLAN correction C1, 2026-09-30): the lateral displacement that
# WentzelVI applies inside a step makes the pre->post chord differ from the navigated path, so the
# binary records the largest displacement and the summed displacement of the steps near Z (chord
# within 1 m of both primitives; include/RegionCrossing.hh). sum_lat_z_mm is a SCALE of the error of the
# length in Z, not a bound (at incidence angle theta the error is ~ lat / sin(theta); for a chord that
# skims a surface it grows like sqrt(2 R lat), R <= 86.3 m). The C++ binary always writes them, after
# max_lat_mm (49 columns; empty when the log is `min` or the run has no region) and declares
# counts.max_lat_z_mm / max_sum_lat_z_mm in the meta.
# The parser accepts them at any position (the base columns keep their order) so that the log of an
# older binary is still readable; K6 makes them mandatory when the meta declares the aggregates, and
# otherwise reports "not verifiable".
MUON_OPTIONAL_COLUMNS = ("max_lat_z_mm", "sum_lat_z_mm")

# column groups of the muones log
_MU_STR = ("geometry_id", "variant", "list_sha256", "prim_fnv", "rng0", "rng1",
           "status", "end_process", "end_vol", "mat_in", "mat_out")
_MU_INT = ("caso", "view", "replica", "id", "detected", "iel", "iaz", "in_rint",
           "n_steps")
# everything else is float64 and may be empty (NaN)

# Durations and resource use of a run (PLAN W3: getrusage CPU and RSS go in the meta, outside the A/A
# comparison). They differ between two otherwise identical runs, so the A/A and A/B comparisons ignore
# EXACTLY these top-level keys of meta.json (the ones RunAction.cc writes) and list the ones they ignored as
# evidence. An explicit allow-list on purpose: a substring rule ("second", "cpu", "user", ...) could hide a
# real difference in some other key. rusage_max_rss_kb is in kilobytes.
TIMING_META_KEYS = ("wall_seconds", "seconds_per_event", "rusage_user_cpu_seconds",
                    "rusage_system_cpu_seconds", "rusage_max_rss_kb")


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fnv1a64(data, h=FNV_OFFSET):
    """FNV-1a 64-bit over ``data`` (bytes), continuing from state ``h``."""
    prime = FNV_PRIME
    mask = MASK64
    for b in data:
        h = ((h ^ b) * prime) & mask
    return h


def hex16(v):
    return "%016x" % (v & MASK64)


def to_py(obj):
    """Recursively converts numpy scalars/arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    return obj


def dumps_json(obj):
    return json.dumps(to_py(obj), indent=1, ensure_ascii=False, sort_keys=False) + "\n"


def write_text_new(path, text):
    """Writes ``text`` (UTF-8, LF) to a file that must not exist yet."""
    with open(path, "xb") as fh:
        fh.write(text.encode("utf-8"))


def utf8_streams():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def run_command(fn, *args):
    """Runs a command-line action. A missing, unreadable or already existing file becomes one line on
    stderr and exit code 2 (unusable input) instead of a traceback; anything else still propagates."""
    try:
        return fn(*args)
    except (FileNotFoundError, FileExistsError, NotADirectoryError, IsADirectoryError, PermissionError) as exc:
        sys.stderr.write("entrada o salida inutilizable: %s\n" % exc)
        return 2


def _loaded_version(name):
    """Version of an optional package only if it is already imported (importing
    scipy or matplotlib just to print a version costs about a second)."""
    mod = sys.modules.get(name)
    return getattr(mod, "__version__", None) if mod is not None else "no usado"


def env_versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": _loaded_version("scipy"), "matplotlib": _loaded_version("matplotlib"),
            "plataforma": "%s %d-bit" % (sys.platform, 64 if sys.maxsize > 2 ** 32 else 32)}


def script_hashes(paths):
    return {os.path.basename(p): sha256_file(p) for p in paths if os.path.isfile(p)}


def here():
    return os.path.dirname(os.path.abspath(__file__))


def tool_paths():
    """Files of this tool family (for the provenance hashes)."""
    d = here()
    names = sorted(n for n in os.listdir(d)
                   if n.startswith("oe1_stage1_") and n.endswith(".py"))
    return [os.path.join(d, n) for n in names]


# --------------------------------------------------------------------------
# Canonical hash and strict reader of the primary list
# --------------------------------------------------------------------------
_CANON_DTYPE = np.dtype([("id", "<i8"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                         ("dx", "<f8"), ("dy", "<f8"), ("dz", "<f8"),
                         ("ekin", "<f8"), ("pdg", "<i4")])


def canonical_fnv(ids, values, pdg):
    """FNV-1a-64 over little-endian int64 id, 7 float64 values, int32 pdg.

    Same layout as ``CanonicalFnv`` in src/PrimaryList.cc (Python:
    ``struct.pack('<q7di', ...)``). ``values`` has shape (n, 7): x, y, z, dx,
    dy, dz, ekin_gev.
    """
    n = len(ids)
    rec = np.empty(n, dtype=_CANON_DTYPE)
    rec["id"] = ids
    for k, name in enumerate(("x", "y", "z", "dx", "dy", "dz", "ekin")):
        rec[name] = values[:, k]
    rec["pdg"] = pdg
    assert rec.dtype.itemsize == 68
    return fnv1a64(rec.tobytes())


# The 60 bytes that EventAction.cc (PrimaryFnv) hashes at BeginOfEventAction, all little-endian: x, y, z
# (float64, mm), dx, dy, dz (float64), ekin (float64, MeV), pdg (int32).
_PRIMARY_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("dx", "<f8"), ("dy", "<f8"),
                           ("dz", "<f8"), ("ekin", "<f8"), ("pdg", "<i4")])


def fnv1a64_rows(byte_rows):
    """FNV-1a-64 of every row of a 2-D uint8 array, at once (uint64 arithmetic wraps modulo 2**64)."""
    rows = np.asarray(byte_rows, dtype=np.uint8)
    h = np.full(rows.shape[0], FNV_OFFSET, dtype=np.uint64)
    prime = np.uint64(FNV_PRIME)
    for k in range(rows.shape[1]):
        h ^= rows[:, k].astype(np.uint64)
        h *= prime
    return h


def primary_fnv_array(vals, pdg):
    """``prim_fnv`` of every primary of a list (uint64 array), recomputed from the list rows.

    Same recipe as ``PrimaryFnv`` in src/EventAction.cc and the README: position x 1000.0 (m -> mm), the
    direction times ``1.0 / sqrt(dx*dx + dy*dy + dz*dz)`` (CLHEP ``unit()``: each component times the reciprocal
    of the norm; dividing by the norm gives other last bits), ekin x 1000.0 (GeV -> MeV) and the PDG code as an
    int32, hashed as little-endian bytes. ``vals`` is (n, 7): x_m, y_m, z_m, dx, dy, dz, ekin_gev. Every
    arithmetic step is a separate numpy operation so that it rounds exactly as the scalar C++ expression.
    """
    vals = np.asarray(vals, dtype=np.float64)
    n = vals.shape[0]
    rec = np.empty(n, dtype=_PRIMARY_DTYPE)
    for k, name in enumerate(("x", "y", "z")):
        rec[name] = vals[:, k] * 1000.0
    dx, dy, dz = vals[:, 3], vals[:, 4], vals[:, 5]
    norm2 = dx * dx
    norm2 = norm2 + dy * dy
    norm2 = norm2 + dz * dz
    inv = 1.0 / np.sqrt(norm2)
    rec["dx"] = dx * inv
    rec["dy"] = dy * inv
    rec["dz"] = dz * inv
    rec["ekin"] = vals[:, 6] * 1000.0
    rec["pdg"] = np.asarray(pdg).astype(np.int32)
    assert rec.dtype.itemsize == 60
    return fnv1a64_rows(rec.view(np.uint8).reshape(n, 60))


_HEX_LUT = np.full(256, 255, dtype=np.uint8)
_HEX_LUT[ord("0"):ord("9") + 1] = np.arange(10, dtype=np.uint8)
_HEX_LUT[ord("a"):ord("f") + 1] = np.arange(10, 16, dtype=np.uint8)
_HEX_LUT[ord("A"):ord("F") + 1] = np.arange(10, 16, dtype=np.uint8)


def hex16_column_to_uint64(col):
    """(values, valid): a column of 16-digit hex strings (numpy ``S`` array) as uint64. ``valid`` is False for a
    value that is not exactly 16 hexadecimal digits (its uint64 is then meaningless)."""
    col = np.asarray(col)
    n = len(col)
    if n == 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=bool)
    fixed = col.astype("S16")
    ok_len = np.char.str_len(col) == 16
    digits = _HEX_LUT[np.frombuffer(fixed.tobytes(), dtype=np.uint8).reshape(n, 16)]
    valid = ok_len & (digits != 255).all(axis=1)
    out = np.zeros(n, dtype=np.uint64)
    for k in range(16):
        out = (out << np.uint64(4)) | digits[:, k].astype(np.uint64)
    return out, valid


_NUM = (rb"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        rb"|-?(?:[iI][nN][fF](?:[iI][nN][iI][tT][yY])?|[nN][aA][nN])")
_INT = rb"-?\d+"
_LIST_LINE = re.compile(
    rb"^(" + _INT + rb")," + rb",".join([rb"(" + _NUM + rb")"] * 7) +
    rb",(" + _INT + rb"),(" + _INT + rb")$")


def parse_primary_bytes(data):
    """Parses list text like ``ParsePrimaryText`` + ``ValidatePrimaryRecords``.

    Returns ``(table, defects)``. ``table`` holds arrays ``id``, ``vals``
    (n, 7), ``pdg``, ``charge`` for the rows that parse. ``defects`` uses the
    names of the C++ ``--check-primaries`` line (snake case).
    """
    d = {
        "lf_only": 1 if b"\r" not in data else 0,
        "header_ok": 0,
        "ends_with_newline": 1 if (len(data) > 0 and data.endswith(b"\n")) else 0,
        "bad_rows": 0, "id_min": 0, "id_max": -1, "ids_contiguous": 1,
        "dup_ids": 0, "dup_states": 0, "nonfinite": 0, "norm_fail": 0,
        "max_norm_dev": 0.0, "dz_nonneg": 0, "ekin_nonpos": 0, "pdg_bad": 0,
        "charge_mismatch": 0, "n_mu_plus": 0, "n_mu_minus": 0}
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines.pop()
    if lines:
        d["header_ok"] = 1 if lines[0] == LIST_HEADER.encode("ascii") else 0
        body = lines[1:]
    else:
        body = []
    groups = []
    for line in body:
        m = _LIST_LINE.match(line)
        if m is None:
            d["bad_rows"] += 1
        else:
            groups.append(m.groups())
    n = len(groups)
    if n:
        cols = list(zip(*groups))
        ids = np.array(cols[0]).astype(np.int64)
        vals = np.empty((n, 7), dtype=np.float64)
        for k in range(7):
            vals[:, k] = np.array(cols[1 + k]).astype(np.float64)
        pdg = np.array(cols[8]).astype(np.int64)
        charge = np.array(cols[9]).astype(np.int64)
    else:
        ids = np.zeros(0, dtype=np.int64)
        vals = np.zeros((0, 7), dtype=np.float64)
        pdg = np.zeros(0, dtype=np.int64)
        charge = np.zeros(0, dtype=np.int64)
    table = {"id": ids, "vals": vals, "pdg": pdg, "charge": charge}
    if n:
        d["id_min"] = int(ids.min())
        d["id_max"] = int(ids.max())
        d["ids_contiguous"] = 1 if bool(np.array_equal(ids, np.arange(n))) else 0
        s = np.sort(ids)
        d["dup_ids"] = int((s[1:] == s[:-1]).sum())
        finite = np.isfinite(vals).all(axis=1)
        d["nonfinite"] = int((~finite).sum())
        with np.errstate(invalid="ignore"):
            v = vals[finite]
            norm = np.sqrt(v[:, 3] * v[:, 3] + v[:, 4] * v[:, 4] + v[:, 5] * v[:, 5])
            dev = np.abs(norm - 1.0)
            if len(dev):
                d["max_norm_dev"] = float(dev.max())
            d["norm_fail"] = int((dev > NORM_TOLERANCE).sum())
            d["dz_nonneg"] = int((~(v[:, 5] < 0.0)).sum())
            d["ekin_nonpos"] = int((~(v[:, 6] > 0.0)).sum())
        is_minus = pdg == 13
        is_plus = pdg == -13
        d["n_mu_minus"] = int(is_minus.sum())
        d["n_mu_plus"] = int(is_plus.sum())
        d["pdg_bad"] = int((~(is_minus | is_plus)).sum())
        d["charge_mismatch"] = int(((is_minus & (charge != -1)) |
                                    (is_plus & (charge != 1))).sum())
        # exact duplicate initial states (bit patterns), independent of the id
        bits = np.empty((n, 8), dtype=np.int64)
        bits[:, :7] = np.ascontiguousarray(vals).view(np.int64)
        bits[:, 7] = pdg
        order = np.lexsort(bits.T[::-1])
        sb = bits[order]
        d["dup_states"] = int((sb[1:] == sb[:-1]).all(axis=1).sum())
    return table, d


def primary_defects_clean(d, n):
    return (n > 0 and d["lf_only"] == 1 and d["header_ok"] == 1 and
            d["ends_with_newline"] == 1 and d["bad_rows"] == 0 and
            d["ids_contiguous"] == 1 and d["dup_ids"] == 0 and d["nonfinite"] == 0 and
            d["norm_fail"] == 0 and d["dz_nonneg"] == 0 and d["ekin_nonpos"] == 0 and
            d["pdg_bad"] == 0 and d["charge_mismatch"] == 0 and d["dup_states"] == 0)


def read_primary_file(path):
    """Returns ``(table, defects, file_sha256, canonical_fnv_int)``."""
    with open(path, "rb") as fh:
        data = fh.read()
    table, d = parse_primary_bytes(data)
    canon = canonical_fnv(table["id"], table["vals"], table["pdg"])
    return table, d, sha256_bytes(data), canon


def sidecar_path(list_path):
    return list_path + ".meta.json"


def read_sidecar(list_path):
    """Returns ``(dict or None, error text or None)``."""
    p = sidecar_path(list_path)
    if not os.path.isfile(p):
        return None, "missing sidecar " + p
    try:
        with open(p, "rb") as fh:
            return json.loads(fh.read().decode("utf-8")), None
    except Exception as exc:  # malformed json
        return None, "invalid sidecar %s: %s" % (p, exc)


def format_check_line(name, n, d, canon, file_sha, sc, sc_err):
    """One ``OE1-CHECK-PRIMARIES`` line like the binary prints (best effort)."""
    have = sc is not None
    format_ok = have and sc.get("format") == LIST_FORMAT
    n_match = have and int(sc.get("n", -1)) == n
    fnv_match = have and str(sc.get("canon_fnv1a64", "")).lower() == hex16(canon)
    sha_checked = have and bool(sc.get("file_sha256"))
    sha_match = sha_checked and sc.get("file_sha256") == file_sha
    ok = (primary_defects_clean(d, n) and format_ok and n_match and fnv_match and
          (not sha_checked or sha_match))
    return ok, (
        "OE1-CHECK-PRIMARIES file=%s n=%d lf_only=%d header_ok=%d ends_with_newline=%d "
        "bad_rows=%d id_min=%d id_max=%d ids_contiguous=%d dup_ids=%d dup_states=%d "
        "nonfinite=%d norm_fail=%d max_norm_dev=%s dz_nonneg=%d ekin_nonpos=%d "
        "pdg_bad=%d charge_mismatch=%d n_mu_plus=%d n_mu_minus=%d canon_fnv1a64=%s "
        "file_sha256=%s sidecar=%s sidecar_n=%s sidecar_fnv_match=%d "
        "sidecar_sha_match=%s status=%s" % (
            name, n, d["lf_only"], d["header_ok"], d["ends_with_newline"],
            d["bad_rows"], d["id_min"], d["id_max"], d["ids_contiguous"],
            d["dup_ids"], d["dup_states"], d["nonfinite"], d["norm_fail"],
            to_chars_like(d["max_norm_dev"]), d["dz_nonneg"], d["ekin_nonpos"], d["pdg_bad"],
            d["charge_mismatch"], d["n_mu_plus"], d["n_mu_minus"], hex16(canon),
            file_sha,
            ("ok" if format_ok else "bad-format") if have else "missing",
            sc.get("n") if have else "na", 1 if fnv_match else 0,
            ("1" if sha_match else "0") if sha_checked else "na",
            "OK" if ok else "DEFECT"))


def parse_kv_line(line):
    """``key=value`` tokens of a summary line (OE1-CHECK-PRIMARIES, OE1-ETAPA1)."""
    out = {}
    for tok in line.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def to_chars_like(v):
    """``repr`` without the trailing '.0' (``std::to_chars(1.0)`` prints ``1``)."""
    t = repr(float(v))
    return t[:-2] if t.endswith(".0") else t


def format_double(v):
    """Shortest round-trip text of a float (``repr``), like ``std::to_chars``.

    ``std::to_chars`` may choose scientific notation where ``repr`` prints
    fixed (e.g. 0.0001); both parse to the same double, which is all the list
    contract needs (the canonical hash is over the values, not the text).
    """
    return repr(float(v))


def write_primary_csv(path, ids, values, pdg, charge):
    """Writes a list file in the C++ format (LF, header, one row per muon)."""
    lines = [LIST_HEADER]
    for k in range(len(ids)):
        row = [str(int(ids[k]))]
        row.extend(format_double(values[k, j]) for j in range(7))
        row.append(str(int(pdg[k])))
        row.append(str(int(charge[k])))
        lines.append(",".join(row))
    text = "\n".join(lines) + "\n"
    with open(path, "xb") as fh:
        fh.write(text.encode("ascii"))
    return text.encode("ascii")


# --------------------------------------------------------------------------
# Pixel convention of SteppingAction.cc
# --------------------------------------------------------------------------
_K_DEG = math.pi / 180.0


def lround(x):
    """C ``lround``: nearest integer, halves away from zero."""
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return -int(math.floor(-x + 0.5))


def bin_index(v, vmin, vmax, n):
    """``binIndex`` of SteppingAction.cc: linspace centres, ``lround``."""
    spacing = (vmax - vmin) / (n - 1)
    idx = lround((v - vmin) / spacing)
    return idx if 0 <= idx < n else -1


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


def look_azimuth_rad(view):
    """``lookAzRad`` of ActionInitialization.cc (same operation order)."""
    return (view * VIEW_STEP_DEG + 180.0) * math.pi / 180.0


def pixel_of_direction(from_vec, view=0):
    """(iEl, iAz, elev_deg, relaz_deg) for the direction the muon came FROM.

    ``from_vec`` is minus the muon's momentum direction at the scoring plane,
    exactly as SteppingAction::UserSteppingAction computes it.
    """
    fx, fy, fz = float(from_vec[0]), float(from_vec[1]), float(from_vec[2])
    elev = math.asin(min(1.0, max(-1.0, fz)))
    rel = wrap_angle(math.atan2(fy, fx) - look_azimuth_rad(view))
    i_az = bin_index(rel / _K_DEG, -AZ_HALF_DEG, AZ_HALF_DEG, N_AZ)
    i_el = bin_index(elev / _K_DEG, EL_MIN_DEG, EL_MAX_DEG, N_EL)
    return i_el, i_az, elev / _K_DEG, rel / _K_DEG


def pixel_centres_deg():
    """(elevation centres, relative-azimuth centres) of the 256 x 256 grid."""
    el = np.linspace(EL_MIN_DEG, EL_MAX_DEG, N_EL)
    az = np.linspace(-AZ_HALF_DEG, AZ_HALF_DEG, N_AZ)
    return el, az


def detector_center(view=0):
    view_az = look_azimuth_rad(view) - math.pi
    return (DET_DIST_M * math.cos(view_az), DET_DIST_M * math.sin(view_az), DET_HEIGHT_M)


# --------------------------------------------------------------------------
# R_int mask: text format read by include/RoiMask.hh
# --------------------------------------------------------------------------
def mask_to_text(mask):
    """256 LF-terminated lines of 256 '0'/'1'; line = iEl, char = iAz."""
    m = np.asarray(mask, dtype=bool)
    if m.shape != (N_EL, N_AZ):
        raise ValueError("mask shape %r != (256, 256)" % (m.shape,))
    rows = []
    for r in range(N_EL):
        rows.append("".join("1" if v else "0" for v in m[r]) + "\n")
    return "".join(rows).encode("ascii")


def mask_from_text(data):
    """Strict inverse of ``mask_to_text``; raises ``ValueError`` on any defect."""
    if len(data) != N_EL * (N_AZ + 1):
        raise ValueError("mask text must be 256 lines of 256 characters (got %d bytes)"
                         % len(data))
    arr = np.frombuffer(data, dtype=np.uint8).reshape(N_EL, N_AZ + 1)
    if not (arr[:, N_AZ] == 10).all():
        raise ValueError("mask line without LF terminator")
    body = arr[:, :N_AZ]
    if not ((body == 48) | (body == 49)).all():
        raise ValueError("mask contains characters other than 0/1")
    return body == 49


def mask_content_sha256(mask):
    return sha256_bytes(np.ascontiguousarray(np.asarray(mask, dtype=bool)).tobytes())


def mask_summary(mask):
    m = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    return {
        "forma": [int(m.shape[0]), int(m.shape[1])],
        "dtype": "bool",
        "pixeles": int(m.sum()),
        "filas": [int(rows.min()), int(rows.max())] if len(rows) else None,
        "columnas": [int(cols.min()), int(cols.max())] if len(cols) else None,
    }


# --------------------------------------------------------------------------
# Readers of the per-run outputs
# --------------------------------------------------------------------------
def read_hist_csv(path):
    """256 x 256 int64 histogram (rows = iEl, columns = iAz)."""
    with open(path, "rb") as fh:
        data = fh.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if len(lines) != N_EL:
        raise ValueError("hist.csv has %d lines, expected 256" % len(lines))
    rows = [line.split(b",") for line in lines]
    for i, parts in enumerate(rows):
        if len(parts) != N_AZ:
            raise ValueError("hist.csv line %d has %d fields, expected 256" % (i, len(parts)))
    flat = np.array([tok for parts in rows for tok in parts])
    return flat.astype(np.int64).reshape(N_EL, N_AZ)


def muon_header_columns(header):
    """Column names of a muones.csv header line (bytes), validated.

    All 47 base columns must be present exactly once and in their own order; the only other
    names allowed are ``MUON_OPTIONAL_COLUMNS`` (anywhere, once each). Raises ``ValueError``.
    """
    try:
        names = tuple(header.decode("ascii").split(","))
    except UnicodeDecodeError:
        raise ValueError("muones.csv header is not ASCII")
    if len(set(names)) != len(names):
        raise ValueError("muones.csv header repeats a column")
    missing = [c for c in MUON_COLUMNS if c not in names]
    unknown = [c for c in names if c not in MUON_COLUMNS and c not in MUON_OPTIONAL_COLUMNS]
    if missing or unknown:
        raise ValueError("muones.csv header differs from the expected 47 columns "
                         "(missing: %s; unknown: %s)" % (missing or "none", unknown or "none"))
    if [c for c in names if c in MUON_COLUMNS] != list(MUON_COLUMNS):
        raise ValueError("muones.csv header has the 47 base columns in another order")
    return names


def muon_columns_present(mu):
    """The columns of a parsed muones log, in canonical order (base columns, then optional ones)."""
    return [c for c in MUON_COLUMNS + MUON_OPTIONAL_COLUMNS if c in mu]


def parse_muones_bytes(data, chunk_rows=25000):
    """Parses ``<stem>_muones.csv`` into a dict of numpy columns.

    Integer columns that are always populated become int64; string columns
    become ``S`` arrays; every other column is float64 with NaN where the
    field is empty (not applicable / not logged). The columns are read by name
    from the header (see ``muon_header_columns``). Raises ``ValueError`` if the
    header is not acceptable or a row has the wrong number of fields.
    """
    lines = data.split(b"\n")
    if not lines:
        raise ValueError("muones.csv is empty")
    names = muon_header_columns(lines[0])
    if lines[-1] == b"":
        lines.pop()
    body = lines[1:]
    ncol = len(names)
    parts = {name: [] for name in names}
    for start in range(0, len(body), chunk_rows):
        rows = []
        for line in body[start:start + chunk_rows]:
            fields = line.split(b",")
            if len(fields) != ncol:
                raise ValueError("muones.csv row with %d fields (expected %d): %r" %
                                 (len(fields), ncol, line[:80]))
            rows.append(fields)
        if not rows:
            continue
        cols = list(zip(*rows))
        for k, name in enumerate(names):
            col = np.array(cols[k])
            if name in _MU_STR:
                parts[name].append(col)
            elif name in _MU_INT:
                parts[name].append(col.astype(np.int64))
            else:
                col = np.where(col == b"", b"nan", col)
                parts[name].append(col.astype(np.float64))
    out = {}
    for name in names:
        if parts[name]:
            out[name] = np.concatenate(parts[name])
        elif name in _MU_STR:
            out[name] = np.zeros(0, dtype="S1")
        elif name in _MU_INT:
            out[name] = np.zeros(0, dtype=np.int64)
        else:
            out[name] = np.zeros(0, dtype=np.float64)
    return out


def read_json(path):
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def timing_meta_paths(meta):
    """Sorted names of the durations / resource figures present in a meta.json.

    Only the TOP-LEVEL keys listed in ``TIMING_META_KEYS`` count (that is where the binary writes them), and
    only with a numeric value: a string, a boolean, a block or the same name deeper in the tree is compared
    like any other field.
    """
    if not isinstance(meta, dict):
        return []
    return sorted(k for k in meta if k in TIMING_META_KEYS and _is_number(meta[k]))


def meta_path_value(meta, path):
    """Value of the dotted ``path`` in a nested dict (KeyError when absent)."""
    cur = meta
    for part in path.split("."):
        cur = cur[part]
    return cur


def meta_without_timing(meta):
    """Copy of ``meta`` without the values listed by ``timing_meta_paths``."""
    if not isinstance(meta, dict):
        return meta
    ignored = set(timing_meta_paths(meta))
    return {k: v for k, v in meta.items() if k not in ignored}


_WARN_RE = re.compile(r"^\*\*\* G4Exception : (\S+)", re.M)


def count_warning_codes(log_text):
    """{code: count} from the ``*** G4Exception : <code>`` lines of a run log."""
    counts = {}
    for code in _WARN_RE.findall(log_text):
        counts[code] = counts.get(code, 0) + 1
    return counts


def warnings_as_text(counts):
    if not counts:
        return "ninguno"
    return ",".join("%s=%d" % (k, counts[k]) for k in sorted(counts))


def find_line(log_text, prefix):
    for line in log_text.splitlines():
        if line.startswith(prefix):
            return line
    return None


# --------------------------------------------------------------------------
# Statistics without SciPy
# --------------------------------------------------------------------------
def _gamma_q(a, x):
    """Regularised upper incomplete gamma Q(a, x) (Numerical Recipes 6.2)."""
    if x <= 0.0:
        return 1.0
    if a <= 0.0:
        raise ValueError("a must be positive")
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(100000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi2_sf(x, df):
    """P(chi2_df >= x)."""
    if df <= 0:
        raise ValueError("df must be positive")
    return _gamma_q(df / 2.0, x / 2.0)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ks_uniform_statistic(u):
    """Two-sided KS statistic D of the sample ``u`` against U(0, 1)."""
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = len(u)
    if n == 0:
        return 0.0
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(max((i / n - u).max(), (u - (i - 1) / n).max()))


def ks_pvalue(d, n):
    """Asymptotic two-sided Kolmogorov tail with Stephens' finite-n correction."""
    if n <= 0:
        return 1.0
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d
    if lam <= 0.0:
        return 1.0
    if lam < 0.27:
        return 1.0
    total = 0.0
    for k in range(1, 201):
        term = 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-18:
            break
    return min(1.0, max(0.0, total))


def ks_uniform_test(u):
    d = ks_uniform_statistic(u)
    return d, ks_pvalue(d, len(u))


def _log_choose_table(n):
    """log(n!) for 0..n by cumulative sums (accurate to ~1e-9 for n <= 1e6)."""
    lf = np.zeros(n + 1)
    if n > 0:
        lf[1:] = np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)))
    return lf


def binom_two_sided_p(k, n, p):
    """Exact two-sided binomial p-value (sum of pmf <= pmf(k), as SciPy's
    ``binomtest`` with the default 'two-sided' alternative)."""
    if n <= 0:
        return 1.0
    lf = _log_choose_table(n)
    i = np.arange(0, n + 1)
    with np.errstate(divide="ignore"):
        logpmf = (lf[n] - lf[i] - lf[n - i] + i * math.log(p) + (n - i) * math.log1p(-p))
    obs = logpmf[k]
    keep = logpmf <= obs + 1e-7  # SciPy: pmf <= pmf(k) * (1 + 1e-7)
    return float(min(1.0, np.exp(logpmf[keep]).sum()))


def _binom_logpmf(i, n, p):
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) +
            i * math.log(p) + (n - i) * math.log1p(-p))


def mcnemar_exact(n01, n10):
    """Exact conditional McNemar test on the discordant pairs.

    Conditional on b = n01 + n10, n01 ~ Binomial(b, 1/2) under H0 (no marginal
    difference). Returns the two-sided exact p-value (twice the smaller tail,
    capped at 1). For b = 0 the test carries no information (p = 1) and the
    result is flagged ``definido = False``.
    """
    b = int(n01) + int(n10)
    if b == 0:
        return {"b": 0, "n01": int(n01), "n10": int(n10), "definido": False,
                "p_dos_colas": 1.0}
    k = min(int(n01), int(n10))
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.exp(_binom_logpmf(i, b, 0.5))
    return {"b": b, "n01": int(n01), "n10": int(n10), "definido": True,
            "p_dos_colas": float(min(1.0, 2.0 * tail))}


def _tango_z(n01, n10, n, d):
    """Tango's (1998) score statistic for H0: p01 - p10 = d.

    The constrained MLE solves 2N q^2 - B q - n10 d (1 - d) = 0 with
    B = (n01 + n10) - d (2N - n01 + n10); q = p10 under H0 and p01 = q + d.
    """
    b = float(n01)
    c = float(n10)
    nn = float(n)
    bb = (b + c) - d * (2.0 * nn - b + c)
    prod = -c * d * (1.0 - d) / (2.0 * nn)  # product of the two roots
    disc = math.sqrt(max(0.0, bb * bb - 4.0 * 2.0 * nn * (-c * d * (1.0 - d))))
    if bb >= 0.0:
        q = (bb + disc) / (4.0 * nn)
    else:
        q_minus = (bb - disc) / (4.0 * nn)
        q = prod / q_minus if q_minus != 0.0 else 0.0
    q = max(q, 0.0, -d)  # feasibility: p10 >= 0 and p01 = q + d >= 0
    var = nn * (2.0 * q + d - d * d)
    num = b - c - nn * d
    if var <= 0.0:
        return 0.0 if num == 0.0 else math.copysign(math.inf, num)
    return num / math.sqrt(var)


def tango_interval(n01, n10, n, alpha=0.05):
    """Tango's score interval for d = p01 - p10 (paired difference).

    It is defined for every table, also when there are no discordant pairs
    (n01 = n10 = 0): then the limits are +-z^2/(N + z^2).
    Returns ``(lower, upper)`` found by bisection on the monotone statistic.
    """
    z = _norm_ppf(1.0 - alpha / 2.0)
    d_hat = (int(n01) - int(n10)) / float(n)

    def g(d):
        if d == d_hat:
            return 0.0
        return _tango_z(n01, n10, n, d)

    lo_bracket = -1.0 + 1e-15
    hi_bracket = 1.0 - 1e-15
    # lower limit: g decreases with d; find g(L) = +z
    a, b = lo_bracket, d_hat
    for _ in range(200):
        mid = 0.5 * (a + b)
        if g(mid) > z:
            a = mid
        else:
            b = mid
        if b - a < 1e-17:
            break
    lower = 0.5 * (a + b)
    a, b = d_hat, hi_bracket
    for _ in range(200):
        mid = 0.5 * (a + b)
        if g(mid) < -z:
            b = mid
        else:
            a = mid
        if b - a < 1e-17:
            break
    upper = 0.5 * (a + b)
    return lower, upper


def _betacf(a, b, x):
    """Continued fraction of the incomplete beta function (modified Lentz; Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def clopper_pearson_upper(k, n, alpha=0.05):
    """Upper limit of the exact two-sided (1 - alpha) Clopper-Pearson interval for a binomial proportion
    with ``k`` successes in ``n`` trials: the p that solves P(X <= k | n, p) = alpha / 2.

    Closed forms for k = 0 (1 - (alpha/2)^(1/n)) and k = n (1); otherwise a bisection on the exact binomial
    tail P(X <= k | n, p) = I_{1-p}(n - k, k + 1) (decreasing in p).
    """
    k, n = int(k), int(n)
    if n <= 0 or k < 0 or k > n:
        raise ValueError("need 0 <= k <= n and n > 0")
    if k == n:
        return 1.0
    tail = alpha / 2.0
    if k == 0:
        return -math.expm1(math.log(tail) / n)
    lo, hi = k / float(n), 1.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if _betai(n - k, k + 1, 1.0 - mid) > tail:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    return 0.5 * (lo + hi)


def clopper_pearson_interval(k, n, alpha=0.05):
    """(lower, upper) of the exact two-sided (1 - alpha) Clopper-Pearson interval (the lower limit uses the
    symmetry k -> n - k, p -> 1 - p)."""
    return 1.0 - clopper_pearson_upper(int(n) - int(k), n, alpha), clopper_pearson_upper(k, n, alpha)


def _norm_ppf(p):
    """Inverse normal CDF by bisection on erfc (adequate for interval limits)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - norm_sf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Generator tables (tables/row_weights_30d.txt, p_cdf.txt, meta.json)
# --------------------------------------------------------------------------
def load_tables(tables_dir):
    rows = []
    with open(os.path.join(tables_dir, "row_weights_30d.txt"), "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append((float(parts[0]), float(parts[1])))
    centres = np.array([r[0] for r in rows])
    weights = np.array([r[1] for r in rows])
    edges = None
    bands = []
    with open(os.path.join(tables_dir, "p_cdf.txt"), "r") as fh:
        for line in fh:
            if line.startswith("# band_edges_deg:"):
                edges = np.array([float(v) for v in line[len("# band_edges_deg:"):].split()])
                continue
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) > 8:
                bands.append(np.array(vals))
    if edges is None or len(edges) != len(bands) + 1:
        raise ValueError("p_cdf.txt: band edges do not match the number of bands")
    with open(os.path.join(tables_dir, "meta.json"), "r") as fh:
        meta = json.load(fh)
    ratio = float(meta.get("charge_ratio_mu_plus_minus", 1.28))
    return {"centres": centres, "weights": weights, "edges": edges, "bands": bands,
            "prob_mu_plus": ratio / (1.0 + ratio), "charge_ratio": ratio,
            "half_step_deg": float(centres[1] - centres[0]) / 2.0}


# --------------------------------------------------------------------------
# Region Z: catalog primitives and crossing_reference.py (reused)
# --------------------------------------------------------------------------
_CROSSING_ENV = "OE1_CROSSING_REFERENCE"


def crossing_reference_candidates(explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get(_CROSSING_ENV):
        cands.append(os.environ[_CROSSING_ENV])
    d = here()
    cands.append(os.path.join(d, os.pardir, "simulation", "geant4_app", "tests",
                              "crossing_reference.py"))
    cands.append(os.path.join(d, "crossing_reference.py"))
    home = os.path.expanduser("~")
    for variant in ("A2", "B2"):
        cands.append(os.path.join(home, "oe1-trabajo", "etapa1", "fuentes", variant,
                                  "simulation", "geant4_app", "tests",
                                  "crossing_reference.py"))
    return [os.path.abspath(c) for c in cands]


def load_crossing_reference(explicit=None):
    """Imports crossing_reference.py by path. Returns ``(module, path, sha256)``."""
    for path in crossing_reference_candidates(explicit):
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("oe1_crossing_reference", path)
            mod = importlib.util.module_from_spec(spec)
            # the file belongs to the director's C++ tree (or to a read-only SC3 copy): never leave
            # a __pycache__ next to it
            previous = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.dont_write_bytecode = previous
            return mod, path, sha256_file(path)
    raise FileNotFoundError(
        "crossing_reference.py not found; set %s or pass --crossing-reference "
        "(searched: %s)" % (_CROSSING_ENV, "; ".join(crossing_reference_candidates(explicit))))


def read_catalog(path):
    """Minimal reader of the geometry catalog JSON (only what the tools need)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    j = json.loads(raw.decode("utf-8"))
    out = {"geometry_id": j["geometry_id"],
           "base_radius_m": float(j["edifice"]["base_radius_m"]),
           "height_m": float(j["edifice"]["height_m"]),
           "anomalies": j.get("anomalies", []),
           "sha256": sha256_bytes(raw)}
    return out


def region_from_catalog(cat, first=0, second=1):
    """(cylinder, ellipsoid) dicts of Z = anomaly `first` AND anomaly `second`."""
    a = cat["anomalies"][first]
    b = cat["anomalies"][second]
    if a["kind"] != "conduit":
        raise ValueError("anomaly %d is not a conduit" % first)
    p = a["params"]
    cyl = {"cx": p["cx_m"], "cy": p["cy_m"], "r": p["radius_m"], "z0": p["z0_m"],
           "z1": p["z1_m"]}
    q = b["params"]
    if b["kind"] == "dense_body":
        ell = {"cx": q["cx_m"], "cy": q["cy_m"], "cz": q["cz_m"], "ax": q["ax_m"],
               "ay": q["ay_m"], "az": q["az_m"]}
    elif b["kind"] == "cavity":
        ell = {"cx": q["cx_m"], "cy": q["cy_m"], "cz": q["cz_m"], "ax": q["radius_m"],
               "ay": q["radius_m"], "az": q["radius_m"]}
    else:
        raise ValueError("anomaly %d is neither dense_body nor cavity" % second)
    return cyl, ell


def reference_matches_catalog(cr, cat):
    """True if crossing_reference.CYL/ELL equal the catalog primitives exactly."""
    cyl, ell = region_from_catalog(cat)
    return all(float(cr.CYL[k]) == float(cyl[k]) for k in cyl) and \
        all(float(cr.ELL[k]) == float(ell[k]) for k in ell)


def line_region_model(cr, start, direction, start_radius_m):
    """Straight-line crossing model for primaries of a list.

    Each primary travels from ``start`` along ``direction`` (unit) to the
    scoring panel; the segment ends at the point the generator aimed at,
    ``start + direction * R_start / hypot(dx, dy)``.
    """
    start = np.asarray(start, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    hor = np.hypot(direction[:, 0], direction[:, 1])
    end = start + direction * (start_radius_m / hor)[:, None]
    return cr.model(np.hstack([start, end]))
