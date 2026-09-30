"""Synthetic fixtures for the OE1 stage-1 tools (tests only).

* ``sample_source_model``: numpy re-implementation of the pilot's sampler, used
  only to make statistically valid lists for the validator tests.
* ``build_attempt``: a complete, self-consistent ``intento-NN`` folder with the
  13 runs, the manifest, the six lists and a small R_int mask, written in the
  exact formats of the C++ binary (47-column muones log, oe1_etapa1 meta key,
  run.log lines). Tests then break single contracts on purpose.
"""
import copy
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.dont_write_bytecode = True  # tests must not leave __pycache__ folders in the source tree

import oe1_stage1_common as C  # noqa: E402

TABLES_DIR = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "simulation", "geant4_app", "tables"))
CATALOG_DIR = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "simulation", "geometry_catalog",
                                           "geometries"))
CROSSING_REFERENCE = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "simulation", "geant4_app",
                                                  "tests", "crossing_reference.py"))
REAL_NPZ_ENV = "OE1_TEST_REAL_NPZ"

CASES = {"geo-train-06": {"short": "g06", "code": 6, "base_radius": 509.7},
         "geo-train-01": {"short": "g01", "code": 1, "base_radius": 656.2}}


# --------------------------------------------------------------------------
def sample_source_model(rng, n, tables, base_radius, view=0, rel_scale=1.0, p_scale=1.0, plus_fraction=None):
    """Same model as PrimaryGeneratorAction::GeneratePrimaries (numpy RNG)."""
    cum = np.cumsum(tables["weights"]) / tables["weights"].sum()
    cum[-1] = 1.0
    row = np.searchsorted(cum, rng.random(n), side="left")
    elev_deg = tables["centres"][row] + (2 * rng.random(n) - 1) * tables["half_step_deg"]
    rel = (2 * rng.random(n) - 1) * C.AZ_HALF_DEG * math.pi / 180.0 * rel_scale
    look = C.look_azimuth_rad(view)
    el = np.radians(elev_deg)
    az = look + rel
    frm = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    arrival = -frm
    band = np.clip(np.searchsorted(tables["edges"], elev_deg, side="right") - 1, 0, len(tables["bands"]) - 1)
    u4 = rng.random(n)
    p = np.empty(n)
    for b, q in enumerate(tables["bands"]):
        sel = band == b
        x = u4[sel] * (len(q) - 1)
        i = np.minimum(x.astype(np.int64), len(q) - 2)
        p[sel] = q[i] + (x - i) * (q[i + 1] - q[i])
    view_az = look - math.pi
    det = np.array([C.DET_DIST_M * math.cos(view_az), C.DET_DIST_M * math.sin(view_az), C.DET_HEIGHT_M])
    panel_y = np.array([-math.sin(view_az), math.cos(view_az), 0.0])
    half = C.DET_SIDE_M / 2
    ua, ub = rng.random(n), rng.random(n)
    impact = det + panel_y * ((2 * ua - 1) * half)[:, None] + np.array([0, 0, 1.0]) * ((2 * ub - 1) * half)[:, None]
    r_start = C.DET_DIST_M + base_radius + 300.0
    path_len = r_start / np.cos(el)
    start = impact - arrival * path_len[:, None]
    plus = rng.random(n) < (tables["prob_mu_plus"] if plus_fraction is None else plus_fraction)
    p = p * p_scale
    ekin = np.sqrt(p * p + C.MUON_MASS_GEV ** 2) - C.MUON_MASS_GEV
    vals = np.hstack([start, arrival, ekin[:, None]])
    pdg = np.where(plus, -13, 13).astype(np.int64)
    charge = np.where(plus, 1, -1).astype(np.int64)
    return np.arange(n, dtype=np.int64), vals, pdg, charge


def write_list(directory, name, ids, vals, pdg, charge, geometry_id, case_code, replica, source_seed,
               base_radius, view=0, extra=None):
    """CSV + sidecar like the C++ ``--write-primaries``. Returns paths and hashes."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    raw = C.write_primary_csv(path, ids, vals, pdg, charge)
    canon = C.canonical_fnv(ids, vals, pdg)
    side = {"format": C.LIST_FORMAT, "kind": "source-model", "geometry_id": geometry_id,
            "view_index": view, "base_radius_m": base_radius,
            "start_radius_m": C.DET_DIST_M + base_radius + 300.0, "case_code": case_code,
            "replica": replica, "source_seed": source_seed, "n": len(ids),
            "canon_fnv1a64": C.hex16(canon), "file_sha256": C.sha256_bytes(raw),
            "binary_sha256": "0" * 64}
    if extra:
        side.update(extra)
    with open(C.sidecar_path(path), "wb") as fh:
        fh.write((json.dumps(side, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    return {"path": path, "sha256": side["file_sha256"], "canon": canon, "n": len(ids)}


# --------------------------------------------------------------------------
# mask
# --------------------------------------------------------------------------
def make_test_mask():
    m = np.zeros((C.N_EL, C.N_AZ), dtype=bool)
    m[30:34, 100:104] = True
    return m


def make_test_masks():
    rint = make_test_mask()
    cond = np.zeros_like(rint)
    cond[36:40, 100:104] = True
    body = np.zeros_like(rint)
    body[42:46, 100:104] = True
    rock = np.zeros_like(rint)
    rock[100:140, 60:120] = True
    sky = np.zeros_like(rint)
    sky[200:250, 60:120] = True
    return {"R_int": rint, "R_cond": cond, "R_body": body, "R_rock": rock, "R_sky": sky}


def write_mask_files(directory):
    """Text mask, frozen JSON and npz for the small test mask."""
    os.makedirs(directory, exist_ok=True)
    masks = make_test_masks()
    npz_path = os.path.join(directory, "roi-mascaras.npz")
    np.savez_compressed(npz_path, **masks)
    text = C.mask_to_text(masks["R_int"])
    text_path = os.path.join(directory, "R_int.mask.txt")
    with open(text_path, "wb") as fh:
        fh.write(text)
    spec = {"formato": "oe1-roi-congelada/1", "mascara": "R_int",
            "origen": {"sha256_archivo": C.sha256_file(npz_path)},
            "hash_contenido": {"valor": C.mask_content_sha256(masks["R_int"])},
            "texto": {"archivo": "R_int.mask.txt", "sha256": C.sha256_bytes(text)}}
    spec.update(C.mask_summary(masks["R_int"]))
    frozen_path = os.path.join(directory, "R_int-congelada.json")
    with open(frozen_path, "wb") as fh:
        fh.write((json.dumps(spec, indent=1) + "\n").encode("utf-8"))
    return {"text": text_path, "frozen": frozen_path, "npz": npz_path, "masks": masks,
            "text_sha": C.sha256_bytes(text)}


# --------------------------------------------------------------------------
# records and files of one run
# --------------------------------------------------------------------------
def hex_of(x):
    return C.hex16(x)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def blank_record(i):
    return {"id": i, "prim_fnv": 0, "rng0": 0, "rng1": 0, "detected": 0, "iel": -1, "iaz": -1, "in_rint": -1,
            "plane": False, "ekin": 0.0, "el": 0.0, "rel": 0.0, "status": "WORLD_EXIT", "end_process": "Transportation",
            "n_steps": 5, "have_end": True, "end_x": 0.0, "end_y": 0.0, "end_z": 0.0, "end_vol": "World",
            "deep": False, "lz": 0.0, "lz_true": 0.0, "lz_rock": 0.0, "lz_a0": 0.0, "lz_a1": 0.0,
            "lz_other": 0.0, "lc": 0.0, "le": 0.0, "rho": 0.0, "n_z": 0, "n_z_bo": 0, "n_entries": 0,
            "n_graze": 0, "mat_in": "", "mat_out": "", "have_ein": False, "e_in": (0.0, 0.0, 0.0),
            "have_eout": False, "e_out": (0.0, 0.0, 0.0), "max_lat": 0.0, "lat_z": 0.0, "sum_lat_z": 0.0}


def base_records(n, caso, replica, seed, mask):
    """Variant-independent truth for one (case, replica)."""
    rng = np.random.default_rng([seed, CASES[caso]["code"], replica])
    recs = []
    mask_pixels = np.argwhere(mask)
    for i in range(n):
        r = blank_record(i)
        r["prim_fnv"] = int(rng.integers(1, 1 << 62))
        r["rng0"] = int(rng.integers(1, 1 << 62))
        r["rng1"] = int(rng.integers(1, 1 << 62))
        r["n_steps"] = int(rng.integers(3, 500))
        # WentzelVI displaces the track inside a step (PLAN C1): a real log has max_lat_mm far above 1e-6 mm
        # for most muons, so the pristine fixtures carry a deterministic 0-5 mm pattern (0 for a tenth)
        r["max_lat"] = 0.0 if i % 10 == 0 else 0.001 * ((i * 7919) % 5000 + 1)
        # lateral displacement of the steps near Z (PLAN C1): a minority of the muons has any; it is part of
        # the displacement over all steps (never above max_lat) and the sum is never below the maximum
        if i % 7 == 0:
            r["lat_z"] = min(0.001 * ((i * 104729) % 9000 + 1), r["max_lat"])
            r["sum_lat_z"] = r["lat_z"] * (1 + i % 5)
        u = rng.random()
        if u < 0.70:
            r["status"] = "DET"
            r["detected"] = 1
            if rng.random() < 0.06:
                iel, iaz = mask_pixels[int(rng.integers(0, len(mask_pixels)))]
            else:
                iel, iaz = int(rng.integers(0, C.N_EL)), int(rng.integers(0, C.N_AZ))
            r["iel"], r["iaz"] = int(iel), int(iaz)
            r["in_rint"] = int(mask[iel, iaz])
            r["plane"] = True
            r["ekin"] = float(rng.uniform(1.0, 200.0))
            r["el"] = C.EL_MIN_DEG + iel * (C.EL_MAX_DEG - C.EL_MIN_DEG) / (C.N_EL - 1)
            r["rel"] = -C.AZ_HALF_DEG + iaz * 2 * C.AZ_HALF_DEG / (C.N_AZ - 1)
            r["end_vol"] = "detector"
            r["end_x"], r["end_y"], r["end_z"] = 1000.0, float(rng.uniform(-0.6, 0.6)), 1.5
        elif u < 0.72:
            r["status"] = "PLANE_OUT"
            r["plane"] = True
            r["ekin"] = float(rng.uniform(1.0, 200.0))
            r["el"] = float(rng.uniform(1.0, 1.9))
            r["rel"] = float(rng.uniform(-30, 30))
            r["end_vol"] = "detector"
        elif u < 0.85:
            r["status"] = "DECAY"
            r["end_process"] = "Decay"
            r["end_vol"] = "edifice"
        elif u < 0.95:
            r["status"] = "MU_CAPTURE"
            r["end_process"] = "muMinusCaptureAtRest"
            r["end_vol"] = "edifice"
        else:
            r["status"] = "WORLD_EXIT"
        recs.append(r)
    return recs


def make_pair_records(n, caso, replica, seed, mask, crossers=None):
    """(records_A, records_B) with the identical primaries and, for geo-train-06,
    a few Z crossers whose lengths are split between anomaly_0 and anomaly_1 in A."""
    a = base_records(n, caso, replica, seed, mask)
    b = copy.deepcopy(a)
    if caso == "geo-train-06":
        picks = crossers if crossers is not None else [5, 17, 29, 41]
        for k, i in enumerate(picks):
            if i >= n:
                continue
            length = 8.0 + 6.0 * k
            for tag, rec in (("A", a[i]), ("B", b[i])):
                rec["deep"] = True
                rec["lz"] = length
                rec["lz_true"] = length * 1.001
                rec["lc"] = length + 3.0
                rec["le"] = length + 1.0
                rec["rho"] = length * 100 * 2.5
                rec["n_z"], rec["n_z_bo"], rec["n_entries"] = 3, 1, 1
                rec["have_ein"] = rec["have_eout"] = True
                rec["e_in"] = (100.0, 99.0, 101.0)
                rec["e_out"] = (95.0, 94.0, 96.0)
                if tag == "A":
                    rec["lz_a0"] = 0.4 * length
                    rec["lz_a1"] = 0.6 * length
                    rec["mat_in"], rec["mat_out"] = "A0", "A1"
                else:
                    rec["lz_a1"] = length
                    rec["mat_in"], rec["mat_out"] = "A1", "A1"
            # crossers diverge afterwards
            b[i]["rng1"] = a[i]["rng1"] ^ 0x5A5A5A
    return a, b


def apply_region_defaults(recs, applicable):
    """Rows of a run whose region is not applicable carry crossed = -1 and empty Z fields."""
    for r in recs:
        r["region"] = applicable


def fmt_num(v):
    return C.to_chars_like(v)


def muon_row(consts, r, full=True, region=True, lateral_columns=False):
    f = []
    f += [str(consts["code"]), consts["geometry_id"], "0", consts["variant"], str(consts["replica"]),
          consts["list_sha256"], str(r["id"]), hex_of(r["prim_fnv"]), hex_of(r["rng0"]), hex_of(r["rng1"])]
    f += [str(r["detected"]), str(r["iel"]), str(r["iaz"]), str(r["in_rint"])]
    if r["plane"]:
        f += [fmt_num(r["ekin"]), fmt_num(r["el"]), fmt_num(r["rel"])]
    else:
        f += ["", "", ""]
    f += [r["status"], r["end_process"], str(r["n_steps"])]
    if full and r["have_end"]:
        f += [fmt_num(r["end_x"]), fmt_num(r["end_y"]), fmt_num(r["end_z"]), r["end_vol"]]
    else:
        f += ["", "", "", ""]
    z = full and region
    f.append(str((1 if r["deep"] else 0) if region else -1) if full else "")
    for k in ("lz", "lz_true", "lz_rock", "lz_a0", "lz_a1", "lz_other", "lc", "le", "rho"):
        f.append(fmt_num(r[k]) if z else "")
    for k in ("n_z", "n_z_bo", "n_entries", "n_graze"):
        f.append(str(r[k]) if z else "")
    f.append(r["mat_in"] if z else "")
    f.append(r["mat_out"] if z else "")
    for have, key in ((r["have_ein"], "e_in"), (r["have_eout"], "e_out")):
        for v in (r[key] if (z and have) else (None, None, None)):
            f.append(fmt_num(v) if v is not None else "")
    f.append(fmt_num(r["max_lat"]) if (full and r["n_steps"] > 0) else "")
    assert len(f) == len(C.MUON_COLUMNS), len(f)
    if lateral_columns:      # max_lat_z_mm, sum_lat_z_mm: only where the region applies
        f.append(fmt_num(r["lat_z"]) if z else "")
        f.append(fmt_num(r["sum_lat_z"]) if z else "")
    return ",".join(f) + "\n"


def write_run(root, entry, recs, lists, maskinfo, tweak_meta=None, wall=12.5, node="guane01",
              job="30600", extra_log="", header_warnings=None, lateral_columns=False):
    """Writes all files of one run folder plus its status file. Returns a dict of paths."""
    caso = entry["caso"]
    run_dir = os.path.join(root, entry["salida"])
    os.makedirs(run_dir, exist_ok=False)
    consts = {"code": entry["case_code"], "geometry_id": caso, "variant": entry["variante"],
              "replica": entry["replica"], "list_sha256": entry["lista_sha256"]}
    region = caso == "geo-train-06"
    n = len(recs)
    recs = sorted(recs, key=lambda r: r["id"])
    header = C.MUON_HEADER + ("," + ",".join(C.MUON_OPTIONAL_COLUMNS) if lateral_columns else "")
    text = header + "\n" + "".join(muon_row(consts, r, True, region, lateral_columns) for r in recs)
    mu_bytes = text.encode("ascii")
    stem = os.path.join(run_dir, "%s_view0" % caso)
    with open(stem + "_muones.csv", "wb") as fh:
        fh.write(mu_bytes)
    hist = np.zeros((C.N_EL, C.N_AZ), dtype=np.int64)
    counts = {s: 0 for s in C.STATUS_NAMES}
    det_e = 0.0
    for r in recs:
        counts[r["status"]] += 1
        if r["status"] == "DET":
            hist[r["iel"], r["iaz"]] += 1
            det_e += r["ekin"]
    with open(stem + "_hist.csv", "wb") as fh:
        fh.write(("\n".join(",".join(map(str, row)) for row in hist.tolist()) + "\n").encode("ascii"))
    n_det = counts["DET"]
    sig = ["World:G4Box:G4_AIR", "detector:G4Box:G4_AIR", "edifice:G4Cons:standard_rock"]
    if region:
        sig.append("anomaly_0:%s:anomaly_0_mat" % ("G4Tubs" if entry["variante"] == "A" else "G4SubtractionSolid"))
        sig.append("anomaly_1:G4Ellipsoid:anomaly_1_mat")
    sig.sort()
    oe = {
        "format": "oe1-etapa1/1", "mode": "list", "variant": entry["variante"], "case_code": entry["case_code"],
        "replica": entry["replica"],
        # PLAN S2: the SHA-256 of the executable (the A and B binaries differ by construction) and of the
        # fixed inputs, as the C++ binary records them
        "binary_sha256": sha256_text("binary of variant " + entry["variante"]),
        "inputs": {"geometry_json": {"file": caso + ".json", "sha256": sha256_text("catalog " + caso)},
                   "tables": {"row_weights_30d.txt": sha256_text("row_weights_30d.txt"),
                              "p_cdf.txt": sha256_text("p_cdf.txt"), "meta.json": sha256_text("tables meta.json")}},
        "list": {"file": os.path.basename(entry["lista"]), "sha256": entry["lista_sha256"], "sha256_verified": True,
                 "canon_fnv1a64": lists[(caso, entry["replica"])]["canon_hex"], "n": n,
                 "sidecar_kind": "source-model", "sidecar_case_code": entry["case_code"],
                 "sidecar_replica": entry["replica"], "source_seed": entry["s_src"]},
        "roi_mask": {"file": "R_int.mask.txt", "sha256": maskinfo["text_sha"], "sha256_verified": True,
                     "ones": int(maskinfo["masks"]["R_int"].sum())},
        "seeds": {"master_cli": entry["s_tr"], "master_applied": entry["s_tr"] % 2147483647,
                  "transport_seed": entry["s_tr"], "source_seed": entry["s_src"],
                  "transport_key": ["event_id", entry["s_tr"], entry["case_code"], C.TAG_TRANSPORT]},
        "runtime": {"master_engine": "MixMaxRng", "run_manager_class": "G4TaskRunManager", "threads_cli": 1,
                    "worker_engines": ["MixMaxRng"], "worker_thread_ids": [0]},
        "region": {"applicable": region, "explicit_none": not region, "geometry_anomalies": 2 if region else 0},
        "counts": {"status": counts, "in_rint": int(sum(1 for r in recs if r["in_rint"] == 1)),
                   "max_lat_mm": max(r["max_lat"] for r in recs)},
        "geometry_signature": {"volumes": sig, "fnv1a64": "0" * 16},
        "checks": {"ids_complete_unique": True, "hist_reconstructed_equal": True, "detected_equal": True,
                   "events_equal": True, "generated_equal": True, "ekin_sum_equal": True, "writes_ok": True},
        "muon_log": {"file": os.path.basename(stem) + "_muones.csv", "level": "full", "rows": n,
                     "columns": 47 + (len(C.MUON_OPTIONAL_COLUMNS) if lateral_columns else 0),
                     "bytes": len(mu_bytes), "fnv1a64": C.hex16(C.fnv1a64(mu_bytes))},
        "errors": [],
    }
    if lateral_columns:
        # the corrected binary (PLAN C1) also aggregates the lateral displacement near Z into the meta
        # (null when the run has no region: the log columns are empty then)
        oe["counts"]["max_lat_z_mm"] = max(r["lat_z"] for r in recs) if region else None
        oe["counts"]["max_sum_lat_z_mm"] = max(r["sum_lat_z"] for r in recs) if region else None
        oe["counts"]["near_z_steps"] = 19 if region else None
        oe["counts"]["near_z_muons"] = 10 if region else None
    meta = {"detected_in_bins": n_det, "detector": {"distance_m": 1000.0, "height_m": 1.5, "side_m": 1.2},
            "ecomug_draws_total": n, "engine": "geant4-test", "events_transported": n, "exposure_days": -1.0,
            "geometry_id": caso, "grid": {"n_az": 256, "n_el": 256}, "mean_ekin_detected_gev":
            (det_e / n_det if n_det else 0.0), "oe1_etapa1": oe, "seed": entry["s_tr"], "threads": 1,
            "view_index": 0, "wall_seconds": wall, "seconds_per_event": wall / n,
            # PLAN W3: the binary writes getrusage(RUSAGE_SELF) as top-level keys next to wall_seconds
            # (user and system CPU seconds, peak RSS in kB); they differ between two identical runs
            "rusage_user_cpu_seconds": wall * 0.98, "rusage_system_cpu_seconds": 0.1 + wall / 1000.0,
            "rusage_max_rss_kb": int(90000 + 100 * wall)}
    if tweak_meta:
        tweak_meta(meta)
    with open(stem + ".meta.json", "wb") as fh:
        fh.write((json.dumps(meta, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    log = ""
    if entry["variante"] == "A" and region:
        log += ("\n-------- WWWW ------- G4Exception-START -------- WWWW -------\n"
                "*** G4Exception : GeomVol1002\n      issued by : G4PVPlacement::CheckOverlaps()\n"
                "Overlap with volume already placed !\n*** This is just a warning message. ***\n"
                "-------- WWWW -------- G4Exception-END --------- WWWW -------\n\n")
    log += extra_log
    log += "RUN DONE %s view 0: events=%d detected=%d wall_s=%r\n" % (caso, n, n_det, wall)
    log += ("OE1-ETAPA1 geometry=%s view=0 variant=%s case_code=%d replica=%d muon_log=full n=%d rows=%d "
            % (caso, entry["variante"], entry["case_code"], entry["replica"], n, n))
    log += " ".join("%s=%d" % (s, counts[s]) for s in C.STATUS_NAMES)
    log += (" in_rint=%d crossed=%s graze_muons=%s ids_ok=1 hist_ok=1 det_ok=1 writes_ok=1 ekin_sum_equal=1 "
            "errors=none status=OK\n" % (oe["counts"]["in_rint"], "0" if region else "NA", "0" if region else "NA"))
    with open(os.path.join(run_dir, "run.log"), "wb") as fh:
        fh.write(log.encode("utf-8"))
    with open(os.path.join(run_dir, "comando.txt"), "wb") as fh:
        fh.write(b"muography_sim --primaries x\n")
    with open(os.path.join(run_dir, "entradas.sha256"), "wb") as fh:
        fh.write(b"0" * 64 + b"  x\n")
    with open(os.path.join(run_dir, "recursos.json"), "wb") as fh:
        fh.write(json.dumps({"wall_s": wall, "user_s": wall * 0.98, "sys_s": 0.1, "maxrss_kb": 90000}).encode())
    write_sums(run_dir)
    os.makedirs(os.path.join(root, "estado"), exist_ok=True)
    with open(os.path.join(root, "estado", entry["run_id"] + ".status"), "wb") as fh:
        fh.write(("%s\tOK\t0\t\t%s\t%s\t2026-09-30T00:00:00Z\t2026-09-30T00:01:00Z\n" %
                  (entry["run_id"], node, job)).encode())
    return {"dir": run_dir, "stem": stem}


def build_attempt_records_for_parse():
    """Bytes of a tiny muones.csv (5 rows, geo-train-06 shape) for the parser tests."""
    consts = {"code": 6, "geometry_id": "geo-train-06", "variant": "B", "replica": 1, "list_sha256": "ab" * 32}
    recs = [blank_record(i) for i in range(5)]
    r = recs[0]
    r.update(status="DET", detected=1, iel=34, iaz=135, in_rint=1, plane=True, ekin=94.0, el=7.93, rel=1.8,
             deep=True, lz=89.67, lz_true=89.76, lz_a1=57.0, lz_a0=32.6, n_z=2, n_entries=1,
             mat_in="A0", mat_out="A1", have_ein=True, have_eout=True, e_in=(98.2, 98.0, 100.0),
             e_out=(96.9, 95.0, 98.0))
    recs[1].update(status="PLANE_OUT", plane=True, ekin=39.0, el=1.5, rel=20.0)
    text = C.MUON_HEADER + "\n" + "".join(muon_row(consts, x, True, True) for x in recs)
    return text.encode("ascii")


def write_sums(run_dir):
    """(Re)writes SHA256SUMS over every regular file except SHA256SUMS itself."""
    names = sorted(n for n in os.listdir(run_dir) if n != "SHA256SUMS" and
                   os.path.isfile(os.path.join(run_dir, n)) and not n.endswith(".part"))
    with open(os.path.join(run_dir, "SHA256SUMS"), "wb") as fh:
        for nme in names:
            fh.write(("%s  %s\n" % (C.sha256_file(os.path.join(run_dir, nme)), nme)).encode())


def refresh_muon_log_meta(stem, n=None):
    """After editing a muones.csv, recompute the FNV and byte count in the meta and the sums."""
    with open(stem + "_muones.csv", "rb") as fh:
        data = fh.read()
    with open(stem + ".meta.json", "rb") as fh:
        meta = json.loads(fh.read().decode())
    meta["oe1_etapa1"]["muon_log"]["fnv1a64"] = C.hex16(C.fnv1a64(data))
    meta["oe1_etapa1"]["muon_log"]["bytes"] = len(data)
    with open(stem + ".meta.json", "wb") as fh:
        fh.write((json.dumps(meta, indent=1, sort_keys=True) + "\n").encode())
    write_sums(os.path.dirname(stem))


# --------------------------------------------------------------------------
def build_attempt(root, n=300, seed=11, tweak=None, replicas=(1, 2, 3), lateral_columns=False):
    """Builds ``root`` (must not exist) and returns a dict with every path.

    ``tweak(records, entries)`` may edit the per-run record lists (before
    writing) so the files stay self-consistent: records[run_id] -> list of dicts.
    ``lateral_columns`` writes the optional max_lat_z_mm / sum_lat_z_mm columns.
    """
    os.makedirs(root)
    tables = C.load_tables(TABLES_DIR)
    maskinfo = write_mask_files(os.path.join(root, "roi"))
    lists_dir = os.path.join(root, "listas")
    lists = {}
    for caso, spec in CASES.items():
        for rep in replicas:
            s_src = 2026092990 if rep == 0 else 2026092900 + rep
            rng = np.random.default_rng([seed, spec["code"], rep, 99])
            ids, vals, pdg, charge = sample_source_model(rng, n, tables, spec["base_radius"])
            name = "%s_v0_r%d_s%d.primarios.csv" % (caso, rep, s_src)
            info = write_list(lists_dir, name, ids, vals, pdg, charge, caso, spec["code"], rep, s_src,
                              spec["base_radius"])
            info["canon_hex"] = C.hex16(info["canon"])
            info["s_src"] = s_src
            # what the binary logs as prim_fnv for each id: the FNV of the primary recomputed from the list rows
            info["prim_fnv"] = C.primary_fnv_array(vals, pdg)
            lists[(caso, rep)] = info
    entries = []
    order = []
    for caso in ("geo-train-06", "geo-train-01"):
        for rep in replicas:
            for var in ("A", "B"):
                order.append((caso, rep, var, ""))
    order.append(("geo-train-01", replicas[0], "A", "-rep"))
    for caso, rep, var, suffix in order:
        info = lists[(caso, rep)]
        run_id = "%s-%s-r%d%s" % (CASES[caso]["short"], var, rep, suffix)
        rol = "repeticion-AA" if suffix else "principal"
        folder = "A-repeticion" if suffix else var
        entries.append({
            "run_id": run_id, "caso": caso, "geometria": os.path.join(root, "catalogo", caso + ".json"),
            "variante": var, "binario": "/fake/build/%s2/muography_sim" % var, "s_src": info["s_src"],
            "s_tr": info["s_src"] - 1000000000, "case_code": CASES[caso]["code"], "lista": info["path"],
            "lista_sha256": info["sha256"], "N": n, "salida": "%s/%s/%d" % (caso, folder, info["s_src"]),
            "rol": rol, "avisos_esperados": "GeomVol1002=1" if (var == "A" and caso == "geo-train-06") else "ninguno",
            "replica": rep})
    records = {}
    for caso in ("geo-train-06", "geo-train-01"):
        for rep in replicas:
            ra, rb = make_pair_records(n, caso, rep, seed, maskinfo["masks"]["R_int"])
            for r_a, r_b in zip(ra, rb):
                r_a["prim_fnv"] = r_b["prim_fnv"] = int(lists[(caso, rep)]["prim_fnv"][r_a["id"]])
            records["%s-A-r%d" % (CASES[caso]["short"], rep)] = ra
            records["%s-B-r%d" % (CASES[caso]["short"], rep)] = rb
    records["g01-A-r%d-rep" % replicas[0]] = copy.deepcopy(records["g01-A-r%d" % replicas[0]])
    if tweak:
        tweak(records, entries)
    attempt = os.path.join(root, "intento-01")
    os.makedirs(attempt)
    paths = {}
    for e in entries:
        wall = 12.5 if not e["run_id"].endswith("-rep") else 13.1
        paths[e["run_id"]] = write_run(attempt, e, records[e["run_id"]], lists, maskinfo, wall=wall,
                                       lateral_columns=lateral_columns)
    manifest = os.path.join(attempt, "manifiesto-corridas.tsv")
    cols = ("run_id", "caso", "geometria", "variante", "binario", "s_src", "s_tr", "case_code", "lista",
            "lista_sha256", "N", "salida", "rol", "avisos_esperados")
    with open(manifest, "wb") as fh:
        fh.write(("\t".join(cols) + "\n").encode())
        for e in entries:
            fh.write(("\t".join(str(e[c]) for c in cols) + "\n").encode())
    return {"root": root, "attempt": attempt, "manifest": manifest, "mask": maskinfo, "lists": lists,
            "lists_dir": lists_dir, "entries": entries, "paths": paths, "records": records, "n": n}


def clone_attempt(src_root, dst_root):
    shutil.copytree(src_root, dst_root)
    return dst_root
