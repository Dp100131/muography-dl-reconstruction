#!/usr/bin/env python3
"""OE1 stage 1: contracts and paired analysis of the 13 runs.

Order of work (PLAN step 6)

  1. Contracts first (``contratos.json``). Blocking: K0 (manifest, inputs and
     the located lists: each list found must have the SHA-256 of its manifest
     row), K6 (completeness and reconstruction of every run; prim_fnv and rng0
     non-zero and unique, and prim_fnv recomputed from the list rows for every
     id), K7 (A/A byte identity), K9 (a) same primary and same random start for
     every id, in both cases, (c) B shows no rock / anomaly_0 length inside Z
     beyond what the lateral displacement explains. Not blocking but they stop
     the attribution to the overlap until explained: K8 (control A vs B), K9 (b)
     (classification of the discordances of muons that do not cross Z,
     diagnostic only, DV9), K9 (c) violations explained by lateral displacement,
     and the detection efficiency far from the campaign's (|z| > 5).
     Plan correction C1 (2026-09-30): the pre->post chord is not the navigated
     path (WentzelVI displaces the track inside a step), so K6 no longer asks for
     max_lat < 1e-6 mm; it asks for max_lat_mm to be reported (finite, equal to
     the meta) and, for the corrected binary (its meta declares the aggregates
     counts.max_lat_z_mm / max_sum_lat_z_mm), for the columns max_lat_z_mm /
     sum_lat_z_mm to be present, consistent and equal to those aggregates
     (sum_lat_z_mm is a SCALE of the error of the length in Z, not a bound; a log
     of an older binary, whose meta declares nothing, is reported as not
     verifiable).
     K7 and K8 ignore exactly the duration and resource keys of the meta that
     the binary writes (common.TIMING_META_KEYS: wall_seconds, seconds_per_event
     and the three rusage_* keys) and list the ones they ignored; K8 also ignores
     oe1_etapa1.binary_sha256, which differs between the A and B executables.
     A blocking failure stamps every output "BLOQUEADO -- no interpretable" and
     the process exits with code 3 (``--diagnostic-only`` still writes the
     stamped tables; without it only ``contratos.json`` is written).
  2. Join A, B and the A/A repeat on (case, replica, id). The A/A repeat is
     EXCLUDED from Delta, the transitions, the pooling and the replica SD.
  3. Per run table; per pair (case, replica): n00, n01, n10, n11 (they add up
     to N), Delta = (n01 - n10) / N, 10^6 Delta with its denominator; per case
     the pooled Delta (denominator 3 N) and the replica mean / SD / range; exact
     counts; Tango's score interval and the exact conditional McNemar test as
     DIAGNOSTICS (no SciPy); the bin-change matrix among muons detected in both;
     the 7 x 7 class matrix; the crossings in A, B, both and the union U with
     the conditional Delta_U and |U|; the straight-line cross-check; maps
     A/N, B/N, (B - A)/N.

Inputs are read only. Outputs: contratos.json, resultados-etapa1.json,
tabla-corridas.csv, transiciones.csv, matriz-clases.csv, cambios-bin.csv,
muones-discordantes.csv, cruces-union.csv, mapas-etapa1.npz, optional figures,
comando-analisis.txt. Python >= 3.9, numpy >= 1.23; SciPy never required.

Exit codes: 0 all blocking contracts hold; 3 blocking contract failed;
2 unusable input (a missing file, including --roi-npz; a bad manifest; a
--roi-frozen that is not valid JSON or lacks the keys the K0 checks read; a
--roi-npz that is not a readable .npz with the five mask arrays; a --replicas
that is not a list of integers). All of it is checked before the output folder
is created, so none leaves an empty folder.
"""
import argparse
import collections
import csv
import datetime
import glob
import io
import json
import math
import os
import re
import socket
import sys

import numpy as np

import oe1_stage1_common as C

MANIFEST_COLUMNS = ("run_id", "caso", "geometria", "variante", "binario", "s_src", "s_tr",
                    "case_code", "lista", "lista_sha256", "N", "salida", "rol",
                    "avisos_esperados")
CLASSES = ("R_int", "R_cond", "R_body", "R_rock", "R_sky", "sin clase", "no detectado")
ROLE_MAIN = "principal"
ROLE_AA = "repeticion-AA"
CASE_AFFECTED = "geo-train-06"
CASE_CONTROL = "geo-train-01"
CASE_ROLE = {CASE_AFFECTED: "afectado", CASE_CONTROL: "control angular"}
# Campaign detection efficiency, detected / events_transported of view 0 in results/dataset/campaign-manifest.csv
# (C11, informative). geo-train-06 is the exact campaign ratio; 0.752094 was the pilot's 3e6-event value.
EFFICIENCY_REFERENCE = {CASE_AFFECTED: 27209253 / 36174382, CASE_CONTROL: 0.73795}
S_TR_OFFSET = 1000000000
STAMP = "BLOQUEADO — no interpretable"
SENTENCE_SD = "DE = 0 entre réplicas no implica incertidumbre cero (PROT:76)."
MAP_CAPTION = "diferencia neta; no identifica muones individuales"
NOTE_INTERVAL = ("El intervalo de la etapa 1 es un diagnóstico: la etapa 1 no está dimensionada (PROT §4) y "
                 "su intervalo no acota el efecto ni decide equivalencia. Las cotas superiores exactas de "
                 "Clopper-Pearson de p_A y p_B se dan aparte, cada una con lo que es dato (ΣY y N medidos en "
                 "esta etapa) y lo que es supuesto (primarios independientes con la misma probabilidad).")
LABEL_CP = ("dato: ΣY (muones detectados dentro de R_int en la variante) y N (primarios generados), medidos en "
            "esta etapa; supuesto: los N primarios son ensayos de Bernoulli independientes con la misma "
            "probabilidad p (réplicas agrupadas), sin comprobar")
CP_ALPHA = 0.05               # two-sided level of the Clopper-Pearson bounds
LZ_TOL_UM_M = 1e-6            # 1 micrometre: K9 (c) and T-S
MAX_LAT_LIMIT_MM = 1e-6       # K6: only the reference for the reported fraction of muons above it
# Largest radius of curvature of the surfaces of Z: the dense-body ellipsoid (semi-axes 72.9, 61.6, 64.6 m) at the
# end of its shortest axis, 72.9^2 / 61.6 = 86.27 m.
R_MAX_M = 72.9 ** 2 / 61.6
EFFICIENCY_Z_LIMIT = 5.0      # K6 (informative): |z| of the efficiency against the campaign's
NOTE_CHORD = ("la cuerda pre→post no es la trayectoria navegada (WentzelVI desplaza lateralmente dentro del "
              "paso, aunque MuHadLateralDisplacement esté desactivado): las longitudes en Z son de la cuerda y "
              "sum_lat_z_mm (Σ del desplazamiento lateral de los pasos cercanos a Z) da la escala del error, no "
              "una cota: con incidencia θ el error de longitud es del orden de lat/sen θ y en una cuerda rasante "
              "crece como √(2·R·lat) (R ≤ %.1f m), de modo que se informa como ≲ 2√(2·R_max·lat); sin esa columna "
              "no hay escala") % R_MAX_M
EPS_MEAN_EKIN = 1e-9
MEAN_EKIN_KEY = "mean_ekin_detected_gev"
UNEXPLAINED_CLASSES = ("solo_elipsoide", "otra")
VARIANT_SOLIDS = {"A": "G4Tubs", "B": "G4SubtractionSolid"}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt(v, nd=None):
    """CSV cell: None -> 'NA'; floats with repr; ints plain."""
    if v is None:
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return "NA"
        return repr(f) if nd is None else ("%.*g" % (nd, f))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def per_million(x):
    return None if x is None else x * 1e6


def denominator_text(n):
    return "cuentas por millón de primarios generados; denominador N = %s" % (
        format(int(n), ",d").replace(",", " "))


def read_text(path):
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", "replace")


class Contracts:
    """Collects the individual checks and builds ``contratos.json``."""

    DESCRIPTIONS = {
        "K6": ("Completitud y reconstrucción de cada corrida (filas = N, IDs únicos, sin ABORTED, "
               "histograma reconstruido igual en las 65 536 celdas, in_rint igual al recalculado, "
               "prim_fnv y rng0 no nulos y únicos, prim_fnv recalculado desde la lista para todo id, "
               "desplazamiento lateral reportado con su escala de error (ya no se exige max_lat < 1e-6 mm: "
               "la cuerda pre→post no es la trayectoria navegada), B sin avisos, clave oe1_etapa1 y "
               "_muones.csv presentes)", True),
        "K7": ("Repetición A/A en geo-train-01 r1: hist y muones idénticos byte a byte y meta "
               "igual salvo tiempos y uso de recursos", True),
        "K8": ("Control geo-train-01, A frente a B: idénticos salvo la columna variant "
               "(se explica antes de atribuir nada, PROT:55)", False),
        "K9": ("A frente a B: (a) prim_fnv y rng0 iguales para todo id, en geo-train-06 y en el control; "
               "(b) clasificación de la discordancia de los que no cruzan Z (solo diagnóstico); (c) en B "
               "lz_a0 y lz_rock <= 1 µm, salvo lo que explica el desplazamiento lateral (no bloqueante)", True),
        "K0": ("Manifiesto y entradas fijas (13 corridas, semillas, listas localizadas con el SHA-256 "
               "del manifiesto, máscara)", True),
    }

    def __init__(self):
        self.checks = []

    def add(self, cid, scope, desc, ok, blocking=True, evidence=None):
        ok = ok if ok is None else bool(ok)
        self.checks.append({"contrato": cid, "ambito": scope, "id": "%s.%d" % (
            cid, 1 + sum(1 for c in self.checks if c["contrato"] == cid and c["ambito"] == scope)),
            "descripcion": desc, "ok": ok, "bloqueante": bool(blocking),
            "evidencia": evidence if evidence is not None else {}})
        return ok

    def failed_blocking(self):
        return [c for c in self.checks if c["ok"] is False and c["bloqueante"]]

    def failed_nonblocking(self):
        return [c for c in self.checks if c["ok"] is False and not c["bloqueante"]]

    def unverifiable(self):
        return [c for c in self.checks if c["ok"] is None]

    def to_json(self, extra=None):
        by = collections.OrderedDict()
        for c in self.checks:
            g = by.setdefault(c["contrato"], collections.OrderedDict())
            g.setdefault(c["ambito"], []).append(c)
        out = collections.OrderedDict()
        out["formato"] = "oe1-contratos/1"
        blocking = self.failed_blocking()
        out["estado"] = STAMP if blocking else "OK"
        out["fecha_utc"] = now_utc()
        out["bloqueantes_fallidos"] = ["%s [%s] %s" % (c["id"], c["ambito"], c["descripcion"]) for c in blocking]
        out["no_bloqueantes_fallidos"] = ["%s [%s] %s" % (c["id"], c["ambito"], c["descripcion"])
                                          for c in self.failed_nonblocking()]
        out["no_verificables"] = ["%s [%s] %s" % (c["id"], c["ambito"], c["descripcion"])
                                  for c in self.unverifiable()]
        contratos = collections.OrderedDict()
        for cid in sorted(by):
            desc, blk = self.DESCRIPTIONS.get(cid, (cid, True))
            allc = [c for grp in by[cid].values() for c in grp]
            if any(c["ok"] is False and c["bloqueante"] for c in allc):
                verdict = "FALLA"
            elif any(c["ok"] is False for c in allc):
                verdict = "REQUIERE_EXPLICACION"
            elif any(c["ok"] is None for c in allc):
                verdict = "OK_CON_AVISOS"
            else:
                verdict = "OK"
            contratos[cid] = {"descripcion": desc, "bloqueante": blk, "veredicto": verdict,
                              "ambitos": by[cid]}
        out["contratos"] = contratos
        if extra:
            out.update(extra)
        return out


# --------------------------------------------------------------------------
# manifest and run loading
# --------------------------------------------------------------------------
class ManifestError(ValueError):
    """The manifest cannot be used: main() reports it and exits with code 2 (unusable input)."""


def read_manifest(path):
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh, delimiter="\t"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError("no se puede leer el manifiesto %s: %s" % (path, exc))
    if not rows:
        raise ManifestError("manifiesto vacío: " + path)
    if tuple(rows[0]) != MANIFEST_COLUMNS:
        raise ManifestError("las columnas del manifiesto no son las esperadas: %r" % (rows[0],))
    entries = []
    for r in rows[1:]:
        if not r or not any(x.strip() for x in r):
            continue
        if len(r) != len(MANIFEST_COLUMNS):
            raise ManifestError("fila del manifiesto con %d campos: %r" % (len(r), r))
        e = dict(zip(MANIFEST_COLUMNS, r))
        try:
            for k in ("s_src", "s_tr", "case_code", "N"):
                e[k] = int(e[k])
        except ValueError as exc:
            raise ManifestError("fila del manifiesto %r: %s" % (r[0], exc))
        m = re.search(r"-r(\d+)", e["run_id"])
        e["replica"] = int(m.group(1)) if m else None
        entries.append(e)
    return entries


def manifest_checks(entries, expected_runs, contracts, reps=(1, 2, 3)):
    ids = [e["run_id"] for e in entries]
    ok_n = len(entries) == expected_runs and len(set(ids)) == len(ids)
    contracts.add("K0", "manifiesto", "el manifiesto tiene %d corridas con run_id únicos" % expected_runs,
                  ok_n, True, {"filas": len(entries), "unicos": len(set(ids))})
    bad_tr = [e["run_id"] for e in entries if e["s_tr"] != e["s_src"] - S_TR_OFFSET]
    contracts.add("K0", "manifiesto", "s_tr = s_src - 1e9 en todas las filas", not bad_tr, True,
                  {"incumplen": bad_tr})
    main = [e for e in entries if e["rol"] == ROLE_MAIN]
    aa = [e for e in entries if e["rol"] == ROLE_AA]
    cells = collections.Counter((e["caso"], e["replica"], e["variante"]) for e in main)
    complete = all(cells.get((c, r, v), 0) == 1 for c in (CASE_AFFECTED, CASE_CONTROL)
                   for r in reps for v in ("A", "B"))
    contracts.add("K0", "manifiesto", "%d corridas principales: 2 casos x réplicas %s x {A, B}" % (
        4 * len(reps), list(reps)), complete and len(main) == 4 * len(reps), True, {"principales": len(main)})
    ok_aa = (len(aa) == 1 and aa[0]["caso"] == CASE_CONTROL and aa[0]["variante"] == "A" and
             aa[0]["replica"] == reps[0])
    contracts.add("K0", "manifiesto", "una repetición A/A de geo-train-01 réplica %d (variante A)" % reps[0],
                  ok_aa, True, {"repeticiones": [e["run_id"] for e in aa]})
    if ok_aa:
        base = [e for e in main if e["caso"] == CASE_CONTROL and e["variante"] == "A" and e["replica"] == reps[0]]
        same = (bool(base) and base[0]["lista_sha256"] == aa[0]["lista_sha256"] and
                base[0]["s_tr"] == aa[0]["s_tr"] and base[0]["s_src"] == aa[0]["s_src"])
        contracts.add("K0", "manifiesto", "la repetición A/A usa la misma lista y la misma s_tr",
                      same, True, {})
    return complete and ok_n and ok_aa


class Run:
    def __init__(self, entry):
        self.entry = entry
        self.run_id = entry["run_id"]
        self.dir = None
        self.problems = []
        self.files = {}
        self.meta = None
        self.hist = None
        self.mu = None
        self.log = None
        self.status = None
        self.recursos = None
        self.sha = {}
        self.muones_len = None
        self.fnv_muones = None
        self.part_files = []


def load_run(entry, attempt, do_fnv=True):
    run = Run(entry)
    run.dir = os.path.join(attempt, entry["salida"])
    if not os.path.isdir(run.dir):
        run.problems.append("no existe la carpeta " + run.dir)
        return run

    def one(pattern, label):
        found = sorted(glob.glob(os.path.join(run.dir, pattern)))
        if len(found) != 1:
            run.problems.append("se esperaba un archivo %s y hay %d" % (label, len(found)))
            return None
        run.files[label] = found[0]
        return found[0]

    hist = one("*_hist.csv", "hist")
    meta = one("*.meta.json", "meta")
    muon = one("*_muones.csv", "muones")
    run.part_files = sorted(glob.glob(os.path.join(run.dir, "*.part")))
    for name in ("run.log", "comando.txt", "entradas.sha256", "SHA256SUMS", "recursos.json"):
        p = os.path.join(run.dir, name)
        if os.path.isfile(p):
            run.files[name] = p
    for label, p in run.files.items():
        run.sha[label] = C.sha256_file(p)
    try:
        if hist:
            run.hist = C.read_hist_csv(hist)
    except ValueError as exc:
        run.problems.append("hist.csv: %s" % exc)
    try:
        if meta:
            run.meta = C.read_json(meta)
    except ValueError as exc:
        run.problems.append("meta.json: %s" % exc)
    try:
        if muon:
            with open(muon, "rb") as fh:
                raw = fh.read()
            run.muones_len = len(raw)
            run.mu = C.parse_muones_bytes(raw)
            if do_fnv:
                run.fnv_muones = C.fnv1a64(raw)
            del raw           # ~30 MB per run at N = 1e5; 13 runs are held at once
    except ValueError as exc:
        run.problems.append("muones.csv: %s" % exc)
    if "run.log" in run.files:
        run.log = read_text(run.files["run.log"])
    if "recursos.json" in run.files:
        try:
            run.recursos = C.read_json(run.files["recursos.json"])
        except ValueError:
            run.recursos = None
    st = os.path.join(attempt, "estado", run.run_id + ".status")
    if os.path.isfile(st):
        parts = read_text(st).rstrip("\n").split("\t")
        run.status = {"run_id": parts[0], "veredicto": parts[1] if len(parts) > 1 else "",
                      "rc": parts[2] if len(parts) > 2 else "", "motivo": parts[3] if len(parts) > 3 else "",
                      "nodo": parts[4] if len(parts) > 4 else None,
                      "trabajo": parts[5] if len(parts) > 5 else None}
    return run


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------
class RoiInputError(ValueError):
    """An ROI input cannot be used: main() reports it and exits with code 2 (unusable input)."""


# keys of R_int-congelada.json that load_roi reads (keep in sync with it)
FROZEN_SPEC_KEYS = (("texto", "sha256"), ("hash_contenido", "valor"), ("origen", "sha256_archivo"))


def read_frozen_spec(path):
    """R_int-congelada.json as a dict. RoiInputError when it is not valid JSON (or not UTF-8), is not an object or
    lacks a key that the K0 checks of load_roi read; a missing or unreadable file stays an OSError."""
    try:
        spec = C.read_json(path)
    except ValueError as exc:               # json.JSONDecodeError and UnicodeDecodeError are ValueErrors
        raise RoiInputError("%s no es un JSON válido: %s" % (path, exc))
    for keys in FROZEN_SPEC_KEYS:
        node = spec
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise RoiInputError("%s no tiene la clave %s" % (path, ".".join(keys)))
            node = node[k]
    return spec


# arrays of roi-mascaras.npz that load_roi reads, in the priority order of the class map (highest first)
ROI_ARRAYS = ("R_int", "R_cond", "R_body", "R_rock", "R_sky")


def read_roi_npz(path):
    """(raw bytes, {name: array}) of roi-mascaras.npz, with every array of ROI_ARRAYS present, boolean or numeric and
    of the shape of the pixel maps. RoiInputError when the file is not a readable .npz archive (empty, truncated, junk
    or corrupt bytes, a bare .npy), lacks one of the arrays or holds it with another shape or type; a missing or
    unreadable file stays an OSError."""
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        if not hasattr(z, "files"):
            raise ValueError("es un array .npy suelto")
        with z:
            arrays = {k: np.array(z[k]) for k in z.files}
    except Exception as exc:  # noqa: BLE001 - only in-memory bytes are decoded here; damaged archives also raise
        # tokenize.TokenError (the .npy header parser), NotImplementedError (zip version or compression method),
        # RuntimeError (the encryption flag) or MemoryError (a forged huge shape), besides ValueError, OSError,
        # EOFError, zipfile.BadZipFile and zlib.error
        raise RoiInputError("%s no es un .npz legible (%s: %s)" % (path, type(exc).__name__, exc))
    for name in ROI_ARRAYS:
        arr = arrays.get(name)
        if arr is None:
            raise RoiInputError("%s no tiene el array %s" % (path, name))
        if arr.shape != (C.N_EL, C.N_AZ) or arr.dtype.kind not in "biuf":
            raise RoiInputError("%s: el array %s es de tipo %s y forma %s; se esperaba una máscara booleana o numérica "
                                "de forma %s" % (path, name, arr.dtype, arr.shape, (C.N_EL, C.N_AZ)))
    return raw, arrays


class RoiContext:
    def __init__(self):
        self.rint = None
        self.text_sha = None
        self.content_sha = None
        self.npz_sha = None
        self.class_map = None            # (256, 256) int8: 0..4 classes, 5 = sin clase
        self.masks = {}
        self.overlaps = 0


def load_roi(mask_path, frozen_path, npz_path, test_mask, contracts):
    roi = RoiContext()
    with open(mask_path, "rb") as fh:
        data = fh.read()
    roi.text_sha = C.sha256_bytes(data)
    try:
        roi.rint = C.mask_from_text(data)
    except ValueError as exc:
        contracts.add("K0", "mascara", "la máscara de texto tiene el formato esperado", False, True,
                      {"error": str(exc)})
        return roi
    roi.content_sha = C.mask_content_sha256(roi.rint)
    spec = read_frozen_spec(frozen_path)
    contracts.add("K0", "mascara", "sha256 del texto de R_int == R_int-congelada.json",
                  roi.text_sha == spec["texto"]["sha256"], True, {"sha256": roi.text_sha})
    contracts.add("K0", "mascara", "sha256 de R_int.tobytes() == R_int-congelada.json",
                  roi.content_sha == spec["hash_contenido"]["valor"], True, {"sha256": roi.content_sha})
    if not test_mask:
        s = C.mask_summary(roi.rint)
        frozen_ok = (roi.content_sha == C.FROZEN_RINT_CONTENT_SHA256 and
                     roi.text_sha == C.FROZEN_RINT_TEXT_SHA256 and
                     s["pixeles"] == C.FROZEN_RINT_PIXELS and
                     tuple(s["filas"]) == C.FROZEN_RINT_ROWS and
                     tuple(s["columnas"]) == C.FROZEN_RINT_COLS and
                     spec["origen"]["sha256_archivo"] == C.FROZEN_NPZ_SHA256)
        contracts.add("K0", "mascara", "R_int coincide con los valores congelados del plan "
                      "(195 píxeles, filas 17-50, columnas 131-138)", frozen_ok, True, s)
    if npz_path:
        raw, arrays = read_roi_npz(npz_path)
        roi.npz_sha = C.sha256_bytes(raw)
        contracts.add("K0", "mascara", "sha256 del npz == R_int-congelada.json",
                      roi.npz_sha == spec["origen"]["sha256_archivo"], True, {"sha256": roi.npz_sha})
        contracts.add("K0", "mascara", "R_int del npz == R_int del texto",
                      bool(np.array_equal(arrays["R_int"], roi.rint)), True, {})
        cmap = np.full((C.N_EL, C.N_AZ), 5, dtype=np.int8)
        for k in range(len(ROI_ARRAYS) - 1, -1, -1):        # highest priority written last
            m = arrays[ROI_ARRAYS[k]].astype(bool)
            if k == 0:
                m = roi.rint
            roi.masks[ROI_ARRAYS[k]] = m
            cmap[m] = k
        stack = np.stack([roi.masks[k] for k in ROI_ARRAYS]).sum(axis=0)
        roi.overlaps = int((stack > 1).sum())
        roi.class_map = cmap
    else:
        cmap = np.full((C.N_EL, C.N_AZ), 5, dtype=np.int8)
        cmap[roi.rint] = 0
        roi.class_map = cmap
    return roi


# --------------------------------------------------------------------------
# K6
# --------------------------------------------------------------------------
def parse_sha256sums(text):
    out = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def expected_solid_types(entry):
    if entry["caso"] == CASE_AFFECTED:
        return {"anomaly_0": VARIANT_SOLIDS[entry["variante"]], "anomaly_1": "G4Ellipsoid"}
    return {}


def efficiency_z(caso, n_det, n):
    """z of the detection efficiency against the campaign's (binomial sigma), None for an unknown case."""
    ref = EFFICIENCY_REFERENCE.get(caso)
    if not ref:
        return None
    return (n_det / n - ref) / math.sqrt(ref * (1 - ref) / n)


def hex_column_state(col):
    """(uint64 values, valid mask, evidence) of a prim_fnv / rng0 column of 16-digit hex strings."""
    values, valid = C.hex16_column_to_uint64(col)
    good = values[valid]
    evid = {"filas": int(len(values)), "no_hexadecimales_de_16_digitos": int((~valid).sum()),
            "ceros": int((good == 0).sum()), "valores_distintos": int(len(np.unique(good)))}
    return values, valid, evid


def k6_checks(run, roi, contracts, lists=None):
    """K6 for one run. ``lists`` (a ListCache) lets the check recompute prim_fnv from the list rows; without
    it that one check is reported as not verifiable."""
    e = run.entry
    scope = run.run_id
    n = e["N"]
    blk = True

    def add(desc, ok, evid=None, blocking=True):
        return contracts.add("K6", scope, desc, ok, blocking, evid)

    add("archivos presentes (hist, meta, _muones.csv), sin .part", not run.problems and not run.part_files and
        run.hist is not None and run.meta is not None and run.mu is not None,
        {"problemas": run.problems, "part": [os.path.basename(p) for p in run.part_files]})
    if run.mu is None or run.meta is None or run.hist is None:
        return
    mu, meta, hist = run.mu, run.meta, run.hist
    # SHA256SUMS of the run folder
    if "SHA256SUMS" in run.files:
        listed = parse_sha256sums(read_text(run.files["SHA256SUMS"]))
        bad = []
        for name, sha in listed.items():
            p = os.path.join(run.dir, name)
            if not os.path.isfile(p) or C.sha256_file(p) != sha:
                bad.append(name)
        needed = [os.path.basename(run.files[k]) for k in ("hist", "meta", "muones") if k in run.files]
        missing = [x for x in needed if x not in listed]
        add("SHA256SUMS de la corrida coincide con los archivos", not bad and not missing,
            {"no_coinciden": bad, "sin_listar": missing})
    else:
        add("SHA256SUMS de la corrida coincide con los archivos", False, {"error": "falta SHA256SUMS"})
    # driver verdict
    if run.status is None:
        add("estado del guion de lanzamiento: OK con código de salida 0", False,
            {"error": "no hay archivo estado/%s.status" % run.run_id})
    else:
        add("estado del guion de lanzamiento: OK con código de salida 0",
            run.status["veredicto"] == "OK" and run.status["rc"] == "0", run.status)
    oe = meta.get("oe1_etapa1")
    add("la clave oe1_etapa1 está en el meta (el binario nuevo, no uno viejo)", isinstance(oe, dict) and
        oe.get("format") == "oe1-etapa1/1" and oe.get("mode") == "list", {"presente": oe is not None})
    if not isinstance(oe, dict):
        return
    checks = oe.get("checks", {})
    need = ("ids_complete_unique", "hist_reconstructed_equal", "detected_equal", "events_equal",
            "generated_equal", "writes_ok")
    add("las comprobaciones del binario son todas verdaderas y no hay errores",
        all(checks.get(k) is True for k in need) and oe.get("errors") == [],
        {"checks": checks, "errors": oe.get("errors")})
    # rows / ids
    ids = mu["id"]
    st = mu["status"]
    rows_ok = (len(ids) == n and meta.get("events_transported") == n and
               meta.get("ecomug_draws_total") == n and oe.get("list", {}).get("n") == n and
               oe.get("muon_log", {}).get("rows") == n)
    add("filas = N = events_transported = ecomug_draws_total = N de la lista", rows_ok,
        {"filas": int(len(ids)), "N": n, "events_transported": meta.get("events_transported"),
         "ecomug_draws_total": meta.get("ecomug_draws_total")})
    ids_ok = bool(np.array_equal(ids, np.arange(len(ids)))) and len(ids) == n
    add("IDs únicos, ordenados y 0..N-1", ids_ok, {})
    # prim_fnv and rng0 must be real values. A column of zeros, or of one repeated value, is identical in A and
    # B (and in an A/A repeat), so K9 (a), K7 and K8 would pass on nothing. Every muon has its own primary and
    # starts from its own engine state (its own key {id, s_tr, case, tag}): N different non-zero values each.
    prim, prim_valid, prim_evid = hex_column_state(mu["prim_fnv"])
    rng0, rng0_valid, rng0_evid = hex_column_state(mu["rng0"])

    def real_values(evid):
        return (evid["filas"] == n and evid["no_hexadecimales_de_16_digitos"] == 0 and evid["ceros"] == 0 and
                evid["valores_distintos"] == n)

    add("prim_fnv y rng0: 16 dígitos hexadecimales, ninguno nulo y N valores distintos cada uno (una columna "
        "vacía o constante haría pasar K7, K8 y K9 (a) en vacío)", real_values(prim_evid) and real_values(rng0_evid),
        {"N": n, "prim_fnv": prim_evid, "rng0": rng0_evid})
    # prim_fnv recomputed from the rows of the located list, for ALL ids (vectorised; the recipe of
    # EventAction.cc / the README): it proves that the muon the binary transported is the listed one, which
    # equality between A and B alone cannot (both could have run the wrong list).
    desc_fnv = "prim_fnv recalculado desde las filas de la lista == columna prim_fnv, para todo id"
    if lists is None:
        add(desc_fnv + " (no verificable: no se pasaron listas)", None, {}, blocking=False)
    else:
        lst = lists.get(e)
        if lst is None:
            add(desc_fnv, False, {"error": lists.problem(e)})
        else:
            table = lst["table"]
            ids_list_ok = (len(table["id"]) == n and bool(np.array_equal(table["id"], np.arange(n))) and
                           bool(np.array_equal(mu["id"], np.arange(n))))
            if not ids_list_ok:
                add(desc_fnv, False, {"error": "los IDs de la lista o del registro no son 0..N-1",
                                      "filas_de_la_lista": int(len(table["id"])), "N": n})
            else:
                fnv_expected = C.primary_fnv_array(table["vals"], table["pdg"])
                differ = ~prim_valid | (prim != fnv_expected)
                first = np.flatnonzero(differ)[:5]
                add(desc_fnv, not bool(differ.any()),
                    {"filas_que_difieren": int(differ.sum()), "N": n,
                     "primeras": [{"id": int(k), "columna": C.hex16(int(prim[k])),
                                   "recalculado": C.hex16(int(fnv_expected[k]))} for k in first]})
    counts = {s: int((st == s.encode("ascii")).sum()) for s in C.STATUS_NAMES}
    add("ninguna fila ABORTED", counts["ABORTED"] == 0, {"ABORTED": counts["ABORTED"]})
    add("los conteos por estado del meta coinciden con las filas",
        oe.get("counts", {}).get("status") == counts, {"filas": counts, "meta": oe.get("counts", {}).get("status")})
    # histogram reconstruction
    det = st == b"DET"
    iel, iaz = mu["iel"], mu["iaz"]
    in_range = bool(((iel[det] >= 0) & (iel[det] < C.N_EL) & (iaz[det] >= 0) & (iaz[det] < C.N_AZ)).all())
    if in_range:
        recon = np.bincount(iel[det] * C.N_AZ + iaz[det], minlength=C.N_EL * C.N_AZ).reshape(C.N_EL, C.N_AZ)
        equal = bool(np.array_equal(recon, hist))
        diff_cells = int((recon != hist).sum())
    else:
        equal, diff_cells = False, None
    add("histograma reconstruido con las filas DET igual a hist.csv en las 65 536 celdas", equal,
        {"celdas_distintas": diff_cells})
    n_det = int(det.sum())
    add("número de filas DET = detected_in_bins = suma de hist.csv",
        n_det == meta.get("detected_in_bins") == int(hist.sum()),
        {"DET": n_det, "detected_in_bins": meta.get("detected_in_bins"), "suma_hist": int(hist.sum())})
    add("la columna detected vale 1 exactamente en las filas DET",
        bool(np.array_equal(mu["detected"], det.astype(np.int64))), {})
    # sanity of the detection efficiency against the campaign's (informative): a value far from it points to a
    # wrong list, geometry or binary that the structural checks above could not see
    z_eff = efficiency_z(e["caso"], n_det, n)
    add("eficiencia de detección compatible con la de la campaña (|z| <= %g; informativo, no bloquea)" %
        EFFICIENCY_Z_LIMIT, None if z_eff is None else abs(z_eff) <= EFFICIENCY_Z_LIMIT,
        {"eficiencia": n_det / n, "referencia_campana": EFFICIENCY_REFERENCE.get(e["caso"]), "z": z_eff},
        blocking=False)
    # in_rint against the offline value
    expected = np.full(len(ids), -1, dtype=np.int64)
    if in_range:
        expected[det] = roi.rint[iel[det], iaz[det]].astype(np.int64)
    same = bool(np.array_equal(mu["in_rint"], expected))
    add("in_rint del binario == valor recalculado fuera de línea (100 %)", same,
        {"distintas": int((mu["in_rint"] != expected).sum())})
    if in_range:
        s_hist = int(hist[roi.rint].sum())
        s_rows = int((expected == 1).sum())
        add("hist[R_int].sum() == número de filas con in_rint = 1", s_hist == s_rows,
            {"hist": s_hist, "filas": s_rows})
    # lateral displacement (PLAN correction C1, 2026-09-30). WentzelVI displaces the track laterally inside a
    # step, so the pre->post chord is NOT the navigated path: max_lat_mm is > 1e-6 mm for ~89 % of the muons
    # and reaches metres for the slowest ones. It is reported, not required to be small. The SCALE of the
    # error of the length in Z (not a bound) is sum_lat_z_mm, which the binary records for the steps near Z.
    lat = mu["max_lat_mm"]
    steps = mu["n_steps"]
    have = ~np.isnan(lat)
    lat_max = float(lat[have].max()) if have.any() else None
    meta_lat = oe.get("counts", {}).get("max_lat_mm")
    lat_ok = (bool(have[steps > 0].all()) and bool(np.isfinite(lat[have]).all()) and
              bool((lat[have] >= 0).all()) and lat_max is not None and isinstance(meta_lat, (int, float)) and
              abs(lat_max - meta_lat) <= 1e-9 * max(abs(lat_max), abs(meta_lat)))
    add("desplazamiento lateral reportado: max_lat_mm finito y presente en toda fila con pasos, con el mismo "
        "máximo que el meta (diagnóstico: ya no se exige < 1e-6 mm)", lat_ok,
        {"max_lat_mm": lat_max, "meta_max_lat_mm": meta_lat, "filas_sin_dato": int((~have & (steps > 0)).sum()),
         "mediana_mm": float(np.median(lat[have])) if have.any() else None,
         "p99_mm": float(np.percentile(lat[have], 99)) if have.any() else None,
         "fraccion_mayor_que_1e-6_mm": float((lat[have] > MAX_LAT_LIMIT_MM).mean()) if have.any() else None})
    z_cols = [c for c in C.MUON_OPTIONAL_COLUMNS if c in mu]
    counts_meta = oe.get("counts", {})
    # the corrected binary (plan C1) always writes the two columns and the two aggregates below; an older
    # binary writes neither. A meta that declares the aggregates makes the columns mandatory for the run.
    declares_z = "max_lat_z_mm" in counts_meta or "max_sum_lat_z_mm" in counts_meta
    if len(z_cols) == len(C.MUON_OPTIONAL_COLUMNS):
        mz, sz = mu["max_lat_z_mm"], mu["sum_lat_z_mm"]
        both = ~np.isnan(mz) & ~np.isnan(sz)
        z_ok = (bool(np.isfinite(mz[~np.isnan(mz)]).all()) and bool(np.isfinite(sz[~np.isnan(sz)]).all()) and
                bool((mz[~np.isnan(mz)] >= 0).all()) and bool((sz[~np.isnan(sz)] >= 0).all()) and
                bool((np.isnan(mz) == np.isnan(sz)).all()) and
                bool((sz[both] >= mz[both] * (1.0 - 1e-9)).all()))
        evid = {"max_lat_z_mm": float(np.nanmax(mz)) if (~np.isnan(mz)).any() else None,
                "max_sum_lat_z_mm": float(np.nanmax(sz)) if (~np.isnan(sz)).any() else None,
                "filas_con_dato": int(both.sum())}
        desc = ("escala del error de la longitud en Z (Σ del desplazamiento transversal, no cota): max_lat_z_mm y "
                "sum_lat_z_mm finitos, no negativos y con "
                "sum_lat_z_mm >= max_lat_z_mm en cada fila")
        if declares_z:
            # with a region (geo-train-06) every row has both values (0 when no step was near Z); without
            # one (geo-train-01) none has; the steps near Z are some of all the steps, so max_lat_z_mm can
            # not exceed max_lat_mm; and the aggregates of the meta are the maxima of the columns
            has_region = e["caso"] == CASE_AFFECTED
            pattern_ok = bool(both.all()) if has_region else bool((~both).all())
            with_lat = both & ~np.isnan(lat)
            pattern_ok = (pattern_ok and bool((mz[with_lat] <= lat[with_lat] * (1.0 + 1e-12)).all()) and
                          bool((mz[both & np.isnan(lat)] == 0.0).all()))

            def same_max(col, meta_val):
                top = float(np.nanmax(col)) if (~np.isnan(col)).any() else None
                if top is None or meta_val is None:
                    return top is None and meta_val is None
                return (isinstance(meta_val, (int, float)) and not isinstance(meta_val, bool) and
                        abs(top - meta_val) <= 1e-9 * max(abs(top), abs(meta_val)))

            meta_ok = (same_max(mz, counts_meta.get("max_lat_z_mm")) and
                       same_max(sz, counts_meta.get("max_sum_lat_z_mm")))
            z_ok = z_ok and pattern_ok and meta_ok
            evid.update({"meta_max_lat_z_mm": counts_meta.get("max_lat_z_mm"),
                         "meta_max_sum_lat_z_mm": counts_meta.get("max_sum_lat_z_mm"),
                         "filas_coherentes": pattern_ok, "maximos_iguales_al_meta": meta_ok})
            desc += ("; con valores en todas las filas si hay región, max_lat_z_mm <= max_lat_mm y los "
                     "máximos iguales a los del meta")
        add(desc, z_ok, evid)
    elif declares_z:
        add("escala del error de la longitud en Z: el meta declara max_lat_z_mm / max_sum_lat_z_mm pero el "
            "registro no tiene las columnas max_lat_z_mm y sum_lat_z_mm", False,
            {"columnas_ausentes": [c for c in C.MUON_OPTIONAL_COLUMNS if c not in mu]})
    else:
        add("escala del error de la longitud en Z: el binario no registra max_lat_z_mm ni sum_lat_z_mm "
            "(no verificable; las longitudes en Z quedan sin escala de error)", None,
            {"columnas_ausentes": [c for c in C.MUON_OPTIONAL_COLUMNS if c not in mu]}, blocking=False)
    # geometry warnings
    log = run.log or ""
    codes = C.count_warning_codes(log)
    if e["variante"] == "B":
        clean = not codes and "GeomVol1002" not in log and "GeomNav" not in log and "G4Exception" not in log
        add("B sin GeomVol1002 / GeomNav / G4Exception en el registro", clean, {"codigos": codes})
    else:
        exp_txt = e["avisos_esperados"].strip() or "ninguno"
        add("avisos de A iguales a los esperados por el manifiesto (diagnóstico histórico)",
            C.warnings_as_text(codes) == exp_txt, {"obtenidos": C.warnings_as_text(codes), "esperados": exp_txt},
            blocking=False)
    # region
    region = oe.get("region", {})
    is06 = e["caso"] == CASE_AFFECTED
    crossed = mu["crossed"]
    if is06:
        cross_ok = bool(np.isin(crossed, (0.0, 1.0)).all())
    else:
        cross_ok = bool((crossed == -1.0).all())
    add("region.applicable = %s y crossed %s" % ("true" if is06 else "false",
                                                 "en {0, 1}" if is06 else "= -1 (no aplica)"),
        region.get("applicable") is is06 and cross_ok,
        {"applicable": region.get("applicable"), "crossed_valores": sorted(set(np.unique(crossed).tolist()))})
    # provenance against the manifest
    seeds = oe.get("seeds", {})
    runtime = oe.get("runtime", {})
    lst = oe.get("list", {})
    # exactly one worker thread ran every event (the binary checks the same); its number is whatever the
    # run manager assigned (>= 0, the master is -1), so it is recorded but not required to be 0
    workers = runtime.get("worker_thread_ids")
    prov = {
        "variant": oe.get("variant") == e["variante"],
        "case_code": oe.get("case_code") == e["case_code"],
        "replica": oe.get("replica") == e["replica"],
        "lista_sha256": lst.get("sha256") == e["lista_sha256"] and lst.get("sha256_verified") is True,
        "semilla_transporte": seeds.get("transport_seed") == e["s_tr"],
        "semilla_maestra": seeds.get("master_cli") == e["s_tr"] and meta.get("seed") == e["s_tr"],
        "fuente": lst.get("source_seed") == e["s_src"],
        "geometria": meta.get("geometry_id") == e["caso"],
        "mascara_sha256": oe.get("roi_mask", {}).get("sha256") == roi.text_sha,
        "gestor": "G4TaskRunManager" in str(runtime.get("run_manager_class", "")),
        "hilos": runtime.get("threads_cli") == 1 and meta.get("threads") == 1,
        "trabajadores": (isinstance(workers, list) and len(workers) == 1 and isinstance(workers[0], int) and
                         not isinstance(workers[0], bool) and workers[0] >= 0),
        "motor": runtime.get("worker_engines") == ["MixMaxRng"] and runtime.get("master_engine") == "MixMaxRng",
        "log_completo": oe.get("muon_log", {}).get("level") == "full",
    }
    const = {
        "caso": bool((mu["caso"] == e["case_code"]).all()),
        "geometry_id": bool((mu["geometry_id"] == e["caso"].encode()).all()),
        "variant": bool((mu["variant"] == e["variante"].encode()).all()),
        "replica": bool((mu["replica"] == e["replica"]).all()),
        "list_sha256": bool((mu["list_sha256"] == e["lista_sha256"].encode()).all()),
        "view": bool((mu["view"] == 0).all()),
    }
    prov.update({"columna_" + k: v for k, v in const.items()})
    add("procedencia coherente con el manifiesto (variante, caso, réplica, lista, semillas, un hilo, "
        "MixMaxRng, G4TaskRunManager, máscara)", all(prov.values()),
        {"falla": [k for k, v in prov.items() if not v], "worker_thread_ids": workers,
         "run_manager_class": runtime.get("run_manager_class")})
    # geometry signature against the variant
    sig = {}
    for entry_ in oe.get("geometry_signature", {}).get("volumes", []):
        parts = entry_.split(":")
        if len(parts) >= 3:
            sig[parts[0]] = parts[1]
    want = expected_solid_types(e)
    sig_ok = all(sig.get(k) == v for k, v in want.items())
    if not is06:
        sig_ok = sig_ok and not any(k.startswith("anomaly_") for k in sig)
    add("la firma de volúmenes corresponde a la variante %s (anomaly_0 = %s)" % (
        e["variante"], VARIANT_SOLIDS[e["variante"]] if is06 else "no aplica"), sig_ok,
        {"firma": {k: v for k, v in sig.items() if k.startswith("anomaly_")}, "esperada": want})
    # muon log integrity
    meta_log = oe.get("muon_log", {})
    fnv_ok = None
    if run.fnv_muones is not None:
        fnv_ok = C.hex16(run.fnv_muones) == meta_log.get("fnv1a64")
    if fnv_ok is not None:
        add("FNV-1a-64 de _muones.csv coincide con muon_log del meta (lo que el binario escribió)", fnv_ok,
            {"fnv_calculado": C.hex16(run.fnv_muones), "fnv_meta": meta_log.get("fnv1a64")})
    else:
        add("FNV-1a-64 de _muones.csv no recalculado (--skip-muon-fnv)", None,
            {"fnv_meta": meta_log.get("fnv1a64")}, blocking=False)
    add("tamaño y número de columnas de _muones.csv coinciden con muon_log del meta",
        meta_log.get("bytes") == run.muones_len and meta_log.get("columns") == len(C.muon_columns_present(mu)),
        {"bytes": run.muones_len, "meta_bytes": meta_log.get("bytes"), "columnas": len(C.muon_columns_present(mu)),
         "meta_columnas": meta_log.get("columns")})
    # run.log lines
    done = C.find_line(log, "RUN DONE")
    summ = C.find_line(log, "OE1-ETAPA1")
    kv = C.parse_kv_line(summ) if summ else {}
    add("run.log contiene RUN DONE y la línea OE1-ETAPA1 con status=OK y errors=none",
        bool(done) and bool(summ) and kv.get("status") == "OK" and kv.get("errors") == "none" and
        kv.get("n") == str(n) and kv.get("rows") == str(n),
        {"RUN DONE": done, "OE1-ETAPA1": summ})
    # mean kinetic energy (informative)
    total = 0.0
    for v in mu["ekin_plane_gev"][det]:
        total += float(v)
    mean = total / n_det if n_det else 0.0
    ref = meta.get(MEAN_EKIN_KEY, 0.0)
    rel = abs(mean - ref) / ref if ref else abs(mean - ref)
    add("energía cinética media en el plano == meta (informativo)", rel <= EPS_MEAN_EKIN,
        {"recalculada": mean, "meta": ref, "diferencia_relativa": rel}, blocking=False)


# --------------------------------------------------------------------------
# K7, K8, K9
# --------------------------------------------------------------------------
def meta_diff(a, b, ignore_paths=()):
    """Dotted paths whose values differ (dict/list aware)."""
    diffs = []

    def walk(x, y, path):
        if path in ignore_paths:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    if (path + "." + k if path else k) not in ignore_paths:
                        diffs.append(path + "." + k if path else k)
                else:
                    walk(x[k], y[k], path + "." + k if path else k)
        elif x != y:
            diffs.append(path)

    walk(a, b, "")
    return diffs


def mask_log(text):
    return re.sub(r"wall_s=\S+", "wall_s=<t>", text or "")


def k7_checks(base, rep, contracts):
    scope = "%s vs %s" % (base.run_id, rep.run_id)
    if base.hist is None or rep.hist is None or base.meta is None or rep.meta is None:
        contracts.add("K7", scope, "A/A: las dos corridas se pudieron leer", False, True, {})
        return
    contracts.add("K7", scope, "hist.csv idéntico byte a byte",
                  base.sha.get("hist") == rep.sha.get("hist"), True,
                  {"sha_a": base.sha.get("hist"), "sha_rep": rep.sha.get("hist")})
    contracts.add("K7", scope, "_muones.csv idéntico byte a byte",
                  base.sha.get("muones") == rep.sha.get("muones"), True,
                  {"sha_a": base.sha.get("muones"), "sha_rep": rep.sha.get("muones")})
    d = meta_diff(C.meta_without_timing(base.meta), C.meta_without_timing(rep.meta))
    contracts.add("K7", scope, "meta.json igual salvo los tiempos y el uso de recursos (%s)" %
                  ", ".join(C.TIMING_META_KEYS), not d, True,
                  {"campos_distintos": d,
                   "claves_de_tiempo_ignoradas": sorted(set(C.timing_meta_paths(base.meta)) |
                                                        set(C.timing_meta_paths(rep.meta)))})
    same_log = mask_log(base.log) == mask_log(rep.log)
    contracts.add("K7", scope, "run.log igual una vez enmascarado wall_s= (informativo)", same_log, False, {})
    nodes = (base.status or {}).get("nodo"), (rep.status or {}).get("nodo")
    if nodes[0] and nodes[1]:
        contracts.add("K7", scope, "A/A en el mismo nodo", nodes[0] == nodes[1], True, {"nodos": list(nodes)})
    else:
        contracts.add("K7", scope, "A/A en el mismo nodo (no verificable: falta el nodo)", None, False, {})


def k8_checks(a, b, contracts):
    scope = "%s vs %s" % (a.run_id, b.run_id)
    if a.hist is None or b.hist is None or a.mu is None or b.mu is None:
        contracts.add("K8", scope, "control: las dos corridas se pudieron leer", False, False, {})
        return
    contracts.add("K8", scope, "hist.csv idéntico entre A y B", a.sha.get("hist") == b.sha.get("hist"),
                  False, {})
    cols_a, cols_b = C.muon_columns_present(a.mu), C.muon_columns_present(b.mu)
    diff_cols = [c for c in cols_a if c != "variant" and c in b.mu and not _col_equal(a.mu[c], b.mu[c])]
    diff_cols += [c for c in cols_a + cols_b if (c in a.mu) != (c in b.mu) and c not in diff_cols]
    contracts.add("K8", scope, "_muones.csv idéntico salvo la columna variant", not diff_cols, False,
                  {"columnas_distintas": diff_cols})
    # muon_log.fnv1a64 is the hash of a file that carries the variant column, so it
    # differs by construction; the columns themselves are compared above. binary_sha256 is the
    # hash of the executable (plan S2): the A and B binaries differ by construction (the SHA-256
    # of the geometry JSON and of the tables, in oe1_etapa1.inputs, must be equal).
    ignored = ("oe1_etapa1.variant", "oe1_etapa1.muon_log.fnv1a64", "oe1_etapa1.binary_sha256")
    d = meta_diff(C.meta_without_timing(a.meta), C.meta_without_timing(b.meta), ignored)
    contracts.add("K8", scope, "meta.json igual salvo tiempos, oe1_etapa1.variant, el SHA-256 del binario y el "
                  "FNV del registro (que incluye la columna variant)", not d, False,
                  {"campos_distintos": d,
                   "claves_de_tiempo_ignoradas": sorted(set(C.timing_meta_paths(a.meta)) |
                                                        set(C.timing_meta_paths(b.meta))),
                   "otras_rutas_ignoradas": list(ignored)})


def _col_equal(x, y):
    if x.dtype.kind == "f":
        return bool(np.array_equal(x, y, equal_nan=True)) if hasattr(np, "array_equal") else False
    return bool(np.array_equal(x, y))


def classify_discordance(a, b, idx):
    """DV9 class of each discordant id (geo-train-06): array of str."""
    lc = np.maximum(np.nan_to_num(a.mu["lc_m"][idx]), np.nan_to_num(b.mu["lc_m"][idx]))
    le = np.maximum(np.nan_to_num(a.mu["le_m"][idx]), np.nan_to_num(b.mu["le_m"][idx]))
    graze = (np.nan_to_num(a.mu["n_graze_z"][idx]) > 0) | (np.nan_to_num(b.mu["n_graze_z"][idx]) > 0)
    cross = (a.mu["crossed"][idx] == 1) | (b.mu["crossed"][idx] == 1)
    endvol_a0 = (a.mu["end_vol"][idx] == b"anomaly_0") | (b.mu["end_vol"][idx] == b"anomaly_0")
    out = np.empty(len(idx), dtype=object)
    for k in range(len(idx)):
        if cross[k]:
            out[k] = "cruza_Z"
        elif graze[k]:
            out[k] = "roce"
        elif lc[k] > 0 or endvol_a0[k]:
            out[k] = "conducto"
        elif le[k] > 0:
            out[k] = "solo_elipsoide"
        else:
            out[k] = "otra"
    return out


def k9a_checks(a, b, contracts, control=False):
    """K9 (a), blocking in the affected case AND in the control: the same ids, the same primary (prim_fnv) and
    the same random start (rng0) for every id. Equal columns only prove something when they hold N different
    non-zero values, so that is required too (K6 checks each run; a constant or empty column would otherwise
    make A equal B by construction). Returns whether the ids agree, or None if a run could not be read."""
    scope = "%s vs %s%s" % (a.run_id, b.run_id, " (control)" if control else "")
    if a.mu is None or b.mu is None:
        contracts.add("K9", scope, "(a) las dos corridas se pudieron leer", False, True, {})
        return None
    n = len(a.mu["id"])
    same_ids = bool(np.array_equal(a.mu["id"], b.mu["id"]))
    verdicts = {}
    for col, label in (("prim_fnv", "prim_fnv"), ("rng0", "rng0")):
        _, _, ea = hex_column_state(a.mu[col])
        _, _, eb = hex_column_state(b.mu[col])
        real = all(x["filas"] == n and x["no_hexadecimales_de_16_digitos"] == 0 and x["ceros"] == 0 and
                   x["valores_distintos"] == n for x in (ea, eb))
        equal = same_ids and bool(np.array_equal(a.mu[col], b.mu[col]))
        verdicts[col] = (equal and real, {
            label + "_distintos": int((a.mu[col] != b.mu[col]).sum()) if same_ids else None,
            "valores_reales_y_distintos_en_A_y_B": real, "A": ea, "B": eb})
    contracts.add("K9", scope, "(a) mismos IDs y prim_fnv igual para todo id (N valores distintos, no nulos)",
                  verdicts["prim_fnv"][0], True, verdicts["prim_fnv"][1])
    contracts.add("K9", scope, "(a) rng0 igual para todo id (N valores distintos, no nulos)",
                  verdicts["rng0"][0], True, verdicts["rng0"][1])
    return same_ids


def lateral_length_scale_m(sum_lat_z_mm):
    """2 * sqrt(2 * R_max * sum_lat_z): the largest length error a chord that skims a surface of radius of
    curvature R_max can make for an accumulated transverse displacement sum_lat_z (mm in, metres out). It is a
    scale, not a bound (plan C1, step 3)."""
    return 2.0 * np.sqrt(2.0 * R_MAX_M * np.nan_to_num(np.asarray(sum_lat_z_mm, dtype=np.float64)) * 1e-3)


def k9c_checks(a, b, contracts):
    """K9 (c): in B nothing of anomaly_0 or of the rock may lie inside Z (lz_a0 and lz_rock <= 1 um). The lengths
    are chord lengths, and a chord can clip Z although the navigated path did not (plan C1), so a violation whose
    size stays within the lateral-displacement scale of its own steps near Z, lz_a0 + lz_rock <=
    2 sqrt(2 R_max sum_lat_z) with R_max = 72.9^2 / 61.6 = 86.27 m, is a NON-BLOCKING class ("explicado por
    desplazamiento lateral", reported, ids listed, holds back the attribution until read). Any other violation
    blocks."""
    scope = "%s vs %s (B)" % (a.run_id, b.run_id)
    a0, rock = b.mu["lz_a0_m"], b.mu["lz_rock_m"]
    have = ~np.isnan(a0) & ~np.isnan(rock)
    complete = len(a0) > 0 and bool(have.all())
    a0z, rockz = np.where(have, a0, 0.0), np.where(have, rock, 0.0)
    viol = have & ((a0z > LZ_TOL_UM_M) | (rockz > LZ_TOL_UM_M))
    excess = a0z + rockz
    scale = lateral_length_scale_m(b.mu["sum_lat_z_mm"]) if "sum_lat_z_mm" in b.mu else np.zeros(len(a0))
    explained = viol & (excess <= scale)
    unexplained = viol & ~explained
    ids = b.mu["id"]
    contracts.add("K9", scope, "(c) en B lz_a0 <= 1 µm y lz_rock <= 1 µm para todo id, salvo lo que explica el "
                  "desplazamiento lateral (contrato no bloqueante siguiente)", complete and not bool(unexplained.any()),
                  True, {"max_lz_a0_m": float(a0z.max()) if len(a0) else None,
                         "max_lz_rock_m": float(rockz.max()) if len(rock) else None,
                         "valores_ausentes": int((~have).sum()), "violaciones": int(viol.sum()),
                         "sin_explicar": int(unexplained.sum()),
                         "ids_sin_explicar": [int(k) for k in ids[unexplained][:20]]})
    contracts.add("K9", scope, "(c) violaciones de 1 µm en B explicadas por el desplazamiento lateral "
                  "(lz_a0 + lz_rock <= 2·√(2·R_max·Σlat), R_max = %g m; clase «explicado por desplazamiento "
                  "lateral», no bloquea)" % R_MAX_M, not bool(explained.any()), False,
                  {"explicados": int(explained.sum()), "ids_explicados": [int(k) for k in ids[explained][:20]],
                   "exceso_max_m": float(excess[explained].max()) if explained.any() else None,
                   "escala_max_m": float(scale[explained].max()) if explained.any() else None,
                   "R_max_m": R_MAX_M,
                   "nota": "escala del error de la cuerda, no cota (plan C1); las columnas sum_lat_z_mm de B "
                           "son necesarias" + ("" if "sum_lat_z_mm" in b.mu else ": AUSENTES, nada se explica")})


def k9_checks(a, b, contracts):
    """K9 of one geo-train-06 pair: (a) blocking, (c) blocking unless explained, and the accounting identity of A."""
    scope = "%s vs %s" % (a.run_id, b.run_id)
    same_ids = k9a_checks(a, b, contracts)
    if same_ids is None:
        return None
    k9c_checks(a, b, contracts)
    # accounting identity of A (informative)
    ident = a.mu["lz_a0_m"] + a.mu["lz_a1_m"] + a.mu["lz_rock_m"] + a.mu["lz_other_m"] - a.mu["lz_m"]
    contracts.add("K9", scope + " (A)", "en A lz_a0 + lz_a1 + lz_rock + lz_other = lz (identidad, informativo)",
                  bool(np.nanmax(np.abs(ident)) <= 1e-9) if len(ident) else False, False,
                  {"max_abs": float(np.nanmax(np.abs(ident))) if len(ident) else None})
    return same_ids


def discordance_table(a, b, roi):
    """Boolean mask of discordant ids and the differing field names."""
    ma, mb = a.mu, b.mu
    fields = collections.OrderedDict()
    fields["detected"] = ma["detected"] != mb["detected"]
    fields["status"] = ma["status"] != mb["status"]
    fields["iel"] = ma["iel"] != mb["iel"]
    fields["iaz"] = ma["iaz"] != mb["iaz"]
    fields["in_rint"] = ma["in_rint"] != mb["in_rint"]
    fields["rng1"] = ma["rng1"] != mb["rng1"]
    anyd = np.zeros(len(ma["id"]), dtype=bool)
    for v in fields.values():
        anyd |= v
    return anyd, fields


# --------------------------------------------------------------------------
# paired quantities
# --------------------------------------------------------------------------
class Pair:
    def __init__(self, caso, replica, a, b):
        self.caso, self.replica, self.a, self.b = caso, replica, a, b
        self.n = a.entry["N"]
        self.det_a = a.mu["status"] == b"DET"
        self.det_b = b.mu["status"] == b"DET"


def y_vector(run, roi):
    """Y_i = 1 when the muon is detected inside R_int (offline mask)."""
    det = run.mu["status"] == b"DET"
    y = np.zeros(len(det), dtype=bool)
    iel = np.where(det, run.mu["iel"], 0)
    iaz = np.where(det, run.mu["iaz"], 0)
    y[det] = roi.rint[iel[det], iaz[det]]
    return y


def interval_block(n01, n10, n, alpha=0.05):
    lo, hi = C.tango_interval(n01, n10, n, alpha)
    mc = C.mcnemar_exact(n01, n10)
    half = 0.5 * (hi - lo)
    return {
        "tango_inf": lo, "tango_sup": hi, "tango_inf_por_millon": per_million(lo),
        "tango_sup_por_millon": per_million(hi), "nivel": 1.0 - alpha,
        "mcnemar_exacto": mc,
        "nota": "diagnóstico; etapa 1 no dimensionada",
        "semiancho": half,
    }


def pair_stats(pair, roi):
    a, b = pair.a, pair.b
    ya = y_vector(a, roi)
    yb = y_vector(b, roi)
    n = pair.n
    n00 = int((~ya & ~yb).sum())
    n01 = int((~ya & yb).sum())
    n10 = int((ya & ~yb).sum())
    n11 = int((ya & yb).sum())
    assert n00 + n01 + n10 + n11 == n
    da, db = pair.det_a, pair.det_b
    m00 = int((~da & ~db).sum())
    m01 = int((~da & db).sum())
    m10 = int((da & ~db).sum())
    m11 = int((da & db).sum())
    res = {"caso": pair.caso, "replica": pair.replica, "N": n, "n00": n00, "n01": n01, "n10": n10,
           "n11": n11, "b": n01 + n10, "delta": (n01 - n10) / n, "sumY_A": int(ya.sum()),
           "sumY_B": int(yb.sum()), "det_A": int(da.sum()), "det_B": int(db.sum()),
           "m00": m00, "m01": m01, "m10": m10, "m11": m11, "delta_eficiencia": (m01 - m10) / n}
    res["delta_por_millon"] = per_million(res["delta"])
    res["denominador"] = denominator_text(n)
    res.update(interval_block(n01, n10, n))
    return res, ya, yb


def class_indices(run, roi):
    det = run.mu["status"] == b"DET"
    cls = np.full(len(det), 6, dtype=np.int64)
    if det.any():
        cls[det] = roi.class_map[run.mu["iel"][det], run.mu["iaz"][det]]
    return cls


def crossing_summary(pair, ya, yb, list_ekin=None):
    a, b = pair.a.mu, pair.b.mu
    ca = a["crossed"] == 1
    cb = b["crossed"] == 1
    both = ca & cb
    union = ca | cb
    nu = int(union.sum())
    out = {"cruces_A": int(ca.sum()), "cruces_B": int(cb.sum()), "cruces_ambas": int(both.sum()),
           "cruces_union": nu, "tamano_U": nu}
    if nu:
        n01 = int((~ya[union] & yb[union]).sum())
        n10 = int((ya[union] & ~yb[union]).sum())
        out["delta_condicional_U"] = (n01 - n10) / nu
        out["n01_U"], out["n10_U"] = n01, n10
        out["nota_U"] = "condicional; denominador |U| = %d; no sustituye a Δ (PROT:69)" % nu
    else:
        out["delta_condicional_U"] = None
        out["n01_U"] = out["n10_U"] = 0
        out["nota_U"] = "|U| = 0: Δ_U no está definido"
    out["lz_total_A_m"] = float(np.nansum(a["lz_m"][ca]))
    out["lz_total_B_m"] = float(np.nansum(b["lz_m"][cb]))
    out["lz_a0_total_A_m"] = float(np.nansum(a["lz_a0_m"]))
    out["lz_a1_total_A_m"] = float(np.nansum(a["lz_a1_m"]))
    out["lz_a1_total_B_m"] = float(np.nansum(b["lz_a1_m"]))
    return out, union


# --------------------------------------------------------------------------
# lists (straight-line cross-check)
# --------------------------------------------------------------------------
def locate_list(entry, lists_dir):
    cands = []
    if lists_dir:
        # the manifest may have been written on another system (POSIX or Windows separators)
        cands.append(os.path.join(lists_dir, re.split(r"[\\/]", entry["lista"])[-1]))
    cands.append(entry["lista"])
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


class ListCache:
    """The primary lists of the manifest, read once. ``get`` returns a list ONLY if it was found and its SHA-256
    equals the manifest row's ``lista_sha256``: a list that does not match is never used (it would give the
    straight-line check, the recomputed prim_fnv and the tables' ekin column a different muon than the one
    that was transported). ``problem`` says why a list was not returned. ``cache`` holds the lists that were
    used (they are the ones the results record)."""

    def __init__(self, lists_dir):
        self.lists_dir = lists_dir
        self.cache = {}          # path -> {"table", "sha256", "sidecar", "path"}: verified and used
        self._read = {}          # path -> the same, whether it matched or not

    def _load(self, path):
        if path not in self._read:
            table, d, sha, canon = C.read_primary_file(path)
            sc, _ = C.read_sidecar(path)
            self._read[path] = {"table": table, "sha256": sha, "sidecar": sc, "path": path}
        return self._read[path]

    def state(self, entry):
        """(estado, path, sha256 of the file or None): estado in "ok", "no_encontrada", "sha_distinto"."""
        path = locate_list(entry, self.lists_dir)
        if path is None:
            return "no_encontrada", None, None
        item = self._load(path)
        return ("ok" if item["sha256"] == entry["lista_sha256"] else "sha_distinto"), path, item["sha256"]

    def get(self, entry):
        estado, path, _ = self.state(entry)
        if estado != "ok":
            return None
        self.cache[path] = self._load(path)
        return self.cache[path]

    def problem(self, entry):
        estado, path, sha = self.state(entry)
        if estado == "no_encontrada":
            return "lista no encontrada (%s); indique --lists-dir" % os.path.basename(entry["lista"])
        if estado == "sha_distinto":
            return "la lista %s tiene SHA-256 %s y el manifiesto dice %s" % (
                os.path.basename(path), sha, entry["lista_sha256"])
        return None


def list_checks(entries, lists, contracts):
    """K0, blocking: every list the manifest names is found and is byte for byte the one the manifest hashes
    (the runs were launched with --primaries-sha256 of that value). One check per distinct list file."""
    seen = collections.OrderedDict()
    for e in entries:
        seen.setdefault(re.split(r"[\\/]", e["lista"])[-1], []).append(e)
    for name, group in seen.items():
        shas = sorted({x["lista_sha256"] for x in group})
        estado, path, sha = lists.state(group[0])
        ok = estado == "ok" and len(shas) == 1
        contracts.add("K0", "listas", "lista localizada == manifiesto: %s" % name, ok, True,
                      {"estado": estado, "corridas": [x["run_id"] for x in group],
                       "sha256_del_archivo": sha, "sha256_del_manifiesto": shas,
                       "ruta_usada": path if ok else None})


def straight_line_check(cr, lst, run, ekin_min):
    """List line versus logged crossing, high-energy muons only."""
    tab = lst["table"]
    sc = lst["sidecar"] or {}
    r_start = sc.get("start_radius_m")
    if r_start is None:
        return None
    model = C.line_region_model(cr, tab["vals"][:, 0:3], tab["vals"][:, 3:6], float(r_start))
    line_len = np.asarray(model["len"])
    line_deep = np.asarray(model["deep"], dtype=bool)
    logged = run.mu["crossed"] == 1
    ekin = tab["vals"][:, 6]
    hi = ekin >= ekin_min
    both_pos = hi & line_deep & logged
    rel = np.abs(run.mu["lz_m"][both_pos] - line_len[both_pos]) / np.maximum(line_len[both_pos], 1e-12)
    table2 = {"recta_cruza_registro_cruza": int((hi & line_deep & logged).sum()),
              "recta_cruza_registro_no": int((hi & line_deep & ~logged).sum()),
              "recta_no_registro_cruza": int((hi & ~line_deep & logged).sum()),
              "recta_no_registro_no": int((hi & ~line_deep & ~logged).sum())}
    disagree = np.flatnonzero(hi & (line_deep != logged))
    all_line = line_deep
    st = run.mu["status"]
    by_status = {s: int(((st == s.encode()) & all_line).sum()) for s in C.STATUS_NAMES}
    return {"ekin_minima_gev": ekin_min, "muones_de_alta_energia": int(hi.sum()),
            "tabla_2x2": table2, "desacuerdos_ids": disagree[:200].tolist(),
            "n_desacuerdos": int(len(disagree)),
            "recta_cruza_todas_las_energias": int(all_line.sum()),
            "recta_cruza_por_estado_final": by_status,
            "diferencia_relativa_de_longitud_max": float(rel.max()) if rel.size else None,
            "diferencia_relativa_de_longitud_media": float(rel.mean()) if rel.size else None}


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def write_csv(path, header, rows, stamp=None):
    with open(path, "x", encoding="utf-8", newline="") as fh:
        if stamp:
            fh.write("# %s\n" % stamp)
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(v) for v in r])


def run_row(run, roi, ya):
    e = run.entry
    mu = run.mu
    st = mu["status"]
    n = e["N"]
    det = int((st == b"DET").sum())
    is06 = e["caso"] == CASE_AFFECTED
    eff = det / n
    z = efficiency_z(e["caso"], det, n)
    codes = C.count_warning_codes(run.log or "")
    cpu_s, cpu_src, rss_kb, rss_src = resources_of(run.meta or {}, run.recursos or {})
    row = [e["run_id"], e["caso"], e["replica"], e["variante"], e["rol"], e["s_src"], e["s_tr"], n,
           det, eff, z, int(ya.sum())]
    for s in C.STATUS_NAMES:
        row.append(int((st == s.encode()).sum()))
    if is06:
        row += [int((mu["crossed"] == 1).sum()), int((np.nan_to_num(mu["n_graze_z"]) > 0).sum()),
                int((np.nan_to_num(mu["lz_m"]) > 0).sum())]
    else:
        row += [None, None, None]
    row += [C.warnings_as_text(codes), (run.meta or {}).get("wall_seconds"), cpu_s, cpu_src, rss_kb, rss_src,
            (run.status or {}).get("nodo"), (run.status or {}).get("trabajo")]
    row += lateral_summary(mu)
    return row


def resources_of(meta, rec):
    """(cpu_s, source of cpu_s, rss_kb, source of rss_kb) of one run.

    Preferred source: the getrusage figures of the process itself that the binary writes into meta.json
    (rusage_user_cpu_seconds + rusage_system_cpu_seconds, and rusage_max_rss_kb in kilobytes; they cover the
    whole process from its start, geometry construction included). Fallback: recursos.json written by
    run_timed.py (user_s + sys_s and maxrss_kb of the child). Each figure names its source ("meta.rusage",
    "recursos.json" or "NA" when neither has it) so the table says which one it used."""
    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    user, sys_, peak = (meta.get(k) for k in ("rusage_user_cpu_seconds", "rusage_system_cpu_seconds",
                                              "rusage_max_rss_kb"))
    if num(user) and num(sys_):
        cpu, cpu_src = user + sys_, "meta.rusage"
    elif num(rec.get("user_s")) and num(rec.get("sys_s")):
        cpu, cpu_src = rec["user_s"] + rec["sys_s"], "recursos.json"
    else:
        cpu, cpu_src = None, "NA"
    if num(peak):
        rss, rss_src = peak, "meta.rusage"
    elif num(rec.get("maxrss_kb")):
        rss, rss_src = rec["maxrss_kb"], "recursos.json"
    else:
        rss, rss_src = None, "NA"
    return cpu, cpu_src, rss, rss_src


def lateral_summary(mu):
    """[max_lat_mm max, fraction > 1e-6 mm, max_lat_z_mm max, sum_lat_z_mm max] of one parsed log."""
    lat = mu["max_lat_mm"]
    have = ~np.isnan(lat)
    top = float(lat[have].max()) if have.any() else None
    frac = float((lat[have] > MAX_LAT_LIMIT_MM).mean()) if have.any() else None
    zmax = zsum = None
    if "max_lat_z_mm" in mu and "sum_lat_z_mm" in mu:
        if (~np.isnan(mu["max_lat_z_mm"])).any():
            zmax = float(np.nanmax(mu["max_lat_z_mm"]))
        if (~np.isnan(mu["sum_lat_z_mm"])).any():
            zsum = float(np.nanmax(mu["sum_lat_z_mm"]))
    return [top, frac, zmax, zsum]


RUN_TABLE_HEADER = (["corrida", "caso", "replica", "variante", "rol", "s_src", "s_tr", "N", "detectados",
                     "eficiencia", "z_eficiencia_vs_campana", "sumY_R_int"] + list(C.STATUS_NAMES) +
                    ["cruces_profundos", "muones_con_roce", "muones_con_LZ_positiva", "avisos_por_codigo",
                     "pared_s", "cpu_s", "fuente_cpu", "rss_kb", "fuente_rss", "nodo", "trabajo", "max_lat_mm",
                     "fraccion_max_lat_mayor_1e-6_mm",
                     "max_lat_z_mm", "max_suma_lat_z_mm"])

TRANSITIONS_HEADER = ["caso", "ambito", "rol_del_caso", "N", "n00", "n01", "n10", "n11", "b_discordantes",
                      "delta", "delta_por_millon", "denominador", "sumY_A", "sumY_B", "detectados_A",
                      "detectados_B", "m00", "m01", "m10", "m11", "delta_eficiencia", "cruces_A",
                      "cruces_B", "cruces_ambas", "cruces_union", "delta_condicional_U", "tamano_U",
                      "bins_distintos_entre_detectados_en_ambas", "muones_discordantes",
                      "tango_inf_por_millon", "tango_sup_por_millon", "mcnemar_p_dos_colas", "nota"]


def transition_row(caso, scope, s, cross, changed_bins, discordant):
    mc = s["mcnemar_exacto"]
    return [caso, scope, CASE_ROLE.get(caso, ""), s["N"], s["n00"], s["n01"], s["n10"], s["n11"], s["b"],
            s["delta"], s["delta_por_millon"], s["denominador"], s["sumY_A"], s["sumY_B"], s["det_A"],
            s["det_B"], s["m00"], s["m01"], s["m10"], s["m11"], s["delta_eficiencia"],
            cross.get("cruces_A"), cross.get("cruces_B"), cross.get("cruces_ambas"),
            cross.get("cruces_union"), cross.get("delta_condicional_U"), cross.get("tamano_U"),
            changed_bins, discordant, s["tango_inf_por_millon"], s["tango_sup_por_millon"],
            mc["p_dos_colas"] if mc["definido"] else None,
            "intervalo Tango y McNemar: diagnóstico; etapa 1 no dimensionada" +
            ("" if mc["definido"] else "; b = 0: la prueba exacta no informa")]


def pool_stats(pair_results):
    n = sum(p["N"] for p in pair_results)
    out = {"N": n}
    for k in ("n00", "n01", "n10", "n11", "sumY_A", "sumY_B", "det_A", "det_B", "m00", "m01", "m10", "m11"):
        out[k] = sum(p[k] for p in pair_results)
    out["b"] = out["n01"] + out["n10"]
    out["delta"] = (out["n01"] - out["n10"]) / n
    out["delta_por_millon"] = per_million(out["delta"])
    out["denominador"] = denominator_text(n)
    out["delta_eficiencia"] = (out["m01"] - out["m10"]) / n
    out.update(interval_block(out["n01"], out["n10"], n))
    deltas = [p["delta"] for p in pair_results]
    rep = {"deltas_por_replica": deltas, "media": float(np.mean(deltas)),
           "DE": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
           "rango": [float(min(deltas)), float(max(deltas))],
           "media_por_millon": per_million(float(np.mean(deltas))),
           "DE_por_millon": per_million(float(np.std(deltas, ddof=1))) if len(deltas) > 1 else None,
           "rango_por_millon": [per_million(min(deltas)), per_million(max(deltas))],
           "frase": SENTENCE_SD}
    out["por_replica"] = rep
    # Stage 1 is not sized (PROT section 4): its interval never bounds the effect, whatever the counts are, so the
    # flag is a constant. What can be stated is data: the exact Clopper-Pearson upper bounds of p_A and p_B,
    # each labelled with what is measured (SigmaY, N) and what is assumed (independent trials, one p).
    out["intervalo_no_acota_el_efecto"] = True
    out["nota_intervalo"] = NOTE_INTERVAL
    out["cotas_superiores_clopper_pearson"] = clopper_pearson_block(out["sumY_A"], out["sumY_B"], n)
    return out


def clopper_pearson_block(sum_a, sum_b, n, alpha=CP_ALPHA):
    """Exact two-sided (1 - alpha) Clopper-Pearson limits of p_A = P(Y_A = 1) and p_B = P(Y_B = 1) from the
    pooled counts, with their labels (data / assumption). No inequality about Delta is derived from them."""
    block = {"nivel_bilateral": 1.0 - alpha, "N": int(n), "etiqueta": LABEL_CP}
    for tag, s in (("p_A", sum_a), ("p_B", sum_b)):
        lo, hi = C.clopper_pearson_interval(s, n, alpha)
        block[tag] = {"suma_Y": int(s), "estimacion": s / float(n), "inf": lo, "sup": hi,
                      "sup_por_millon": per_million(hi)}
    return block


# --------------------------------------------------------------------------
# figures (optional)
# --------------------------------------------------------------------------
def make_figures(out_dir, maps, roi, stamp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    os.makedirs(os.path.join(out_dir, "figuras"), exist_ok=True)
    made = []
    for caso, m in maps.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        ext = [-C.AZ_HALF_DEG, C.AZ_HALF_DEG, C.EL_MIN_DEG, C.EL_MAX_DEG]
        for ax, key, title in ((axes[0], "A_sobre_N", "A / N"), (axes[1], "B_sobre_N", "B / N"),
                               (axes[2], "B_menos_A_sobre_N", "(B − A) / N")):
            arr = m[key]
            vmax = float(np.abs(arr).max()) or 1.0
            im = ax.imshow(arr, origin="lower", extent=ext, aspect="auto",
                           cmap="coolwarm" if key.startswith("B_menos") else "viridis",
                           vmin=-vmax if key.startswith("B_menos") else 0.0, vmax=vmax)
            ax.contour(np.linspace(ext[0], ext[1], C.N_AZ), np.linspace(ext[2], ext[3], C.N_EL),
                       roi.rint.astype(float), levels=[0.5], colors="black", linewidths=0.6)
            ax.set_title("%s — %s" % (caso, title), fontsize=9)
            ax.set_xlabel("acimut relativo (grados)")
            ax.set_ylabel("elevación (grados)")
            fig.colorbar(im, ax=ax)
        fig.suptitle("Mapas agrupados de %s; (B−A)/N: %s%s" % (caso, MAP_CAPTION,
                                                                   ("  [" + stamp + "]") if stamp else ""),
                     fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, "figuras", "mapa-%s.png" % caso)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        made.append(path)
    return made


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_replicas(text):
    """'1,2,3' -> (1, 2, 3). argparse reports a bad value as a usage error (exit code 2) before anything is read."""
    try:
        return tuple(int(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("%r no es una lista de enteros separados por comas (p. ej. 1,2,3)" % text)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt", required=True, help="carpeta intento-NN (contiene las corridas y estado/)")
    ap.add_argument("--manifest", help="manifiesto-corridas.tsv (por defecto <attempt>/manifiesto-corridas.tsv)")
    ap.add_argument("--roi-mask", required=True, help="R_int.mask.txt")
    ap.add_argument("--roi-frozen", required=True, help="R_int-congelada.json")
    ap.add_argument("--roi-npz", help="roi-mascaras.npz (clases descriptivas R_cond, R_body, R_rock, R_sky)")
    ap.add_argument("--lists-dir", help="carpeta con las listas (si no están en la ruta del manifiesto)")
    ap.add_argument("--out", required=True, help="carpeta de salida nueva")
    ap.add_argument("--crossing-reference", help="ruta de crossing_reference.py")
    ap.add_argument("--diagnostic-only", action="store_true",
                    help="con un contrato bloqueante fallido escribe igualmente las tablas, marcadas BLOQUEADO")
    ap.add_argument("--expected-runs", type=int, help="por defecto 4 x réplicas + 1")
    ap.add_argument("--replicas", default="1,2,3", type=parse_replicas,
                    help="réplicas de la etapa (las pruebas técnicas T-F usan '0')")
    ap.add_argument("--high-energy-gev", type=float, default=1000.0,
                    help="energía cinética inicial mínima para la verificación de la recta")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--skip-muon-fnv", action="store_true", help="no recalcula el FNV de _muones.csv (más rápido)")
    ap.add_argument("--test-mask", action="store_true",
                    help="SOLO PRUEBAS: acepta una máscara distinta de la congelada del plan")
    return ap


def main(argv=None):
    C.utf8_streams()
    args = build_parser().parse_args(argv)
    argv_used = list(sys.argv if argv is None else ["oe1_stage1_analysis.py"] + list(argv))
    # a file that cannot be opened is one line on stderr and exit code 2, not a traceback
    return C.run_command(run_analysis, args, argv_used)


def run_analysis(args, argv_used):
    attempt = os.path.abspath(args.attempt)
    manifest_path = args.manifest or os.path.join(attempt, "manifiesto-corridas.tsv")
    inputs = [attempt, manifest_path, args.roi_mask, args.roi_frozen] + ([args.roi_npz] if args.roi_npz else [])
    for p in inputs:
        if not os.path.exists(p):
            print("falta la entrada: " + p, file=sys.stderr)
            return 2
    if os.path.exists(args.out):
        print("la carpeta de salida ya existe (no se sobrescribe): " + args.out, file=sys.stderr)
        return 2
    reps = args.replicas
    expected_runs = args.expected_runs if args.expected_runs else 4 * len(reps) + 1
    # The manifest and the ROI inputs are read BEFORE the output folder is created (load_roi reads them again
    # later): an unusable input is exit code 2 and leaves no empty folder behind.
    try:
        entries = read_manifest(manifest_path)
    except ManifestError as exc:
        print("manifiesto inutilizable: %s" % exc, file=sys.stderr)
        return 2
    with open(args.roi_mask, "rb"):         # a folder or a file without read permission fails here (OSError, exit 2)
        pass
    for flag, reader, path in (("--roi-frozen", read_frozen_spec, args.roi_frozen),
                               ("--roi-npz", read_roi_npz, args.roi_npz)):
        if path:
            try:
                reader(path)
            except RoiInputError as exc:
                print("%s inutilizable: %s" % (flag, exc), file=sys.stderr)
                return 2
    os.makedirs(args.out)
    contracts = Contracts()
    manifest_checks(entries, expected_runs, contracts, reps)
    lists = ListCache(args.lists_dir)
    list_checks(entries, lists, contracts)
    roi = load_roi(args.roi_mask, args.roi_frozen, args.roi_npz, args.test_mask, contracts)
    if roi.rint is None:
        write_blocked(args.out, contracts, argv_used, manifest_path, None)
        return 3

    # -- runs
    runs = collections.OrderedDict()
    for e in entries:
        runs[e["run_id"]] = load_run(e, attempt, do_fnv=not args.skip_muon_fnv)
    for run in runs.values():
        k6_checks(run, roi, contracts, lists)
    by_key = {(r.entry["caso"], r.entry["replica"], r.entry["variante"]): r
              for r in runs.values() if r.entry["rol"] == ROLE_MAIN}
    aa = [r for r in runs.values() if r.entry["rol"] == ROLE_AA]
    if aa:
        base = by_key.get((CASE_CONTROL, reps[0], "A"))
        if base is not None:
            k7_checks(base, aa[0], contracts)
    for rep in reps:
        a, b = by_key.get((CASE_CONTROL, rep, "A")), by_key.get((CASE_CONTROL, rep, "B"))
        if a is not None and b is not None:
            k8_checks(a, b, contracts)
            k9a_checks(a, b, contracts, control=True)     # same primary and random start: blocking here too
        a, b = by_key.get((CASE_AFFECTED, rep, "A")), by_key.get((CASE_AFFECTED, rep, "B"))
        if a is not None and b is not None:
            k9_checks(a, b, contracts)

    blocked = bool(contracts.failed_blocking())
    usable = all(r.mu is not None for r in runs.values() if r.entry["rol"] == ROLE_MAIN)
    stamp = STAMP if blocked else None
    if blocked and not args.diagnostic_only or not usable:
        write_blocked(args.out, contracts, argv_used, manifest_path, entries)
        print("CONTRATOS: %s (%d bloqueantes fallidos)" % (STAMP, len(contracts.failed_blocking())))
        for c in contracts.failed_blocking()[:40]:
            print("  - %s [%s] %s" % (c["id"], c["ambito"], c["descripcion"]))
        return 3

    # -- pairs and results
    cr = cr_path = cr_sha = None
    try:
        cr, cr_path, cr_sha = C.load_crossing_reference(args.crossing_reference)
    except FileNotFoundError:
        pass
    cases = collections.OrderedDict()
    run_rows = []
    trans_rows, class_rows, bin_rows, disc_rows, union_rows = [], [], [], [], []
    maps = {}
    y_cache = {}
    pair_results_by_case = collections.defaultdict(list)
    unexplained_total = 0
    k9b_records = []
    for caso in (CASE_AFFECTED, CASE_CONTROL):
        pairs = []
        for rep in reps:
            a, b = by_key.get((caso, rep, "A")), by_key.get((caso, rep, "B"))
            if a is not None and b is not None and a.mu is not None and b.mu is not None:
                pairs.append(Pair(caso, rep, a, b))
        pooled_maps = {"A": np.zeros((C.N_EL, C.N_AZ)), "B": np.zeros((C.N_EL, C.N_AZ))}
        n_total = 0
        pair_json = []
        cls_pool = np.zeros((7, 7), dtype=np.int64)
        cls_ids_pool = collections.defaultdict(list)
        bin_pool = collections.Counter()
        cross_pool = collections.Counter()
        disc_pool = 0
        for pair in pairs:
            s, ya, yb = pair_stats(pair, roi)
            y_cache[pair.a.run_id], y_cache[pair.b.run_id] = ya, yb
            # class matrix
            ca, cb = class_indices(pair.a, roi), class_indices(pair.b, roi)
            mat = np.zeros((7, 7), dtype=np.int64)
            np.add.at(mat, (ca, cb), 1)
            cls_pool += mat
            ids = pair.a.mu["id"]
            for i in range(7):
                for j in range(7):
                    if mat[i, j] and i != j:
                        sel = np.flatnonzero((ca == i) & (cb == j))
                        cls_ids_pool[(i, j)].extend([(pair.replica, int(x)) for x in ids[sel][:50]])
            for i in range(7):
                for j in range(7):
                    if mat[i, j]:
                        sel = np.flatnonzero((ca == i) & (cb == j)) if i != j else []
                        class_rows.append([caso, "replica %d" % pair.replica, CLASSES[i], CLASSES[j],
                                           int(mat[i, j]),
                                           " ".join(str(int(x)) for x in ids[sel][:50]) +
                                           (" ...(+%d)" % (len(sel) - 50) if len(sel) > 50 else "") if i != j else "",
                                           "control angular" if caso == CASE_CONTROL else ""])
            # bin changes among detected in both
            both = pair.det_a & pair.det_b
            d_el = pair.b.mu["iel"][both] - pair.a.mu["iel"][both]
            d_az = pair.b.mu["iaz"][both] - pair.a.mu["iaz"][both]
            cnt = collections.Counter(zip(d_el.tolist(), d_az.tolist()))
            changed = int(sum(v for k, v in cnt.items() if k != (0, 0)))
            for (de, dz), v in sorted(cnt.items()):
                bin_rows.append([caso, "replica %d" % pair.replica, de, dz, v,
                                 "mismo bin" if (de, dz) == (0, 0) else "cambio de bin"])
            bin_pool.update(cnt)
            # crossings
            if caso == CASE_AFFECTED:
                cross, union = crossing_summary(pair, ya, yb)
                cross_pool.update({k: v for k, v in cross.items() if k in
                                   ("cruces_A", "cruces_B", "cruces_ambas", "cruces_union", "n01_U", "n10_U")})
                union_rows.extend(union_table(pair, ya, yb, union, lists))
            else:
                cross = {}
            # discordant muons
            anyd, fields = discordance_table(pair.a, pair.b, roi)
            idx = np.flatnonzero(anyd)
            disc_pool += len(idx)
            if caso == CASE_AFFECTED:
                cls = classify_discordance(pair.a, pair.b, idx) if len(idx) else np.array([], dtype=object)
            else:
                cls = np.array(["control_sin_region"] * len(idx), dtype=object)
            lst = lists.get(pair.a.entry)
            for pos, k in enumerate(idx):
                names = [nm for nm, v in fields.items() if v[k]]
                disc_rows.append(discordant_row(pair, int(k), ya, yb, names, cls[pos], lst))
            counts_cls = collections.Counter(cls.tolist())
            unexplained = sum(counts_cls.get(c, 0) for c in UNEXPLAINED_CLASSES) if caso == CASE_AFFECTED else \
                len(idx)
            unexplained_total += unexplained
            k9b_records.append({"caso": caso, "replica": pair.replica, "discordantes": int(len(idx)),
                                "por_clase": dict(counts_cls), "sin_explicar": int(unexplained)})
            if caso == CASE_AFFECTED:
                contracts.add("K9", "%s vs %s" % (pair.a.run_id, pair.b.run_id),
                              "(b) discordancias clasificadas (solo diagnóstico; DV9): sin clases inesperadas",
                              unexplained == 0, False,
                              {"discordantes": int(len(idx)), "por_clase": dict(counts_cls),
                               "inesperadas": int(unexplained)})
            trans_rows.append(transition_row(caso, "réplica %d" % pair.replica, s, cross, changed, int(len(idx))))
            pair_json.append(dict(s, cruces=cross, bins_cambiados=changed, discordantes=int(len(idx)),
                                  discordantes_por_clase=dict(counts_cls)))
            pair_results_by_case[caso].append(s)
            n_total += pair.n
            # maps
            for key, run_ in (("A", pair.a), ("B", pair.b)):
                pooled_maps[key] += run_.hist
        # run rows
        for rep in reps:
            for var in ("A", "B"):
                r = by_key.get((caso, rep, var))
                if r is not None and r.mu is not None:
                    ya_ = y_cache.get(r.run_id)
                    if ya_ is None:
                        ya_ = y_vector(r, roi)
                    run_rows.append(run_row(r, roi, ya_))
        if not pairs:
            continue
        pooled = pool_stats(pair_results_by_case[caso])
        cross_total = {}
        if caso == CASE_AFFECTED:
            cross_total = {"cruces_A": cross_pool["cruces_A"], "cruces_B": cross_pool["cruces_B"],
                           "cruces_ambas": cross_pool["cruces_ambas"], "cruces_union": cross_pool["cruces_union"],
                           "tamano_U": cross_pool["cruces_union"]}
            nu = cross_pool["cruces_union"]
            cross_total["delta_condicional_U"] = ((cross_pool["n01_U"] - cross_pool["n10_U"]) / nu) if nu else None
        changed_pool = int(sum(v for k, v in bin_pool.items() if k != (0, 0)))
        trans_rows.append(transition_row(caso, "agrupado", pooled, cross_total, changed_pool, disc_pool))
        for i in range(7):
            for j in range(7):
                if cls_pool[i, j]:
                    ids_txt = ""
                    if i != j:
                        lst_ids = cls_ids_pool[(i, j)]
                        ids_txt = " ".join("r%d:%d" % x for x in lst_ids[:50]) + (
                            " ...(+%d)" % (len(lst_ids) - 50) if len(lst_ids) > 50 else "")
                    class_rows.append([caso, "agrupado", CLASSES[i], CLASSES[j], int(cls_pool[i, j]), ids_txt,
                                       "control angular" if caso == CASE_CONTROL else ""])
        for (de, dz), v in sorted(bin_pool.items()):
            bin_rows.append([caso, "agrupado", de, dz, v, "mismo bin" if (de, dz) == (0, 0) else "cambio de bin"])
        maps[caso] = {"A_sobre_N": pooled_maps["A"] / n_total, "B_sobre_N": pooled_maps["B"] / n_total,
                      "B_menos_A_sobre_N": (pooled_maps["B"] - pooled_maps["A"]) / n_total}
        cases[caso] = {
            "rol": CASE_ROLE[caso],
            "etiqueta_mascara": ("máscara angular R_int (control), no una región de intersección"
                                 if caso == CASE_CONTROL else "R_int (región de intersección)"),
            "pares": pair_json, "agrupado": pooled,
            "matriz_clases_agrupada": {CLASSES[i]: {CLASSES[j]: int(cls_pool[i, j]) for j in range(7)}
                                       for i in range(7)},
            "cruces_agrupados": cross_total,
            "bins_cambiados_agrupados": changed_pool,
            "mapas": {"leyenda": MAP_CAPTION},
        }
    # A/A repetition: run row only (excluded from everything else)
    for r in aa:
        if r.mu is not None:
            run_rows.append(run_row(r, roi, y_vector(r, roi)))

    # straight-line cross-check (geo-train-06)
    straight = {}
    if cr is not None:
        for rep in reps:
            for var in ("A", "B"):
                r = by_key.get((CASE_AFFECTED, rep, var))
                if r is None or r.mu is None:
                    continue
                lst = lists.get(r.entry)
                if lst is None:
                    straight[r.run_id] = {"omitida": lists.problem(r.entry) or "lista no disponible"}
                    continue
                straight[r.run_id] = straight_line_check(cr, lst, r, args.high_energy_gev)
    else:
        straight = {"omitida": "no se encontró crossing_reference.py"}

    atrib_blocked = bool([c for c in contracts.failed_nonblocking() if c["contrato"] in ("K8", "K9")]) or \
        unexplained_total > 0
    ctr = contracts.to_json({"atribucion_al_solape_bloqueada": bool(atrib_blocked),
                             "discordancias_no_cruzan_K9b": k9b_records})
    stamp_line = stamp
    # -- write
    out = args.out
    C.write_text_new(os.path.join(out, "contratos.json"), C.dumps_json(ctr))
    write_csv(os.path.join(out, "tabla-corridas.csv"), RUN_TABLE_HEADER, run_rows, stamp_line)
    write_csv(os.path.join(out, "transiciones.csv"), TRANSITIONS_HEADER, trans_rows, stamp_line)
    write_csv(os.path.join(out, "matriz-clases.csv"),
              ["caso", "ambito", "clase_A", "clase_B", "n", "ids_fuera_de_la_diagonal", "etiqueta"], class_rows, stamp_line)
    write_csv(os.path.join(out, "cambios-bin.csv"),
              ["caso", "ambito", "delta_iEl", "delta_iAz", "n", "tipo"], bin_rows, stamp_line)
    write_csv(os.path.join(out, "muones-discordantes.csv"), DISCORDANT_HEADER, disc_rows, stamp_line)
    write_csv(os.path.join(out, "cruces-union.csv"), UNION_HEADER, union_rows, stamp_line)
    np.savez_compressed(os.path.join(out, "mapas-etapa1.npz"),
                        **{"%s__%s" % (caso, k): v for caso, m in maps.items() for k, v in m.items()})
    figs = [] if args.no_figures else make_figures(out, maps, roi, stamp_line)
    results = {
        "formato": "oe1-resultados-etapa1/1",
        "estado": stamp_line or "OK",
        "naturaleza": ("Etapa 1 diagnóstica: no dimensionada; ningún intervalo de esta etapa acota "
                       "el efecto ni decide equivalencia."),
        "leyendas": {"delta": "Δ = promedio(Y_B − Y_A) = (n01 − n10)/N",
                     "y": "Y = 1 si el muon se detecta dentro de R_int",
                     "millon": "10⁶·Δ en cuentas por millón de primarios generados",
                     "frase_DE": SENTENCE_SD, "mapas": MAP_CAPTION,
                     "repeticion_AA": "la repetición A/A queda fuera de Δ, transiciones, agrupado y DE",
                     "cuerda": NOTE_CHORD,
                     "recursos": ("cpu_s = rusage_user_cpu_seconds + rusage_system_cpu_seconds y rss_kb = "
                                  "rusage_max_rss_kb del meta.json (getrusage del proceso, desde su arranque; "
                                  "kB); si el meta no los tiene, user_s + sys_s y maxrss_kb de recursos.json "
                                  "(run_timed.py, del proceso hijo). fuente_cpu y fuente_rss dicen cuál se usó "
                                  "en cada fila"),
                     "cotas_clopper_pearson": LABEL_CP},
        # PLAN W3: getrusage figures written by the binary into the meta, exactly as found (units are the
        # binary's own); the CPU and RSS columns of tabla-corridas.csv use them and name their source
        "recursos_del_meta": {
            rid: {p: C.meta_path_value(r.meta, p) for p in C.timing_meta_paths(r.meta)
                  if p.startswith("rusage_")}
            for rid, r in runs.items() if r.meta is not None},
        "desplazamiento_lateral": {
            rid: dict(zip(("max_lat_mm", "fraccion_max_lat_mayor_1e-6_mm", "max_lat_z_mm", "max_suma_lat_z_mm"),
                          lateral_summary(r.mu)))
            for rid, r in runs.items() if r.mu is not None},
        "atribucion_al_solape_bloqueada": bool(atrib_blocked),
        "advertencias": ([] if args.roi_npz else
                         ["sin roi-mascaras.npz: la matriz de clases solo distingue R_int y 'sin clase'"]),
        "casos": cases,
        "verificacion_de_la_recta": straight,
        "entradas": {
            "manifiesto_sha256": C.sha256_file(manifest_path),
            "R_int_texto_sha256": roi.text_sha, "R_int_contenido_sha256": roi.content_sha,
            "npz_sha256": roi.npz_sha, "solapes_entre_mascaras_descriptivas": roi.overlaps,
            "corridas": {rid: {k: v for k, v in r.sha.items() if k in ("hist", "muones", "meta", "run.log")}
                         for rid, r in runs.items()},
            "listas": {os.path.basename(p): d["sha256"] for p, d in lists.cache.items()},
        },
        "herramienta": {"scripts_sha256": C.script_hashes(C.tool_paths()), "entorno": C.env_versions(),
                        "crossing_reference": {"ruta": cr_path, "sha256": cr_sha}},
        "figuras": [os.path.relpath(f, out) for f in figs],
    }
    C.write_text_new(os.path.join(out, "resultados-etapa1.json"), C.dumps_json(results))
    write_command(out, argv_used, manifest_path, results["herramienta"])
    if stamp_line:
        C.write_text_new(os.path.join(out, "BLOQUEADO.txt"), stamp_line + "\n")
    print("CONTRATOS: %s" % (stamp_line or "OK"))
    if atrib_blocked:
        print("ATENCIÓN: la atribución al solape queda bloqueada hasta explicar K8 / K9(b).")
    for caso, cdat in cases.items():
        g = cdat["agrupado"]
        print("%s: n01=%d n10=%d N=%d  Δ=%s por millón  Tango [%.3f, %.3f] por millón" % (
            caso, g["n01"], g["n10"], g["N"], fmt(g["delta_por_millon"]),
            g["tango_inf_por_millon"], g["tango_sup_por_millon"]))
    return 3 if blocked else 0


def write_blocked(out, contracts, argv_used, manifest_path, entries):
    C.write_text_new(os.path.join(out, "contratos.json"), C.dumps_json(contracts.to_json()))
    C.write_text_new(os.path.join(out, "BLOQUEADO.txt"), STAMP + "\n")
    write_command(out, argv_used, manifest_path,
                  {"scripts_sha256": C.script_hashes(C.tool_paths()), "entorno": C.env_versions()})


def write_command(out, argv_used, manifest_path, tool):
    lines = ["fecha_utc=%s" % now_utc(), "host=%s" % socket.gethostname(), "cwd=%s" % os.getcwd(),
             "python=%s" % tool.get("entorno", {}).get("python"), "numpy=%s" % tool.get("entorno", {}).get("numpy"),
             "manifiesto_sha256=%s" % (C.sha256_file(manifest_path) if manifest_path and os.path.exists(manifest_path) else "NA")]
    for name, sha in sorted(tool.get("scripts_sha256", {}).items()):
        lines.append("script_sha256[%s]=%s" % (name, sha))
    lines.append("comando=" + " ".join(_quote(a) for a in argv_used))
    C.write_text_new(os.path.join(out, "comando-analisis.txt"), "\n".join(lines) + "\n")


def _quote(s):
    return s if re.match(r"^[\w@%+=:,./-]+$", s) else "'" + s.replace("'", "'\\''") + "'"


DISCORDANT_HEADER = ["caso", "replica", "id", "ekin_inicial_gev", "campos_distintos", "clase_K9b", "Y_A", "Y_B",
                     "estado_A", "estado_B", "iel_A", "iaz_A", "iel_B", "iaz_B", "rng1_igual", "cruza_A",
                     "cruza_B", "lz_A_m", "lz_B_m", "lc_A_m", "le_A_m", "roce_A", "roce_B", "end_vol_A",
                     "end_vol_B"]

UNION_HEADER = ["caso", "replica", "id", "ekin_inicial_gev", "cruza_A", "cruza_B", "Y_A", "Y_B", "estado_A",
                "estado_B", "lz_A_m", "lz_B_m", "lz_verdadera_A_m", "lz_verdadera_B_m", "lz_a0_A_m",
                "lz_a1_A_m", "lz_a1_B_m", "mat_in_A", "mat_in_B", "mat_out_A", "mat_out_B", "e_in_A_gev",
                "e_in_B_gev", "e_out_A_gev", "e_out_B_gev", "roce_A", "roce_B", "pasos_bilaterales_fuera_A",
                "pasos_bilaterales_fuera_B", "escala_error_lz_A_mm", "escala_error_lz_B_mm", "max_lat_z_A_mm",
                "max_lat_z_B_mm"]


def cell_opt(mu, name, k):
    """``cell`` for a column the binary may not record (None when the column is absent)."""
    return cell(mu[name], k) if name in mu else None


def cell(arr, k, integer=False):
    """One value of a parsed column as a plain Python object (None when empty)."""
    v = arr[k]
    if isinstance(v, (bytes, np.bytes_)):
        return bytes(v).decode("ascii")
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return None
        return int(v) if integer else float(v)
    return v.item() if hasattr(v, "item") else v


def discordant_row(pair, k, ya, yb, names, cls, lst):
    a, b = pair.a.mu, pair.b.mu
    ekin = float(lst["table"]["vals"][k, 6]) if lst is not None else None
    return [pair.caso, pair.replica, int(a["id"][k]), ekin, ";".join(names), cls, int(ya[k]), int(yb[k]),
            cell(a["status"], k), cell(b["status"], k), int(a["iel"][k]), int(a["iaz"][k]), int(b["iel"][k]),
            int(b["iaz"][k]), int(a["rng1"][k] == b["rng1"][k]), cell(a["crossed"], k, True),
            cell(b["crossed"], k, True), cell(a["lz_m"], k), cell(b["lz_m"], k), cell(a["lc_m"], k),
            cell(a["le_m"], k), cell(a["n_graze_z"], k, True), cell(b["n_graze_z"], k, True),
            cell(a["end_vol"], k), cell(b["end_vol"], k)]


def union_table(pair, ya, yb, union, lists):
    rows = []
    a, b = pair.a.mu, pair.b.mu
    lst = lists.get(pair.a.entry)
    for k in np.flatnonzero(union):
        ekin = float(lst["table"]["vals"][k, 6]) if lst is not None else None
        rows.append([pair.caso, pair.replica, int(a["id"][k]), ekin, cell(a["crossed"], k, True),
                     cell(b["crossed"], k, True), int(ya[k]), int(yb[k]), cell(a["status"], k), cell(b["status"], k),
                     cell(a["lz_m"], k), cell(b["lz_m"], k), cell(a["lz_true_m"], k), cell(b["lz_true_m"], k),
                     cell(a["lz_a0_m"], k), cell(a["lz_a1_m"], k), cell(b["lz_a1_m"], k), cell(a["mat_in"], k),
                     cell(b["mat_in"], k), cell(a["mat_out"], k), cell(b["mat_out"], k), cell(a["e_in_gev"], k),
                     cell(b["e_in_gev"], k), cell(a["e_out_gev"], k), cell(b["e_out_gev"], k),
                     cell(a["n_graze_z"], k, True), cell(b["n_graze_z"], k, True),
                     cell(a["n_steps_z_both_out"], k, True), cell(b["n_steps_z_both_out"], k, True),
                     cell_opt(a, "sum_lat_z_mm", k), cell_opt(b, "sum_lat_z_mm", k),
                     cell_opt(a, "max_lat_z_mm", k), cell_opt(b, "max_lat_z_mm", k)])
    return rows


if __name__ == "__main__":
    sys.exit(main())
