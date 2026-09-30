#!/usr/bin/env python3
"""OE1 stage 1: frozen R_int mask (export, verify) and the K-ROI orientation proof.

Subcommands
  export  Reads the pilot's ``roi-mascaras.npz`` (read-only), checks its frozen
          fingerprints, and writes the text mask read by ``--roi-mask`` of
          ``muography_sim`` (256 LF lines of '0'/'1'; line = iEl, character =
          iAz), its SHA-256 and ``R_int-congelada.json`` (hashes, provenance,
          orientation). Nothing is ever overwritten.
  verify  Re-checks a text mask against ``R_int-congelada.json`` (and, if given,
          against the npz). Needs only numpy.
  kroi    Geometric proof of the mask orientation with the conventions of
          ``SteppingAction.cc`` (detector (1000, 0, 1.5), view 0 looks at
          azimuth 180 deg, ``relAz = wrap(atan2(y, x) - lookAz)``, linspace
          pixel centres, ``lround``), plus a chord-length map L_Z(pixel) of the
          3D region Z along the central rays, computed with the reused
          ``crossing_reference.py``. Exit code 3 if the mask contradicts the
          geometry (PROT:59 forbids redefining it: stop and report).

Python >= 3.9, numpy >= 1.23; matplotlib only for the optional PNG.
"""
import argparse
import io
import math
import os
import shutil
import sys

import numpy as np

import oe1_stage1_common as C

# The centroid of Z (audit of the director, ENTREGA_OE1.md:21).
Z_CENTROID_M = (21.36, -30.71, 137.81)
# Acceptance of K-ROI (fixed before looking at the numbers).
KROI_MIN_MASS_IN_RINT = 0.5      # share of the L_Z mass that falls inside R_int
KROI_MAX_MIRROR_RATIO = 0.1      # mirrored share must be <= this fraction of the R_int share
KROI_LINE_LENGTH_M = 3000.0      # central rays are traced this far from the detector
KROI_L_INT_THRESHOLD_M = 20.0    # the pilot defined R_int as l_int >= 20 m


FROZEN = {
    "npz_sha256": C.FROZEN_NPZ_SHA256,
    "content_sha256": C.FROZEN_RINT_CONTENT_SHA256,
    "text_sha256": C.FROZEN_RINT_TEXT_SHA256,      # the exported R_int.mask.txt the binary loads
    "shape": C.FROZEN_RINT_SHAPE,
    "pixels": C.FROZEN_RINT_PIXELS,
    "rows": C.FROZEN_RINT_ROWS,
    "cols": C.FROZEN_RINT_COLS,
}


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def _load_npz_bytes(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    return raw, arrays


def check_rint_against(mask, npz_sha, expect):
    """List of (name, ok, detail) comparing the mask with the frozen values."""
    checks = []
    if expect is None:
        return checks
    checks.append(("sha256 del archivo npz", npz_sha == expect["npz_sha256"], npz_sha))
    checks.append(("forma (256,256)", tuple(mask.shape) == tuple(expect["shape"]),
                   list(mask.shape)))
    checks.append(("dtype bool", mask.dtype == np.bool_, str(mask.dtype)))
    content = C.mask_content_sha256(mask)
    checks.append(("sha256 de R_int.tobytes()", content == expect["content_sha256"], content))
    s = C.mask_summary(mask)
    checks.append(("pixeles", s["pixeles"] == expect["pixels"], s["pixeles"]))
    checks.append(("filas", tuple(s["filas"] or ()) == tuple(expect["rows"]), s["filas"]))
    checks.append(("columnas", tuple(s["columnas"] or ()) == tuple(expect["cols"]),
                   s["columnas"]))
    return checks


def build_frozen_spec(mask, npz_sha, text_bytes, text_name, tool_shas):
    text_sha = C.sha256_bytes(text_bytes)
    spec = {
        "formato": "oe1-roi-congelada/1",
        "mascara": "R_int",
        "descripcion": ("R_int del piloto (712c786), congelada para la etapa 1. "
                        "PROT:59 prohibe redefinirla; la orientacion se demuestra "
                        "geometricamente con oe1_stage1_roi.py kroi."),
        "origen": {
            "repositorio": "muografia-documentada",
            "commit": C.FROZEN_RINT_PROVENANCE_COMMIT,
            "ruta": C.FROZEN_RINT_PROVENANCE_PATH,
            "clave_npz": "R_int",
            "sha256_archivo": npz_sha,
            "construida_por": "oe1_pilot_roi.py preregister (repositorio de codigo, commit fa21f57)",
        },
        "hash_contenido": {
            "algoritmo": "sha256",
            "sobre": "R_int.tobytes() (orden C, bool, 1 byte por pixel)",
            "valor": C.mask_content_sha256(mask),
        },
        "orientacion": {
            "indices": "[iEl, iAz]",
            "filas": "iEl = linea de hist.csv (elevacion 2 a 47 grados, centros de linspace)",
            "columnas": ("iAz = columna de hist.csv (acimut relativo -30 a +30 grados, "
                         "centros de linspace)"),
            "transformaciones": "ninguna (sin transposicion ni volteo)",
        },
        "texto": {
            "archivo": text_name,
            "formato": ("256 lineas terminadas en LF de 256 caracteres '0'/'1'; "
                        "linea = iEl, caracter = iAz (RoiMask.hh)"),
            "bytes": len(text_bytes),
            "sha256": text_sha,
        },
        "herramienta": {"scripts_sha256": tool_shas, "entorno": C.env_versions()},
    }
    spec.update(C.mask_summary(mask))
    return spec


def export_mask(npz_path, out_dir, expect=FROZEN, copy_npz=False):
    """Writes text mask, its sha256 file and the frozen spec. Returns a summary."""
    raw, arrays = _load_npz_bytes(npz_path)
    if "R_int" not in arrays:
        raise SystemExit("el npz no contiene la clave R_int")
    mask = arrays["R_int"]
    npz_sha = C.sha256_bytes(raw)
    checks = check_rint_against(mask, npz_sha, expect)
    failed = [c for c in checks if not c[1]]
    if failed:
        raise SystemExit("R_int no coincide con los valores congelados: " +
                         "; ".join("%s (obtenido %r)" % (c[0], c[2]) for c in failed))
    text = C.mask_to_text(mask)
    # the text file is what the binary loads (--roi-mask-sha256): its SHA-256 is hard-coded too, so an export
    # that drifted from the plan cannot pass on the strength of its own spec
    if expect is not None and expect.get("text_sha256") not in (None, C.sha256_bytes(text)):
        raise SystemExit("el texto exportado de R_int no tiene el sha256 congelado del plan (%s): obtenido %s" %
                         (expect["text_sha256"], C.sha256_bytes(text)))
    os.makedirs(out_dir, exist_ok=True)
    name = "R_int.mask.txt"
    text_path = os.path.join(out_dir, name)
    with open(text_path, "xb") as fh:
        fh.write(text)
    text_sha = C.sha256_bytes(text)
    with open(os.path.join(out_dir, name + ".sha256"), "xb") as fh:
        fh.write(("%s  %s\n" % (text_sha, name)).encode("ascii"))
    spec = build_frozen_spec(mask, npz_sha, text, name,
                             C.script_hashes([os.path.abspath(__file__),
                                              os.path.join(C.here(), "oe1_stage1_common.py")]))
    C.write_text_new(os.path.join(out_dir, "R_int-congelada.json"), C.dumps_json(spec))
    if copy_npz:
        dest = os.path.join(out_dir, os.path.basename(npz_path))
        with open(dest, "xb") as fh:
            fh.write(raw)
        if C.sha256_file(dest) != npz_sha:
            raise SystemExit("la copia del npz no coincide byte a byte")
    return {"text_path": text_path, "text_sha256": text_sha, "npz_sha256": npz_sha,
            "content_sha256": spec["hash_contenido"]["valor"], "summary": C.mask_summary(mask),
            "checks": checks}


def cmd_export(args):
    res = export_mask(args.npz, args.out_dir, FROZEN, copy_npz=args.copy_npz)
    print("R_int exportada a %s" % res["text_path"])
    print("  sha256 del texto        : %s" % res["text_sha256"])
    print("  sha256 del npz          : %s" % res["npz_sha256"])
    print("  sha256 de R_int.tobytes(): %s" % res["content_sha256"])
    s = res["summary"]
    print("  forma %s, %d pixeles, filas %s, columnas %s" %
          (tuple(s["forma"]), s["pixeles"], tuple(s["filas"]), tuple(s["columnas"])))
    for name, ok, detail in res["checks"]:
        print("  [%s] %s" % ("OK" if ok else "FALLA", name))
    return 0


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def verify_mask(mask_path, frozen_json, npz_path=None, expect=FROZEN):
    """Returns a list of (name, ok, detail)."""
    out = []
    spec = C.read_json(frozen_json)
    with open(mask_path, "rb") as fh:
        data = fh.read()
    out.append(("sha256 del texto == congelada.json", C.sha256_bytes(data) == spec["texto"]["sha256"],
                C.sha256_bytes(data)))
    try:
        mask = C.mask_from_text(data)
    except ValueError as exc:
        out.append(("formato del texto", False, str(exc)))
        return out
    out.append(("formato del texto", True, "256 x 256"))
    out.append(("sha256 de R_int.tobytes() == congelada.json",
                C.mask_content_sha256(mask) == spec["hash_contenido"]["valor"],
                C.mask_content_sha256(mask)))
    s = C.mask_summary(mask)
    for key in ("forma", "pixeles", "filas", "columnas"):
        out.append((key + " == congelada.json", s[key] == spec[key], s[key]))
    if expect is not None:
        if expect.get("text_sha256") is not None:
            out.append(("sha256 del texto == valor congelado del plan",
                        C.sha256_bytes(data) == expect["text_sha256"], C.sha256_bytes(data)))
        out.append(("sha256 de R_int.tobytes() == valor congelado del plan",
                    C.mask_content_sha256(mask) == expect["content_sha256"], None))
        out.append(("pixeles/filas/columnas == plan",
                    (s["pixeles"], tuple(s["filas"]), tuple(s["columnas"])) ==
                    (expect["pixels"], tuple(expect["rows"]), tuple(expect["cols"])), s))
    if npz_path:
        raw, arrays = _load_npz_bytes(npz_path)
        out.append(("sha256 del npz == congelada.json",
                    C.sha256_bytes(raw) == spec["origen"]["sha256_archivo"], C.sha256_bytes(raw)))
        out.append(("R_int del npz == texto", bool(np.array_equal(arrays.get("R_int"), mask)), None))
    return out


def cmd_verify(args):
    res = verify_mask(args.mask, args.frozen, args.npz, None if args.no_plan_check else FROZEN)
    ok_all = True
    for name, ok, detail in res:
        ok_all = ok_all and ok
        print("[%s] %s" % ("OK" if ok else "FALLA", name))
    print("VERIFICACION R_int:", "OK" if ok_all else "FALLA")
    return 0 if ok_all else 3


# --------------------------------------------------------------------------
# kroi
# --------------------------------------------------------------------------
def rays_from_detector(view=0, length_m=KROI_LINE_LENGTH_M):
    """Segments (65536, 6) along the central ray of every pixel, plus the pixel grid."""
    el_deg, az_deg = C.pixel_centres_deg()
    look = C.look_azimuth_rad(view)
    el = np.radians(el_deg)[:, None] * np.ones((1, C.N_AZ))
    az = look + np.radians(az_deg)[None, :] * np.ones((C.N_EL, 1))
    fx = np.cos(el) * np.cos(az)
    fy = np.cos(el) * np.sin(az)
    fz = np.sin(el)
    d = np.array(C.detector_center(view))
    a = np.tile(d, (C.N_EL * C.N_AZ, 1))
    b = a + length_m * np.stack([fx.ravel(), fy.ravel(), fz.ravel()], axis=1)
    return np.hstack([a, b]), (el_deg, az_deg)


def z_membership(cr, pts):
    """Boolean membership in Z (convention of auditar_oe1.margin: inside <=> margin < 0)."""
    cyl, ell = cr.CYL, cr.ELL
    c = np.array([cyl["cx"], cyl["cy"], 0.5 * (cyl["z0"] + cyl["z1"])])
    s = np.array([cyl["r"], cyl["r"], 0.5 * (cyl["z1"] - cyl["z0"])])
    v = (pts - c) / s
    mc = np.maximum(v[:, 0] ** 2 + v[:, 1] ** 2 - 1, np.abs(v[:, 2]) - 1)
    ce = np.array([ell["cx"], ell["cy"], ell["cz"]])
    se = np.array([ell["ax"], ell["ay"], ell["az"]])
    ve = (pts - ce) / se
    me = (ve * ve).sum(axis=1) - 1
    return np.maximum(mc, me) < 0


def z_bounding_box(cr):
    cyl, ell = cr.CYL, cr.ELL
    lo = np.array([max(cyl["cx"] - cyl["r"], ell["cx"] - ell["ax"]),
                   max(cyl["cy"] - cyl["r"], ell["cy"] - ell["ay"]),
                   max(cyl["z0"], ell["cz"] - ell["az"])])
    hi = np.array([min(cyl["cx"] + cyl["r"], ell["cx"] + ell["ax"]),
                   min(cyl["cy"] + cyl["r"], ell["cy"] + ell["ay"]),
                   min(cyl["z1"], ell["cz"] + ell["az"])])
    return lo, hi


def pixels_of_points(points, view=0):
    """Pixel (iEl, iAz) of the line of sight from the detector to each point,
    with the SteppingAction convention (scalar loop over ``pixel_of_direction``)."""
    d = np.array(C.detector_center(view))
    out = np.empty((len(points), 2), dtype=np.int64)
    for k, p in enumerate(points):
        mom = (d - p) / np.linalg.norm(d - p)    # muon momentum direction at the panel
        i_el, i_az, _, _ = C.pixel_of_direction(-mom, view)
        out[k] = (i_el, i_az)
    return out


def kroi_compute(cr, rint, l_int=None, view=0, n_points=20000, seed=20260930):
    """All numbers of the K-ROI proof. ``rint`` is the bool R_int array."""
    rint = np.asarray(rint, dtype=bool)
    mirror = rint[:, ::-1]           # azimuth mirror: iAz -> 255 - iAz (relAz -> -relAz)
    flip_el = rint[::-1, :]          # negative control: elevation reversed
    d = np.array(C.detector_center(view))
    cen = np.array(Z_CENTROID_M)
    mom = (d - cen) / np.linalg.norm(d - cen)
    i_el, i_az, el_deg, rel_deg = C.pixel_of_direction(-mom, view)
    res = {
        "convenciones": {
            "detector_m": list(d),
            "acimut_de_mirada_grados": math.degrees(C.look_azimuth_rad(view)),
            "relAz": "wrap(atan2(from.y, from.x) - lookAz), from = -direccion del muon en el panel",
            "pixel": "binIndex con centros de linspace y lround (SteppingAction.cc)",
            "espejo": "iAz -> 255 - iAz (relAz -> -relAz)",
        },
        "centroide": {
            "punto_m": list(Z_CENTROID_M),
            "elevacion_grados": el_deg, "acimut_relativo_grados": rel_deg,
            "pixel": [i_el, i_az],
            "en_R_int": bool(0 <= i_el < C.N_EL and 0 <= i_az < C.N_AZ and rint[i_el, i_az]),
            "pixel_espejo": [i_el, C.N_AZ - 1 - i_az] if i_az >= 0 else None,
            "espejo_en_R_int": bool(i_az >= 0 and rint[i_el, C.N_AZ - 1 - i_az]),
        },
    }
    segs, _ = rays_from_detector(view)
    model = cr.model(segs)
    lz = np.asarray(model["len"], dtype=np.float64).reshape(C.N_EL, C.N_AZ)
    total = float(lz.sum())
    pos = lz > 0
    lz20 = lz >= KROI_L_INT_THRESHOLD_M

    def share(m):
        return float(lz[m].sum() / total) if total > 0 else 0.0

    def jaccard(a, b):
        u = int((a | b).sum())
        return float((a & b).sum() / u) if u else 0.0

    res["mapa_LZ"] = {
        "descripcion": "longitud (m) de la recta central de cada pixel dentro de Z",
        "pixeles_con_LZ_positiva": int(pos.sum()),
        "pixeles_con_LZ_ge_20_m": int(lz20.sum()),
        "LZ_maxima_m": float(lz.max()),
        "LZ_total_m": total,
        "pixel_de_LZ_maxima": [int(v) for v in np.unravel_index(int(lz.argmax()), lz.shape)],
        "centro_de_masa_pixel": ([float((lz * np.arange(C.N_EL)[:, None]).sum() / total),
                                  float((lz * np.arange(C.N_AZ)[None, :]).sum() / total)]
                                 if total > 0 else None),
    }
    res["solape"] = {
        "masa_LZ_en_R_int": share(rint),
        "masa_LZ_en_espejo_azimut": share(mirror),
        "masa_LZ_en_elevacion_invertida": share(flip_el),
        "pixeles_LZ_positiva_en_R_int": int((pos & rint).sum()),
        "pixeles_LZ_positiva_en_espejo": int((pos & mirror).sum()),
        "jaccard_LZ_ge_20_con_R_int": jaccard(lz20, rint),
        "jaccard_LZ_ge_20_con_espejo": jaccard(lz20, mirror),
        "jaccard_LZ_ge_20_con_elevacion_invertida": jaccard(lz20, flip_el),
        "R_int_pixeles": int(rint.sum()),
    }
    if l_int is not None:
        li = np.asarray(l_int, dtype=np.float64)
        both = pos | (li > 0)
        res["comparacion_con_l_int_del_piloto"] = {
            "descripcion": ("l_int del npz (marcha de rayos con paso de 2 m, float32) "
                            "frente a L_Z exacta"),
            "diferencia_maxima_m": float(np.abs(li - lz).max()),
            "diferencia_media_abs_m": float(np.abs(li - lz).mean()),
            "correlacion": float(np.corrcoef(li.ravel(), lz.ravel())[0, 1]) if li.std() > 0 and lz.std() > 0 else None,
            "pixeles_l_int_positiva": int((li > 0).sum()),
            "pixeles_solo_uno_positivo": int(((li > 0) != pos).sum()),
            "pixeles_union": int(both.sum()),
        }
    # Independent view: pixels of points sampled inside Z (membership by the
    # normalised margins of auditar_oe1, no ray model involved).
    rng = np.random.default_rng(seed)
    lo, hi = z_bounding_box(cr)
    pts = lo + rng.random((int(n_points) * 6, 3)) * (hi - lo)
    pts = pts[z_membership(cr, pts)][: int(n_points)]
    if len(pts):
        px = pixels_of_points(pts, view)
        valid = (px[:, 0] >= 0) & (px[:, 1] >= 0)
        vi = px[valid]
        res["puntos_de_Z"] = {
            "muestras": int(len(pts)), "semilla": seed,
            "fraccion_en_R_int": float(rint[vi[:, 0], vi[:, 1]].mean()) if len(vi) else None,
            "fraccion_en_espejo": float(mirror[vi[:, 0], vi[:, 1]].mean()) if len(vi) else None,
        }
    thr = {"masa_minima_en_R_int": KROI_MIN_MASS_IN_RINT,
           "razon_maxima_espejo_sobre_R_int": KROI_MAX_MIRROR_RATIO}
    checks = [
        ("el pixel del centroide esta en R_int", res["centroide"]["en_R_int"]),
        ("el pixel espejo del centroide NO esta en R_int", not res["centroide"]["espejo_en_R_int"]),
        ("masa de L_Z en R_int >= %.2f" % KROI_MIN_MASS_IN_RINT,
         res["solape"]["masa_LZ_en_R_int"] >= KROI_MIN_MASS_IN_RINT),
        ("masa de L_Z en el espejo <= %.2f x masa en R_int" % KROI_MAX_MIRROR_RATIO,
         res["solape"]["masa_LZ_en_espejo_azimut"] <=
         KROI_MAX_MIRROR_RATIO * res["solape"]["masa_LZ_en_R_int"]),
    ]
    res["criterios"] = thr
    res["comprobaciones"] = [{"descripcion": n, "ok": bool(o)} for n, o in checks]
    res["pixel_esperado_34_135"] = bool((i_el, i_az) == (34, 135))
    res["veredicto"] = "OK" if all(o for _, o in checks) else "CONTRADICE LA MASCARA"
    return res, lz


def _plot_kroi(path, lz, rint, l_int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 3 if l_int is not None else 2, figsize=(5 * (3 if l_int is not None else 2), 4.6))
    ext = [-C.AZ_HALF_DEG, C.AZ_HALF_DEG, C.EL_MIN_DEG, C.EL_MAX_DEG]
    mirror = rint[:, ::-1]
    for ax, cont, title in ((axes[0], rint, "R_int (contorno) sobre L_Z"),
                            (axes[1], mirror, "espejo en acimut (contorno) sobre L_Z")):
        im = ax.imshow(lz, origin="lower", extent=ext, aspect="auto", cmap="viridis")
        ax.contour(np.linspace(ext[0], ext[1], C.N_AZ), np.linspace(ext[2], ext[3], C.N_EL),
                   cont.astype(float), levels=[0.5], colors="red", linewidths=1.0)
        ax.set_xlim(-2, 8)
        ax.set_ylim(3, 13)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("acimut relativo (grados)")
        ax.set_ylabel("elevacion (grados)")
        fig.colorbar(im, ax=ax, label="L_Z (m)")
    if l_int is not None:
        ax = axes[2]
        im = ax.imshow(l_int - lz, origin="lower", extent=ext, aspect="auto", cmap="coolwarm")
        ax.set_xlim(-2, 8)
        ax.set_ylim(3, 13)
        ax.set_title("l_int del piloto - L_Z exacta", fontsize=9)
        ax.set_xlabel("acimut relativo (grados)")
        fig.colorbar(im, ax=ax, label="m")
    fig.suptitle("K-ROI: orientacion de R_int frente a la longitud de Z por pixel", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def cmd_kroi(args):
    cr, cr_path, cr_sha = C.load_crossing_reference(args.crossing_reference)
    raw, arrays = _load_npz_bytes(args.npz)
    rint = arrays["R_int"]
    npz_sha = C.sha256_bytes(raw)
    res, lz = kroi_compute(cr, rint, arrays.get("l_int"), view=args.view)
    res["entradas"] = {"npz_sha256": npz_sha,
                       "R_int_contenido_sha256": C.mask_content_sha256(rint),
                       "crossing_reference": {"ruta": cr_path, "sha256": cr_sha}}
    res["herramienta"] = {"scripts_sha256": C.script_hashes(C.tool_paths()),
                          "entorno": C.env_versions()}
    os.makedirs(args.out_dir, exist_ok=True)
    C.write_text_new(os.path.join(args.out_dir, "kroi.json"), C.dumps_json(res))
    png = os.path.join(args.out_dir, "kroi.png")
    made = False if args.no_figure else _plot_kroi(png, lz, rint, arrays.get("l_int"))
    c = res["centroide"]
    s = res["solape"]
    print("K-ROI centroide de Z: elevacion %.4f, acimut relativo %+.4f -> pixel %s (en R_int: %s)" %
          (c["elevacion_grados"], c["acimut_relativo_grados"], tuple(c["pixel"]), c["en_R_int"]))
    print("  espejo en acimut: pixel %s (en R_int: %s)" % (tuple(c["pixel_espejo"]), c["espejo_en_R_int"]))
    print("  masa de L_Z en R_int %.4f | espejo %.4f | elevacion invertida %.4f" %
          (s["masa_LZ_en_R_int"], s["masa_LZ_en_espejo_azimut"], s["masa_LZ_en_elevacion_invertida"]))
    print("  Jaccard(L_Z>=20 m, R_int) %.4f | espejo %.4f" %
          (s["jaccard_LZ_ge_20_con_R_int"], s["jaccard_LZ_ge_20_con_espejo"]))
    if "comparacion_con_l_int_del_piloto" in res:
        k = res["comparacion_con_l_int_del_piloto"]
        print("  l_int del piloto: |dif| max %.3f m, media %.4f m, correlacion %s" %
              (k["diferencia_maxima_m"], k["diferencia_media_abs_m"], k["correlacion"]))
    print("  figura:", png if made else "no generada")
    print("K-ROI veredicto:", res["veredicto"])
    return 0 if res["veredicto"] == "OK" else 3


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="exporta R_int a texto con hash y R_int-congelada.json")
    e.add_argument("--npz", required=True, help="roi-mascaras.npz (solo lectura)")
    e.add_argument("--out-dir", required=True, help="carpeta de salida (los archivos no deben existir)")
    e.add_argument("--copy-npz", action="store_true", help="copia el npz byte a byte a la salida")
    e.set_defaults(fn=cmd_export)
    v = sub.add_parser("verify", help="verifica el texto de la mascara contra el JSON congelado")
    v.add_argument("--mask", required=True)
    v.add_argument("--frozen", required=True)
    v.add_argument("--npz")
    v.add_argument("--no-plan-check", action="store_true",
                   help="(solo pruebas) no compara con los valores congelados del plan")
    v.set_defaults(fn=cmd_verify)
    k = sub.add_parser("kroi", help="prueba geometrica de la orientacion de R_int")
    k.add_argument("--npz", required=True)
    k.add_argument("--out-dir", required=True)
    k.add_argument("--view", type=int, default=0)
    k.add_argument("--crossing-reference")
    k.add_argument("--no-figure", action="store_true")
    k.set_defaults(fn=cmd_kroi)
    return ap


def main(argv=None):
    C.utf8_streams()
    args = build_parser().parse_args(argv)
    return C.run_command(args.fn, args)


if __name__ == "__main__":
    sys.exit(main())
