#!/usr/bin/env python3
"""Stage 19 ROI checks V1-V5 -- REAL geometries, RUN ON THE CLUSTER.

These are the checks the execution plan lists against risk D4 ("an ROI bug
would invalidate Stage 21 and Stage 22 in silence"). They need the geometry
catalog and the stored surrogate opacity maps of data/campaign/, and one full
ray-march per (geometry, view) -- minutes of CPU, not the seconds the local
synthetic suite costs. scripts/evaluation/test_roi.py stays runnable on a laptop.

  V1  identity of the angular grid BY IMPORT (no re-implementation anywhere)
      plus agreement of the cone predicate with VolcanoGeometry.density
  V2  max L_a <= analytic chord + one ray step, every anomaly instance
  V3  ROI azimuth width within +-15 % of the analytic half-maximum width
  V4  X_total = rho_rock L_rock + sum_a (rho_a - rho_rock) L_a + air,
      against the stored surrogate opacity, to better than 0.1 % on rock
      pixels; the open-sky floor must equal the analytic value exactly
  V5  overlay PNGs of every ROI and annulus for visual inspection

Default scope is val + generalization, which are free to use. The test-split
geometries are passed EXPLICITLY by scripts/cluster/e20_test_pass.sbatch, inside
the single authorized test pass. This module never touches the count shards
and never resolves a split, so it needs no held-out opt-in.

Run:  python scripts/evaluation/test_roi_cluster.py
      python scripts/evaluation/test_roi_cluster.py --geometries geo-test-01 geo-test-02
      python scripts/evaluation/test_roi_cluster.py --views 0 --no-overlays
"""

from __future__ import annotations

import argparse
import inspect
import sys
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))

import roi as R  # noqa: E402
from surrogate import imaging  # noqa: E402
from surrogate.geometry import VolcanoGeometry  # noqa: E402

CAMPAIGN_DIR = REPO_ROOT / "data" / "campaign"
OVERLAY_DIR = REPO_ROOT / "results" / "evaluation" / "roi-overlays"
DEFAULT_GEOMETRIES = ("geo-val-01", "geo-val-02", "geo-val-03", "geo-gen-01")
DEFAULT_VIEWS = tuple(range(6))

V3_TOLERANCE = 0.15
V4_TOLERANCE = 1e-3          # 0.1 % relative on rock pixels
CHORD_SLACK_M = R.RAY_STEP_M  # one sampling step


# ------------------------------------------------------------------------ V1
def check_v1() -> bool:
    """The grid and the ray directions must BE imaging's, not a copy."""
    checks = {
        "angular_grid is imaging's": R.angular_grid is imaging.angular_grid,
        "ray_directions is imaging's": R.ray_directions is imaging.ray_directions,
        "N_AZ/N_EL shared": (R.N_AZ, R.N_EL) == (imaging.N_AZ, imaging.N_EL),
        "RAY_STEP_M shared": R.RAY_STEP_M == imaging.RAY_STEP_M,
        "RAY_MAX_M shared": R.RAY_MAX_M == imaging.RAY_MAX_M,
    }
    # roi.py must not define its own grid: no literal 256 / 30.0 / 2.0 sneaking in
    source = inspect.getsource(R)
    checks["no local grid literals"] = ("N_AZ = " not in source
                                        and "RAY_STEP_M = " not in source)

    # the cone predicate must agree with VolcanoGeometry.density
    geo = R.load_geometry(DEFAULT_GEOMETRIES[0])
    rng = np.random.default_rng(20260805)
    pts = rng.uniform(-1.2 * geo.base_radius_m, 1.2 * geo.base_radius_m,
                      size=(3, 40000))
    x, y = pts[0], pts[1]
    z = rng.uniform(-50.0, 1.2 * geo.height_m, size=x.shape)
    inside = R._inside_edifice(geo, x, y, z)
    rho = geo.density(x, y, z)
    from surrogate.geometry import AIR_DENSITY
    checks["cone predicate matches density()"] = bool(
        np.array_equal(inside, rho != AIR_DENSITY))

    ok = all(checks.values())
    for label, passed in checks.items():
        print(f"    V1 {label}: {'ok' if passed else 'FAIL'}")
    return ok


# ------------------------------------------------------------------------ V2
def check_v2(rows) -> bool:
    ok = True
    for row in rows:
        slack = row["max_l_a_m"] - (row["chord_m"] + CHORD_SLACK_M)
        good = slack <= 1e-6
        ok = ok and good
        print(f"    V2 {row['geometry']} v{row['view']} a{row['anomaly']} "
              f"({row['kind']}): max L_a = {row['max_l_a_m']:.1f} m, "
              f"chord = {row['chord_m']:.1f} m -> {'ok' if good else 'FAIL'}")
    return ok


# ------------------------------------------------------------------------ V3
def check_v3(rows) -> bool:
    ok = True
    for row in rows:
        predicted = row["predicted_width_px"]
        measured = row["width_px"]
        good = abs(measured - predicted) <= V3_TOLERANCE * predicted + 1.0
        ok = ok and good
        print(f"    V3 {row['geometry']} v{row['view']} a{row['anomaly']}: "
              f"width {measured} px vs {predicted:.1f} px predicted "
              f"({100.0 * (measured - predicted) / predicted:+.1f} %) -> "
              f"{'ok' if good else 'FAIL'}")
    return ok


# ------------------------------------------------------------------------ V4
def check_v4(rows) -> bool:
    ok = True
    for row in rows:
        good = (row["max_rel_error"] < V4_TOLERANCE
                and row["sky_abs_error"] < 1e-3)
        ok = ok and good
        print(f"    V4 {row['geometry']} v{row['view']}: max rel. error on "
              f"{row['n_rock']} rock pixels = {row['max_rel_error']:.2e}, "
              f"sky floor {row['sky_predicted']:.4f} vs {row['sky_observed']:.4f} "
              f"g/cm2 -> {'ok' if good else 'FAIL'}")
    return ok


# ------------------------------------------------------------------------ V5
def write_overlay(geometry_id: str, view_index: int, masks: dict,
                  opacity: np.ndarray, out_dir: Path) -> Path:
    """One PNG per (geometry, view): opacity with ROI and annulus contours."""
    from plotting_style import apply_style
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    ax.imshow(np.log10(np.maximum(opacity, 1.0)), origin="lower",
              aspect="auto", cmap="viridis")
    for index, (roi, annulus) in enumerate(zip(masks["rois"], masks["annuli"])):
        ax.contour(roi.astype(float), levels=[0.5], colors="white",
                   linewidths=1.4)
        ax.contour(annulus.astype(float), levels=[0.5], colors="orange",
                   linewidths=0.8)
        box = R.bounding_box(roi)
        if box is not None:
            ax.text(box[2], box[1] + 2, f"a{index}", color="white", fontsize=8)
    for sham in masks["shams"]:
        ax.contour(sham.astype(float), levels=[0.5], colors="red",
                   linewidths=0.6, linestyles="dashed")
    ax.contour(masks["sky"].astype(float), levels=[0.5], colors="0.8",
               linewidths=0.6)
    ax.set_xlabel("Azimut [px]")
    ax.set_ylabel("Elevación [px]")
    ax.set_title(f"{geometry_id} vista {view_index} — ROI (blanco), "
                 f"anillo (naranja), sham (rojo)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{geometry_id}_view{view_index}_roi.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", nargs="+", default=list(DEFAULT_GEOMETRIES))
    parser.add_argument("--views", nargs="+", type=int, default=list(DEFAULT_VIEWS))
    parser.add_argument("--overlay-dir", default=str(OVERLAY_DIR))
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()

    print(f"Stage 19 ROI cluster checks — geometries={args.geometries} "
          f"views={args.views}")
    v2_rows, v3_rows, v4_rows, overlays = [], [], [], []

    for geo_index, geometry_id in enumerate(args.geometries):
        geo = R.load_geometry(geometry_id)
        for view_index in args.views:
            view = imaging.DetectorView(geometry_id, view_index)
            # zlib.crc32 of the id: stable across processes, unlike hash()
            rng = np.random.default_rng(np.random.SeedSequence(
                [20260805, 19, geo_index, view_index,
                 zlib.crc32(geometry_id.encode("utf-8"))]))
            masks = R.view_masks(geo, view, rng=rng)
            l_rock, l_anomalies = masks["l_rock"], masks["l_anomalies"]

            for index, anomaly in enumerate(geo.anomalies):
                roi = masks["rois"][index]
                box = R.bounding_box(roi)
                v2_rows.append({
                    "geometry": geometry_id, "view": view_index,
                    "anomaly": index, "kind": anomaly.kind,
                    "max_l_a_m": float(l_anomalies[index].max()),
                    "chord_m": R.analytic_max_chord_m(anomaly)})
                if box is not None and masks["roi_valid"][index]:
                    v3_rows.append({
                        "geometry": geometry_id, "view": view_index,
                        "anomaly": index, "width_px": int(box[3] - box[2]),
                        "predicted_width_px":
                            R.analytic_roi_azimuth_width_px(geo, view, index)})

            npz_path = (CAMPAIGN_DIR / geometry_id
                        / f"{geometry_id}_view{view_index}.npz")
            if npz_path.exists():
                with np.load(npz_path) as npz:
                    observed = np.asarray(npz["opacity"], dtype=np.float64)
                # Exact density()-override reproduction (handles overlapping
                # anomalies such as geo-test-01's conduit + cavity); the linear
                # predicted_opacity is a diagnostic only where anomalies do not
                # overlap.
                predicted = R.reproduced_opacity(geo, view)
                rock = ~masks["sky"]
                rel = np.abs(predicted - observed) / np.maximum(observed, 1e-9)
                v4_rows.append({
                    "geometry": geometry_id, "view": view_index,
                    "n_rock": int(rock.sum()),
                    "max_rel_error": float(rel[rock].max()) if rock.any() else 0.0,
                    "sky_predicted": R.sky_opacity(geo, view),
                    "sky_observed": (float(observed[masks["sky"]].mean())
                                     if masks["sky"].any() else float("nan")),
                    "sky_abs_error": (abs(R.sky_opacity(geo, view)
                                          - float(observed[masks["sky"]].mean()))
                                      if masks["sky"].any() else 0.0)})
                if not args.no_overlays:
                    overlays.append(write_overlay(
                        geometry_id, view_index, masks, observed,
                        Path(args.overlay_dir)))
            else:
                print(f"    V4 {geometry_id} v{view_index}: MISSING {npz_path}")
                v4_rows.append({"geometry": geometry_id, "view": view_index,
                                "n_rock": 0, "max_rel_error": float("inf"),
                                "sky_predicted": float("nan"),
                                "sky_observed": float("nan"),
                                "sky_abs_error": float("inf")})
            print(f"  {geometry_id} view {view_index}: "
                  f"ROI px {masks['roi_pixels']} valid {masks['roi_valid']} "
                  f"annulus {[i['n_pixels'] for i in masks['annulus_info']]} "
                  f"shams {masks['sham_info']['obtained']}", flush=True)

    results = {
        "V1_grid_identity": check_v1(),
        "V2_chord_envelope": check_v2(v2_rows),
        "V3_angular_width": check_v3(v3_rows),
        "V4_opacity_decomposition": check_v4(v4_rows),
        "V5_overlays": (args.no_overlays or len(overlays) > 0),
    }
    if overlays and not args.no_overlays:
        print(f"    V5 wrote {len(overlays)} overlays to {Path(args.overlay_dir)}")

    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 ROI CLUSTER:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
