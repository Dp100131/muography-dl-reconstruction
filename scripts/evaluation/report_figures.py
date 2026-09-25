#!/usr/bin/env python3
"""Stage 20/21/22 figures — everything drawn from the cached tidy metrics.

Five figure families, all from results/evaluation/<split>/metrics.csv (plus
octiles.csv and the prediction cache for the gallery). Nothing re-reads a
shard, so a figure can be regenerated as often as the drafting needs it.

  stage20_quality_vs_level    SSIM and PSNR against the kept fraction, log-x,
                              with the cluster interval per arm
  stage20_per_sample          the 360 pairs behind one cell mean: a strip plot
                              over a violin, so the pseudo-replication the
                              statistics discard is still visible
  stage22_equivalence         the raw curve with the enhanced scores projected
                              onto it and the resulting factor annotated. The
                              projection comes from factor_analysis
                              -- Brent on the PCHIP fit, with the bootstrap
                              interval and the structural cap -- so the figure
                              and the factor table can never disagree.
  stage21_detectability       z against the physical perturbation DX by
                              path-length octile, with the detection threshold
  stage21_gallery             input / classical / L1 / cWGAN / reference with
                              the ROI contour and C, CNR, z printed

The stochastic ceiling drawn on the quality curve is READ from the JSON that
ceiling.py regenerates, not typed in as 0.905: the Stage 18 number
is only citable because a script reproduces it.

Figure text is Spanish (the deliverable language); identifiers stay English.

Run:  python scripts/evaluation/report_figures.py --cache results/evaluation/val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

import cache as cache_module  # noqa: E402
import equivalence as EQ  # noqa: E402
import report_tables as RT  # noqa: E402
import stats as S  # noqa: E402
from muography_dataset import preprocess_pair_corrected  # noqa: E402
from plotting_style import apply_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_FIG_DIR = REPO_ROOT / "figures" / "results"
DEFAULT_CEILING_JSON = REPO_ROOT / "results" / "evaluation" / "ceiling.json"
CURVE_ARMS = ("B", "C", "D", "E", "F")
MARKERS = {"B": "o", "C": "s", "D": "^", "E": "v", "F": "D"}
GALLERY_ARMS = ("D", "E", "F", "A")
N_WORST_CASES = 10               # execution plan 2.3 item 5


def load_ceiling(path=DEFAULT_CEILING_JSON):
    """The regenerated stochastic SSIM ceiling, or None when it is absent."""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("ssim_cell_mean")
    return float(value) if value is not None and np.isfinite(value) else None


def _cell_summary(rows, arm, level, metric, b, seed):
    subset = RT.select(rows, arm=arm, scope="global", fraction=level)
    if not subset:
        return None
    grouped = RT.by_cell(subset, metric)
    if not grouped:
        return None
    cells = list(grouped.values())
    estimate = float(np.mean([cell.mean() for cell in cells]))
    boot = S.cluster_bootstrap(cells, b=b, rng=np.random.default_rng(seed))
    interval = S.choose_ci(boot, estimate, S.jackknife_over_cells(cells))
    return {"estimate": estimate, "lo": interval["lo"], "hi": interval["hi"],
            "n_cells": len(cells),
            "pairs": np.concatenate(cells) if cells else np.array([])}


def figure_quality_vs_level(rows, path, metric="ssim", b=2000, seed=20260805,
                            ceiling=None):
    levels = sorted({row["fraction"] for row in rows
                     if row.get("scope") == "global"
                     and row.get("arm") in CURVE_ARMS
                     and row.get("fraction") is not None})
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    drawn = 0
    for arm in CURVE_ARMS:
        xs, ys, lo, hi = [], [], [], []
        for level in levels:
            summary = _cell_summary(rows, arm, level, metric, b, seed)
            if summary is None:
                continue
            xs.append(level)
            ys.append(summary["estimate"])
            lo.append(summary["estimate"] - summary["lo"])
            hi.append(summary["hi"] - summary["estimate"])
        if not xs:
            continue
        drawn += 1
        ax.errorbar(xs, ys, yerr=[lo, hi], marker=MARKERS.get(arm, "o"),
                    capsize=4, label=RT.ARM_NAMES.get(arm, arm))
    if ceiling is not None:
        ax.axhline(ceiling, linestyle="--", color="0.4", linewidth=1.0)
        ax.text(levels[0] if levels else 0.01, ceiling,
                f" techo estocástico {ceiling:.3f}", va="bottom", fontsize=8,
                color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("Fracción de eventos conservada")
    ax.set_ylabel("SSIM" if metric == "ssim" else "PSNR [dB]")
    ax.set_title("Calidad frente al nivel de estadística "
                 "(media por celda, IC 95 % cluster)")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return drawn


def figure_per_sample(rows, path, level=0.05, metric="ssim"):
    """The 360 individual pairs behind each cell mean, arm by arm."""
    data, labels = [], []
    for arm in CURVE_ARMS:
        subset = RT.select(rows, arm=arm, scope="global", fraction=level)
        values = np.array([row[metric] for row in subset
                           if row.get(metric) is not None], dtype=np.float64)
        if values.size:
            data.append(values)
            labels.append(RT.ARM_NAMES.get(arm, arm))
    if not data:
        return 0
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.violinplot(data, showextrema=False, widths=0.8)
    rng = np.random.default_rng(7)
    for index, values in enumerate(data, start=1):
        jitter = index + 0.10 * rng.standard_normal(values.size)
        ax.plot(jitter, values, ".", markersize=2.5, alpha=0.45)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("SSIM" if metric == "ssim" else "PSNR [dB]")
    ax.set_title(f"Dispersión por par a f = {level * 100:g} % "
                 f"(n = {data[0].size}; descriptiva, no inferencial)")
    fig.savefig(path)
    plt.close(fig)
    return len(data)


def figure_equivalence(factors, path, metric="ssim", convention="corrected",
                       method="cWGAN-GP"):
    """The raw curve with the SOLVED equivalent fractions projected onto it.

    `factors` is factor_analysis.FactorAnalysis.to_json() (live, or
    reloaded from factor_tables.json). The projection is the Brent root of the
    PCHIP fit, with the bootstrap interval and the structural cap, exactly as
    section 6 defines it -- NOT the smallest grid point that happens to clear
    the enhanced score. That older shortcut reported the nearest of nine
    tabulated fractions, which rounds a factor of 17.4x up to 20x at f = 0.01
    and, worse, drew a number the factor table did not contain.
    """
    key = f"{convention}|{metric}"
    curve_summary = (factors.get("curves") or {}).get(key)
    if not curve_summary:
        return 0
    fractions = np.asarray(curve_summary["fractions"], dtype=np.float64)
    means = np.asarray(curve_summary["mean_curve"], dtype=np.float64)
    curve = EQ.fit_raw_curve(fractions, means[np.newaxis, :])

    rows = [row for row in factors.get("factor_rows", [])
            if row["convention"] == convention and row["metric"] == metric
            and row["method"] == method]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    dense = np.logspace(np.log10(fractions.min()), np.log10(fractions.max()),
                        200)
    ax.plot(dense, curve(dense), "-", linewidth=1.4,
            label=f"curva cruda ({RT.ARM_NAMES.get('C', 'cruda')}, "
                  f"{'corregida 1/f' if convention == 'corrected' else 'cruda'})")
    ax.plot(fractions, means, "o", markersize=4, color="0.35",
            label="grilla densa medida (K=20)")

    drawn = 0
    for row in sorted(rows, key=lambda r: r["level"]):
        value, f_prime = row.get("score"), row.get("f_prime")
        if value is None or not np.isfinite(value) or not np.isfinite(f_prime):
            continue
        drawn += 1
        boot = row.get("bootstrap") or {}
        ax.axhline(value, linewidth=0.6, color="0.8")
        ax.plot([row["level"], f_prime], [value, value], ":", color="0.4")
        ax.plot([f_prime], [value], "*", markersize=9)
        interval = (f" IC [{boot['lo']:.1f}, {boot['hi']:.1f}]"
                    if boot and np.isfinite(boot.get("lo", np.nan)) else "")
        suffix = {EQ.LOWER_BOUND: " (cota inferior)",
                  EQ.NO_GAIN: " (sin ganancia medible)"}.get(row["status"], "")
        ax.annotate(f"f={row['level'] * 100:g} % → f'={f_prime * 100:.2g} % "
                    f"(N={row['factor']:.1f}×{interval}, "
                    f"cota {row['structural_cap']:.0f}×){suffix}",
                    xy=(f_prime, value), fontsize=7.5,
                    xytext=(3, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Fracción de eventos conservada f'")
    ax.set_ylabel("SSIM" if metric == "ssim" else "PSNR [dB]")
    ax.set_title(f"Curva de equivalencia y factor de reducción — {method}")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return 1 if drawn else 0


def figure_detectability(octile_rows, path, z_thr=4.0, arms=("D", "E", "F")):
    """z against the physical perturbation DX, by path-length octile."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    drawn = 0
    for arm in arms:
        xs = [row["delta_opacity_mean"] for row in octile_rows
              if row["arm"] == arm and row["delta_opacity_mean"] is not None
              and row["z"] is not None]
        ys = [row["z"] for row in octile_rows
              if row["arm"] == arm and row["delta_opacity_mean"] is not None
              and row["z"] is not None]
        if not xs:
            continue
        drawn += 1
        ax.plot(np.abs(xs), np.abs(ys), MARKERS.get(arm, "o"), markersize=4,
                alpha=0.6, label=RT.ARM_NAMES.get(arm, arm))
    ax.axhline(z_thr, linestyle="--", color="0.35", linewidth=1.0)
    ax.text(ax.get_xlim()[0], z_thr, f" z_thr = {z_thr:.2f}", va="bottom",
            fontsize=8, color="0.3")
    ax.set_xlabel("|ΔX| = |Δρ| L · 100  [g/cm²]")
    ax.set_ylabel("|z| de detección")
    ax.set_title("Detectabilidad por octiles de longitud de camino")
    if drawn:
        ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return drawn


def figure_gallery(panels, path, title="Galería de casos"):
    """panels: [{'label': str, 'image': 2-D array, 'contour': mask|None,
    'caption': str}] rendered as one row."""
    if not panels:
        return 0
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(2.6 * len(panels), 3.2))
    axes = np.atleast_1d(axes)
    for axis, panel in zip(axes, panels):
        axis.imshow(panel["image"], origin="lower", cmap="viridis",
                    vmin=0.0, vmax=1.0)
        if panel.get("contour") is not None:
            axis.contour(np.asarray(panel["contour"], dtype=float),
                         levels=[0.5], colors="white", linewidths=1.0)
        axis.set_title(panel["label"], fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        if panel.get("caption"):
            axis.set_xlabel(panel["caption"], fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path)
    plt.close(fig)
    return len(panels)


def _cached_arm(cache_root, geometry, arm):
    """(images, extras) of one cached (geometry, arm) block, or None."""
    path = Path(cache_root) / geometry / f"arm_{arm}.npz"
    if not path.exists():
        return None
    return cache_module.load_predictions(path)


def _pick(extra, view, fraction, k):
    """Row index of one (view, fraction, k) inside a cached block."""
    matches = np.flatnonzero(
        (np.asarray(extra["view"]) == view)
        & np.isclose(np.asarray(extra["fraction"], dtype=np.float64), fraction,
                     rtol=0.0, atol=1e-9)
        & (np.asarray(extra["k"]) == k))
    return int(matches[0]) if matches.size else None


def worst_cases(rows, arm="F", limit=N_WORST_CASES):
    """The `limit` worst ROI rows of one arm by local SSIM (plan 2.3 item 5)."""
    candidates = [row for row in RT.select(rows, arm=arm, scope="roi",
                                           roi_valid=1)
                  if row.get("local_ssim") is not None
                  and np.isfinite(row["local_ssim"])]
    return sorted(candidates, key=lambda row: row["local_ssim"])[:limit]


def gallery_panels(cache_root, case):
    """The five panels of one case, straight from the Stage 20 cache.

    The INPUT panel is regenerated from the integer reference counts stored in
    masks.npz, thinned with the arm-I seed of the same (fraction, k): arms B
    and C hold no images in the cache, so the honest alternative to a
    regenerated input is no input panel at all -- and a gallery that shows the
    enhanced image without the degraded one it came from is not evidence of
    anything. The panel is labelled as a regeneration, not passed off as the
    shard's own realization.
    """
    geometry, view = case["geometry"], case["view"]
    fraction, k = case["fraction"], case["k"]
    mask_path = Path(cache_root) / geometry / "masks.npz"
    if not mask_path.exists():
        return []
    masks = cache_module.load_masks(mask_path)
    views = [int(v) for v in masks["views"]]
    if view not in views:
        return []
    row_index = views.index(view)
    contour = None
    if masks.get("rois") is not None and np.asarray(masks["rois"]).size:
        rois = np.asarray(masks["rois"])
        if case["anomaly_idx"] is not None and \
                0 <= case["anomaly_idx"] < rois.shape[1]:
            contour = rois[row_index, case["anomaly_idx"]]

    panels = []
    counts = np.asarray(masks.get("ref_counts"))
    if counts.size and counts.ndim == 3:
        ref_counts = counts[row_index].astype(np.int64)
        fraction_index = next(
            (i for i, f in enumerate(EQ.DENSE_FRACTIONS)
             if np.isclose(f, fraction, rtol=0.0, atol=1e-9)), None)
        if fraction_index is not None:
            import run_evaluation as RE

            rng = np.random.default_rng(
                RE.thinning_seed(fraction_index, k, 0, view))
            low = rng.binomial(ref_counts, fraction)
            image, _ = preprocess_pair_corrected(low, ref_counts, fraction)
            panels.append({"label": f"entrada f={fraction * 100:g} %",
                           "image": image, "contour": contour,
                           "caption": "regenerada del cache (semilla arm I)"})

    labels = {"D": "clásica", "E": "U-Net L1", "F": "cWGAN-GP",
              "A": "referencia 30 d"}
    for arm in GALLERY_ARMS:
        block = _cached_arm(cache_root, geometry, arm)
        if block is None:
            continue
        images, extra = block
        index = _pick(extra, view, 1.0 if arm == "A" else fraction,
                      -1 if arm == "A" else k)
        if index is None:
            continue
        caption = ""
        if arm == case["arm"]:
            caption = (f"C={case['C']:.3f} CNR={case['CNR']:.2f} "
                       f"z={case['z']:.1f}")
        panels.append({"label": labels[arm], "image": images[index],
                       "contour": contour, "caption": caption})
    return panels


def load_octiles(path):
    import csv
    rows = []
    numeric = ("fraction", "l_lo_m", "l_hi_m", "l_mean_m",
               "delta_opacity_mean", "C", "z")
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for column in numeric:
                row[column] = RT._number(row.get(column))
            for column in ("view", "k", "anomaly_idx", "octile", "n_roi"):
                row[column] = RT._number(row.get(column), int)
            rows.append(row)
    return rows


def resolve_factors(cache, path_hint, bootstrap, seed=20260805):
    """The factor analysis: reloaded from factor_tables.json, else computed.

    Preferring the file makes the figure and the table the same numbers by
    construction, and skips a bootstrap that has already been paid for.
    """
    candidate = Path(path_hint) if path_hint else \
        RT.DEFAULT_OUT / "factor_tables.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8")), str(candidate)
    import factor_analysis as FA

    analysis = FA.build(cache, b=bootstrap, seed=seed)
    return analysis.to_json(), f"computed in-process (B={bootstrap})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--metric", default="ssim", choices=("ssim", "psnr_db"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--z-thr", type=float, default=None)
    parser.add_argument("--ceiling", type=float, default=None,
                        help="overrides the value read from --ceiling-json")
    parser.add_argument("--ceiling-json", default=str(DEFAULT_CEILING_JSON),
                        help="output of ceiling.py; the ceiling is "
                             "READ from it, never typed in")
    parser.add_argument("--factors", default=None,
                        help="factor_tables.json from "
                             "factor_analysis.py; computed "
                             "in-process when absent")
    parser.add_argument("--factor-bootstrap", type=int, default=2000,
                        help="B used only when the factor tables have to be "
                             "computed here; the reported numbers come from "
                             "factor_analysis.py at its own B")
    parser.add_argument("--gallery-cases", type=int, default=N_WORST_CASES)
    args = parser.parse_args()

    apply_style()
    cache = Path(args.cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = RT.load_metrics(cache / "metrics.csv")
    thresholds = RT.load_thresholds()
    z_thr = args.z_thr if args.z_thr is not None else thresholds["z_thr"]
    ceiling = (args.ceiling if args.ceiling is not None
               else load_ceiling(args.ceiling_json))
    print(f"Stage 20 figures — {len(rows)} rows from {cache}")
    print(f"  stochastic ceiling: "
          + (f"{ceiling:.4f} (from {args.ceiling_json})" if ceiling is not None
             else "not available — run ceiling.py"))

    written = []
    path = out_dir / "stage20_quality_vs_level.png"
    if figure_quality_vs_level(rows, path, args.metric, args.bootstrap,
                               ceiling=ceiling):
        written.append(path)
    path = out_dir / "stage20_per_sample.png"
    if figure_per_sample(rows, path, 0.05, args.metric):
        written.append(path)

    factors, source = resolve_factors(cache, args.factors,
                                      args.factor_bootstrap)
    print(f"  factor analysis: {source}")
    path = out_dir / "stage22_equivalence.png"
    if figure_equivalence(factors, path, args.metric):
        written.append(path)

    octiles_path = cache / "octiles.csv"
    if octiles_path.exists():
        path = out_dir / "stage21_detectability.png"
        if figure_detectability(load_octiles(octiles_path), path, z_thr):
            written.append(path)

    # the gallery of the worst cases: the qualitative half of Stage 21, and
    # the only place a reader sees what a number like local_ssim = 0.4 looks
    # like. It was implemented and never called.
    cases = worst_cases(rows, "F", args.gallery_cases)
    for rank, case in enumerate(cases, start=1):
        panels = gallery_panels(cache, case)
        if not panels:
            continue
        path = out_dir / f"stage21_gallery_{rank:02d}.png"
        title = (f"Peor caso {rank}: {case['geometry']} vista {case['view']} "
                 f"f={case['fraction'] * 100:g} % k={case['k']} "
                 f"({case['anomaly_kind']}, SSIM local "
                 f"{case['local_ssim']:.3f})")
        if figure_gallery(panels, path, title):
            written.append(path)
    if cases and not any("gallery" in p.name for p in written):
        print("  WARNING: no gallery panel could be built — the prediction "
              "cache or masks.npz is missing from this cache root")

    for item in written:
        print(f"  wrote {item}")
    print("STAGE 20 FIGURES:", "PASSED" if written else "FAILED")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
