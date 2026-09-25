#!/usr/bin/env python3
"""Stage 22 — the exposure-reduction factor. Section 6 of protocol v1.1, run.

equivalence.py holds the machinery (PAVA, PCHIP over log10 f',
Brent, the boundary rules, the structural cap, the cluster bootstrap) but
nothing in the repository called it: the factor existed as a capability, not as
a result. This module is the driver that turns it into one, from the Stage 20
cache and nothing else.

WHAT IT PRODUCES, and why each column is there:

  one row per METHOD          raw 1/f identity, the frozen classical, the
                              U-Net L1, the cWGAN -- because H9 of the
                              execution plan measured the classical alone at
                              about 27x at 1 %. Most of the absolute factor is
                              the deterministic 1/f amplitude restoration, not
                              learning. Reporting only the cWGAN row would read
                              as concealment when a reader recomputes it.
  an INCREMENTAL column       method factor minus the raw-identity factor: the
                              part attributable to the model, stated next to
                              the absolute one instead of left to be inferred.
  both CONVENTIONS            the raw curve in the `raw` convention (comparable
                              with the literature) and in the 1/f-corrected one
                              (the conservative bound). Same enhanced score,
                              two reference curves.
  SSIM and PSNR               the primary metric and its companion.
  per-anomaly-type CNR        via the analytic Poisson anchor
                              N_CNR = (CNR_enh / CNR_raw)^2, which is what
                              section 6 step 11 asks for: the dense grid stores
                              global metrics only, so a CNR curve to invert does
                              not exist and inventing one would be worse than
                              using the closed form.
  the STRUCTURAL CAP          N <= 1/f, printed in EVERY row. At f = 0.01 no
                              method can be worth more than 100x because 100 %
                              of the events is all there is; a factor quoted
                              without its cap reads as an unbounded gain.

UNCERTAINTY. B = 10 000 replicates of the TWO-STAGE cluster bootstrap of
section 5.4 -- cells drawn with replacement, then the K = 20 realizations
drawn within each selected cell -- re-deriving BOTH curves in every replicate
(section 6 step 9). B = 10 000 is the pre-registered number and is what the
CLI runs; the tests use a smaller B for speed and say so. Resampling only the
enhanced arm would treat the raw curve as known when it was measured in the
same pass and carries the same cluster structure.

Run:  python scripts/evaluation/factor_analysis.py --cache results/evaluation/val
      python scripts/evaluation/factor_analysis.py --cache <root> --bootstrap 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import equivalence as EQ  # noqa: E402
import report_tables as RT  # noqa: E402
import stats as S  # noqa: E402

B_DEFAULT = 10_000               # pre-registered (section 5.4), see the header
SEED_DEFAULT = 20260805
ALPHA = 0.05

ARM_THINNING_GRID = "I"
RAW, CORRECTED = "raw", "corrected"
CONVENTIONS = (RAW, CORRECTED)
CONVENTION_LABELS = {RAW: "cruda", CORRECTED: "corregida 1/f"}
METRICS = ("ssim", "psnr_db")
METRIC_LABELS = {"ssim": "SSIM", "psnr_db": "PSNR [dB]"}

RAW_IDENTITY = "crudo 1/f (identidad)"
METHODS = (
    (RAW_IDENTITY, RT.ARM_RAW_CORRECTED),
    ("mejor clásica", RT.ARM_CLASSICAL),
    ("U-Net L1", RT.ARM_UNET_L1),
    ("cWGAN-GP", RT.ARM_CWGAN),
)


# --------------------------------------------------------------- cache access
def cell_key(row) -> str:
    return f"{row['geometry']}|v{row['view']}"


def raw_curves_by_cell(rows, convention: str, metric: str,
                       fractions=EQ.DENSE_FRACTIONS):
    """{cell: (K, n_fractions)} from the dense thinning grid (arm I).

    Two dimensions per cell, not one: the two-stage bootstrap needs the
    individual realizations to resample within a drawn cell. A cell whose K
    differs between fractions cannot form a rectangle, so it collapses to its
    per-fraction means and the caller is told how many cells that happened to.
    """
    grid = RT.select(rows, arm=ARM_THINNING_GRID, scope="global",
                     convention=convention)
    per_cell = {}
    for row in grid:
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue
        index = next((i for i, f in enumerate(fractions)
                      if np.isclose(f, row["fraction"], rtol=0.0, atol=1e-9)),
                     None)
        if index is None:
            continue
        per_cell.setdefault(cell_key(row), {}).setdefault(index, []).append(
            float(value))

    curves, collapsed, incomplete = {}, [], []
    for cell, by_fraction in sorted(per_cell.items()):
        if len(by_fraction) != len(fractions):
            incomplete.append(cell)
            continue
        sizes = {len(values) for values in by_fraction.values()}
        if len(sizes) == 1:
            k = sizes.pop()
            curves[cell] = np.array([[by_fraction[i][j]
                                      for i in range(len(fractions))]
                                     for j in range(k)], dtype=np.float64)
        else:
            collapsed.append(cell)
            curves[cell] = np.array([[float(np.mean(by_fraction[i]))
                                      for i in range(len(fractions))]],
                                    dtype=np.float64)
    return curves, {"cells": len(curves), "collapsed_cells": collapsed,
                    "incomplete_cells": incomplete}


def enhanced_by_cell(rows, arm: str, level: float, metric: str):
    """{cell: array of the per-realization scores of one arm at one level}."""
    per_cell = {}
    for row in RT.select(rows, arm=arm, scope="global", fraction=level):
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue
        per_cell.setdefault(cell_key(row), []).append(float(value))
    return {cell: np.asarray(values, dtype=np.float64)
            for cell, values in sorted(per_cell.items())}


def align(curves: dict, enhanced: dict):
    """Restrict both mappings to the cells they share, in one order."""
    shared = [cell for cell in curves if cell in enhanced]
    return ({cell: curves[cell] for cell in shared},
            {cell: enhanced[cell] for cell in shared})


# ------------------------------------------------------------------- factors
def factor_row(rows, method: str, arm: str, level: float, convention: str,
               metric: str, curves: dict, b: int, seed: int) -> dict:
    """One method x level x convention x metric factor, with its interval."""
    enhanced = enhanced_by_cell(rows, arm, level, metric)
    paired_curves, paired_enhanced = align(curves, enhanced)
    common = {"method": method, "arm": arm, "level": float(level),
              "convention": convention, "metric": metric,
              "structural_cap": EQ.structural_cap(level)}
    if not paired_curves or not paired_enhanced:
        return {**common, "status": "sin datos", "factor": float("nan"),
                "f_prime": float("nan"), "equivalent_days": float("nan"),
                "n_cells": 0, "bootstrap": None}
    result = EQ.bootstrap_factor(
        EQ.DENSE_FRACTIONS, paired_curves, paired_enhanced, float(level),
        b=b, rng=np.random.default_rng(seed), alpha=ALPHA)
    return {**common, "status": result["status"], "factor": result["factor"],
            "capped_factor": result.get("capped_factor", float("nan")),
            "f_prime": result["f_prime"],
            "equivalent_days": result["equivalent_days"],
            "score": result["ssim_enhanced"],
            "curve_at_lo": result["curve_at_lo"],
            "curve_at_hi": result["curve_at_hi"],
            "note": result.get("note", ""),
            "n_cells": len(paired_curves),
            "bootstrap": result["bootstrap"]}


def cnr_factor_rows(rows, level: float, z_arms=None, b: int = 2000,
                    seed: int = SEED_DEFAULT) -> list:
    """Per-anomaly-type CNR factors from the analytic Poisson anchor.

    CNR of a Poisson image scales as sqrt(f), so the exposure factor implied by
    a CNR ratio is that ratio squared (section 6 step 11). The dense grid holds
    global metrics only -- no ROI, therefore no CNR curve to invert -- so this
    closed form is the honest route rather than a curve fabricated from four
    frozen levels. The denominator is the raw 1/f arm at the SAME level, and
    the interval is a cluster bootstrap over cells of the two cell-mean CNRs.
    """
    z_arms = ([arm for _name, arm in METHODS if arm != RT.ARM_RAW_CORRECTED]
              if z_arms is None else z_arms)
    reference = {}
    for row in RT.select(rows, arm=RT.ARM_RAW_CORRECTED, scope="roi",
                         fraction=level, roi_valid=1):
        if row.get("CNR") is not None and np.isfinite(row["CNR"]):
            reference.setdefault(row["anomaly_kind"], {}).setdefault(
                cell_key(row), []).append(float(row["CNR"]))

    out = []
    for arm in z_arms:
        enhanced = {}
        for row in RT.select(rows, arm=arm, scope="roi", fraction=level,
                             roi_valid=1):
            if row.get("CNR") is not None and np.isfinite(row["CNR"]):
                enhanced.setdefault(row["anomaly_kind"], {}).setdefault(
                    cell_key(row), []).append(float(row["CNR"]))
        for kind in sorted(set(reference) & set(enhanced)):
            cells = sorted(set(reference[kind]) & set(enhanced[kind]))
            if not cells:
                continue
            ratios = [np.asarray(enhanced[kind][c], dtype=np.float64).mean()
                      / np.asarray(reference[kind][c], dtype=np.float64).mean()
                      for c in cells
                      if np.asarray(reference[kind][c]).mean() != 0.0]
            if not ratios:
                continue
            per_cell = [np.asarray([r], dtype=np.float64) for r in ratios]
            boot = S.cluster_bootstrap(
                per_cell, b=b, rng=np.random.default_rng(seed))
            lo, hi = S.percentile_ci(boot ** 2, ALPHA)
            mean_ratio = float(np.mean(ratios))
            out.append({
                "method": next(name for name, a in METHODS if a == arm),
                "arm": arm, "level": float(level), "metric": "CNR",
                "anomaly_kind": kind, "n_cells": len(cells),
                "cnr_ratio": mean_ratio,
                "factor": EQ.cnr_cross_check(mean_ratio, 1.0),
                "ci_lo": float(lo), "ci_hi": float(hi),
                "structural_cap": EQ.structural_cap(level),
            })
    return out


class FactorAnalysis:
    """The whole of section 6 over one cache: curves, rows, incremental column."""

    def __init__(self, rows, levels=RT.LEVELS, b=B_DEFAULT, seed=SEED_DEFAULT):
        self.levels = tuple(levels)
        self.b = int(b)
        self.seed = int(seed)
        self.curves, self.curve_info, self.rows = {}, {}, []
        self.cnr_rows = []

        for convention in CONVENTIONS:
            for metric in METRICS:
                curves, info = raw_curves_by_cell(rows, convention, metric)
                self.curve_info[(convention, metric)] = info
                if not curves:
                    continue
                self.curves[(convention, metric)] = EQ.fit_raw_curve(
                    EQ.DENSE_FRACTIONS,
                    {c: np.asarray(v).mean(axis=0) for c, v in curves.items()})
                for level in self.levels:
                    for method, arm in METHODS:
                        self.rows.append(factor_row(
                            rows, method, arm, level, convention, metric,
                            curves, self.b, self.seed))

        for level in self.levels:
            self.cnr_rows.extend(cnr_factor_rows(
                rows, level, b=min(self.b, 2000), seed=self.seed))
        self._add_incremental()

    def _add_incremental(self):
        """method factor - raw-identity factor, at the same level/convention/metric."""
        baseline = {(row["level"], row["convention"], row["metric"]):
                    row["factor"] for row in self.rows
                    if row["method"] == RAW_IDENTITY}
        for row in self.rows:
            reference = baseline.get((row["level"], row["convention"],
                                      row["metric"]), float("nan"))
            row["raw_identity_factor"] = reference
            row["incremental_factor"] = (row["factor"] - reference
                                         if np.isfinite(row["factor"])
                                         and np.isfinite(reference)
                                         else float("nan"))

    def curve(self, convention=CORRECTED, metric="ssim"):
        return self.curves.get((convention, metric))

    def rows_for(self, convention=CORRECTED, metric="ssim"):
        return [row for row in self.rows
                if row["convention"] == convention and row["metric"] == metric]

    def to_json(self) -> dict:
        return {
            "stage": "22 (exposure reduction factor)",
            "bootstrap": {"b": self.b, "seed": self.seed,
                          "scheme": "two-stage cluster (cells, then K within "
                                    "each drawn cell), both curves re-derived "
                                    "per replicate",
                          "alpha": ALPHA},
            "dense_fractions": list(EQ.DENSE_FRACTIONS),
            "structural_caps": {str(k): v for k, v in
                                EQ.STRUCTURAL_CAPS.items()},
            "curves": {f"{convention}|{metric}": curve.summary()
                       for (convention, metric), curve in self.curves.items()},
            "curve_inputs": {f"{c}|{m}": info
                             for (c, m), info in self.curve_info.items()},
            "factor_rows": self.rows,
            "cnr_rows": self.cnr_rows,
        }


# -------------------------------------------------------------------- tables
def _fmt(value, digits=2, suffix=""):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}{suffix}"


def markdown_tables(analysis: FactorAnalysis) -> str:
    parts = []
    for convention in CONVENTIONS:
        for metric in METRICS:
            rows = analysis.rows_for(convention, metric)
            if not rows:
                continue
            header = ["Nivel", "Método", "f'", "N", "IC 95 % (bootstrap)",
                      "N incremental", "Cota estructural", "Días equiv.",
                      "Estado", "n celdas"]
            body = []
            for row in rows:
                boot = row.get("bootstrap") or {}
                body.append([
                    f"{row['level'] * 100:g} %", row["method"],
                    _fmt(row["f_prime"], 4),
                    _fmt(row["factor"], 2, "×"),
                    (f"[{_fmt(boot.get('lo'), 2)}, {_fmt(boot.get('hi'), 2)}]"
                     if boot else "—"),
                    _fmt(row.get("incremental_factor"), 2, "×"),
                    _fmt(row["structural_cap"], 0, "×"),
                    _fmt(row["equivalent_days"], 1),
                    row["status"], str(row["n_cells"])])
            parts.append(
                f"### Factor por método — curva {CONVENTION_LABELS[convention]}"
                f", métrica {METRIC_LABELS[metric]}\n\n"
                + RT.markdown_table(header, body))

    if analysis.cnr_rows:
        header = ["Nivel", "Método", "Tipo de anomalía", "Razón CNR",
                  "N = (razón)²", "IC 95 %", "Cota estructural", "n celdas"]
        body = [[f"{row['level'] * 100:g} %", row["method"],
                 row["anomaly_kind"], _fmt(row["cnr_ratio"], 3),
                 _fmt(row["factor"], 2, "×"),
                 f"[{_fmt(row['ci_lo'], 2)}, {_fmt(row['ci_hi'], 2)}]",
                 _fmt(row["structural_cap"], 0, "×"), str(row["n_cells"])]
                for row in analysis.cnr_rows]
        parts.append("### Factor CNR por tipo de anomalía "
                     "(ancla analítica N = (razón CNR)², conteo Poisson)\n\n"
                     + RT.markdown_table(header, body))

    caps = ", ".join(f"f={k:g} → {v}×"
                     for k, v in sorted(EQ.STRUCTURAL_CAPS.items()))
    parts.append(
        "\n".join([
            "",
            f"Incertidumbre: bootstrap cluster de dos etapas "
            f"(celdas con reemplazo, luego K dentro de cada celda sorteada), "
            f"B = {analysis.b}, ambas curvas re-derivadas por réplica "
            f"(sección 6, paso 9).",
            "",
            f"**Cotas estructurales N ≤ 1/f: {caps}.** Ningún método puede "
            f"valer más que la cota: el 100 % de los eventos es todo lo que "
            f"hay.",
            "",
            "La columna «N incremental» es el factor del método menos el de "
            "la identidad cruda 1/f: separa la restauración determinista de "
            "amplitud de lo aportado por el modelo aprendido (hallazgo H9 del "
            "plan de ejecución, pre-registrado).",
        ]))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------- main
def build(cache_root, b=B_DEFAULT, seed=SEED_DEFAULT) -> FactorAnalysis:
    rows = RT.load_metrics(Path(cache_root) / "metrics.csv")
    summary = RT.load_summary(cache_root)
    RT.assert_thresholds_are_frozen(summary, RT.load_thresholds())
    return FactorAnalysis(rows, b=b, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default=str(RT.DEFAULT_OUT))
    parser.add_argument("--bootstrap", type=int, default=B_DEFAULT,
                        help="B of the two-stage cluster bootstrap "
                             f"(pre-registered: {B_DEFAULT})")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    analysis = build(args.cache, args.bootstrap, args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = analysis.to_json()
    (out_dir / "factor_tables.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out_dir / "factor_tables.md").write_text(markdown_tables(analysis),
                                              encoding="utf-8")

    print(f"Stage 22 factor analysis — B = {analysis.b}, "
          f"{len(analysis.rows)} factor rows, {len(analysis.cnr_rows)} CNR rows")
    for row in analysis.rows_for(CORRECTED, "ssim"):
        boot = row.get("bootstrap") or {}
        print(f"  f={row['level'] * 100:>4g} % {row['method']:<24} "
              f"N = {_fmt(row['factor'], 2, 'x'):>8} "
              f"[{_fmt(boot.get('lo'), 2)}, {_fmt(boot.get('hi'), 2)}] "
              f"incremental {_fmt(row.get('incremental_factor'), 2, 'x')} "
              f"cap {_fmt(row['structural_cap'], 0, 'x')}")
    print(f"  wrote {out_dir / 'factor_tables.md'}")
    print(f"  wrote {out_dir / 'factor_tables.json'}")
    print("STAGE 22 FACTOR ANALYSIS:", "PASSED" if analysis.rows else "FAILED")
    return 0 if analysis.rows else 1


if __name__ == "__main__":
    sys.exit(main())
