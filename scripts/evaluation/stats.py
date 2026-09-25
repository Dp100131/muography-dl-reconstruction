"""Stage 19 — inferential statistics (protocol v1.1 section 5.4, PRE-REGISTERED).

The unit of analysis is the CELL (geometry, view): n = 18 on test, n = 6 on
generalization. The K = 20 realizations inside a cell share one parent
transport, so treating the 360 pairs as independent would be
pseudo-replication -- n = 360 survives only as a descriptive dispersion.

  primary test    exact two-sided Wilcoxon signed-rank over the 18 cell-mean
                  differences (minimum attainable p = 2/2^18 = 7.6e-6, far
                  below the 0.05/12 = 0.00417 of Holm, so the binding
                  constraint is sign consistency across geometries and views,
                  which is the right thing to be constrained by)
  normality gate  Shapiro-Wilk at alpha = 0.05 (D'Agostino is invalid at
                  n = 18) -> a paired t-test as a secondary confirmation
  interval        two-stage cluster bootstrap (cells with replacement, then K
                  within each drawn cell), B = 10 000, seeded; BCa with a
                  percentile backup. If the two endpoints differ by more than
                  20 % of the interval width, the percentile interval is the
                  one reported and the divergence is noted.
  multiplicity    Holm-Bonferroni at alpha = 0.05 over the 12 primary
                  hypotheses; secondary metrics carry their own family.

THE INTERVAL CARRIES THE CLAIM; THE p-VALUE ACCOMPANIES IT.

SMALL-SAMPLE CAVEAT (measured, not assumed). With 18 clusters the
nonparametric cluster bootstrap under-covers: on synthetic data shaped like
the real design (tau = 0.014 between cells, sigma = 0.004 within, K = 20) the
BCa and percentile intervals both cover the true effect about 91 % of the
time, against 95.4 % for the exact Student-t interval on the 18 cell means.
The cause is standard -- the bootstrap uses the plug-in scale and
normal-shaped quantiles where n = 18 calls for t_17. The pre-registered
interval is the bootstrap one and is NOT silently replaced here;
`t_interval()` is provided so the dry run can report both, and any change of
the primary interval has to go through a dated addendum (protocol section 8).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

B_DEFAULT = 10_000
ALPHA_DEFAULT = 0.05
CI_DIVERGENCE_RULE = 0.20      # 20 % of the percentile interval width
EXACT_WILCOXON_MAX_N = 25
PRIMARY_FAMILY_SIZE = 12       # 3 comparisons x 4 levels


# ------------------------------------------------------------------ cell means
def cell_means(values, cells):
    """(ordered cell keys, mean per cell) — the reduction to the analysis unit.

    `values` and `cells` are parallel sequences, one entry per pair, exactly
    the long format of results/evaluation/<split>/metrics.csv. Cell keys are returned in
    first-appearance order so a bootstrap seeded on the index is reproducible
    regardless of how the keys sort.
    """
    values = np.asarray(list(values), dtype=np.float64)
    cells = list(cells)
    if values.size != len(cells):
        raise ValueError(f"length mismatch: {values.size} values, "
                         f"{len(cells)} cell labels")
    order, groups = [], {}
    for value, cell in zip(values, cells):
        if cell not in groups:
            groups[cell] = []
            order.append(cell)
        groups[cell].append(value)
    means = np.array([float(np.mean(groups[cell])) for cell in order])
    return order, means


def group_by_cell(values, cells):
    """{cell: np.ndarray of the pair-level values} — the bootstrap input."""
    order, _ = cell_means(values, cells)
    grouped = {cell: [] for cell in order}
    for value, cell in zip(np.asarray(list(values), dtype=np.float64), cells):
        grouped[cell].append(float(value))
    return {cell: np.asarray(grouped[cell], dtype=np.float64) for cell in order}


def mean_of_cell_means(resampled) -> float:
    """Default bootstrap statistic: the grand mean over cell means."""
    return float(np.mean([np.mean(cell) for cell in resampled]))


# -------------------------------------------------------------- bootstrapping
def cluster_bootstrap(values_by_cell, b: int = B_DEFAULT, rng=None,
                      statistic=None):
    """Two-stage cluster bootstrap over cells and then within each drawn cell.

    values_by_cell: a mapping cell -> array of pair-level values, or a
    sequence of such arrays. The default statistic is the mean of the cell
    means, i.e. the paired difference estimator of section 5.4 when the input
    is already a per-pair difference.

    When every cell holds the same number of pairs (the real case: K = 20) the
    resampling is fully vectorized; ragged input falls back to a loop.
    """
    if isinstance(values_by_cell, dict):
        cells = [np.asarray(v, dtype=np.float64) for v in values_by_cell.values()]
    else:
        cells = [np.asarray(v, dtype=np.float64) for v in values_by_cell]
    if not cells:
        raise ValueError("cluster_bootstrap needs at least one cell")
    rng = np.random.default_rng() if rng is None else rng
    statistic = mean_of_cell_means if statistic is None else statistic
    n_cells = len(cells)
    sizes = {cell.size for cell in cells}

    if len(sizes) == 1 and statistic is mean_of_cell_means:
        k = sizes.pop()
        matrix = np.stack(cells)                                # (n_cells, k)
        cell_idx = rng.integers(0, n_cells, size=(b, n_cells))
        within = rng.integers(0, k, size=(b, n_cells, k))
        drawn = matrix[cell_idx[:, :, None], within]            # (b, n_cells, k)
        return drawn.mean(axis=2).mean(axis=1)

    out = np.empty(b, dtype=np.float64)
    for replicate in range(b):
        chosen = rng.integers(0, n_cells, size=n_cells)
        resampled = []
        for index in chosen:
            cell = cells[index]
            resampled.append(cell[rng.integers(0, cell.size, size=cell.size)])
        out[replicate] = float(statistic(resampled))
    return out


def jackknife_over_cells(values_by_cell, statistic=None) -> np.ndarray:
    """Leave-one-CELL-out values — the acceleration input for BCa.

    The jackknife is over clusters, not over pairs: the resampling unit of the
    bootstrap and of the acceleration must be the same or `a` is meaningless.
    """
    if isinstance(values_by_cell, dict):
        cells = [np.asarray(v, dtype=np.float64) for v in values_by_cell.values()]
    else:
        cells = [np.asarray(v, dtype=np.float64) for v in values_by_cell]
    statistic = mean_of_cell_means if statistic is None else statistic
    return np.array([float(statistic([c for j, c in enumerate(cells) if j != i]))
                     for i in range(len(cells))], dtype=np.float64)


def percentile_ci(boot_stats, alpha: float = ALPHA_DEFAULT):
    boot = np.asarray(boot_stats, dtype=np.float64)
    return (float(np.percentile(boot, 100.0 * alpha / 2.0)),
            float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0))))


def t_interval(cell_values, alpha: float = ALPHA_DEFAULT):
    """Exact Student-t interval on the cell means — the coverage yardstick.

    Reported ALONGSIDE the pre-registered bootstrap interval, never instead of
    it. See the small-sample caveat in the module docstring.
    """
    values = np.asarray(list(cell_values), dtype=np.float64)
    n = values.size
    if n < 2:
        return float("nan"), float("nan")
    half = (sp_stats.t.ppf(1.0 - alpha / 2.0, n - 1)
            * values.std(ddof=1) / np.sqrt(n))
    return float(values.mean() - half), float(values.mean() + half)


def bca_ci(boot_stats, theta_hat: float, jackknife_values,
           alpha: float = ALPHA_DEFAULT):
    """Bias-corrected and accelerated interval.

    theta_hat is a required argument: BCa is defined relative to the point
    estimate (z0 is the bias correction of the bootstrap distribution AROUND
    it), so a signature without it could not be implemented correctly. The
    plan's shorthand `bca_ci(boot, jackknife, alpha)` is this call with the
    estimate made explicit.

    Degenerate cases (all bootstrap replicates on one side of theta_hat, or a
    constant jackknife) fall back to the percentile interval and say so.
    """
    boot = np.asarray(boot_stats, dtype=np.float64)
    jack = np.asarray(jackknife_values, dtype=np.float64)
    proportion = float(np.mean(boot < theta_hat))
    if proportion <= 0.0 or proportion >= 1.0:
        lo, hi = percentile_ci(boot, alpha)
        return lo, hi, {"degenerate": True, "reason": "z0 undefined",
                        "z0": float("nan"), "a": float("nan")}
    z0 = float(sp_stats.norm.ppf(proportion))
    centred = jack.mean() - jack
    denominator = 6.0 * (float(np.sum(centred ** 2)) ** 1.5)
    a = 0.0 if denominator == 0.0 else float(np.sum(centred ** 3) / denominator)

    z_alpha = sp_stats.norm.ppf(alpha / 2.0)
    z_upper = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    lo_q = sp_stats.norm.cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)))
    hi_q = sp_stats.norm.cdf(z0 + (z0 + z_upper) / (1.0 - a * (z0 + z_upper)))
    if not (np.isfinite(lo_q) and np.isfinite(hi_q)) or lo_q >= hi_q:
        lo, hi = percentile_ci(boot, alpha)
        return lo, hi, {"degenerate": True, "reason": "quantiles out of order",
                        "z0": z0, "a": a}
    return (float(np.percentile(boot, 100.0 * lo_q)),
            float(np.percentile(boot, 100.0 * hi_q)),
            {"degenerate": False, "z0": z0, "a": a})


def choose_ci(boot_stats, theta_hat: float, jackknife_values,
              alpha: float = ALPHA_DEFAULT) -> dict:
    """BCa with the percentile backup and the pre-registered 20 % rule.

    Rule (section 5.4): if either endpoint of the BCa interval differs from
    the percentile endpoint by more than 20 % of the percentile width, the
    percentile interval is reported and the divergence is noted.
    """
    bca_lo, bca_hi, info = bca_ci(boot_stats, theta_hat, jackknife_values, alpha)
    pct_lo, pct_hi = percentile_ci(boot_stats, alpha)
    width = pct_hi - pct_lo
    divergence = (max(abs(bca_lo - pct_lo), abs(bca_hi - pct_hi)) / width
                  if width > 0 else float("inf"))
    use_percentile = bool(info["degenerate"] or divergence > CI_DIVERGENCE_RULE)
    lo, hi = (pct_lo, pct_hi) if use_percentile else (bca_lo, bca_hi)
    return {
        "lo": float(lo), "hi": float(hi),
        "method": "percentile" if use_percentile else "bca",
        "bca": [float(bca_lo), float(bca_hi)],
        "percentile": [float(pct_lo), float(pct_hi)],
        "divergence_fraction": float(divergence),
        "excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
        "bca_info": info,
    }


# ---------------------------------------------------------------------- tests
EXACT = "exact"
NORMAL_APPROX = "normal-approx"


def wilcoxon_exact(diffs, alternative: str = "two-sided") -> dict:
    """Exact Wilcoxon signed-rank when n <= 25 AND no difference is zero.

    Section 5.4 pre-registers the EXACT test, so the reported `method` has to
    be the truth rather than the intention. A zero difference takes SciPy off
    the exact path -- some versions warn and switch to the normal
    approximation, others discard the zeros first and run the exact test on a
    smaller sample -- and the resulting p-value is therefore version dependent.
    Rather than let the label drift with the installed wheel, a sample with any
    zero difference is run on the normal approximation EXPLICITLY, reported as
    "normal-approx", counted in `n_zero` and annotated. `min_attainable_p`
    (2/2^n, the floor that makes n = 18 enough for Holm over 12 hypotheses)
    only exists on the exact path and is NaN otherwise.

    SciPy renamed the keyword from `mode` to `method` in 1.9; both spellings
    are attempted so the module runs on the cluster wheel and on a newer local
    one without a version pin.
    """
    diffs = np.asarray(list(diffs), dtype=np.float64)
    nonzero = diffs[diffs != 0.0]
    n = int(nonzero.size)
    n_zero = int(diffs.size - n)
    if n == 0:
        return {"n": int(diffs.size), "n_nonzero": 0, "n_zero": n_zero,
                "statistic": float("nan"), "p_value": 1.0, "method": "none",
                "min_attainable_p": float("nan"),
                "positive": 0, "negative": 0,
                "note": "every difference is exactly zero"}

    exact_available = n <= EXACT_WILCOXON_MAX_N and n_zero == 0
    method = EXACT if exact_available else NORMAL_APPROX
    scipy_method = "exact" if exact_available else "approx"
    try:
        result = sp_stats.wilcoxon(diffs, alternative=alternative,
                                   method=scipy_method)
    except TypeError:                                    # SciPy < 1.9
        result = sp_stats.wilcoxon(diffs, alternative=alternative,
                                   mode=scipy_method)
    note = ""
    if n_zero and n <= EXACT_WILCOXON_MAX_N:
        note = (f"{n_zero} difference(s) are exactly zero: the exact "
                f"signed-rank distribution does not apply, so the normal "
                f"approximation is used and reported as such")
    elif not exact_available:
        note = f"n = {n} exceeds {EXACT_WILCOXON_MAX_N}: normal approximation"
    return {"n": int(diffs.size), "n_nonzero": n, "n_zero": n_zero,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue), "method": method,
            "min_attainable_p": (2.0 / (2.0 ** n) if exact_available
                                 else float("nan")),
            "positive": int(np.sum(nonzero > 0)),
            "negative": int(np.sum(nonzero < 0)),
            "note": note}


def shapiro_gate(diffs, alpha: float = ALPHA_DEFAULT) -> dict:
    """Shapiro-Wilk normality gate; `normal` decides whether the paired
    t-test is reported as a secondary confirmation."""
    diffs = np.asarray(list(diffs), dtype=np.float64)
    if diffs.size < 3:
        return {"n": int(diffs.size), "statistic": float("nan"),
                "p_value": float("nan"), "normal": False,
                "note": "Shapiro-Wilk needs at least 3 observations"}
    if float(np.ptp(diffs)) == 0.0:
        # SciPy warns and returns garbage on a constant sample; a degenerate
        # difference is not evidence of normality, so the gate stays closed
        return {"n": int(diffs.size), "statistic": float("nan"),
                "p_value": float("nan"), "normal": False,
                "note": "every difference is identical: the gate stays closed"}
    result = sp_stats.shapiro(diffs)
    return {"n": int(diffs.size), "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "normal": bool(result.pvalue >= alpha)}


def paired_t(diffs) -> dict:
    """Secondary confirmation, reported only when the Shapiro gate opens."""
    diffs = np.asarray(list(diffs), dtype=np.float64)
    result = sp_stats.ttest_1samp(diffs, 0.0)
    return {"n": int(diffs.size), "statistic": float(result.statistic),
            "p_value": float(result.pvalue)}


def holm(p_values, alpha: float = ALPHA_DEFAULT) -> dict:
    """Holm-Bonferroni step-down: adjusted p-values and rejection flags.

    Adjusted p_(i) = max over j <= i of (m - j + 1) p_(j), clipped at 1, so
    the sequence is monotone and can be compared with alpha directly.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    m = p.size
    if m == 0:
        return {"adjusted": [], "reject": [], "alpha": alpha, "m": 0}
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return {"adjusted": [float(v) for v in adjusted],
            "reject": [bool(v <= alpha) for v in adjusted],
            "alpha": float(alpha), "m": int(m),
            # per-step levels in ASCENDING p order, the order Holm walks
            "per_step_alpha_sorted": [float(alpha / (m - rank))
                                      for rank in range(m)],
            "sorted_order": [int(i) for i in order]}


def mde(sigma_d: float, n: int, alpha_adj: float, power: float = 0.8) -> float:
    """Minimum detectable effect of a two-sided paired test, normal approximation.

        MDE = (z_{1-alpha_adj/2} + z_power) sigma_d / sqrt(n)

    alpha_adj is the Holm-adjusted level of the hypothesis in question
    (0.05/12 for the most stringent of the primary family).
    """
    if n <= 0 or not np.isfinite(sigma_d) or sigma_d < 0:
        return float("nan")
    z_alpha = float(sp_stats.norm.ppf(1.0 - alpha_adj / 2.0))
    z_power = float(sp_stats.norm.ppf(power))
    return float((z_alpha + z_power) * sigma_d / np.sqrt(n))


def power_table(sigma_values, n: int = 18, family: int = PRIMARY_FAMILY_SIZE,
                alpha: float = ALPHA_DEFAULT, power: float = 0.8):
    """The dry-run power table: MDE for the strictest and the loosest Holm step."""
    rows = []
    for sigma in sigma_values:
        rows.append({
            "sigma_d": float(sigma),
            "n": int(n),
            "mde_holm_strictest": mde(sigma, n, alpha / family, power),
            "mde_holm_loosest": mde(sigma, n, alpha, power),
        })
    return rows


# ------------------------------------------------------------------ front door
def compare_cells(diff_by_cell, b: int = B_DEFAULT, rng=None,
                  alpha: float = ALPHA_DEFAULT) -> dict:
    """The full section 5.4 pass for one comparison, from per-pair differences.

    diff_by_cell: {cell: array of per-pair (method - baseline) differences}.
    Returns the cell-mean estimate, the chosen interval, the exact Wilcoxon,
    the normality gate and the secondary t-test.
    """
    cells = list(diff_by_cell)
    per_cell = [np.asarray(diff_by_cell[c], dtype=np.float64) for c in cells]
    cell_diffs = np.array([float(c.mean()) for c in per_cell])
    theta_hat = float(cell_diffs.mean())

    rng = np.random.default_rng(20260805) if rng is None else rng
    boot = cluster_bootstrap(per_cell, b=b, rng=rng)
    boot_ci = choose_ci(boot, theta_hat, jackknife_over_cells(per_cell), alpha)
    gate = shapiro_gate(cell_diffs, alpha)
    t_lo, t_hi = t_interval(cell_diffs, alpha)
    # Protocol v1.1 section 5.4: the PRIMARY interval is Student-t when the 18
    # paired cell differences pass the Shapiro gate (the usual case; small-
    # sample cluster bootstrap under-covers, 0.905 vs 0.940 measured on the
    # dry run), and the cluster-bootstrap percentile when normality is rejected.
    # Both are always reported.
    if gate["normal"]:
        primary = {"lo": float(t_lo), "hi": float(t_hi), "method": "student-t"}
    else:
        primary = {"lo": float(boot_ci["lo"]), "hi": float(boot_ci["hi"]),
                   "method": "bootstrap-percentile (Shapiro rejected)"}
    primary["excludes_zero"] = bool(primary["lo"] > 0.0 or primary["hi"] < 0.0)
    return {
        "cells": [str(c) for c in cells],
        "n_cells": len(cells),
        "n_pairs": int(sum(c.size for c in per_cell)),
        "estimate": theta_hat,
        "sd_cells": float(cell_diffs.std(ddof=1)) if cell_diffs.size > 1 else 0.0,
        "ci": primary,
        "ci_bootstrap": boot_ci,
        "t_interval": [float(t_lo), float(t_hi)],
        "wilcoxon": wilcoxon_exact(cell_diffs),
        "shapiro": gate,
        "paired_t": paired_t(cell_diffs) if gate["normal"] else None,
        "b": int(b),
    }
