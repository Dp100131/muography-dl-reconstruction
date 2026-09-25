"""Stage 19 — exposure-reduction factor (protocol v1.1 section 6, PRE-REGISTERED).

The claim of the thesis is "this model buys N times the exposure". The number
comes from inverting the no-model quality curve: if the enhanced image at
fraction f scores what a raw image scores at fraction f', then the model is
worth N = f'/f, and 30 f' is the equivalent number of measuring days.

The pre-registered recipe, step by step (section 6):

  1. both curves from the SAME test pass
  2. raw grid F' = {0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90},
     K = 20 realizations, seeded
  3. both input conventions (raw = comparable with the literature,
     1/f-corrected = the conservative bound)
  4. cell means as the unit
  5. per-cell monotonicity guard: PAVA, with the number of adjusted cells
     REPORTED, never silently applied
  6. PCHIP interpolation over log10 f'
  7. Brent root finding on [0.01, 0.90]; N = f'/f
  8. boundaries: above the curve at 0.90 -> lower bound ">= 0.90/f"; below the
     curve at 0.01 -> "no measurable gain"; the STRUCTURAL cap N <= 1/f is
     printed next to every factor
  9. uncertainty: cluster bootstrap re-deriving BOTH curves per replicate
 10. days = 30 f'

The structural cap matters: at f = 0.01 no method can be worth more than 100x,
because 100 % of the events is all there is. A factor quoted without it would
be read as an unbounded gain.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

DENSE_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90)
SOLVE_LO, SOLVE_HI = 0.01, 0.90
REFERENCE_DAYS = 30.0

# N <= 1/f: the whole dataset is the ceiling (section 6 step 8)
STRUCTURAL_CAPS = {0.5: 2, 0.1: 10, 0.05: 20, 0.01: 100}

FACTOR = "factor"
LOWER_BOUND = "lower_bound"
NO_GAIN = "no_gain"


def structural_cap(f_input: float) -> float:
    """1/f, from the frozen table when the level is one of the four."""
    for level, cap in STRUCTURAL_CAPS.items():
        if np.isclose(level, f_input, rtol=0.0, atol=1e-9):
            return float(cap)
    return float(1.0 / f_input)


# ---------------------------------------------------------------------- PAVA
def pava(y, weights=None) -> np.ndarray:
    """Pool adjacent violators — the least-squares non-decreasing fit.

    Quality must not fall when more events are kept; a cell whose measured
    curve dips is Poisson jitter, not physics. PAVA is the minimal repair:
    it replaces each violating run by its weighted mean and leaves everything
    else untouched.
    """
    y = np.asarray(y, dtype=np.float64).copy()
    n = y.size
    if n < 2:
        return y
    w = (np.ones(n, dtype=np.float64) if weights is None
         else np.asarray(weights, dtype=np.float64).copy())
    values, counts, blocks = [], [], []
    for i in range(n):
        values.append(y[i] * w[i])
        counts.append(w[i])
        blocks.append(1)
        while len(values) > 1 and (values[-2] / counts[-2]
                                   > values[-1] / counts[-1]):
            # pop FIRST: `values[-2] += values.pop()` would resolve the target
            # index against the already-shortened list
            value, count, size = values.pop(), counts.pop(), blocks.pop()
            values[-1] += value
            counts[-1] += count
            blocks[-1] += size
    out = np.empty(n, dtype=np.float64)
    position = 0
    for value, count, size in zip(values, counts, blocks):
        out[position:position + size] = value / count
        position += size
    return out


def is_monotone(y, tolerance: float = 0.0) -> bool:
    return bool(np.all(np.diff(np.asarray(y, dtype=np.float64)) >= -tolerance))


# ---------------------------------------------------------------------- curve
class RawCurve:
    """Quality of the no-model image as a function of the kept fraction.

    Holds the per-cell PCHIP interpolators over log10 f' and the mean curve.
    `__call__` evaluates the mean curve, which is the one section 6 inverts;
    the per-cell fits exist so the bootstrap can re-derive the curve from a
    resample of cells without re-reading anything.
    """

    def __init__(self, fractions, cell_curves, n_adjusted, adjusted_cells):
        self.fractions = np.asarray(fractions, dtype=np.float64)
        self.log_f = np.log10(self.fractions)
        self.cell_curves = np.asarray(cell_curves, dtype=np.float64)
        self.mean_curve = self.cell_curves.mean(axis=0)
        self.n_cells = self.cell_curves.shape[0]
        self.n_adjusted = int(n_adjusted)
        self.adjusted_cells = list(adjusted_cells)
        self._interp = PchipInterpolator(self.log_f, self.mean_curve,
                                         extrapolate=False)
        # LAZY: the bootstrap of section 6 step 9 builds one RawCurve per
        # replicate and inverts only the mean curve, so eagerly fitting one
        # PCHIP per cell would multiply the cost of B = 10 000 replicates by
        # the 18 cells for output nobody reads on that path
        self._cell_interps = None

    def __call__(self, fraction):
        return np.asarray(self._interp(np.log10(np.asarray(fraction,
                                                           dtype=np.float64))))

    def cell(self, index, fraction):
        if self._cell_interps is None:
            self._cell_interps = [PchipInterpolator(self.log_f, row,
                                                    extrapolate=False)
                                  for row in self.cell_curves]
        return float(self._cell_interps[index](np.log10(float(fraction))))

    def summary(self) -> dict:
        return {
            "fractions": [float(f) for f in self.fractions],
            "mean_curve": [float(v) for v in self.mean_curve],
            "n_cells": self.n_cells,
            "n_cells_adjusted_by_pava": self.n_adjusted,
            "adjusted_cells": self.adjusted_cells,
            "monotone_after_pava": is_monotone(self.mean_curve, 1e-12),
        }


def fit_raw_curve(fractions, ssim_per_cell, cell_labels=None) -> RawCurve:
    """Per-cell PCHIP over log10 f', with the PAVA guard and its count.

    ssim_per_cell: (n_cells, n_fractions) or a mapping cell -> sequence.
    Fractions are sorted ascending internally, so callers may pass the grid in
    any order.
    """
    if isinstance(ssim_per_cell, dict):
        cell_labels = list(ssim_per_cell) if cell_labels is None else cell_labels
        matrix = np.asarray([ssim_per_cell[c] for c in cell_labels],
                            dtype=np.float64)
    else:
        matrix = np.atleast_2d(np.asarray(ssim_per_cell, dtype=np.float64))
        if cell_labels is None:
            cell_labels = [f"cell{i}" for i in range(matrix.shape[0])]
    fractions = np.asarray(fractions, dtype=np.float64)
    if matrix.shape[1] != fractions.size:
        raise ValueError(f"curve needs one value per fraction: got "
                         f"{matrix.shape[1]} for {fractions.size} fractions")
    order = np.argsort(fractions)
    fractions, matrix = fractions[order], matrix[:, order]

    repaired, adjusted = [], []
    for label, row in zip(cell_labels, matrix):
        if is_monotone(row, 1e-12):
            repaired.append(row)
        else:
            repaired.append(pava(row))
            adjusted.append(str(label))
    return RawCurve(fractions, np.asarray(repaired), len(adjusted), adjusted)


# ------------------------------------------------------------------- solving
def solve_factor(curve, ssim_enhanced: float, f_input: float,
                 lo: float = SOLVE_LO, hi: float = SOLVE_HI) -> dict:
    """Invert the raw curve at ssim_enhanced; apply the boundary rules.

    Returns a dict with `status` in {factor, lower_bound, no_gain}, the
    equivalent fraction f', the factor N = f'/f, the equivalent days 30 f' and
    the structural cap 1/f, which is ALWAYS reported next to N.
    """
    cap = structural_cap(f_input)
    at_hi, at_lo = float(curve(hi)), float(curve(lo))
    common = {"ssim_enhanced": float(ssim_enhanced), "f_input": float(f_input),
              "structural_cap": cap, "curve_at_hi": at_hi, "curve_at_lo": at_lo,
              "solve_range": [float(lo), float(hi)]}

    if not np.isfinite(ssim_enhanced):
        return {**common, "status": NO_GAIN, "f_prime": float("nan"),
                "factor": float("nan"), "equivalent_days": float("nan"),
                "note": "the enhanced score is not finite"}
    if ssim_enhanced >= at_hi:
        return {**common, "status": LOWER_BOUND, "f_prime": float(hi),
                "factor": float(hi / f_input),
                "capped_factor": float(min(hi / f_input, cap)),
                "equivalent_days": REFERENCE_DAYS * hi,
                "note": f"above the raw curve at f'={hi}: the factor is a "
                        f"lower bound (>= {hi}/f)"}
    if ssim_enhanced <= at_lo:
        return {**common, "status": NO_GAIN, "f_prime": float(lo),
                "factor": float(lo / f_input),
                "capped_factor": float(min(lo / f_input, cap)),
                "equivalent_days": REFERENCE_DAYS * lo,
                "note": f"below the raw curve at f'={lo}: no measurable gain"}

    f_prime = float(brentq(lambda f: float(curve(f)) - ssim_enhanced, lo, hi,
                           xtol=1e-12, rtol=8.9e-16))
    return {**common, "status": FACTOR, "f_prime": f_prime,
            "factor": float(f_prime / f_input),
            "capped_factor": float(min(f_prime / f_input, cap)),
            "equivalent_days": REFERENCE_DAYS * f_prime,
            "note": ""}


def _cell_stack(by_cell, labels):
    """(n_cells, K, n_fractions) from per-cell curves or per-cell realizations.

    A cell entry may be a single curve (n_fractions,) -- already averaged over
    the K realizations -- or the raw stack (K, n_fractions). Both are promoted
    to three dimensions so one code path serves the single-stage and the
    two-stage bootstrap.
    """
    stacks = []
    for label in labels:
        array = np.atleast_2d(np.asarray(by_cell[label], dtype=np.float64))
        stacks.append(array)
    widths = {array.shape[1] for array in stacks}
    if len(widths) != 1:
        raise ValueError(f"cells disagree on the number of fractions: {widths}")
    depths = {array.shape[0] for array in stacks}
    if len(depths) != 1:
        raise ValueError(f"cells disagree on the number of realizations: "
                         f"{depths}; a ragged K breaks the two-stage draw")
    return np.stack(stacks)


def bootstrap_factor(fractions, ssim_by_cell, enhanced_by_cell, f_input: float,
                     b: int = 2000, rng=None, alpha: float = 0.05,
                     resampler=None) -> dict:
    """Cluster bootstrap that re-derives BOTH curves in every replicate.

    Resampling only the enhanced score would treat the raw curve as known;
    it is measured in the same pass and carries the same cluster structure, so
    step 9 of section 6 requires re-fitting it per replicate. Replicates whose
    solution hits a boundary are counted, not discarded: a factor whose
    interval touches the grid edge must say so.

    ssim_by_cell[cell] and enhanced_by_cell[cell] may each be

        1-D  the cell curve / the cell mean, already reduced over K. The draw
             is then a ONE-stage cluster bootstrap over cells.
        2-D  (K, n_fractions) -- and (K,) for the enhanced arm -- the raw
             realizations. The draw is then the TWO-STAGE cluster bootstrap of
             section 5.4: cells with replacement, then K within each drawn
             cell, which is what propagates the within-cell Poisson jitter
             into the factor interval instead of assuming it away.

    `resampler` overrides the cell-level draw: a callable (rng, n_cells) that
    returns an integer index array, so a caller can inject its own cluster
    scheme without this module having to know about it.
    """
    labels = list(ssim_by_cell)
    curves = _cell_stack(ssim_by_cell, labels)              # (cells, K, F)
    enhanced_raw = [np.atleast_1d(np.asarray(enhanced_by_cell[c],
                                             dtype=np.float64))
                    for c in labels]
    two_stage = curves.shape[1] > 1
    rng = np.random.default_rng(20260805) if rng is None else rng
    n_cells = len(labels)
    draw_cells = (resampler if resampler is not None
                  else (lambda r, n: r.integers(0, n, size=n)))

    cell_curves = curves.mean(axis=1)                       # (cells, F)
    enhanced = np.array([float(values.mean()) for values in enhanced_raw])
    point = solve_factor(fit_raw_curve(fractions, cell_curves, labels),
                         float(enhanced.mean()), f_input)

    factors, statuses = [], {FACTOR: 0, LOWER_BOUND: 0, NO_GAIN: 0}
    for _ in range(b):
        pick = np.asarray(draw_cells(rng, n_cells), dtype=int)
        if two_stage:
            rows, drawn_enhanced = [], []
            for index in pick:
                block = curves[index]
                within = rng.integers(0, block.shape[0], size=block.shape[0])
                rows.append(block[within].mean(axis=0))
                values = enhanced_raw[index]
                drawn_enhanced.append(float(
                    values[rng.integers(0, values.size, size=values.size)].mean()))
            matrix = np.asarray(rows, dtype=np.float64)
            score = float(np.mean(drawn_enhanced))
        else:
            matrix = cell_curves[pick]
            score = float(enhanced[pick].mean())
        curve = fit_raw_curve(fractions, matrix, [labels[i] for i in pick])
        result = solve_factor(curve, score, f_input)
        statuses[result["status"]] += 1
        factors.append(result["factor"])
    factors = np.asarray(factors, dtype=np.float64)
    finite = factors[np.isfinite(factors)]
    return {
        **point,
        "bootstrap": {
            "b": int(b),
            "two_stage": bool(two_stage),
            "k_per_cell": int(curves.shape[1]),
            "lo": float(np.percentile(finite, 100.0 * alpha / 2.0))
            if finite.size else float("nan"),
            "hi": float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
            if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "status_counts": statuses,
            "boundary_fraction": float(
                (statuses[LOWER_BOUND] + statuses[NO_GAIN]) / b) if b else 0.0,
        },
    }


def cnr_cross_check(cnr_enhanced: float, cnr_raw_reference: float) -> float:
    """Analytic anchor: CNR scales as sqrt(f) for Poisson counts, so the
    exposure factor implied by a CNR ratio is that ratio squared. Section 6
    step 11 requires it next to the CNR-based factor."""
    if not (np.isfinite(cnr_enhanced) and np.isfinite(cnr_raw_reference)) \
            or cnr_raw_reference == 0.0:
        return float("nan")
    return float((cnr_enhanced / cnr_raw_reference) ** 2)
