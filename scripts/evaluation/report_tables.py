#!/usr/bin/env python3
"""Stage 20/21/22 — master tables from the cached tidy metrics.

Reads ONLY results/evaluation/<split>/metrics.csv (plus the frozen thresholds
of protocol v1.1) and emits Markdown fragments that the reports include
verbatim.
Nothing here re-reads a shard: Stages 21 and 22 are pure post-processing.

The primary family, the S1/S2/S3 verdicts and the detectability rates are
built on the TEST geometries ONLY (protocol v1.1 section 5.4: n = 18 cells =
3 test geometries x 6 views). geo-gen-01 (n = 6) is the generalization
geometry and is reported apart as descriptive -- never folded into the Holm
family it would only dilute. main() splits the rows on the `geometry_split`
column before any table is built; a cache without that column (the validation
dry run predates it) is treated as a single test family so the rehearsal
tables still build.

Tables emitted (execution plan sections 2.2 to 2.5):

  01_levels          per level: mean +- SD over the 360 pairs (DESCRIPTIVE)
                     and the cell mean over the 18 cells with its interval
                     (INFERENTIAL). The two are never mixed: n = 360 is
                     pseudo-replication and only ever describes dispersion.
  02_primary         the 12 pre-registered hypotheses -- cWGAN against raw
                     1/f, against the frozen classical and against the U-Net
                     L1, at each of the four levels -- with the cluster
                     bootstrap interval, the exact Wilcoxon and Holm.
  03_verdicts        S1 / S2 / S3 against the frozen thresholds.
  04_detectability   preserved / lost / inflated / hallucinated rates with
                     cluster intervals, by level and arm.
  05_generalization  the levels table restricted to geo-test's absent twin
                     geo-gen-01 (n = 6 cells), DESCRIPTIVE only -- no Wilcoxon,
                     no Holm, no verdict: it shows the direction of transfer to
                     an unseen geometry, not a confirmatory test of it
                     (subpotenciado por diseno).

Text is Spanish (the deliverable language); identifiers stay English.

Run:  python scripts/evaluation/report_tables.py --cache results/evaluation/val
      python scripts/evaluation/report_tables.py --cache <root> --out results/evaluation/fragments
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import detectability as det  # noqa: E402
import stats as S  # noqa: E402

THRESHOLDS_PATH = REPO_ROOT / "results" / "evaluation" / "protocol_thresholds.json"
DEFAULT_OUT = REPO_ROOT / "results" / "evaluation" / "fragments"

LEVELS = (0.50, 0.10, 0.05, 0.01)
ARM_REFERENCE, ARM_RAW_SHARED, ARM_RAW_CORRECTED = "A", "B", "C"
ARM_CLASSICAL, ARM_UNET_L1, ARM_CWGAN = "D", "E", "F"

LEVEL_TABLE_ARMS = (ARM_RAW_SHARED, ARM_RAW_CORRECTED, ARM_CLASSICAL,
                    ARM_UNET_L1, ARM_CWGAN)
ARM_NAMES = {
    ARM_REFERENCE: "referencia", ARM_RAW_SHARED: "crudo",
    ARM_RAW_CORRECTED: "crudo 1/f", ARM_CLASSICAL: "clásica congelada",
    ARM_UNET_L1: "U-Net L1", ARM_CWGAN: "cWGAN-GP",
}
# the pre-registered primary family: 3 comparisons x 4 levels
PRIMARY_BASELINES = (ARM_RAW_CORRECTED, ARM_CLASSICAL, ARM_UNET_L1)

DEFAULT_THRESHOLDS = {
    "provisional": True,
    "z_thr": 4.0,
    "preserved_tolerance_X": det.PRESERVED_TOLERANCE_X,
    "inflation_tolerance_Y": det.INFLATION_TOLERANCE_Y,
    "s2_floors": {"0.50": {"ssim": 0.885, "psnr_db": None},
                  "0.10": {"ssim": 0.780, "psnr_db": None},
                  "0.05": {"ssim": 0.763, "psnr_db": None},
                  "0.01": {"ssim": 0.750, "psnr_db": None}},
    "s3": {"preserved_min": 0.90, "hallucination_max": 0.05, "lost_max": 0.10,
           "levels_required": ["0.50", "0.10", "0.05"]},
    "ceiling_ssim": 0.905,
}

NUMERIC_COLUMNS = ("fraction", "model_fraction", "ssim", "psnr_db",
                   "local_ssim", "mean_roi", "mean_bg", "std_bg", "C", "CNR",
                   "z", "delta_rho", "delta_opacity_max", "variance_ratio",
                   "nyquist_band_ratio", "residual_high_freq_power",
                   "residual_low_freq_power", "residual_std")
INTEGER_COLUMNS = ("view", "k", "anomaly_idx", "geo_index", "roi_pixels",
                   "annulus_pixels", "n_roi", "n_bg", "roi_valid")


# ----------------------------------------------------------------- loading
def _number(value, cast=float):
    # Float columns coerce to NaN on empty/unparseable so downstream
    # numpy predicates (isfinite, comparisons) behave; integer columns
    # keep None because NaN is not an int.
    missing = float("nan") if cast is float else None
    if value in ("", None):
        return missing
    try:
        return cast(value)
    except (TypeError, ValueError):
        return missing


def load_metrics(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for column in NUMERIC_COLUMNS:
                if column in row:
                    row[column] = _number(row[column])
            for column in INTEGER_COLUMNS:
                if column in row:
                    row[column] = _number(row[column], int)
            rows.append(row)
    return rows


def load_thresholds(path=THRESHOLDS_PATH):
    path = Path(path)
    if not path.exists():
        return {**DEFAULT_THRESHOLDS, "missing_file": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("provisional", False)
    payload["missing_file"] = None
    return payload


def load_summary(cache_root):
    """The run summary written by the driver, or {} when it is absent."""
    path = Path(cache_root) / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def assert_thresholds_are_frozen(summary, thresholds, path=THRESHOLDS_PATH):
    """HARD ABORT when the test split is scored against unfrozen thresholds.

    Protocol v1.1 section 1 (reinforced by section 8): the thresholds are
    pre-registered, and a verdict computed against provisional numbers is a
    post-hoc verdict wearing a pre-registration's clothes. On validation that
    is a rehearsal and a warning is right; on the test split -- which is read
    once and can never be re-read -- there is no recovering from it, because
    the numbers would already have been seen before the thresholds were fixed.
    So this refuses to produce the tables at all rather than print a caveat
    somebody might quote without.
    """
    if str(summary.get("split")) != "test":
        return
    if thresholds.get("missing_file"):
        raise RuntimeError(
            f"REFUSING to score the TEST split: the frozen thresholds file "
            f"{thresholds['missing_file']} does not exist, so these tables "
            f"would be built on the provisional defaults of report_tables.py. "
            f"Freeze protocol v1.1 to zero TBDs and write "
            f"results/evaluation/protocol_thresholds.json first.")
    if thresholds.get("provisional"):
        raise RuntimeError(
            f"REFUSING to score the TEST split: {path} still declares "
            f"\"provisional\": true. z_thr, X, the S2 floors and the S3 rates "
            f"have to be frozen from the DRY RUN before the test numbers are "
            f"looked at; deciding them afterwards is the one thing the "
            f"pre-registration exists to prevent.")


def select(rows, **criteria):
    def matches(row):
        for key, wanted in criteria.items():
            value = row.get(key)
            if isinstance(wanted, (list, tuple, set)):
                if value not in wanted:
                    return False
            elif isinstance(wanted, float) and isinstance(value, float):
                # np.isclose, never ==: the fractions round-trip through a CSV
                # and a float32 shard header, and an exact comparison would
                # silently drop a whole level from the table
                if not np.isclose(value, wanted, rtol=0.0, atol=1e-9):
                    return False
            elif value != wanted:
                return False
        return True
    return [row for row in rows if matches(row)]


def cell_of(row):
    return f"{row['geometry']}|v{row['view']}"


def by_cell(rows, column="ssim"):
    """{cell: array of per-pair values} — the bootstrap and Wilcoxon input."""
    grouped = {}
    for row in rows:
        value = row.get(column)
        if value is None or not np.isfinite(value):
            continue
        grouped.setdefault(cell_of(row), []).append(float(value))
    return {cell: np.asarray(values, dtype=np.float64)
            for cell, values in sorted(grouped.items())}


def paired_differences(rows_a, rows_b, column="ssim"):
    """{cell: array of (a - b) per matched (geometry, view, fraction, k)}.

    Pairing is explicit: a difference is only formed when both arms measured
    the same realization of the same view. An unmatched row is dropped and
    counted, never silently averaged away.
    """
    def index(rows):
        out = {}
        for row in rows:
            value = row.get(column)
            if value is None or not np.isfinite(value):
                continue
            out[(row["geometry"], row["view"], row["fraction"], row["k"])] = \
                float(value)
        return out

    left, right = index(rows_a), index(rows_b)
    keys = sorted(set(left) & set(right))
    grouped = {}
    for key in keys:
        grouped.setdefault(f"{key[0]}|v{key[1]}", []).append(left[key] - right[key])
    dropped = len(set(left) ^ set(right))
    return ({cell: np.asarray(values, dtype=np.float64)
             for cell, values in sorted(grouped.items())}, len(keys), dropped)


def split_by_geometry(rows):
    """Partition the rows into the test family and the generalization family.

    Protocol v1.1 section 5.4 binds the primary family and the S1/S2/S3
    verdicts to the TEST geometries only (n = 18 cells = 3 test geometries x 6
    views). geo-gen-01 is a single geometry (n = 6 cells): its family of tests
    is underpowered by design, so it is reported apart as descriptive and never
    enters the Holm family it would only dilute. A cache whose metrics.csv has
    no `geometry_split` column -- the validation dry run predates it -- carries
    every row into the test family and leaves the generalization family empty,
    so the rehearsal tables still build unchanged.
    """
    if not any(row.get("geometry_split") for row in rows):
        return list(rows), []
    test = [row for row in rows if row.get("geometry_split") == "test"]
    generalization = [row for row in rows
                      if row.get("geometry_split") == "generalization"]
    return test, generalization


# ------------------------------------------------------------------ helpers
def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def markdown_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------- tables
def table_levels(rows, metric="ssim", b=10_000, seed=20260805) -> str:
    """Descriptive dispersion (n pairs) next to the inferential cell mean."""
    header = ["Nivel", "Brazo", "Media ± DE (n pares)", "Media por celda",
              "IC 95 % (método)", "n celdas"]
    body = []
    for level in LEVELS:
        for arm in LEVEL_TABLE_ARMS:
            subset = select(rows, arm=arm, scope="global", fraction=level)
            if not subset:
                continue
            values = np.array([r[metric] for r in subset
                               if r.get(metric) is not None], dtype=np.float64)
            grouped = by_cell(subset, metric)
            cell_values = np.array([v.mean() for v in grouped.values()])
            rng = np.random.default_rng(seed)
            boot = S.cluster_bootstrap(list(grouped.values()), b=b, rng=rng)
            interval = S.choose_ci(boot, float(cell_values.mean()),
                                   S.jackknife_over_cells(list(grouped.values())))
            body.append([
                f"{level * 100:g} %", ARM_NAMES.get(arm, arm),
                f"{fmt(values.mean())} ± {fmt(values.std(ddof=1))} (n={values.size})",
                fmt(float(cell_values.mean())),
                f"[{fmt(interval['lo'])}, {fmt(interval['hi'])}] "
                f"({interval['method']})",
                str(len(grouped))])
    return markdown_table(header, body)


def primary_family(rows, metric="ssim", b=10_000, seed=20260805):
    """The 12 pre-registered comparisons and the Holm result over them.

    Computed once and shared by table_primary and table_verdicts: the S1
    verdict has to read the SAME Holm decision the primary table prints, not a
    per-level test of its own that ignores the multiplicity correction.
    """
    comparisons, p_values = [], []
    for level in LEVELS:
        treatment = select(rows, arm=ARM_CWGAN, scope="global", fraction=level)
        for baseline_arm in PRIMARY_BASELINES:
            baseline = select(rows, arm=baseline_arm, scope="global",
                              fraction=level)
            diff_by_cell, matched, dropped = paired_differences(
                treatment, baseline, metric)
            if not diff_by_cell:
                comparisons.append({"level": level, "baseline": baseline_arm,
                                    "report": None, "matched": 0,
                                    "dropped": dropped})
                p_values.append(1.0)
                continue
            report = S.compare_cells(diff_by_cell, b=b,
                                     rng=np.random.default_rng(seed))
            comparisons.append({"level": level, "baseline": baseline_arm,
                                "report": report, "matched": matched,
                                "dropped": dropped})
            p_values.append(report["wilcoxon"]["p_value"])
    return comparisons, S.holm(p_values)


def find_comparison(comparisons, holm, level, baseline_arm):
    """(comparison, rejects H0 after Holm) for one cell of the family."""
    for index, item in enumerate(comparisons):
        if (item["baseline"] == baseline_arm
                and np.isclose(item["level"], level, rtol=0.0, atol=1e-9)):
            return item, bool(holm["reject"][index])
    return None, False


def table_primary(rows, metric="ssim", b=10_000, seed=20260805) -> str:
    """The 12 pre-registered hypotheses with the interval, Wilcoxon and Holm."""
    comparisons, holm = primary_family(rows, metric, b, seed)
    header = ["Nivel", "Comparación", "Δ media por celda", "IC 95 %", "Método",
              "Wilcoxon p", "p ajustado (Holm)", "Rechaza H0", "n celdas"]
    body = []
    for index, item in enumerate(comparisons):
        label = f"cWGAN − {ARM_NAMES.get(item['baseline'], item['baseline'])}"
        report = item["report"]
        if report is None:
            body.append([f"{item['level'] * 100:g} %", label, "—", "—", "—",
                         "—", "—", "sin datos", "0"])
            continue
        interval = report["ci"]
        body.append([
            f"{item['level'] * 100:g} %", label,
            fmt(report["estimate"]),
            f"[{fmt(interval['lo'])}, {fmt(interval['hi'])}]",
            interval["method"],
            f"{report['wilcoxon']['p_value']:.2e}",
            f"{holm['adjusted'][index]:.2e}",
            "sí" if holm["reject"][index] else "no",
            str(report["n_cells"])])
    note = (f"\n\nFamilia primaria de {holm['m']} hipótesis, Holm–Bonferroni "
            f"α = {holm['alpha']}. El intervalo porta la afirmación; el valor "
            f"p lo acompaña. Unidad de análisis = celda (geometría, vista).")
    return markdown_table(header, body) + note


def table_verdicts(rows, thresholds, metric="ssim", b=10_000,
                   seed=20260805) -> str:
    """S1 / S2 / S3 against the frozen thresholds of protocol v1.1."""
    floors = thresholds.get("s2_floors", {})
    comparisons, holm = primary_family(rows, metric, b, seed)
    header = ["Nivel", "S1 (vence a la clásica)", "S2 (piso absoluto)",
              "S3 (detectabilidad)", "Veredicto"]
    body = []
    for level in LEVELS:
        key = f"{level:.2f}"
        # S1 is a member of the 12-hypothesis primary family, so the interval
        # is not enough on its own: section 5.4 pre-registers Holm-Bonferroni
        # over the whole family, and a per-level interval that excludes zero
        # while its Holm-adjusted p sits above alpha is exactly the multiple
        # comparison the correction exists to refuse.
        item, rejected = find_comparison(comparisons, holm, level,
                                         ARM_CLASSICAL)
        report = item["report"] if item else None
        if report is not None:
            excludes_zero = bool(report["ci"]["excludes_zero"]
                                 and report["estimate"] > 0)
            s1_pass = bool(excludes_zero and rejected)
            s1 = (f"Δ = {fmt(report['estimate'])} "
                  f"[{fmt(report['ci']['lo'])}, {fmt(report['ci']['hi'])}], "
                  f"Holm p = {holm['adjusted'][comparisons.index(item)]:.2e} → "
                  + ("CUMPLE" if s1_pass else
                     ("NO CUMPLE (IC excluye 0 pero no sobrevive Holm)"
                      if excludes_zero else "NO CUMPLE")))
        else:
            s1_pass, s1 = False, "sin datos"

        floor = (floors.get(key) or {}).get("ssim")
        cells = by_cell(select(rows, arm=ARM_CWGAN, scope="global",
                               fraction=level), metric)
        observed = (float(np.mean([v.mean() for v in cells.values()]))
                    if cells else float("nan"))
        if floor is None or not np.isfinite(observed):
            s2_pass, s2 = False, "sin umbral congelado"
        else:
            s2_pass = observed >= floor
            s2 = (f"{fmt(observed)} vs piso {fmt(float(floor))} → "
                  f"{'CUMPLE' if s2_pass else 'NO CUMPLE'}")

        rates = detectability_rates(rows, ARM_CWGAN, level,
                                    z_thr_for(thresholds, ARM_CWGAN, level),
                                    thresholds)
        s3_rules = thresholds.get("s3", DEFAULT_THRESHOLDS["s3"])
        required = key in s3_rules.get("levels_required", [])
        if rates["denominator"] == 0:
            s3_pass, s3 = False, "sin datos"
        else:
            s3_pass = (rates["preserved"] >= s3_rules["preserved_min"]
                       and rates["hallucinated"] <= s3_rules["hallucination_max"]
                       and rates["lost"] <= s3_rules["lost_max"])
            s3 = (f"pres. {rates['preserved']:.1%}, perd. {rates['lost']:.1%}, "
                  f"aluc. {rates['hallucinated']:.1%} → "
                  + ("CUMPLE" if s3_pass else
                     ("NO CUMPLE" if required else "informativo")))
        verdict = ("CUMPLE" if (s1_pass and s2_pass
                                and (s3_pass or not required))
                   else "NO CUMPLE")
        body.append([f"{level * 100:g} %", s1, s2, s3, verdict])
    note = ("\n\nUmbrales " + ("PROVISIONALES (dry-run pendiente)"
                               if thresholds.get("provisional")
                               else "congelados en protocol-v1.1.md")
            + f"; z_thr = {z_thr_for(thresholds, ARM_CWGAN, 0.05)} "
              f"(calibrado por brazo cuando v1.1 lo provee; escalar "
              f"{thresholds['z_thr']} como respaldo), "
              f"X = {thresholds['preserved_tolerance_X']}, "
              f"Y = {thresholds['inflation_tolerance_Y']}. Al 1 % la "
              f"detectabilidad se reporta sin exigir: es el límite de "
              f"aplicabilidad declarado.")
    return markdown_table(header, body) + note


def z_thr_for(thresholds, arm, level) -> float:
    """Per-(arm, level) calibrated detection threshold with scalar fallback.

    The dry run showed the sham |z| distribution shifts by arm (denoising
    shrinks the annulus noise the statistic divides by), so a single scalar
    calibrated on references does not transfer. When protocol_thresholds.json
    carries z_thr_by_arm (protocol v1.1), each arm is judged against its own
    validation-calibrated 99th percentile; the scalar remains the fallback
    for provisional runs and for arms without an entry (the reference).
    """
    by_arm = thresholds.get("z_thr_by_arm") or {}
    per_arm = by_arm.get(arm) or {}
    value = per_arm.get(f"{level:.2f}")
    return float(value) if value is not None else float(thresholds["z_thr"])


def detectability_rates(rows, arm, level, z_thr, thresholds) -> dict:
    """Preserved / lost / inflated rates plus the sham hallucination rate.

    ONLY roi_valid == 1 rows enter the rates. Protocol v1.1 section 5.2
    excludes an ROI below 64 px because its statistics are not measurable, and
    the driver now emits those rows with explicit NaN statistics so the
    exclusion is visible rather than silent. Letting them through here would
    classify a NaN z as "lost" and depress the preservation rate with an
    artefact of the mask size; dropping them without saying so would hide a
    changed denominator. So they are filtered out AND counted, and
    `excluded_roi` is printed as its own row of the table.

    The rates read the independent FLAGS of det.classify, not its label: with
    X frozen above Y an inflated case can also be inside the preservation
    tolerance, and it must appear in both numerators.
    """
    x = thresholds.get("preserved_tolerance_X", det.PRESERVED_TOLERANCE_X)
    y = thresholds.get("inflation_tolerance_Y", det.INFLATION_TOLERANCE_Y)

    roi_rows_all = select(rows, arm=arm, scope="roi", fraction=level)
    if roi_rows_all and all(row.get("roi_valid") is None
                            for row in roi_rows_all):
        raise ValueError(
            "metrics.csv carries no roi_valid column: the ROI exclusions of "
            "protocol v1.1 section 5.2 cannot be honoured, and scoring "
            "without them would silently mix unmeasurable masks into the "
            "detectability rates")

    reference = {}
    for row in select(rows, arm=ARM_REFERENCE, scope="roi", roi_valid=1):
        reference[(row["geometry"], row["view"], row["anomaly_idx"])] = row

    classifications, per_cell = [], {}
    excluded, unmatched = 0, 0
    for row in roi_rows_all:
        if row.get("roi_valid") != 1:
            excluded += 1
            continue
        key = (row["geometry"], row["view"], row["anomaly_idx"])
        if key not in reference:
            unmatched += 1
            continue
        item = det.classify(row, reference[key], z_thr, x, y)
        classifications.append(item)
        per_cell.setdefault(cell_of(row), []).append(item)

    preserved, _num, denominator = det.flag_rate(classifications, det.PRESERVED)
    lost, _n, _d = det.flag_rate(classifications, det.LOST)
    inflated, _n, _d = det.flag_rate(classifications, det.INFLATED)

    sham_rows = select(rows, arm=arm, scope="sham", fraction=level)
    sham_labels = [det.classify_sham(row, z_thr) for row in sham_rows]
    hallucinated = (sum(1 for label in sham_labels if label == det.HALLUCINATED)
                    / len(sham_labels)) if sham_labels else float("nan")
    return {
        "preserved": preserved, "lost": lost, "inflated": inflated,
        "hallucinated": hallucinated,
        "denominator": denominator,
        "undetected_ref": sum(1 for item in classifications
                              if item[det.UNDETECTED_REF]),
        "excluded_roi": int(excluded),
        "unmatched_roi": int(unmatched),
        "n_sham": len(sham_labels),
        "per_cell": per_cell,
    }


def table_detectability(rows, thresholds, b=10_000, seed=20260805) -> str:
    header = ["Nivel", "Brazo", "Preservadas", "IC 95 %", "Perdidas",
              "Infladas", "Alucinación (sham)", "n ROI", "n sham",
              "Ref. no detectable", "ROI excluidas (<64 px)"]
    body = []
    excluded_total = 0
    for level in LEVELS:
        for arm in (ARM_RAW_CORRECTED, ARM_CLASSICAL, ARM_UNET_L1, ARM_CWGAN):
            rates = detectability_rates(rows, arm, level,
                                        z_thr_for(thresholds, arm, level),
                                        thresholds)
            excluded_total += rates["excluded_roi"]
            if rates["denominator"] == 0:
                continue
            per_cell = [np.array([1.0 if item[det.PRESERVED] else 0.0
                                  for item in items
                                  if not item[det.UNDETECTED_REF]])
                        for items in rates["per_cell"].values()]
            per_cell = [cell for cell in per_cell if cell.size]
            if per_cell:
                rng = np.random.default_rng(seed)
                boot = S.cluster_bootstrap(per_cell, b=b, rng=rng)
                interval = S.choose_ci(
                    boot, float(np.mean([c.mean() for c in per_cell])),
                    S.jackknife_over_cells(per_cell))
                ci = f"[{interval['lo']:.1%}, {interval['hi']:.1%}]"
            else:
                ci = "—"
            body.append([
                f"{level * 100:g} %", ARM_NAMES.get(arm, arm),
                f"{rates['preserved']:.1%}", ci, f"{rates['lost']:.1%}",
                f"{rates['inflated']:.1%}", f"{rates['hallucinated']:.1%}",
                str(rates["denominator"]), str(rates["n_sham"]),
                str(rates["undetected_ref"]), str(rates["excluded_roi"])])
    # the exclusions get their own row: a changed denominator that only shows
    # up as a smaller n is a changed denominator nobody notices
    body.append(["—", "**ROI excluidas (todas las filas)**", "—", "—", "—",
                 "—", "—", "—", "—", "—", f"**{excluded_total}**"])
    note = (f"\n\nz_thr calibrado por (brazo, nivel) — percentil 99 de |z| "
            f"sobre ROIs sham de validación de CADA brazo (escalar de "
            f"respaldo {thresholds['z_thr']}). «Ref. no detectable» sale del "
            f"denominador y se reporta aparte (válvula de la sección 5.3). "
            f"«ROI excluidas» son las ROI con `roi_valid = 0` (< 64 px o sin "
            f"anillo medible, sección 5.2): quedan fuera de toda tasa y se "
            f"cuentan aquí, nunca se descartan en silencio. Las tasas leen "
            f"banderas independientes, de modo que un caso inflado cuenta como "
            f"inflado aunque también esté dentro de la tolerancia X.")
    return markdown_table(header, body) + note


def table_generalization(rows, metric="ssim", b=10_000, seed=20260805) -> str:
    """Descriptive levels table for the generalization geometry (n = 6 cells).

    Protocol v1.1 section 5.4: geo-gen-01 is one geometry, six views, so its
    tests are underpowered by design and are NEVER folded into the primary Holm
    family. It is reported here as a descriptive companion -- the same per-level
    cell means and cluster-bootstrap intervals as the test table, with no
    Wilcoxon and no S1/S2/S3 verdict -- so a reader can read the direction of
    transfer to an unseen geometry without mistaking it for a confirmatory
    result.
    """
    if not rows:
        return ("_Sin filas de la geometría de generalización en la caché "
                "(columna `geometry_split` ausente o vacía): no aplica en esta "
                "corrida._")
    note = ("\n\n**Descriptivo, no confirmatorio.** Una sola geometría "
            "(geo-gen-01), seis vistas → n = 6 celdas: subpotenciado por "
            "diseño. Estas filas NO entran en la familia primaria ni en el "
            "Holm de la sección 5.4, y no se reporta Wilcoxon ni veredicto "
            "S1/S2/S3 sobre ellas. Es la dirección de la transferencia a una "
            "geometría no vista, no una prueba de ella.")
    return table_levels(rows, metric, b, seed) + note


def calibrate_z_threshold(rows, percentile: float = 99.0) -> dict:
    """z_thr from the sham ROIs measured on the REFERENCE images.

    This is the dry-run measurement that protocol v1.1 freezes as a number.
    Running it on the test split would be circular, so it is computed on the
    validation references and then never recomputed.
    """
    values = [abs(row["z"]) for row in select(rows, arm=ARM_REFERENCE,
                                              scope="sham")
              if row.get("z") is not None and np.isfinite(row["z"])]
    if not values:
        return {"z_thr": float("nan"), "n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"z_thr": float(np.percentile(array, percentile)),
            "percentile": percentile, "n": int(array.size),
            "max": float(array.max()), "median": float(np.median(array))}


# ---------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True,
                        help="cache root holding metrics.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--thresholds", default=str(THRESHOLDS_PATH))
    parser.add_argument("--metric", default="ssim", choices=("ssim", "psnr_db"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    metrics_path = Path(args.cache) / "metrics.csv"
    rows = load_metrics(metrics_path)
    thresholds = load_thresholds(args.thresholds)
    summary = load_summary(args.cache)
    # HARD ABORT before anything is written: on the test split, unfrozen or
    # missing thresholds are not a caveat, they are a broken pre-registration
    assert_thresholds_are_frozen(summary, thresholds, args.thresholds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Protocol v1.1 section 5.4: the primary family and the S1/S2/S3 verdicts
    # live on the test geometries alone (n = 18 cells). The generalization
    # geometry (geo-gen-01, n = 6) is descriptive and reported apart -- mixing
    # the two would build a 24-cell family the pre-registration never declared.
    test_rows, generalization_rows = split_by_geometry(rows)

    print(f"Stage 20 tables — {len(rows)} rows from {metrics_path} "
          f"(split {summary.get('split', 'unknown')}); "
          f"{len(test_rows)} test rows, {len(generalization_rows)} "
          f"generalization rows")
    if thresholds.get("provisional"):
        print("  WARNING: using PROVISIONAL thresholds — protocol v1.1 is not "
              "frozen yet; every verdict below is a dry-run rehearsal")

    calibration = calibrate_z_threshold(rows)
    print(f"  z_thr calibration from reference shams: "
          f"{calibration['z_thr']:.3f} (n = {calibration['n']})")

    fragments = {
        "01_levels": table_levels(test_rows, args.metric, args.bootstrap,
                                  args.seed),
        "02_primary": table_primary(test_rows, args.metric, args.bootstrap,
                                    args.seed),
        "03_verdicts": table_verdicts(test_rows, thresholds, args.metric,
                                      args.bootstrap, args.seed),
        "04_detectability": table_detectability(test_rows, thresholds,
                                                args.bootstrap, args.seed),
        "05_generalization": table_generalization(generalization_rows,
                                                  args.metric, args.bootstrap,
                                                  args.seed),
    }
    for name, text in fragments.items():
        path = out_dir / f"{name}.md"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"  wrote {path}")

    (out_dir / "z_thr_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8")
    print("STAGE 20 TABLES: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
