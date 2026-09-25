#!/usr/bin/env python3
"""Post-hoc diagnosis of the pre-registered z detector (exploratory, read-only).

Answers one question with numbers: is the S3 failure in the MODEL or in the
DETECTOR? If the reference arm (A, full statistics, no model involved) cannot
be separated from sham regions by |z|, the instrument is the problem.

Reads only the frozen test cache produced by the single test pass; never
touches the test set itself. Output goes to stdout and to
results/evaluation/fragments/06_detector_diagnostic.txt.

Usage:
    python scripts/evaluation/diagnose_detector.py
"""

from __future__ import annotations

import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "results/evaluation/test/metrics.csv"
OUT = REPO / "results/evaluation/fragments/06_detector_diagnostic.txt"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC: P(score of a real ROI > score of a sham ROI).

    0.5 = the statistic cannot tell them apart; 1.0 = perfect separation.
    """
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def report() -> None:
    df = pd.read_csv(CACHE, low_memory=False)

    print("=" * 78)
    print("ESTRUCTURA DEL CACHE")
    print("=" * 78)
    print(f"filas totales: {len(df):,}")
    print("\nfilas por scope:")
    print(df["scope"].value_counts().to_string())
    print("\nbrazos:")
    print(df[["arm", "arm_label"]].drop_duplicates().to_string(index=False))

    roi = df[(df["scope"] == "roi") & df["z"].notna()].copy()
    sham = df[(df["scope"] == "sham") & df["z"].notna()].copy()
    for d in (roi, sham):
        d["absz"] = d["z"].abs()
        d["abscnr"] = d["CNR"].abs()
        d["absC"] = d["C"].abs()

    print()
    print("=" * 78)
    print("PRUEBA DECISIVA — EL BRAZO A ES LA REFERENCIA DE 30 DIAS (SIN MODELO)")
    print("=" * 78)
    ra = roi[roi["arm"] == "A"]
    sa = sham[sham["arm"] == "A"]
    if len(ra) and len(sa):
        thr = np.percentile(sa["absz"], 99)
        print(f"anomalias reales   n={len(ra):5d}  |z| mediana={ra['absz'].median():7.2f}  max={ra['absz'].max():7.2f}")
        print(f"regiones falsas    n={len(sa):5d}  |z| mediana={sa['absz'].median():7.2f}  max={sa['absz'].max():7.2f}")
        print(f"umbral = percentil 99 de las falsas: {thr:.2f}")
        print(f"anomalias reales que lo superan: {(ra['absz'] > thr).mean() * 100:.1f}%")
        print(f"AUC con |z|: {auc(ra['absz'].to_numpy(), sa['absz'].to_numpy()):.3f}")

    print()
    print("=" * 78)
    print("COMPARACION DE ESTADISTICOS — AUC (0.5 = no distingue, 1.0 = perfecto)")
    print("=" * 78)
    print(f"{'brazo':<6}{'f':>7}{'n_roi':>7}{'n_sham':>8}{'AUC |z|':>10}{'AUC |CNR|':>11}{'AUC |C|':>10}")
    for arm in sorted(roi["arm"].unique()):
        for f in sorted(roi[roi["arm"] == arm]["fraction"].unique()):
            r = roi[(roi["arm"] == arm) & (roi["fraction"] == f)]
            s = sham[(sham["arm"] == arm) & (sham["fraction"] == f)]
            if len(r) < 5 or len(s) < 5:
                continue
            print(
                f"{arm:<6}{f:>7.2f}{len(r):>7d}{len(s):>8d}"
                f"{auc(r['absz'].to_numpy(), s['absz'].to_numpy()):>10.3f}"
                f"{auc(r['abscnr'].to_numpy(), s['abscnr'].to_numpy()):>11.3f}"
                f"{auc(r['absC'].to_numpy(), s['absC'].to_numpy()):>10.3f}"
            )

    print()
    print("=" * 78)
    print("POR QUE FALLA — EL RUIDO DEL FONDO NO ES ESTADISTICO")
    print("=" * 78)
    a = df[(df["arm"] == "A") & (df["scope"] == "sham") & df["std_bg"].notna()]
    if len(a):
        ratio = (a["std_bg"] / a["mean_bg"]).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"dispersion relativa del anillo de fondo (referencia): mediana={ratio.median():.4f}")
        print("si el fondo fuera plano, esa dispersion seria solo ruido de conteo;")
        print("aqui incluye el gradiente de espesor de la montana, que es deterministico.")
        print()
        print(f"n efectivo asumido por el estadistico z (pixeles del anillo): mediana={a['n_bg'].median():.0f}")
        print("el z divide por la raiz de ese n, asi que asume miles de medidas independientes;")
        print("los pixeles vecinos estan correlacionados, por lo que el n real es mucho menor")
        print("y el z queda inflado tanto para anomalias reales como para regiones falsas.")

    print()
    print("=" * 78)
    print("POR TIPO DE ANOMALIA (brazo A, referencia)")
    print("=" * 78)
    if len(ra):
        g = ra.groupby("anomaly_kind")["absz"].agg(["count", "median", "max"])
        print(g.to_string())


def main() -> int:
    if not CACHE.exists():
        print(f"cache not found: {CACHE}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh, redirect_stdout(fh):
        print(f"source: {CACHE.relative_to(REPO).as_posix()} (frozen test cache, read-only)")
        report()
    sys.stdout.write(OUT.read_text(encoding="utf-8"))
    print()
    print(f"wrote {OUT.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
