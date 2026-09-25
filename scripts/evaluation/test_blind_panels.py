#!/usr/bin/env python3
"""Stage 19 blind-panel tests — synthetic 4-image cache in, 60 panels out.

The properties that make the inspection blind are exactly the ones asserted
here: the file names leak nothing about the arm, the order differs per rater
and is reproducible from the seed, the key is complete, and the reveal
arithmetic (Fleiss' kappa, sensitivity, false alarms) is right on hand-built
answers.

Run:  python scripts/evaluation/test_blind_panels.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import blind_panels as BP  # noqa: E402
import cache as C  # noqa: E402

ARMS = ("D", "E", "F")
LEVELS = (0.50, 0.10, 0.05, 0.01)


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def _raises(exception, fn, *args) -> bool:
    try:
        fn(*args)
    except exception:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _synthetic_cache(root: Path, n_views=6):
    """Two geometries with a cached reference arm, three method arms and masks."""
    rng = np.random.default_rng(20260805)
    for geometry in ("geo-test-01", "geo-test-02"):
        directory = root / geometry
        views, fractions, ks, images = [], [], [], []
        for view in range(n_views):
            views.append(view)
            fractions.append(1.0)
            ks.append(-1)
            images.append(rng.random((32, 32)).astype(np.float32))
        C.save_predictions(directory / "arm_A.npz", np.stack(images),
                           view=np.array(views), fraction=np.array(fractions),
                           k=np.array(ks))
        for arm in ARMS:
            views, fractions, ks, images = [], [], [], []
            for view in range(n_views):
                for level in LEVELS:
                    for k in range(2):
                        views.append(view)
                        fractions.append(level)
                        ks.append(k)
                        images.append(rng.random((32, 32)).astype(np.float32))
            C.save_predictions(directory / f"arm_{arm}.npz", np.stack(images),
                               view=np.array(views),
                               fraction=np.array(fractions), k=np.array(ks))
        centroids = np.zeros((n_views, 1, 2))
        centroids[:, 0, 0] = 16.0
        centroids[:, 0, 1] = 16.0
        C.save_masks(directory / "masks.npz",
                     views=np.arange(n_views, dtype=np.int32),
                     roi_centroids=centroids)
    return root


def test_build() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = _synthetic_cache(Path(tmp) / "cache")
        catalog = BP.catalog_from_cache(cache_root)
        panels = BP.select_panels(catalog)
        out = Path(tmp) / "blind"
        key = BP.build(panels, out, raters=3)

        rater_dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
        pngs = {name: sorted(p.stem for p in (out / name).glob("*.png"))
                for name in rater_dirs}
        first = key["raters"]["1"]
        arms = {entry["arm"] for entry in first}
        anchors = [entry for entry in first if entry["arm"] == "A"]

        checks = {
            "60 panels": len(panels) == BP.N_PANELS,
            "48 method panels": len(panels) - len(anchors) == BP.N_METHOD_PANELS,
            "12 reference anchors": len(anchors) == BP.N_ANCHOR_PANELS,
            "three rater folders": rater_dirs == ["rater_1", "rater_2",
                                                  "rater_3"],
            "60 PNGs per rater": all(len(v) == BP.N_PANELS
                                     for v in pngs.values()),
            "the key covers every rater": set(key["raters"]) == {"1", "2", "3"},
            "the key is complete": all(len(entries) == BP.N_PANELS
                                       for entries in key["raters"].values()),
            "every arm is represented": arms >= {"A", "D", "E", "F"},
            "each rubric exists": all(
                (out / name / f"rubric_{name.split('_')[1]}.md").exists()
                for name in rater_dirs),
            "each response template exists": all(
                (out / name / f"respuestas_{name.split('_')[1]}.csv").exists()
                for name in rater_dirs),
        }

        # no file name may leak the arm, the level, the geometry or the k
        leak_tokens = ["arm", "cwgan", "unet", "classical", "referen",
                       "geo-test", "0.05", "0.5", "_A", "_D", "_E", "_F"]
        names = [name for values in pngs.values() for name in values]
        checks["file names leak nothing"] = all(
            token.lower() not in name.lower()
            for name in names for token in leak_tokens)
        checks["file names are 16 hex characters"] = all(
            len(name) == 16 and all(c in "0123456789abcdef" for c in name)
            for name in names)

        # the presentation order must differ per rater and be reproducible
        order_1 = [e["panel_id"] for e in key["raters"]["1"]]
        order_2 = [e["panel_id"] for e in key["raters"]["2"]]
        identity_1 = [(e["arm"], e["level"], e["geometry"], e["view"], e["k"])
                      for e in key["raters"]["1"]]
        identity_2 = [(e["arm"], e["level"], e["geometry"], e["view"], e["k"])
                      for e in key["raters"]["2"]]
        checks["orders differ between raters"] = identity_1 != identity_2
        checks["ids differ between raters"] = set(order_1) & set(order_2) == set()
        checks["the same seed reproduces the order"] = bool(np.array_equal(
            np.random.default_rng(BP.panel_seed(1)).permutation(len(panels)),
            np.random.default_rng(BP.panel_seed(1)).permutation(len(panels))))
        checks["the seed formula matches the plan"] = (
            BP.panel_seed(2).entropy == [20260805 * 10 ** 6 + 21 * 10 ** 4
                                         + 9 * 10 ** 2 + 2])

        # the rubric must carry the three Spanish questions
        rubric = (out / "rater_1" / "rubric_1.md").read_text(encoding="utf-8")
        checks["the rubric asks the three questions"] = all(
            token in rubric for token in ("¿Distingue una anomalía?",
                                          "Calidad general",
                                          "¿Observa artefactos?"))
        checks["the rubric lists every panel"] = rubric.count(".png") >= BP.N_PANELS
        # (a) the rubric must state the pixel size a mark is measured in
        display = BP.IMAGE_SIDE_PX * BP.PANEL_UPSCALE
        checks["the rubric states the panel pixel size"] = all(
            token in rubric for token in
            (f"{BP.IMAGE_SIDE_PX}×{BP.IMAGE_SIDE_PX}",
             f"{display}×{display}", f"×{BP.PANEL_UPSCALE}",
             f"0–{display - 1}"))
        sealed = json.loads((out / "key_sealed.json").read_text(
            encoding="utf-8"))
        checks["the sealed key is valid JSON"] = isinstance(sealed, dict)
        checks["the key records the hit radius"] = key["hit_radius_px"] == 15
        checks["the key records the upscale and the panel size"] = (
            sealed["panel_upscale"] == BP.PANEL_UPSCALE
            and sealed["display_side_px"] == display)

        # (b) a random 128-bit salt, stored ONLY in the sealed key
        salt = sealed["panel_id_salt"]
        checks["the key carries a 128-bit salt"] = (
            isinstance(salt, str) and len(salt) == 32
            and all(c in "0123456789abcdef" for c in salt))
        checks["the ids are reproducible FROM the salt"] = all(
            BP.panel_hash({k: entry[k] for k in
                           ("arm", "level", "geometry", "view", "k")},
                          1, salt) == entry["panel_id"]
            for entry in sealed["raters"]["1"])
        # without the salt the mapping cannot be rebuilt from this source
        checks["the ids are NOT reproducible without the salt"] = all(
            BP.panel_hash({k: entry[k] for k in
                           ("arm", "level", "geometry", "view", "k")},
                          1, "0" * 32) != entry["panel_id"]
            for entry in sealed["raters"]["1"])
        checks["an unsalted hash is refused outright"] = _raises(
            ValueError, BP.panel_hash, {"arm": "F"}, 1, "")
        checks["two builds draw different salts"] = (
            BP.new_salt() != BP.new_salt())
        # the salt must not leak into any rubric or response template
        leaked = [path for path in out.rglob("*")
                  if path.is_file() and path.name != "key_sealed.json"
                  and salt in _safe_read(path)]
        checks["the salt appears nowhere but the sealed key"] = not leaked

        # (c) every anomaly centroid of the view travels with the panel
        checks["all the view's centroids travel with the key"] = all(
            isinstance(entry.get("roi_centroids"), list)
            for entry in sealed["raters"]["1"])

        # the stride sampling exists so the 48 method panels cover every arm
        # and every level; taking the head of each cached block would put them
        # all on view 0 at the highest fraction
        method_levels = Counter(entry["level"] for entry in first
                                if entry["arm"] != "A")
        method_arms = Counter(entry["arm"] for entry in first
                              if entry["arm"] != "A")
        checks["method panels cover every level"] = (
            set(method_levels) == set(LEVELS)
            and max(method_levels.values()) - min(method_levels.values()) <= 4)
        checks["method panels are balanced across arms"] = (
            set(method_arms) == set(ARMS)
            and max(method_arms.values()) - min(method_arms.values()) <= 2)
        checks["the anchors are the reference arm"] = all(
            entry["level"] == 1.0 for entry in anchors)

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  build: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_reveal_arithmetic() -> bool:
    """Fleiss' kappa, sensitivity and false alarms on hand-built answers."""
    perfect = np.array([[3, 0], [3, 0], [0, 3], [0, 3]], dtype=float)
    chance = np.array([[2, 1], [1, 2], [2, 1], [1, 2]], dtype=float)
    checks = {
        "perfect agreement gives kappa = 1":
            abs(BP.fleiss_kappa(perfect) - 1.0) < 1e-12,
        "partial agreement is below 1": BP.fleiss_kappa(chance) < 0.5,
        "unequal rater counts are undefined":
            not np.isfinite(BP.fleiss_kappa(np.array([[3, 0], [2, 0]],
                                                     dtype=float))),
        "an empty table is undefined":
            not np.isfinite(BP.fleiss_kappa(np.zeros((0, 2)))),
        "McNemar counts the discordant pairs":
            BP.mcnemar([(True, False)] * 8 + [(False, True)] * 1)["p_value"]
            < 0.05,
        "McNemar with no discordance is p = 1":
            BP.mcnemar([(True, True)] * 5)["p_value"] == 1.0,
    }

    # the panels are rendered at an integer upscale, so a rater's mark is in
    # DISPLAY pixels: the centroid at image (100, 100) sits at display
    # (300, 300), and a mark at (312, 291) is 4 / 3 image pixels away
    upscale = BP.PANEL_UPSCALE
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        key = {"raters": {"1": [], "2": []}, "hit_radius_px": 15,
               "panel_upscale": upscale,
               "paired_comparisons": [["F", "D"]]}
        for rater in ("1", "2"):
            rows = []
            for index, arm in enumerate(("F", "F", "D", "D")):
                panel_id = f"{rater}{index:015d}"
                key["raters"][rater].append({
                    "position": index + 1, "panel_id": panel_id, "arm": arm,
                    "level": 0.05, "geometry": "geo-test-01",
                    "view": index % 2, "k": 0,
                    # two anomalies: the rater marks the SECOND one
                    "roi_centroids": [[20.0, 20.0], [100.0, 100.0]]})
                # cWGAN: detected and correctly localized; classical: missed
                detected = arm == "F"
                rows.append({"panel_id": panel_id,
                             "anomalia": "si" if detected else "no",
                             "fila": str(104 * upscale) if detected else "",
                             "columna": str(97 * upscale) if detected else "",
                             "calidad": "5" if detected else "2",
                             "artefactos": "no", "comentario": ""})
            with open(out / f"respuestas_{rater}.csv", "w", newline="",
                      encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["panel_id", "anomalia", "fila",
                                        "columna", "calidad", "artefactos",
                                        "comentario"])
                writer.writeheader()
                writer.writerows(rows)
        summary = BP.reveal(key, BP.read_responses(out))

    per_arm = summary["per_arm"]
    checks["the cWGAN arm is detected every time"] = (
        per_arm["F@0.05"]["detection_rate"] == 1.0)
    checks["the classical arm is never detected"] = (
        per_arm["D@0.05"]["detection_rate"] == 0.0)
    checks["a mark within 15 image px counts as localized"] = (
        per_arm["F@0.05"]["localization_rate"] == 1.0)
    checks["quality is averaged"] = per_arm["F@0.05"]["mean_quality"] == 5.0
    checks["kappa is computed over complete items"] = (
        summary["n_items_with_full_votes"] == 4
        and abs(summary["fleiss_kappa"] - 1.0) < 1e-12)

    # (a) the upscale conversion: without it a mark on the anomaly reads as
    # ~200 px away and the localization rate collapses to zero
    checks["display coordinates are divided by the upscale"] = (
        BP.mark_in_image_pixels({"fila": "312", "columna": "291"}, 3)
        == (104.0, 97.0))
    checks["an unparseable mark is None, not a crash"] = (
        BP.mark_in_image_pixels({"fila": "", "columna": ""}, 3) is None)
    checks["the summary states the units and the upscale"] = (
        summary["panel_upscale"] == upscale
        and "image pixels" in summary["hit_radius_units"])

    # (c) nearest centroid, not centroid 0
    centroids = [[20.0, 20.0], [100.0, 100.0]]
    _near, distance = BP.nearest_centroid((104.0, 97.0), centroids)
    _far, far_distance = BP.nearest_centroid((160.0, 160.0), centroids)
    checks["the hit test uses the NEAREST anomaly"] = abs(distance - 4.0) < 1e-9
    checks["a mark on no anomaly is far from all of them"] = far_distance > 15
    checks["a view with no finite centroid is handled"] = (
        BP.nearest_centroid((10.0, 10.0), [[float("nan"), 1.0]])[0] is None)

    # (d) the pre-registered paired comparison actually runs
    mcnemar = summary["mcnemar"]
    checks["the paired McNemar test is computed"] = "F_vs_D" in mcnemar
    checks["it pairs the same case under two arms"] = (
        mcnemar["F_vs_D"]["n_pairs"] == 4)
    checks["it counts the discordant pairs the right way round"] = (
        mcnemar["F_vs_D"]["b"] == 4 and mcnemar["F_vs_D"]["c"] == 0)
    checks["the McNemar p-value is reported"] = (
        0.0 < mcnemar["F_vs_D"]["p_value"] <= 1.0)

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  reveal: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19 blind-panel tests")
    results = {
        "build": test_build(),
        "reveal": test_reveal_arithmetic(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 BLIND PANELS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
