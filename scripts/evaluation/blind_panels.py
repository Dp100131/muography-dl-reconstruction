#!/usr/bin/env python3
"""Stage 21 — blind visual inspection: 60 anonymized panels and a sealed key.

Global SSIM can rise while local fidelity falls, and no metric in this thesis
is a human observer. Stage 21 therefore asks three people (both authors and
the co-director) to judge the panels WITHOUT knowing which method produced
each one, and only afterwards opens the key.

Design (execution plan section 2.3, item 7):

  60 panels   48 method panels + 12 reference anchors. The anchors are the
              30-day images: a rater who cannot see the anomaly in an anchor
              is telling us about the anomaly, not about the method, and that
              is exactly what the "not detectable at 100 %" valve needs.
  identical   every panel is rendered by the same function, same colour map,
              same limits, same size. A rater must not be able to recognize an
              arm by its rendering.
  hash names  the file name is a SHA-256 prefix of the panel identity plus a
              RANDOM 128-BIT SALT that exists only inside key_sealed.json. The
              identity space is tiny -- four arms x four levels x a handful of
              geometries, views and realizations -- so a hash over the identity
              alone is invertible by brute force in seconds against a checkout
              of this file. The salt is what makes the blinding a fact rather
              than an inconvenience, and it lives in the sealed key so that
              opening the key is still the only way to read the mapping.
  per rater   the presentation order is shuffled independently for each rater
              with SeedSequence([20260805e6 + 21e4 + 9e2 + rater]).
  sealed key  key_sealed.json maps every hashed name back to its panel. It is
              written once, before any rating exists, and is not opened until
              every rubric comes back.

Rubric (Spanish, three questions): is an anomaly visible and where, overall
quality 1-5, and are there artefacts. --reveal reads the returned CSVs and
computes Fleiss' kappa, sensitivity (a mark inside the ROI +- 15 IMAGE pixels,
after dividing the rater's coordinates by the display upscale), the false-alarm
rate and the pre-registered paired McNemar tests between arms.

Run:  python scripts/evaluation/blind_panels.py --cache results/evaluation/val --raters 3
      python scripts/evaluation/blind_panels.py --reveal --responses results/evaluation/blind/responses
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import cache as cache_module  # noqa: E402
from plotting_style import apply_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_BLIND_DIR = REPO_ROOT / "results" / "evaluation" / "blind"
MASTER_SEED = 20260805
STAGE = 21
PANEL_CONFIG = 9              # the config index of the blind panel stream
N_METHOD_PANELS = 48
N_ANCHOR_PANELS = 12
N_PANELS = N_METHOD_PANELS + N_ANCHOR_PANELS
DEFAULT_RATERS = 3
HIT_RADIUS_PX = 15            # in IMAGE pixels, not in displayed pixels
PANEL_UPSCALE = 3            # 256 px -> 768 px, nearest neighbour
IMAGE_SIDE_PX = 256
SALT_BITS = 128
# the pre-registered paired arm comparisons of the blind inspection
PAIRED_COMPARISONS = (("F", "D"), ("F", "E"), ("E", "D"))

RUBRIC_TEMPLATE = """# Inspección ciega — evaluador {rater}

Etapa 21 del trabajo de grado. Cada panel es una imagen muográfica de
{image_side}×{image_side} píxeles, mostrada ampliada a {display_side}×{display_side}
(ampliación entera ×{upscale}, sin interpolación), producida por un método que
usted NO conoce. Responda las tres preguntas para cada panel, en el orden en
que aparecen. Tiempo estimado: 30 minutos.

**No consulte con los demás evaluadores hasta entregar sus respuestas.**

## Preguntas (para cada panel)

1. **¿Distingue una anomalía?** `si` / `no`. Si responde `si`, indique la
   posición aproximada del centro en píxeles como `fila,columna` medidos sobre
   la imagen **tal como se muestra**, es decir en el rango
   0–{display_max} (origen abajo a la izquierda). El análisis convierte esas
   coordenadas a la grilla original dividiéndolas por {upscale}.
2. **Calidad general de la imagen**, en una escala de 1 a 5
   (1 = inservible, 3 = utilizable con reservas, 5 = excelente).
3. **¿Observa artefactos?** `si` / `no`. Si responde `si`, describa en pocas
   palabras (por ejemplo: «cuadriculado», «manchas», «bordes duros»).

## Cómo responder

Complete el archivo `respuestas_{rater}.csv` que acompaña a esta rúbrica. Una
fila por panel, con las columnas ya creadas:

    panel_id, anomalia, fila, columna, calidad, artefactos, comentario

- `anomalia`: `si` o `no`
- `fila`, `columna`: enteros; déjelos vacíos si respondió `no`
- `calidad`: entero de 1 a 5
- `artefactos`: `si` o `no`
- `comentario`: texto libre, opcional

## Orden de los paneles

{listing}

---

Los paneles están anonimizados y su orden es distinto para cada evaluador. La
llave que asocia cada panel con su método está sellada y no se abrirá hasta
recibir las respuestas de todos los evaluadores.
"""


def panel_seed(rater: int):
    """SeedSequence([20260805e6 + 21e4 + 9e2 + rater]) — one stream per rater."""
    base = (MASTER_SEED * 10 ** 6 + STAGE * 10 ** 4
            + PANEL_CONFIG * 10 ** 2 + int(rater))
    return np.random.SeedSequence([base])


def new_salt(bits: int = SALT_BITS) -> str:
    """A fresh random salt, hex encoded. Written ONLY into key_sealed.json."""
    return secrets.token_hex(bits // 8)


def panel_hash(identity: dict, rater: int, salt: str) -> str:
    """Stable, opaque file name: no substring of it leaks the arm.

    The salt is required. Without one the digest is a function of a handful of
    enumerable fields and anybody holding this source can rebuild the mapping
    from the file names alone, which would make the inspection blind in
    appearance only.
    """
    if not salt:
        raise ValueError("panel_hash needs a salt: an unsalted digest over the "
                         "panel identity is invertible by brute force")
    payload = json.dumps({**identity, "rater": int(rater), "salt": str(salt)},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render_panel(image: np.ndarray, path: Path) -> Path:
    """One panel, rendered identically for every arm. No title, no colour bar.

    plt.imsave writes the array straight through the colour map: one output
    pixel per image pixel, no axes, no resampling, no figure furniture that a
    rater could read as a hint. The nearest-neighbour upscale by an integer
    factor keeps the panel comfortable on screen without inventing detail --
    an interpolated enlargement would smooth the very noise under judgement.
    """
    array = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    if PANEL_UPSCALE > 1:
        array = np.repeat(np.repeat(array, PANEL_UPSCALE, axis=0),
                          PANEL_UPSCALE, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, array, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower",
               format="png")
    return path


def select_panels(catalog, n_method=N_METHOD_PANELS, n_anchor=N_ANCHOR_PANELS,
                  rng=None):
    """Balance the method panels across arms and levels, then add the anchors.

    catalog: [{'arm', 'level', 'geometry', 'view', 'k', 'image', 'roi'}].
    Entries whose arm is 'A' are the reference anchors.
    """
    rng = np.random.default_rng(20260805) if rng is None else rng
    methods = [item for item in catalog if item["arm"] != "A"]
    anchors = [item for item in catalog if item["arm"] == "A"]

    buckets = {}
    for item in methods:
        buckets.setdefault((item["arm"], item["level"]), []).append(item)
    chosen, keys = [], sorted(buckets)
    index = 0
    while len(chosen) < min(n_method, len(methods)):
        bucket = buckets[keys[index % len(keys)]]
        if bucket:
            chosen.append(bucket.pop(rng.integers(len(bucket))))
        index += 1
        if index > 10 * max(1, len(methods)):
            break
    picked_anchors = anchors[:n_anchor]
    if anchors and len(picked_anchors) < n_anchor:
        picked_anchors = [anchors[i % len(anchors)] for i in range(n_anchor)]
    return chosen + picked_anchors


def build(panels, out_dir: Path, raters: int, salt: str | None = None) -> dict:
    """Render every panel once per rater and write the sealed key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    salt = new_salt() if salt is None else salt
    display_side = IMAGE_SIDE_PX * PANEL_UPSCALE
    key = {"generated_utc": datetime.now(timezone.utc).isoformat(
        timespec="seconds"),
        "stage": "21 (blind inspection)",
        "n_panels": len(panels), "n_raters": int(raters),
        "hit_radius_px": HIT_RADIUS_PX,
        "panel_upscale": PANEL_UPSCALE,
        "image_side_px": IMAGE_SIDE_PX,
        "display_side_px": display_side,
        # SEALED WITH THE KEY: the salt is the blinding. It is generated here,
        # written nowhere else, and never printed.
        "panel_id_salt": salt,
        "panel_id_derivation": "sha256(identity + rater + salt)[:16]",
        "seed_derivation": "SeedSequence([20260805e6 + 21e4 + 9e2 + rater])",
        "paired_comparisons": [list(pair) for pair in PAIRED_COMPARISONS],
        "raters": {}}

    for rater in range(1, raters + 1):
        rng = np.random.default_rng(panel_seed(rater))
        order = rng.permutation(len(panels))
        rater_dir = out_dir / f"rater_{rater}"
        rater_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for position, panel_index in enumerate(order, start=1):
            panel = panels[int(panel_index)]
            identity = {"arm": panel["arm"], "level": panel["level"],
                        "geometry": panel["geometry"], "view": panel["view"],
                        "k": panel["k"]}
            panel_id = panel_hash(identity, rater, salt)
            render_panel(panel["image"], rater_dir / f"{panel_id}.png")
            entries.append({"position": position, "panel_id": panel_id,
                            **identity,
                            # ALL the anomalies of the view: a rater who marks
                            # the second anomaly of a three-anomaly geometry
                            # located something real
                            "roi_centroids": panel.get("roi_centroids") or []})
        key["raters"][str(rater)] = entries

        listing = "\n".join(f"{entry['position']:>3}. `{entry['panel_id']}.png`"
                            for entry in entries)
        (rater_dir / f"rubric_{rater}.md").write_text(
            RUBRIC_TEMPLATE.format(rater=rater, listing=listing,
                                   image_side=IMAGE_SIDE_PX,
                                   display_side=display_side,
                                   display_max=display_side - 1,
                                   upscale=PANEL_UPSCALE),
            encoding="utf-8")
        with open(rater_dir / f"respuestas_{rater}.csv", "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["panel_id", "anomalia", "fila", "columna",
                             "calidad", "artefactos", "comentario"])
            for entry in entries:
                writer.writerow([entry["panel_id"], "", "", "", "", "", ""])

    cache_module.write_json_atomic(out_dir / "key_sealed.json", key)
    return key


# ------------------------------------------------------------------- reveal
def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: (n_items, n_categories) of rater votes. NaN when undefined."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.size == 0:
        return float("nan")
    n_raters = counts.sum(axis=1)
    if not np.all(n_raters == n_raters[0]) or n_raters[0] < 2:
        return float("nan")
    n = float(n_raters[0])
    agreement = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1.0))
    p_bar = float(agreement.mean())
    proportions = counts.sum(axis=0) / (counts.shape[0] * n)
    p_expected = float((proportions ** 2).sum())
    if abs(1.0 - p_expected) < 1e-12:
        return float("nan")
    return (p_bar - p_expected) / (1.0 - p_expected)


def mcnemar(pairs) -> dict:
    """Exact McNemar over paired (a_detected, b_detected) booleans."""
    from scipy import stats as sp_stats

    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    if b + c == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    p = float(sp_stats.binomtest(b, b + c, 0.5).pvalue)
    return {"b": int(b), "c": int(c), "p_value": p}


def read_responses(directory: Path) -> dict:
    """{rater: {panel_id: response}} from the returned CSVs."""
    responses = {}
    for path in sorted(Path(directory).glob("respuestas_*.csv")):
        rater = path.stem.split("_")[-1]
        rows = {}
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("panel_id"):
                    rows[row["panel_id"]] = row
        responses[rater] = rows
    return responses


def mark_in_image_pixels(row, upscale: int):
    """(row, col) of a rater's mark, converted from DISPLAY to image pixels.

    The panels are shown at an integer upscale, so a mark read off the screen
    is in display coordinates. Comparing it directly with a centroid measured
    on the 256 x 256 grid would put every mark roughly three times too far from
    the anomaly and drive the localization rate towards zero -- the sensitivity
    of the blind inspection would be an artefact of the renderer.
    """
    try:
        return (float(row["fila"]) / float(upscale),
                float(row["columna"]) / float(upscale))
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return None


def nearest_centroid(mark, centroids):
    """(centroid, Chebyshev distance) of the closest anomaly of that view.

    Hit-testing against centroid 0 asks "did the rater find the FIRST anomaly
    of this geometry", which is not the question: a geometry can carry three,
    and marking any real one is a correct localization. With one anomaly the
    two rules coincide, which is exactly why the bug would survive a smoke test
    and then quietly under-count sensitivity on the multi-anomaly geometries.
    """
    best, best_distance = None, float("inf")
    for centroid in centroids or []:
        if centroid is None or len(centroid) != 2:
            continue
        if not all(np.isfinite(float(v)) for v in centroid):
            continue
        distance = max(abs(mark[0] - float(centroid[0])),
                       abs(mark[1] - float(centroid[1])))
        if distance < best_distance:
            best, best_distance = centroid, distance
    return best, best_distance


def paired_mcnemar(key: dict, responses: dict,
                   comparisons=PAIRED_COMPARISONS) -> dict:
    """The pre-registered paired arm comparisons, by exact McNemar.

    Pairing is on (rater, geometry, view, level, k): the same case shown to the
    same person under two methods. Unpaired panels are dropped and the surviving
    count travels, because a McNemar over four pairs says nothing and the reader
    has to be able to see that.
    """
    detections = {}
    for rater, entries in key["raters"].items():
        answers = responses.get(rater, {})
        for entry in entries:
            row = answers.get(entry["panel_id"])
            if row is None:
                continue
            case = (rater, entry["geometry"], entry["view"], entry["level"],
                    entry["k"])
            detections[(case, entry["arm"])] = said_yes(row)

    out = {}
    cases = sorted({case for case, _arm in detections}, key=str)
    for first, second in comparisons:
        pairs = [(detections[(case, first)], detections[(case, second)])
                 for case in cases
                 if (case, first) in detections and (case, second) in detections]
        if not pairs:
            continue
        out[f"{first}_vs_{second}"] = {**mcnemar(pairs), "n_pairs": len(pairs)}
    return out


def said_yes(row) -> bool:
    return str(row.get("anomalia", "")).strip().lower() in ("si", "sí", "1",
                                                            "yes", "y")


def reveal(key: dict, responses: dict) -> dict:
    """Sensitivity, false alarms, quality, Fleiss' kappa and McNemar."""
    upscale = int(key.get("panel_upscale", PANEL_UPSCALE))
    radius = float(key.get("hit_radius_px", HIT_RADIUS_PX))

    per_arm, votes, quality = {}, {}, {}
    for rater, entries in key["raters"].items():
        answers = responses.get(rater, {})
        for entry in entries:
            row = answers.get(entry["panel_id"])
            if row is None:
                continue
            detected = said_yes(row)
            arm = entry["arm"]
            bucket = per_arm.setdefault(
                (arm, entry["level"]),
                {"n": 0, "detected": 0, "hits": 0, "quality": [],
                 "artifacts": 0})
            bucket["n"] += 1
            bucket["detected"] += int(detected)
            bucket["artifacts"] += int(
                str(row.get("artefactos", "")).strip().lower()
                in ("si", "sí", "1", "yes", "y"))
            try:
                bucket["quality"].append(float(row["calidad"]))
            except (TypeError, ValueError, KeyError):
                pass
            centroids = entry.get("roi_centroids")
            if centroids is None and entry.get("roi_centroid"):
                centroids = [entry["roi_centroid"]]   # keys written before W7
            if detected and centroids:
                mark = mark_in_image_pixels(row, upscale)
                if mark is not None:
                    _centroid, distance = nearest_centroid(mark, centroids)
                    if distance <= radius:
                        bucket["hits"] += 1
            identity = (entry["arm"], entry["level"], entry["geometry"],
                        entry["view"], entry["k"])
            votes.setdefault(identity, []).append(detected)
            quality.setdefault(identity, []).append(row.get("calidad"))

    complete = [v for v in votes.values()
                if len(v) == len(key["raters"]) and len(key["raters"]) >= 2]
    counts = np.array([[sum(1 for d in v if d), sum(1 for d in v if not d)]
                       for v in complete]) if complete else np.zeros((0, 2))

    summary = {"per_arm": {}, "fleiss_kappa": fleiss_kappa(counts),
               "n_items_with_full_votes": int(len(complete)),
               "n_raters": len(key["raters"]), "hit_radius_px": radius,
               "panel_upscale": upscale,
               "hit_radius_units": "image pixels (marks divided by the upscale)",
               "localization_rule": "nearest anomaly centroid of the view",
               # the pre-registered paired arm comparisons; mcnemar existed and
               # was never called, so the design's paired test was not run
               "mcnemar": paired_mcnemar(
                   key, responses,
                   [tuple(pair) for pair in key.get("paired_comparisons",
                                                    PAIRED_COMPARISONS)])}
    for (arm, level), bucket in sorted(per_arm.items(), key=lambda kv: str(kv[0])):
        summary["per_arm"][f"{arm}@{level}"] = {
            "n_judgements": bucket["n"],
            "detection_rate": bucket["detected"] / bucket["n"]
            if bucket["n"] else float("nan"),
            "localization_rate": bucket["hits"] / bucket["n"]
            if bucket["n"] else float("nan"),
            "artifact_rate": bucket["artifacts"] / bucket["n"]
            if bucket["n"] else float("nan"),
            "mean_quality": float(np.mean(bucket["quality"]))
            if bucket["quality"] else float("nan"),
        }
    return summary


# ---------------------------------------------------------------------- main
PANEL_ARMS = ("A", "D", "E", "F")   # reference anchors + classical + L1 + cWGAN


def _centroid_lookup(geometry_dir: Path):
    """{view: [centroid per anomaly]} from the cached geometry masks."""
    path = geometry_dir / "masks.npz"
    if not path.exists():
        return {}
    masks = cache_module.load_masks(path)
    views = [int(v) for v in masks["views"]]
    centroids = np.asarray(masks["roi_centroids"], dtype=np.float64)
    return {view: centroids[row] for row, view in enumerate(views)}


def catalog_from_cache(cache_root: Path, limit_per_arm: int = 48):
    """Build the panel catalog from the Stage 20 cache — no shard is opened.

    The reference arm is cached too, so the 12 anchors come from the same
    place as the 48 method panels and are rendered by the same function.

    Entries are sampled at an EVEN STRIDE through each cached block rather
    than from its head: a block is ordered (view, fraction, k), so taking the
    first N would draw every candidate from view 0 at the highest level and
    the level balance of select_panels would have nothing to balance.
    """
    catalog = []
    for geometry_dir in sorted(Path(cache_root).iterdir()):
        if not geometry_dir.is_dir():
            continue
        centroids = _centroid_lookup(geometry_dir)
        for path in sorted(geometry_dir.glob("arm_*.npz")):
            arm = path.stem.split("_")[-1]
            if arm not in PANEL_ARMS:
                continue
            images, extra = cache_module.load_predictions(path)
            total = images.shape[0]
            take = min(limit_per_arm, total)
            picks = (np.unique(np.linspace(0, total - 1, take).astype(int))
                     if take else np.array([], dtype=int))
            for index in picks:
                view = int(extra["view"][index])
                per_view = centroids.get(view)
                # EVERY finite centroid of the view, not just the first: a
                # geometry can carry three anomalies and a rater who marks the
                # second one has localized a real anomaly
                view_centroids = [
                    [float(row[0]), float(row[1])]
                    for row in (per_view if per_view is not None else [])
                    if np.isfinite(row).all()]
                catalog.append({
                    "arm": arm,
                    "level": float(extra["fraction"][index]),
                    "geometry": geometry_dir.name,
                    "view": view,
                    "k": int(extra["k"][index]),
                    "image": images[index],
                    "roi_centroids": view_centroids,
                })
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=None,
                        help="Stage 20 cache root (build mode)")
    parser.add_argument("--out", default=str(DEFAULT_BLIND_DIR))
    parser.add_argument("--raters", type=int, default=DEFAULT_RATERS)
    parser.add_argument("--reveal", action="store_true")
    parser.add_argument("--responses", default=None,
                        help="directory holding respuestas_*.csv (reveal mode)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    apply_style()

    if args.reveal:
        key = json.loads((out_dir / "key_sealed.json").read_text(
            encoding="utf-8"))
        responses = read_responses(Path(args.responses or out_dir))
        summary = reveal(key, responses)
        cache_module.write_json_atomic(out_dir / "reveal_summary.json", summary)
        print(json.dumps(summary, indent=2))
        print(f"wrote {out_dir / 'reveal_summary.json'}")
        print("STAGE 21 BLIND REVEAL: PASSED")
        return 0

    if args.cache is None:
        print("ERROR: build mode needs --cache", file=sys.stderr)
        return 1
    catalog = catalog_from_cache(Path(args.cache))
    panels = select_panels(catalog)
    key = build(panels, out_dir, args.raters)
    arms = Counter(entry["arm"] for entry in key["raters"]["1"])
    print(f"Stage 21 blind panels — {key['n_panels']} panels x "
          f"{key['n_raters']} raters")
    print(f"  arms represented: {dict(sorted(arms.items()))}")
    print(f"  wrote {out_dir / 'key_sealed.json'} (SEALED — do not open until "
          f"every rubric is back)")
    print("STAGE 21 BLIND PANELS: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
