#!/usr/bin/env python3
"""Stage 19 driver tests — NO SHARD IS EVER OPENED HERE.

The driver's real acceptance test is the validation dry run
(`python scripts/evaluation/run_evaluation.py --split val`), which is a cluster job.
What is checked locally is the part that must be right before that run
happens: the access-log writer, the arm resolution, the seed derivations, the
refusal to proceed when the held-out guard was opened by somebody else, and
the row builders and mask packer -- those last two on a small SYNTHETIC
geometry cast through the real ray-caster, so the driver's inner loop is not
shipped unexercised.

Run:  python scripts/evaluation/test_run_evaluation.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))

import run_evaluation as RE  # noqa: E402


def test_access_log() -> bool:
    """Append-only: an open entry, then a close entry with every hash."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test_access_log.json"
        argv = ["scripts/evaluation/run_evaluation.py", "--split", "test"]
        first = RE.access_log_open(path, argv)
        RE.access_log_close(path, first, {"geo-test-01.npz": "a" * 64},
                            {"geo-test-01/arm_F.npz": "b" * 64},
                            {"row_counts": {"global": 3}})
        second = RE.access_log_open(path, argv + ["--again"])
        entries = json.loads(path.read_text(encoding="utf-8"))

        checks = {
            "the log is a list": isinstance(entries, list),
            "nothing is overwritten": len(entries) == 3,
            "the first entry is an open": entries[0]["event"] == "open",
            "the second entry is its close": (
                entries[1]["event"] == "close"
                and entries[1]["access_id"] == first["access_id"]),
            "a later run appends rather than replaces": (
                entries[2]["event"] == "open"
                and entries[2]["access_id"] != first["access_id"]
                and entries[2]["argv"][-1] == "--again"),
            "the open entry records the provenance": all(
                key in entries[0] for key in
                ("utc", "git_commit", "argv", "host", "slurm_job_id", "user")),
            "the git commit is resolved":
                entries[0]["git_commit"] not in ("", None),
            "the close entry carries the shard hashes":
                entries[1]["shards_read"] == {"geo-test-01.npz": "a" * 64},
            "the close entry carries the cache hashes":
                entries[1]["cache_written"] == {"geo-test-01/arm_F.npz": "b" * 64},
            "second is distinct from first":
                second["access_id"] != first["access_id"],
        }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  access log: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_guard_refusal() -> bool:
    """A pre-set guard must abort the run; a clean environment must not."""
    checks = {
        "a pre-set guard raises": _raises(RuntimeError,
                                          RE.assert_guard_not_preset, True),
        "the refusal names the variable":
            RE.ALLOW_TEST_ENV in _message(RuntimeError,
                                          RE.assert_guard_not_preset, True),
        "a clean environment passes": _does_not_raise(
            RE.assert_guard_not_preset, False),
        "this test process started clean": RE.GUARD_PRESET_AT_IMPORT is False,
        "val never touches the guard": _does_not_raise(
            lambda: RE.resolve_geometries("val")),
    }
    geometries, entry = RE.resolve_geometries("val")
    checks["val resolves the three validation geometries"] = (
        len(geometries) == 3 and all(s == "val" for _g, s in geometries))
    checks["val writes no access-log entry"] = entry is None
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  guard refusal: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_arm_resolution() -> bool:
    checks = {
        "the default is every arm": RE.resolve_arms(None, False) == list(RE.ALL_ARMS),
        "'all' is the default too": RE.resolve_arms(["all"], False)
                                    == list(RE.ALL_ARMS),
        "--skip-gpu drops exactly the GPU arms":
            RE.resolve_arms(None, True) == [a for a in RE.ALL_ARMS
                                            if a not in RE.GPU_ARMS],
        "a subset keeps the canonical order":
            RE.resolve_arms(["I", "D", "B"], False)
            == [RE.ARM_REFERENCE, "B", "D", "I"],
        "the reference arm is always present":
            RE.ARM_REFERENCE in RE.resolve_arms(["F"], False),
        "an unknown arm raises": _raises(ValueError, RE.resolve_arms, ["Z"], False),
        "--skip-gpu leaves no arm needing torch":
            not set(RE.resolve_arms(None, True)) & set(RE.GPU_ARMS),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  arm resolution: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_seed_derivation() -> bool:
    """The thinning seeds must equal the formula written in section 6 step 2."""
    expected_base = 20260805 * 10 ** 6 + 22 * 10 ** 4 + 3 * 10 ** 2 + 7
    sequence = RE.thinning_seed(3, 7, 2, 5)
    checks = {
        "the base seed matches the formula":
            sequence.entropy == [expected_base, 2, 5],
        "the base value is 20260805220307": expected_base == 20260805220307,
        "the same inputs reproduce the stream": bool(np.array_equal(
            np.random.default_rng(RE.thinning_seed(3, 7, 2, 5)).random(4),
            np.random.default_rng(RE.thinning_seed(3, 7, 2, 5)).random(4))),
        "a different k gives a different stream": not np.array_equal(
            np.random.default_rng(RE.thinning_seed(3, 7, 2, 5)).random(4),
            np.random.default_rng(RE.thinning_seed(3, 8, 2, 5)).random(4)),
        "a different view gives a different stream": not np.array_equal(
            np.random.default_rng(RE.thinning_seed(3, 7, 2, 5)).random(4),
            np.random.default_rng(RE.thinning_seed(3, 7, 2, 4)).random(4)),
        "the sham stream is separate from the thinning stream":
            RE.sham_seed(2, 5).entropy[0] != RE.thinning_seed(0, 0, 2, 5).entropy[0],
        "the sham base is the Stage 21 stream":
            RE.sham_seed(2, 5).entropy == [20260805 * 10 ** 6 + 21 * 10 ** 4, 2, 5],
        "the dense grid is the nine pre-registered fractions":
            RE.DENSE_FRACTIONS == (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50,
                                   0.75, 0.90),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  seed derivation: base(fi=3, k=7) = {expected_base} -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_frozen_configuration() -> bool:
    """The frozen tables must match the execution plan, character for character."""
    checks = {
        "the classical winners are the frozen four":
            RE.FROZEN_CLASSICAL == {0.50: "nlm_log_h0.5",
                                    0.10: "nlm_anscombe_h1.2",
                                    0.05: "nlm_anscombe_h1.2",
                                    0.01: "gaussian_s2.5"},
        "the classical arm runs 1/f corrected":
            RE.CLASSICAL_CONVENTION == "corrected",
        "the cWGAN arm uses the Stage 17 finals":
            set(RE.CWGAN_RUNS.values()) == {"e17-final-f50", "e17-final-f10",
                                            "e17-final-f5", "e17-final-f1"},
        "the U-Net arm uses the Stage 19.5 retrains":
            set(RE.UNET_L1_RUNS.values()) == {"e19-unetl1-f50", "e19-unetl1-f10",
                                              "e19-unetl1-f5", "e19-unetl1-f1"},
        "the shared-contract continuity twin is pre-registered":
            RE.UNET_L1_CONTINUITY_RUNS == (("e19-unetl1-f1-shared", 0.01),),
        "the U-Net roster carries five runs, two of them at 1 %": (
            len(RE.model_runs_for_arm(RE.ARM_UNET_L1)) == 5
            and sorted(run for run, fraction, _levels
                       in RE.model_runs_for_arm(RE.ARM_UNET_L1)
                       if fraction == 0.01)
            == ["e19-unetl1-f1", "e19-unetl1-f1-shared"]),
        "the cross-level matrix skips the diagonal": all(
            model_fraction not in levels
            for _run, model_fraction, levels
            in RE.model_runs_for_arm(RE.ARM_CROSS_LEVEL)),
        "the cross-level matrix has 12 off-diagonal cells":
            sum(len(levels) for _r, _f, levels
                in RE.model_runs_for_arm(RE.ARM_CROSS_LEVEL)) == 12,
        "the seed replicas run at f = 0.05": all(
            fraction == 0.05 for _r, fraction, _l
            in RE.model_runs_for_arm(RE.ARM_SEED_REPLICAS)),
        "no-model arms declare no generator":
            RE.model_runs_for_arm(RE.ARM_RAW_SHARED) == [],
        "the tidy schema carries the required columns": all(
            column in RE.CSV_COLUMNS for column in
            ("split", "arm", "model_run", "geometry", "view", "fraction", "k",
             "ssim", "psnr_db", "anomaly_idx", "C", "CNR", "z", "local_ssim")),
        "the archived e17 configs use ref_equivalent": all(
            RE.input_scaling_for(run) == "ref_equivalent"
            for run in RE.CWGAN_RUNS.values()),
        # a missing record used to default to ref_equivalent. That default is
        # exactly the Stage 18 contract mismatch waiting to happen again: it
        # produces a wrong image with a plausible SSIM and no error anywhere.
        "an unknown run RAISES instead of defaulting":
            _raises(FileNotFoundError, RE.input_scaling_for, "e99-not-a-run"),
        "the refusal names the run and the records directory": all(
            token in _message(FileNotFoundError, RE.input_scaling_for,
                              "e99-not-a-run")
            for token in ("e99-not-a-run", "records")),
        "fractions are matched with np.isclose, never ==": (
            RE.is_frozen_fraction(0.05 + 1e-15)
            and RE.is_frozen_fraction(float(np.float32(0.01)))
            and not RE.is_frozen_fraction(0.02)),
        "the frozen classical winner survives a float32 round trip":
            RE.frozen_classical_for(float(np.float32(0.05)))
            == "nlm_anscombe_h1.2",
        "an unfrozen level has no classical winner":
            _raises(KeyError, RE.frozen_classical_for, 0.02),
        "texture is measured on the raw 1/f and classical arms too":
            set(RE.TEXTURE_ARMS) == {RE.ARM_RAW_CORRECTED, RE.ARM_CLASSICAL,
                                     RE.ARM_UNET_L1, RE.ARM_CWGAN},
        "texture is measured at the two adjudicating levels":
            RE.TEXTURE_FRACTIONS == (0.05, 0.01),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  frozen configuration: {sum(checks.values())}/{len(checks)} "
          f"checks -> {'ok' if ok else 'FAIL'}")
    return ok


def test_row_builders_and_mask_packing() -> bool:
    """The row builders and the mask packer, on a synthetic geometry.

    No shard is opened: a small cone with one spherical cavity is cast through
    the real ray-caster, which is enough to exercise every writer the driver
    uses inside its view loop -- the part that a pure-logic test would leave
    completely uncovered.
    """
    import roi as roi_module
    from surrogate.geometry import Anomaly, VolcanoGeometry

    geo = VolcanoGeometry(
        geometry_id="synthetic-run", base_radius_m=200.0, height_m=380.0,
        rock_density=2.65,
        anomalies=[Anomaly("cavity", 0.25, {"cx_m": 0.0, "cy_m": 0.0,
                                            "cz_m": 190.0, "radius_m": 45.0})])
    masks = RE.build_view_masks(geo, geo.geometry_id, 0, 0, n_sham=4)

    rng = np.random.default_rng(5)
    x_ref = np.clip(0.6 + 0.05 * rng.standard_normal((256, 256)), 0.0, 1.0)
    image = np.clip(x_ref + 0.02 * rng.standard_normal((256, 256)), 0.0, 1.0)
    image[masks["rois"][0]] += 0.08

    context = {"split": "val", "arm": "F", "arm_label": "cWGAN-GP",
               "model_run": "e17-final-f5", "geometry": geo.geometry_id,
               "geometry_split": "val", "geo_index": 0, "view": 0,
               "fraction": 0.05, "model_fraction": 0.05,
               "convention": "ref_equivalent", "k": 3}

    g_row = RE.global_row(context, image, x_ref)
    r_rows = RE.roi_rows(context, image, x_ref, masks, geo)
    s_rows = RE.sham_rows(context, image, masks)
    t_row = RE.texture_row(context, image, x_ref, masks)
    o_rows = RE.octile_rows(context, image, masks, geo)
    ref_counts = np.random.default_rng(11).poisson(4000.0,
                                                   size=(256, 256)).astype(
                                                       np.uint32)
    masks["ref_counts"] = ref_counts
    packed = RE.pack_masks([(0, masks)], geo)

    # an ROI that cannot be measured must still emit a row, flagged
    blanked = dict(masks)
    blanked["rois"] = [np.zeros_like(masks["rois"][0])]
    blanked["roi_pixels"] = [0]
    blanked["roi_valid"] = [False]
    excluded = RE.roi_rows(context, image, x_ref, blanked, geo)

    checks = {
        "the ROI is big enough to measure":
            masks["roi_pixels"][0] >= roi_module.MIN_ROI_PIXELS,
        "the annulus reaches its floor":
            masks["annulus_info"][0]["n_pixels"] >= 200,
        "shams were sampled": masks["sham_info"]["obtained"] > 0,
        "the global row carries every schema column":
            set(g_row) == set(RE.CSV_COLUMNS),
        "the global row is scoped and indexed":
            g_row["scope"] == "global" and g_row["anomaly_idx"] == -1,
        "the global SSIM is a number": float(g_row["ssim"]) > 0.0,
        "one ROI row per anomaly": len(r_rows) == 1,
        "the ROI row detects the injected contrast": float(r_rows[0]["z"]) > 3.0,
        # SIGNED, like octiles.csv: a cavity in rock has a negative density
        # contrast and therefore a negative opacity perturbation
        "the ROI row records the SIGNED physical perturbation":
            float(r_rows[0]["delta_opacity_max"]) < 0.0
            and float(r_rows[0]["delta_rho"]) < 0.0,
        "the ROI row records the local SSIM":
            0.0 < float(r_rows[0]["local_ssim"]) <= 1.0,
        "a measurable ROI is flagged valid": r_rows[0]["roi_valid"] == 1,
        # W10: an excluded or empty ROI is REPORTED, never skipped
        "an empty ROI still emits a row": len(excluded) == 1,
        "the excluded row is flagged invalid": excluded[0]["roi_valid"] == 0,
        "the excluded row carries explicit NaN statistics": all(
            excluded[0][key] == "nan"
            for key in RE.STAT_COLUMNS + ("local_ssim",)),
        "the excluded row says why":
            "no measurable statistic" in excluded[0]["notes"]
            and "below 64 px" in excluded[0]["notes"],
        "the excluded row keeps the anomaly identity":
            excluded[0]["anomaly_kind"] == "cavity"
            and excluded[0]["anomaly_idx"] == 0,
        "one sham row per sham": len(s_rows) == masks["sham_info"]["obtained"],
        "sham rows are scoped": all(row["scope"] == "sham" for row in s_rows),
        "the texture row carries the three scalars": all(
            t_row[key] not in ("",) for key in
            ("variance_ratio", "nyquist_band_ratio", "residual_std")),
        "eight octile rows": len(o_rows) == 8,
        "octile rows use the octile schema":
            set(o_rows[0]) == set(RE.OCTILE_COLUMNS),
        "the effective octile count travels":
            all(row["n_octiles"] == len(o_rows) for row in o_rows),
        "octiles are ordered by path length": all(
            float(a["l_mean_m"]) < float(b["l_mean_m"])
            for a, b in zip(o_rows, o_rows[1:])),
        "the packed masks keep one row per view": packed["views"].tolist() == [0],
        "the packed ROIs have the anomaly axis":
            packed["rois"].shape == (1, 1, 256, 256),
        "path lengths are stored as float16":
            packed["l_anomalies"].dtype == np.float16,
        "the ROI centroid is finite": bool(
            np.isfinite(packed["roi_centroids"][0, 0]).all()),
        "the sham count travels even when the stack is ragged":
            packed["sham_counts"].tolist()
            == [masks["sham_info"]["obtained"]],
        "the density contrast travels with the masks":
            abs(float(packed["anomaly_delta_rho"][0]) - (0.25 - 2.65)) < 1e-9,
        "the anomaly kind travels with the masks":
            packed["anomaly_kinds"].tolist() == ["cavity"],
        "the anomaly kinds are pickle-free":
            packed["anomaly_kinds"].dtype != object,
        # T5: without the integer counts arm I is not regenerable and the
        # gallery has no input panel (arms B and C are not cached)
        "the integer reference counts travel with the masks":
            packed["ref_counts"].shape == (1, 256, 256)
            and packed["ref_counts"].dtype == np.uint32,
        "the counts survive the round trip":
            bool(np.array_equal(packed["ref_counts"][0], ref_counts)),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  row builders: ROI {masks['roi_pixels'][0]} px, annulus "
          f"{masks['annulus_info'][0]['n_pixels']} px, "
          f"{masks['sham_info']['obtained']} shams, z = {float(r_rows[0]['z']):.1f} "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


def _arm_i_metrics(ref_counts, geo_index, view, fractions, k_max):
    """The driver's arm-I inner loop, verbatim: seed -> thin -> preprocess -> metric."""
    import predict as pred

    ref_counts = np.asarray(ref_counts)
    ref64 = ref_counts.astype(np.int64)
    scale = pred.shared_scale(ref_counts)
    x_ref = np.log1p(ref_counts.astype(np.float32)) / np.float32(scale)
    rows = []
    for fraction_index, fraction in enumerate(fractions):
        for k in range(k_max):
            rng = np.random.default_rng(
                RE.thinning_seed(fraction_index, k, geo_index, view))
            low = rng.binomial(ref64, fraction)
            for convention in (RE.RAW, RE.CORRECTED):
                image = (RE.preprocess_pair(low, ref_counts)[0]
                         if convention == RE.RAW
                         else RE.preprocess_pair_corrected(low, ref_counts,
                                                           fraction)[0])
                rows.append((float(fraction), k, convention,
                             RE.compute_ssim(image, x_ref),
                             RE.compute_psnr(x_ref, image)))
    return rows


def test_arm_i_is_regenerable_from_the_cache() -> bool:
    """Arm I stores no image. The cache must still make it reproducible.

    The dense thinning grid is 9 fractions x K = 20 x 2 conventions per view,
    stored as metrics only on the explicit promise that the whole grid is
    regenerable bit for bit from its seeds. A seed alone does not regenerate
    anything: the binomial thins the INTEGER reference counts, which live in
    the shard, and after Stage 20 the shard may never be opened again. This
    test rebuilds the grid from masks.npz and asserts the metrics come back
    identical to the last bit -- if it fails, arm I is a promise the cache
    cannot keep and the Stage 22 curve has no lower half.
    """
    import cache as cache_module
    from surrogate.geometry import Anomaly, VolcanoGeometry

    fractions = (0.01, 0.10, 0.50)
    geo = VolcanoGeometry(
        geometry_id="synthetic-armi", base_radius_m=200.0, height_m=380.0,
        rock_density=2.65,
        anomalies=[Anomaly("cavity", 0.25, {"cx_m": 0.0, "cy_m": 0.0,
                                            "cz_m": 190.0, "radius_m": 45.0})])
    rng = np.random.default_rng(20260805)
    ref_counts = rng.poisson(3000.0, size=(64, 64)).astype(np.uint32)
    masks = {"l_rock": np.ones((64, 64), dtype=np.float32),
             "sky": np.zeros((64, 64), dtype=bool),
             "edge": np.zeros((64, 64), dtype=bool),
             "rois": [np.zeros((64, 64), dtype=bool)],
             "annuli": [np.zeros((64, 64), dtype=bool)],
             "l_anomalies": {0: np.zeros((64, 64), dtype=np.float32)},
             "roi_valid": [False], "shams": [],
             "ref_counts": ref_counts}

    original = _arm_i_metrics(ref_counts, 2, 3, fractions, k_max=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "masks.npz"
        cache_module.save_masks(path, **RE.pack_masks([(3, masks)], geo))
        restored = cache_module.load_masks(path)
        counts_back = restored["ref_counts"][
            list(restored["views"]).index(3)]
        regenerated = _arm_i_metrics(counts_back, 2, 3, fractions, k_max=2)

    identical = all(
        a[:3] == b[:3] and a[3] == b[3] and a[4] == b[4]
        for a, b in zip(original, regenerated))
    checks = {
        "the grid has one row per (fraction, k, convention)":
            len(original) == len(fractions) * 2 * 2,
        "the counts survive the cache without loss":
            bool(np.array_equal(counts_back, ref_counts)),
        "the regenerated grid is the same size":
            len(regenerated) == len(original),
        "every regenerated metric is bit-identical": identical,
        "the metrics are not all trivially equal to each other":
            len({row[3] for row in original}) == len(original),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    if not identical:
        for a, b in zip(original, regenerated):
            if a != b:
                print(f"    f={a[0]} k={a[1]} {a[2]}: {a[3]!r} vs {b[3]!r}")
    print(f"  arm I regeneration: {len(original)} rows, bit-identical from "
          f"ref_counts + seeds -> {'ok' if ok else 'FAIL'}")
    return ok


def test_resume_protocol() -> bool:
    """The crash protocol: what is kept, what is redone, what is logged."""
    import cache as cache_module

    arms = [RE.ARM_REFERENCE, RE.ARM_CLASSICAL]
    geometries = [("geo-test-01", "test"), ("geo-test-02", "test"),
                  ("geo-test-03", "test")]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "cache"
        manifest = root / cache_module.MANIFEST_NAME
        metrics = root / "metrics.csv"
        octiles = root / "octiles.csv"

        # geo-test-01: cache AND rows -> complete
        # geo-test-02: rows but the cache never got written (the crash)
        # geo-test-03: neither
        for arm in ("A", "D"):
            cache_module.save_predictions(
                root / "geo-test-01" / f"arm_{arm}.npz",
                np.zeros((2, 8, 8), dtype=np.float32),
                view=np.array([0, 0]), fraction=np.array([1.0, 0.05]),
                k=np.array([-1, 0]), model_run=np.array(["", "g"], dtype="U32"))
            cache_module.manifest_update(
                manifest, root / "geo-test-01" / f"arm_{arm}.npz", root)
        cache_module.save_masks(root / "geo-test-01" / "masks.npz",
                                views=np.array([0], dtype=np.int32))
        cache_module.manifest_update(manifest,
                                     root / "geo-test-01" / "masks.npz", root)

        for path, columns in ((metrics, RE.CSV_COLUMNS),
                              (octiles, RE.OCTILE_COLUMNS)):
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(columns))
                writer.writeheader()
                for geometry in ("geo-test-01", "geo-test-02"):
                    for index in range(3):
                        row = {column: "" for column in columns}
                        row.update({"geometry": geometry, "arm": "A",
                                    "view": index})
                        writer.writerow(row)

        done = RE.completed_geometries(root, manifest, metrics, geometries, arms)
        partial = RE.partial_cache_hashes(root)
        kept_metrics = RE.prune_tables(metrics, RE.CSV_COLUMNS, done)
        kept_octiles = RE.prune_tables(octiles, RE.OCTILE_COLUMNS, done)
        with open(metrics, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            surviving = {row["geometry"] for row in reader}
            header_ok = reader.fieldnames == list(RE.CSV_COLUMNS)

        # a half-written npz must NOT count as a completed geometry
        (root / "geo-test-01" / "arm_D.npz").write_bytes(b"truncated")
        after_corruption = RE.completed_geometries(root, manifest, metrics,
                                                   geometries, arms)

        log = Path(tmp) / "access.json"
        RE.access_log_open(log, ["run_evaluation.py", "--split", "test"])
        entry = RE.access_log_restart(
            log, ["run_evaluation.py", "--split", "test", "--resume"],
            "SLURM time limit hit on geo-test-02", partial)
        entries = json.loads(log.read_text(encoding="utf-8"))

    checks = {
        "only the geometry with cache AND rows is complete":
            done == ["geo-test-01"],
        "the partial cache is hashed file by file":
            len(partial) >= 4 and all(len(v) == 64 for v in partial.values()),
        "pruning keeps only the completed geometry":
            surviving == {"geo-test-01"} and kept_metrics == 3,
        "pruning keeps the octile table in step": kept_octiles == 3,
        "pruning rewrites a full header": header_ok,
        "a corrupted cache file un-completes its geometry":
            after_corruption == [],
        "the restart is an event of its own, not a second open":
            entries[1]["event"] == "restart" and entries[0]["event"] == "open",
        "the restart records the reason verbatim":
            entries[1]["reason"] == "SLURM time limit hit on geo-test-02",
        "the restart carries the sha256 of the partial cache":
            entries[1]["partial_cache"] == dict(sorted(partial.items()))
            and entries[1]["n_partial_files"] == len(partial),
        "the restart keeps the provenance an open would have": all(
            key in entry for key in ("utc", "git_commit", "argv", "host",
                                     "slurm_job_id", "user", "access_id")),
        "the log is still append-only": len(entries) == 2,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  resume protocol: {sum(checks.values())}/{len(checks)} checks "
          f"(kept {done}) -> {'ok' if ok else 'FAIL'}")
    return ok


def _raises(exception, fn, *args) -> bool:
    try:
        fn(*args)
    except exception:
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    unexpected {type(exc).__name__}: {exc}")
        return False
    return False


def _does_not_raise(fn, *args) -> bool:
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        print(f"    unexpected {type(exc).__name__}: {exc}")
        return False
    return True


def _message(exception, fn, *args) -> str:
    try:
        fn(*args)
    except exception as exc:
        return str(exc)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def main() -> int:
    print("Stage 19 driver tests (no shard opened; the validation dry run is "
          "the real acceptance test)")
    results = {
        "access_log": test_access_log(),
        "guard_refusal": test_guard_refusal(),
        "arm_resolution": test_arm_resolution(),
        "seed_derivation": test_seed_derivation(),
        "frozen_configuration": test_frozen_configuration(),
        "row_builders": test_row_builders_and_mask_packing(),
        "arm_i_regeneration": test_arm_i_is_regenerable_from_the_cache(),
        "resume_protocol": test_resume_protocol(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 RUN EVALUATION:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
