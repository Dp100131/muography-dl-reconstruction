#!/usr/bin/env python3
"""Stage 19 inference tests — hash gate and classical dispatch, no torch needed.

The generator path is exercised as far as a machine without PyTorch allows:
the manifest lookup, the path re-rooting, the size check and the SHA-256
refusal all run against a tiny fake checkpoint and a fake manifest in a
temporary directory. The torch-dependent tail (build, load_state_dict, eval)
is covered on the cluster by the dry run, which reproduces the frozen
validation SSIM of e17-final-f5.

Run:  python scripts/evaluation/test_predict.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

import predict as P  # noqa: E402

try:
    import pywt  # noqa: F401
    HAVE_PYWT = True
except ImportError:
    HAVE_PYWT = False


def _fake_run(root: Path, run_id: str, payload: bytes):
    """A fake checkpoint plus the manifest entry that describes it truthfully."""
    checkpoint = root / "runs" / run_id / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(payload)
    return {
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "sha256": P.sha256_of(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "config_sha8": "deadbeef",
        "best_val_ssim": 0.7834,
    }


def test_hash_gate() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        entry = _fake_run(root, "e17-fake-f5", b"weights-v1" * 100)
        manifest_path = root / "weights-manifest.json"
        manifest_path.write_text(json.dumps(
            {"runs_root": str(root / "runs"), "runs": [entry]}, indent=2),
            encoding="utf-8")

        good = P.verify_checkpoint("e17-fake-f5", manifest_path)
        checks = {
            "clean checkpoint verifies": good["sha256"] == entry["sha256"],
            "unknown run raises KeyError": _raises(
                KeyError, P.manifest_entry, "e17-nope", manifest_path),
        }

        # corrupt the payload but keep the length: only the hash can catch it
        path = Path(entry["checkpoint"])
        data = bytearray(path.read_bytes())
        data[7] = (data[7] + 1) % 256
        path.write_bytes(bytes(data))
        checks["a same-length corruption is refused"] = _raises(
            RuntimeError, P.verify_checkpoint, "e17-fake-f5", manifest_path)
        message = _message(RuntimeError, P.verify_checkpoint, "e17-fake-f5",
                           manifest_path)
        checks["the refusal names sha256"] = "sha256" in message

        # a truncated file is caught by the cheaper size check first
        path.write_bytes(b"short")
        checks["a truncated file is refused"] = _raises(
            RuntimeError, P.verify_checkpoint, "e17-fake-f5", manifest_path)
        checks["the size refusal names bytes"] = "bytes" in _message(
            RuntimeError, P.verify_checkpoint, "e17-fake-f5", manifest_path)

        path.unlink()
        checks["a missing file raises FileNotFoundError"] = _raises(
            FileNotFoundError, P.verify_checkpoint, "e17-fake-f5", manifest_path)

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  hash gate: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_path_rerooting() -> bool:
    """The tracked manifest holds cluster paths; --runs-root must re-root them."""
    entry = {"run_id": "e17-final-f5",
             "checkpoint": "/home/user/muography-train/runs/e17-final-f5"
                           "/checkpoints/best.pt"}
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "m.json"
        manifest_path.write_text(json.dumps(
            {"runs_root": "/home/user/muography-train/runs",
             "runs": [entry]}), encoding="utf-8")
        default = P.resolve_checkpoint(entry, None, manifest_path)
        rerooted = P.resolve_checkpoint(entry, Path(tmp) / "local", manifest_path)
        unrelated = P.resolve_checkpoint(
            {"run_id": "e17-final-f5", "checkpoint": "/somewhere/else/best.pt"},
            Path(tmp) / "local", manifest_path)
    checks = {
        "without runs_root the manifest path is used":
            default.as_posix().endswith("runs/e17-final-f5/checkpoints/best.pt"),
        "with runs_root the suffix is preserved":
            rerooted.as_posix().endswith(
                "local/e17-final-f5/checkpoints/best.pt"),
        "an unrelated path falls back to the conventional layout":
            unrelated.as_posix().endswith(
                "local/e17-final-f5/checkpoints/best.pt"),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  path re-rooting: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_real_manifest_and_records() -> bool:
    """The tracked manifest and the archived configs must stay consistent."""
    manifest = json.loads(P.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    run_ids = [entry["run_id"] for entry in manifest["runs"]]
    checks = {
        "the four frozen levels are present": all(
            f"e17-final-f{level}" in run_ids for level in (50, 10, 5, 1)),
        "the two seed replicas are present":
            "e17-final-f5-s1" in run_ids and "e17-final-f5-s2" in run_ids,
        "every entry carries a 64-hex sha256": all(
            len(entry["sha256"]) == 64 for entry in manifest["runs"]),
    }
    generators = {}
    for run_id in run_ids:
        record = P.run_config(run_id)
        if record is not None:
            generators[run_id] = P.generator_name_for(run_id)
    checks["archived configs resolve a generator name"] = len(generators) > 0
    checks["the final runs use unet_dc_5"] = all(
        generators.get(f"e17-final-f{level}") == "unet_dc_5"
        for level in (50, 10, 5, 1) if f"e17-final-f{level}" in generators)
    checks["config_sha8 matches the archived record"] = all(
        P.run_config(entry["run_id"])["config_sha8"] == entry["config_sha8"]
        for entry in manifest["runs"] if P.run_config(entry["run_id"]))
    checks["a run without a record and without a payload raises"] = _raises(
        FileNotFoundError, P.generator_name_for, "e99-does-not-exist")

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  real manifest: {len(run_ids)} runs, generators "
          f"{sorted(set(generators.values()))} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_classical_dispatch() -> bool:
    """The dispatch must be the real classical_baselines function, not a copy."""
    from classical_baselines import denoise as reference_denoise

    rng = np.random.default_rng(20260805)
    ref_counts = rng.poisson(4000.0, size=(48, 48)).astype(np.uint32)
    fraction = 0.05
    low = rng.binomial(ref_counts.astype(np.int64), fraction).astype(np.uint16)
    scale = P.shared_scale(ref_counts)

    counts_only = P.classical_denoise("gaussian_s2.5", None, low, fraction)
    expected_counts = reference_denoise("gaussian", {"sigma": 2.5}, low)
    corrected = P.classical_denoise("gaussian_s2.5", None, low, fraction,
                                    P.CORRECTED, scale)
    raw = P.classical_denoise("gaussian_s2.5", None, low, fraction, P.RAW, scale)
    expected_corrected = (np.log1p(expected_counts / np.float32(fraction))
                          / np.float32(scale))

    checks = {
        "counts match classical_baselines.denoise bit for bit":
            bool(np.array_equal(counts_only, expected_counts)),
        "the corrected convention matches _scored":
            float(np.abs(corrected - expected_corrected).max()) < 1e-6,
        "the corrected image is brighter than the raw one":
            float(corrected.mean()) > float(raw.mean()),
        "the output lands in [0, 1.2]":
            0.0 <= float(corrected.min()) and float(corrected.max()) < 1.2,
        "median dispatch also works": np.isfinite(
            P.classical_denoise("median_r2", None, low, fraction,
                                P.CORRECTED, scale)).all(),
        "an unknown key raises": _raises(
            ValueError, P.classical_denoise, "not_a_method", None, low, fraction),
        "an unknown convention raises": _raises(
            ValueError, P.classical_denoise, "gaussian_s2.5", None, low,
            fraction, "sideways", scale),
        "explicit params override the grid": not np.array_equal(
            P.classical_denoise("gaussian_s2.5", {"sigma": 0.8}, low, fraction),
            counts_only),
        "the frozen winner keys resolve": all(
            P.classical_spec(key)[0] in ("gaussian", "nlm_log", "nlm_anscombe")
            for key in ("nlm_log_h0.5", "nlm_anscombe_h1.2", "gaussian_s2.5")),
    }
    if HAVE_PYWT:
        nlm = P.classical_denoise("nlm_anscombe_h1.2", None, low, fraction,
                                  P.CORRECTED, scale)
        checks["NLM dispatch runs"] = bool(np.isfinite(nlm).all())
    else:
        print("    note: PyWavelets is absent — the NLM winners are resolved "
              "by key but not executed locally (they run on the cluster)")

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  classical dispatch: {sum(checks.values())}/{len(checks)} checks "
          f"-> {'ok' if ok else 'FAIL'}")
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


def _message(exception, fn, *args) -> str:
    try:
        fn(*args)
    except exception as exc:
        return str(exc)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def test_supplementary_manifest_and_preflight() -> bool:
    """The Stage 19.5 supplement is merged, and the pre-flight checks all runs.

    The e19 retrains do not exist in the Phase C manifest and cannot be
    appended to it without changing the sha of a signed artefact. They travel
    in `<name>-e19.json`, which every consumer sees merged. The pre-flight then
    verifies EVERY generator before the driver resolves a split: discovering a
    missing entry after the guard has opened would leave an append-only log
    recording an opening that produced nothing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        main_entry = _fake_run(root, "e17-final-fake", b"phase-c-weights" * 50)
        supplement_entry = _fake_run(root, "e19-unetl1-fake", b"retrain" * 90)
        override = _fake_run(root, "e17-final-fake", b"phase-c-weights" * 50)
        override["best_val_ssim"] = 0.9999

        manifest_path = root / "weights-manifest.json"
        manifest_path.write_text(json.dumps(
            {"runs_root": str(root / "runs"), "runs": [main_entry]}),
            encoding="utf-8")
        supplement_path = P.supplementary_manifest_path(manifest_path)

        alone = P.load_manifest(manifest_path)
        checks = {
            "the supplement path is <name>-e19.json":
                supplement_path.name == "weights-manifest-e19.json",
            "an absent supplement changes nothing":
                sorted(alone["runs_by_id"]) == ["e17-final-fake"],
            "a run that is in neither manifest raises": _raises(
                KeyError, P.manifest_entry, "e19-nowhere", manifest_path),
        }

        supplement_path.write_text(json.dumps(
            {"runs_root": str(root / "runs"),
             "runs": [supplement_entry, override]}), encoding="utf-8")
        merged = P.load_manifest(manifest_path)
        checks["the supplement is merged into one roster"] = (
            sorted(merged["runs_by_id"])
            == ["e17-final-fake", "e19-unetl1-fake"])
        checks["a retrain resolves through the merged roster"] = (
            P.manifest_entry("e19-unetl1-fake", manifest_path)["run_id"]
            == "e19-unetl1-fake")
        checks["the supplement wins on a duplicate, and says so"] = (
            merged["runs_by_id"]["e17-final-fake"]["best_val_ssim"] == 0.9999
            and merged["overridden"] == ["e17-final-fake"])
        checks["each entry names the file it came from"] = (
            merged["runs_by_id"]["e19-unetl1-fake"]["manifest"]
            == str(supplement_path))
        checks["a supplemented run passes the hash gate"] = (
            P.verify_checkpoint("e19-unetl1-fake", manifest_path)["sha256"]
            == supplement_entry["sha256"])

        # the pre-flight: every problem collected, nothing verified halfway
        records = root / "records"
        for run_id in ("e17-final-fake", "e19-unetl1-fake"):
            (records / run_id).mkdir(parents=True, exist_ok=True)
            (records / run_id / "config.json").write_text(json.dumps(
                {"config": {"generator": "unet_dc_5",
                            "input_scaling": "ref_equivalent"}}),
                encoding="utf-8")
        ok_runs = P.verify_all_checkpoints(
            ["e17-final-fake", "e19-unetl1-fake", "e17-final-fake"],
            manifest_path, records_dir=records)
        checks["the pre-flight verifies every distinct run"] = (
            sorted(ok_runs) == ["e17-final-fake", "e19-unetl1-fake"])

        message = _message(RuntimeError, P.verify_all_checkpoints,
                           ["e17-final-fake", "e19-missing", "e19-also-gone"],
                           manifest_path, None, records)
        checks["the pre-flight collects EVERY problem before raising"] = (
            "e19-missing" in message and "e19-also-gone" in message)
        checks["the pre-flight says nothing was opened"] = (
            "nothing was opened" in message)

        # a run whose checkpoint is fine but whose config record is gone
        (records / "e19-unetl1-fake" / "config.json").unlink()
        no_record = _message(RuntimeError, P.verify_all_checkpoints,
                             ["e19-unetl1-fake"], manifest_path, None, records)
        checks["a missing config record is a pre-flight failure"] = (
            "no archived config record" in no_record)

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  supplement and pre-flight: {sum(checks.values())}/{len(checks)} "
          f"checks -> {'ok' if ok else 'FAIL'}")
    return ok


def _message(exception, fn, *args) -> str:
    try:
        fn(*args)
    except exception as exc:
        return str(exc)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def main() -> int:
    print("Stage 19 inference tests  [torch absent: generator build is "
          "cluster-only]")
    results = {
        "hash_gate": test_hash_gate(),
        "path_rerooting": test_path_rerooting(),
        "supplement_and_preflight": test_supplementary_manifest_and_preflight(),
        "real_manifest": test_real_manifest_and_records(),
        "classical_dispatch": test_classical_dispatch(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 PREDICT:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
