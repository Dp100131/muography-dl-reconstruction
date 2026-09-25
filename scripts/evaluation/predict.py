"""Stage 19 — inference: load a frozen generator, or dispatch a classical method.

Two rules govern this module.

  1. NOTHING LOADS WITHOUT A HASH. `load_generator` computes the SHA-256 of the
     checkpoint and compares it with models/weights-manifest.json BEFORE calling
     torch.load. A mismatch raises; it never warns and continues. The weights
     are the single irreproducible asset of the project (a 280 MB file per
     level, only on one node until the external backup is done), so a silently
     corrupted or swapped checkpoint would poison the one test pass with no
     way to notice afterwards.
  2. THIS MODULE NEVER TOUCHES A SPLIT. It takes arrays and returns arrays.
     The held-out guard lives in scripts/training/muography_dataset.py and its
     single authorized call-site is scripts/evaluation/run_evaluation.py.

torch is imported LAZILY inside the functions that need it: the classical
dispatch and the manifest verification run on a machine without PyTorch,
which is where the module is developed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

DEFAULT_MANIFEST = REPO_ROOT / "models" / "weights-manifest.json"
# The Stage 19.5 retrains are recorded in a SUPPLEMENT rather than appended to
# the Phase C manifest: that file is a frozen artefact whose sha is quoted in
# the Phase C reports, and rewriting it to add runs that did not exist then
# would invalidate those quotations. Same schema, merged on read.
SUPPLEMENTARY_SUFFIX = "-e19"
RECORDS_DIR = REPO_ROOT / "results" / "training" / "records"
GENERATOR_KEY = "generator"        # key of the state dict inside the payload

RAW = "raw"
CORRECTED = "corrected"
CONVENTIONS = (RAW, CORRECTED)


# ------------------------------------------------------------- manifest layer
def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def supplementary_manifest_path(manifest_path=DEFAULT_MANIFEST) -> Path:
    """`<name>-e19.json` beside the given manifest (may not exist)."""
    path = Path(manifest_path)
    return path.with_name(f"{path.stem}{SUPPLEMENTARY_SUFFIX}{path.suffix}")


def load_manifest(manifest_path=DEFAULT_MANIFEST) -> dict:
    """The main manifest MERGED with its optional Stage 19.5 supplement.

    Both files share the schema {runs_root, runs: [{run_id, checkpoint,
    sha256, bytes, config_sha8, best_val_ssim}]}. A run_id present in both is
    taken from the supplement (it is the newer measurement) and the fact is
    recorded in `overridden`. Each merged entry carries the `runs_root` of the
    file it came from, so a supplement written on a different node still
    re-roots correctly.
    """
    main_path = Path(manifest_path)
    sources = [(main_path, json.loads(main_path.read_text(encoding="utf-8")))]
    supplement = supplementary_manifest_path(main_path)
    if supplement.exists():
        sources.append((supplement,
                        json.loads(supplement.read_text(encoding="utf-8"))))

    runs, overridden = {}, []
    for path, payload in sources:
        root = payload.get("runs_root")
        for entry in payload.get("runs", []):
            run_id = entry.get("run_id")
            if run_id in runs:
                overridden.append(run_id)
            merged = dict(entry)
            merged.setdefault("runs_root", root)
            merged["manifest"] = str(path)
            runs[run_id] = merged
    return {"runs": list(runs.values()),
            "runs_by_id": runs,
            "sources": [str(path) for path, _payload in sources],
            "overridden": sorted(set(overridden))}


def manifest_entry(run_id: str, manifest_path=DEFAULT_MANIFEST) -> dict:
    manifest = load_manifest(manifest_path)
    entry = manifest["runs_by_id"].get(run_id)
    if entry is not None:
        return entry
    known = sorted(manifest["runs_by_id"])
    raise KeyError(f"run {run_id!r} is not in {manifest['sources']}; "
                   f"known: {known}")


def resolve_checkpoint(entry: dict, runs_root=None,
                       manifest_path=DEFAULT_MANIFEST) -> Path:
    """Re-root the manifest path onto the local runs directory when asked.

    The manifest records paths relative to `runs_root` (a cluster path in the
    original run). `runs_root` here lets the same manifest drive a run from a
    pulled copy without editing the tracked file. Absent that override, a
    relative checkpoint path is resolved against REPO_ROOT so the manifest
    works out of the box against a `runs/` directory in the repo root.
    """
    checkpoint = Path(entry["checkpoint"])
    if runs_root is None:
        if not checkpoint.is_absolute():
            return REPO_ROOT / checkpoint
        return checkpoint
    recorded_root = entry.get("runs_root")
    if recorded_root is None:
        recorded_root = json.loads(
            Path(manifest_path).read_text(encoding="utf-8")).get("runs_root")
    runs_root = Path(runs_root)
    if recorded_root:
        try:
            return runs_root / checkpoint.relative_to(recorded_root)
        except ValueError:
            pass
    return runs_root / entry["run_id"] / "checkpoints" / "best.pt"


def verify_checkpoint(run_id: str, manifest_path=DEFAULT_MANIFEST,
                      runs_root=None) -> dict:
    """Hash the file and compare with the manifest. Raises on any mismatch."""
    entry = manifest_entry(run_id, manifest_path)
    path = resolve_checkpoint(entry, runs_root, manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint for {run_id!r} not found: {path}")
    size = path.stat().st_size
    expected_size = entry.get("bytes")
    if expected_size is not None and int(expected_size) != size:
        raise RuntimeError(
            f"{run_id}: checkpoint is {size} bytes, the manifest says "
            f"{expected_size} — refusing to load")
    digest = sha256_of(path)
    if digest != entry["sha256"]:
        raise RuntimeError(
            f"{run_id}: sha256 mismatch — file {digest}, manifest "
            f"{entry['sha256']} — refusing to load. The weights are the one "
            f"irreproducible asset of the project; fix the copy, do not "
            f"regenerate the manifest.")
    return {"run_id": run_id, "path": str(path), "sha256": digest,
            "bytes": int(size), "config_sha8": entry.get("config_sha8"),
            "best_val_ssim": entry.get("best_val_ssim"),
            "manifest": entry.get("manifest")}


def verify_all_checkpoints(run_ids, manifest_path=DEFAULT_MANIFEST,
                           runs_root=None, records_dir=RECORDS_DIR) -> dict:
    """PRE-FLIGHT: every generator hashed and every archived config present.

    Called BEFORE any split is resolved and before the held-out guard is
    opened. A missing manifest entry or a stale checkpoint discovered halfway
    through the single test pass would leave the access log claiming an
    opening that produced nothing -- and the log is append-only, so that entry
    can never be taken back. Every run is checked and EVERY problem is
    collected before raising, so one job fixes the whole roster instead of one
    job per missing file.
    """
    ok, problems = {}, []
    for run_id in list(dict.fromkeys(run_ids)):
        try:
            provenance = verify_checkpoint(run_id, manifest_path, runs_root)
        except (KeyError, FileNotFoundError, RuntimeError) as exc:
            problems.append(f"{run_id}: {exc}")
            continue
        if run_config(run_id, records_dir) is None:
            problems.append(f"{run_id}: no archived config record in "
                            f"{Path(records_dir) / run_id / 'config.json'} — "
                            f"the input contract and the architecture would "
                            f"have to be guessed")
            continue
        ok[run_id] = provenance
    if problems:
        raise RuntimeError(
            "checkpoint pre-flight failed for "
            f"{len(problems)} of {len(list(dict.fromkeys(run_ids)))} runs; "
            "nothing was opened:\n  - " + "\n  - ".join(problems))
    return ok


def run_config(run_id: str, records_dir=RECORDS_DIR):
    """The archived training config of a run, or None if the record is gone."""
    path = Path(records_dir) / run_id / "config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generator_name_for(run_id: str, records_dir=RECORDS_DIR, payload=None) -> str:
    """Generator architecture of a run: the archived record first.

    The checkpoint payload also carries its config (engine.save_checkpoint
    stores it), so a run whose record folder was lost can still be rebuilt --
    but the record is preferred because it is tracked in git and the payload
    is not.
    """
    record = run_config(run_id, records_dir)
    if record is not None:
        return str(record["config"]["generator"])
    if payload is not None and isinstance(payload.get("config"), dict):
        return str(payload["config"]["generator"])
    raise FileNotFoundError(
        f"no archived config for {run_id!r} in {records_dir} and the "
        f"checkpoint carries none — cannot know which architecture to build")


# ------------------------------------------------------------------ generator
def load_generator(run_id: str, manifest_path=DEFAULT_MANIFEST, runs_root=None,
                   device: str = "cpu", records_dir=RECORDS_DIR):
    """(eval-mode generator, provenance dict). Verifies the hash FIRST."""
    provenance = verify_checkpoint(run_id, manifest_path, runs_root)

    import torch  # lazy: the module is developed on a machine without torch
    from models import build_generator

    payload = torch.load(provenance["path"], map_location="cpu",
                         weights_only=False)
    if GENERATOR_KEY not in payload:
        raise KeyError(f"{run_id}: checkpoint has no {GENERATOR_KEY!r} key; "
                       f"found {sorted(payload)}")
    name = generator_name_for(run_id, records_dir, payload)
    model = build_generator(name)
    model.load_state_dict(payload[GENERATOR_KEY], strict=True)
    model.to(torch.device(device))
    model.eval()

    provenance.update({
        "generator": name,
        "epoch": payload.get("epoch"),
        "checkpoint_config_sha8": payload.get("config_sha8"),
        "device": str(device),
    })
    stored_sha8 = payload.get("config_sha8")
    if (stored_sha8 and provenance.get("config_sha8")
            and stored_sha8 != provenance["config_sha8"]):
        raise RuntimeError(
            f"{run_id}: checkpoint config_sha8 {stored_sha8} != manifest "
            f"{provenance['config_sha8']}")
    return model, provenance


def predict_batch(model, x_low_batch: np.ndarray, device: str = "cpu",
                  batch_size: int = 8) -> np.ndarray:
    """Forward-only inference on (N, H, W) or (N, 1, H, W) float arrays.

    Returns float32 clipped to [0, 1]: the cache stores uint16 on that range
    and a generator can overshoot it by a few 1e-3 at the extremes.
    """
    import torch

    x = np.asarray(x_low_batch, dtype=np.float32)
    if x.ndim == 2:
        x = x[np.newaxis, np.newaxis]
    elif x.ndim == 3:
        x = x[:, np.newaxis]
    elif x.ndim != 4:
        raise ValueError(f"expected 2-, 3- or 4-D input, got shape {x.shape}")

    torch_device = torch.device(device)
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            chunk = torch.from_numpy(x[start:start + batch_size]).to(torch_device)
            outputs.append(model(chunk).detach().to("cpu").numpy())
    out = np.concatenate(outputs, axis=0)[:, 0]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ classical
def classical_spec(method_key: str):
    """(method, params) of a grid key, straight from training.classical_baselines.

    The grid is the frozen Stage 13 one; the winners are read from it rather
    than re-declared, so a hyper-parameter can never drift between the
    selection run and the test pass.
    """
    from classical_baselines import method_grid

    for key, method, params in method_grid():
        if key == method_key:
            return method, dict(params)
    # nlm_* disappear from the grid when PyWavelets is absent; resolve the
    # frozen winners by name so the failure is a clear import error later,
    # not a confusing "unknown method".
    fallback = {
        "nlm_log_h0.5": ("nlm_log", {"h_mult": 0.5}),
        "nlm_log_h0.8": ("nlm_log", {"h_mult": 0.8}),
        "nlm_log_h1.2": ("nlm_log", {"h_mult": 1.2}),
        "nlm_anscombe_h0.5": ("nlm_anscombe", {"h_mult": 0.5}),
        "nlm_anscombe_h0.8": ("nlm_anscombe", {"h_mult": 0.8}),
        "nlm_anscombe_h1.2": ("nlm_anscombe", {"h_mult": 1.2}),
    }
    if method_key in fallback:
        return fallback[method_key]
    raise ValueError(f"unknown classical method key {method_key!r}")


def classical_denoise(method_key: str, params, low_counts: np.ndarray,
                      fraction: float, convention: str = CORRECTED,
                      scale: float | None = None) -> np.ndarray:
    """Denoise raw counts and (optionally) map them into the [0, 1] image space.

    scale=None returns denoised COUNTS (float32). Passing the shared scale
    log1p(ref.max()) returns the preprocessed image, reproducing
    classical_baselines._scored exactly:

        raw        x = log1p(denoise(low)) / scale
        corrected  x = log1p(denoise(low) / f) / scale

    The scale is an argument rather than something derived here because this
    module is forbidden from reading the reference itself.
    """
    from classical_baselines import denoise

    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, "
                         f"got {convention!r}")
    method, resolved = classical_spec(method_key)
    if params:
        resolved = dict(params)
    denoised = denoise(method, resolved, np.asarray(low_counts))
    if scale is None:
        return denoised.astype(np.float32)
    if not scale > 0.0:
        raise ValueError(f"scale must be positive, got {scale!r}")
    counts = (denoised if convention == RAW
              else denoised / np.float32(fraction))
    return (np.log1p(counts) / np.float32(scale)).astype(np.float32)


def shared_scale(ref_counts: np.ndarray) -> float:
    """log1p(ref.max()) — the oracle scale of the frozen preprocessing contract."""
    return float(np.log1p(np.float32(np.asarray(ref_counts).max())))
