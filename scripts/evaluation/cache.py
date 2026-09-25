"""Stage 19 — the prediction cache. WHAT IS NOT CACHED IS UNMEASURABLE FOREVER.

The test split is opened exactly once (Stage 20). Every downstream analysis --
detectability (Stage 21), the reduction factor (Stage 22), the blind panels,
every figure and every table -- reads this cache and nothing else. A quantity
that was not written here cannot be recomputed without a second read, which
the protocol forbids.

Storage: uint16 on the FIXED scale 65535, so a value is quantized to
1.5e-5 -- far below any difference the thesis claims. float16 was rejected
because its resolution collapses to ~5e-4 near 1.0, exactly where the
reference images live: the round-trip test asserts that SSIM computed on the
restored arrays reproduces SSIM on the originals to better than 1e-4.

Every written file is hashed into `cache-manifest.json` (tracked in git while
the cache itself is ignored), so Stage 23 can put a sha256 next to every
claim, and a re-run from the cache is provably the same data.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

UINT16_SCALE = 65535
MANIFEST_NAME = "cache-manifest.json"
PREDICTIONS_KEY = "predictions"


def sha256_of(path) -> str:
    """Streaming SHA-256 (1 MiB blocks) — the same helper run_campaign uses."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize(arr_float01: np.ndarray) -> np.ndarray:
    """[0, 1] float -> uint16 on the fixed scale (clip, then round-half-even)."""
    clipped = np.clip(np.asarray(arr_float01, dtype=np.float64), 0.0, 1.0)
    return np.rint(clipped * UINT16_SCALE).astype(np.uint16)


def dequantize(arr_uint16: np.ndarray) -> np.ndarray:
    return (np.asarray(arr_uint16, dtype=np.float32)
            / np.float32(UINT16_SCALE)).astype(np.float32)


def save_predictions(path, arr_float01: np.ndarray, **extra) -> Path:
    """Write one (geometry, arm) prediction block, compressed, atomically.

    `extra` arrays travel alongside (fractions, k indices, view indices ...)
    so a cache file is self-describing: the tidy table indexes into it by
    position and the positions must be recoverable from the file itself.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {PREDICTIONS_KEY: quantize(arr_float01),
               "uint16_scale": np.int64(UINT16_SCALE)}
    payload.update({key: np.asarray(value) for key, value in extra.items()})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def load_predictions(path):
    """(float32 predictions in [0, 1], the extra arrays that travelled with it)."""
    with np.load(Path(path)) as npz:
        scale = int(npz["uint16_scale"]) if "uint16_scale" in npz else UINT16_SCALE
        if scale != UINT16_SCALE:
            raise ValueError(f"{path}: cache scale {scale} != {UINT16_SCALE}; "
                             "the file was written by a different contract")
        predictions = dequantize(npz[PREDICTIONS_KEY])
        extra = {key: np.asarray(npz[key]) for key in npz.files
                 if key not in (PREDICTIONS_KEY, "uint16_scale")}
    return predictions, extra


def save_masks(path, **arrays) -> Path:
    """Persist the geometry-derived masks of one geometry, compressed.

    Boolean masks compress to a few percent of their raw size, so they are
    stored as plain arrays rather than bit-packed: a mask that is awkward to
    read back is a mask nobody checks. Path lengths travel as float16 (a 2 m
    ray step over at most ~1 400 m needs three significant digits).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    pickled = sorted(k for k, v in payload.items() if v.dtype == object)
    if pickled:
        # an object array would only be readable through pickle, and the cache
        # is read back by Stages 21 and 22 with allow_pickle=False on purpose:
        # fail here, while the pass can still be re-run, not months later
        raise TypeError(f"save_masks refuses object arrays {pickled}: use a "
                        f"fixed-width dtype (e.g. np.array(names, dtype='U32'))")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def load_masks(path) -> dict:
    with np.load(Path(path), allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def read_manifest(manifest_path) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {"uint16_scale": UINT16_SCALE, "files": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _describe(file_path: Path) -> dict:
    """Shape/dtype for a prediction block, the array roster for anything else.

    The manifest tracks EVERY cache artefact, not only the prediction blocks:
    the mask archives, the tidy tables and the run summary all carry claims of
    Stage 23 and all need a sha256 next to them. Only a prediction block has a
    canonical array, so the description degrades gracefully -- an npz without
    the predictions key is described by its array names, and a plain file (a
    CSV, a JSON) carries no description at all.
    """
    if file_path.suffix != ".npz":
        return {}
    with np.load(file_path, allow_pickle=False) as npz:
        names = list(npz.files)
        if PREDICTIONS_KEY in names:
            array = npz[PREDICTIONS_KEY]
            return {"shape": list(array.shape), "dtype": str(array.dtype)}
        return {"arrays": sorted(names)}


def manifest_update(manifest_path, file_path, root=None) -> dict:
    """Hash one cache file into the manifest; sorted keys, atomic replace.

    Keys are POSIX-style paths relative to `root` (the manifest's own folder by
    default) so the manifest is identical on Windows and on the cluster.
    Accepts any cache artefact (prediction block, mask archive, CSV, JSON).
    """
    manifest_path = Path(manifest_path)
    file_path = Path(file_path)
    root = Path(root) if root is not None else manifest_path.parent
    try:
        key = file_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        key = file_path.name

    manifest = read_manifest(manifest_path)
    manifest.setdefault("uint16_scale", UINT16_SCALE)
    manifest.setdefault("files", {})
    manifest["files"][key] = {
        "sha256": sha256_of(file_path),
        "bytes": int(file_path.stat().st_size),
        **_describe(file_path),
    }
    manifest["files"] = dict(sorted(manifest["files"].items()))
    manifest["n_files"] = len(manifest["files"])
    manifest["total_bytes"] = int(sum(entry["bytes"]
                                      for entry in manifest["files"].values()))
    write_json_atomic(manifest_path, manifest)
    return manifest


def verify_manifest(manifest_path, root=None):
    """[(key, 'missing' | 'sha256 mismatch')] — empty when the cache is intact."""
    manifest_path = Path(manifest_path)
    root = Path(root) if root is not None else manifest_path.parent
    problems = []
    for key, entry in read_manifest(manifest_path).get("files", {}).items():
        path = root / key
        if not path.exists():
            problems.append((key, "missing"))
        elif sha256_of(path) != entry["sha256"]:
            problems.append((key, "sha256 mismatch"))
    return problems


def write_json_atomic(path, payload) -> Path:
    """json.dump to a temporary file, fsync, then os.replace (never truncate)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path
