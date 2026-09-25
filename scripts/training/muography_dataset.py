"""Dataset access layer — preprocessing contract and (optional) PyTorch Dataset.

The dataset shards (dataset/shards/<geometry>.npz) store RAW counts:
  ref_counts:  (n_views, H, W)                  uint32 — 100% reference
  low_counts:  (n_views, n_fracs, K, H, W)      uint16 — subsampled inputs
  fractions:   (n_fracs,)                        float
plus per-shard metadata JSON. Preprocessing is applied at load time so the
contract lives in exactly one place (this module).

Two input conventions (the A/B of Stage 13.4 decides which one Stage 15 uses):

  SHARED ("shared", Phase B v0.9 default)
      x = log1p(counts) / log1p(ref_max_of_the_same_view)
      -> the low-statistics input keeps its systematically lower amplitude
         (~f x) and the network must learn the statistical amplification.

  REF_EQUIVALENT ("ref_equivalent")
      x_low = log1p(counts / f) / log1p(ref_max_of_the_same_view)
      -> counts are rescaled by the KNOWN factor 1/f before log1p (a
         deterministic, invertible transform available at inference time), so
         input and reference share amplitude and only noise differs.

Both use the same scale (an oracle: log1p(ref.max())); the operational
estimator log1p(max(low)/f) and its sensitivity are reported in Stage 13.2.

Held-out data is guarded: split "test" and "generalization" raise unless
MUOGRAPHY_ALLOW_TEST=1 — the test set is opened exactly once, in Stage 20.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = REPO_ROOT / "dataset" / "shards"
SPLITS_PATH = REPO_ROOT / "results" / "dataset" / "splits.json"

SHARED = "shared"
REF_EQUIVALENT = "ref_equivalent"
INPUT_SCALINGS = (SHARED, REF_EQUIVALENT)

HELD_OUT_SPLITS = ("test", "generalization")
ALLOW_TEST_ENV = "MUOGRAPHY_ALLOW_TEST"
AZIMUTH_AXIS = -1  # array layout is (elevation, azimuth)
UINT16_MAX = 65535  # the shards store low_counts as uint16 (build_dataset.py)
# Independent RNG stream for the on-the-fly re-thinning draw: mixing it into
# the same generator as the hflip draw would shift every flip decision, so a
# rethin=True run could not be compared with a rethin=False one at equal seed.
RETHIN_STREAM = 0x52455448  # "RETH"


def preprocess_pair(low_counts: np.ndarray, ref_counts: np.ndarray):
    """Return (x_low, x_ref) float32 in [0, ~1] using the SAME scale."""
    scale = np.log1p(np.float32(ref_counts.max()))
    x_ref = np.log1p(ref_counts.astype(np.float32)) / scale
    x_low = np.log1p(low_counts.astype(np.float32)) / scale
    return x_low, x_ref


def preprocess_pair_corrected(low_counts: np.ndarray, ref_counts: np.ndarray,
                              fraction: float):
    """Reference-equivalent variant: rescale counts by 1/f BEFORE log1p.

    Same shared scale as preprocess_pair, so x_ref is bit-identical between
    the two conventions and only the input changes.
    """
    if not fraction > 0.0:
        raise ValueError(f"fraction must be positive, got {fraction!r}")
    scale = np.log1p(np.float32(ref_counts.max()))
    x_ref = np.log1p(ref_counts.astype(np.float32)) / scale
    x_low = np.log1p(low_counts.astype(np.float32) / np.float32(fraction)) / scale
    return x_low, x_ref


def load_shard(geometry_id: str):
    """Return (arrays, meta) with the count arrays MATERIALIZED in RAM.

    np.load returns a lazy NpzFile: every arr[...] access re-inflates the whole
    compressed member (~0.577 s/pair measured). Materializing once costs about
    64.5 MB per shard and takes the per-pair cost to ~1.5 ms.
    """
    with np.load(SHARD_DIR / f"{geometry_id}.npz") as npz:
        arrays = {
            "ref_counts": np.asarray(npz["ref_counts"]),
            "low_counts": np.asarray(npz["low_counts"]),
            "fractions": np.asarray(npz["fractions"]),
        }
    meta = json.loads((SHARD_DIR / f"{geometry_id}.meta.json").read_text(
        encoding="utf-8"))
    return arrays, meta


def geometry_ids_for_split(split: str):
    """Geometry ids of a split, with the held-out guard (Stage 20 opens test)."""
    if split in HELD_OUT_SPLITS and os.environ.get(ALLOW_TEST_ENV) != "1":
        raise RuntimeError(
            f"split {split!r} is held out: the test/generalization data is "
            f"single-use and is opened exactly once, in Stage 20. Set "
            f"{ALLOW_TEST_ENV}=1 and record the use in the research log if "
            f"this really is that one time.")
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))["splits"]
    if split not in splits:
        raise KeyError(f"unknown split {split!r}; available: {sorted(splits)}")
    return list(splits[split])


def iter_pairs(geometry_ids, fractions=None):
    """Yield (geometry_id, view, fraction, k, low_counts, ref_counts)."""
    for gid in geometry_ids:
        data, _ = load_shard(gid)
        fracs = data["fractions"]
        for vi in range(data["ref_counts"].shape[0]):
            ref = data["ref_counts"][vi]
            for fi, frac in enumerate(fracs):
                if fractions is not None and frac not in fractions:
                    continue
                for k in range(data["low_counts"].shape[2]):
                    yield gid, vi, float(frac), k, data["low_counts"][vi, fi, k], ref


try:  # PyTorch is optional at this stage (framework acta pending signature)
    import torch
    from torch.utils.data import Dataset

    class MuographyPairs(Dataset):
        """Map-style dataset over preprocessed (low, ref) pairs.

        Shards are materialized eagerly (~64.5 MB each; ~738 MB for the 12
        train geometries), which makes num_workers=0 the optimal setting.

        hflip mirrors the AZIMUTH axis only — physically valid by the symmetry
        of the angular grid. A vertical flip would invert elevation and break
        the cos^2(theta) gradient of the muon flux, so it is never applied.
        The flip draw is reproducible: a NumPy Generator seeded from
        (seed + epoch_offset + index). Call set_epoch() once per epoch so the
        augmentation actually varies; leaving it unset reproduces the plain
        seed + index draw.

        rethin=True redraws the low-statistics input from the reference counts
        at every access: low ~ Binomial(ref_counts, f), clipped to uint16.
        That is the SAME generative process the frozen shards were built with
        (scripts/analysis/build_dataset.py, "Binomial(counts, f) — exact
        real-event subsampling"), so it is an exact augmentation, not an approximation:
        instead of cycling the K=20 stored realizations it draws a fresh one
        per epoch. The shards are only read, never rewritten. Reference
        counts, and therefore the target, are untouched. Use it on the TRAIN
        split only — a validation set that changes every epoch would make the
        early-stopping curve meaningless.
        """

        def __init__(self, split=None, geometry_ids=None, fractions=None,
                     input_scaling=SHARED, hflip=False, seed=None,
                     rethin=False):
            # geometry_ids= bypasses the split guard: trusted callers only
            # (the Stage 20 test-set protection lives in geometry_ids_for_split).
            if (split is None) == (geometry_ids is None):
                raise ValueError("pass exactly one of split= or geometry_ids=")
            if input_scaling not in INPUT_SCALINGS:
                raise ValueError(
                    f"input_scaling must be one of {INPUT_SCALINGS}, "
                    f"got {input_scaling!r}")
            if hflip and seed is None:
                raise ValueError("hflip=True requires an explicit seed")
            if rethin and seed is None:
                raise ValueError("rethin=True requires an explicit seed")

            self.split = split
            self.geometry_ids = (list(geometry_ids) if geometry_ids is not None
                                 else geometry_ids_for_split(split))
            self.input_scaling = input_scaling
            self.hflip = bool(hflip)
            self.rethin = bool(rethin)
            self.seed = int(seed) if seed is not None else 0
            self._epoch_offset = 0

            self.index = []
            self._cache = {}
            for gid in self.geometry_ids:
                data, _ = load_shard(gid)
                self._cache[gid] = data
                n_views, n_fracs, k_max = data["low_counts"].shape[:3]
                for vi in range(n_views):
                    for fi in range(n_fracs):
                        frac = float(data["fractions"][fi])
                        if fractions is not None and frac not in fractions:
                            continue
                        for k in range(k_max):
                            self.index.append((gid, vi, fi, k))
            if not self.index:
                raise ValueError(
                    f"empty dataset: geometries={self.geometry_ids} "
                    f"fractions={fractions}")

        def set_epoch(self, epoch: int) -> None:
            """Decorrelate the hflip draws between epochs, reproducibly."""
            self._epoch_offset = int(epoch) * 1_000_003

        def fraction_at(self, i: int) -> float:
            gid, _, fi, _ = self.index[i]
            return float(self._cache[gid]["fractions"][fi])

        def __len__(self):
            return len(self.index)

        def __getitem__(self, i):
            gid, vi, fi, k = self.index[i]
            data = self._cache[gid]
            low = data["low_counts"][vi, fi, k]
            ref = data["ref_counts"][vi]
            fraction = float(data["fractions"][fi])

            if self.rethin:
                rng = np.random.default_rng(
                    [self.seed + self._epoch_offset + i, RETHIN_STREAM])
                low = np.minimum(
                    rng.binomial(ref.astype(np.int64), fraction),
                    UINT16_MAX).astype(np.uint16)

            if self.input_scaling == REF_EQUIVALENT:
                x_low, x_ref = preprocess_pair_corrected(low, ref, fraction)
            else:
                x_low, x_ref = preprocess_pair(low, ref)

            if self.hflip:
                rng = np.random.default_rng(self.seed + self._epoch_offset + i)
                if rng.random() < 0.5:
                    x_low = np.ascontiguousarray(np.flip(x_low, axis=AZIMUTH_AXIS))
                    x_ref = np.ascontiguousarray(np.flip(x_ref, axis=AZIMUTH_AXIS))

            return (torch.from_numpy(x_low).unsqueeze(0),
                    torch.from_numpy(x_ref).unsqueeze(0))

except ImportError:  # pragma: no cover
    pass
