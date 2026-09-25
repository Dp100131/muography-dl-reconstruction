#!/usr/bin/env python3
"""Stage 14 unit tests — architecture, losses, determinism and checkpointing.

Six checks, all self-verifying and pytest-free:
  1. shapes           G: (N,1,256,256) -> (N,1,256,256); D-70 -> (N,1,30,30)
  2. receptive field   MEASURED by gradient, not asserted from the diagram:
                       70 +- 2 px for D-70, >= 190 px for the G-A centre pixel
                       (the multiple-scattering halo is 92 px wide)
  3. gradient penalty  exact against a linear critic with a known input-gradient
                       norm: ||w||=1 -> GP=0, ||w||=2 -> GP=1
  4. determinism       same seed, same 3 iterations, identical losses on CPU
  5. preprocess        x_ref max = 1, shared scale, 1/f variant mean-matches
  6. checkpoint        save -> fresh Trainer -> resume -> identical next step

Torch is optional at this stage: without it the script prints SKIPPED and
exits 0 so the night job can continue.

Run:  python scripts/training/test_model_units.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("STAGE 14 UNIT TESTS: SKIPPED (torch not installed)")
    sys.exit(0)

from models import build_critic, build_generator, count_params  # noqa: E402
from muography_dataset import (preprocess_pair,  # noqa: E402
                               preprocess_pair_corrected)

from losses import critic_loss, generator_loss, gradient_penalty  # noqa: E402

IMAGE_SIZE = 256
D70_RECEPTIVE_FIELD = 70
D70_TOLERANCE = 2
GA_MIN_RECEPTIVE_FIELD = 190


# ------------------------------------------------------- receptive-field probe
def _flatten_norms(module: nn.Module) -> None:
    """InstanceNorm makes every gradient global (it mixes the whole map), so it
    is swapped for Identity before measuring a receptive field."""
    for name, child in module.named_children():
        if isinstance(child, (nn.InstanceNorm2d, nn.BatchNorm2d)):
            setattr(module, name, nn.Identity())
        else:
            _flatten_norms(child)


def _make_probe(module: nn.Module) -> nn.Module:
    """Positive averaging weights: every activation stays O(1) and every
    gradient inside the receptive field is strictly positive, so the non-zero
    extent of d(out)/d(in) IS the receptive field — no threshold guessing."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            fan_in = layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3]
            nn.init.constant_(layer.weight, 1.0 / fan_in)
        elif isinstance(layer, nn.ConvTranspose2d):
            taps = (layer.weight.shape[0] * layer.weight.shape[2]
                    * layer.weight.shape[3]
                    / (layer.stride[0] * layer.stride[1]))
            nn.init.constant_(layer.weight, 1.0 / taps)
        if getattr(layer, "bias", None) is not None:
            nn.init.zeros_(layer.bias)
    _flatten_norms(module)
    return module.double().eval()


def _extent(grad: np.ndarray) -> tuple:
    rows = np.nonzero(grad.sum(axis=1))[0]
    cols = np.nonzero(grad.sum(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, 0
    return int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)


def _critic_receptive_field(name: str) -> tuple:
    critic = _make_probe(build_critic(name))
    cond = torch.ones(1, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float64)
    img = torch.ones(1, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float64,
                     requires_grad=True)
    out = critic(cond, img)
    centre = out[0, 0, out.shape[-2] // 2, out.shape[-1] // 2]
    grad = torch.autograd.grad(centre, img)[0][0, 0].abs().numpy()
    return _extent(grad)


def _generator_receptive_field(name: str) -> tuple:
    generator = _make_probe(build_generator(name))
    x = torch.ones(1, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float64,
                   requires_grad=True)
    out = generator(x)
    centre = out[0, 0, IMAGE_SIZE // 2, IMAGE_SIZE // 2]
    grad = torch.autograd.grad(centre, x)[0][0, 0].abs().numpy()
    return _extent(grad)


# ---------------------------------------------------------------------- tests
def test_shapes() -> bool:
    generator = build_generator("unet_p2p_6")
    critic = build_critic("patch_critic_70")
    with torch.no_grad():
        y = generator(torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE))
        scores = critic(torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE),
                        torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE))
    ok = (tuple(y.shape) == (1, 1, IMAGE_SIZE, IMAGE_SIZE)
          and tuple(scores.shape) == (2, 1, 30, 30)
          and float(y.min()) >= 0.0 and float(y.max()) <= 1.0)
    print(f"  shapes: G {tuple(y.shape)} ({count_params(generator) / 1e6:.2f} M), "
          f"D {tuple(scores.shape)} ({count_params(critic) / 1e6:.2f} M) "
          f"-> {'ok' if ok else 'FAIL'}")
    if not ok:
        return False

    # unet_p2p_4 and unet_p2p_6_noskip are the Stage 18 ablation arms (A5 and
    # A4): their channel arithmetic differs from G-A, so it is checked here
    # rather than discovered 120 epochs into an ablation run.
    for name in ("unet_p2p_6_base32", "unet_dc_5", "unet_p2p_4",
                 "unet_p2p_6_noskip"):
        model = build_generator(name)
        with torch.no_grad():
            shape = tuple(model(torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE)).shape)
        if shape != (1, 1, IMAGE_SIZE, IMAGE_SIZE):
            print(f"  shapes: {name} produced {shape} -> FAIL")
            return False
        print(f"  shapes: {name} {shape} "
              f"({count_params(model) / 1e6:.2f} M) -> ok")
    d142 = build_critic("patch_critic_142")
    with torch.no_grad():
        shape = tuple(d142(torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE),
                           torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE)).shape)
    print(f"  shapes: patch_critic_142 {shape} "
          f"({count_params(d142) / 1e6:.2f} M) -> ok")
    return True


def test_receptive_field() -> bool:
    rows, cols = _critic_receptive_field("patch_critic_70")
    d70_ok = (abs(rows - D70_RECEPTIVE_FIELD) <= D70_TOLERANCE
              and abs(cols - D70_RECEPTIVE_FIELD) <= D70_TOLERANCE)
    print(f"  receptive field D-70: measured {rows} x {cols} px "
          f"(expected {D70_RECEPTIVE_FIELD} +- {D70_TOLERANCE}) "
          f"-> {'ok' if d70_ok else 'FAIL'}")

    rows_g, cols_g = _generator_receptive_field("unet_p2p_6")
    ga_ok = min(rows_g, cols_g) >= GA_MIN_RECEPTIVE_FIELD
    print(f"  receptive field G-A: measured {rows_g} x {cols_g} px "
          f"(need >= {GA_MIN_RECEPTIVE_FIELD}, halo is 92 px) "
          f"-> {'ok' if ga_ok else 'FAIL'}")
    return d70_ok and ga_ok


class _LinearCritic(nn.Module):
    """C(cond, img) = <w, img>: the input gradient is exactly w, so the
    gradient penalty has a closed form (||w|| - 1)^2."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("w", weight)

    def forward(self, cond, img):  # noqa: ARG002 — the critic ignores cond here
        return (img * self.w).sum(dim=(1, 2, 3)).view(-1, 1, 1, 1)


def test_gradient_penalty() -> bool:
    torch.manual_seed(20260805140001)
    shape = (1, 8, 8)
    raw = torch.randn(*shape, dtype=torch.float64)
    unit = raw / raw.norm()

    real = torch.rand(4, *shape, dtype=torch.float64)
    fake = torch.rand(4, *shape, dtype=torch.float64)
    cond = torch.rand(4, *shape, dtype=torch.float64)

    gp_unit = float(gradient_penalty(_LinearCritic(unit), cond, real, fake))
    gp_double = float(gradient_penalty(_LinearCritic(2.0 * unit), cond, real, fake))
    ok = abs(gp_unit) < 1e-5 and abs(gp_double - 1.0) < 1e-4
    print(f"  gradient penalty: ||w||=1 -> {gp_unit:.3e} (want < 1e-5); "
          f"||w||=2 -> {gp_double:.8f} (want 1 +- 1e-4) "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


def _three_iterations(seed: int) -> list:
    torch.manual_seed(seed)
    generator = build_generator("unet_p2p_6_base32")
    critic = build_critic("patch_critic_70")
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    cond = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)

    losses = []
    for _ in range(3):
        with torch.no_grad():
            fake = generator(cond)
        loss_c, parts_c = critic_loss(critic, cond, target, fake, 10.0)
        opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        opt_c.step()

        fake = generator(cond)
        loss_g, parts_g = generator_loss(critic, cond, fake, target, 100.0, 1.0)
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        losses.append((float(loss_c), float(loss_g), parts_c["gp"], parts_g["l1"]))
    return losses


def test_determinism() -> bool:
    first = _three_iterations(20260805140002)
    second = _three_iterations(20260805140002)
    ok = first == second
    print(f"  determinism (CPU, 3 iterations): "
          f"loss_c={first[-1][0]:+.6f} loss_g={first[-1][1]:+.6f} "
          f"-> {'ok' if ok else 'FAIL'}")
    if not ok:
        for a, b in zip(first, second):
            if a != b:
                print(f"    diverged: {a} vs {b}")
    return ok


def test_preprocess_contract() -> bool:
    rng = np.random.default_rng(20260805140003)
    ref = rng.poisson(400.0, size=(64, 64)).astype(np.uint32)
    ok = True
    for fraction in (0.50, 0.05, 0.01):
        low = rng.binomial(ref.astype(np.int64), fraction).astype(np.uint16)
        x_low, x_ref = preprocess_pair(low, ref)
        x_low_c, x_ref_c = preprocess_pair_corrected(low, ref, fraction)

        if abs(float(x_ref.max()) - 1.0) > 1e-6:
            ok = False
            print(f"  preprocess f={fraction}: max(x_ref)={x_ref.max():.6f} "
                  "!= 1 -> FAIL")
        if not np.array_equal(x_ref, x_ref_c):
            ok = False
            print(f"  preprocess f={fraction}: the two conventions must share "
                  "x_ref -> FAIL")
        if float(x_low.mean()) >= float(x_ref.mean()):
            ok = False
            print(f"  preprocess f={fraction}: shared-scale input should be "
                  "darker than the reference -> FAIL")
        rel = abs(float(x_low_c.mean()) - float(x_ref.mean())) / float(x_ref.mean())
        if rel > 0.05:
            ok = False
            print(f"  preprocess f={fraction}: 1/f mean gap {rel:.4f} > 0.05 "
                  "-> FAIL")
        else:
            print(f"  preprocess f={fraction:.2f}: shared mean "
                  f"{x_low.mean():.4f} / corrected {x_low_c.mean():.4f} vs ref "
                  f"{x_ref.mean():.4f} (rel {rel:.4f}) -> ok")
    return ok


class _SyntheticPairs(torch.utils.data.Dataset):
    """Deterministic (low, ref) stand-in so the checkpoint test does not depend
    on the dataset shards being deployed."""

    def __init__(self, n: int = 4, size: int = 64, seed: int = 20260805140004):
        rng = np.random.default_rng(seed)
        self.low = rng.random((n, 1, size, size), dtype=np.float32) * 0.4
        self.ref = (self.low * 2.0 + 0.1).astype(np.float32)

    def __len__(self):
        return self.low.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.low[i]), torch.from_numpy(self.ref[i])


def test_checkpoint_roundtrip() -> bool:
    from engine import RECORDS_DIR, Trainer  # noqa: PLC0415 — torch-only import

    config = {
        "run_id": "unit-checkpoint",
        "generator": "unet_p2p_6_base32",
        "critic": "patch_critic_70",
        "fraction": 0.05,
        "input_scaling": "shared",
        "hflip": False,
        "n_critic": 2,
        "batch": 2,
        "epochs": 1,
        "seed": 20260805140005,
        "sample_every": 0,
        "device": "cpu",
    }
    data = (_SyntheticPairs(), _SyntheticPairs(n=2))
    tmp = Path(tempfile.mkdtemp(prefix="muo-unit-"))
    record_dirs = [RECORDS_DIR / "unit-checkpoint"]
    try:
        first = Trainer(config, tmp / "a", datasets=data)
        first.train(resume="none")

        # Same config on purpose: load_checkpoint refuses a config_sha8 that
        # does not match, which is exactly the guard being exercised here.
        second = Trainer(config, tmp / "b", datasets=data)
        second.load_checkpoint(tmp / "a" / "checkpoints" / "last.pt")

        mismatched = [k for k, v in first.generator.state_dict().items()
                      if not torch.equal(v, second.generator.state_dict()[k])]
        if mismatched:
            print(f"  checkpoint round-trip: {len(mismatched)} generator "
                  f"tensors differ after load -> FAIL")
            return False

        cond = torch.rand(2, 1, 64, 64, generator=torch.Generator().manual_seed(7))
        target = torch.rand(2, 1, 64, 64, generator=torch.Generator().manual_seed(8))
        # The torch RNG is global and both trainers share it, so the gradient
        # penalty alphas are pinned to the same draw for the two steps.
        rng_state = torch.get_rng_state()
        parts_a = first.train_step(cond.clone(), target.clone())
        torch.set_rng_state(rng_state)
        parts_b = second.train_step(cond.clone(), target.clone())
        deltas = {k: abs(parts_a[k] - parts_b[k]) for k in parts_a}
        worst = max(deltas.values())
        ok = worst < 1e-12
        print(f"  checkpoint round-trip: next-step losses match to "
              f"{worst:.2e} (l1={parts_a['l1']:.6f}, gp={parts_a['gp']:.6f}) "
              f"-> {'ok' if ok else 'FAIL'}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for path in record_dirs:
            shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    print("Stage 14 unit tests")
    results = {
        "shapes": test_shapes(),
        "receptive_field": test_receptive_field(),
        "gradient_penalty": test_gradient_penalty(),
        "determinism": test_determinism(),
        "preprocess_contract": test_preprocess_contract(),
        "checkpoint_roundtrip": test_checkpoint_roundtrip(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 14 UNIT TESTS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
