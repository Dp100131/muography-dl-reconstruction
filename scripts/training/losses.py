"""Stage 14 losses — Wasserstein critic with gradient penalty (WGAN-GP).

Exact protocol formulation, ported from the Stage 4 toy benchmark
without semantic changes:

    L_critic    = E[C(cond, fake)] - E[C(cond, real)] + lambda_gp * GP
    L_generator = -lambda_adv * E[C(cond, fake)] + lambda_l1 * L1(fake, real)

    GP = E[(||d C(cond, x_hat) / d x_hat||_2 - 1)^2],  x_hat = a*real + (1-a)*fake

with one alpha per sample. lambda_adv = 0 turns the generator into the plain
U-Net L1 baseline of Stage 13.3 (no critic is evaluated at all).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gradient_penalty(critic, cond: torch.Tensor, real: torch.Tensor,
                     fake: torch.Tensor) -> torch.Tensor:
    """One-centered gradient penalty on the real/fake interpolation line."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(cond, mixed)
    grads = torch.autograd.grad(outputs=scores, inputs=mixed,
                                grad_outputs=torch.ones_like(scores),
                                create_graph=True, retain_graph=True)[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def critic_loss(critic, cond: torch.Tensor, real: torch.Tensor,
                fake: torch.Tensor, lambda_gp: float = 10.0):
    """Return (loss, parts) for one critic step. `fake` must be detached."""
    d_real = critic(cond, real).mean()
    d_fake = critic(cond, fake).mean()
    gp = gradient_penalty(critic, cond, real, fake)
    loss = d_fake - d_real + lambda_gp * gp
    parts = {
        "d_real": float(d_real.detach()),
        "d_fake": float(d_fake.detach()),
        "w_est": float((d_real - d_fake).detach()),  # Wasserstein estimate
        "gp": float(gp.detach()),
    }
    return loss, parts


def generator_loss(critic, cond: torch.Tensor, fake: torch.Tensor,
                   real: torch.Tensor, lambda_l1: float = 100.0,
                   lambda_adv: float = 1.0):
    """Return (loss, parts) for one generator step.

    critic=None or lambda_adv=0 skips the adversarial term entirely, which is
    the pure-L1 mode used by scripts/training/train_unet_l1.py.
    """
    l1 = F.l1_loss(fake, real)
    loss = lambda_l1 * l1
    adv = 0.0
    if critic is not None and lambda_adv != 0.0:
        score = critic(cond, fake).mean()
        loss = loss - lambda_adv * score
        adv = float((-score).detach())
    return loss, {"l1": float(l1.detach()), "adv": adv}
