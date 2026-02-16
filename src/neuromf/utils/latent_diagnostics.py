"""Lightweight latent-space diagnostics (no VAE required).

Provides statistics and comparison functions for generated latent tensors,
used by both the ``SampleCollectorCallback`` during training and the
``decode_samples.py`` CLI for post-hoc analysis.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def compute_latent_stats(z: Tensor) -> dict[str, Any]:
    """Per-channel mean/std/min/max/skewness/kurtosis for a batch of latents.

    Args:
        z: Latent tensor of shape ``(B, C, ...)``.

    Returns:
        Dict with per-channel stats: ``mean`` (C,), ``std`` (C,),
        ``min`` (C,), ``max`` (C,), ``skewness`` (C,), ``kurtosis`` (C,),
        plus scalar ``global_min`` and ``global_max``.
    """
    C = z.shape[1]
    z_flat = z.detach().float().reshape(z.shape[0], C, -1)  # (B, C, N)

    # Pool over batch and spatial dims
    z_pool = z_flat.reshape(C, -1)  # (C, B*N)

    mean = z_pool.mean(dim=1)
    std = z_pool.std(dim=1)
    z_min = z_pool.min(dim=1).values
    z_max = z_pool.max(dim=1).values

    # Standardised moments for skewness and kurtosis
    centered = z_pool - mean.unsqueeze(1)
    std_safe = std.clamp(min=1e-8).unsqueeze(1)
    normed = centered / std_safe

    skewness = normed.pow(3).mean(dim=1)
    kurtosis = normed.pow(4).mean(dim=1) - 3.0  # excess kurtosis

    return {
        "mean": mean,
        "std": std,
        "min": z_min,
        "max": z_max,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "global_min": float(z.min()),
        "global_max": float(z.max()),
    }


def compute_nfe_consistency(
    samples: dict[str, Tensor],
) -> dict[str, float]:
    """MSE and cosine similarity between NFE-1 and NFE-N for each N > 1.

    Lower MSE means 1-NFE is converging to multi-step quality.

    Args:
        samples: Dict mapping NFE keys (e.g. ``"nfe_1"``, ``"nfe_5"``)
            to tensors of shape ``(B, C, ...)``.

    Returns:
        Dict with ``mse_1vsN`` and ``cosine_1vsN`` for each N > 1.
    """
    ref_key = "nfe_1"
    if ref_key not in samples:
        return {}

    ref = samples[ref_key].detach().float()
    ref_flat = ref.flatten(1)
    result: dict[str, float] = {}

    for key, val in sorted(samples.items()):
        if key == ref_key:
            continue
        val_f = val.detach().float()
        val_flat = val_f.flatten(1)

        mse = (ref_flat - val_flat).pow(2).mean().item()
        cos = torch.nn.functional.cosine_similarity(ref_flat, val_flat, dim=1).mean().item()

        n_str = key.replace("nfe_", "")
        result[f"mse_1vs{n_str}"] = mse
        result[f"cosine_1vs{n_str}"] = cos

    return result


def compute_inter_epoch_delta(
    current: Tensor,
    previous: Tensor,
) -> dict[str, float]:
    """L2 distance and cosine similarity between same-seed samples across epochs.

    Measures how much generation changes as training progresses.

    Args:
        current: Latent tensor from current epoch ``(B, C, ...)``.
        previous: Latent tensor from previous epoch ``(B, C, ...)``.

    Returns:
        Dict with ``l2_distance`` and ``cosine_similarity``.
    """
    cur_flat = current.detach().float().flatten(1)
    prev_flat = previous.detach().float().flatten(1)

    l2 = (cur_flat - prev_flat).norm(dim=1).mean().item()
    cos = torch.nn.functional.cosine_similarity(cur_flat, prev_flat, dim=1).mean().item()

    return {"l2_distance": l2, "cosine_similarity": cos}
