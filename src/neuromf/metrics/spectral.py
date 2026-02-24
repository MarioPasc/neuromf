"""High-frequency energy analysis via 3D FFT.

Computes the ratio of spectral energy above a cutoff frequency to total
spectral energy. Used to quantify VAE smoothing vs generative model smoothing
(Phase 5, Axis 5).

Reference: Phase 5 spec section 1, Axis 5.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_hf_energy_ratio(
    volume: Tensor,
    cutoff_fraction: float = 0.5,
) -> float:
    """High-frequency energy ratio via 3D FFT.

    Computes rho(k0) = sum(|F(x)|^2 for |k| > k0) / sum(|F(x)|^2),
    where k0 = cutoff_fraction * k_max (Nyquist frequency).

    Args:
        volume: 3D volume of shape ``(D, H, W)`` or ``(1, 1, D, H, W)``.
        cutoff_fraction: Fraction of Nyquist frequency for cutoff (0.5 = half).

    Returns:
        HF energy ratio in [0, 1]. Higher means more high-frequency content.
    """
    # Squeeze to 3D
    if volume.ndim == 5:
        volume = volume.squeeze(0).squeeze(0)
    elif volume.ndim == 4:
        volume = volume.squeeze(0)
    assert volume.ndim == 3, f"Expected 3D volume, got {volume.ndim}D"

    D, H, W = volume.shape

    # 3D FFT and power spectrum
    F = torch.fft.fftn(volume.float())
    power = torch.abs(F) ** 2
    total_energy = power.sum().item()
    if total_energy < 1e-12:
        return 0.0

    # Build radial frequency grid (normalised to [0, 1])
    freq_d = torch.fft.fftfreq(D, device=volume.device)
    freq_h = torch.fft.fftfreq(H, device=volume.device)
    freq_w = torch.fft.fftfreq(W, device=volume.device)

    fd, fh, fw = torch.meshgrid(freq_d, freq_h, freq_w, indexing="ij")
    # Radial frequency: max possible is 0.5 (Nyquist) per axis
    radial_freq = torch.sqrt(fd**2 + fh**2 + fw**2)

    # Nyquist frequency for radial distance
    k_max = 0.5 * (3**0.5)  # sqrt(3) * 0.5 for 3D
    k0 = cutoff_fraction * k_max

    hf_mask = radial_freq > k0
    hf_energy = power[hf_mask].sum().item()

    return hf_energy / total_energy
