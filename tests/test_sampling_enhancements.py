"""Phase 5 tests: Post-hoc sampling enhancements.

Tests for stochastic perturbation, norm correction gamma, variance rescaling
integration, and post_process_fn wiring through LatentGenerator.

Test naming convention: test_P5_T{N}_{description}
All tests use mock/tiny models — no GPU or real weights required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from neuromf.sampling.variance_rescaling import stochastic_perturb, variance_rescale

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal model for testing: returns z_t * 0.5 (x-prediction)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z_t * 0.5


# ---------------------------------------------------------------------------
# P5-T20: stochastic_perturb shape preserved
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T20_stochastic_perturb_shape_preserved() -> None:
    """Output shape must match input shape for any ndim."""
    z = torch.randn(4, 4, 8, 8, 8)
    sigma = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out = stochastic_perturb(z, sigma)
    assert out.shape == z.shape


# ---------------------------------------------------------------------------
# P5-T21: stochastic_perturb increases variance per channel
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T21_stochastic_perturb_increases_variance() -> None:
    """Perturbing zeros with sigma_inject should produce per-channel std ~ sigma_inject."""
    n = 1000
    z = torch.zeros(n, 4, 8, 8, 8)
    sigma = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out = stochastic_perturb(z, sigma)

    # Per-channel std across batch and spatial dims
    per_ch_std = out.std(dim=(0, 2, 3, 4))
    for c in range(4):
        assert abs(per_ch_std[c].item() - sigma[c].item()) < 0.05, (
            f"Channel {c}: expected std ~{sigma[c]:.3f}, got {per_ch_std[c]:.3f}"
        )


# ---------------------------------------------------------------------------
# P5-T22: stochastic_perturb zero sigma is identity
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T22_stochastic_perturb_zero_sigma_identity() -> None:
    """sigma_inject=0 should return input unchanged."""
    z = torch.randn(2, 4, 8, 8, 8)
    sigma = torch.zeros(4)
    out = stochastic_perturb(z, sigma)
    torch.testing.assert_close(out, z)


# ---------------------------------------------------------------------------
# P5-T23: norm_correction wired through LatentGenerator
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T23_norm_correction_wired_through(tmp_path: Path) -> None:
    """LatentGenerator with gamma!=1.0 must produce different output than gamma=1.0."""
    from neuromf.generation.latent_generator import LatentGenerator

    model = TinyModel()
    device = torch.device("cpu")
    latent_shape = (4, 8, 8, 8)
    noise = torch.randn(2, *latent_shape)

    # Generate with gamma=1.0 (baseline)
    gen1 = LatentGenerator(model, "x", device, norm_correction=1.0)
    path1 = tmp_path / "nfe1_g1.h5"
    gen1.generate(
        n_samples=2,
        nfe=1,
        output_path=path1,
        latent_shape=latent_shape,
        shared_noise=noise,
    )

    # Generate with gamma=2.0 (corrected)
    gen2 = LatentGenerator(model, "x", device, norm_correction=2.0)
    path2 = tmp_path / "nfe1_g2.h5"
    gen2.generate(
        n_samples=2,
        nfe=1,
        output_path=path2,
        latent_shape=latent_shape,
        shared_noise=noise,
    )

    # Read back and verify they differ

    lat1 = HDF5Manager_read_all(path1, 2, latent_shape)
    lat2 = HDF5Manager_read_all(path2, 2, latent_shape)
    assert not torch.allclose(lat1, lat2, atol=1e-4), (
        "Outputs with gamma=1.0 and gamma=2.0 should differ"
    )


def HDF5Manager_read_all(path: Path, n: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Read all latents from an HDF5 archive."""
    from neuromf.generation.h5_manager import H5Manager

    latents = []
    for i in range(n):
        latents.append(H5Manager.read_latent(path, i))
    return torch.stack(latents)


# ---------------------------------------------------------------------------
# P5-T24: post_process_fn applied before HDF5 write
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T24_post_process_fn_applied(tmp_path: Path) -> None:
    """A post_process_fn that adds 1.0 should shift all latent values by ~1.0."""
    from neuromf.generation.latent_generator import LatentGenerator

    model = TinyModel()
    device = torch.device("cpu")
    latent_shape = (4, 8, 8, 8)
    noise = torch.randn(2, *latent_shape)

    def add_one(z: torch.Tensor) -> torch.Tensor:
        return z + 1.0

    # Without post-processing
    path_raw = tmp_path / "raw.h5"
    gen = LatentGenerator(model, "x", device)
    gen.generate(
        n_samples=2,
        nfe=1,
        output_path=path_raw,
        latent_shape=latent_shape,
        shared_noise=noise,
    )

    # With post-processing
    path_pp = tmp_path / "pp.h5"
    gen.generate(
        n_samples=2,
        nfe=1,
        output_path=path_pp,
        latent_shape=latent_shape,
        shared_noise=noise,
        post_process_fn=add_one,
    )

    raw = HDF5Manager_read_all(path_raw, 2, latent_shape)
    pp = HDF5Manager_read_all(path_pp, 2, latent_shape)

    # The difference should be approximately 1.0 (float16 precision)
    diff = (pp - raw).mean().item()
    assert abs(diff - 1.0) < 0.05, f"Expected mean diff ~1.0, got {diff:.4f}"


# ---------------------------------------------------------------------------
# P5-T25: variance_rescale integration
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T25_variance_rescale_integration() -> None:
    """Variance rescaling should shift per-channel stats to target values."""
    # Generate data with known per-channel stats
    n = 500
    z = torch.randn(n, 4, 8, 8, 8)
    # Scale each channel differently
    z[:, 0] *= 0.5
    z[:, 1] *= 2.0
    z[:, 2] = z[:, 2] + 3.0
    z[:, 3] *= 0.1

    mu_target = torch.tensor([0.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1, 1)
    sigma_target = torch.tensor([1.0, 1.0, 1.0, 1.0]).view(1, 4, 1, 1, 1)

    rescaled = variance_rescale(z, mu_target, sigma_target)

    # Check per-channel stats of rescaled
    for c in range(4):
        ch = rescaled[:, c]
        mu = ch.mean().item()
        sigma = ch.std().item()
        assert abs(mu) < 0.15, f"Channel {c}: mean={mu:.3f}, expected ~0.0"
        assert abs(sigma - 1.0) < 0.15, f"Channel {c}: std={sigma:.3f}, expected ~1.0"


# ---------------------------------------------------------------------------
# P5-T26: enhancement config parsing from OmegaConf
# ---------------------------------------------------------------------------


@pytest.mark.phase5
@pytest.mark.critical
def test_P5_T26_enhancement_config_parsing() -> None:
    """Verify that the enhancements section parses correctly from YAML-like dict."""
    from omegaconf import OmegaConf

    cfg_dict = {
        "generation": {
            "enhancements": {
                "norm_correction": 1.65,
                "stochastic_perturbation": {
                    "enabled": True,
                    "sigma_inject": [0.1, 0.2, 0.3, 0.4],
                    "auto_calibrate": False,
                    "calibration_n_samples": 50,
                },
                "variance_rescale": {"enabled": True},
                "comparison_mode": True,
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    enh = OmegaConf.to_container(cfg.generation.enhancements, resolve=True)

    assert enh["norm_correction"] == 1.65
    assert enh["stochastic_perturbation"]["enabled"] is True
    assert enh["stochastic_perturbation"]["sigma_inject"] == [0.1, 0.2, 0.3, 0.4]
    assert enh["variance_rescale"]["enabled"] is True
    assert enh["comparison_mode"] is True
