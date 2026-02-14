"""Tests for real latent smoke test — Phase 3 verification.

P3-T13: Load real .pt latent files and forward through UNet.
"""

from pathlib import Path

import pytest
import torch

from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper

LATENTS_DIR = Path("/media/mpascual/Sandisk2TB/research/neuromf/results/latents")


@pytest.mark.phase3
@pytest.mark.informational
def test_P3_T13_real_latent_smoke_test(device: torch.device) -> None:
    """P3-T13: Load 5 real .pt files, forward through UNet, check output.

    Skipped if latent files are not available (e.g., Phase 1 not run yet).
    """
    if not LATENTS_DIR.exists():
        pytest.skip(f"Latents directory not found: {LATENTS_DIR}")

    pt_files = sorted(LATENTS_DIR.glob("*.pt"))[:5]
    if len(pt_files) < 5:
        pytest.skip(f"Need at least 5 .pt files, found {len(pt_files)}")

    # Load latents and detect spatial size
    latents = []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        z = data["z"] if isinstance(data, dict) else data
        assert z.ndim == 4 and z.shape[0] == 4, f"Unexpected shape {z.shape} in {pt_file.name}"
        spatial = z.shape[1]
        if spatial not in (32, 48):
            pytest.fail(f"Unexpected spatial dim {spatial} in {pt_file.name} (expected 32 or 48)")
        assert torch.isfinite(z).all(), f"NaN/Inf in {pt_file.name}"
        latents.append(z)

    spatial = latents[0].shape[1]  # 32 or 48

    # Create model — spatial dim must be divisible by 8 (3 downsample levels)
    torch.manual_seed(42)
    config = MAISIUNetConfig(prediction_type="x")
    model = MAISIUNetWrapper(config)
    model.eval()

    # Forward pass with first latent (batch=1)
    z = latents[0].unsqueeze(0)  # (1, 4, S, S, S)
    r = torch.tensor([0.2])
    t = torch.tensor([0.7])

    with torch.no_grad():
        out = model(z, r, t)

    assert out.shape == (1, 4, spatial, spatial, spatial), f"Output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"

    # Basic statistics sanity check
    for i, z_i in enumerate(latents):
        assert z_i.mean().abs() < 5.0, f"Latent {i} mean suspiciously large: {z_i.mean():.2f}"
        assert z_i.std() < 10.0, f"Latent {i} std suspiciously large: {z_i.std():.2f}"
