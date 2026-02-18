"""Real-data integration tests for augmentation & spatial masking (Phase 4d).

Exercises augmentation transforms and masking against the **real** pre-computed
latents (HDF5 shards) and optionally decodes through the MAISI VAE.
All tests skip gracefully when the external drive is not mounted.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import torch

from neuromf.data.latent_augmentation import (
    PerChannelGaussianNoise,
    build_latent_augmentation,
)
from neuromf.data.latent_hdf5 import build_global_index, discover_shards, read_sample
from neuromf.utils.latent_stats import load_latent_stats

# ---------------------------------------------------------------------------
# Paths & skip guards
# ---------------------------------------------------------------------------
LATENTS_DIR = Path("/media/mpascual/Sandisk2TB/research/neuromf/results/latents")
STATS_PATH = LATENTS_DIR / "latent_stats.json"
VAE_WEIGHTS = Path(
    "/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/"
    "NV-Generate-MR/models/autoencoder_v2.pt"
)

skip_no_latents = pytest.mark.skipif(
    not LATENTS_DIR.exists() or not list(LATENTS_DIR.glob("*.h5")),
    reason="Real latent HDF5 shards not available",
)
skip_no_vae = pytest.mark.skipif(not VAE_WEIGHTS.exists(), reason="VAE checkpoint not available")


def _load_n_latents(n: int) -> list[torch.Tensor]:
    """Load the first *n* raw (un-normalised) latent tensors from HDF5 shards."""
    shard_paths = discover_shards(LATENTS_DIR)
    if not shard_paths:
        return []
    global_idx = build_global_index(shard_paths)
    latents = []
    for shard_path, local_idx in global_idx[:n]:
        with h5py.File(str(shard_path), "r") as f:
            z, _ = read_sample(f, local_idx)
        latents.append(z.float())
    return latents


# ---------------------------------------------------------------------------
# Fast tests (~2-5 s each)
# ---------------------------------------------------------------------------


@pytest.mark.phase4
@pytest.mark.informational
class TestRealDataAugmentation:
    """Phase 4d real-data augmentation integration tests."""

    @skip_no_latents
    def test_P4d_T11_augmentation_preserves_channel_stats(self) -> None:
        """P4d-T11: Geometric augmentation is measure-preserving on real latents.

        Flip_d (depth/L-R axis) should not change per-channel mean or std.
        """
        latents = _load_n_latents(20)

        # Geometric-only config (no noise, no scale) — safe transforms only
        config = {
            "enabled": True,
            "transforms": {
                "flip_d": {"prob": 1.0},
            },
        }
        pipeline = build_latent_augmentation(config)

        orig_means = torch.zeros(4)
        orig_stds = torch.zeros(4)
        aug_means = torch.zeros(4)
        aug_stds = torch.zeros(4)

        n_aug = 10
        for z in latents:
            for c in range(4):
                orig_means[c] += z[c].mean().item()
                orig_stds[c] += z[c].std().item()
            for _ in range(n_aug):
                z_aug = pipeline(z.clone())
                for c in range(4):
                    aug_means[c] += z_aug[c].mean().item()
                    aug_stds[c] += z_aug[c].std().item()

        n_orig = len(latents)
        n_total_aug = n_orig * n_aug
        orig_means /= n_orig
        orig_stds /= n_orig
        aug_means /= n_total_aug
        aug_stds /= n_total_aug

        for c in range(4):
            mean_rel_err = abs(aug_means[c] - orig_means[c]) / (abs(orig_means[c]) + 1e-8)
            std_rel_err = abs(aug_stds[c] - orig_stds[c]) / (abs(orig_stds[c]) + 1e-8)
            assert mean_rel_err < 0.05, (
                f"Channel {c}: mean shifted {mean_rel_err:.1%} "
                f"(orig={orig_means[c]:.4f}, aug={aug_means[c]:.4f})"
            )
            assert std_rel_err < 0.05, (
                f"Channel {c}: std shifted {std_rel_err:.1%} "
                f"(orig={orig_stds[c]:.4f}, aug={aug_stds[c]:.4f})"
            )

    @skip_no_latents
    def test_P4d_T12_noise_calibrated_to_real_stats(self) -> None:
        """P4d-T12: Per-channel noise magnitude matches real channel stds.

        Noise std for channel c should be approximately
        ``std_fraction * real_channel_std[c]``.
        """
        stats = load_latent_stats(STATS_PATH)
        per_ch = stats["per_channel"]
        channel_stds = [per_ch[f"channel_{c}"]["std"] for c in range(4)]

        std_fraction = 0.05
        noise_fn = PerChannelGaussianNoise(
            prob=1.0,
            std_fraction=std_fraction,
            channel_stds=channel_stds,
        )

        latents = _load_n_latents(20)

        # Accumulate noise residuals over multiple samples
        noise_residuals = []
        for z in latents:
            z_noisy = noise_fn(z.clone())
            noise_residuals.append(z_noisy - z)

        noise_stack = torch.stack(noise_residuals)  # (20, 4, 48, 48, 48)

        for c in range(4):
            expected_std = std_fraction * channel_stds[c]
            actual_std = noise_stack[:, c].std().item()
            ratio = actual_std / expected_std
            assert 0.7 < ratio < 1.3, (
                f"Channel {c}: noise std ratio {ratio:.2f} "
                f"(actual={actual_std:.5f}, expected={expected_std:.5f})"
            )

    @skip_no_latents
    def test_P4d_T13_augmented_values_in_distribution(self) -> None:
        """P4d-T13: Full augmentation doesn't produce extreme outliers.

        No augmented value should exceed 2x the original range.
        """
        latents = _load_n_latents(20)

        # Compute original range per channel
        all_vals = torch.stack(latents)  # (20, 4, 48, 48, 48)
        orig_min = all_vals.amin(dim=(0, 2, 3, 4))  # (4,)
        orig_max = all_vals.amax(dim=(0, 2, 3, 4))  # (4,)
        orig_range = orig_max - orig_min

        # Full augmentation with all safe transforms at prob=1.0
        config = {
            "enabled": True,
            "transforms": {
                "flip_d": {"prob": 1.0},
                "gaussian_noise": {"prob": 1.0, "std_fraction": 0.05},
                "intensity_scale": {"prob": 1.0, "factors": 0.05},
            },
        }
        pipeline = build_latent_augmentation(config, latent_stats_path=STATS_PATH)

        for z in latents:
            for _ in range(5):
                z_aug = pipeline(z.clone())
                for c in range(4):
                    ch_min = z_aug[c].min().item()
                    ch_max = z_aug[c].max().item()
                    lower_bound = orig_min[c].item() - 2 * orig_range[c].item()
                    upper_bound = orig_max[c].item() + 2 * orig_range[c].item()
                    assert ch_min > lower_bound, (
                        f"Channel {c}: min={ch_min:.3f} below "
                        f"2x-expanded lower bound {lower_bound:.3f}"
                    )
                    assert ch_max < upper_bound, (
                        f"Channel {c}: max={ch_max:.3f} above "
                        f"2x-expanded upper bound {upper_bound:.3f}"
                    )

    @skip_no_latents
    def test_P4d_T14_real_dataset_with_augmentation(self) -> None:
        """P4d-T14: LatentDataset with augmentation produces valid stochastic outputs."""
        from neuromf.data.latent_dataset import LatentDataset

        config = {
            "enabled": True,
            "transforms": {
                "flip_d": {"prob": 0.5},
                "gaussian_noise": {"prob": 1.0, "std_fraction": 0.05},
                "intensity_scale": {"prob": 0.3, "factors": 0.05},
            },
        }
        aug_pipeline = build_latent_augmentation(config, latent_stats_path=STATS_PATH)

        ds = LatentDataset(
            LATENTS_DIR,
            normalise=True,
            transform=aug_pipeline,
        )
        assert len(ds) >= 100, f"Expected >= 100 latents, got {len(ds)}"

        # Load 10 items, check shapes and finiteness
        for i in range(10):
            item = ds[i]
            z = item["z"]
            assert z.shape == (4, 48, 48, 48), f"Item {i}: shape {z.shape}"
            assert torch.isfinite(z).all(), f"Item {i}: non-finite values"

        # Stochasticity check: same index, different outputs
        outputs = [ds[0]["z"] for _ in range(5)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "All 5 reads of index 0 are identical — augmentation not stochastic"


# ---------------------------------------------------------------------------
# Slow test (~130 s — VAE decode on CPU)
# ---------------------------------------------------------------------------


@pytest.mark.phase4
@pytest.mark.informational
@pytest.mark.slow
class TestRealDataVAEDecode:
    """Phase 4d VAE decode integration test (CPU, ~130s)."""

    @skip_no_latents
    @skip_no_vae
    def test_P4d_T15_flip_decode_ssim(self) -> None:
        """P4d-T15: Flip in latent space → decode → flip back ≈ original in pixel space.

        Brain bilateral symmetry means L-R flip SSIM should be > 0.85.
        """
        from monai.metrics import SSIMMetric

        from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

        # Load real latent stats for denormalization
        stats = load_latent_stats(STATS_PATH)
        per_ch = stats["per_channel"]
        means = torch.tensor([per_ch[f"channel_{c}"]["mean"] for c in range(4)]).view(1, 4, 1, 1, 1)
        stds = torch.tensor([per_ch[f"channel_{c}"]["std"] for c in range(4)]).view(1, 4, 1, 1, 1)

        # Load first latent from HDF5 shard
        shard_paths = discover_shards(LATENTS_DIR)
        global_idx = build_global_index(shard_paths)
        shard_path, local_idx = global_idx[0]
        with h5py.File(str(shard_path), "r") as f:
            z_tensor, _ = read_sample(f, local_idx)
        z_raw = z_tensor.float().unsqueeze(0)  # (1, 4, 48, 48, 48)

        # z_raw is un-normalised from encoding — use directly
        z_orig = z_raw

        # Flip along axis 2 (left-right in RAS convention)
        z_flip = z_orig.flip(dims=[2])

        # Decode through VAE
        vae_cfg = MAISIVAEConfig(
            weights_path=str(VAE_WEIGHTS),
            norm_float16=False,
            num_splits=6,
        )
        vae = MAISIVAEWrapper(vae_cfg, device="cpu")
        vae.eval()

        with torch.no_grad():
            x_orig = vae.decode(z_orig).float()  # (1, 1, 192, 192, 192)
            x_flip = vae.decode(z_flip).float()

        # Flip decoded back to align with original
        x_flip_back = x_flip.flip(dims=[2])

        # Sanity: pixel range for skull-stripped brain
        x_min = x_orig.min().item()
        x_max = x_orig.max().item()
        assert x_max > 0.1, f"Decoded max={x_max:.3f} too low"
        assert x_max < 5.0, f"Decoded max={x_max:.3f} unexpectedly high"

        # SSIM between original and flip-back
        data_range = x_orig.max() - x_orig.min()
        ssim_fn = SSIMMetric(spatial_dims=3, data_range=data_range.item())
        ssim_val = ssim_fn(x_orig, x_flip_back).mean().item()
        assert ssim_val > 0.85, (
            f"SSIM={ssim_val:.4f} between original and flip-back decode "
            f"(expected > 0.85 due to bilateral brain symmetry)"
        )


# ---------------------------------------------------------------------------
# Masking test with real latents (~5 s, uses tiny UNet)
# ---------------------------------------------------------------------------


@pytest.mark.phase4
@pytest.mark.informational
class TestRealDataMasking:
    """Phase 4d masking integration with real latents."""

    @skip_no_latents
    def test_P4d_T16_masking_loss_on_real_latents(self) -> None:
        """P4d-T16: MeanFlowPipeline with masking produces sane loss on real data."""
        from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper
        from neuromf.wrappers.meanflow_loss import MeanFlowPipeline, MeanFlowPipelineConfig

        # Tiny UNet (same pattern as test_spatial_masking.py)
        unet_cfg = MAISIUNetConfig(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=[8, 16, 32, 64],
            attention_levels=[False] * 4,
            num_res_blocks=1,
            num_head_channels=[0, 0, 0, 8],
            norm_num_groups=8,
            use_flash_attention=False,
        )
        model = MAISIUNetWrapper(unet_cfg)

        # Load 4 real latents, center-crop to 16^3 for tiny UNet compatibility
        latents = _load_n_latents(4)
        # Crop from center: 48 -> 16 means offset = (48 - 16) // 2 = 16
        off = (48 - 16) // 2
        z_0 = torch.stack([z[:, off : off + 16, off : off + 16, off : off + 16] for z in latents])
        eps = torch.randn_like(z_0)
        B = z_0.shape[0]
        t = torch.rand(B).clamp(0.05, 0.95)
        r = t * torch.rand(B) * 0.5

        # Unmasked pipeline
        pipeline_unmasked = MeanFlowPipeline(
            MeanFlowPipelineConfig(
                jvp_strategy="finite_difference",
                norm_eps=1.0,
                spatial_mask_ratio=0.0,
            )
        )
        # Masked pipeline
        pipeline_masked = MeanFlowPipeline(
            MeanFlowPipelineConfig(
                jvp_strategy="finite_difference",
                norm_eps=1.0,
                spatial_mask_ratio=0.5,
            )
        )

        # Run both with same inputs (but masks are stochastic)
        torch.manual_seed(42)
        result_unmasked = pipeline_unmasked(model, z_0, eps, t, r)
        torch.manual_seed(43)
        result_masked = pipeline_masked(model, z_0, eps, t, r)

        loss_unmasked = result_unmasked["raw_loss"].item()
        loss_masked = result_masked["raw_loss"].item()

        # (a) Both produce finite, positive loss
        assert torch.isfinite(result_unmasked["raw_loss"]), "Unmasked loss not finite"
        assert torch.isfinite(result_masked["raw_loss"]), "Masked loss not finite"
        assert loss_unmasked > 0, "Unmasked loss should be positive"
        assert loss_masked > 0, "Masked loss should be positive"

        # (b) Masked loss within 3x of unmasked (normalization keeps scale)
        ratio = loss_masked / loss_unmasked
        assert 1.0 / 3.0 < ratio < 3.0, (
            f"Masked/unmasked ratio={ratio:.2f} outside [0.33, 3.0] "
            f"(unmasked={loss_unmasked:.4f}, masked={loss_masked:.4f})"
        )

        # (c) Run masked multiple times — variance should be higher than unmasked
        n_runs = 5
        masked_losses = []
        unmasked_losses = []
        for i in range(n_runs):
            torch.manual_seed(100 + i)
            res_m = pipeline_masked(model, z_0, eps, t, r)
            masked_losses.append(res_m["raw_loss"].item())
            torch.manual_seed(200 + i)
            res_u = pipeline_unmasked(model, z_0, eps, t, r)
            unmasked_losses.append(res_u["raw_loss"].item())

        var_masked = torch.tensor(masked_losses).var().item()
        var_unmasked = torch.tensor(unmasked_losses).var().item()
        # Masked should have higher variance (fewer voxels → noisier estimate)
        # Unmasked variance should be ~0 since only random seed for noise matters,
        # but the pipeline is deterministic given same seed for t/r/eps
        assert var_masked >= var_unmasked, (
            f"Expected masked variance ({var_masked:.6f}) >= unmasked variance ({var_unmasked:.6f})"
        )
