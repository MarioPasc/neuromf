"""CLI script for MAISI VAE reconstruction validation on FOMO-60K brain MRI.

Produces publication-quality figures, latent space statistics, negative
controls, and rich HTML/Markdown reports.

Usage:
    ~/.conda/envs/neuromf/bin/python experiments/cli/validate_vae.py \
        --config configs/vae_validation.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from rich.logging import RichHandler

from neuromf.data.fomo60k import FOMO60KConfig, get_fomo60k_file_list
from neuromf.data.mri_preprocessing import build_mri_preprocessing_from_config
from neuromf.metrics.ssim_psnr import compute_psnr, compute_ssim_3d
from neuromf.utils.visualisation import (
    plot_error_heatmap,
    plot_latent_histograms,
    plot_metrics_distribution,
    plot_reconstruction_comparison,
)
from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Number of voxels to subsample per channel per volume for histograms
_HIST_SUBSAMPLE = 4096


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def run_negative_controls(
    vae: MAISIVAEWrapper,
    device: torch.device,
) -> dict:
    """Run negative controls proving the VAE behaves as expected.

    Tests:
        1. Gaussian noise input -> low SSIM (not identity)
        2. Uniform noise input -> low SSIM (not identity)
        3. Blank (zeros) -> finite output
        4. Constant (0.5) -> finite output
        5. Wrong scale factor: sf=1.0 vs correct sf
        6. Encoding stochasticity: seeded deterministic, unseeded stochastic

    Args:
        vae: Loaded MAISI VAE wrapper.
        device: Torch device.

    Returns:
        Dict with results for each negative control.
    """
    results: dict = {}
    shape = (1, 1, 128, 128, 128)

    # 1. Gaussian noise
    logger.info("Negative control: Gaussian noise input")
    x_noise = torch.randn(shape, device=device).clamp(0.0, 1.0)
    x_hat_noise = vae.reconstruct(x_noise)
    noise_ssim = compute_ssim_3d(x_noise.cpu(), x_hat_noise.cpu())
    results["gaussian_noise"] = {
        "ssim": noise_ssim,
        "expected": "< 0.5",
        "pass": noise_ssim < 0.5,
    }
    logger.info("  Gaussian noise SSIM: %.4f (expect < 0.5)", noise_ssim)
    del x_noise, x_hat_noise

    # 2. Uniform noise
    logger.info("Negative control: Uniform noise input")
    x_uniform = torch.rand(shape, device=device)
    x_hat_uniform = vae.reconstruct(x_uniform)
    uniform_ssim = compute_ssim_3d(x_uniform.cpu(), x_hat_uniform.cpu())
    results["uniform_noise"] = {
        "ssim": uniform_ssim,
        "expected": "< 0.5",
        "pass": uniform_ssim < 0.5,
    }
    logger.info("  Uniform noise SSIM: %.4f (expect < 0.5)", uniform_ssim)
    del x_uniform, x_hat_uniform

    # 3. Blank (zeros)
    logger.info("Negative control: Blank (zeros) input")
    x_blank = torch.zeros(shape, device=device)
    z_blank = vae.encode(x_blank)
    x_hat_blank = vae.decode(z_blank)
    blank_finite = bool(torch.isfinite(z_blank).all() and torch.isfinite(x_hat_blank).all())
    results["blank_zeros"] = {
        "latent_finite": blank_finite,
        "recon_finite": blank_finite,
        "pass": blank_finite,
    }
    logger.info("  Blank input: all finite = %s", blank_finite)
    del x_blank, z_blank, x_hat_blank

    # 4. Constant (0.5)
    logger.info("Negative control: Constant (0.5) input")
    x_const = torch.full(shape, 0.5, device=device)
    z_const = vae.encode(x_const)
    x_hat_const = vae.decode(z_const)
    const_finite = bool(torch.isfinite(z_const).all() and torch.isfinite(x_hat_const).all())
    results["constant_half"] = {
        "latent_finite": const_finite,
        "recon_finite": const_finite,
        "pass": const_finite,
    }
    logger.info("  Constant 0.5 input: all finite = %s", const_finite)
    del x_const, z_const, x_hat_const

    # 5. Wrong scale factor
    logger.info("Negative control: Wrong scale factor (1.0 vs %.10f)", vae.config.scale_factor)
    x_test = torch.randn(shape, device=device).clamp(0.0, 1.0)
    z_test = vae.encode(x_test)

    # Correct scale factor
    x_hat_correct = vae.decode(z_test)
    ssim_correct = compute_ssim_3d(x_test.cpu(), x_hat_correct.cpu())

    # Wrong scale factor (1.0): bypass wrapper, call decoder directly
    z_wrong_scale = z_test / 1.0  # sf=1.0 means no scaling
    use_autocast = vae.config.norm_float16 and z_test.is_cuda
    with torch.no_grad(), torch.amp.autocast(device_type=z_test.device.type, enabled=use_autocast):
        x_hat_wrong = vae.model.decode_stage_2_outputs(z_wrong_scale)
    x_hat_wrong = x_hat_wrong.float()
    ssim_wrong = compute_ssim_3d(x_test.cpu(), x_hat_wrong.cpu())

    sf_matters = ssim_correct > ssim_wrong
    results["wrong_scale_factor"] = {
        "ssim_correct_sf": ssim_correct,
        "ssim_wrong_sf": ssim_wrong,
        "correct_sf": vae.config.scale_factor,
        "wrong_sf": 1.0,
        "pass": sf_matters,
    }
    logger.info(
        "  Scale factor: correct sf SSIM=%.4f, wrong sf SSIM=%.4f, matters=%s",
        ssim_correct,
        ssim_wrong,
        sf_matters,
    )
    del x_test, z_test, x_hat_correct, x_hat_wrong

    # 6. Encoding stochasticity
    logger.info("Negative control: Encoding stochasticity")
    x_determ = torch.randn(shape, device=device).clamp(0.0, 1.0)

    # Seeded: should be deterministic
    torch.manual_seed(42)
    z1 = vae.encode(x_determ)
    torch.manual_seed(42)
    z2 = vae.encode(x_determ)
    seeded_match = bool(torch.allclose(z1, z2, atol=1e-6))

    # Also get (z_mu, z_sigma) via model.encode() to document posterior
    with torch.no_grad():
        use_autocast = vae.config.norm_float16 and x_determ.is_cuda
        with torch.amp.autocast(device_type=x_determ.device.type, enabled=use_autocast):
            z_mu, z_sigma = vae.model.encode(x_determ)
        z_mu = z_mu.float()
        z_sigma = z_sigma.float()
    mean_sigma = z_sigma.mean().item()

    # Unseeded: should differ (stochastic sampling)
    z3 = vae.encode(x_determ)
    z4 = vae.encode(x_determ)
    unseeded_differ = not torch.allclose(z3, z4, atol=1e-6)

    results["encoding_stochasticity"] = {
        "seeded_match": seeded_match,
        "unseeded_differ": unseeded_differ,
        "mean_posterior_sigma": mean_sigma,
        "pass": seeded_match and unseeded_differ,
        "note": (
            "encode_stage_2_inputs calls sampling(z_mu, z_sigma) with torch.randn_like. "
            "model.encode() returns (z_mu, z_sigma) for deterministic access."
        ),
    }
    logger.info(
        "  Stochasticity: seeded_match=%s, unseeded_differ=%s, mean_sigma=%.6f",
        seeded_match,
        unseeded_differ,
        mean_sigma,
    )
    del x_determ, z1, z2, z3, z4, z_mu, z_sigma

    torch.cuda.empty_cache()

    all_pass = all(r["pass"] for r in results.values())
    results["all_pass"] = all_pass
    logger.info("Negative controls: %s", "ALL PASS" if all_pass else "SOME FAILED")
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_enhanced_html_report(
    metrics: list[dict],
    mean_ssim: float,
    mean_psnr: float,
    elapsed: float,
    negative_controls: dict,
    latent_stats: dict,
    figures_dir: Path,
    output_path: Path,
) -> None:
    """Write a rich HTML verification report with embedded figure references.

    Args:
        metrics: Per-volume metrics dicts.
        mean_ssim: Mean SSIM across all volumes.
        mean_psnr: Mean PSNR across all volumes.
        elapsed: Total processing time in seconds.
        negative_controls: Results from ``run_negative_controls``.
        latent_stats: Per-channel latent statistics.
        figures_dir: Directory containing generated figures.
        output_path: Path to write the HTML file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ssim_status = "PASS" if mean_ssim > 0.90 else "FAIL"
    psnr_status = "PASS" if mean_psnr > 30.0 else "FAIL"
    overall = "ALL PASS" if ssim_status == "PASS" and psnr_status == "PASS" else "FAIL"
    status_color = "#2ecc71" if overall == "ALL PASS" else "#e74c3c"

    # Figure paths relative to HTML location
    fig_rel = Path(figures_dir).relative_to(output_path.parent)

    # Per-volume rows
    per_vol_rows = ""
    for m in metrics:
        ds = m.get("dataset", "")
        per_vol_rows += f"<tr><td>{ds}</td><td>{m['filename']}</td><td>{m['ssim']:.4f}</td><td>{m['psnr']:.2f}</td></tr>\n"

    # Reconstruction figures
    recon_figures_html = ""
    recon_pngs = sorted(figures_dir.glob("reconstruction_*.png"))
    for png in recon_pngs:
        rel = png.relative_to(output_path.parent)
        name = png.stem.replace("reconstruction_", "")
        recon_figures_html += f'<figure><img src="{rel}" alt="{name}" style="max-width:100%"><figcaption>{name}</figcaption></figure>\n'

    # Negative controls table
    nc_rows = ""
    for name, result in negative_controls.items():
        if name == "all_pass":
            continue
        status = "PASS" if result["pass"] else "FAIL"
        color = "#2ecc71" if result["pass"] else "#e74c3c"
        detail = ", ".join(f"{k}={v}" for k, v in result.items() if k not in ("pass", "note"))
        nc_rows += f'<tr><td>{name}</td><td>{detail}</td><td style="color:{color};font-weight:bold">{status}</td></tr>\n'

    # Latent stats table
    latent_rows = ""
    for ch_key, stats in latent_stats.get("per_channel", {}).items():
        latent_rows += (
            f"<tr><td>{ch_key}</td>"
            f"<td>{stats['mean']:.4f}</td>"
            f"<td>{stats['std']:.4f}</td>"
            f"<td>{stats['min']:.4f}</td>"
            f"<td>{stats['max']:.4f}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Phase 0 — VAE Validation Report</title>
<style>
body {{ font-family: 'Georgia', serif; max-width: 1000px; margin: 2em auto; line-height: 1.5; color: #333; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ color: #555; margin-top: 2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
.status {{ font-size: 1.5em; font-weight: bold; color: {status_color}; }}
figure {{ display: inline-block; margin: 0.5em; text-align: center; }}
figcaption {{ font-size: 0.85em; color: #666; }}
details {{ margin: 1em 0; }}
summary {{ cursor: pointer; font-weight: bold; color: #555; }}
.section {{ margin: 2em 0; padding: 1em; background: #fafafa; border-radius: 6px; }}
</style></head><body>
<h1>Phase 0 — VAE Validation Report</h1>
<p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<p><strong>Volumes:</strong> {len(metrics)} &nbsp;|&nbsp;
<strong>Time:</strong> {elapsed:.1f}s</p>
<p class="status">{overall}</p>

<h2>1. Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
<tr><td>Mean SSIM</td><td>{mean_ssim:.4f}</td><td>&gt; 0.90</td><td>{ssim_status}</td></tr>
<tr><td>Mean PSNR</td><td>{mean_psnr:.2f} dB</td><td>&gt; 30.0</td><td>{psnr_status}</td></tr>
</table>

<h2>2. Reconstruction Quality</h2>
<div class="section">
{recon_figures_html if recon_figures_html else "<p>No reconstruction figures generated.</p>"}
</div>

<h2>3. Metrics Distributions</h2>
<div class="section">
<figure><img src="{fig_rel}/ssim_distribution.png" alt="SSIM Distribution" style="max-width:48%"></figure>
<figure><img src="{fig_rel}/psnr_distribution.png" alt="PSNR Distribution" style="max-width:48%"></figure>
</div>

<h2>4. Error Analysis</h2>
<div class="section">
<figure><img src="{fig_rel}/mean_error_heatmap.png" alt="Mean Error Heatmap" style="max-width:100%"></figure>
<p>Mean absolute reconstruction error averaged across {len(metrics)} volumes.
Brighter regions indicate higher error, typically at tissue boundaries.</p>
</div>

<h2>5. Latent Space</h2>
<div class="section">
<figure><img src="{fig_rel}/latent_histograms.png" alt="Latent Histograms" style="max-width:100%"></figure>
<table>
<tr><th>Channel</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
{latent_rows}
</table>
</div>

<h2>6. Negative Controls</h2>
<div class="section">
<table>
<tr><th>Control</th><th>Details</th><th>Status</th></tr>
{nc_rows}
</table>
</div>

<h2>7. Per-Volume Results</h2>
<details>
<summary>Click to expand ({len(metrics)} volumes)</summary>
<table>
<tr><th>Dataset</th><th>Volume</th><th>SSIM</th><th>PSNR (dB)</th></tr>
{per_vol_rows}
</table>
</details>

</body></html>"""

    output_path.write_text(html)
    logger.info("Enhanced HTML report written to %s", output_path)


def generate_enhanced_markdown_report(
    metrics: list[dict],
    mean_ssim: float,
    mean_psnr: float,
    elapsed: float,
    negative_controls: dict,
    latent_stats: dict,
    output_path: Path,
) -> None:
    """Write an enhanced markdown verification report.

    Args:
        metrics: Per-volume metrics dicts.
        mean_ssim: Mean SSIM across all volumes.
        mean_psnr: Mean PSNR across all volumes.
        elapsed: Total processing time in seconds.
        negative_controls: Results from ``run_negative_controls``.
        latent_stats: Per-channel latent statistics.
        output_path: Path to write the markdown file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ssim_status = "PASS" if mean_ssim > 0.90 else "FAIL"
    psnr_status = "PASS" if mean_psnr > 30.0 else "FAIL"
    overall = "ALL PASS" if ssim_status == "PASS" and psnr_status == "PASS" else "FAIL"

    lines = [
        "# Phase 0 — VAE Validation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Volumes:** {len(metrics)}",
        f"**Total time:** {elapsed:.1f}s",
        f"**Status:** {overall}",
        "",
        "## Summary",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
        f"| Mean SSIM | {mean_ssim:.4f} | > 0.90 | {ssim_status} |",
        f"| Mean PSNR | {mean_psnr:.2f} dB | > 30.0 | {psnr_status} |",
        "",
        "## Latent Space Statistics",
        "",
        "| Channel | Mean | Std | Min | Max |",
        "|---------|------|-----|-----|-----|",
    ]
    for ch_key, stats in latent_stats.get("per_channel", {}).items():
        lines.append(
            f"| {ch_key} | {stats['mean']:.4f} | {stats['std']:.4f} "
            f"| {stats['min']:.4f} | {stats['max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Negative Controls",
            "",
            "| Control | Pass | Details |",
            "|---------|------|---------|",
        ]
    )
    for name, result in negative_controls.items():
        if name == "all_pass":
            continue
        detail = ", ".join(f"{k}={v}" for k, v in result.items() if k not in ("pass", "note"))
        lines.append(f"| {name} | {'PASS' if result['pass'] else 'FAIL'} | {detail} |")

    lines.extend(
        [
            "",
            "## Per-Volume Results",
            "",
            "| Dataset | Volume | SSIM | PSNR (dB) |",
            "|---------|--------|------|-----------|",
        ]
    )
    for m in metrics:
        ds = m.get("dataset", "")
        lines.append(f"| {ds} | {m['filename']} | {m['ssim']:.4f} | {m['psnr']:.2f} |")

    output_path.write_text("\n".join(lines))
    logger.info("Enhanced markdown report written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run VAE validation on FOMO-60K volumes with figures and negative controls."""
    parser = argparse.ArgumentParser(description="MAISI VAE reconstruction validation")
    parser.add_argument("--config", type=str, required=True, help="Path to vae_validation.yaml")
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Directory containing base.yaml and fomo60k.yaml (default: configs/ relative to repo root)",
    )
    args = parser.parse_args()

    # Load merged config (base + fomo60k + vae_validation)
    if args.configs_dir:
        configs_dir = Path(args.configs_dir)
    else:
        configs_dir = Path(__file__).resolve().parents[2] / "configs"
    base_cfg = OmegaConf.load(configs_dir / "base.yaml")
    fomo60k_cfg = OmegaConf.load(configs_dir / "fomo60k.yaml")
    vae_cfg = OmegaConf.load(args.config)
    config = OmegaConf.merge(base_cfg, fomo60k_cfg, vae_cfg)
    OmegaConf.resolve(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build VAE
    vae_config = MAISIVAEConfig.from_omegaconf(config)
    vae = MAISIVAEWrapper(vae_config, device=device)

    # Build preprocessing
    transform = build_mri_preprocessing_from_config(config)

    # Get file list from FOMO-60K
    fomo60k_config = FOMO60KConfig.from_omegaconf(config)
    file_list = get_fomo60k_file_list(fomo60k_config, n_volumes=config.data.n_validation)

    # Output dirs
    figures_dir = Path(config.output.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    latent_stats_dir = Path(config.output.latent_stats_dir)
    latent_stats_dir.mkdir(parents=True, exist_ok=True)

    n_volumes = len(file_list)
    n_channels = config.vae.latent_channels

    # Accumulators for latent stats (Welford online algorithm)
    channel_sum = torch.zeros(n_channels, dtype=torch.float64)
    channel_sum_sq = torch.zeros(n_channels, dtype=torch.float64)
    channel_min = torch.full((n_channels,), float("inf"), dtype=torch.float64)
    channel_max = torch.full((n_channels,), float("-inf"), dtype=torch.float64)
    channel_count = 0  # total voxels per channel

    # Subsampled latent values for histograms
    channel_samples: dict[int, list[np.ndarray]] = {ch: [] for ch in range(n_channels)}

    # Mean error accumulator (running mean in pixel space)
    target_shape = tuple(config.data.target_shape)
    mean_error_accum = np.zeros(target_shape, dtype=np.float64)

    # Per-dataset reconstruction save counter (at least 2 per dataset)
    n_per_dataset = config.output.get("n_save_per_dataset", 2)
    dataset_save_counts: dict[str, int] = {}

    # Run validation
    metrics_list: list[dict] = []
    t_start = time.time()

    for i, data_dict in enumerate(file_list):
        filepath = Path(data_dict["image"])
        filename = filepath.name
        # Extract dataset name: .../FOMO60K/{dataset}/sub_X/ses_Y/t1.nii.gz
        dataset_name = filepath.parent.parent.parent.name
        logger.info("[%d/%d] Processing %s (%s)", i + 1, n_volumes, filename, dataset_name)

        # Preprocess
        processed = transform(data_dict)
        x = processed["image"].unsqueeze(0).to(device)  # (1, 1, 192, 192, 192)

        # Encode
        z = vae.encode(x)  # (1, 4, 48, 48, 48)

        # Accumulate latent stats
        z_cpu = z.squeeze(0).cpu().to(torch.float64)  # (4, 48, 48, 48)
        for ch in range(n_channels):
            ch_data = z_cpu[ch]
            channel_sum[ch] += ch_data.sum()
            channel_sum_sq[ch] += (ch_data**2).sum()
            channel_min[ch] = min(channel_min[ch], ch_data.min())
            channel_max[ch] = max(channel_max[ch], ch_data.max())
            # Subsample for histograms
            flat = ch_data.flatten().numpy()
            indices = np.random.default_rng(seed=i * n_channels + ch).choice(
                len(flat), size=min(_HIST_SUBSAMPLE, len(flat)), replace=False
            )
            channel_samples[ch].append(flat[indices])
        channel_count += z_cpu.shape[1] * z_cpu.shape[2] * z_cpu.shape[3]
        del z_cpu

        # Decode
        x_hat = vae.decode(z)

        # Metrics
        psnr_val = compute_psnr(x, x_hat)
        ssim_val = compute_ssim_3d(x.cpu(), x_hat.cpu())
        metrics_list.append(
            {
                "filename": filename,
                "dataset": dataset_name,
                "ssim": ssim_val,
                "psnr": psnr_val,
            }
        )
        logger.info("  SSIM=%.4f  PSNR=%.2f dB", ssim_val, psnr_val)

        # Accumulate mean absolute error
        error_np = (x - x_hat).abs().squeeze().cpu().float().numpy()
        mean_error_accum += error_np / n_volumes

        # Save reconstruction examples: at least n_per_dataset per dataset
        ds_count = dataset_save_counts.get(dataset_name, 0)
        if ds_count < n_per_dataset:
            dataset_save_counts[dataset_name] = ds_count + 1
            recon_dir = Path(config.output.reconstructions_dir)
            recon_dir.mkdir(parents=True, exist_ok=True)
            # Use subject ID for readable names
            subject_id = filepath.parent.parent.name
            recon_np = x_hat.clamp(0.0, 1.0).squeeze().cpu().float().numpy()
            nii = nib.Nifti1Image(recon_np, affine=np.eye(4))
            out_path = recon_dir / f"recon_{dataset_name}_{subject_id}.nii.gz"
            nib.save(nii, str(out_path))
            logger.info("  Saved reconstruction to %s", out_path)

            # Reconstruction comparison figure (center-cropped)
            orig_np = x.squeeze().cpu().float().numpy()
            vol_label = f"{dataset_name} / {subject_id}"
            plot_reconstruction_comparison(
                orig_np,
                recon_np,
                vol_label,
                figures_dir / f"reconstruction_{dataset_name}_{subject_id}",
                crop_frac=0.75,
            )

        # Free VRAM
        del x, x_hat, z
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Compute final latent statistics
    # ------------------------------------------------------------------
    logger.info("Computing final latent statistics...")
    latent_stats: dict = {"per_channel": {}}
    for ch in range(n_channels):
        mean = (channel_sum[ch] / channel_count).item()
        var = (channel_sum_sq[ch] / channel_count - mean**2).item()
        std = var**0.5
        latent_stats["per_channel"][f"channel_{ch}"] = {
            "mean": mean,
            "std": std,
            "min": channel_min[ch].item(),
            "max": channel_max[ch].item(),
        }
        logger.info(
            "  Channel %d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            ch,
            mean,
            std,
            channel_min[ch].item(),
            channel_max[ch].item(),
        )

    # Merge subsampled values for histograms
    merged_samples = {ch: np.concatenate(channel_samples[ch]) for ch in range(n_channels)}

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    logger.info("Generating figures...")

    # SSIM & PSNR distributions
    ssim_values = [m["ssim"] for m in metrics_list]
    psnr_values = [m["psnr"] for m in metrics_list]
    mean_ssim = float(np.mean(ssim_values))
    mean_psnr = float(np.mean(psnr_values))
    logger.info("Mean SSIM=%.4f  Mean PSNR=%.2f dB  Time=%.1fs", mean_ssim, mean_psnr, elapsed)

    plot_metrics_distribution(ssim_values, "SSIM", 0.90, figures_dir / "ssim_distribution")
    plot_metrics_distribution(psnr_values, "PSNR (dB)", 30.0, figures_dir / "psnr_distribution")

    # Latent histograms
    plot_latent_histograms(
        merged_samples,
        {ch: latent_stats["per_channel"][f"channel_{ch}"] for ch in range(n_channels)},
        figures_dir / "latent_histograms",
    )

    # Mean error heatmap
    plot_error_heatmap(mean_error_accum.astype(np.float32), figures_dir / "mean_error_heatmap")

    # ------------------------------------------------------------------
    # Negative controls
    # ------------------------------------------------------------------
    logger.info("Running negative controls...")
    negative_controls = run_negative_controls(vae, device)

    # ------------------------------------------------------------------
    # Write metrics.json (backward compatible + extended)
    # ------------------------------------------------------------------
    metrics_dir = Path(config.output.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    results = {
        "mean_ssim": mean_ssim,
        "mean_psnr": mean_psnr,
        "n_volumes": len(metrics_list),
        "elapsed_seconds": elapsed,
        "per_volume": metrics_list,
        "latent_statistics": latent_stats,
        "negative_controls": negative_controls,
    }
    metrics_path.write_text(json.dumps(results, indent=2))
    logger.info("Metrics written to %s", metrics_path)

    # Write separate latent_stats.json
    latent_stats_path = latent_stats_dir / "latent_stats.json"
    latent_stats_path.write_text(json.dumps(latent_stats, indent=2))
    logger.info("Latent stats written to %s", latent_stats_path)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    generate_enhanced_markdown_report(
        metrics_list,
        mean_ssim,
        mean_psnr,
        elapsed,
        negative_controls,
        latent_stats,
        Path(config.output.report_md),
    )
    generate_enhanced_html_report(
        metrics_list,
        mean_ssim,
        mean_psnr,
        elapsed,
        negative_controls,
        latent_stats,
        figures_dir,
        Path(config.output.report_html),
    )

    # Final verdict
    if mean_ssim > 0.90 and mean_psnr > 30.0:
        logger.info("Phase 0 VAE validation: PASSED")
    else:
        logger.warning(
            "Phase 0 VAE validation: FAILED (SSIM=%.4f, PSNR=%.2f)", mean_ssim, mean_psnr
        )


if __name__ == "__main__":
    main()
