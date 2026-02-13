"""CLI script for encoding all FOMO-60K volumes through the frozen MAISI VAE.

Produces ``.pt`` latent files, per-channel statistics, and rich HTML/Markdown
reports. Designed for both local (batch_size=1) and cluster (multi-worker)
execution.

Usage:
    ~/.conda/envs/neuromf/bin/python experiments/cli/encode_dataset.py \
        --config configs/encode_dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.logging import RichHandler
from tqdm import tqdm

from neuromf.data.fomo60k import FOMO60KConfig, get_fomo60k_file_list
from neuromf.data.mri_preprocessing import build_mri_preprocessing_from_config
from neuromf.metrics.ssim_psnr import compute_psnr, compute_ssim_3d
from neuromf.utils.latent_stats import (
    LatentStatsAccumulator,
    save_latent_stats,
)
from neuromf.utils.visualisation import (
    plot_channel_stats_bar,
    plot_correlation_heatmap,
    plot_latent_histograms,
    plot_reconstruction_comparison,
)
from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Voxels to subsample per channel per volume for histograms
_HIST_SUBSAMPLE = 4096


def _build_latent_filename(filepath: Path) -> str:
    """Build a unique .pt filename from the FOMO-60K path hierarchy.

    Convention: ``{dataset}_{participant_id}_{session_id}.pt``
    e.g. ``PT005_IXI_sub_002_ses_1.pt``

    Args:
        filepath: Absolute path to a NIfTI file within FOMO-60K.

    Returns:
        Filename string (no directory).
    """
    session_id = filepath.parent.name
    participant_id = filepath.parent.parent.name
    dataset_name = filepath.parent.parent.parent.name
    return f"{dataset_name}_{participant_id}_{session_id}.pt"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_html_report(
    encoding_log: list[dict],
    stats: dict,
    round_trip_results: list[dict],
    elapsed: float,
    figures_dir: Path,
    output_path: Path,
) -> None:
    """Write a rich HTML verification report.

    Args:
        encoding_log: Per-volume encoding results.
        stats: Latent statistics dict.
        round_trip_results: SSIM values for round-trip checked volumes.
        elapsed: Total time in seconds.
        figures_dir: Directory containing generated figures.
        output_path: HTML output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_success = sum(1 for e in encoding_log if e["status"] == "success")
    n_failed = sum(1 for e in encoding_log if e["status"] == "failed")
    n_total = len(encoding_log)

    overall = "ALL PASS" if n_failed == 0 else "FAILURES DETECTED"
    status_color = "#2ecc71" if n_failed == 0 else "#e74c3c"

    fig_rel = Path(figures_dir).relative_to(output_path.parent)

    # Per-channel stats table
    ch_rows = ""
    for ch_key, ch_stats in stats.get("per_channel", {}).items():
        ch_rows += (
            f"<tr><td>{ch_key}</td>"
            f"<td>{ch_stats['mean']:.4f}</td>"
            f"<td>{ch_stats['std']:.4f}</td>"
            f"<td>{ch_stats['skewness']:.4f}</td>"
            f"<td>{ch_stats['kurtosis']:.4f}</td>"
            f"<td>{ch_stats['min']:.4f}</td>"
            f"<td>{ch_stats['max']:.4f}</td></tr>\n"
        )

    # Round-trip quality table
    rt_rows = ""
    for rt in round_trip_results:
        rt_rows += (
            f"<tr><td>{rt['dataset']}</td>"
            f"<td>{rt['filename']}</td>"
            f"<td>{rt['ssim']:.4f}</td>"
            f"<td>{rt['psnr']:.2f}</td></tr>\n"
        )

    # Reconstruction figures
    recon_html = ""
    recon_pngs = sorted(figures_dir.glob("reconstruction_*.png"))
    for png in recon_pngs:
        rel = png.relative_to(output_path.parent)
        name = png.stem.replace("reconstruction_", "")
        recon_html += (
            f'<figure><img src="{rel}" alt="{name}" style="max-width:100%">'
            f"<figcaption>{name}</figcaption></figure>\n"
        )

    # Encoding log table (expandable)
    log_rows = ""
    for entry in encoding_log:
        status_str = entry["status"]
        color = "#2ecc71" if status_str == "success" else "#e74c3c"
        t = entry.get("encoding_time_s", 0)
        log_rows += (
            f"<tr><td>{entry.get('dataset', '')}</td>"
            f"<td>{entry.get('filename', '')}</td>"
            f"<td>{t:.1f}s</td>"
            f'<td style="color:{color};font-weight:bold">{status_str}</td></tr>\n'
        )

    # Disk usage estimate
    disk_mb = n_success * 0.5  # ~0.5 MB per latent

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Phase 1 — Latent Pre-computation Report</title>
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
<h1>Phase 1 — Latent Pre-computation Report</h1>
<p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<p class="status">{overall}</p>

<h2>1. Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total volumes</td><td>{n_total}</td></tr>
<tr><td>Encoded successfully</td><td>{n_success}</td></tr>
<tr><td>Failed</td><td>{n_failed}</td></tr>
<tr><td>Total time</td><td>{elapsed:.1f}s ({elapsed / 3600:.1f}h)</td></tr>
<tr><td>Disk usage (approx)</td><td>{disk_mb:.0f} MB</td></tr>
</table>

<h2>2. Latent Statistics</h2>
<div class="section">
<table>
<tr><th>Channel</th><th>Mean</th><th>Std</th><th>Skewness</th><th>Kurtosis</th><th>Min</th><th>Max</th></tr>
{ch_rows}
</table>
<figure><img src="{fig_rel}/channel_stats_bar.png" alt="Channel Stats" style="max-width:60%"></figure>
</div>

<h2>3. Latent Distributions</h2>
<div class="section">
<figure><img src="{fig_rel}/latent_histograms.png" alt="Latent Histograms" style="max-width:100%"></figure>
</div>

<h2>4. Cross-Channel Correlation</h2>
<div class="section">
<figure><img src="{fig_rel}/correlation_heatmap.png" alt="Correlation Heatmap" style="max-width:50%"></figure>
</div>

<h2>5. Round-Trip Quality</h2>
<div class="section">
<table>
<tr><th>Dataset</th><th>Volume</th><th>SSIM</th><th>PSNR (dB)</th></tr>
{rt_rows}
</table>
{recon_html if recon_html else "<p>No reconstruction figures generated.</p>"}
</div>

<h2>6. Encoding Log</h2>
<details>
<summary>Click to expand ({n_total} volumes)</summary>
<table>
<tr><th>Dataset</th><th>Volume</th><th>Time</th><th>Status</th></tr>
{log_rows}
</table>
</details>

</body></html>"""

    output_path.write_text(html)
    logger.info("HTML report written to %s", output_path)


def generate_markdown_report(
    encoding_log: list[dict],
    stats: dict,
    round_trip_results: list[dict],
    elapsed: float,
    output_path: Path,
) -> None:
    """Write a markdown verification report.

    Args:
        encoding_log: Per-volume encoding results.
        stats: Latent statistics dict.
        round_trip_results: SSIM values for round-trip checked volumes.
        elapsed: Total time in seconds.
        output_path: Markdown output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_success = sum(1 for e in encoding_log if e["status"] == "success")
    n_failed = sum(1 for e in encoding_log if e["status"] == "failed")
    n_total = len(encoding_log)

    overall = "ALL PASS" if n_failed == 0 else "FAILURES DETECTED"

    lines = [
        "# Phase 1 — Latent Pre-computation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Status:** {overall}",
        f"**Volumes:** {n_success}/{n_total} encoded, {n_failed} failed",
        f"**Total time:** {elapsed:.1f}s ({elapsed / 3600:.1f}h)",
        "",
        "## Latent Statistics",
        "",
        "| Channel | Mean | Std | Skewness | Kurtosis | Min | Max |",
        "|---------|------|-----|----------|----------|-----|-----|",
    ]
    for ch_key, ch_stats in stats.get("per_channel", {}).items():
        lines.append(
            f"| {ch_key} | {ch_stats['mean']:.4f} | {ch_stats['std']:.4f} "
            f"| {ch_stats['skewness']:.4f} | {ch_stats['kurtosis']:.4f} "
            f"| {ch_stats['min']:.4f} | {ch_stats['max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Round-Trip Quality",
            "",
            "| Dataset | Volume | SSIM | PSNR (dB) |",
            "|---------|--------|------|-----------|",
        ]
    )
    for rt in round_trip_results:
        lines.append(
            f"| {rt['dataset']} | {rt['filename']} | {rt['ssim']:.4f} | {rt['psnr']:.2f} |"
        )

    output_path.write_text("\n".join(lines))
    logger.info("Markdown report written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Encode all FOMO-60K volumes and generate reports."""
    parser = argparse.ArgumentParser(description="FOMO-60K latent pre-computation (Phase 1)")
    parser.add_argument("--config", type=str, required=True, help="Path to encode_dataset.yaml")
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Directory containing base.yaml and fomo60k.yaml (default: configs/ relative to repo root)",
    )
    args = parser.parse_args()

    # Load merged config
    if args.configs_dir:
        configs_dir = Path(args.configs_dir)
    else:
        configs_dir = Path(__file__).resolve().parents[2] / "configs"
    base_cfg = OmegaConf.load(configs_dir / "base.yaml")
    fomo60k_cfg = OmegaConf.load(configs_dir / "fomo60k.yaml")
    encode_cfg = OmegaConf.load(args.config)
    config = OmegaConf.merge(base_cfg, fomo60k_cfg, encode_cfg)
    OmegaConf.resolve(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build VAE
    vae_config = MAISIVAEConfig.from_omegaconf(config)
    vae = MAISIVAEWrapper(vae_config, device=device)

    # Build preprocessing
    transform = build_mri_preprocessing_from_config(config)

    # Get file list
    fomo60k_config = FOMO60KConfig.from_omegaconf(config)
    file_list = get_fomo60k_file_list(fomo60k_config)
    n_volumes = len(file_list)
    logger.info("Total volumes to encode: %d", n_volumes)

    # Output directories
    latent_dir = Path(config.output.latent_dir)
    latent_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(config.output.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_channels = config.vae.latent_channels
    seed = config.encoding.seed
    n_save_per_dataset = config.encoding.n_save_per_dataset
    n_round_trip_check = config.data.n_round_trip_check

    # Online stats accumulator
    acc = LatentStatsAccumulator(n_channels=n_channels)

    # Subsampled latent values for histograms
    channel_samples: dict[int, list[np.ndarray]] = {ch: [] for ch in range(n_channels)}

    # Per-dataset reconstruction save counter
    dataset_save_counts: dict[str, int] = {}

    # Encoding log and round-trip results
    encoding_log: list[dict] = []
    round_trip_results: list[dict] = []
    n_round_trip_done = 0

    t_start = time.time()

    for i, data_dict in enumerate(tqdm(file_list, desc="Encoding", unit="vol")):
        filepath = Path(data_dict["image"])
        filename = filepath.name
        dataset_name = filepath.parent.parent.parent.name
        latent_filename = _build_latent_filename(filepath)

        entry: dict = {
            "index": i,
            "dataset": dataset_name,
            "filename": latent_filename,
            "source_path": str(filepath),
            "status": "pending",
            "encoding_time_s": 0.0,
        }

        t_vol = time.time()
        try:
            # Preprocess
            processed = transform(data_dict)
            x = processed["image"].unsqueeze(0).to(device)  # (1, 1, 192, 192, 192)

            # Deterministic encoding
            torch.manual_seed(seed + i)
            z = vae.encode(x)  # (1, 4, 48, 48, 48)
            z_squeezed = z.squeeze(0).cpu()  # (4, 48, 48, 48)

            # Save .pt file
            save_data = {
                "z": z_squeezed,
                "metadata": {
                    "subject_id": filepath.parent.parent.name,
                    "session_id": filepath.parent.name,
                    "dataset": dataset_name,
                    "source_path": str(filepath),
                    "latent_filename": latent_filename,
                },
            }
            torch.save(save_data, latent_dir / latent_filename)

            # Update online stats
            acc.update(z_squeezed)

            # Subsample for histograms
            z_np = z_squeezed.numpy()
            rng = np.random.default_rng(seed=i)
            for ch in range(n_channels):
                flat = z_np[ch].ravel()
                indices = rng.choice(len(flat), size=min(_HIST_SUBSAMPLE, len(flat)), replace=False)
                channel_samples[ch].append(flat[indices])

            # Round-trip check for first n_round_trip_check volumes
            if n_round_trip_done < n_round_trip_check:
                x_hat = vae.decode(z)
                ssim_val = compute_ssim_3d(x.cpu(), x_hat.cpu())
                psnr_val = compute_psnr(x, x_hat)
                round_trip_results.append(
                    {
                        "dataset": dataset_name,
                        "filename": latent_filename,
                        "ssim": ssim_val,
                        "psnr": psnr_val,
                    }
                )
                n_round_trip_done += 1
                logger.info(
                    "  Round-trip SSIM=%.4f PSNR=%.2f (%s)", ssim_val, psnr_val, latent_filename
                )
                del x_hat

            # Save reconstruction examples per dataset
            ds_count = dataset_save_counts.get(dataset_name, 0)
            if ds_count < n_save_per_dataset:
                dataset_save_counts[dataset_name] = ds_count + 1
                x_hat_recon = vae.decode(z)
                orig_np = x.squeeze().cpu().float().numpy()
                recon_np = x_hat_recon.clamp(0.0, 1.0).squeeze().cpu().float().numpy()
                subject_id = filepath.parent.parent.name
                plot_reconstruction_comparison(
                    orig_np,
                    recon_np,
                    f"{dataset_name} / {subject_id}",
                    figures_dir / f"reconstruction_{dataset_name}_{subject_id}",
                    crop_frac=0.75,
                )
                del x_hat_recon

            entry["status"] = "success"
            entry["encoding_time_s"] = time.time() - t_vol

            # Free GPU memory
            del x, z, z_squeezed
            torch.cuda.empty_cache()

        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            entry["traceback"] = traceback.format_exc()
            entry["encoding_time_s"] = time.time() - t_vol
            logger.error("FAILED [%d] %s: %s", i, latent_filename, exc)

        encoding_log.append(entry)

    elapsed = time.time() - t_start
    n_success = sum(1 for e in encoding_log if e["status"] == "success")
    n_failed = sum(1 for e in encoding_log if e["status"] == "failed")
    logger.info(
        "Encoding complete: %d/%d success, %d failed, %.1fs (%.1fh)",
        n_success,
        n_volumes,
        n_failed,
        elapsed,
        elapsed / 3600,
    )

    # ------------------------------------------------------------------
    # Compute final statistics
    # ------------------------------------------------------------------
    logger.info("Finalising latent statistics...")
    stats = acc.finalize()
    stats["n_files"] = n_success
    save_latent_stats(stats, Path(config.output.stats_path))

    for ch in range(n_channels):
        ch_stats = stats["per_channel"][f"channel_{ch}"]
        logger.info(
            "  Channel %d: mean=%.4f, std=%.4f, skew=%.4f, kurt=%.4f",
            ch,
            ch_stats["mean"],
            ch_stats["std"],
            ch_stats["skewness"],
            ch_stats["kurtosis"],
        )

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    logger.info("Generating figures...")

    # Latent histograms
    merged_samples = {ch: np.concatenate(channel_samples[ch]) for ch in range(n_channels)}
    plot_latent_histograms(
        merged_samples,
        {ch: stats["per_channel"][f"channel_{ch}"] for ch in range(n_channels)},
        figures_dir / "latent_histograms",
    )

    # Channel stats bar chart
    plot_channel_stats_bar(stats, figures_dir / "channel_stats_bar")

    # Correlation heatmap
    corr = np.array(stats["cross_channel_correlation"])
    plot_correlation_heatmap(
        corr,
        [f"Ch {c}" for c in range(n_channels)],
        figures_dir / "correlation_heatmap",
    )

    # ------------------------------------------------------------------
    # Write encoding log
    # ------------------------------------------------------------------
    encoding_log_path = Path(config.output.encoding_log)
    encoding_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        "n_total": n_volumes,
        "n_success": n_success,
        "n_failed": n_failed,
        "elapsed_seconds": elapsed,
        "entries": encoding_log,
    }
    encoding_log_path.write_text(json.dumps(log_data, indent=2, default=str))
    logger.info("Encoding log written to %s", encoding_log_path)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    generate_markdown_report(
        encoding_log,
        stats,
        round_trip_results,
        elapsed,
        Path(config.output.report_md),
    )
    generate_html_report(
        encoding_log,
        stats,
        round_trip_results,
        elapsed,
        figures_dir,
        Path(config.output.report_html),
    )

    # Final verdict
    if n_failed == 0:
        logger.info("Phase 1 latent pre-computation: PASSED (%d volumes)", n_success)
    else:
        logger.warning("Phase 1 latent pre-computation: %d FAILURES out of %d", n_failed, n_volumes)


if __name__ == "__main__":
    main()
