"""Post-hoc comparison of x-prediction vs u-prediction ablation arms.

Loads training_summary.json and sample_archive.pt from both arms, generates
IEEE-compliant comparison figures and a markdown report. Designed to be run
after both SLURM jobs have completed.

Usage:
    python experiments/ablations/xpred_vs_upred/compare_arms.py \
        --results-dir /path/to/results/ablations \
        --output-dir  /path/to/results/ablations/xpred_vs_upred_report

Directory structure expected per arm::

    <results-dir>/<arm_dir>/
        diagnostics/aggregate_results/training_summary.json
        samples/sample_archive.pt  (optional)
        checkpoints/               (not used here)
        logs/                      (TensorBoard; not used here)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Add project root so we can import experiments.utils
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.utils.settings import (
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Arm definitions — must match directory names from launch.sh
# ============================================================================

ARM_CONFIGS: dict[str, dict[str, str]] = {
    "xpred": {
        "dir": "xpred_exact_jvp",
        "label": "x-pred + exact JVP",
        "color": PAUL_TOL_BRIGHT["blue"],
        "marker": "o",
    },
    "upred": {
        "dir": "upred_fd_jvp",
        "label": "u-pred + FD-JVP",
        "color": PAUL_TOL_BRIGHT["red"],
        "marker": "s",
    },
}

# ============================================================================
# Data loading
# ============================================================================


def load_training_summary(arm_dir: Path) -> list[dict[str, Any]]:
    """Load the per-epoch training summary JSON from an arm directory.

    The diagnostics callback writes to:
    ``<arm_dir>/diagnostics/aggregate_results/training_summary.json``

    Args:
        arm_dir: Path to the arm results directory.

    Returns:
        List of per-epoch metric dictionaries.
    """
    primary = arm_dir / "diagnostics" / "aggregate_results" / "training_summary.json"
    if primary.exists():
        with open(primary) as f:
            return json.load(f)

    # Fallback: search common alternative locations
    for candidate in [
        arm_dir / "logs" / "training_summary.json",
        arm_dir / "training_summary.json",
    ]:
        if candidate.exists():
            logger.info("Using fallback training summary at %s", candidate)
            with open(candidate) as f:
                return json.load(f)

    logger.warning("No training summary found in %s", arm_dir)
    return []


def load_sample_archive(arm_dir: Path) -> dict[str, Any] | None:
    """Load the sample archive .pt file if it exists.

    Args:
        arm_dir: Path to the arm results directory.

    Returns:
        Archive dict or None.
    """
    import torch

    archive_path = arm_dir / "samples" / "sample_archive.pt"
    if not archive_path.exists():
        return None

    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    logger.info("Loaded sample archive with %d epochs", len(archive.get("epochs", [])))
    return archive


# ============================================================================
# Metric extraction helpers
# ============================================================================


def _safe_array(entries: list[dict], key: str, nested: str | None = None) -> np.ndarray:
    """Extract a metric as a numpy array, handling missing/nested keys.

    Args:
        entries: List of per-epoch summary dicts.
        key: Top-level key in each entry.
        nested: If not None, extract ``entry[key][nested]``.

    Returns:
        Array of values (NaN for missing entries).
    """
    vals = []
    for e in entries:
        v = e.get(key)
        if v is None:
            vals.append(float("nan"))
        elif nested is not None:
            vals.append(float(v.get(nested, float("nan"))) if isinstance(v, dict) else float("nan"))
        else:
            vals.append(float(v))
    return np.array(vals)


def extract_epochs(entries: list[dict]) -> np.ndarray:
    """Extract epoch numbers."""
    return np.array([e["epoch"] for e in entries])


# ============================================================================
# Plotting helpers
# ============================================================================


def _plot_metric_pair(
    ax: plt.Axes,
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    ylabel: str,
    title: str,
    *,
    logy: bool = False,
    legend: bool = True,
) -> None:
    """Plot one metric for both arms on a single axes.

    Args:
        ax: Target axes.
        data: Mapping arm_id -> (epochs, values).
        ylabel: Y-axis label.
        title: Axes title.
        logy: Use log scale on y-axis.
        legend: Show legend.
    """
    for arm_id, (epochs, values) in data.items():
        cfg = ARM_CONFIGS[arm_id]
        mask = np.isfinite(values)
        if not mask.any():
            continue
        ax.plot(
            epochs[mask],
            values[mask],
            label=cfg["label"],
            color=cfg["color"],
            linewidth=PLOT_SETTINGS["line_width"],
            alpha=0.85,
            marker=cfg["marker"],
            markersize=2.5,
            markevery=max(1, mask.sum() // 15),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=PLOT_SETTINGS["axes_titlesize"])
    if logy:
        ax.set_yscale("log")
    if legend:
        ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])


def _savefig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure as both PNG and PDF."""
    fig.savefig(
        output_dir / f"{name}.png",
        dpi=PLOT_SETTINGS["dpi_print"],
        bbox_inches="tight",
    )
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s.{png,pdf}", name)


# ============================================================================
# Figure generators
# ============================================================================


def plot_loss_overview(
    summaries: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Figure 1: Loss overview (2x2 grid).

    Panels: (a) total weighted loss, (b) raw loss, (c) FM component, (d) MF component.
    """
    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", 0.8))

    specs = [
        ("loss_mean", None, "Weighted Loss", "Weighted Loss (iMF)", False),
        ("raw_loss_mean", None, "Raw Loss", "Raw Loss (pre-adaptive)", True),
        ("raw_loss_fm_mean", None, "FM Loss", "FM Component (raw)", True),
        ("raw_loss_mf_mean", None, "MF Loss", "MF Component (raw)", True),
    ]

    for ax, (key, nested, ylabel, title, logy) in zip(axes.flat, specs):
        arm_data = {}
        for arm_id, entries in summaries.items():
            epochs = extract_epochs(entries)
            values = _safe_array(entries, key, nested)
            arm_data[arm_id] = (epochs, values)
        _plot_metric_pair(ax, arm_data, ylabel, title, logy=logy)

    # Panel labels
    for ax, label in zip(axes.flat, "abcd"):
        ax.text(
            -0.12,
            1.05,
            f"({label})",
            transform=ax.transAxes,
            fontsize=PLOT_SETTINGS["panel_label_fontsize"],
            fontweight="bold",
            va="bottom",
        )

    fig.suptitle(
        "x-Prediction vs u-Prediction: Training Loss",
        fontweight="bold",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "fig1_loss_overview")


def plot_numerical_stability(
    summaries: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Figure 2: Numerical stability diagnostics (2x2 grid).

    Panels: (a) JVP norm, (b) compound V vs target V norm, (c) V/target ratio, (d) v_tilde norm.
    This is the key diagnostic — JVP explosion was the root cause of v2_baseline failure.
    """
    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", 0.8))

    # (a) JVP norm
    arm_data = {}
    for arm_id, entries in summaries.items():
        epochs = extract_epochs(entries)
        jvp = _safe_array(entries, "velocity_norms", "jvp_norm")
        arm_data[arm_id] = (epochs, jvp)
    _plot_metric_pair(axes[0, 0], arm_data, "JVP Norm", "(a) JVP Norm (du/dt)", logy=True)

    # (b) Compound V and target V norms (overlay per arm)
    ax = axes[0, 1]
    for arm_id, entries in summaries.items():
        cfg = ARM_CONFIGS[arm_id]
        epochs = extract_epochs(entries)
        cv = _safe_array(entries, "velocity_norms", "compound_v_norm")
        tv = _safe_array(entries, "velocity_norms", "target_v_norm")

        mask_cv = np.isfinite(cv)
        mask_tv = np.isfinite(tv)
        if mask_cv.any():
            ax.plot(
                epochs[mask_cv],
                cv[mask_cv],
                label=f"{cfg['label']} (compound V)",
                color=cfg["color"],
                linewidth=PLOT_SETTINGS["line_width"],
                linestyle="-",
            )
        if mask_tv.any():
            ax.plot(
                epochs[mask_tv],
                tv[mask_tv],
                label=f"{cfg['label']} (target)",
                color=cfg["color"],
                linewidth=PLOT_SETTINGS["line_width"],
                linestyle="--",
                alpha=0.6,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Norm")
    ax.set_title("(b) Compound V vs Target V Norm")
    ax.set_yscale("log")
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"] - 1, ncol=1)

    # (c) Compound V / Target V ratio
    arm_data = {}
    for arm_id, entries in summaries.items():
        epochs = extract_epochs(entries)
        cv = _safe_array(entries, "velocity_norms", "compound_v_norm")
        tv = _safe_array(entries, "velocity_norms", "target_v_norm")
        ratio = np.where(tv > 0, cv / tv, np.nan)
        arm_data[arm_id] = (epochs, ratio)
    ax_c = axes[1, 0]
    _plot_metric_pair(ax_c, arm_data, "Ratio", "(c) Compound V / Target V", logy=True)
    ax_c.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label="Ideal (1.0)")
    ax_c.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    # (d) v_tilde norm
    arm_data = {}
    for arm_id, entries in summaries.items():
        epochs = extract_epochs(entries)
        vt = _safe_array(entries, "velocity_norms", "v_tilde_norm")
        arm_data[arm_id] = (epochs, vt)
    _plot_metric_pair(
        axes[1, 1], arm_data, "Norm", "(d) $\\tilde{v}$ Norm (model instant velocity)"
    )

    fig.suptitle(
        "Numerical Stability: JVP and Velocity Norms",
        fontweight="bold",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "fig2_numerical_stability")


def plot_self_consistency(
    summaries: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Figure 3: MeanFlow self-consistency metrics (1x3 grid).

    Panels: (a) cosine(V, v_c), (b) cosine(v_tilde, v_c), (c) relative error.
    """
    fig, axes = plt.subplots(1, 3, figsize=get_figure_size("double", 0.4))

    specs = [
        ("cosine_sim_V_vc", "Cosine Sim", "(a) cos(V, $v_c$)"),
        ("cosine_sim_vtilde_vc", "Cosine Sim", "(b) cos($\\tilde{v}$, $v_c$)"),
        ("relative_error", "Relative Error", "(c) Relative Error"),
    ]

    for ax, (key, ylabel, title) in zip(axes, specs):
        arm_data = {}
        for arm_id, entries in summaries.items():
            epochs = extract_epochs(entries)
            values = _safe_array(entries, key)
            arm_data[arm_id] = (epochs, values)
        _plot_metric_pair(ax, arm_data, ylabel, title)

        # Reference lines for cosine similarity
        if "cosine" in key.lower():
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.6, alpha=0.4)
            ax.set_ylim(None, 1.05)

    fig.suptitle(
        "MeanFlow Self-Consistency",
        fontweight="bold",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, output_dir, "fig3_self_consistency")


def plot_training_health(
    summaries: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Figure 4: Training health diagnostics (2x2 grid).

    Panels: (a) gradient norm, (b) relative update norm, (c) clip fraction, (d) EMA divergence.
    """
    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", 0.8))

    specs = [
        ("grad_norm_mean", "Grad Norm", "(a) Gradient Norm", True),
        ("relative_update_norm", "Rel. Update", "(b) Relative Update Norm", True),
        ("grad_clip_fraction", "Clip Fraction", "(c) Gradient Clip Fraction", False),
        ("ema_divergence", "EMA Divergence", "(d) EMA Divergence", True),
    ]

    for ax, (key, ylabel, title, logy) in zip(axes.flat, specs):
        arm_data = {}
        for arm_id, entries in summaries.items():
            epochs = extract_epochs(entries)
            values = _safe_array(entries, key)
            arm_data[arm_id] = (epochs, values)
        _plot_metric_pair(ax, arm_data, ylabel, title, logy=logy)

    fig.suptitle(
        "Training Health: Gradients and Updates",
        fontweight="bold",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, output_dir, "fig4_training_health")


def plot_throughput(
    summaries: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Figure 5: Wall-clock throughput comparison.

    Shows epoch time and cumulative training time.
    """
    fig, axes = plt.subplots(1, 2, figsize=get_figure_size("double", 0.4))

    # (a) Per-epoch time
    arm_data = {}
    for arm_id, entries in summaries.items():
        epochs = extract_epochs(entries)
        times = _safe_array(entries, "epoch_time_sec")
        arm_data[arm_id] = (epochs, times)
    _plot_metric_pair(axes[0], arm_data, "Time (s)", "(a) Epoch Wall Time")

    # (b) Cumulative time (hours)
    for arm_id, entries in summaries.items():
        cfg = ARM_CONFIGS[arm_id]
        times = _safe_array(entries, "epoch_time_sec")
        mask = np.isfinite(times)
        if not mask.any():
            continue
        cumulative_hrs = np.nancumsum(times) / 3600.0
        epochs = extract_epochs(entries)
        axes[1].plot(
            epochs[mask],
            cumulative_hrs[mask],
            label=cfg["label"],
            color=cfg["color"],
            linewidth=PLOT_SETTINGS["line_width"],
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cumulative Time (h)")
    axes[1].set_title("(b) Cumulative Training Time")
    axes[1].legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    fig.suptitle(
        "Training Throughput",
        fontweight="bold",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, output_dir, "fig5_throughput")


def plot_prediction_stats(
    summaries: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Figure 6: Prediction output statistics.

    x-pred arm logs x_hat_stats; u-pred arm logs u_pred_stats. Shows the
    evolution of the model's raw output distribution.
    """
    fig, axes = plt.subplots(1, 4, figsize=get_figure_size("double", 0.35))

    stat_keys = ["mean", "std", "min", "max"]
    stat_labels = ["Mean", "Std", "Min", "Max"]

    # Map arm_id to the correct stats key
    stats_key_map = {
        "xpred": "x_hat_stats",
        "upred": "u_pred_stats",
    }

    for ax, stat_key, stat_label in zip(axes, stat_keys, stat_labels):
        for arm_id, entries in summaries.items():
            cfg = ARM_CONFIGS[arm_id]
            sk = stats_key_map[arm_id]
            epochs = extract_epochs(entries)
            values = _safe_array(entries, sk, stat_key)
            mask = np.isfinite(values)
            if not mask.any():
                continue
            ax.plot(
                epochs[mask],
                values[mask],
                label=cfg["label"],
                color=cfg["color"],
                linewidth=PLOT_SETTINGS["line_width"],
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(stat_label)
        ax.set_title(f"Output {stat_label}")
        ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"] - 1)

    fig.suptitle(
        "Prediction Output Statistics ($\\hat{x}_0$ vs $u$)",
        fontweight="bold",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _savefig(fig, output_dir, "fig6_prediction_stats")


def plot_adaptive_weights(
    summaries: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Figure 7: Adaptive weighting dynamics (1x2 grid).

    The adaptive weight should be ~1.0 when normalization works correctly.
    If it diverges, the loss landscape is being masked.
    """
    fig, axes = plt.subplots(1, 2, figsize=get_figure_size("double", 0.4))

    specs = [
        ("adaptive_weight", "mean", "Weight Mean", "(a) Adaptive Weight Mean"),
        ("adaptive_weight", "std", "Weight Std", "(b) Adaptive Weight Std"),
    ]

    for ax, (key, nested, ylabel, title) in zip(axes, specs):
        arm_data = {}
        for arm_id, entries in summaries.items():
            epochs = extract_epochs(entries)
            values = _safe_array(entries, key, nested)
            arm_data[arm_id] = (epochs, values)
        _plot_metric_pair(ax, arm_data, ylabel, title)

    fig.suptitle(
        "Adaptive Weighting Dynamics",
        fontweight="bold",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, output_dir, "fig7_adaptive_weights")


def plot_validation(
    summaries: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Figure 8: Validation loss (if available).

    Shows val_loss and val_raw_loss for generalization assessment.
    """
    # Check if any arm has validation data
    has_val = any(
        any(e.get("val_loss") is not None for e in entries) for entries in summaries.values()
    )
    if not has_val:
        logger.info("No validation data found; skipping fig8_validation")
        return

    fig, axes = plt.subplots(1, 2, figsize=get_figure_size("double", 0.4))

    specs = [
        ("val_loss", "Val Loss", "(a) Validation Loss (weighted)"),
        ("val_raw_loss", "Val Raw Loss", "(b) Validation Loss (raw)"),
    ]

    for ax, (key, ylabel, title) in zip(axes, specs):
        arm_data = {}
        for arm_id, entries in summaries.items():
            epochs = extract_epochs(entries)
            values = _safe_array(entries, key)
            arm_data[arm_id] = (epochs, values)
        _plot_metric_pair(ax, arm_data, ylabel, title, logy=True)

    fig.suptitle(
        "Validation Loss",
        fontweight="bold",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, output_dir, "fig8_validation")


def plot_sample_quality(
    archives: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Figure 9: Sample quality evolution from sample archives.

    Shows NFE consistency (1-NFE vs multi-step MSE) and latent mean/std
    over training epochs.
    """
    if not archives:
        logger.info("No sample archives found; skipping fig9_sample_quality")
        return

    fig, axes = plt.subplots(1, 3, figsize=get_figure_size("double", 0.4))

    # (a) 1-NFE vs 5-step MSE
    for arm_id, archive in archives.items():
        cfg = ARM_CONFIGS[arm_id]
        epoch_list = sorted(archive.get("epochs", []))
        mse_vals = []
        for ep in epoch_list:
            ep_data = archive.get(f"epoch_{ep:04d}", {})
            nfe_con = ep_data.get("nfe_consistency", {})
            mse = nfe_con.get("mse_1vs5", nfe_con.get("mse_1vs10", float("nan")))
            mse_vals.append(float(mse) if mse is not None else float("nan"))

        mse_arr = np.array(mse_vals)
        ep_arr = np.array(epoch_list)
        mask = np.isfinite(mse_arr)
        if mask.any():
            axes[0].plot(
                ep_arr[mask],
                mse_arr[mask],
                label=cfg["label"],
                color=cfg["color"],
                linewidth=PLOT_SETTINGS["line_width"],
                marker=cfg["marker"],
                markersize=PLOT_SETTINGS["marker_size"],
            )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("(a) 1-NFE vs Multi-step MSE")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    # (b) Channel-mean of generated latents at 1-NFE
    for arm_id, archive in archives.items():
        cfg = ARM_CONFIGS[arm_id]
        epoch_list = sorted(archive.get("epochs", []))
        ch_means = []
        for ep in epoch_list:
            ep_data = archive.get(f"epoch_{ep:04d}", {})
            stats = ep_data.get("stats", {}).get("nfe_1", {})
            mean_list = stats.get("mean", [])
            ch_means.append(np.mean(mean_list) if mean_list else float("nan"))

        arr = np.array(ch_means)
        ep_arr = np.array(epoch_list)
        mask = np.isfinite(arr)
        if mask.any():
            axes[1].plot(
                ep_arr[mask],
                arr[mask],
                label=cfg["label"],
                color=cfg["color"],
                linewidth=PLOT_SETTINGS["line_width"],
            )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean")
    axes[1].set_title("(b) 1-NFE Latent Mean")
    axes[1].legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    # (c) Channel-std of generated latents at 1-NFE
    for arm_id, archive in archives.items():
        cfg = ARM_CONFIGS[arm_id]
        epoch_list = sorted(archive.get("epochs", []))
        ch_stds = []
        for ep in epoch_list:
            ep_data = archive.get(f"epoch_{ep:04d}", {})
            stats = ep_data.get("stats", {}).get("nfe_1", {})
            std_list = stats.get("std", [])
            ch_stds.append(np.mean(std_list) if std_list else float("nan"))

        arr = np.array(ch_stds)
        ep_arr = np.array(epoch_list)
        mask = np.isfinite(arr)
        if mask.any():
            axes[2].plot(
                ep_arr[mask],
                arr[mask],
                label=cfg["label"],
                color=cfg["color"],
                linewidth=PLOT_SETTINGS["line_width"],
            )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Std")
    axes[2].set_title("(c) 1-NFE Latent Std")
    axes[2].legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    fig.suptitle(
        "Sample Quality Evolution",
        fontweight="bold",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, output_dir, "fig9_sample_quality")


# ============================================================================
# Summary report
# ============================================================================


def _last_n_mean(arr: np.ndarray, n: int = 10) -> float:
    """Average over the last N non-NaN values."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid[-n:]))


def _fmt(val: float, precision: int = 4) -> str:
    """Format a float or return 'N/A'."""
    if np.isnan(val) or np.isinf(val):
        return "N/A"
    if abs(val) > 1e6:
        return f"{val:.2e}"
    return f"{val:.{precision}f}"


def generate_summary_report(
    summaries: dict[str, list[dict]],
    archives: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate a comprehensive markdown report comparing the two arms.

    Args:
        summaries: Mapping arm_id -> training summary entries.
        archives: Mapping arm_id -> sample archive (may be empty).
        output_dir: Directory to save the report.
    """
    lines: list[str] = []

    lines.append("# Ablation Report: x-Prediction vs u-Prediction\n")
    lines.append(
        "**Ablation:** `experiments/ablations/xpred_vs_upred/`  \n"
        "**Arms:** x-pred + exact JVP (no grad checkpoint) vs "
        "u-pred + FD-JVP (with grad checkpoint)  "
    )
    lines.append("")

    # --- Training overview table ---
    lines.append("## Training Overview\n")
    lines.append("| Setting | x-pred arm | u-pred arm |")
    lines.append("|---------|-----------|-----------|")

    settings = [
        ("Prediction type", "x", "u"),
        ("JVP strategy", "Exact (`torch.func.jvp`)", "Finite Difference ($h=10^{-3}$)"),
        ("Gradient checkpointing", "OFF", "ON"),
        ("Flash attention", "OFF", "ON"),
        ("Batch/GPU", "2", "16"),
        ("GPUs", "4", "2"),
        ("Accumulate grad batches", "16", "4"),
        ("Effective batch", "2x4x16=128", "16x2x4=128"),
    ]

    for label, val_x, val_u in settings:
        lines.append(f"| {label} | {val_x} | {val_u} |")

    # Append actual epoch counts from data
    x_epochs = len(summaries.get("xpred", []))
    u_epochs = len(summaries.get("upred", []))
    lines.append(f"| Epochs completed | {x_epochs} | {u_epochs} |")

    # --- Throughput table ---
    lines.append("\n## Throughput\n")
    lines.append("| Metric | x-pred arm | u-pred arm |")
    lines.append("|--------|-----------|-----------|")

    for label, key, unit in [
        ("Median epoch time", "epoch_time_sec", "s"),
        ("Total training time", "_total_h", "h"),
    ]:
        vals = {}
        for arm_id, entries in summaries.items():
            times = _safe_array(entries, "epoch_time_sec")
            valid = times[np.isfinite(times)]
            if key == "epoch_time_sec":
                vals[arm_id] = f"{float(np.median(valid)):.1f} {unit}" if len(valid) > 0 else "N/A"
            else:
                vals[arm_id] = (
                    f"{float(np.sum(valid)) / 3600.0:.1f} {unit}" if len(valid) > 0 else "N/A"
                )
        lines.append(f"| {label} | {vals.get('xpred', 'N/A')} | {vals.get('upred', 'N/A')} |")

    # --- Final metrics table (last 10 epoch average) ---
    lines.append("\n## Final Metrics (last 10 epoch average)\n")
    lines.append("| Metric | x-pred arm | u-pred arm |")
    lines.append("|--------|-----------|-----------|")

    metric_specs = [
        ("Weighted loss", "loss_mean", None),
        ("Raw loss", "raw_loss_mean", None),
        ("FM loss (raw)", "raw_loss_fm_mean", None),
        ("MF loss (raw)", "raw_loss_mf_mean", None),
        ("JVP norm", "velocity_norms", "jvp_norm"),
        ("Compound V norm", "velocity_norms", "compound_v_norm"),
        ("Target V norm", "velocity_norms", "target_v_norm"),
        ("V/target ratio", None, None),  # computed
        ("cos(V, $v_c$)", "cosine_sim_V_vc", None),
        ("Relative error", "relative_error", None),
        ("Grad norm", "grad_norm_mean", None),
        ("Relative update norm", "relative_update_norm", None),
        ("EMA divergence", "ema_divergence", None),
        ("Validation loss", "val_loss", None),
        ("Validation raw loss", "val_raw_loss", None),
    ]

    for label, key, nested in metric_specs:
        vals = {}
        for arm_id, entries in summaries.items():
            if key is None:
                # Computed metric: V/target ratio
                cv = _safe_array(entries, "velocity_norms", "compound_v_norm")
                tv = _safe_array(entries, "velocity_norms", "target_v_norm")
                ratio = np.where(tv > 0, cv / tv, np.nan)
                vals[arm_id] = _fmt(_last_n_mean(ratio))
            else:
                arr = _safe_array(entries, key, nested)
                vals[arm_id] = _fmt(_last_n_mean(arr))
        lines.append(f"| {label} | {vals.get('xpred', 'N/A')} | {vals.get('upred', 'N/A')} |")

    # --- Prediction-specific stats ---
    lines.append("\n## Prediction Output Statistics (last 10 epoch average)\n")
    lines.append("| Stat | x-pred ($\\hat{x}_0$) | u-pred ($u$) |")
    lines.append("|------|---------------------|-------------|")

    for stat in ["mean", "std", "min", "max"]:
        vals = {}
        for arm_id, entries in summaries.items():
            sk = "x_hat_stats" if arm_id == "xpred" else "u_pred_stats"
            arr = _safe_array(entries, sk, stat)
            vals[arm_id] = _fmt(_last_n_mean(arr))
        lines.append(
            f"| {stat.capitalize()} | {vals.get('xpred', 'N/A')} | {vals.get('upred', 'N/A')} |"
        )

    # --- Sample quality (if archives available) ---
    if archives:
        lines.append("\n## Sample Quality (from sample archive)\n")
        lines.append("| Metric | x-pred arm | u-pred arm |")
        lines.append("|--------|-----------|-----------|")

        for arm_id, archive in archives.items():
            epoch_list = sorted(archive.get("epochs", []))
            if epoch_list:
                last_ep = epoch_list[-1]
                ep_data = archive.get(f"epoch_{last_ep:04d}", {})
                nfe_con = ep_data.get("nfe_consistency", {})
                stats_1nfe = ep_data.get("stats", {}).get("nfe_1", {})

                lines.append(f"\n**{ARM_CONFIGS[arm_id]['label']}** (epoch {last_ep}):")
                for k, v in sorted(nfe_con.items()):
                    lines.append(f"- {k}: {_fmt(float(v))}")
                if stats_1nfe:
                    lines.append(f"- 1-NFE mean: {stats_1nfe.get('mean', 'N/A')}")
                    lines.append(f"- 1-NFE std: {stats_1nfe.get('std', 'N/A')}")

    # --- Convergence analysis ---
    lines.append("\n## Convergence Analysis\n")

    for arm_id, entries in summaries.items():
        cfg = ARM_CONFIGS[arm_id]
        raw = _safe_array(entries, "raw_loss_mean")
        valid = raw[np.isfinite(raw)]
        if len(valid) < 2:
            continue

        first_10 = float(np.mean(valid[: min(10, len(valid))]))
        last_10 = float(np.mean(valid[-min(10, len(valid)) :]))
        reduction = (first_10 - last_10) / first_10 * 100 if first_10 > 0 else 0

        jvp = _safe_array(entries, "velocity_norms", "jvp_norm")
        jvp_valid = jvp[np.isfinite(jvp)]
        jvp_first = float(np.mean(jvp_valid[:10])) if len(jvp_valid) >= 10 else float("nan")
        jvp_last = float(np.mean(jvp_valid[-10:])) if len(jvp_valid) >= 10 else float("nan")
        jvp_ratio = jvp_last / jvp_first if jvp_first > 0 else float("nan")

        lines.append(f"### {cfg['label']}\n")
        lines.append(
            f"- Raw loss reduction: {reduction:.1f}% ({_fmt(first_10)} -> {_fmt(last_10)})"
        )
        lines.append(
            f"- JVP norm change: {_fmt(jvp_ratio, 2)}x ({_fmt(jvp_first)} -> {_fmt(jvp_last)})"
        )

        # Check for divergence
        mf = _safe_array(entries, "raw_loss_mf_mean")
        mf_valid = mf[np.isfinite(mf)]
        if len(mf_valid) >= 20:
            mf_first = float(np.mean(mf_valid[:10]))
            mf_last = float(np.mean(mf_valid[-10:]))
            if mf_last > mf_first * 10:
                lines.append(
                    f"- **WARNING:** MF loss diverged {mf_last / mf_first:.0f}x "
                    f"({_fmt(mf_first)} -> {_fmt(mf_last)})"
                )
            elif mf_last < mf_first * 0.5:
                lines.append(f"- MF loss converged well: {_fmt(mf_first)} -> {_fmt(mf_last)}")
        lines.append("")

    # --- Decision ---
    lines.append("## Decision\n")

    # Determine winner by raw loss
    raw_x = _last_n_mean(_safe_array(summaries.get("xpred", [{}]), "raw_loss_mean"))
    raw_u = _last_n_mean(_safe_array(summaries.get("upred", [{}]), "raw_loss_mean"))

    if np.isnan(raw_x) or np.isnan(raw_u):
        lines.append("**Insufficient data** — one or both arms did not complete.\n")
    else:
        if raw_x < raw_u:
            winner = "x-pred + exact JVP"
            pct = (raw_u - raw_x) / raw_u * 100
        else:
            winner = "u-pred + FD-JVP"
            pct = (raw_x - raw_u) / raw_x * 100

        lines.append(f"**Winner: {winner}** with {pct:.1f}% lower final raw loss.\n")

    lines.append("### Decision Matrix\n")
    lines.append("| Outcome | Action |")
    lines.append("|---------|--------|")
    lines.append("| x-pred wins by >5% | Adopt x-pred + exact JVP; accept higher VRAM cost |")
    lines.append("| Difference <5% | Adopt u-pred + FD-JVP for memory efficiency |")
    lines.append("| u-pred wins | Confirm u-pred + FD-JVP as production config |")
    lines.append("| Both arms diverge | Investigate further (LR, data proportion, norm_eps) |")

    lines.append("\n### Confounds\n")
    lines.append("- This ablation conflates prediction type with JVP method and memory strategy")
    lines.append("- x-pred arm: exact JVP + no grad checkpointing + no flash attention")
    lines.append("- u-pred arm: FD-JVP + grad checkpointing + flash attention")
    lines.append("- Sample quality (FID, SSIM) should be evaluated separately via Phase 5 pipeline")
    lines.append("- A clean disentanglement requires additional arms (deferred to Phase 6)")

    lines.append("\n## Figures\n")
    lines.append("| Figure | Description |")
    lines.append("|--------|-------------|")
    lines.append("| `fig1_loss_overview` | Weighted, raw, FM, and MF loss curves |")
    lines.append("| `fig2_numerical_stability` | JVP norms, compound V, velocity ratios |")
    lines.append("| `fig3_self_consistency` | Cosine similarities and relative error |")
    lines.append("| `fig4_training_health` | Gradient norms, update norms, clip fraction, EMA |")
    lines.append("| `fig5_throughput` | Per-epoch and cumulative wall time |")
    lines.append("| `fig6_prediction_stats` | Output mean/std/min/max evolution |")
    lines.append("| `fig7_adaptive_weights` | Adaptive weighting dynamics |")
    lines.append("| `fig8_validation` | Validation loss (if available) |")
    lines.append("| `fig9_sample_quality` | NFE consistency and latent stats from samples |")
    lines.append("")

    report_path = output_dir / "ablation_report.md"
    report_path.write_text("\n".join(lines))
    logger.info("Summary report saved to %s", report_path)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run the full comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Compare x-pred vs u-pred ablation arms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Expected directory structure:\n"
            "  <results-dir>/xpred_exact_jvp/diagnostics/aggregate_results/training_summary.json\n"
            "  <results-dir>/upred_fd_jvp/diagnostics/aggregate_results/training_summary.json\n"
            "  <results-dir>/xpred_exact_jvp/samples/sample_archive.pt  (optional)\n"
            "  <results-dir>/upred_fd_jvp/samples/sample_archive.pt      (optional)"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Root directory containing xpred_exact_jvp/ and upred_fd_jvp/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for report output (default: <results-dir>/xpred_vs_upred_report)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "xpred_vs_upred_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply IEEE publication style
    apply_ieee_style()

    # Load data from each arm
    summaries: dict[str, list[dict[str, Any]]] = {}
    archives: dict[str, dict[str, Any]] = {}

    for arm_id, cfg in ARM_CONFIGS.items():
        arm_dir = results_dir / cfg["dir"]
        if not arm_dir.exists():
            logger.warning("Arm '%s' directory not found: %s", arm_id, arm_dir)
            continue

        entries = load_training_summary(arm_dir)
        if entries:
            summaries[arm_id] = entries
            logger.info("Loaded %s: %d epochs", cfg["label"], len(entries))

        archive = load_sample_archive(arm_dir)
        if archive is not None:
            archives[arm_id] = archive

    if not summaries:
        logger.error("No training data found. Check --results-dir path.")
        return

    if len(summaries) < 2:
        logger.warning("Only %d arm(s) have data — generating partial report.", len(summaries))

    # Generate all figures
    plot_loss_overview(summaries, output_dir)
    plot_numerical_stability(summaries, output_dir)
    plot_self_consistency(summaries, output_dir)
    plot_training_health(summaries, output_dir)
    plot_throughput(summaries, output_dir)
    plot_prediction_stats(summaries, output_dir)
    plot_adaptive_weights(summaries, output_dir)
    plot_validation(summaries, output_dir)
    plot_sample_quality(archives, output_dir)

    # Generate markdown report
    generate_summary_report(summaries, archives, output_dir)

    logger.info("Analysis complete. Report at: %s", output_dir)


if __name__ == "__main__":
    main()
