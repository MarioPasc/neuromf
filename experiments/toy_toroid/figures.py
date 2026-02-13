"""Publication-quality figure generation for Phase 2 toroid experiment.

Generates Figs 2a-2f as PDF+PNG using IEEE style from experiments/utils/settings.py.
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Add project root to path for imports
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.utils.settings import (
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
)

log = logging.getLogger(__name__)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure as both PDF and PNG."""
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path.stem)


def fig2a_loss_convergence(results_dir: Path, fig_dir: Path) -> None:
    """Fig 2a: Training loss convergence (Ablation A)."""
    log_file = results_dir / "ablation_a" / "baseline_D4_u-pred_p2.0_dp0.75" / "training_log.json"
    if not log_file.exists():
        log.warning("Ablation A log not found — skipping fig2a")
        return

    with open(log_file) as f:
        data = json.load(f)

    losses = data["epoch_losses"]
    losses_fm = data["epoch_losses_fm"]
    losses_mf = data["epoch_losses_mf"]
    epochs = list(range(1, len(losses) + 1))

    w, h = get_figure_size("single", height_ratio=0.8)
    fig, ax = plt.subplots(figsize=(w, h))

    ax.plot(
        epochs,
        losses,
        color=PAUL_TOL_BRIGHT["blue"],
        linewidth=PLOT_SETTINGS["line_width"],
        label="Total loss",
    )
    ax.plot(
        epochs,
        losses_fm,
        color=PAUL_TOL_BRIGHT["red"],
        linewidth=PLOT_SETTINGS["line_width"],
        linestyle="--",
        label="FM loss",
        alpha=0.8,
    )
    ax.plot(
        epochs,
        losses_mf,
        color=PAUL_TOL_BRIGHT["green"],
        linewidth=PLOT_SETTINGS["line_width"],
        linestyle=":",
        label="MF loss",
        alpha=0.8,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])
    ax.set_title("(a) Training Loss Convergence", fontsize=PLOT_SETTINGS["axes_titlesize"])

    _save_fig(fig, fig_dir / "fig2a_loss_convergence")


def fig2b_dim_scaling(results_dir: Path, fig_dir: Path) -> None:
    """Fig 2b: Dimensionality scaling 2x4 grid (Ablation B) — KEY FIGURE."""
    dims = [4, 16, 64, 256]
    pred_types = ["x", "u"]

    # Check if data exists
    first_dir = results_dir / "ablation_b" / f"D{dims[0]}_{pred_types[0]}-pred"
    if not (first_dir / "samples.pt").exists():
        log.warning("Ablation B samples not found — skipping fig2b")
        return

    w, h = get_figure_size("double", height_ratio=0.45)
    fig, axes = plt.subplots(2, 4, figsize=(w, h))

    colors = {"x": PAUL_TOL_BRIGHT["blue"], "u": PAUL_TOL_BRIGHT["red"]}

    for row, pred in enumerate(pred_types):
        for col, dim in enumerate(dims):
            ax = axes[row, col]
            run_dir = results_dir / "ablation_b" / f"D{dim}_{pred}-pred"
            samples_file = run_dir / "samples.pt"
            metrics_file = run_dir / "metrics.json"

            if not samples_file.exists():
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            data = torch.load(samples_file, weights_only=True)
            samples = data["one_step"]  # (N, D)

            # Project to first 2 components (z1, z2) for visualisation
            if dim > 4:
                # Load projection matrix from dataset
                from neuromf.data.toroid_dataset import ToroidConfig, ToroidDataset

                ds = ToroidDataset(ToroidConfig(n_samples=10, ambient_dim=dim), seed=42)
                samples_4d = ds.project_to_r4(samples)
            else:
                samples_4d = samples

            ax.scatter(
                samples_4d[:, 0].numpy(),
                samples_4d[:, 1].numpy(),
                s=1,
                alpha=0.3,
                color=colors[pred],
                rasterized=True,
            )

            # Reference unit circle (unnormalised torus: pair norm = 1.0)
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.5)

            # Annotate with MMD
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                mmd = m.get("one_step", {}).get("mmd", None)
                if mmd is not None:
                    ax.text(
                        0.95,
                        0.05,
                        f"MMD={mmd:.4f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=PLOT_SETTINGS["annotation_fontsize"],
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    )

            ax.set_aspect("equal")
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)

            if row == 0:
                ax.set_title(f"D={dim}", fontsize=PLOT_SETTINGS["axes_titlesize"])
            if col == 0:
                ax.set_ylabel(f"{pred}-pred", fontsize=PLOT_SETTINGS["axes_labelsize"])

            # Remove tick labels for inner axes
            if col > 0:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_xticklabels([])

    fig.suptitle(
        "(b) Dimensionality Scaling: $(z_1, z_2)$ Projections",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, fig_dir / "fig2b_dim_scaling")


def fig2c_nfe_tradeoff(results_dir: Path, fig_dir: Path) -> None:
    """Fig 2c: NFE vs quality dual-axis (Ablation D)."""
    nfe_file = results_dir / "ablation_d" / "nfe_sweep.json"
    if not nfe_file.exists():
        log.warning("Ablation D results not found — skipping fig2c")
        return

    with open(nfe_file) as f:
        nfe_data = json.load(f)

    nfes = sorted([int(k) for k in nfe_data.keys()])
    torus_dists = [nfe_data[str(n)]["mean_torus_distance"] for n in nfes]
    mmds = [nfe_data[str(n)]["mmd"] for n in nfes]

    w, h = get_figure_size("single", height_ratio=0.8)
    fig, ax1 = plt.subplots(figsize=(w, h))

    color1 = PAUL_TOL_BRIGHT["blue"]
    color2 = PAUL_TOL_BRIGHT["red"]

    ax1.plot(
        nfes,
        torus_dists,
        "o-",
        color=color1,
        linewidth=PLOT_SETTINGS["line_width"],
        markersize=PLOT_SETTINGS["marker_size"],
        label="Torus distance",
    )
    ax1.set_xlabel("Number of Function Evaluations (NFE)")
    ax1.set_ylabel("Mean Torus Distance", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.plot(
        nfes,
        mmds,
        "s--",
        color=color2,
        linewidth=PLOT_SETTINGS["line_width"],
        markersize=PLOT_SETTINGS["marker_size"],
        label="MMD",
    )
    ax2.set_ylabel("MMD", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=PLOT_SETTINGS["legend_fontsize"])

    ax1.set_title("(c) NFE vs Quality Tradeoff", fontsize=PLOT_SETTINGS["axes_titlesize"])
    fig.tight_layout()
    _save_fig(fig, fig_dir / "fig2c_nfe_tradeoff")


def fig2d_angular_distributions(results_dir: Path, fig_dir: Path) -> None:
    """Fig 2d: Angular distribution recovery 2x2 (Ablation A)."""
    samples_file = results_dir / "ablation_a" / "baseline_D4_u-pred_p2.0_dp0.75" / "samples.pt"
    metrics_file = results_dir / "ablation_a" / "baseline_D4_u-pred_p2.0_dp0.75" / "metrics.json"
    if not samples_file.exists():
        log.warning("Ablation A samples not found — skipping fig2d")
        return

    data = torch.load(samples_file, weights_only=True)

    with open(metrics_file) as f:
        metrics = json.load(f)

    w, h = get_figure_size("double", height_ratio=0.45)
    fig, axes = plt.subplots(2, 2, figsize=(w, h))

    uniform_y = 1.0 / (2 * np.pi)
    bins = 40

    for row, (name, key) in enumerate([("1-NFE", "one_step"), ("10-step", "multi_step")]):
        samples = data[key]
        theta1 = torch.atan2(samples[:, 1], samples[:, 0]).numpy()
        theta2 = torch.atan2(samples[:, 3], samples[:, 2]).numpy()

        m = metrics.get(key, {})

        for col, (theta, label, ks_key) in enumerate(
            [
                (theta1, r"$\theta_1$", "theta1_ks_pvalue"),
                (theta2, r"$\theta_2$", "theta2_ks_pvalue"),
            ]
        ):
            ax = axes[row, col]
            ax.hist(
                theta,
                bins=bins,
                density=True,
                alpha=0.7,
                color=PAUL_TOL_BRIGHT["blue"],
                edgecolor="white",
                linewidth=0.3,
            )
            ax.axhline(
                uniform_y,
                color=PAUL_TOL_BRIGHT["red"],
                linestyle="--",
                linewidth=PLOT_SETTINGS["line_width"],
                label="Uniform",
            )

            ks_p = m.get(ks_key, None)
            if ks_p is not None:
                ax.text(
                    0.95,
                    0.95,
                    f"KS p={ks_p:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=PLOT_SETTINGS["annotation_fontsize"],
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

            if row == 1:
                ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel(f"{name}\nDensity")
            ax.set_xlim(-np.pi, np.pi)

    fig.suptitle(
        "(d) Angular Distribution Recovery", fontsize=PLOT_SETTINGS["axes_titlesize"], y=1.02
    )
    fig.tight_layout()
    _save_fig(fig, fig_dir / "fig2d_angular_distributions")


def fig2e_lp_impact(results_dir: Path, fig_dir: Path) -> None:
    """Fig 2e: Lp norm impact bar chart (Ablation C)."""
    p_values = [1.0, 1.5, 2.0, 3.0]
    torus_dists = []
    mmds = []
    coverages = []

    for p in p_values:
        metrics_file = results_dir / "ablation_c" / f"p{p:.1f}" / "metrics.json"
        if not metrics_file.exists():
            log.warning("Ablation C p=%.1f not found — skipping fig2e", p)
            return
        with open(metrics_file) as f:
            m = json.load(f)
        one = m.get("one_step", {})
        torus_dists.append(one.get("mean_torus_distance", 0))
        mmds.append(one.get("mmd", 0))
        coverages.append(one.get("coverage", 0))

    w, h = get_figure_size("single", height_ratio=0.8)
    fig, ax = plt.subplots(figsize=(w, h))

    x = np.arange(len(p_values))
    bar_w = PLOT_SETTINGS["bar_width"]

    ax.bar(
        x - bar_w,
        torus_dists,
        bar_w,
        label="Torus dist.",
        color=PAUL_TOL_BRIGHT["blue"],
        alpha=PLOT_SETTINGS["bar_alpha"],
    )
    ax.bar(
        x, mmds, bar_w, label="MMD", color=PAUL_TOL_BRIGHT["red"], alpha=PLOT_SETTINGS["bar_alpha"]
    )
    ax.bar(
        x + bar_w,
        coverages,
        bar_w,
        label="Coverage",
        color=PAUL_TOL_BRIGHT["green"],
        alpha=PLOT_SETTINGS["bar_alpha"],
    )

    ax.set_xlabel("$L_p$ norm exponent $p$")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.1f}" for p in p_values])
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])
    ax.set_title("(e) $L_p$ Norm Impact", fontsize=PLOT_SETTINGS["axes_titlesize"])

    fig.tight_layout()
    _save_fig(fig, fig_dir / "fig2e_lp_impact")


def fig2f_data_proportion(results_dir: Path, fig_dir: Path) -> None:
    """Fig 2f: data_proportion effect (Ablation E)."""
    dp_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    mmds_1nfe = []
    mmds_multi = []

    for dp in dp_values:
        metrics_file = results_dir / "ablation_e" / f"dp{dp:.2f}" / "metrics.json"
        if not metrics_file.exists():
            log.warning("Ablation E dp=%.2f not found — skipping fig2f", dp)
            return
        with open(metrics_file) as f:
            m = json.load(f)
        mmds_1nfe.append(m.get("one_step", {}).get("mmd", 0))
        mmds_multi.append(m.get("multi_step", {}).get("mmd", 0))

    w, h = get_figure_size("single", height_ratio=0.8)
    fig, ax = plt.subplots(figsize=(w, h))

    ax.plot(
        dp_values,
        mmds_1nfe,
        "o-",
        color=PAUL_TOL_BRIGHT["blue"],
        linewidth=PLOT_SETTINGS["line_width"],
        markersize=PLOT_SETTINGS["marker_size"],
        label="1-NFE",
    )
    ax.plot(
        dp_values,
        mmds_multi,
        "s--",
        color=PAUL_TOL_BRIGHT["red"],
        linewidth=PLOT_SETTINGS["line_width"],
        markersize=PLOT_SETTINGS["marker_size"],
        label="10-step",
    )

    ax.set_xlabel("data\\_proportion")
    ax.set_ylabel("MMD")
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])
    ax.set_title("(f) FM/MF Balance Effect", fontsize=PLOT_SETTINGS["axes_titlesize"])

    fig.tight_layout()
    _save_fig(fig, fig_dir / "fig2f_data_proportion")


def generate_all_figures(results_dir: Path) -> None:
    """Generate all Phase 2 figures."""
    apply_ieee_style()

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig2a_loss_convergence(results_dir, fig_dir)
    fig2b_dim_scaling(results_dir, fig_dir)
    fig2c_nfe_tradeoff(results_dir, fig_dir)
    fig2d_angular_distributions(results_dir, fig_dir)
    fig2e_lp_impact(results_dir, fig_dir)
    fig2f_data_proportion(results_dir, fig_dir)

    log.info("All figures saved to %s", fig_dir)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()
    generate_all_figures(Path(args.results_dir))
