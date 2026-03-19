"""Compare two NeuroMF training runs (v1 vs v2) with diagnostic figures.

Produces three comparison figures and a summary JSON highlighting key
differences in training dynamics, adaptive weighting, and curriculum.

Usage:
    ~/.conda/envs/neuromf/bin/python experiments/analysis/compare_runs.py \
        --run-v1 /path/to/xpred_exact_jvp \
        --run-v2 /path/to/run_20260306_041930 \
        --output-dir /path/to/output/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Paul Tol bright palette
_COLORS = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

V1_COLOR = _COLORS["blue"]
V2_COLOR = _COLORS["red"]


def _apply_style() -> None:
    """Set rcParams for comparison plots."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _safe_get(entry: dict[str, Any], *keys: str) -> float | None:
    """Traverse nested dict safely, return float or None."""
    obj: Any = entry
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return None
    try:
        return float(obj)
    except (TypeError, ValueError):
        return None


def _extract_series(
    data: list[dict[str, Any]], *keys: str
) -> tuple[list[int], list[float]]:
    """Extract (epochs, values) series from training summary entries."""
    pairs = []
    for d in data:
        val = _safe_get(d, *keys)
        if val is not None:
            pairs.append((d["epoch"], val))
    if not pairs:
        return [], []
    return list(zip(*pairs))


def _load_run_data(run_dir: Path) -> dict[str, Any]:
    """Load training summary and eval summary from a run directory."""
    summary_path = run_dir / "diagnostics" / "aggregate_results" / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"training_summary.json not found at {summary_path}")

    with open(summary_path) as f:
        training_data = json.load(f)

    eval_data: dict[str, Any] | None = None
    eval_path = run_dir / "diagnostics" / "eval_cache" / "eval_summary.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)

    config_dirs = list((run_dir / "logs" / "config_snapshots").iterdir())
    config: dict[str, Any] = {}
    if config_dirs:
        cfg_path = config_dirs[0] / "resolved_config.yaml"
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                config = yaml.safe_load(f)

    return {
        "training": training_data,
        "eval": eval_data,
        "config": config,
        "dir": run_dir,
    }


def _get_fid_series(run: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Extract FID series — 3D from eval_summary or 2.5D from training_summary."""
    eval_data = run.get("eval")
    if eval_data:
        fid_entries = [
            (e["train_epoch"], e["fid_3d"])
            for e in eval_data.get("per_epoch", [])
            if "fid_3d" in e
        ]
        if fid_entries:
            return list(zip(*fid_entries))

    # Fallback to 2.5D
    return _extract_series(run["training"], "val_fid_avg")


def plot_training_dynamics_comparison(
    v1: dict[str, Any],
    v2: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Figure 1: 3x3 training dynamics comparison."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))

    d1, d2 = v1["training"], v2["training"]

    def _dual_plot(
        ax: plt.Axes,
        key_or_keys: str | tuple[str, ...],
        ylabel: str,
        title: str,
        log_scale: bool = False,
    ) -> None:
        keys = (key_or_keys,) if isinstance(key_or_keys, str) else key_or_keys
        ep1, val1 = _extract_series(d1, *keys)
        ep2, val2 = _extract_series(d2, *keys)
        if ep1:
            ax.plot(ep1, val1, color=V1_COLOR, linewidth=1.2, label="v1", alpha=0.8)
        if ep2:
            ax.plot(ep2, val2, color=V2_COLOR, linewidth=1.2, label="v2", alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale("log")

    # (a) FID
    ax = axes[0, 0]
    ep1_fid, val1_fid = _get_fid_series(v1)
    ep2_fid, val2_fid = _get_fid_series(v2)
    if ep1_fid:
        ax.plot(ep1_fid, val1_fid, color=V1_COLOR, linewidth=1.2, label="v1 (2.5D)", alpha=0.8)
    if ep2_fid:
        ax.plot(ep2_fid, val2_fid, color=V2_COLOR, linewidth=1.2, label="v2 (3D)", alpha=0.8)
    ax.set_ylabel("FID")
    ax.set_title("(a) FID (different metrics!)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # (b) Adaptive-weighted loss
    _dual_plot(axes[0, 1], "loss_mean", "Loss", "(b) Adaptive-Weighted Loss")

    # (c) Raw loss
    _dual_plot(axes[0, 2], "raw_loss_mean", "Raw Loss", "(c) Raw Loss (pre-adaptive)", log_scale=True)

    # (d) cos(V, v_c)
    _dual_plot(axes[1, 0], "cosine_sim_V_vc", "Cosine Sim", r"(d) cos(V, $v_c$)")

    # (e) cos(v_tilde, v_c)
    _dual_plot(axes[1, 1], "cosine_sim_vtilde_vc", "Cosine Sim", r"(e) cos($\tilde{v}$, $v_c$)")

    # (f) V/target norm ratio
    ax = axes[1, 2]
    for data, color, label in [(d1, V1_COLOR, "v1"), (d2, V2_COLOR, "v2")]:
        ratios = []
        for d in data:
            vn = _safe_get(d, "velocity_norms", "compound_v_norm")
            tn = _safe_get(d, "velocity_norms", "target_v_norm")
            if vn is not None and tn is not None and tn > 0:
                ratios.append((d["epoch"], vn / tn))
        if ratios:
            eps, vals = zip(*ratios)
            ax.plot(eps, vals, color=color, linewidth=1.2, label=label, alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("||V|| / ||target||")
    ax.set_title("(f) Compound V / Target Norm Ratio")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # (g) Gradient norm
    _dual_plot(axes[2, 0], "grad_norm_mean", "Grad Norm", "(g) Gradient Norm Mean", log_scale=True)
    axes[2, 0].set_xlabel("Epoch")

    # (h) Gradient clip fraction
    _dual_plot(axes[2, 1], "grad_clip_fraction", "Clip Fraction", "(h) Gradient Clip Fraction")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylim(-0.05, 1.05)

    # (i) Learning rate
    _dual_plot(axes[2, 2], "learning_rate", "LR", "(i) Learning Rate", log_scale=True)
    axes[2, 2].set_xlabel("Epoch")

    fig.suptitle("Training Dynamics: v1 vs v2", fontsize=12, y=1.0)
    fig.tight_layout()

    out = output_dir / "comparison_training_dynamics"
    fig.savefig(str(out.with_suffix(".pdf")))
    fig.savefig(str(out.with_suffix(".png")))
    plt.close(fig)
    logger.info("Saved %s", out)
    return out.with_suffix(".png")


def plot_adaptive_weight_pathology(
    v1: dict[str, Any],
    v2: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Figure 2: 2x2 adaptive weight pathology analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    d1, d2 = v1["training"], v2["training"]

    # (a) Adaptive weight mean
    ax = axes[0, 0]
    for data, color, label in [(d1, V1_COLOR, "v1"), (d2, V2_COLOR, "v2")]:
        ep, val = _extract_series(data, "adaptive_weight", "mean")
        if ep:
            ax.plot(ep, val, color=color, linewidth=1.2, label=label, alpha=0.8)
    ax.set_ylabel("Adaptive Weight Mean")
    ax.set_title("(a) Adaptive Weight Mean")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Loss mean (showing scale difference)
    ax = axes[0, 1]
    for data, color, label in [(d1, V1_COLOR, "v1"), (d2, V2_COLOR, "v2")]:
        ep, val = _extract_series(data, "loss_mean")
        if ep:
            ax.plot(ep, val, color=color, linewidth=1.2, label=label, alpha=0.8)
    ax.set_ylabel("Adaptive-Weighted Loss")
    ax.set_title("(b) Loss Scale: v1 ~2.0 vs v2 ~3000")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Raw loss U vs V (v2 only, showing dual-head balance)
    ax = axes[1, 0]
    for data, color, label in [(d1, V1_COLOR, "v1"), (d2, V2_COLOR, "v2")]:
        ep_u, val_u = _extract_series(data, "raw_loss_u_mean")
        ep_v, val_v = _extract_series(data, "raw_loss_v_mean")
        if ep_u:
            ax.plot(ep_u, val_u, color=color, linewidth=1.2, label=f"{label} loss_u",
                    linestyle="-", alpha=0.8)
        if ep_v:
            ax.plot(ep_v, val_v, color=color, linewidth=1.0, label=f"{label} loss_v",
                    linestyle="--", alpha=0.6)
    ax.set_ylabel("Raw Loss")
    ax.set_title("(c) Dual-Head Raw Loss (U vs V)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # (d) Per-block gradient norms at last epoch (bar chart)
    ax = axes[1, 1]
    block_names = None
    for data, color, label, offset in [(d1, V1_COLOR, "v1", -0.2), (d2, V2_COLOR, "v2", 0.2)]:
        last = data[-1]
        block_grads = last.get("block_grad_norms", {})
        if block_grads:
            names = list(block_grads.keys())
            vals = list(block_grads.values())
            if block_names is None:
                block_names = names
            x = np.arange(len(names))
            ax.bar(x + offset, vals, width=0.35, color=color, label=label, alpha=0.8)
    if block_names:
        ax.set_xticks(np.arange(len(block_names)))
        ax.set_xticklabels(block_names, rotation=45, ha="right", fontsize=5)
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"(d) Per-Block Gradient Norms (last epoch)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Adaptive Weight Pathology: norm_p=1.0 (v1) vs norm_p=0.5 (v2)", fontsize=11, y=1.0)
    fig.tight_layout()

    out = output_dir / "comparison_adaptive_pathology"
    fig.savefig(str(out.with_suffix(".pdf")))
    fig.savefig(str(out.with_suffix(".png")))
    plt.close(fig)
    logger.info("Saved %s", out)
    return out.with_suffix(".png")


def plot_progressive_gap_curriculum(
    v2: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Figure 3: 1x3 progressive gap curriculum analysis (v2 only)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    d2 = v2["training"]

    # (a) h (gap) distribution evolution via sampling stats
    ax = axes[0]
    sample_epochs = [0, 50, 100, 200, 359]
    colors_iter = [_COLORS["blue"], _COLORS["cyan"], _COLORS["green"],
                   _COLORS["yellow"], _COLORS["red"]]
    for ep_target, color in zip(sample_epochs, colors_iter):
        # Find nearest epoch
        entry = min(d2, key=lambda d: abs(d["epoch"] - ep_target))
        h_mean = _safe_get(entry, "sampling", "h_mean")
        h_zero_frac = _safe_get(entry, "sampling", "h_zero_frac")
        if h_mean is not None:
            ax.bar(
                ep_target, h_mean, width=15, color=color, alpha=0.7,
                label=f"ep {entry['epoch']} (h={h_mean:.3f})"
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Gap h")
    ax.set_title("(a) Mean Gap h Over Training")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # (b) Per-time-bin cosine similarity
    ax = axes[1]
    bins = ["near_data", "mid", "near_noise"]
    for bin_name, color, marker in zip(
        bins,
        [_COLORS["blue"], _COLORS["green"], _COLORS["red"]],
        ["o", "s", "^"],
    ):
        key = f"cosine_sim_V_vc_{bin_name}"
        ep_vals = [(d["epoch"], d.get(key)) for d in d2 if d.get(key) is not None]
        if ep_vals:
            eps, vals = zip(*ep_vals)
            ax.plot(eps, vals, color=color, linewidth=1.0, label=bin_name,
                    marker=marker, markersize=2, alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("cos(V, v_c)")
    ax.set_title("(b) Per-Time-Bin cos(V, v_c)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) JVP decomposition: temporal vs spatial
    ax = axes[2]
    ep_temp, val_temp = _extract_series(d2, "jvp_decomposition", "temporal_norm")
    ep_spat, val_spat = _extract_series(d2, "jvp_decomposition", "spatial_norm")
    if ep_temp:
        ax.plot(ep_temp, val_temp, color=_COLORS["blue"], linewidth=1.2, label="Temporal")
    if ep_spat:
        ax.plot(ep_spat, val_spat, color=_COLORS["red"], linewidth=1.2, label="Spatial")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Norm (log)")
    ax.set_title("(c) JVP Decomposition: Temporal vs Spatial")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("v2 Training Dynamics Detail", fontsize=11, y=1.02)
    fig.tight_layout()

    out = output_dir / "comparison_v2_details"
    fig.savefig(str(out.with_suffix(".pdf")))
    fig.savefig(str(out.with_suffix(".png")))
    plt.close(fig)
    logger.info("Saved %s", out)
    return out.with_suffix(".png")


def write_comparison_summary(
    v1: dict[str, Any],
    v2: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write a machine-readable comparison JSON."""
    d1, d2 = v1["training"], v2["training"]

    def _best_metric(data: list[dict[str, Any]], key: str) -> float | None:
        vals = [d.get(key) for d in data if d.get(key) is not None]
        return min(vals) if vals else None

    def _last_metric(data: list[dict[str, Any]], key: str) -> float | None:
        for d in reversed(data):
            val = d.get(key)
            if val is not None:
                return val
        return None

    # FID
    v1_fid_ep, v1_fid_vals = _get_fid_series(v1)
    v2_fid_ep, v2_fid_vals = _get_fid_series(v2)

    summary = {
        "v1": {
            "dir": str(v1["dir"]),
            "total_epochs": len(d1),
            "total_steps": d1[-1].get("global_step", 0) if d1 else 0,
            "best_fid": min(v1_fid_vals) if v1_fid_vals else None,
            "best_fid_epoch": v1_fid_ep[int(np.argmin(v1_fid_vals))] if v1_fid_vals else None,
            "fid_mode": "2.5D",
            "final_cos_V_vc": _last_metric(d1, "cosine_sim_V_vc"),
            "final_cos_vtilde_vc": _last_metric(d1, "cosine_sim_vtilde_vc"),
            "final_grad_clip_frac": _last_metric(d1, "grad_clip_fraction"),
            "final_loss_mean": _last_metric(d1, "loss_mean"),
            "norm_p": v1["config"].get("meanflow", {}).get("norm_p", 1.0),
        },
        "v2": {
            "dir": str(v2["dir"]),
            "total_epochs": len(d2),
            "total_steps": d2[-1].get("global_step", 0) if d2 else 0,
            "best_fid": min(v2_fid_vals) if v2_fid_vals else None,
            "best_fid_epoch": v2_fid_ep[int(np.argmin(v2_fid_vals))] if v2_fid_vals else None,
            "fid_mode": "3D",
            "final_cos_V_vc": _last_metric(d2, "cosine_sim_V_vc"),
            "final_cos_vtilde_vc": _last_metric(d2, "cosine_sim_vtilde_vc"),
            "final_grad_clip_frac": _last_metric(d2, "grad_clip_fraction"),
            "final_loss_mean": _last_metric(d2, "loss_mean"),
            "norm_p": v2["config"].get("meanflow", {}).get("norm_p", 0.5),
        },
        "config_diff": {
            "norm_p": {"v1": 1.0, "v2": 0.5, "impact": "PRIMARY PATHOLOGY"},
            "lr_schedule": {"v1": "cosine", "v2": "constant"},
            "warmup_steps": {"v1": 0, "v2": 1000},
            "boundary_fraction": {"v1": "N/A", "v2": 0.0},
            "progressive_gap": {"v1": False, "v2": True},
            "v_head_res_blocks": {"v1": 1, "v2": 2},
            "backbone_lr_factor": {"v1": 1.0, "v2": 0.1},
            "devices": {"v1": 6, "v2": 3},
            "accum": {"v1": 11, "v2": 22},
            "fid_mode": {"v1": "2.5D", "v2": "3D"},
            "early_stop_patience": {"v1": 10, "v2": 15},
        },
        "diagnosis": {
            "primary": "norm_p=0.5 breaks adaptive weighting: loss=L/sqrt(L+1)~sqrt(L) instead of L/(L+1)~1.0",
            "secondary": "boundary_fraction=0.0 — 1-NFE improvement never activated",
            "tertiary": "constant LR + progressive gap: no LR decay when curriculum opens to harder samples",
        },
    }

    out = output_dir / "comparison_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved comparison summary to %s", out)
    return out


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare two NeuroMF training runs.")
    parser.add_argument("--run-v1", type=str, required=True, help="v1 run directory.")
    parser.add_argument("--run-v2", type=str, required=True, help="v2 run directory.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to v2's figures/.")
    args = parser.parse_args()

    _apply_style()

    v1_dir = Path(args.run_v1)
    v2_dir = Path(args.run_v2)
    output_dir = Path(args.output_dir) if args.output_dir else v2_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading v1 from %s", v1_dir)
    v1 = _load_run_data(v1_dir)
    logger.info("Loading v2 from %s", v2_dir)
    v2 = _load_run_data(v2_dir)

    plot_training_dynamics_comparison(v1, v2, output_dir)
    plot_adaptive_weight_pathology(v1, v2, output_dir)
    plot_progressive_gap_curriculum(v2, output_dir)
    write_comparison_summary(v1, v2, output_dir)

    logger.info("All comparison outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
