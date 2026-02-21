"""End-of-training performance dashboard (3x3 grid).

Reads ``training_summary.json`` (per-epoch diagnostics) and produces a
single publication-quality figure with 9 panels covering loss, quality
metrics, cosine similarities, JVP norms, learning rate, and gradients.

Panels:
    (a) FID (avg + per-plane)     (b) SWD               (c) Raw loss (total + FM + MF)
    (d) cos(V, v_c)               (e) cos(v_tilde, v_c)  (f) JVP norm
    (g) Learning rate             (h) Grad norm + clip    (i) V/target norm ratio
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# IEEE-friendly colorblind-safe palette (Paul Tol bright)
_COLORS = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}


def _apply_dashboard_style() -> None:
    """Set rcParams for a dense 3x3 dashboard."""
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


def plot_training_dashboard(
    training_summary_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Generate a 3x3 training performance dashboard.

    Args:
        training_summary_path: Path to ``training_summary.json`` (list of
            per-epoch dicts).
        output_dir: Directory to save the output figure.

    Returns:
        Path to the saved PNG file.
    """
    _apply_dashboard_style()

    training_summary_path = Path(training_summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(training_summary_path) as f:
        data: list[dict[str, Any]] = json.load(f)

    if not data:
        logger.warning("Empty training_summary.json; skipping dashboard.")
        return output_dir / "training_dashboard.png"

    epochs = [d["epoch"] for d in data]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    panel_labels = "abcdefghi"

    # ── (a) FID ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    fid_avg = [(d["epoch"], d["val_fid_avg"]) for d in data if "val_fid_avg" in d]
    fid_xy = [(d["epoch"], d["val_fid_xy"]) for d in data if "val_fid_xy" in d]
    fid_yz = [(d["epoch"], d["val_fid_yz"]) for d in data if "val_fid_yz" in d]
    fid_zx = [(d["epoch"], d["val_fid_zx"]) for d in data if "val_fid_zx" in d]

    if fid_avg:
        ep_fid, v_fid = zip(*fid_avg)
        ax.plot(ep_fid, v_fid, color=_COLORS["blue"], linewidth=1.5, label="Avg")
    if fid_xy:
        ax.plot(*zip(*fid_xy), color=_COLORS["cyan"], linewidth=0.8, alpha=0.6, label="XY")
    if fid_yz:
        ax.plot(*zip(*fid_yz), color=_COLORS["green"], linewidth=0.8, alpha=0.6, label="YZ")
    if fid_zx:
        ax.plot(*zip(*fid_zx), color=_COLORS["yellow"], linewidth=0.8, alpha=0.6, label="ZX")
    ax.set_ylabel("FID")
    ax.set_title("(a) FID (2.5D)")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── (b) SWD ──────────────────────────────────────────────────────────
    ax = axes[0, 1]
    swd_vals = [(d["epoch"], d["train_swd"]) for d in data if "train_swd" in d]
    if swd_vals:
        ep_swd, v_swd = zip(*swd_vals)
        ax.plot(ep_swd, v_swd, color=_COLORS["blue"], linewidth=1.2)
    ax.set_ylabel("SWD")
    ax.set_title("(b) Sliced Wasserstein Distance")
    ax.grid(True, alpha=0.3)

    # ── (c) Raw loss ─────────────────────────────────────────────────────
    ax = axes[0, 2]
    raw_total = [(_safe_get(d, "epoch"), _safe_get(d, "raw_loss_mean")) for d in data]
    raw_fm = [(_safe_get(d, "epoch"), _safe_get(d, "raw_loss_fm_mean")) for d in data]
    raw_mf = [(_safe_get(d, "epoch"), _safe_get(d, "raw_loss_mf_mean")) for d in data]

    _plot_line(ax, raw_total, _COLORS["blue"], "Total", lw=1.5)
    _plot_line(ax, raw_fm, _COLORS["green"], "FM", lw=1.0)
    _plot_line(ax, raw_mf, _COLORS["red"], "MF", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Raw Loss (log)")
    ax.set_title("(c) Raw Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── (d) cos(V, v_c) ─────────────────────────────────────────────────
    ax = axes[1, 0]
    cos_V = [(d["epoch"], d["cosine_sim_V_vc"]) for d in data if "cosine_sim_V_vc" in d]
    if cos_V:
        ax.plot(*zip(*cos_V), color=_COLORS["blue"], linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(r"(d) cos(V, $v_c$)")
    ax.grid(True, alpha=0.3)

    # ── (e) cos(v_tilde, v_c) ───────────────────────────────────────────
    ax = axes[1, 1]
    cos_vt = [(d["epoch"], d["cosine_sim_vtilde_vc"]) for d in data if "cosine_sim_vtilde_vc" in d]
    if cos_vt:
        ax.plot(*zip(*cos_vt), color=_COLORS["green"], linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(r"(e) cos($\tilde{v}$, $v_c$)")
    ax.grid(True, alpha=0.3)

    # ── (f) JVP norm ─────────────────────────────────────────────────────
    ax = axes[1, 2]
    jvp = [(_safe_get(d, "epoch"), _safe_get(d, "velocity_norms", "jvp_norm")) for d in data]
    _plot_line(ax, jvp, _COLORS["red"], None, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("JVP Norm (log)")
    ax.set_title("(f) JVP Norm")
    ax.grid(True, alpha=0.3)

    # ── (g) Learning rate ────────────────────────────────────────────────
    ax = axes[2, 0]
    lr = [(d["epoch"], d["learning_rate"]) for d in data if "learning_rate" in d]
    if lr:
        ax.plot(*zip(*lr), color=_COLORS["purple"], linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR (log)")
    ax.set_title("(g) Learning Rate")
    ax.grid(True, alpha=0.3)

    # ── (h) Grad norm + clip fraction ────────────────────────────────────
    ax = axes[2, 1]
    grad_norm = [(d["epoch"], d["grad_norm_mean"]) for d in data if "grad_norm_mean" in d]
    clip_frac = [(d["epoch"], d["grad_clip_fraction"]) for d in data if "grad_clip_fraction" in d]

    if grad_norm:
        ax.plot(*zip(*grad_norm), color=_COLORS["blue"], linewidth=1.2, label="Grad norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm", color=_COLORS["blue"])
    ax.set_title("(h) Gradient Norm & Clip Fraction")
    ax.grid(True, alpha=0.3)

    if clip_frac:
        ax2 = ax.twinx()
        ax2.plot(
            *zip(*clip_frac), color=_COLORS["red"], linewidth=1.0, alpha=0.7, label="Clip frac"
        )
        ax2.set_ylabel("Clip Fraction", color=_COLORS["red"])
        ax2.set_ylim(-0.05, 1.05)
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    # ── (i) V/target norm ratio ──────────────────────────────────────────
    ax = axes[2, 2]
    ratios: list[tuple[int, float]] = []
    for d in data:
        vn = _safe_get(d, "velocity_norms", "compound_v_norm")
        tn = _safe_get(d, "velocity_norms", "target_v_norm")
        if vn is not None and tn is not None and tn > 0:
            ratios.append((d["epoch"], vn / tn))
    if ratios:
        ax.plot(*zip(*ratios), color=_COLORS["blue"], linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||V|| / ||target||")
    ax.set_title("(i) Compound V / Target Norm Ratio")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Performance Dashboard", fontsize=12, y=1.0)
    fig.tight_layout()

    out_stem = output_dir / "training_dashboard"
    fig.savefig(str(out_stem.with_suffix(".pdf")))
    fig.savefig(str(out_stem.with_suffix(".png")))
    plt.close(fig)

    logger.info("Saved training dashboard to %s", out_stem)
    return out_stem.with_suffix(".png")


def _plot_line(
    ax: plt.Axes,
    xy_pairs: list[tuple[float | None, float | None]],
    color: str,
    label: str | None,
    lw: float = 1.2,
) -> None:
    """Plot a line from (x, y) pairs, skipping None values.

    Args:
        ax: Matplotlib axes.
        xy_pairs: List of (x, y) tuples (None entries skipped).
        color: Line color.
        label: Legend label (None to skip).
        lw: Line width.
    """
    filtered = [(x, y) for x, y in xy_pairs if x is not None and y is not None]
    if not filtered:
        return
    xs, ys = zip(*filtered)
    ax.plot(xs, ys, color=color, linewidth=lw, label=label)
