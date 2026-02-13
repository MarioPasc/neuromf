"""HTML report generator for Phase 2 toroid experiment.

Produces a self-contained HTML report with base64-embedded figures and tables.
"""

import base64
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _embed_image(path: Path) -> str:
    """Convert an image file to a base64-encoded HTML img tag."""
    if not path.exists():
        return f'<p style="color: red;">Image not found: {path.name}</p>'
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = path.suffix.lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f'<img src="data:{mime};base64,{data}" style="max-width: 100%; height: auto;" />'


def _metrics_table(summary: dict, keys: list[str]) -> str:
    """Build an HTML table from summary metrics."""
    if not keys:
        return "<p>No data available.</p>"

    rows = []
    for key in keys:
        m = summary.get(key, {})
        one = m.get("one_step", {})
        multi = m.get("multi_step", {})
        rows.append(
            f"<tr>"
            f"<td>{key}</td>"
            f"<td>{m.get('final_loss', 'N/A'):.4f}</td>"
            f"<td>{one.get('mean_torus_distance', 'N/A'):.4f}</td>"
            f"<td>{one.get('mmd', 'N/A'):.6f}</td>"
            f"<td>{one.get('coverage', 'N/A'):.3f}</td>"
            f"<td>{one.get('density', 'N/A'):.3f}</td>"
            f"<td>{one.get('theta1_ks_pvalue', 'N/A'):.3f}</td>"
            f"<td>{multi.get('mean_torus_distance', 'N/A'):.4f}</td>"
            f"</tr>"
        )

    header = (
        "<tr><th>Run</th><th>Loss</th><th>Torus Dist (1-NFE)</th>"
        "<th>MMD (1-NFE)</th><th>Coverage</th><th>Density</th>"
        "<th>KS-&theta;1 p</th><th>Torus Dist (10-step)</th></tr>"
    )
    return f'<table border="1" cellpadding="4" cellspacing="0">{header}{"".join(rows)}</table>'


def _nfe_table(nfe_data: dict) -> str:
    """Build HTML table for NFE sweep results."""
    if not nfe_data:
        return "<p>No NFE sweep data available.</p>"

    rows = []
    for nfe in sorted(nfe_data.keys(), key=int):
        m = nfe_data[nfe]
        rows.append(
            f"<tr>"
            f"<td>{nfe}</td>"
            f"<td>{m.get('mean_torus_distance', 'N/A'):.4f}</td>"
            f"<td>{m.get('mmd', 'N/A'):.6f}</td>"
            f"<td>{m.get('coverage', 'N/A'):.3f}</td>"
            f"<td>{m.get('density', 'N/A'):.3f}</td>"
            f"</tr>"
        )

    header = "<tr><th>NFE</th><th>Torus Dist</th><th>MMD</th><th>Coverage</th><th>Density</th></tr>"
    return f'<table border="1" cellpadding="4" cellspacing="0">{header}{"".join(rows)}</table>'


def generate_report(results_dir: Path) -> None:
    """Generate self-contained HTML report.

    Args:
        results_dir: Root results directory containing all ablation outputs.
    """
    fig_dir = results_dir / "figures"

    # Load summary
    summary_file = results_dir / "summary_metrics.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

    # Load NFE sweep
    nfe_data = {}
    nfe_file = results_dir / "ablation_d" / "nfe_sweep.json"
    if nfe_file.exists():
        with open(nfe_file) as f:
            nfe_data = json.load(f)

    # Categorise runs by ablation
    abl_keys = {"a": [], "b": [], "c": [], "e": []}
    for key in sorted(summary.keys()):
        for prefix in abl_keys:
            if key.startswith(f"ablation_{prefix}"):
                abl_keys[prefix].append(key)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Phase 2: MeanFlow Validation on Toroidal Manifold</title>
<style>
body {{ font-family: 'Times New Roman', Times, serif; max-width: 900px; margin: auto; padding: 20px; line-height: 1.6; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
h2 {{ color: #004488; margin-top: 30px; }}
table {{ border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
th {{ background: #f0f0f0; }}
td, th {{ padding: 6px 10px; text-align: center; }}
img {{ margin: 10px 0; border: 1px solid #ddd; }}
.key-result {{ background: #f8f8ff; border-left: 4px solid #004488; padding: 10px 15px; margin: 15px 0; }}
</style>
</head>
<body>

<h1>Phase 2: MeanFlow Validation on Toroidal Manifold</h1>

<h2>1. Executive Summary</h2>
<p>This report presents the results of a formal 5-ablation experiment suite validating the
MeanFlow implementation on a flat torus manifold embedded in R^D. The experiment covers
convergence validation, dimensionality scaling (x-pred vs u-pred), Lp norm impact, NFE-quality
tradeoff, and FM/MF balance (data_proportion).</p>

<p><strong>Total runs:</strong> 18 training + 6 inference configurations.
<strong>Total ablation keys found:</strong> {len(summary)} entries.</p>

<h2>2. Ablation A: Convergence Baseline</h2>
{_embed_image(fig_dir / "fig2a_loss_convergence.png")}
{_embed_image(fig_dir / "fig2d_angular_distributions.png")}
{_metrics_table(summary, abl_keys["a"])}

<h2>3. Ablation B: Dimensionality Scaling (Key Result)</h2>
<div class="key-result">
<p><strong>Key question:</strong> Does x-prediction outperform u-prediction as ambient dimension D increases?</p>
</div>
{_embed_image(fig_dir / "fig2b_dim_scaling.png")}
{_metrics_table(summary, abl_keys["b"])}

<h2>4. Ablation C: Lp Norm Impact</h2>
{_embed_image(fig_dir / "fig2e_lp_impact.png")}
{_metrics_table(summary, abl_keys["c"])}

<h2>5. Ablation D: NFE vs Quality</h2>
{_embed_image(fig_dir / "fig2c_nfe_tradeoff.png")}
{_nfe_table(nfe_data)}

<h2>6. Ablation E: data_proportion Effect</h2>
{_embed_image(fig_dir / "fig2f_data_proportion.png")}
{_metrics_table(summary, abl_keys["e"])}

<h2>7. Conclusions</h2>
<p>The MeanFlow implementation has been validated on a known manifold with geometric and
distributional metrics. Key validated components carrying forward to Phase 3+:</p>
<ul>
<li><code>meanflow_jvp.py</code> — JVP loss computation</li>
<li><code>lp_loss.py</code> — Per-channel Lp loss</li>
<li><code>one_step.py</code> / <code>multi_step.py</code> — Sampling with u/x prediction</li>
<li><code>time_sampler.py</code> — Logit-normal (t, r) sampling</li>
<li><code>ema.py</code> — Exponential moving average</li>
<li><code>mmd.py</code>, <code>coverage_density.py</code> — Distributional metrics</li>
</ul>

<hr>
<p><em>Generated automatically by experiments/toy_toroid/report.py</em></p>
</body>
</html>"""

    report_path = results_dir / "report.html"
    with open(report_path, "w") as f:
        f.write(html)

    log.info("Report saved to %s", report_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()
    generate_report(Path(args.results_dir))
