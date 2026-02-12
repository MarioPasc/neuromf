# Phase 8: Paper Figures and Tables

**Depends on:** Phase 7 (gate must be OPEN)
**Modules touched:** `src/neuromf/utils/`, `experiments/`
**Estimated effort:** 1–2 sessions

---

## 1. Objective

Generate all publication-quality figures and tables for the paper from experiment results accumulated in Phases 4–7. This phase produces the visual and tabular content that goes into the manuscript.

## 2. Theoretical Background

From `docs/main/methodology_expanded.md` §12:

**Target venues:**
1. *Medical Image Analysis* (Elsevier, IF ~10.9) — comprehensive method + clinical application
2. *IEEE Transactions on Medical Imaging* (IF ~10.6) — emphasis on methodology
3. *MICCAI 2026* (conference, 8-page format) — condensed method-only version

## 3. External Code to Leverage

No external code needed. Use matplotlib/seaborn for all figures.

## 4. Implementation Specification

### Required Figures (from tech guide §10.1)

| Figure | Content | Source Phase |
|---|---|---|
| Fig. 1 | Method overview: VAE → Latent MeanFlow → Decode | Diagram (TikZ / draw.io) |
| Fig. 2 | Toy toroid results: (a) training loss, (b) generated samples on torus, (c) angular distribution | Phase 2 |
| Fig. 3 | Sample brain MRI: real vs. generated (axial/coronal/sagittal slices) | Phase 5 |
| Fig. 4 | FID vs. NFE curve: MeanFlow vs. Rectified Flow vs. MAISI-v2 | Phase 6 |
| Fig. 5 | $L_p$ sweep: FID and SSIM vs. $p$ | Phase 6 |
| Fig. 6 | x-pred vs. u-pred comparison (FID distribution, sample quality) | Phase 6 |
| Fig. 7 | FCD joint synthesis: generated FLAIR + mask overlays | Phase 7 |
| Fig. 8 | SynthSeg regional volume distributions: real vs. generated | Phase 5 |

### Required Tables (from tech guide §10.2)

| Table | Content |
|---|---|
| Table 1 | Main comparison: FID, SSIM, NFE, time — ours vs. MAISI, MAISI-v2, MOTFM, Med-DDPM |
| Table 2 | x-pred vs. u-pred ablation (FID ± std, SSIM ± std) |
| Table 3 | $L_p$ sweep results |
| Table 4 | Joint synthesis: image quality + mask Dice for different $L_p$ settings |

### `src/neuromf/utils/visualisation.py`
- **Purpose:** Publication-quality figure generation utilities.
- **Key functions:**
  ```python
  def setup_publication_style() -> None: ...  # serif font, 10pt, etc.
  def plot_volume_slices(volume: torch.Tensor, axes: list[str], save_path: Path) -> None: ...
  def plot_loss_curve(log_path: Path, save_path: Path) -> None: ...
  def plot_fid_vs_nfe(results: dict, save_path: Path) -> None: ...
  def plot_lp_sweep(results: pd.DataFrame, save_path: Path) -> None: ...
  def plot_volume_distributions(real: dict, gen: dict, save_path: Path) -> None: ...
  ```

### Figure standards (from paper-figure-generator agent):
- **Font:** serif (Times New Roman or Computer Modern), size 10pt for labels, 8pt for ticks
- **Figsize:** single-column (3.5 inches wide) or double-column (7 inches wide)
- **Save as** both PDF (vector) and PNG (300 DPI)
- **Colour palettes:** colorblind-friendly (seaborn "colorblind" or "Set2")
- **No titles on figures** — titles go in captions in the paper
- **Error bars:** mean ± std where multiple seeds exist

## 5. Data and I/O

- **Input:** All experiment results from Phases 2–7
  - `experiments/toy_toroid/` — toroid results
  - `experiments/stage1_healthy/metrics/` — main evaluation metrics
  - `experiments/ablations/` — all ablation results
  - `experiments/stage2_fcd/` — FCD results
- **Output:**
  - `experiments/paper_figures/fig_{1..8}.pdf` and `.png`
  - `experiments/paper_tables/table_{1..4}.tex` (LaTeX) and `.csv`

## 6. Verification Tests

No formal verification tests for Phase 8. The gate criterion is: **all figures and tables generated without error**.

## 7. Expected Outputs

- `src/neuromf/utils/visualisation.py` (updated)
- `experiments/paper_figures/fig_{1..8}.{pdf,png}`
- `experiments/paper_tables/table_{1..4}.{tex,csv}`
- `experiments/phase_8/verification_report.md` — confirming all figures/tables generated

## 8. Failure Modes and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Missing experiment results | Cannot generate some figures | Low | Phase gating ensures all prior phases complete |
| Font not available on system | Visual inconsistency | Low | Fall back to DejaVu Serif or Computer Modern |
| Large figure files | Slow compilation | Low | Optimise PDF vector graphics; rasterise dense plots |
