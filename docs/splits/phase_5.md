# Phase 5: Evaluation Suite

**Depends on:** Phase 4 (gate must be OPEN)
**Modules touched:** `src/neuromf/metrics/`, `experiments/cli/`, `configs/`
**Estimated effort:** 2–3 sessions

---

## 1. Objective

Generate 2,000 synthetic brain MRI volumes at 1-NFE, compute all evaluation metrics (FID, 3D-FID, SSIM, PSNR, SynthSeg Dice, spectral analysis), and compare against baselines. This phase produces the numbers that go into the paper's main results table.

## 2. Theoretical Background

From `docs/main/methodology_expanded.md`:

### §10.1 Image Quality Metrics

| Metric | Description | Implementation |
|---|---|---|
| **FID** (Fréchet Inception Distance) | Distribution-level quality; computed slice-wise (axial) on 2D slices extracted from 3D volumes | `torch-fidelity` or custom with InceptionV3 |
| **3D-FID** | FID computed on 3D feature vectors extracted by Med3D (Chen et al., 2019) or SynthSeg encoder | Custom implementation |
| **SSIM** | Per-volume structural similarity; averaged over all volumes | `skimage.metrics.structural_similarity` |
| **PSNR** | Peak signal-to-noise ratio | Standard formula |

### §10.2 Morphological Metrics (via SynthSeg)

Following MOTFM (Yazdani et al., 2025), apply **SynthSeg** (Billot et al., 2023) to both real and synthetic volumes, then compare:

| Metric | Description |
|---|---|
| **SynthSeg Dice** | Dice overlap between SynthSeg labels of real and synthetic volumes (paired by nearest neighbour in feature space) |
| **Volume correlation** | Pearson $r$ between regional volumes (hippocampus, ventricles, cortex) in real vs. synthetic |
| **Morphological realism** | Distribution of regional volumes: KL divergence between real and synthetic histograms |

### §10.3 VAE Smoothing Quantification

To explicitly quantify the VAE smoothing artefact:
1. Compute the high-frequency energy ratio $\rho = \sum_{|\mathbf{k}| > k_0} |F(\hat{\mathbf{x}})|^2 / \sum_{\mathbf{k}} |F(\hat{\mathbf{x}})|^2$ for real, VAE-reconstructed, and generated volumes, where $F$ denotes the 3D DFT and $k_0$ is a frequency cutoff.
2. Report $\rho_{\text{real}}$, $\rho_{\text{VAE-recon}}$, $\rho_{\text{generated}}$ to disentangle smoothing from the VAE vs. the generative model.

### §10.4 Sampling Speed

| Method | NFE | Time per volume (A100) |
|---|---|---|
| MAISI (DDPM) | 50 | ~50s |
| MAISI-v2 (Rectified Flow) | 5–50 | ~5–50s |
| MOTFM | 10–50 | ~10–50s |
| Med-DDPM | 1000 | ~1000s |
| **Ours (MeanFlow)** | **1** | **~1s** |

## 3. External Code to Leverage

### `src/external/MOTFM/`
- **Insights doc:** `docs/papers/motfm_2025/insights.md`
- **Specific files:** `inferer.py` (sampling/generation), `utils/utils_fm.py` (evaluation protocol)
- **What to extract:** FID computation protocol (slice-wise), evaluation pipeline structure
- **What to AVOID:** MOTFM's pixel-space flow matching training code (irrelevant to our latent approach)

## 4. Implementation Specification

### `src/neuromf/metrics/fid.py`
- **Purpose:** Slice-wise FID computation on axial/coronal/sagittal slices.
- **Key functions:**
  ```python
  def compute_fid(real_dir: Path, generated_dir: Path, slice_axis: str = "axial") -> float: ...
  ```

### `src/neuromf/metrics/fid_3d.py`
- **Purpose:** 3D-FID via Med3D features.
- **Key functions:**
  ```python
  def compute_3d_fid(real_dir: Path, generated_dir: Path) -> float: ...
  ```

### `src/neuromf/metrics/ssim_psnr.py`
- **Purpose:** SSIM and PSNR for 3D volumes.
- **Key functions:**
  ```python
  def compute_ssim(real: torch.Tensor, generated: torch.Tensor) -> float: ...
  def compute_psnr(real: torch.Tensor, generated: torch.Tensor) -> float: ...
  ```

### `src/neuromf/metrics/synthseg_metrics.py`
- **Purpose:** SynthSeg-based morphological evaluation.
- **Key functions:**
  ```python
  def run_synthseg(volume_dir: Path, output_dir: Path) -> None: ...
  def compute_volume_correlations(real_labels: Path, gen_labels: Path) -> dict: ...
  ```

### `src/neuromf/metrics/spectral.py`
- **Purpose:** High-frequency energy analysis for VAE smoothing quantification.
- **Key functions:**
  ```python
  def compute_hf_energy_ratio(volume: torch.Tensor, k0: int) -> float: ...
  ```

### `experiments/cli/generate.py`
- **Purpose:** Generate N synthetic volumes using trained model.
- **Usage:**
  ```bash
  python experiments/cli/generate.py --config configs/evaluate.yaml --checkpoint best_ema.ckpt --n_samples 2000 --nfe 1
  ```

### `experiments/cli/evaluate.py`
- **Purpose:** Compute all metrics on generated vs. real volumes.
- **Usage:**
  ```bash
  python experiments/cli/evaluate.py --generated_dir experiments/stage1_healthy/generated --real_dir /path/to/test_set
  ```

### `configs/evaluate.yaml`
- **Key fields:** checkpoint path, n_samples, nfe, output dirs, metric selection

## 5. Data and I/O

- **Input:** Trained MeanFlow checkpoint (from Phase 4), test set real volumes
- **Output:**
  - `experiments/stage1_healthy/generated/` — 2,000 `.nii.gz` synthetic volumes
  - `experiments/stage1_healthy/metrics/metrics.json` — all computed metrics
  - `experiments/stage1_healthy/metrics/synthseg/` — SynthSeg labels and stats
  - Timing logs

## 6. Verification Tests

| Test ID | Description | Pass Criterion | Critical? | Implementation Hint |
|---|---|---|---|---|
| P5-T1 | 2000 samples generated without error | 2000 `.nii.gz` files in output dir | CRITICAL | Count files after generation |
| P5-T2 | FID (slice-wise, axial) < 50 | Competitive with published 3D brain MRI methods | CRITICAL | Check `metrics.json` |
| P5-T3 | SSIM (mean over volumes) > 0.70 | Generated volumes have structural similarity to real | CRITICAL | Check `metrics.json` |
| P5-T4 | SynthSeg runs on generated volumes | No SynthSeg failures | DESIRABLE | Check SynthSeg logs |
| P5-T5 | SynthSeg regional volumes correlate with real | Pearson $r > 0.7$ for hippocampus, ventricles | DESIRABLE | Check `metrics.json` |
| P5-T6 | Sampling speed < 2 seconds per volume (A100) | 1-NFE is fast | CRITICAL | Timing log |

**Phase 5 is PASSED when P5-T1, P5-T2, P5-T3, P5-T6 are ALL green. P5-T4 and P5-T5 are desirable.**

## 7. Expected Outputs

- `src/neuromf/metrics/fid.py`
- `src/neuromf/metrics/fid_3d.py`
- `src/neuromf/metrics/ssim_psnr.py`
- `src/neuromf/metrics/synthseg_metrics.py`
- `src/neuromf/metrics/spectral.py`
- `experiments/cli/generate.py`
- `experiments/cli/evaluate.py`
- `configs/evaluate.yaml`
- `experiments/stage1_healthy/generated/` — synthetic volumes
- `experiments/stage1_healthy/metrics/` — all metrics
- `experiments/phase_5/verification_report.md`

## 8. Failure Modes and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| FID > 50 | Weakens paper | Medium | Check sample quality visually; may need more training epochs; compare with VAE reconstruction FID as upper bound |
| SynthSeg fails on synthetic volumes | Limits evaluation | Low | Use alternative: FreeSurfer or manual QC |
| Generation OOM | Blocks P5-T1 | Low | Generate in smaller batches (e.g., 8 at a time) |
| Slice-wise FID not representative | Misleading metrics | Medium | Also compute 3D-FID; report both |
