# Phase 1: Latent Pre-computation Pipeline

**Depends on:** Phase 0 (gate must be OPEN)
**Modules touched:** `src/neuromf/data/`, `src/neuromf/utils/`, `tests/`, `configs/`, `experiments/cli/`
**Estimated effort:** 1–2 sessions

---

## 1. Objective

Build the MRI preprocessing pipeline, encode all training volumes through the frozen MAISI VAE, store latents as `.pt` files, and compute per-channel latent statistics. This decouples the VAE from the training loop — standard LDM practice that eliminates VAE overhead during MeanFlow training.

## 2. Theoretical Background

From `docs/main/methodology_expanded.md`:

### §2.1 The Latent Diffusion / Latent Flow Matching Paradigm

Let $\mathcal{E}_\phi: \mathbb{R}^{1 \times H \times W \times D} \to \mathbb{R}^{C \times H' \times W' \times D'}$ and $\mathcal{D}_\phi: \mathbb{R}^{C \times H' \times W' \times D'} \to \mathbb{R}^{1 \times H \times W \times D}$ denote the frozen encoder and decoder of a pretrained VAE (MAISI), where $H' = H/f$, $W' = W/f$, $D' = D/f$ for spatial downsampling factor $f$. The encoder maps MRI volumes $\mathbf{x} \in \mathcal{X}$ to latent representations $\mathbf{z}_0 = \mathcal{E}_\phi(\mathbf{x}) \in \mathcal{Z}$.

The latent interpolation used for flow matching is:

$$
\mathbf{z}_t = (1 - t)\,\mathbf{z}_0 + t\,\boldsymbol{\epsilon}, \quad \mathbf{z}_0 = \mathcal{E}_\phi(\mathbf{x}),\; \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\tag{17}
$$

### §2.4 VAE Nonlinearity and Its Implications

**Distribution shape in latent space.** The MAISI VAE is trained with a KL regulariser calibrated so that the marginal latent distribution $q(\mathbf{z}) = \mathbb{E}_{\mathbf{x} \sim p_0}[q_\phi(\mathbf{z} | \mathbf{x})]$ has per-channel statistics $\mu_c \approx 0$, $\sigma_c \in [0.9, 1.1]$. However, $q(\mathbf{z})$ is not exactly Gaussian — it is a mixture of Gaussians (one per data point), which can exhibit heavy tails, multimodality, and non-trivial channel correlations.

**Implication:** Latent normalisation is important. Following standard LDM practice (Rombach et al., 2022), we compute per-channel statistics $(\mu_c, \sigma_c)$ from the training latents and normalise: $\tilde{\mathbf{z}}_0 = (\mathbf{z}_0 - \boldsymbol{\mu}) / \boldsymbol{\sigma}$. This ensures the flow endpoints are well-matched.

## 3. External Code to Leverage

### `src/external/NV-Generate-CTMR/`
- **Insights doc:** `docs/papers/maisi_2024/insights.md`
- **Specific files to study:** MONAI transform pipelines, preprocessing examples
- **What to extract/wrap:** Preprocessing transform chain (LoadImaged, EnsureChannelFirstd, Spacingd, etc.)
- **What to AVOID:** Do not replicate the full MAISI training pipeline; only use preprocessing transforms

## 4. Implementation Specification

### `src/neuromf/data/mri_preprocessing.py`
- **Purpose:** NIfTI → preprocessed 128^3 normalised tensor pipeline.
- **Key functions:**
  ```python
  def get_preprocessing_transforms(config: PreprocessingConfig) -> monai.transforms.Compose: ...
  def preprocess_volume(nifti_path: Path, config: PreprocessingConfig) -> torch.Tensor: ...
  ```
- **Pipeline:** NIfTI → SynthStrip skull-stripping → N4 bias correction → resample 1mm^3 → crop/pad 128^3 → normalise [0,1]
- **Dependencies:** MONAI transforms (`LoadImaged`, `EnsureChannelFirstd`, `Spacingd`, `CropForegroundd`, `Resized`, `ScaleIntensityd`)

### `src/neuromf/data/latent_dataset.py`
- **Purpose:** PyTorch Dataset of pre-computed `.pt` latents.
- **Key classes:**
  ```python
  class LatentDataset(Dataset):
      def __init__(self, latent_dir: Path, normalise: bool = True, stats_path: Optional[Path] = None) -> None: ...
      def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...
  ```
- **Each `.pt` file contains:** `{"z": tensor(4, 32, 32, 32), "metadata": {"subject_id": str, ...}}`

### `src/neuromf/utils/latent_stats.py`
- **Purpose:** Compute and store per-channel latent statistics.
- **Key functions:**
  ```python
  def compute_latent_stats(latent_dir: Path) -> dict: ...
  def save_latent_stats(stats: dict, output_path: Path) -> None: ...
  ```
- **Computes:** Per-channel mean, std, skewness, kurtosis; cross-channel correlation matrix; PCA explained variance ratio (top 50 components)

### `configs/encode_dataset.yaml`
- **Key fields:**
  - `vae.weights_path`: path to MAISI VAE weights
  - `data.dataset_root`: path to IXI/OASIS dataset
  - `data.output_dir`: path for `.pt` latent files
  - `data.batch_size`: 4
  - `preprocessing`: skull-strip, N4, spacing, crop, normalise options

### `experiments/cli/encode_dataset.py`
- **Purpose:** CLI to encode all volumes and compute stats.
- **Usage:** `python experiments/cli/encode_dataset.py --config configs/encode_dataset.yaml`

## 5. Data and I/O

- **Input:** IXI NIfTI brain MRI volumes (T1W), path from config
- **Output:**
  - `{output_dir}/{subject_id}.pt` — one file per volume, containing `{"z": tensor(4,32,32,32), "metadata": {...}}`
  - `{output_dir}/latent_stats.json` — per-channel statistics
- **Tensor shapes:** Input `(1, 128, 128, 128)` → Latent `(4, 32, 32, 32)` (no batch dim in saved file)

## 6. Verification Tests

| Test ID | Description | Pass Criterion | Critical? | Implementation Hint |
|---|---|---|---|---|
| P1-T1 | All volumes encode without error | 0 failures in batch encoding | CRITICAL | Check logs for exceptions during encoding |
| P1-T2 | Latent shape correct | Every `.pt` file has `z.shape == (4, 32, 32, 32)` | CRITICAL | Iterate all `.pt` files and assert shape |
| P1-T3 | Per-channel mean ≈ 0 | \|μ_c\| < 0.5 for all c | CRITICAL | Check `latent_stats.json` |
| P1-T4 | Per-channel std ∈ [0.5, 2.0] | σ_c ∈ [0.5, 2.0] for all c | CRITICAL | Check `latent_stats.json` |
| P1-T5 | No NaN/Inf in latents | `torch.isfinite(z).all()` for all files | CRITICAL | Scan all `.pt` files |
| P1-T6 | Latent dataset loads correctly | `LatentDataset.__getitem__` returns correct shape and type | CRITICAL | Unit test with mock `.pt` files |
| P1-T7 | Round-trip: decode(load(.pt)) ≈ original | SSIM > 0.89 for 5 random volumes | CRITICAL | Decode stored latents and compare to originals |

**Suggested test file:** `tests/test_latent_dataset.py`

**Phase 1 is PASSED when ALL of P1-T1 through P1-T7 are green.**

## 7. Expected Outputs

- `src/neuromf/data/mri_preprocessing.py` — preprocessing pipeline
- `src/neuromf/data/latent_dataset.py` — latent dataset class
- `src/neuromf/utils/latent_stats.py` — statistics computation
- `configs/encode_dataset.yaml` — encoding config
- `experiments/cli/encode_dataset.py` — encoding CLI
- `tests/test_latent_dataset.py` — verification tests
- `{output_dir}/*.pt` — encoded latent files
- `{output_dir}/latent_stats.json` — per-channel statistics
- `experiments/phase_1/verification_report.md` — test results

## 8. Failure Modes and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| SynthStrip not available | Blocks preprocessing | Low | Install via `pip install synthstrip`; fallback to MONAI `SkullStripd` |
| N4 bias correction crashes on some volumes | Blocks P1-T1 | Low | Wrap in try/except, skip problematic volumes, log warnings |
| Latent stats far from expected (μ ≠ 0, σ ≠ 1) | Informational | Medium | This just means normalisation is needed; compute and apply |
| Disk space insufficient for all latents | Blocks P1-T1 | Low | Each latent is ~0.5 MB; 1000 volumes = ~500 MB. Check disk before starting |
