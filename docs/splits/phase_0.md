# Phase 0: Environment Bootstrap and VAE Validation

**Depends on:** None (first phase)
**Modules touched:** `src/neuromf/wrappers/`, `tests/`, `configs/`, `experiments/cli/`, `experiments/vae_validation/`
**Estimated effort:** 1–2 sessions

---

## 1. Objective

Set up the conda environment, build the MAISI VAE wrapper, and validate that the frozen VAE faithfully reconstructs brain MRI at 128^3 resolution. This phase establishes the foundation encoder-decoder that all subsequent phases depend on.

## 2. Theoretical Background

From `docs/main/methodology_expanded.md` §6:

### 6.1 Architecture Summary

The MAISI VAE (Guo et al., 2024) is a 3D VAE-GAN with the following properties:

| Property | Value |
|---|---|
| Input | 1-channel 3D volume, multiples of 128 |
| Encoder stages | 3 stages of 2x strided 3D convolution → 4x spatial downsampling |
| Latent channels | 4 (with KL regularisation) |
| Decoder | Symmetric to encoder, with skip-less transposed convolutions |
| Training losses | L1 reconstruction + LPIPS perceptual + PatchGAN adversarial + KL |
| KL calibration | Adaptive KL weight to maintain σ_c ∈ [0.9, 1.1] |
| Training data | 37,243 CT + 17,887 MRI volumes (including brain MRI) |

### 6.2 Reconstruction Quality Validation Protocol

Before training any generative model, we must validate that the frozen MAISI VAE faithfully reconstructs brain MRI at our target resolution. The protocol:

1. Select a held-out set of N_val = 100 brain MRI volumes from IXI.
2. Preprocess: skull-strip, resample to 128^3 at 1mm^3 isotropic, intensity normalise to [0, 1].
3. Encode and decode: x̂ = D_φ(E_φ(x)).
4. Compute metrics:
   - **SSIM** (Structural Similarity Index): > 0.90 required
   - **PSNR** (Peak Signal-to-Noise Ratio): > 30 dB required
   - **LPIPS** (Learned Perceptual Image Patch Similarity): < 0.10 required (adapted for 3D slices)
5. Visual inspection of cortical boundaries (the known VAE smoothing region).

### 6.3 Latent Statistics Characterisation

After encoding the full training set, compute and report:
- Per-channel mean μ_c = E[z_{0,c}] and standard deviation σ_c = std(z_{0,c})
- Per-channel skewness γ_c and kurtosis κ_c (to quantify deviation from Gaussianity)
- Channel correlation matrix R ∈ R^{4×4} where R_{ij} = corr(z_{0,i}, z_{0,j})
- PCA spectrum of the flattened latents (to estimate effective dimensionality)

## 3. External Code to Leverage

### `src/external/NV-Generate-CTMR/`

- **Insights doc:** `docs/papers/maisi_2024/insights.md` (run `/review-external maisi_2024 NV-Generate-CTMR` first)
- **Specific files to study:**
  - VAE architecture definition and weight loading scripts
  - Preprocessing pipeline examples (MONAI transforms)
  - Configuration files for model instantiation
- **What to extract/wrap:** VAE model class instantiation, weight loading, encode/decode methods
- **What to AVOID:** Do not use the diffusion UNet training scripts from this repo (we use MeanFlow, not DDPM/Rectified Flow for generation)

## 4. Implementation Specification

### `src/neuromf/wrappers/maisi_vae.py`
- **Purpose:** Frozen MAISI VAE encoder-decoder wrapper for 3D medical volumes.
- **Key classes/functions:**
  ```python
  @dataclass
  class MAISIVAEConfig:
      weights_path: str
      spatial_dims: int = 3
      in_channels: int = 1
      latent_channels: int = 4
      downsample_factor: int = 4

  class MAISIVAEWrapper:
      def __init__(self, config: MAISIVAEConfig) -> None: ...
      def encode(self, x: torch.Tensor) -> torch.Tensor: ...
      def decode(self, z: torch.Tensor) -> torch.Tensor: ...
      def reconstruct(self, x: torch.Tensor) -> torch.Tensor: ...
  ```
- **Dependencies:** MONAI's `AutoencoderKL` (or equivalent from NV-Generate-CTMR), torch
- **Wraps:** MAISI VAE from `src/external/NV-Generate-CTMR/`
- **Requirements:** All params frozen (`.requires_grad_(False)`), bf16 inference support, shape assertions (128^3 input → 4×32^3 latent)

### `configs/vae_validation.yaml`
- **Key fields:**
  - `vae.weights_path`: path to MAISI VAE weights
  - `data.dataset_root`: path to IXI dataset
  - `data.n_validation`: 20 (quick) or 100 (full)
  - `output_dir`: `experiments/vae_validation/`

### `experiments/cli/validate_vae.py`
- **Purpose:** CLI script that loads VAE, encodes+decodes IXI volumes, computes metrics.
- **Output:** `experiments/vae_validation/metrics.json` with SSIM, PSNR, LPIPS per volume.

### `tests/test_maisi_vae_wrapper.py`
- **Purpose:** Unit tests for the VAE wrapper (P0-T1 through P0-T7).

## 5. Data and I/O

- **Input:** IXI NIfTI brain MRI volumes (T1W), path configured in `configs/vae_validation.yaml`
- **Preprocessing:** skull-strip (SynthStrip), N4 bias correction, resample to 1mm^3, crop/pad to 128^3, normalise [0,1]
- **Output:**
  - `experiments/vae_validation/metrics.json` — per-volume SSIM, PSNR, LPIPS
  - `experiments/vae_validation/reconstructions/` — sample reconstructed NIfTI files for visual inspection
- **Tensor shapes:** Input `(B, 1, 128, 128, 128)` → Latent `(B, 4, 32, 32, 32)` → Reconstructed `(B, 1, 128, 128, 128)`

## 6. Verification Tests

| Test ID | Description | Pass Criterion | Critical? | Implementation Hint |
|---|---|---|---|---|
| P0-T1 | MAISI VAE weights load without error | No exceptions | CRITICAL | `python -c "from neuromf.wrappers.maisi_vae import MAISIVAEWrapper; w = MAISIVAEWrapper(config)"` |
| P0-T2 | Encode produces correct shape | `z.shape == (B, 4, 32, 32, 32)` for `x.shape == (B, 1, 128, 128, 128)` | CRITICAL | Unit test with random tensor |
| P0-T3 | Decode produces correct shape | `x_hat.shape == (B, 1, 128, 128, 128)` for `z.shape == (B, 4, 32, 32, 32)` | CRITICAL | Unit test with random tensor |
| P0-T4 | Round-trip reconstruction SSIM > 0.90 | Mean SSIM over 20 IXI volumes > 0.90 | CRITICAL | Run `validate_vae.py`, check `metrics.json` |
| P0-T5 | Round-trip PSNR > 30 dB | Mean PSNR > 30.0 | CRITICAL | Same as P0-T4 |
| P0-T6 | VAE is frozen (no grads) | All `param.requires_grad == False` | CRITICAL | Assert all VAE params frozen |
| P0-T7 | bf16 inference works | No NaN/Inf in output | CRITICAL | Test with `torch.autocast("cuda", dtype=torch.bfloat16)` |

**Suggested test file:** `tests/test_maisi_vae_wrapper.py`

**Phase 0 is PASSED when ALL of P0-T1 through P0-T7 are green.**

## 7. Expected Outputs

- `src/neuromf/wrappers/maisi_vae.py` — VAE wrapper module
- `configs/vae_validation.yaml` — validation config
- `experiments/cli/validate_vae.py` — validation CLI script
- `tests/test_maisi_vae_wrapper.py` — verification tests
- `experiments/vae_validation/metrics.json` — reconstruction quality metrics
- `experiments/vae_validation/reconstructions/` — sample reconstructions
- `experiments/phase_0/verification_report.md` — test results

## 8. Failure Modes and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| MAISI VAE poorly reconstructs FLAIR | Blocks Phase 0 | Low | VAE was trained on MRI including FLAIR; fine-tune if needed |
| Weight loading fails due to architecture mismatch | Blocks Phase 0 | Low | Check `NV-Generate-CTMR/` for exact model class and config; match architecture params exactly |
| bf16 produces NaN on certain inputs | Blocks P0-T7 | Low | Fall back to fp32 for validation; investigate which layers are numerically unstable |
