# SLURM Job Reference

Atomic SLURM launcher+worker pairs. Each directory is a self-contained job that
can be submitted independently or composed by orchestration scripts.

## Convention

Every `launch.sh` follows the same pattern:

- **Env vars with Picasso defaults:** `REPO_SRC`, `CONFIGS_DIR`, `RESULTS_DST`, `CONDA_ENV_NAME`
  (all overridable)
- **Common CLI arg:** `--depends-on JOB_ID` for job chaining
- **Output discipline:** status messages to `stderr`; job ID only to `stdout`
  so orchestration scripts can capture it: `JOB_ID=$(bash slurm/X/launch.sh ...)`
- **Worker reference:** co-located `worker.sh`

## Quick Reference

| Directory | Purpose | Phase | Model / Component | Configs (Picasso) | CLI Entry Point | GPUs | Wall-time | Memory |
|-----------|---------|-------|-------------------|-------------------|-----------------|------|-----------|--------|
| `validate_vae/` | VAE reconstruction metrics | 0 | Frozen MAISI VAE | `base.yaml`, `fomo60k.yaml`, `vae_validation.yaml` | `experiments/cli/validate_vae.py` | 1 | 1h | 32G |
| `encode/` | Latent pre-computation | 1 | Frozen MAISI VAE | `base.yaml`, `fomo60k.yaml`, `encode_dataset.yaml` | `experiments/cli/encode_dataset.py` | 1 | 10h | 64G |
| `toroid_sweep/` | Toy ablation experiment | 2 | Toy MLP (CPU-only) | `experiments/toy_toroid/configs/{base,ablation_*}.yaml` | `experiments/toy_toroid/sweep.py` | 1* | 23h | 32G |
| `train/` | MeanFlow training | 4 | NeuroiMF (MAISI UNet) | `base.yaml`, `train_meanflow.yaml` | `experiments/cli/train.py` | 1-8 | 7d | 64G/GPU |
| `augmentation_viz/` | Augmentation figures | 4 | MAISI VAE (decode) | `base.yaml`, `train_meanflow.yaml` | `scripts/augmentation_viz/run_all.py` | 1 | 1h | 32G |
| `generate/` | NeuroiMF sample generation | 5 | NeuroiMF + MAISI VAE | `base.yaml`, `generate.yaml` | `generate_latents.py`, `prepare_real_test.py`, `decode_volumes.py`, `visualize_generation.py` | 1 | 24h | 128G |
| `generate_motfm/` | MOTFM sample generation | 5 | MOTFM baseline | `base.yaml`, `generate.yaml`, `motfm/fomo60k_unconditional_3d.yaml` | `experiments/MOTFM/generate_volumes.py` | 1 | 24h | 128G |
| `evaluate/` | Feature extraction + metrics | 5 | R3D-18 (FID) | `base.yaml`, `generate.yaml` | `experiments/cli/compute_metrics.py` | 1 | 23h | 128G |
| `analyze/` | Bootstrap CIs, figures, tables | 5 | None (statistical) | None (reads metrics JSONs) | `experiments/cli/run_analysis.py` | 1 | 6h | 64G |
| `skull_strip/` | HD-BET skull stripping | -- | HD-BET | None (CLI args only) | `scripts/skull_strip_defaced.py` | 1-N | 3h | 32G |
| `motfm_prepare/` | FOMO-60K to MOTFM HDF5 | -- | None (data conversion) | `base.yaml`, `generate.yaml` | `experiments/MOTFM/prepare_data.py` | 0 (CPU) | 6h | 500G |
| `motfm_train/` | MOTFM baseline training | -- | MOTFM (DiffusionModelUNet) | `motfm/fomo60k_unconditional_3d.yaml` | `experiments/MOTFM/train_motfm.py` | 1 | 3d | 64G |

\* Toroid sweep is CPU-only; GPU requested only for DGX node allocation.

## Detailed Usage

### validate_vae

Validates the frozen MAISI VAE on 20 FOMO-60K volumes. Produces SSIM, PSNR,
negative controls, latent stats, and an HTML report.

```bash
bash slurm/validate_vae/launch.sh
```

### encode

Encodes all FOMO-60K volumes through the frozen MAISI VAE into HDF5 latent
shards with per-channel statistics.

```bash
bash slurm/encode/launch.sh
```

### toroid_sweep

Runs the full Phase 2 toy experiment: 18 training runs (ablations A-E),
NFE sweep, publication figures, CSV tables, and an HTML report.

```bash
bash slurm/toroid_sweep/launch.sh
```

### train

Trains the Latent MeanFlow (NeuroiMF) model on pre-computed latents.
Supports multi-GPU DDP via `N_GPUS` env var and checkpoint resumption.

```bash
bash slurm/train/launch.sh                           # 6 GPUs (default)
N_GPUS=4 bash slurm/train/launch.sh                  # 4 GPUs
bash slurm/train/launch.sh --resume /path/to/last.ckpt
```

The `TRAIN_CONFIG` env var can inject additional config layers for ablations:
```bash
TRAIN_CONFIG="${CONFIGS_DIR}/train_meanflow.yaml ${CONFIGS_DIR}/ablation_lp_sweep.yaml" \
  bash slurm/train/launch.sh
```

### augmentation_viz

Generates 18 PNGs (6 augmentation visualizations x 3 anatomical planes)
using latent augmentations decoded through the MAISI VAE.

```bash
bash slurm/augmentation_viz/launch.sh
VIZ_SEED=123 bash slurm/augmentation_viz/launch.sh
```

### generate

Generates NeuroiMF samples: latents at NFE 1/2/5/10/25/50, real test volumes
from original NIfTI, decoded volumes at NFE 1/10/50, and visualization figures.

```bash
bash slurm/generate/launch.sh --run-dir /path/to/NeuroiMF_01032026
bash slurm/generate/launch.sh --run-dir /path/to/run --n-samples 500
```

### generate_motfm

Generates MOTFM baseline samples: real test volumes, MOTFM volumes at
NFE 1/10/50, and visualization figures.

```bash
bash slurm/generate_motfm/launch.sh --run-dir /path/to/MOTFM_01032026
bash slurm/generate_motfm/launch.sh --run-dir /path --checkpoint /path/to/last.ckpt
```

### evaluate

Extracts R3D-18 features and computes metrics (FID-3D, MMD, Coverage,
Density, MS-SSIM, PSNR, spectral). Generic: works for both NeuroiMF and
MOTFM results directories.

```bash
bash slurm/evaluate/launch.sh --run-dir /path/to/NeuroiMF_01032026
bash slurm/evaluate/launch.sh --run-dir /path/to/MOTFM_01032026
```

### analyze

Runs post-hoc analysis: bootstrap confidence intervals, publication figures
(PDF+PNG), LaTeX tables, statistical tests, and manifest generation.

```bash
bash slurm/analyze/launch.sh
bash slurm/analyze/launch.sh --results-dir /path/to/run --n-bootstrap 1000
bash slurm/analyze/launch.sh --skip-qualitative --skip-bootstrap
bash slurm/analyze/launch.sh --nfe 1 10 50
```

### skull_strip

Runs HD-BET skull-stripping on defaced FOMO-60K datasets. Supports
Phase A (validation, 9 subjects, 1 GPU) and Phase B (batch, N parallel GPUs).
Phase B auto-submits a visualization dependency job.

```bash
bash slurm/skull_strip/launch.sh                    # Phase B, 3 workers
bash slurm/skull_strip/launch.sh --phase A          # Validation only
bash slurm/skull_strip/launch.sh --num-workers 6    # 6 parallel GPUs
```

### motfm_prepare

Converts FOMO-60K NIfTI volumes to MOTFM HDF5 format. CPU-only, high memory
(loads all volumes). Uses the same train/val/test split as Phase 1.

```bash
bash slurm/motfm_prepare/launch.sh
```

### motfm_train

Trains the MOTFM baseline on FOMO-60K using the vendored trainer with
HDF5-backed lazy loading.

```bash
bash slurm/motfm_train/launch.sh
bash slurm/motfm_train/launch.sh --depends-on 12345
```

## Orchestration

Multi-model dispatch scripts live in `experiments/slurm/phase_5/`:

| Script | What it does |
|--------|-------------|
| `experiments/slurm/phase_5/generate.sh` | Dispatches `slurm/generate/` and/or `slurm/generate_motfm/` |
| `experiments/slurm/phase_5/evaluate.sh` | Dispatches `slurm/evaluate/` once (NeuroiMF) or twice (+ MOTFM) |

```bash
# Generate both models
bash experiments/slurm/phase_5/generate.sh --motfm

# Generate MOTFM only
bash experiments/slurm/phase_5/generate.sh --motfm --skip-neuromf

# Evaluate both
bash experiments/slurm/phase_5/evaluate.sh --motfm \
    --results-dir .../NeuroiMF_01032026 \
    --motfm-results-dir .../MOTFM_01032026
```

## Typical Pipeline

```bash
# 1. Validate VAE
VAL_JOB=$(bash slurm/validate_vae/launch.sh)

# 2. Encode dataset
ENC_JOB=$(bash slurm/encode/launch.sh --depends-on $VAL_JOB)

# 3. Train NeuroiMF
TRAIN_JOB=$(bash slurm/train/launch.sh --depends-on $ENC_JOB)

# 4. Generate samples
GEN_JOB=$(bash slurm/generate/launch.sh \
    --run-dir .../NeuroiMF_01032026 --depends-on $TRAIN_JOB)

# 5. Evaluate
EVAL_JOB=$(bash slurm/evaluate/launch.sh \
    --run-dir .../NeuroiMF_01032026 --depends-on $GEN_JOB)

# 6. Analyze
bash slurm/analyze/launch.sh --depends-on $EVAL_JOB \
    --results-dir .../NeuroiMF_01032026
```

## MOTFM Baseline Pipeline

```bash
# 1. Prepare data (CPU-only, ~500G RAM)
PREP_JOB=$(bash slurm/motfm_prepare/launch.sh)

# 2. Train MOTFM
TRAIN_JOB=$(bash slurm/motfm_train/launch.sh --depends-on $PREP_JOB)

# 3. Generate + evaluate
GEN_JOB=$(bash slurm/generate_motfm/launch.sh \
    --run-dir .../MOTFM_01032026 --depends-on $TRAIN_JOB)
EVAL_JOB=$(bash slurm/evaluate/launch.sh \
    --run-dir .../MOTFM_01032026 --depends-on $GEN_JOB)
```
