#!/usr/bin/env bash
#SBATCH -J neuromf_p0_vae
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=vae_validation_%j.out
#SBATCH --error=vae_validation_%j.err

# =============================================================================
# PHASE 0: VAE VALIDATION WORKER
#
# Validates the frozen MAISI VAE on 20 FOMO-60K volumes at 192^3 resolution.
# Produces reconstruction metrics (SSIM, PSNR), negative controls, latent
# statistics, figures, and HTML/Markdown reports.
#
# Expected env vars (exported by validate_vae.sh launcher):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 0 VAE validation started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Devices:', torch.cuda.device_count())"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Verify configs exist
for f in "${CONFIGS_DIR}/base.yaml" "${CONFIGS_DIR}/fomo60k.yaml" "${CONFIGS_DIR}/vae_validation.yaml"; do
    if [ -f "$f" ]; then
        echo "[OK]   $f"
    else
        echo "[MISS] $f"
        echo "ERROR: Required config file missing. Aborting."
        exit 1
    fi
done

# Quick import check
python -c "
from neuromf.wrappers.maisi_vae import MAISIVAEWrapper, MAISIVAEConfig
from neuromf.data.fomo60k import get_fomo60k_file_list, FOMO60KConfig
from neuromf.data.mri_preprocessing import build_mri_preprocessing_from_config
from neuromf.metrics.ssim_psnr import compute_psnr, compute_ssim_3d
print('All imports OK')
"

# Verify VAE weights and data exist
python -c "
from omegaconf import OmegaConf
from pathlib import Path
cfg = OmegaConf.merge(
    OmegaConf.load('${CONFIGS_DIR}/base.yaml'),
    OmegaConf.load('${CONFIGS_DIR}/fomo60k.yaml'),
    OmegaConf.load('${CONFIGS_DIR}/vae_validation.yaml'),
)
OmegaConf.resolve(cfg)
w = Path(cfg.paths.maisi_vae_weights)
d = Path(cfg.paths.fomo60k_root)
print(f'VAE weights: {w} (exists={w.exists()})')
print(f'FOMO-60K root: {d} (exists={d.exists()})')
print(f'Target shape: {cfg.data.target_shape}')
print(f'num_splits: {cfg.vae.num_splits}')
assert w.exists(), f'VAE weights not found at {w}'
assert d.exists(), f'FOMO-60K not found at {d}'
print('Pre-flight checks PASSED')
"

if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED — aborting."
    exit 1
fi

# ========================================================================
# RUN VAE VALIDATION
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING PHASE 0 VAE VALIDATION"
echo "=========================================="
echo "Config dir: ${CONFIGS_DIR}"
echo "Output:     ${RESULTS_DST}/phase_0/"
echo ""

python experiments/cli/validate_vae.py \
    --config "${CONFIGS_DIR}/vae_validation.yaml" \
    --configs-dir "${CONFIGS_DIR}"

VALIDATE_EXIT=$?

# ========================================================================
# POST-FLIGHT: Verify outputs
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

EXPECTED_FILES=(
    "${RESULTS_DST}/phase_0/vae_validation/metrics/metrics.json"
    "${RESULTS_DST}/phase_0/vae_validation/latent_stats/latent_stats.json"
    "${RESULTS_DST}/phase_0/vae_validation/figures/ssim_distribution.png"
    "${RESULTS_DST}/phase_0/vae_validation/figures/psnr_distribution.png"
    "${RESULTS_DST}/phase_0/vae_validation/figures/latent_histograms.png"
    "${RESULTS_DST}/phase_0/vae_validation/figures/mean_error_heatmap.png"
    "${RESULTS_DST}/phase_0/verification_report.html"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   $f (${SIZE} bytes)"
    else
        echo "[MISS] $f"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo "WARNING: ${MISSING} expected files missing."
fi

# Show metrics summary
if [ -f "${RESULTS_DST}/phase_0/vae_validation/metrics/metrics.json" ]; then
    echo ""
    echo "Validation metrics summary:"
    python -c "
import json
with open('${RESULTS_DST}/phase_0/vae_validation/metrics/metrics.json') as f:
    m = json.load(f)
print(f'  Mean SSIM:  {m[\"mean_ssim\"]:.4f}  (threshold > 0.90)')
print(f'  Mean PSNR:  {m[\"mean_psnr\"]:.2f} dB  (threshold > 30.0)')
print(f'  Volumes:    {m[\"n_volumes\"]}')
print(f'  Time:       {m[\"elapsed_seconds\"]:.1f}s')
nc = m.get('negative_controls', {})
nc_pass = nc.get('all_pass', 'N/A')
print(f'  Neg. controls: {\"ALL PASS\" if nc_pass else \"SOME FAILED\"}')
"
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PHASE 0 VAE VALIDATION COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Exit code:  ${VALIDATE_EXIT}"

if [ "$VALIDATE_EXIT" -eq 0 ]; then
    echo "Phase 0 VAE validation completed successfully."
else
    echo "Phase 0 VAE validation FAILED with exit code ${VALIDATE_EXIT}."
    exit "${VALIDATE_EXIT}"
fi
