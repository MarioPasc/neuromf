#!/usr/bin/env bash
#SBATCH -J neuromf_p1_encode
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=encode_%j.out
#SBATCH --error=encode_%j.err

# =============================================================================
# PHASE 1: LATENT ENCODING WORKER
#
# Encodes all FOMO-60K volumes through the frozen MAISI VAE on a single A100.
# Produces .pt latent files, latent_stats.json, figures, and reports.
#
# Expected env vars (exported by encode_dataset.sh launcher):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 1 encoding started at: $(date)"
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
for f in "${CONFIGS_DIR}/base.yaml" "${CONFIGS_DIR}/fomo60k.yaml" "${CONFIGS_DIR}/encode_dataset.yaml"; do
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
from neuromf.utils.latent_stats import LatentStatsAccumulator
print('All imports OK')
"

# Verify VAE weights exist
python -c "
from omegaconf import OmegaConf
from pathlib import Path
cfg = OmegaConf.merge(
    OmegaConf.load('${CONFIGS_DIR}/base.yaml'),
    OmegaConf.load('${CONFIGS_DIR}/fomo60k.yaml'),
    OmegaConf.load('${CONFIGS_DIR}/encode_dataset.yaml'),
)
OmegaConf.resolve(cfg)
w = Path(cfg.paths.maisi_vae_weights)
d = Path(cfg.paths.fomo60k_root)
print(f'VAE weights: {w} (exists={w.exists()})')
print(f'FOMO-60K root: {d} (exists={d.exists()})')
assert w.exists(), f'VAE weights not found at {w}'
assert d.exists(), f'FOMO-60K not found at {d}'
print('Pre-flight checks PASSED')
"

if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED — aborting."
    exit 1
fi

# ========================================================================
# RUN ENCODING
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING PHASE 1 ENCODING"
echo "=========================================="
echo "Config dir: ${CONFIGS_DIR}"
echo "Output:     ${RESULTS_DST}/latents/"
echo ""

python experiments/cli/encode_dataset.py \
    --config "${CONFIGS_DIR}/encode_dataset.yaml" \
    --configs-dir "${CONFIGS_DIR}"

ENCODE_EXIT=$?

# ========================================================================
# POST-FLIGHT: Verify outputs
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

LATENT_DIR="${RESULTS_DST}/latents"
PT_COUNT=$(find "${LATENT_DIR}" -name "*.pt" 2>/dev/null | wc -l)
echo "Latent .pt files: ${PT_COUNT}"

EXPECTED_FILES=(
    "${LATENT_DIR}/latent_stats.json"
    "${RESULTS_DST}/phase_1/encoding_log.json"
    "${RESULTS_DST}/phase_1/figures/latent_histograms.png"
    "${RESULTS_DST}/phase_1/figures/channel_stats_bar.png"
    "${RESULTS_DST}/phase_1/figures/correlation_heatmap.png"
    "${RESULTS_DST}/phase_1/verification_report.html"
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

# Show latent stats summary
if [ -f "${LATENT_DIR}/latent_stats.json" ]; then
    echo ""
    echo "Latent stats summary:"
    python -c "
import json
with open('${LATENT_DIR}/latent_stats.json') as f:
    stats = json.load(f)
print(f'  Files: {stats[\"n_files\"]}')
print(f'  Voxels/channel: {stats[\"n_voxels_per_channel\"]}')
for ch, s in stats['per_channel'].items():
    print(f'  {ch}: mean={s[\"mean\"]:.4f}, std={s[\"std\"]:.4f}, skew={s[\"skewness\"]:.4f}, kurt={s[\"kurtosis\"]:.4f}')
"
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PHASE 1 ENCODING COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Latents:    ${PT_COUNT} files in ${LATENT_DIR}"
echo "Exit code:  ${ENCODE_EXIT}"

if [ "$ENCODE_EXIT" -eq 0 ]; then
    echo "Phase 1 encoding completed successfully."
else
    echo "Phase 1 encoding FAILED with exit code ${ENCODE_EXIT}."
    exit "${ENCODE_EXIT}"
fi
