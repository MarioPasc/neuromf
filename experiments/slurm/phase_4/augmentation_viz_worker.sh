#!/usr/bin/env bash
#SBATCH -J neuromf_aug_viz
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=aug_viz_%j.out
#SBATCH --error=aug_viz_%j.err

# =============================================================================
# AUGMENTATION VISUALISATION SUITE — SLURM WORKER
#
# Runs the full augmentation visualisation suite (6 visualisations, 18 PNGs).
# ~113 VAE decodes, ~15 min on 1 A100 40GB.
#
# Expected env vars (exported by augmentation_viz.sh launcher):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME, VIZ_SEED
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Augmentation visualisation started at: $(date)"
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
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Verify configs
for f in "${CONFIGS_DIR}/base.yaml" "${REPO_SRC}/configs/train_meanflow.yaml" "${CONFIGS_DIR}/train_meanflow.yaml"; do
    if [ -f "$f" ]; then
        echo "[OK]   $f"
    else
        echo "[MISS] $f"
        echo "ERROR: Required config file missing. Aborting."
        exit 1
    fi
done

# Verify latent shards and stats
LATENT_DIR="${RESULTS_DST}/latents"
H5_COUNT=$(find "${LATENT_DIR}" -name "*.h5" 2>/dev/null | wc -l)
echo "Latent HDF5 shards: ${H5_COUNT}"
if [ "$H5_COUNT" -eq 0 ]; then
    echo "ERROR: No .h5 shard files found in ${LATENT_DIR}. Run Phase 1 first."
    exit 1
fi

if [ ! -f "${LATENT_DIR}/latent_stats.json" ]; then
    echo "ERROR: latent_stats.json not found. Run Phase 1 first."
    exit 1
fi
echo "[OK]   ${LATENT_DIR}/latent_stats.json"

# Quick import check
python -c "
from neuromf.wrappers.maisi_vae import MAISIVAEWrapper
from neuromf.data.latent_dataset import LatentDataset
from neuromf.data.latent_augmentation import PerChannelGaussianNoise
from neuromf.utils.latent_stats import load_latent_stats
print('All imports OK')
"

if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED — aborting."
    exit 1
fi

# ========================================================================
# RUN VISUALISATION SUITE
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING AUGMENTATION VISUALISATION SUITE"
echo "=========================================="

OUTPUT_DIR="${RESULTS_DST}/phase_4/augmentation_viz"
mkdir -p "${OUTPUT_DIR}"

python scripts/augmentation_viz/run_all.py \
    --configs-dir "${CONFIGS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${VIZ_SEED}"

VIZ_EXIT=$?

# ========================================================================
# POST-FLIGHT: Verify outputs
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

PNG_COUNT=$(find "${OUTPUT_DIR}" -name "*.png" 2>/dev/null | wc -l)
echo "PNG files: ${PNG_COUNT} (expected: 18)"

if [ "$PNG_COUNT" -ge 18 ]; then
    echo "[OK] All 18 PNGs generated"
else
    echo "[WARN] Only ${PNG_COUNT}/18 PNGs found"
fi

ls -lh "${OUTPUT_DIR}"/*.png 2>/dev/null || echo "No PNGs found"

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "AUGMENTATION VISUALISATION COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 60))m $(($ELAPSED % 60))s"
echo "PNGs:       ${PNG_COUNT}"
echo "Output:     ${OUTPUT_DIR}"
echo "Exit code:  ${VIZ_EXIT}"

if [ "$VIZ_EXIT" -eq 0 ]; then
    echo "Augmentation visualisation completed successfully."
else
    echo "Augmentation visualisation FAILED with exit code ${VIZ_EXIT}."
    exit "${VIZ_EXIT}"
fi
