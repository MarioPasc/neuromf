#!/usr/bin/env bash
#SBATCH -J neuromf_p4_train
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# =============================================================================
# PHASE 4: MEANFLOW TRAINING WORKER
#
# Trains the Latent MeanFlow model on pre-computed latents using a single A100.
# Produces model checkpoints, TensorBoard logs, and periodic sample images.
#
# Expected env vars (exported by train.sh launcher):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME, RESUME_CKPT
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 4 training started at: $(date)"
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
python -c "import pytorch_lightning; print('Lightning', pytorch_lightning.__version__)"

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
for f in "${CONFIGS_DIR}/base.yaml" "${REPO_SRC}/configs/train_meanflow.yaml" "${CONFIGS_DIR}/train_meanflow.yaml"; do
    if [ -f "$f" ]; then
        echo "[OK]   $f"
    else
        echo "[MISS] $f"
        echo "ERROR: Required config file missing. Aborting."
        exit 1
    fi
done

# Verify latent dir has .pt files and stats
LATENT_DIR="${RESULTS_DST}/latents"
PT_COUNT=$(find "${LATENT_DIR}" -name "*.pt" 2>/dev/null | wc -l)
echo "Latent .pt files: ${PT_COUNT}"
if [ "$PT_COUNT" -eq 0 ]; then
    echo "ERROR: No .pt files found in ${LATENT_DIR}. Run Phase 1 first."
    exit 1
fi

if [ ! -f "${LATENT_DIR}/latent_stats.json" ]; then
    echo "ERROR: latent_stats.json not found in ${LATENT_DIR}. Run Phase 1 first."
    exit 1
fi
echo "[OK]   ${LATENT_DIR}/latent_stats.json"

# Quick import check
python -c "
from neuromf.models.latent_meanflow import LatentMeanFlow
from neuromf.data.latent_dataset import LatentDataset, latent_collate_fn
from neuromf.wrappers.maisi_unet import MAISIUNetWrapper
from neuromf.wrappers.meanflow_loss import MeanFlowPipeline
print('All imports OK')
"

if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED — aborting."
    exit 1
fi

# ========================================================================
# RUN TRAINING
# ========================================================================
echo ""
echo "=========================================="
echo "RUNNING PHASE 4 TRAINING"
echo "=========================================="
echo "Config dir: ${CONFIGS_DIR}"
echo "Checkpoints: ${RESULTS_DST}/training_checkpoints/"
echo "Logs:        ${RESULTS_DST}/phase_4/logs/"
echo "Samples:     ${RESULTS_DST}/phase_4/samples/"
echo ""

TRAIN_CMD="python experiments/cli/train.py \
    --config ${CONFIGS_DIR}/train_meanflow.yaml \
    --configs-dir ${CONFIGS_DIR}"

# Add resume flag if checkpoint specified
if [ -n "${RESUME_CKPT:-}" ] && [ -f "${RESUME_CKPT}" ]; then
    echo "Resuming from: ${RESUME_CKPT}"
    TRAIN_CMD="${TRAIN_CMD} --resume ${RESUME_CKPT}"
fi

eval "${TRAIN_CMD}"

TRAIN_EXIT=$?

# ========================================================================
# POST-FLIGHT: Verify outputs
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

CKPT_DIR="${RESULTS_DST}/training_checkpoints"
CKPT_COUNT=$(find "${CKPT_DIR}" -name "*.ckpt" 2>/dev/null | wc -l)
echo "Checkpoint files: ${CKPT_COUNT}"

if [ -f "${CKPT_DIR}/last.ckpt" ]; then
    SIZE=$(stat -c%s "${CKPT_DIR}/last.ckpt" 2>/dev/null || echo "?")
    echo "[OK]   last.ckpt (${SIZE} bytes)"
else
    echo "[MISS] last.ckpt"
fi

SAMPLE_COUNT=$(find "${RESULTS_DST}/phase_4/samples" -name "*.png" 2>/dev/null | wc -l)
echo "Sample images: ${SAMPLE_COUNT}"

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PHASE 4 TRAINING COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Checkpts:   ${CKPT_COUNT} files in ${CKPT_DIR}"
echo "Samples:    ${SAMPLE_COUNT} images"
echo "Exit code:  ${TRAIN_EXIT}"

if [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "Phase 4 training completed successfully."
else
    echo "Phase 4 training FAILED with exit code ${TRAIN_EXIT}."
    exit "${TRAIN_EXIT}"
fi
