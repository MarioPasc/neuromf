#!/usr/bin/env bash
#SBATCH -J neuromf_p4_train
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:2
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# =============================================================================
# PHASE 4: MEANFLOW TRAINING WORKER
#
# Trains the Latent MeanFlow model on pre-computed latents.
# Supports single-GPU and multi-GPU (DDP) within one DGX node.
# Lightning handles process spawning — no srun needed.
#
# Dataset: ~6,471 scans (8 datasets), 3-way split (85/10/5)
# ~5,500 train scans -> ~43 steps/epoch -> ~12,900 total steps at 300 epochs
#
# Expected env vars (exported by train.sh launcher):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME, RESUME_CKPT, N_GPUS
#   TRAIN_CONFIG (optional): space-separated config paths for --config.
#     Defaults to "${CONFIGS_DIR}/train_meanflow.yaml" for backward compat.
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 4 training started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"

# ========================================================================
# NCCL CONFIGURATION (multi-GPU communication)
# ========================================================================
# NCCL is the GPU communication backend for DDP. Within a DGX node,
# GPUs communicate over NVLink (fast) with no special tuning needed.
# These settings provide safe defaults and useful debug info.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"       # WARN for production, INFO for debugging
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}" # Enable InfiniBand if available
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" # Enable P2P (NVLink) transfers

# Avoid NCCL timeout on slow filesystems during checkpoint saving
export TORCH_NCCL_BLOCKING_WAIT=0

N_GPUS="${N_GPUS:-2}"
echo "GPUs requested: ${N_GPUS}"

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

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "NCCL_DEBUG=${NCCL_DEBUG}"

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

# Verify latent dir has HDF5 shard files and stats
LATENT_DIR="${RESULTS_DST}/latents"
H5_COUNT=$(find "${LATENT_DIR}" -name "*.h5" 2>/dev/null | wc -l)
echo "Latent HDF5 shards: ${H5_COUNT}"
if [ "$H5_COUNT" -eq 0 ]; then
    echo "ERROR: No .h5 shard files found in ${LATENT_DIR}. Run Phase 1 first."
    exit 1
fi

if [ ! -f "${LATENT_DIR}/latent_stats.json" ]; then
    echo "ERROR: latent_stats.json not found in ${LATENT_DIR}. Run Phase 1 first."
    exit 1
fi
echo "[OK]   ${LATENT_DIR}/latent_stats.json"

# Verify GPU count matches request
VISIBLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Visible GPUs: ${VISIBLE_GPUS} (requested: ${N_GPUS})"
if [ "$VISIBLE_GPUS" -lt "$N_GPUS" ]; then
    echo "WARNING: Only ${VISIBLE_GPUS} GPUs visible but ${N_GPUS} requested."
    echo "         Training will use ${VISIBLE_GPUS} GPUs."
fi

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
echo "Config dir:  ${CONFIGS_DIR}"
echo "GPUs:        ${N_GPUS}"
echo "Checkpoints: ${RESULTS_DST}/training_checkpoints/"
echo "Logs:        ${RESULTS_DST}/phase_4/logs/"
echo "Samples:     ${RESULTS_DST}/phase_4/samples/"
echo ""

# TRAIN_CONFIG allows ablation launchers to inject extra config layers
TRAIN_CONFIG="${TRAIN_CONFIG:-${CONFIGS_DIR}/train_meanflow.yaml}"
echo "Config(s):   ${TRAIN_CONFIG}"

TRAIN_CMD="python experiments/cli/train.py \
    --config ${TRAIN_CONFIG} \
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
echo "GPUs:       ${N_GPUS}"
echo "Checkpts:   ${CKPT_COUNT} files in ${CKPT_DIR}"
echo "Samples:    ${SAMPLE_COUNT} images"
echo "Exit code:  ${TRAIN_EXIT}"

if [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "Phase 4 training completed successfully."
else
    echo "Phase 4 training FAILED with exit code ${TRAIN_EXIT}."
    exit "${TRAIN_EXIT}"
fi
