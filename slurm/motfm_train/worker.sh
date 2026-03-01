#!/usr/bin/env bash
#SBATCH -J motfm_train
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:4
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# =============================================================================
# MOTFM TRAINING WORKER
#
# Trains MOTFM on FOMO-60K (unconditional 3D, 192^3) using the vendored
# trainer.py from src/external/MOTFM/. No modifications to vendored code.
#
# Supports multi-GPU DDP. Lightning auto-detects SLURM-allocated GPUs via
# devices="auto" in config. DDP strategy is resolved by _resolve_strategy().
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME, MOTFM_CONFIG
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "MOTFM training started at: $(date)"
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

# Add MOTFM + experiments/MOTFM to PYTHONPATH
export PYTHONPATH="${REPO_SRC}/src/external/MOTFM:${REPO_SRC}/experiments/MOTFM:${PYTHONPATH:-}"

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
python -c "import flow_matching; print('flow_matching OK')"
python -c "from generative.networks.nets import DiffusionModelUNet; print('monai-generative OK')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

VISIBLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Visible GPUs: ${VISIBLE_GPUS}"
if [ "$VISIBLE_GPUS" -gt 1 ]; then
    echo "Multi-GPU DDP: ${VISIBLE_GPUS} GPUs detected (eff. batch = 1 × 1 × ${VISIBLE_GPUS} = ${VISIBLE_GPUS})"
fi

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Verify config
if [ -f "${MOTFM_CONFIG}" ]; then
    echo "[OK]   Config: ${MOTFM_CONFIG}"
else
    echo "[MISS] Config not found: ${MOTFM_CONFIG}"
    exit 1
fi

# Verify HDF5 data file (primary — required by train_motfm.py)
H5_PATH=$(python -c "
import yaml
with open('${MOTFM_CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['data_args'].get('h5_path', ''))
")

if [ -n "${H5_PATH}" ] && [ -f "${H5_PATH}" ]; then
    SIZE=$(stat -c%s "${H5_PATH}" 2>/dev/null || echo "?")
    SIZE_GB=$(echo "scale=2; ${SIZE} / 1073741824" | bc 2>/dev/null || echo "?")
    echo "[OK]   HDF5 data: ${H5_PATH} (${SIZE_GB} GB)"
else
    echo "[MISS] HDF5 data not found: ${H5_PATH}"
    echo "       Run motfm_prepare first."
    exit 1
fi

# Quick import check
python -c "
from trainer import FlowMatchingLightningModule
from utils.utils_fm import build_model
from h5_dataset import H5VolumeDataset
print('MOTFM + H5 dataset imports OK')
"

# Pickle is optional — only needed for MOTFM's original trainer, not train_motfm.py
PICKLE_PATH=$(python -c "
import yaml
with open('${MOTFM_CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['data_args'].get('pickle_path', ''))
")
if [ -n "${PICKLE_PATH}" ] && [ -f "${PICKLE_PATH}" ]; then
    echo "[INFO] Pickle also available: ${PICKLE_PATH} (not used by train_motfm.py)"
else
    echo "[INFO] No pickle file (not needed — using HDF5-backed lazy loading)"
fi

# ========================================================================
# TRAINING
# ========================================================================
echo ""
echo "=========================================="
echo "TRAINING MOTFM"
echo "=========================================="
echo "Config: ${MOTFM_CONFIG}"
echo "Using memory-efficient H5-backed training wrapper (train_motfm.py)"

# Log system memory before training
echo "System memory:"
free -h

python -u "${REPO_SRC}/experiments/MOTFM/train_motfm.py" \
    --config_path "${MOTFM_CONFIG}"

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

CKPT_DIR="${RESULTS_DST}/motfm/checkpoints/fomo60k_unconditional_3d"

if [ -d "${CKPT_DIR}" ]; then
    echo "Checkpoint directory: ${CKPT_DIR}"
    ls -lh "${CKPT_DIR}"/*.ckpt 2>/dev/null || echo "  (no .ckpt files found)"

    if [ -f "${CKPT_DIR}/last.ckpt" ]; then
        SIZE=$(stat -c%s "${CKPT_DIR}/last.ckpt" 2>/dev/null || echo "?")
        SIZE_MB=$(echo "scale=1; ${SIZE} / 1048576" | bc 2>/dev/null || echo "?")
        echo "[OK]   last.ckpt (${SIZE_MB} MB)"
    else
        echo "[WARN] last.ckpt not found"
    fi
else
    echo "[WARN] Checkpoint directory not found: ${CKPT_DIR}"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "MOTFM TRAINING COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
