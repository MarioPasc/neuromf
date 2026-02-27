#!/usr/bin/env bash
#SBATCH -J motfm_train
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# =============================================================================
# MOTFM TRAINING WORKER
#
# Trains MOTFM on FOMO-60K (unconditional 3D, 192^3) using the vendored
# trainer.py from src/external/MOTFM/. No modifications to vendored code.
#
# Expected env vars (exported by train.sh):
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

# Add MOTFM to PYTHONPATH (required for trainer.py internal imports)
export PYTHONPATH="${REPO_SRC}/src/external/MOTFM:${PYTHONPATH:-}"

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import flow_matching; print('flow_matching OK')"
python -c "from generative.networks.nets import DiffusionModelUNet; print('monai-generative OK')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

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

# Verify data pickle
PICKLE_PATH=$(python -c "
import yaml
with open('${MOTFM_CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['data_args']['pickle_path'])
")

if [ -f "${PICKLE_PATH}" ]; then
    SIZE=$(stat -c%s "${PICKLE_PATH}" 2>/dev/null || echo "?")
    SIZE_GB=$(echo "scale=2; ${SIZE} / 1073741824" | bc 2>/dev/null || echo "?")
    echo "[OK]   Data pickle: ${PICKLE_PATH} (${SIZE_GB} GB)"
else
    echo "[MISS] Data pickle not found: ${PICKLE_PATH}"
    echo "       Run prepare_data.sh first."
    exit 1
fi

# Quick import check
python -c "
from trainer import FlowMatchingLightningModule, FlowMatchingDataModule
from utils.utils_fm import build_model
print('MOTFM trainer imports OK')
"

# ========================================================================
# TRAINING
# ========================================================================
echo ""
echo "=========================================="
echo "TRAINING MOTFM"
echo "=========================================="
echo "Config: ${MOTFM_CONFIG}"

python -u "${REPO_SRC}/src/external/MOTFM/trainer.py" \
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
