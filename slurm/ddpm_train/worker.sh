#!/usr/bin/env bash
#SBATCH -J ddpm_train
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --constraint=dgx

# =============================================================================
# DDPM BASELINE TRAINING WORKER
#
# Expected env vars (from launch.sh):
#   REPO_SRC, CONDA_ENV_NAME, DDPM_CONFIG, RESULTS_DST
# =============================================================================

set -euo pipefail

echo "DDPM baseline training started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"

# ========================================================================
# ENVIRONMENT
# ========================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module; assuming conda in PATH."

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[python] $(which python || true)"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# PYTHONPATH: MOTFM external (for build_model), DDPM experiments, MOTFM experiments (for H5DataModule)
export PYTHONPATH="${REPO_SRC}/src/external/MOTFM:${REPO_SRC}/experiments/DDPM:${REPO_SRC}/experiments/MOTFM:${PYTHONPATH:-}"

# ========================================================================
# PRE-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

echo "Config: ${DDPM_CONFIG}"
[ ! -f "${DDPM_CONFIG}" ] && echo "FATAL: Config not found" >&2 && exit 1

# Verify HDF5 data exists
H5_PATH=$(python -c "
import yaml
with open('${DDPM_CONFIG}') as f: c = yaml.safe_load(f)
print(c['data_args']['h5_path'])
")
echo "HDF5 path: ${H5_PATH}"
[ ! -f "${H5_PATH}" ] && echo "FATAL: HDF5 not found: ${H5_PATH}" >&2 && exit 1

python -c "
from ddpm_module import DDPMLightningModule
from train_motfm import H5DataModule
print('[OK] DDPM imports verified')
"

echo "[OK] Pre-flight passed"

# ========================================================================
# TRAINING
# ========================================================================
echo ""
echo "=========================================="
echo "STARTING DDPM TRAINING"
echo "=========================================="

python -u experiments/DDPM/train_ddpm.py --config_path "${DDPM_CONFIG}"

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="

CKPT_DIR=$(python -c "
import yaml
with open('${DDPM_CONFIG}') as f: c = yaml.safe_load(f)
print(c['train_args']['checkpoint_dir'])
")
echo "Checkpoint dir: ${CKPT_DIR}"
if [ -d "${CKPT_DIR}" ]; then
    echo "Checkpoints found:"
    ls -lh "${CKPT_DIR}"/fomo60k_unconditional_3d/*.ckpt 2>/dev/null || echo "  (no .ckpt files)"
fi

echo "Finished at: $(date)"
