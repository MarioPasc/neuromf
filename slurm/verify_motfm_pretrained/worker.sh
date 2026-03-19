#!/usr/bin/env bash
#SBATCH -J motfm_verify
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# MOTFM PRETRAINED VERIFICATION WORKER
#
# Generates volumes from the author-provided MOTFM checkpoint at multiple NFE
# levels and saves mid-slice PNGs + a summary grid for visual inspection.
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONDA_ENV_NAME, VERIFY_CHECKPOINT, VERIFY_CONFIG, VERIFY_OUTPUT
# =============================================================================

set -euo pipefail

echo "MOTFM pretrained verification started at: $(date)"
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
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Add MOTFM + experiments to PYTHONPATH
export PYTHONPATH="${REPO_SRC}/src/external/MOTFM:${REPO_SRC}/experiments/MOTFM:${PYTHONPATH:-}"

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

echo "Config:     ${VERIFY_CONFIG}"
echo "Checkpoint: ${VERIFY_CHECKPOINT}"
echo "Output:     ${VERIFY_OUTPUT}"

if [ ! -f "${VERIFY_CONFIG}" ]; then
    echo "FATAL: Config not found: ${VERIFY_CONFIG}" >&2
    exit 1
fi

if [ ! -f "${VERIFY_CHECKPOINT}" ]; then
    echo "FATAL: Checkpoint not found: ${VERIFY_CHECKPOINT}" >&2
    exit 1
fi

python -c "
import torch
ckpt = torch.load('${VERIFY_CHECKPOINT}', map_location='cpu', weights_only=False)
print(f'Checkpoint keys: {list(ckpt.keys())}')
print(f'Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
print(f'Global step: {ckpt.get(\"global_step\", \"N/A\")}')
n_params = sum(v.numel() for v in ckpt['state_dict'].values())
print(f'Parameters: {n_params:,} ({n_params*4/1e6:.1f} MB fp32)')
"

echo "[OK] Pre-flight passed"

# ========================================================================
# GENERATE VERIFICATION VOLUMES
# ========================================================================
echo ""
echo "=========================================="
echo "GENERATING VERIFICATION VOLUMES"
echo "=========================================="

python -u experiments/MOTFM/verify_pretrained.py \
    --config "${VERIFY_CONFIG}" \
    --checkpoint "${VERIFY_CHECKPOINT}" \
    --nfe 1 2 5 10 50 \
    --n-samples 4 \
    --output-dir "${VERIFY_OUTPUT}" \
    --device auto

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="
echo "Output directory: ${VERIFY_OUTPUT}"
echo "  Summary grid:   ${VERIFY_OUTPUT}/summary_grid.png"
echo "  Volume stats:   ${VERIFY_OUTPUT}/volume_stats.txt"

if [ -f "${VERIFY_OUTPUT}/volume_stats.txt" ]; then
    echo ""
    echo "Volume statistics:"
    cat "${VERIFY_OUTPUT}/volume_stats.txt"
fi

END_TIME=$(date +%s)
echo ""
echo "Total wall time: $(( END_TIME - SLURM_JOB_START_TIME )) seconds"
echo "Finished at: $(date)"
