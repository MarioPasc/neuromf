#!/usr/bin/env bash
#SBATCH -J ddpm_gen
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# Expected env: REPO_SRC, CONDA_ENV_NAME, DDPM_RUN_DIR, DDPM_CHECKPOINT, DDPM_CONFIG, GEN_N_SAMPLES

set -euo pipefail
echo "DDPM generation started at: $(date)"

# Environment
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module."
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

export PYTHONPATH="${REPO_SRC}/src/external/MOTFM:${REPO_SRC}/experiments/DDPM:${REPO_SRC}/experiments/MOTFM:${PYTHONPATH:-}"

python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Pre-flight
[ ! -f "${DDPM_CHECKPOINT}" ] && echo "FATAL: Checkpoint not found: ${DDPM_CHECKPOINT}" >&2 && exit 1

N_SAMPLES="${GEN_N_SAMPLES:-500}"
GEN_DIR="${DDPM_RUN_DIR}/generation"
DDPM_VOL_DIR="${GEN_DIR}/volumes"
mkdir -p "${DDPM_VOL_DIR}"

# Stage 1: Real test volumes (same as MOTFM pipeline)
echo "=========================================="
echo "STAGE 1: Real test volumes"
echo "=========================================="
python -u experiments/cli/prepare_real_test.py \
    --config configs/picasso/generate.yaml \
    --configs-dir configs/picasso \
    --output-dir "${GEN_DIR}" 2>&1 || echo "[WARN] prepare_real_test failed (may already exist)"

# Stage 2: DDPM generation at NFE=1, 10, 50
echo "=========================================="
echo "STAGE 2: DDPM Generation"
echo "=========================================="
python -u experiments/DDPM/generate_volumes.py \
    --config "${DDPM_CONFIG}" \
    --checkpoint "${DDPM_CHECKPOINT}" \
    --nfe 1 10 50 \
    --n-samples "${N_SAMPLES}" \
    --output-dir "${DDPM_VOL_DIR}" \
    --volume-size 192

echo "=========================================="
echo "GENERATION COMPLETE"
echo "=========================================="
echo "Volumes: ${DDPM_VOL_DIR}"
ls -lh "${DDPM_VOL_DIR}"/*.h5 2>/dev/null || echo "(no h5 files)"
echo "Finished at: $(date)"
