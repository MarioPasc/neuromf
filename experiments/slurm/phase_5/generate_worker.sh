#!/usr/bin/env bash
#SBATCH -J neuromf_p5_gen
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=generate_%j.out
#SBATCH --error=generate_%j.err

# =============================================================================
# PHASE 5: GENERATION PIPELINE WORKER
#
# Stage 1: Generate latents at all NFE levels (1, 2, 5, 10, 25, 50)
# Stage 2: Prepare real test volumes
# Stage 3: Decode volumes at selected NFE levels (1, 10, 50)
# Stage 4: Visualization sanity-check figures (CPU-only)
#
# Expected env vars (exported by generate.sh):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 5 generation started at: $(date)"
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

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Verify checkpoint
CKPT_PATH="${RESULTS_DST}/ablations/xpred_exact_jvp/checkpoints/best_raw_epoch=589_train/raw_loss=1295477.5000.ckpt"
if [ -f "${CKPT_PATH}" ]; then
    echo "[OK]   Checkpoint: ${CKPT_PATH}"
else
    echo "[MISS] Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

# Verify latent stats
STATS_PATH="${RESULTS_DST}/latents/latent_stats.json"
if [ -f "${STATS_PATH}" ]; then
    echo "[OK]   Latent stats: ${STATS_PATH}"
else
    echo "[MISS] Latent stats not found: ${STATS_PATH}"
    exit 1
fi

# Quick import check
python -c "
from neuromf.generation import LatentGenerator, VolumeDecoder, H5Manager
from neuromf.wrappers.maisi_unet import MAISIUNetWrapper
from neuromf.wrappers.maisi_vae import MAISIVAEWrapper
print('All imports OK')
"

# ========================================================================
# STAGE 1: GENERATE LATENTS
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 1: GENERATING LATENTS"
echo "=========================================="

python experiments/cli/generate_latents.py \
    --config "${CONFIGS_DIR}/generate.yaml" \
    --configs-dir "${CONFIGS_DIR}" \
    --checkpoint "${CKPT_PATH}" \
    --use-ema \
    --nfe 1 2 5 10 25 50 \
    --n-samples 2000 \
    --batch-size 8

echo "Stage 1 complete."

# ========================================================================
# STAGE 2: PREPARE REAL TEST VOLUMES
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 2: PREPARING REAL TEST VOLUMES"
echo "=========================================="

python experiments/cli/prepare_real_test.py \
    --config "${CONFIGS_DIR}/generate.yaml" \
    --configs-dir "${CONFIGS_DIR}" \
    --output-dir "${RESULTS_DST}/phase_5/generation"

echo "Stage 2 complete."

# ========================================================================
# STAGE 3: DECODE VOLUMES
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 3: DECODING VOLUMES"
echo "=========================================="

python experiments/cli/decode_volumes.py \
    --config "${CONFIGS_DIR}/generate.yaml" \
    --configs-dir "${CONFIGS_DIR}" \
    --nfe 1 10 50

echo "Stage 3 complete."

# ========================================================================
# STAGE 4: VISUALIZATION (CPU-only, reads ~12 slices — runs in seconds)
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 4: VISUALIZATION"
echo "=========================================="

GEN_DIR="${RESULTS_DST}/phase_5/generation"

python experiments/cli/visualize_generation.py \
    --generation-dir "${GEN_DIR}" \
    --output-dir "${GEN_DIR}/figures" \
    --n-subjects 3 \
    --nfe 1 10 50

echo "Stage 4 complete."

# ========================================================================
# POST-FLIGHT
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT VERIFICATION"
echo "=========================================="

for nfe in 1 2 5 10 25 50; do
    f="${RESULTS_DST}/phase_5/generation/latents/nfe_$(printf '%03d' $nfe).h5"
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   nfe_$(printf '%03d' $nfe).h5 (latent, ${SIZE} bytes)"
    else
        echo "[MISS] nfe_$(printf '%03d' $nfe).h5 (latent)"
    fi
done

for nfe in 1 10 50; do
    f="${RESULTS_DST}/phase_5/generation/volumes/nfe_$(printf '%03d' $nfe).h5"
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        echo "[OK]   nfe_$(printf '%03d' $nfe).h5 (volume, ${SIZE} bytes)"
    else
        echo "[MISS] nfe_$(printf '%03d' $nfe).h5 (volume)"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PHASE 5 GENERATION COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
