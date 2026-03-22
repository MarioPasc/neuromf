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
# NEUROMF GENERATION WORKER
#
# All paths are read from the merged config via env vars set by launch.sh.
# The only required env vars are REPO_SRC, CONFIGS_DIR, RUN_DIR, CONDA_ENV_NAME.
# Optional: CKPT_PATH, GEN_N_SAMPLES, ENHANCEMENT_FLAGS.
#
# Stage 1: Generate latents at all NFE levels (1, 2, 5, 10, 25, 50)
# Stage 2: Prepare real test volumes (from original NIfTI, no VAE round-trip)
# Stage 3: Decode volumes at selected NFE levels (1, 10, 50)
# Stage 4: Visualization sanity-check figures (CPU-only)
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "Phase 5 NeuroiMF generation started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Run directory: ${RUN_DIR}"

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

# Resolve n_samples (CLI override > default 500)
N_SAMPLES="${GEN_N_SAMPLES:-500}"
echo "n_samples: ${N_SAMPLES}"

# ========================================================================
# PRE-FLIGHT CHECKS — read all paths from merged config
# ========================================================================
echo ""
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Read checkpoint + latent_stats from merged config (unless overridden via env)
if [ -z "${CKPT_PATH:-}" ] || [ -z "${STATS_PATH:-}" ]; then
    echo "Reading paths from config: ${CONFIGS_DIR}/generate.yaml"
    CONFIG_PATHS=$(python -c "
import json
from pathlib import Path
from neuromf.config import load_merged_config
cfg = load_merged_config(
    [Path('${CONFIGS_DIR}/generate.yaml')],
    configs_dir=Path('${CONFIGS_DIR}'),
    include_generate=True,
)
print(json.dumps({
    'checkpoint': str(cfg.paths.checkpoint),
    'latent_stats': str(cfg.paths.latent_stats),
}))
")
    if [ -z "${CKPT_PATH:-}" ]; then
        CKPT_PATH=$(echo "${CONFIG_PATHS}" | python -c "import sys,json; print(json.load(sys.stdin)['checkpoint'])")
    fi
    if [ -z "${STATS_PATH:-}" ]; then
        STATS_PATH=$(echo "${CONFIG_PATHS}" | python -c "import sys,json; print(json.load(sys.stdin)['latent_stats'])")
    fi
fi

# Verify checkpoint
if [ -f "${CKPT_PATH}" ]; then
    echo "[OK]   Checkpoint: ${CKPT_PATH}"
else
    echo "[MISS] Checkpoint not found: ${CKPT_PATH}"
    echo "       Update paths.checkpoint in ${CONFIGS_DIR}/generate.yaml"
    exit 1
fi

# Verify latent stats
if [ -f "${STATS_PATH}" ]; then
    echo "[OK]   Latent stats: ${STATS_PATH}"
else
    echo "[MISS] Latent stats not found: ${STATS_PATH}"
    echo "       Update paths.latent_stats in ${CONFIGS_DIR}/generate.yaml"
    exit 1
fi

# Quick import check
python -c "
from neuromf.generation import LatentGenerator, VolumeDecoder, H5Manager
from neuromf.wrappers.maisi_unet import MAISIUNetWrapper
from neuromf.wrappers.maisi_vae import MAISIVAEWrapper
from neuromf.data.latent_dataset import export_split_manifest
from neuromf.data.mri_preprocessing import build_mri_preprocessing_from_config
print('All imports OK')
"

# ========================================================================
# STAGE 1: GENERATE LATENTS
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 1: GENERATING LATENTS"
echo "=========================================="

GEN_DIR="${RUN_DIR}/generation"

python experiments/cli/generate_latents.py \
    --config "${CONFIGS_DIR}/generate.yaml" \
    --configs-dir "${CONFIGS_DIR}" \
    --checkpoint "${CKPT_PATH}" \
    --use-ema \
    --nfe 1 2 5 10 25 50 \
    --n-samples "${N_SAMPLES}" \
    --batch-size 8 \
    --output-dir "${GEN_DIR}" \
    ${ENHANCEMENT_FLAGS:-}

echo "Stage 1 complete."

# ========================================================================
# STAGE 2: PREPARE REAL TEST VOLUMES (from original NIfTI, no VAE)
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 2: PREPARING REAL TEST VOLUMES"
echo "=========================================="
echo "(Reading original NIfTI files — no VAE round-trip)"

python experiments/cli/prepare_real_test.py \
    --config "${CONFIGS_DIR}/generate.yaml" \
    --configs-dir "${CONFIGS_DIR}" \
    --output-dir "${GEN_DIR}"

echo "Stage 2 complete."

# ========================================================================
# STAGE 3: DECODE VOLUMES
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 3: DECODING VOLUMES"
echo "=========================================="

# Decode from comparison dirs if they exist, otherwise standard latents/
if [ -d "${GEN_DIR}/latents_baseline" ] && [ -d "${GEN_DIR}/latents_enhanced" ]; then
    echo "Comparison mode detected — decoding both baseline and enhanced"
    python experiments/cli/decode_volumes.py \
        --config "${CONFIGS_DIR}/generate.yaml" \
        --configs-dir "${CONFIGS_DIR}" \
        --nfe 1 10 50 \
        --latent-dir "${GEN_DIR}/latents_baseline" \
        --output-dir "${GEN_DIR}/volumes_baseline"

    python experiments/cli/decode_volumes.py \
        --config "${CONFIGS_DIR}/generate.yaml" \
        --configs-dir "${CONFIGS_DIR}" \
        --nfe 1 10 50 \
        --latent-dir "${GEN_DIR}/latents_enhanced" \
        --output-dir "${GEN_DIR}/volumes_enhanced"
else
    python experiments/cli/decode_volumes.py \
        --config "${CONFIGS_DIR}/generate.yaml" \
        --configs-dir "${CONFIGS_DIR}" \
        --nfe 1 10 50 \
        --latent-dir "${GEN_DIR}/latents" \
        --output-dir "${GEN_DIR}/volumes"
fi

echo "Stage 3 complete."

# ========================================================================
# STAGE 4: VISUALIZATION (CPU-only, reads ~12 slices — runs in seconds)
# ========================================================================
echo ""
echo "=========================================="
echo "STAGE 4: VISUALIZATION"
echo "=========================================="

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

# Check latent archives (comparison or standard)
if [ -d "${GEN_DIR}/latents_baseline" ] && [ -d "${GEN_DIR}/latents_enhanced" ]; then
    for mode in baseline enhanced; do
        echo "--- ${mode} ---"
        for nfe in 1 2 5 10 25 50; do
            f="${GEN_DIR}/latents_${mode}/nfe_$(printf '%03d' $nfe).h5"
            if [ -f "$f" ]; then
                SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
                echo "[OK]   nfe_$(printf '%03d' $nfe).h5 (latent ${mode}, ${SIZE} bytes)"
            else
                echo "[MISS] nfe_$(printf '%03d' $nfe).h5 (latent ${mode})"
            fi
        done
    done
    for mode in baseline enhanced; do
        for nfe in 1 10 50; do
            f="${GEN_DIR}/volumes_${mode}/nfe_$(printf '%03d' $nfe).h5"
            if [ -f "$f" ]; then
                SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
                echo "[OK]   nfe_$(printf '%03d' $nfe).h5 (volume ${mode}, ${SIZE} bytes)"
            else
                echo "[MISS] nfe_$(printf '%03d' $nfe).h5 (volume ${mode})"
            fi
        done
    done
else
    for nfe in 1 2 5 10 25 50; do
        f="${GEN_DIR}/latents/nfe_$(printf '%03d' $nfe).h5"
        if [ -f "$f" ]; then
            SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
            echo "[OK]   nfe_$(printf '%03d' $nfe).h5 (latent, ${SIZE} bytes)"
        else
            echo "[MISS] nfe_$(printf '%03d' $nfe).h5 (latent)"
        fi
    done
    for nfe in 1 10 50; do
        f="${GEN_DIR}/volumes/nfe_$(printf '%03d' $nfe).h5"
        if [ -f "$f" ]; then
            SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
            echo "[OK]   nfe_$(printf '%03d' $nfe).h5 (volume, ${SIZE} bytes)"
        else
            echo "[MISS] nfe_$(printf '%03d' $nfe).h5 (volume)"
        fi
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "NEUROMF GENERATION COMPLETED"
echo "=========================================="
echo "Run dir:    ${RUN_DIR}"
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
