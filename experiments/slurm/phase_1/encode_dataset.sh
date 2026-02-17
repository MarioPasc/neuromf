#!/usr/bin/env bash
# =============================================================================
# PHASE 1: LATENT PRE-COMPUTATION — SLURM LAUNCHER
#
# Login-node script that submits the Phase 1 encoding pipeline on Picasso.
# Uses a single A100 GPU to encode all ~1,100 FOMO-60K volumes through
# the frozen MAISI VAE.
#
# Expected time: ~1-2h on A100 (vs ~6h on RTX 4060)
#
# Usage (from login node):
#   bash experiments/slurm/phase_1/encode_dataset.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "PHASE 1: LATENT PRE-COMPUTATION LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_1_latent_encoding"
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Configs:     ${CONFIGS_DIR}"
echo "  Results:     ${RESULTS_DST}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo ""

# Create output directories
mkdir -p "${RESULTS_DST}/latents"
mkdir -p "${RESULTS_DST}/phase_1/figures"

# ========================================================================
# SUBMIT JOB
# ========================================================================
JOB_ID=$(sbatch --parsable \
    --job-name="neuromf_p1_encode" \
    --time=0-10:00:00 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=64G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${RESULTS_DST}/phase_1/encode_%j.out" \
    --error="${RESULTS_DST}/phase_1/encode_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/encode_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${RESULTS_DST}/phase_1/encode_${JOB_ID}.{out,err}"
echo "Results:   ${RESULTS_DST}/latents/"
echo ""
echo "After completion, run Phase 1 tests:"
echo "  python -m pytest tests/test_latent_dataset.py -v --tb=short"
