#!/usr/bin/env bash
# =============================================================================
# PHASE 5: GENERATION PIPELINE — SLURM LAUNCHER
#
# Login-node script that submits the Phase 5 generation + decode job.
# Generates latents at all NFE levels, then decodes selected NFE levels.
#
# Usage (from login node):
#   bash experiments/slurm/phase_5/generate.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "PHASE 5: GENERATION PIPELINE LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_5_generation"
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Configs:     ${CONFIGS_DIR}"
echo "  Results:     ${RESULTS_DST}"
echo ""

# Create output directories
mkdir -p "${RESULTS_DST}/phase_5/generation/latents"
mkdir -p "${RESULTS_DST}/phase_5/generation/volumes"
mkdir -p "${RESULTS_DST}/phase_5/features"
mkdir -p "${RESULTS_DST}/phase_5/metrics"

# ========================================================================
# SUBMIT JOB
# ========================================================================
JOB_ID=$(sbatch --parsable \
    --job-name="neuromf_p5_gen" \
    --time=1-00:00:00 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${RESULTS_DST}/phase_5/generate_%j.out" \
    --error="${RESULTS_DST}/phase_5/generate_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/generate_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${RESULTS_DST}/phase_5/generate_${JOB_ID}.{out,err}"
echo "Latents:   ${RESULTS_DST}/phase_5/generation/latents/"
echo "Volumes:   ${RESULTS_DST}/phase_5/generation/volumes/"
