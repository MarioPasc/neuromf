#!/usr/bin/env bash
# =============================================================================
# MOTFM DATA PREPARATION — SLURM LAUNCHER
#
# Login-node script that submits the FOMO-60K → MOTFM .pkl conversion job.
# CPU-only, high memory (loading all 1,379 volumes at 192^3).
#
# Usage (from login node):
#   bash experiments/slurm/motfm/prepare_data.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "MOTFM DATA PREPARATION LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="motfm_prepare_data"
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

export MOTFM_OUTPUT="${RESULTS_DST}/motfm/data/fomo60k_3d.pkl"
export SPLIT_MANIFEST="${RESULTS_DST}/latents/split_manifest.json"

echo "Configuration:"
echo "  Repo:            ${REPO_SRC}"
echo "  Configs:         ${CONFIGS_DIR}"
echo "  Results:         ${RESULTS_DST}"
echo "  Output pickle:   ${MOTFM_OUTPUT}"
echo "  Split manifest:  ${SPLIT_MANIFEST}"
echo ""

# Create output directories
mkdir -p "${RESULTS_DST}/motfm/data"

# ========================================================================
# SUBMIT JOB
# ========================================================================
JOB_ID=$(sbatch --parsable \
    --job-name="motfm_data_prep" \
    --time=0-06:00:00 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=500G \
    --partition=cpu \
    --output="${RESULTS_DST}/motfm/prepare_data_%j.out" \
    --error="${RESULTS_DST}/motfm/prepare_data_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/prepare_data_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${RESULTS_DST}/motfm/prepare_data_${JOB_ID}.{out,err}"
echo "Output:    ${MOTFM_OUTPUT}"
