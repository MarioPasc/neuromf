#!/usr/bin/env bash
# =============================================================================
# AUGMENTATION VISUALISATION SUITE — SLURM LAUNCHER
#
# Login-node script that submits the augmentation visualisation job on Picasso.
# Produces 18 PNGs (6 visualisations x 3 planes) in ~15 min on 1 A100.
#
# Usage (from login node):
#   bash experiments/slurm/phase_4/augmentation_viz.sh
#   VIZ_SEED=123 bash experiments/slurm/phase_4/augmentation_viz.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "AUGMENTATION VISUALISATION LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"
export VIZ_SEED="${VIZ_SEED:-42}"

OUTPUT_DIR="${RESULTS_DST}/phase_4/augmentation_viz"

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Configs:     ${CONFIGS_DIR}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo "  Seed:        ${VIZ_SEED}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ========================================================================
# SUBMIT JOB
# ========================================================================
JOB_ID=$(sbatch --parsable \
    --job-name="neuromf_aug_viz" \
    --time=1:00:00 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=32G \
    --constraint=dgx \
    --gres=gpu:1 \
    --output="${RESULTS_DST}/phase_4/aug_viz_%j.out" \
    --error="${RESULTS_DST}/phase_4/aug_viz_%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/augmentation_viz_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${RESULTS_DST}/phase_4/aug_viz_${JOB_ID}.{out,err}"
echo "Output:    ${OUTPUT_DIR}"
