#!/usr/bin/env bash
# =============================================================================
# POST-TRAINING FIGURE GENERATION
#
# Submits a lightweight SLURM job (1 GPU, 1h) to generate figures from
# sample archives. Runs VAE decode + all evolution plots.
#
# Usage (from login node):
#   bash experiments/ablations/xpred_vs_upred/generate_figures.sh               # xpred (default)
#   bash experiments/ablations/xpred_vs_upred/generate_figures.sh xpred         # xpred arm
#   bash experiments/ablations/xpred_vs_upred/generate_figures.sh rflow         # rflow arm
#   bash experiments/ablations/xpred_vs_upred/generate_figures.sh upred         # upred arm
#   bash experiments/ablations/xpred_vs_upred/generate_figures.sh all           # all arms
#   bash experiments/ablations/xpred_vs_upred/generate_figures.sh --skip-decode # latent plots only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/generate_figures_worker.sh"

if [ ! -f "${WORKER_SCRIPT}" ]; then
    echo "ERROR: Worker script not found at ${WORKER_SCRIPT}"
    exit 1
fi

echo "=========================================="
echo "FIGURE GENERATION LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# SHARED CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="neuromf"
export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

# Parse arguments
ARM="xpred"
SKIP_DECODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        xpred)         ARM="xpred"; shift ;;
        rflow)         ARM="rflow"; shift ;;
        upred)         ARM="upred"; shift ;;
        all)           ARM="all"; shift ;;
        --skip-decode) SKIP_DECODE="--skip-decode"; shift ;;
        *)             echo "Unknown argument: $1"; exit 1 ;;
    esac
done
export SKIP_DECODE

# ========================================================================
# HELPER: submit figure generation for one arm
# ========================================================================
submit_figures() {
    local ARM_NAME="$1"
    local ABL_DIR="${RESULTS_DST}/ablations/${ARM_NAME}"

    # Verify samples exist
    if [ ! -f "${ABL_DIR}/samples/sample_archive.pt" ]; then
        echo "SKIP: No sample_archive.pt for ${ARM_NAME} at ${ABL_DIR}/samples/"
        return
    fi

    export SAMPLES_DIR="${ABL_DIR}/samples"
    export FIGURES_DIR="${ABL_DIR}/figures"
    mkdir -p "${FIGURES_DIR}"

    local RESOURCES=""
    local TIME="01:00:00"
    if [ -z "${SKIP_DECODE}" ]; then
        # VAE decode needs 1 GPU (~8GB VRAM for 192^3 decode)
        RESOURCES="--gres=gpu:1 --constraint=dgx"
        TIME="01:00:00"
    else
        # Latent-only plots: CPU is enough
        RESOURCES=""
        TIME="00:15:00"
    fi

    echo "--- Submitting figures for ${ARM_NAME} ---"

    JOB_ID=$(sbatch --parsable \
        --job-name="figs_${ARM_NAME}" \
        --time="${TIME}" \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=32G \
        ${RESOURCES} \
        --output="${ABL_DIR}/figures_%j.out" \
        --error="${ABL_DIR}/figures_%j.err" \
        --export=ALL \
        "${WORKER_SCRIPT}")

    echo "  Job ID:  ${JOB_ID}"
    echo "  Samples: ${SAMPLES_DIR}"
    echo "  Output:  ${FIGURES_DIR}"
    echo "  Decode:  $([ -z "${SKIP_DECODE}" ] && echo 'yes (1 GPU)' || echo 'no (CPU only)')"
    echo ""
}

# ========================================================================
# SUBMIT
# ========================================================================

if [ "${ARM}" = "xpred" ] || [ "${ARM}" = "all" ]; then
    submit_figures "xpred_exact_jvp"
fi

if [ "${ARM}" = "rflow" ] || [ "${ARM}" = "all" ]; then
    submit_figures "xpred_exact_jvp_rflow"
fi

if [ "${ARM}" = "upred" ] || [ "${ARM}" = "all" ]; then
    submit_figures "upred_fd_jvp"
fi

echo "=========================================="
echo "FIGURE JOBS SUBMITTED"
echo "=========================================="
echo "Monitor: squeue -u \$USER"
