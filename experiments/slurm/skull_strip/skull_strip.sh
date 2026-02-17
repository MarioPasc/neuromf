#!/usr/bin/env bash
# =============================================================================
# SKULL-STRIP DEFACED DATASETS — SLURM LAUNCHER
#
# Login-node script that:
#   1. Pre-downloads HD-BET weights (compute nodes have no internet)
#   2. Submits a 3-GPU array job for parallel skull-stripping
#   3. Submits a dependency job to generate the summary visualization
#
# Phase B uses 3 GPUs in parallel (round-robin work distribution).
# Expected time: ~45 min on 3×A100 (fast mode, ~2,800 volumes at ~4s each).
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf
#
#   # Batch mode (default): 3 parallel GPUs
#   bash experiments/slurm/skull_strip/skull_strip.sh
#
#   # Phase A validation only (single GPU, 9 subjects):
#   bash experiments/slurm/skull_strip/skull_strip.sh --phase A
#
#   # Custom number of workers:
#   bash experiments/slurm/skull_strip/skull_strip.sh --num-workers 6
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "SKULL-STRIP DEFACED DATASETS — LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="neuromf"
export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export FOMO60K_ROOT="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/FOMO60K"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"
export SKULL_STRIP_PHASE="B"
NUM_WORKERS=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            SKULL_STRIP_PHASE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash experiments/slurm/skull_strip/skull_strip.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --phase A|B       Validation (A, single GPU) or batch (B, default)"
            echo "  --num-workers N   Number of parallel GPU workers for Phase B (default: 3)"
            echo "  --help, -h        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done
export SKULL_STRIP_PHASE
export NUM_WORKERS

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  FOMO-60K:    ${FOMO60K_ROOT}"
echo "  Results:     ${RESULTS_DST}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo "  Phase:       ${SKULL_STRIP_PHASE}"
echo "  Workers:     ${NUM_WORKERS}"
echo ""

# ========================================================================
# PRE-DOWNLOAD: HD-BET weights (compute nodes have no internet)
# ========================================================================
echo "Checking HD-BET model weights..."

# Activate conda on login node to access the package
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

python -c "
from brainles_hd_bet.utils import get_params_fname, maybe_download_parameters
missing = [i for i in range(5) if not get_params_fname(i).exists()]
if missing:
    print(f'  Downloading HD-BET weights for folds: {missing}')
    for fold in missing:
        maybe_download_parameters(fold)
    print('  HD-BET weights: download complete')
else:
    print('  HD-BET weights: all 5 folds already present')
" || {
    echo "WARNING: Could not verify HD-BET weights. Skull stripping may fail on compute nodes."
    echo "         Manual download:"
    echo "         python -c \"from brainles_hd_bet.utils import maybe_download_parameters; [maybe_download_parameters(i) for i in range(5)]\""
}
echo ""

# ========================================================================
# CREATE OUTPUT DIRECTORIES
# ========================================================================
mkdir -p "${RESULTS_DST}/skull_strip"

# ========================================================================
# SUBMIT JOBS
# ========================================================================

if [ "${SKULL_STRIP_PHASE}" == "A" ]; then
    # ---- Phase A: single GPU, validation only ----
    JOB_ID=$(sbatch --parsable \
        --job-name="neuromf_ss_valA" \
        --time=0-01:00:00 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=32G \
        --constraint=dgx \
        --gres=gpu:1 \
        --output="${RESULTS_DST}/skull_strip/ss_phaseA_%j.out" \
        --error="${RESULTS_DST}/skull_strip/ss_phaseA_%j.err" \
        --export=ALL \
        "${SCRIPT_DIR}/skull_strip_worker.sh")

    echo "=========================================="
    echo "JOB SUBMITTED (Phase A — single GPU)"
    echo "=========================================="
    echo "Job ID:    ${JOB_ID}"
    echo "Monitor:   squeue -j ${JOB_ID}"
    echo "Logs:      ${RESULTS_DST}/skull_strip/ss_phaseA_${JOB_ID}.{out,err}"
    echo ""
    echo "After completion, inspect visualizations at:"
    echo "  ${FOMO60K_ROOT}/_skull_strip_validation/"
    echo ""
    echo "Then run Phase B:"
    echo "  bash experiments/slurm/skull_strip/skull_strip.sh --phase B"

else
    # ---- Phase B: array job with N workers, each gets 1 GPU ----
    ARRAY_MAX=$((NUM_WORKERS - 1))

    ARRAY_JOB_ID=$(sbatch --parsable \
        --job-name="neuromf_ss_batch" \
        --time=0-03:00:00 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=32G \
        --constraint=dgx \
        --gres=gpu:1 \
        --array="0-${ARRAY_MAX}" \
        --output="${RESULTS_DST}/skull_strip/ss_phaseB_%A_%a.out" \
        --error="${RESULTS_DST}/skull_strip/ss_phaseB_%A_%a.err" \
        --export=ALL \
        "${SCRIPT_DIR}/skull_strip_worker.sh")

    # ---- Dependency job: summary visualization after all workers finish ----
    # Needs --gres=gpu:1 because --constraint=dgx routes to GPU partition
    VIZ_JOB_ID=$(sbatch --parsable \
        --job-name="neuromf_ss_viz" \
        --time=0-00:15:00 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=16G \
        --constraint=dgx \
        --gres=gpu:1 \
        --dependency="afterok:${ARRAY_JOB_ID}" \
        --output="${RESULTS_DST}/skull_strip/ss_viz_%j.out" \
        --error="${RESULTS_DST}/skull_strip/ss_viz_%j.err" \
        --export=ALL \
        "${SCRIPT_DIR}/skull_strip_visualize.sh")

    echo "=========================================="
    echo "JOBS SUBMITTED (Phase B — ${NUM_WORKERS} parallel GPUs)"
    echo "=========================================="
    echo "Array job:  ${ARRAY_JOB_ID} (workers 0-${ARRAY_MAX})"
    echo "Viz job:    ${VIZ_JOB_ID} (runs after array completes)"
    echo ""
    echo "Monitor:"
    echo "  squeue -j ${ARRAY_JOB_ID}                   # Worker status"
    echo "  squeue -j ${VIZ_JOB_ID}                     # Viz job status"
    echo ""
    echo "Logs:"
    echo "  ${RESULTS_DST}/skull_strip/ss_phaseB_${ARRAY_JOB_ID}_<worker>.{out,err}"
    echo "  ${RESULTS_DST}/skull_strip/ss_viz_${VIZ_JOB_ID}.{out,err}"
    echo ""
    echo "After completion:"
    echo "  Log:           ${FOMO60K_ROOT}/_skull_strip_log.json"
    echo "  Summary plot:  ${FOMO60K_ROOT}/_skull_strip_summary.png"
    echo ""
    echo "Cancel all:"
    echo "  scancel ${ARRAY_JOB_ID} ${VIZ_JOB_ID}"
fi
