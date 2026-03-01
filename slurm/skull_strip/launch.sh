#!/usr/bin/env bash
# =============================================================================
# SKULL-STRIP DEFACED DATASETS — SLURM LAUNCHER (atomic)
#
# Login-node script that:
#   1. Pre-downloads HD-BET weights (compute nodes have no internet)
#   2. Submits a 3-GPU array job for parallel skull-stripping
#   3. Submits a dependency job to generate the summary visualization
#
# Phase B uses 3 GPUs in parallel (round-robin work distribution).
# Expected time: ~45 min on 3xA100 (fast mode, ~2,800 volumes at ~4s each).
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf
#
#   # Batch mode (default): 3 parallel GPUs
#   bash slurm/skull_strip/launch.sh
#
#   # Phase A validation only (single GPU, 9 subjects):
#   bash slurm/skull_strip/launch.sh --phase A
#
#   # Custom number of workers:
#   bash slurm/skull_strip/launch.sh --num-workers 6
#
#   # Chain after another job:
#   bash slurm/skull_strip/launch.sh --depends-on 12345
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==========================================" >&2
echo "SKULL-STRIP DEFACED DATASETS — LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export FOMO60K_ROOT="${FOMO60K_ROOT:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/FOMO60K}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"
export SKULL_STRIP_PHASE="B"
NUM_WORKERS=3
DEPENDS_ON=""

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
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash slurm/skull_strip/launch.sh [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --phase A|B       Validation (A, single GPU) or batch (B, default)" >&2
            echo "  --num-workers N   Number of parallel GPU workers for Phase B (default: 3)" >&2
            echo "  --depends-on ID   Wait for job ID to finish" >&2
            echo "  --help, -h        Show this help" >&2
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done
export SKULL_STRIP_PHASE
export NUM_WORKERS

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  FOMO-60K:    ${FOMO60K_ROOT}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "  Conda env:   ${CONDA_ENV_NAME}" >&2
echo "  Phase:       ${SKULL_STRIP_PHASE}" >&2
echo "  Workers:     ${NUM_WORKERS}" >&2
echo "" >&2

# ========================================================================
# PRE-DOWNLOAD: HD-BET weights (compute nodes have no internet)
# ========================================================================
echo "Checking HD-BET model weights..." >&2

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
    echo "WARNING: Could not verify HD-BET weights. Skull stripping may fail on compute nodes." >&2
    echo "         Manual download:" >&2
    echo "         python -c \"from brainles_hd_bet.utils import maybe_download_parameters; [maybe_download_parameters(i) for i in range(5)]\"" >&2
}
echo "" >&2

# ========================================================================
# CREATE OUTPUT DIRECTORIES
# ========================================================================
mkdir -p "${RESULTS_DST}/skull_strip"

# ========================================================================
# SUBMIT JOBS
# ========================================================================

if [ "${SKULL_STRIP_PHASE}" == "A" ]; then
    # ---- Phase A: single GPU, validation only ----
    SBATCH_ARGS=(
        --parsable
        --job-name="neuromf_ss_valA"
        --time=0-01:00:00
        --ntasks=1
        --cpus-per-task=8
        --mem=32G
        --constraint=dgx
        --gres=gpu:1
        --output="${RESULTS_DST}/skull_strip/ss_phaseA_%j.out"
        --error="${RESULTS_DST}/skull_strip/ss_phaseA_%j.err"
        --export=ALL
    )

    if [ -n "${DEPENDS_ON}" ]; then
        SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    fi

    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

    echo "==========================================" >&2
    echo "JOB SUBMITTED (Phase A — single GPU)" >&2
    echo "==========================================" >&2
    echo "Job ID:    ${JOB_ID}" >&2
    echo "Monitor:   squeue -j ${JOB_ID}" >&2
    echo "Logs:      ${RESULTS_DST}/skull_strip/ss_phaseA_${JOB_ID}.{out,err}" >&2
    echo "" >&2
    echo "After completion, inspect visualizations at:" >&2
    echo "  ${FOMO60K_ROOT}/_skull_strip_validation/" >&2
    echo "" >&2
    echo "Then run Phase B:" >&2
    echo "  bash slurm/skull_strip/launch.sh --phase B" >&2

    # Job ID to stdout for orchestration capture
    echo "${JOB_ID}"

else
    # ---- Phase B: array job with N workers, each gets 1 GPU ----
    ARRAY_MAX=$((NUM_WORKERS - 1))

    SBATCH_ARGS=(
        --parsable
        --job-name="neuromf_ss_batch"
        --time=0-03:00:00
        --ntasks=1
        --cpus-per-task=8
        --mem=32G
        --constraint=dgx
        --gres=gpu:1
        --array="0-${ARRAY_MAX}"
        --output="${RESULTS_DST}/skull_strip/ss_phaseB_%A_%a.out"
        --error="${RESULTS_DST}/skull_strip/ss_phaseB_%A_%a.err"
        --export=ALL
    )

    if [ -n "${DEPENDS_ON}" ]; then
        SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    fi

    ARRAY_JOB_RAW=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

    # --parsable may return "JOBID;CLUSTER" — extract just the numeric ID
    ARRAY_JOB_ID="${ARRAY_JOB_RAW%%;*}"
    echo "Array job submitted: ${ARRAY_JOB_ID} (raw: ${ARRAY_JOB_RAW})" >&2

    # ---- Dependency job: summary visualization after all workers finish ----
    # afterany tolerates individual task failures (afterok would cancel viz if any worker fails)
    # Needs --gres=gpu:1 because --constraint=dgx routes to GPU partition
    VIZ_JOB_ID=$(sbatch --parsable \
        --job-name="neuromf_ss_viz" \
        --time=0-00:15:00 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=16G \
        --constraint=dgx \
        --gres=gpu:1 \
        --dependency="afterany:${ARRAY_JOB_ID}" \
        --output="${RESULTS_DST}/skull_strip/ss_viz_%j.out" \
        --error="${RESULTS_DST}/skull_strip/ss_viz_%j.err" \
        --export=ALL \
        "${SCRIPT_DIR}/visualize.sh")

    echo "==========================================" >&2
    echo "JOBS SUBMITTED (Phase B — ${NUM_WORKERS} parallel GPUs)" >&2
    echo "==========================================" >&2
    echo "Array job:  ${ARRAY_JOB_ID} (workers 0-${ARRAY_MAX})" >&2
    echo "Viz job:    ${VIZ_JOB_ID} (runs after array completes)" >&2
    echo "" >&2
    echo "Monitor:" >&2
    echo "  squeue -j ${ARRAY_JOB_ID}                   # Worker status" >&2
    echo "  squeue -j ${VIZ_JOB_ID}                     # Viz job status" >&2
    echo "" >&2
    echo "Logs:" >&2
    echo "  ${RESULTS_DST}/skull_strip/ss_phaseB_${ARRAY_JOB_ID}_<worker>.{out,err}" >&2
    echo "  ${RESULTS_DST}/skull_strip/ss_viz_${VIZ_JOB_ID}.{out,err}" >&2
    echo "" >&2
    echo "After completion:" >&2
    echo "  Log:           ${FOMO60K_ROOT}/_skull_strip_log.json" >&2
    echo "  Summary plot:  ${FOMO60K_ROOT}/_skull_strip_summary.png" >&2
    echo "" >&2
    echo "Cancel all:" >&2
    echo "  scancel ${ARRAY_JOB_ID} ${VIZ_JOB_ID}" >&2

    # Job IDs to stdout for orchestration capture
    echo "${ARRAY_JOB_ID}"
fi
