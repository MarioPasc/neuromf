#!/usr/bin/env bash
# =============================================================================
# VELOCITY SMOOTHNESS ABLATION — SMOKE TEST
#
# Exercises the exact same code path as the full ablation (smoothness loss,
# FD Hutchinson estimator, v-head Jacobian penalty) on minimal resources:
# 1 GPU, 5 batches/epoch, 3 epochs. Verifies the extra v_fn forward pass
# doesn't OOM and the smoothness loss is computed correctly.
#
# Config chain: picasso/train_meanflow.yaml + v_smoothness/config.yaml
#             + picasso/smoke_test.yaml
#
# Usage (from login node):
#   bash experiments/ablations/v_smoothness/smoke_launch.sh
#   bash experiments/ablations/v_smoothness/smoke_launch.sh --depends-on 12345
#
# Expected runtime: ~5-10 min on 1 A100.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/../../../slurm/train/worker.sh"

if [ ! -f "${WORKER_SCRIPT}" ]; then
    echo "ERROR: Worker script not found at ${WORKER_SCRIPT}" >&2
    exit 1
fi

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
export RESUME_CKPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash experiments/ablations/v_smoothness/smoke_launch.sh [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "V-SMOOTHNESS ABLATION — SMOKE TEST" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="v_smoothness_smoke"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

# Single GPU for smoke test
export N_GPUS=1

# Config chain: Picasso overlay + smoothness ablation + smoke test limits
export TRAIN_CONFIG="${CONFIGS_DIR}/train_meanflow.yaml ${SCRIPT_DIR}/config.yaml ${CONFIGS_DIR}/smoke_test.yaml"

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Config chain: ${TRAIN_CONFIG}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "  Conda env:   ${CONDA_ENV_NAME}" >&2
echo "  GPUs:        ${N_GPUS}" >&2
echo "" >&2

# SLURM log output directory (run dir is created by train.py)
ABL_DIR="${RESULTS_DST}/ablations/v_smoothness"
mkdir -p "${ABL_DIR}"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_vsmooth_smoke"
    --time=1:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=64G
    --constraint=dgx
    --gres="gpu:1"
    --output="${ABL_DIR}/smoke_%j.out"
    --error="${ABL_DIR}/smoke_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency:  afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${WORKER_SCRIPT}")

echo "==========================================" >&2
echo "SMOKE TEST SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "GPUs:      ${N_GPUS}" >&2
echo "Wall time: 1 hour" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Output:    ${ABL_DIR}/smoke_${JOB_ID}.{out,err}" >&2
echo "Run dir:   ${RESULTS_DST}/runs/run_<timestamp>/" >&2
echo "" >&2
echo "After completion, verify:" >&2
echo "  ls ${RESULTS_DST}/runs/       # find your run directory" >&2
echo "  # Check smoothness loss is being computed:" >&2
echo "  cat ${RESULTS_DST}/runs/run_*/diagnostics/aggregate_results/training_summary.json | python -m json.tool | grep smoothness" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
