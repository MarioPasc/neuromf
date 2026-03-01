#!/usr/bin/env bash
# =============================================================================
# SMOKE TEST LAUNCHER — Pre-training verification on Picasso
#
# Runs the exact same training pipeline (config chain, model, loss, JVP,
# FID callback, checkpointing) on minimal resources: 1 GPU, 5 batches/epoch,
# 3 epochs. Catches config errors, import failures, OOM, and pipeline bugs
# before committing to a multi-day training run.
#
# Reuses worker.sh with TRAIN_CONFIG pointing to the smoke overlay.
#
# Usage (from login node):
#   bash slurm/train/smoke_launch.sh
#   bash slurm/train/smoke_launch.sh --depends-on 12345
#
# Expected runtime: ~5-10 min compute on 1 A100 (1h wall time for buffer).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
            echo "Usage: bash slurm/train/smoke_launch.sh [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "SMOKE TEST LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="smoke_test"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

# Single GPU for smoke test
export N_GPUS=1

# Smoke overlay: merged on top of picasso/train_meanflow.yaml
export TRAIN_CONFIG="${CONFIGS_DIR}/train_meanflow.yaml ${CONFIGS_DIR}/smoke_test.yaml"

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Configs:     ${CONFIGS_DIR}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "  Conda env:   ${CONDA_ENV_NAME}" >&2
echo "  GPUs:        ${N_GPUS}" >&2
echo "  Config chain: ${TRAIN_CONFIG}" >&2
echo "" >&2

# Create output directories
mkdir -p "${RESULTS_DST}/smoke_test/checkpoints"
mkdir -p "${RESULTS_DST}/smoke_test/logs"
mkdir -p "${RESULTS_DST}/smoke_test/samples"
mkdir -p "${RESULTS_DST}/smoke_test/diagnostics"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_smoke"
    --time=1:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=64G
    --constraint=dgx
    --gres="gpu:1"
    --output="${RESULTS_DST}/smoke_test/smoke_%j.out"
    --error="${RESULTS_DST}/smoke_test/smoke_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency:  afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "SMOKE TEST SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "GPUs:      ${N_GPUS}" >&2
echo "Wall time: 1 hour" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Output:    ${RESULTS_DST}/smoke_test/smoke_${JOB_ID}.{out,err}" >&2
echo "Checkpts:  ${RESULTS_DST}/smoke_test/checkpoints/" >&2
echo "" >&2
echo "After completion, verify:" >&2
echo "  ls ${RESULTS_DST}/smoke_test/checkpoints/*.ckpt" >&2
echo "  cat ${RESULTS_DST}/smoke_test/diagnostics/training_summary.json | python -m json.tool" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
