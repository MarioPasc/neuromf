#!/usr/bin/env bash
# =============================================================================
# x-pred vs u-pred ABLATION LAUNCHER
#
# Submits the x-pred training run on Picasso. The u-pred baseline already ran
# and diverged at epoch 150 — it serves as the comparison reference.
#
# The x-pred arm inherits the best known config from base + Picasso overlay
# (x-pred, exact JVP, t_h conditioning, v-head, 1500 epochs, augmentation).
# The ablation overlay only redirects output paths.
#
# Usage (from login node):
#   bash experiments/ablations/xpred_vs_upred/launch.sh               # x-pred (default)
#   bash experiments/ablations/xpred_vs_upred/launch.sh xpred         # explicit x-pred
#   bash experiments/ablations/xpred_vs_upred/launch.sh --xpred-only  # alias
#   bash experiments/ablations/xpred_vs_upred/launch.sh upred         # u-pred (re-run)
#   bash experiments/ablations/xpred_vs_upred/launch.sh both          # both arms
#   bash experiments/ablations/xpred_vs_upred/launch.sh --resume /path/to/ckpt  # resume x-pred
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/../../slurm/phase_4/train_worker.sh"

# Validate worker script exists
if [ ! -f "${WORKER_SCRIPT}" ]; then
    echo "ERROR: Worker script not found at ${WORKER_SCRIPT}"
    exit 1
fi

echo "=========================================="
echo "x-pred vs u-pred ABLATION LAUNCHER"
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
ARM="xpred"  # default: x-pred only
RESUME_CKPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        xpred|--xpred-only) ARM="xpred"; shift ;;
        upred|--upred-only) ARM="upred"; shift ;;
        both|--both)        ARM="both"; shift ;;
        --resume)           RESUME_CKPT="$2"; shift 2 ;;
        *)                  echo "Unknown argument: $1"; exit 1 ;;
    esac
done
export RESUME_CKPT

# ========================================================================
# HELPER: submit one arm
# ========================================================================
submit_arm() {
    local ARM_NAME="$1"
    local CONFIG_FILE="$2"
    local ARM_GPUS="$3"
    local WALL_TIME="${4:-7-00:00:00}"

    local CPUS=$((16 * ARM_GPUS))
    local MEM=$((64 * ARM_GPUS))
    if [ "$CPUS" -gt 128 ]; then CPUS=128; fi
    if [ "$MEM" -gt 480 ]; then MEM=480; fi

    echo "--- Submitting ${ARM_NAME} (${ARM_GPUS} GPUs, ${WALL_TIME}) ---"

    # Config chain: Picasso overlay + ablation overlay
    export TRAIN_CONFIG="${CONFIGS_DIR}/train_meanflow.yaml ${SCRIPT_DIR}/configs/${CONFIG_FILE}"
    export N_GPUS="${ARM_GPUS}"

    # Create output directories
    local ABL_DIR="${RESULTS_DST}/ablations/${ARM_NAME}"
    mkdir -p "${ABL_DIR}/checkpoints"
    mkdir -p "${ABL_DIR}/logs"
    mkdir -p "${ABL_DIR}/samples"
    mkdir -p "${ABL_DIR}/diagnostics"

    JOB_ID=$(sbatch --parsable \
        --job-name="neuromf_${ARM_NAME}" \
        --time="${WALL_TIME}" \
        --ntasks=1 \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}G" \
        --constraint=dgx \
        --gres="gpu:${ARM_GPUS}" \
        --output="${ABL_DIR}/train_%j.out" \
        --error="${ABL_DIR}/train_%j.err" \
        --export=ALL \
        "${WORKER_SCRIPT}")

    echo "  Job ID:    ${JOB_ID}"
    echo "  Config:    ${TRAIN_CONFIG}"
    echo "  GPUs:      ${ARM_GPUS}"
    echo "  Wall time: ${WALL_TIME}"
    echo "  Logs:      ${ABL_DIR}/train_${JOB_ID}.{out,err}"
    echo ""
}

# ========================================================================
# SUBMIT ARMS
# ========================================================================

if [ "${ARM}" = "xpred" ] || [ "${ARM}" = "both" ]; then
    # x-pred: 6 GPUs (batch=2 x 6 GPUs x 11 accum = 132), 7-day wall time
    # Inherits all settings from base (x-pred, exact JVP, t_h, 1500 epochs)
    submit_arm "xpred_exact_jvp" "xpred_exact_jvp.yaml" 6 "7-00:00:00"
fi

if [ "${ARM}" = "upred" ] || [ "${ARM}" = "both" ]; then
    # u-pred: 2 GPUs (batch=16 x 2 GPUs x 4 accum = 128), 3-day wall time
    # NOTE: u-pred already collapsed at epoch 150 in previous run. Re-running
    # is only needed if configs changed. The existing results serve as baseline.
    submit_arm "upred_fd_jvp" "upred_fd_jvp.yaml" 2 "3-00:00:00"
fi

echo "=========================================="
echo "ABLATION JOBS SUBMITTED"
echo "=========================================="
echo "Monitor: squeue -u \$USER"
echo "Compare results in: ${RESULTS_DST}/ablations/"
echo ""
echo "After completion, compare:"
echo "  tensorboard --logdir_spec xpred:${RESULTS_DST}/ablations/xpred_exact_jvp/logs,upred:${RESULTS_DST}/ablations/upred_fd_jvp/logs"
echo ""
echo "Generate report:"
echo "  python experiments/ablations/xpred_vs_upred/compare_arms.py --results-dir ${RESULTS_DST}/ablations"
