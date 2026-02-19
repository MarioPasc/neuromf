#!/usr/bin/env bash
# =============================================================================
# x-pred vs u-pred ABLATION LAUNCHER
#
# Submits one or both ablation arms on Picasso. Each arm runs independently.
# x-pred uses 4 GPUs (batch_size=2 OOMs at 2 GPUs with exact JVP).
# u-pred uses 2 GPUs (gradient checkpointing fits batch_size=16).
#
# Usage (from login node):
#   bash experiments/ablations/xpred_vs_upred/launch.sh          # both arms
#   bash experiments/ablations/xpred_vs_upred/launch.sh xpred    # x-pred only
#   bash experiments/ablations/xpred_vs_upred/launch.sh upred    # u-pred only
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

# Which arms to run
ARM="${1:-both}"

# ========================================================================
# HELPER: submit one arm
# ========================================================================
submit_arm() {
    local ARM_NAME="$1"
    local CONFIG_FILE="$2"
    local ARM_GPUS="$3"
    local WALL_TIME="${4:-5-00:00:00}"

    local CPUS=$((16 * ARM_GPUS))
    local MEM=$((64 * ARM_GPUS))

    echo "--- Submitting ${ARM_NAME} (${ARM_GPUS} GPUs) ---"

    # Full config chain: picasso overlay + ablation overlay
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
    echo "  Effective batch: see config (both arms = 128)"
    echo "  Logs:      ${ABL_DIR}/train_${JOB_ID}.{out,err}"
    echo ""
}

# ========================================================================
# SUBMIT ARMS
# ========================================================================

if [ "${ARM}" = "xpred" ] || [ "${ARM}" = "both" ]; then
    # x-pred: 4 GPUs (batch_size=2 x 4 GPUs x 16 accum = 128)
    # No gradient checkpointing → slower per step, allow 5 days
    submit_arm "xpred_exact_jvp" "xpred_exact_jvp.yaml" 4 "5-00:00:00"
fi

if [ "${ARM}" = "upred" ] || [ "${ARM}" = "both" ]; then
    # u-pred: 2 GPUs (batch_size=16 x 2 GPUs x 4 accum = 128)
    # Gradient checkpointing → faster per step, 3 days should suffice
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
