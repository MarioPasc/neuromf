#!/usr/bin/env bash
# =============================================================================
# NeuroMF v2 TRAINING — SLURM LAUNCHER (two-stage, fully automated)
#
# Submits a chained pipeline:
#   1. Stage 1 (GPU): FM pretraining with rflow init, 500 epochs
#   2. Bridge  (CPU): Finds best FID checkpoint from Stage 1
#   3. Stage 2 (GPU): α-Flow MF fine-tuning from Stage 1 checkpoint
#
# "both" mode chains all three automatically via SLURM --dependency.
# No manual intervention needed between stages.
#
# Usage (from Picasso login node):
#   bash slurm/train_v2/launch.sh                     # both stages (default)
#   bash slurm/train_v2/launch.sh both                # same as above
#   bash slurm/train_v2/launch.sh stage1              # Stage 1 only
#   bash slurm/train_v2/launch.sh stage2 --resume /path/to/ckpt
#   bash slurm/train_v2/launch.sh both --depends-on 12345
#
# Environment variables (override defaults):
#   N_GPUS=3              Number of A100 GPUs (default: 3)
#   REPO_SRC=...          Repository path on Picasso
#   RESULTS_DST=...       Results directory
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
STAGE="${1:-both}"
shift || true

DEPENDS_ON=""
export RESUME_CKPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            export RESUME_CKPT="$2"
            shift 2
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/train_v2/launch.sh {stage1|stage2|both} [--resume CKPT] [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "NeuroMF v2 TRAINING LAUNCHER" >&2
echo "==========================================" >&2
echo "Time:  $(date)" >&2
echo "Stage: ${STAGE}" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"
export N_GPUS="${N_GPUS:-3}"

# Scale resources with GPU count
CPUS=$((16 * N_GPUS))
MEM=$((64 * N_GPUS))
if [ "$CPUS" -gt 128 ]; then CPUS=128; fi
if [ "$MEM" -gt 480 ]; then MEM=480; fi

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Configs:     ${CONFIGS_DIR}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "  GPUs:        ${N_GPUS}" >&2
echo "  CPUs:        ${CPUS}" >&2
echo "  Memory:      ${MEM}G" >&2
echo "  Resume:      ${RESUME_CKPT:-none}" >&2
echo "" >&2

# Config chains
STAGE1_CONFIG="${REPO_SRC}/configs/train_v2_stage1.yaml ${CONFIGS_DIR}/train_v2_stage1.yaml"
STAGE2_CONFIG="${REPO_SRC}/configs/train_v2_stage2.yaml ${CONFIGS_DIR}/train_v2_stage2.yaml"
export STAGE2_CONFIG  # bridge.sh reads this
export STAGE2_WALL_TIME="10-00:00:00"

# ========================================================================
# HELPER: submit a GPU training job
# ========================================================================
submit_gpu_job() {
    local JOB_NAME="$1"
    local TRAIN_CONFIG_VAR="$2"
    local WALL_TIME="$3"
    local LOG_PREFIX="$4"
    local DEP="$5"

    local SBATCH_ARGS=(
        --parsable
        --job-name="${JOB_NAME}"
        --time="${WALL_TIME}"
        --ntasks=1
        --cpus-per-task="${CPUS}"
        --mem="${MEM}G"
        --constraint=dgx
        --gres="gpu:${N_GPUS}"
        --output="${RESULTS_DST}/${LOG_PREFIX}_%j.out"
        --error="${RESULTS_DST}/${LOG_PREFIX}_%j.err"
        --export=ALL
    )

    if [ -n "${DEP}" ]; then
        SBATCH_ARGS+=(--dependency="afterok:${DEP}")
        echo "  Dependency: afterok:${DEP}" >&2
    fi

    export TRAIN_CONFIG="${TRAIN_CONFIG_VAR}"

    local RAW_ID
    RAW_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")
    # Clean job ID: strip whitespace, take only digits
    local JOB_ID
    JOB_ID=$(echo "${RAW_ID}" | tr -dc '0-9')

    echo "  Job ID:    ${JOB_ID}" >&2
    echo "  Wall time: ${WALL_TIME}" >&2
    echo "  Logs:      ${RESULTS_DST}/${LOG_PREFIX}_${JOB_ID}.{out,err}" >&2
    echo "" >&2

    echo "${JOB_ID}"
}

# ========================================================================
# HELPER: submit the CPU bridge job
# ========================================================================
submit_bridge_job() {
    local DEP="$1"

    local SBATCH_ARGS=(
        --parsable
        --job-name="neuromf_v2_bridge"
        --time="0-00:10:00"
        --ntasks=1
        --cpus-per-task=2
        --mem=4G
        --output="${RESULTS_DST}/v2_bridge_%j.out"
        --error="${RESULTS_DST}/v2_bridge_%j.err"
        --export=ALL
    )

    if [ -n "${DEP}" ]; then
        SBATCH_ARGS+=(--dependency="afterok:${DEP}")
        echo "  Dependency: afterok:${DEP}" >&2
    fi

    local RAW_ID
    RAW_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/bridge.sh")
    local JOB_ID
    JOB_ID=$(echo "${RAW_ID}" | tr -dc '0-9')

    echo "  Job ID:    ${JOB_ID}" >&2
    echo "  Logs:      ${RESULTS_DST}/v2_bridge_${JOB_ID}.{out,err}" >&2
    echo "" >&2

    echo "${JOB_ID}"
}

# ========================================================================
# CREATE OUTPUT DIRECTORIES
# ========================================================================
mkdir -p "${RESULTS_DST}/v2_stage1"
mkdir -p "${RESULTS_DST}/v2_stage2"

# ========================================================================
# SUBMIT JOBS
# ========================================================================

case "${STAGE}" in
    stage1)
        echo "--- Stage 1: FM Pretraining (500 epochs) ---" >&2
        JOB1=$(submit_gpu_job \
            "neuromf_v2_s1" \
            "${STAGE1_CONFIG}" \
            "5-00:00:00" \
            "v2_stage1/train" \
            "${DEPENDS_ON}")
        echo "==========================================" >&2
        echo "Stage 1 submitted: Job ${JOB1}" >&2
        echo "Monitor: squeue -j ${JOB1}" >&2
        echo "${JOB1}"
        ;;

    stage2)
        if [ -z "${RESUME_CKPT}" ]; then
            echo "ERROR: Stage 2 requires --resume /path/to/stage1/best.ckpt" >&2
            exit 1
        fi
        echo "--- Stage 2: α-Flow MF Fine-tuning (1000 epochs) ---" >&2
        echo "  Resuming from: ${RESUME_CKPT}" >&2
        JOB2=$(submit_gpu_job \
            "neuromf_v2_s2" \
            "${STAGE2_CONFIG}" \
            "10-00:00:00" \
            "v2_stage2/train" \
            "${DEPENDS_ON}")
        echo "==========================================" >&2
        echo "Stage 2 submitted: Job ${JOB2}" >&2
        echo "Monitor: squeue -j ${JOB2}" >&2
        echo "${JOB2}"
        ;;

    both)
        echo "==========================================" >&2
        echo "FULLY AUTOMATED TWO-STAGE PIPELINE" >&2
        echo "==========================================" >&2
        echo "" >&2

        # --- Stage 1: GPU job ---
        echo "--- [1/3] Stage 1: FM Pretraining (500 epochs, ~5 days) ---" >&2
        JOB1=$(submit_gpu_job \
            "neuromf_v2_s1" \
            "${STAGE1_CONFIG}" \
            "5-00:00:00" \
            "v2_stage1/train" \
            "${DEPENDS_ON}")

        # --- Bridge: CPU job (finds best checkpoint, submits Stage 2) ---
        echo "--- [2/3] Bridge: Find best checkpoint (afterok:${JOB1}) ---" >&2
        JOB_BRIDGE=$(submit_bridge_job "${JOB1}")

        # --- Stage 2 is submitted BY the bridge script ---
        echo "--- [3/3] Stage 2 will be submitted by bridge job ${JOB_BRIDGE} ---" >&2
        echo "  Stage 2 config:  ${STAGE2_CONFIG}" >&2
        echo "  Stage 2 wall:    ${STAGE2_WALL_TIME}" >&2
        echo "" >&2

        echo "==========================================" >&2
        echo "PIPELINE SUBMITTED" >&2
        echo "==========================================" >&2
        echo "" >&2
        echo "Job chain:" >&2
        echo "  Stage 1 (GPU):  ${JOB1}  →  5 days, ${N_GPUS} GPUs" >&2
        echo "  Bridge  (CPU):  ${JOB_BRIDGE}  →  afterok:${JOB1}, finds best ckpt" >&2
        echo "  Stage 2 (GPU):  (submitted by bridge)  →  10 days, ${N_GPUS} GPUs" >&2
        echo "" >&2
        echo "Monitor:" >&2
        echo "  squeue -u \$(whoami)" >&2
        echo "  tail -f ${RESULTS_DST}/v2_stage1/train_${JOB1}.out" >&2
        echo "  tail -f ${RESULTS_DST}/v2_bridge_${JOB_BRIDGE}.out" >&2
        echo "" >&2
        echo "Total estimated time: ~15 days" >&2

        # Output Stage 1 job ID (for external chaining)
        echo "${JOB1}"
        ;;

    *)
        echo "Unknown stage: ${STAGE}" >&2
        echo "Usage: bash slurm/train_v2/launch.sh {stage1|stage2|both}" >&2
        exit 1
        ;;
esac
