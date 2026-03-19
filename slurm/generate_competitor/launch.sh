#!/usr/bin/env bash
# =============================================================================
# GENERIC COMPETITOR GENERATION — SLURM LAUNCHER
#
# Model-agnostic generation script.  Uses the competitor registry to
# instantiate the correct generator by name.
#
# Usage (from login node):
#   bash slurm/generate_competitor/launch.sh \
#       --model motfm --run-dir /path/to/MOTFM_01032026
#   bash slurm/generate_competitor/launch.sh \
#       --model ddpm --checkpoint /path/to/best.ckpt --n-samples 500
#   bash slurm/generate_competitor/launch.sh \
#       --model scoresde --config /path/config.yaml --checkpoint /path/ckpt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
MODEL=""
RUN_DIR_ARG=""
CHECKPOINT_ARG=""
CONFIG_ARG=""
N_SAMPLES=""
VOLUME_SIZE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";         shift 2 ;;
        --run-dir)       RUN_DIR_ARG="$2";   shift 2 ;;
        --checkpoint)    CHECKPOINT_ARG="$2"; shift 2 ;;
        --config)        CONFIG_ARG="$2";     shift 2 ;;
        --n-samples)     N_SAMPLES="$2";      shift 2 ;;
        --volume-size)   VOLUME_SIZE="$2";    shift 2 ;;
        --depends-on)    DEPENDS_ON="$2";     shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "" >&2
            echo "Usage: bash slurm/generate_competitor/launch.sh --model NAME [OPTIONS]" >&2
            echo "" >&2
            echo "Required:" >&2
            echo "  --model NAME          Competitor name (e.g. motfm, ddpm)" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --run-dir PATH        Output directory (default: auto)" >&2
            echo "  --checkpoint PATH     Model checkpoint (default: model-specific)" >&2
            echo "  --config PATH         Model config (default: model-specific)" >&2
            echo "  --n-samples N         Number of volumes (default: 500)" >&2
            echo "  --volume-size N       Spatial size per axis (default: from config)" >&2
            echo "  --depends-on ID       SLURM dependency" >&2
            exit 1
            ;;
    esac
done

if [ -z "${MODEL}" ]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

export COMPETITOR_MODEL="${MODEL}"

# Model-specific defaults
MODEL_LOWER=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')
case "${MODEL_LOWER}" in
    motfm)
        DEFAULT_CONFIG="${CONFIGS_DIR}/motfm/fomo60k_unconditional_3d.yaml"
        DEFAULT_CHECKPOINT="${RESULTS_DST}/motfm/checkpoints/fomo60k_unconditional_3d/last.ckpt"
        ;;
    ddpm)
        DEFAULT_CONFIG="${CONFIGS_DIR}/ddpm/fomo60k_unconditional_3d.yaml"
        DEFAULT_CHECKPOINT="${RESULTS_DST}/ddpm/checkpoints/fomo60k_unconditional_3d/last.ckpt"
        ;;
    *)
        DEFAULT_CONFIG=""
        DEFAULT_CHECKPOINT=""
        ;;
esac

export COMPETITOR_CONFIG="${CONFIG_ARG:-${DEFAULT_CONFIG}}"
export COMPETITOR_CHECKPOINT="${CHECKPOINT_ARG:-${DEFAULT_CHECKPOINT}}"
export GEN_N_SAMPLES="${N_SAMPLES:-500}"
export GEN_VOLUME_SIZE="${VOLUME_SIZE:-}"

# Resolve run directory
if [ -n "${RUN_DIR_ARG}" ]; then
    export COMPETITOR_RUN_DIR="${RUN_DIR_ARG}"
else
    RUN_DATE=$(date +%d%m%Y)
    export COMPETITOR_RUN_DIR="${RESULTS_DST}/phase_5/${MODEL}_${RUN_DATE}"
fi

# Validate
if [ -z "${COMPETITOR_CONFIG}" ]; then
    echo "ERROR: --config is required for model '${MODEL}' (no default)" >&2
    exit 1
fi
if [ -z "${COMPETITOR_CHECKPOINT}" ]; then
    echo "ERROR: --checkpoint is required for model '${MODEL}' (no default)" >&2
    exit 1
fi

echo "==========================================" >&2
echo "COMPETITOR GENERATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Model:      ${MODEL}" >&2
echo "Config:     ${COMPETITOR_CONFIG}" >&2
echo "Checkpoint: ${COMPETITOR_CHECKPOINT}" >&2
echo "Run dir:    ${COMPETITOR_RUN_DIR}" >&2
echo "n_samples:  ${GEN_N_SAMPLES}" >&2
echo "" >&2

# Create output directories
mkdir -p "${COMPETITOR_RUN_DIR}/generation/volumes"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="${MODEL_LOWER}_gen"
    --time=1-00:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=128G
    --constraint=dgx
    --gres=gpu:1
    --output="${COMPETITOR_RUN_DIR}/generate_%j.out"
    --error="${COMPETITOR_RUN_DIR}/generate_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency: afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:  ${JOB_ID}" >&2
echo "Monitor: squeue -j ${JOB_ID}" >&2

# Job ID to stdout for orchestration
echo "${JOB_ID}"
