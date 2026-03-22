#!/usr/bin/env bash
# =============================================================================
# NEUROMF GENERATION — SLURM LAUNCHER (atomic)
#
# Submits a single NeuroiMF generation + decode job. Generates latents at
# multiple NFE levels, prepares real test volumes, decodes, and visualizes.
#
# Usage (from login node):
#   bash slurm/generate/launch.sh --run-dir /path/to/NeuroiMF_01032026
#   bash slurm/generate/launch.sh --run-dir /path/to/run --n-samples 500
#   bash slurm/generate/launch.sh --run-dir /path/to/run --depends-on 12345
#   bash slurm/generate/launch.sh --run-dir /path/to/run \
#       --checkpoint /path/to/best.ckpt \
#       --norm-correction 1.65 --auto-calibrate --variance-rescale --comparison
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
N_SAMPLES=""
RUN_DIR_ARG=""
CHECKPOINT_ARG=""
ENHANCEMENT_PARTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR_ARG="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_ARG="$2"
            shift 2
            ;;
        --norm-correction)
            ENHANCEMENT_PARTS+=("--norm-correction" "$2")
            shift 2
            ;;
        --sigma-inject)
            ENHANCEMENT_PARTS+=("--sigma-inject" "$2" "$3" "$4" "$5")
            shift 5
            ;;
        --auto-calibrate)
            ENHANCEMENT_PARTS+=("--auto-calibrate")
            shift
            ;;
        --variance-rescale)
            ENHANCEMENT_PARTS+=("--variance-rescale")
            shift
            ;;
        --comparison)
            ENHANCEMENT_PARTS+=("--comparison")
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/generate/launch.sh --run-dir PATH [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --run-dir PATH       Output directory for this run (required)" >&2
            echo "  --n-samples N        Override number of samples (default: from config)" >&2
            echo "  --depends-on ID      Wait for job ID to finish" >&2
            echo "  --checkpoint PATH    Override checkpoint path" >&2
            echo "  --norm-correction F  Norm correction gamma (1.0 = no correction)" >&2
            echo "  --auto-calibrate     Auto-compute sigma_inject from baseline" >&2
            echo "  --variance-rescale   Enable per-channel variance rescaling" >&2
            echo "  --comparison         Generate both baseline + enhanced" >&2
            exit 1
            ;;
    esac
done

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

# Resolve run directory
if [ -n "${RUN_DIR_ARG}" ]; then
    export RUN_DIR="${RUN_DIR_ARG}"
else
    RUN_DATE=$(date +%d%m%Y)
    export RUN_DIR="${RESULTS_DST}/NeuroMF/runs/run_${RUN_DATE}"
    echo "WARNING: --run-dir not specified, using default: ${RUN_DIR}" >&2
fi

# Verify the parent of RUN_DIR is accessible (catches running locally with Picasso defaults)
RUN_DIR_PARENT="$(dirname "${RUN_DIR}")"
if [ ! -d "${RUN_DIR_PARENT}" ] && ! mkdir -p "${RUN_DIR_PARENT}" 2>/dev/null; then
    echo "ERROR: Cannot create run directory parent: ${RUN_DIR_PARENT}" >&2
    echo "       Are you running on Picasso? If running locally, pass --run-dir /local/path" >&2
    exit 1
fi

export EXPERIMENT_NAME="phase_5_gen_neuromf"
export GEN_N_SAMPLES="${N_SAMPLES}"

# Checkpoint override
if [ -n "${CHECKPOINT_ARG}" ]; then
    export CKPT_PATH="${CHECKPOINT_ARG}"
fi

# Enhancement flags (passed through to generate_latents.py via worker.sh)
export ENHANCEMENT_FLAGS="${ENHANCEMENT_PARTS[*]:-}"

echo "==========================================" >&2
echo "NEUROMF GENERATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "Run directory: ${RUN_DIR}" >&2
if [ -n "${N_SAMPLES}" ]; then
    echo "n_samples:     ${N_SAMPLES} (override)" >&2
fi
if [ -n "${CHECKPOINT_ARG}" ]; then
    echo "checkpoint:    ${CHECKPOINT_ARG}" >&2
fi
if [ -n "${ENHANCEMENT_FLAGS}" ]; then
    echo "enhancements:  ${ENHANCEMENT_FLAGS}" >&2
fi
echo "" >&2

# Create output directories
mkdir -p "${RUN_DIR}/generation/latents"
mkdir -p "${RUN_DIR}/generation/volumes"
mkdir -p "${RUN_DIR}/features"
mkdir -p "${RUN_DIR}/metrics"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_p5_gen"
    --time=1-00:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=128G
    --constraint=dgx
    --gres=gpu:1
    --output="${RUN_DIR}/generate_%j.out"
    --error="${RUN_DIR}/generate_%j.err"
    --export=ALL
)
if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency: afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "NEUROMF JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${RUN_DIR}/generate_${JOB_ID}.{out,err}" >&2
echo "Latents:   ${RUN_DIR}/generation/latents/" >&2
echo "Volumes:   ${RUN_DIR}/generation/volumes/" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
