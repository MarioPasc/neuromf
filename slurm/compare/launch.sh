#!/usr/bin/env bash
# =============================================================================
# COMPARISON ANALYSIS — SLURM LAUNCHER (atomic)
#
# Submits a single comparison job across all 3 models (NeuroiMF, MOTFM, DDPM).
# Reads pre-computed features/ and metrics/ from each model's results directory,
# then produces cross-model figures, tables, and statistical tests.
#
# Usage (from login node):
#   bash slurm/compare/launch.sh \
#       --neuroimf-dir /path/to/NeuroiMF \
#       --motfm-dir /path/to/MOTFM \
#       --ddpm-dir /path/to/DDPM
#
#   bash slurm/compare/launch.sh \
#       --neuroimf-dir /path/to/NeuroiMF \
#       --motfm-dir /path/to/MOTFM \
#       --ddpm-dir /path/to/DDPM \
#       --output-dir /path/to/comparison \
#       --depends-on "12345:12346:12347" \
#       --n-bootstrap 2000 \
#       --nfe 1 10 50
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
NEUROIMF_DIR_ARG=""
MOTFM_DIR_ARG=""
DDPM_DIR_ARG=""
OUTPUT_DIR_ARG=""
export N_BOOTSTRAP=1000
export SKIP_BOOTSTRAP=0
export SKIP_QUALITATIVE=0
export NFE_LEVELS="1 10 50"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --neuroimf-dir)
            NEUROIMF_DIR_ARG="$2"
            shift 2
            ;;
        --motfm-dir)
            MOTFM_DIR_ARG="$2"
            shift 2
            ;;
        --ddpm-dir)
            DDPM_DIR_ARG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_ARG="$2"
            shift 2
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        --n-bootstrap)
            export N_BOOTSTRAP="$2"
            shift 2
            ;;
        --nfe)
            export NFE_LEVELS=""
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                export NFE_LEVELS="${NFE_LEVELS} $1"
                shift
            done
            ;;
        --skip-bootstrap)
            export SKIP_BOOTSTRAP=1
            shift
            ;;
        --skip-qualitative)
            export SKIP_QUALITATIVE=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/compare/launch.sh [OPTIONS]" >&2
            echo "" >&2
            echo "Required:" >&2
            echo "  --neuroimf-dir PATH     NeuroiMF results directory" >&2
            echo "  --motfm-dir PATH        MOTFM results directory" >&2
            echo "  --ddpm-dir PATH         DDPM results directory" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --output-dir PATH       Comparison output (default: RESULTS_DST/phase_5/comparison_YYYYMMDD)" >&2
            echo "  --depends-on ID         SLURM dependency (colon-separated for multiple)" >&2
            echo "  --n-bootstrap N         Bootstrap samples (default: 1000)" >&2
            echo "  --nfe 1 10 50           NFE levels (default: 1 10 50)" >&2
            echo "  --skip-bootstrap        Skip slow bootstrap CI computation" >&2
            echo "  --skip-qualitative      Skip volume-dependent figures" >&2
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${NEUROIMF_DIR_ARG}" ]; then
    echo "ERROR: --neuroimf-dir is required" >&2
    exit 1
fi
if [ -z "${MOTFM_DIR_ARG}" ]; then
    echo "ERROR: --motfm-dir is required" >&2
    exit 1
fi
if [ -z "${DDPM_DIR_ARG}" ]; then
    echo "ERROR: --ddpm-dir is required" >&2
    exit 1
fi

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

export NEUROIMF_DIR="${NEUROIMF_DIR_ARG}"
export MOTFM_DIR="${MOTFM_DIR_ARG}"
export DDPM_DIR="${DDPM_DIR_ARG}"

# Resolve output directory
if [ -n "${OUTPUT_DIR_ARG}" ]; then
    export COMPARE_OUTPUT_DIR="${OUTPUT_DIR_ARG}"
else
    COMPARE_DATE=$(date +%Y%m%d)
    export COMPARE_OUTPUT_DIR="${RESULTS_DST}/phase_5/comparison_${COMPARE_DATE}"
fi

echo "==========================================" >&2
echo "COMPARISON ANALYSIS LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2
echo "Configuration:" >&2
echo "  NeuroiMF dir:     ${NEUROIMF_DIR}" >&2
echo "  MOTFM dir:        ${MOTFM_DIR}" >&2
echo "  DDPM dir:         ${DDPM_DIR}" >&2
echo "  Output dir:       ${COMPARE_OUTPUT_DIR}" >&2
echo "  NFE levels:       ${NFE_LEVELS}" >&2
echo "  Bootstrap:        ${N_BOOTSTRAP} samples" >&2
echo "  Skip bootstrap:   ${SKIP_BOOTSTRAP}" >&2
echo "  Skip qualitative: ${SKIP_QUALITATIVE}" >&2
echo "" >&2

# ========================================================================
# PRE-FLIGHT: CHECK REQUIRED INPUTS
# ========================================================================
echo "Pre-flight checks:" >&2

MISSING=0
for model_label in NeuroiMF MOTFM DDPM; do
    case "${model_label}" in
        NeuroiMF) model_dir="${NEUROIMF_DIR}" ;;
        MOTFM)    model_dir="${MOTFM_DIR}" ;;
        DDPM)     model_dir="${DDPM_DIR}" ;;
    esac

    if [ -d "${model_dir}/features" ]; then
        echo "  [OK]   ${model_label}: features/" >&2
    else
        echo "  [MISS] ${model_label}: features/ — not found" >&2
        MISSING=1
    fi

    if [ -d "${model_dir}/metrics" ]; then
        echo "  [OK]   ${model_label}: metrics/" >&2
    else
        echo "  [MISS] ${model_label}: metrics/ — not found" >&2
        MISSING=1
    fi
done

if [ "${MISSING}" -eq 1 ] && [ -z "${DEPENDS_ON}" ]; then
    echo "" >&2
    echo "WARNING: Some required directories are missing." >&2
    echo "  If evaluation jobs are still running, use --depends-on ID1:ID2:ID3" >&2
    echo "  to chain the comparison after they complete." >&2
    echo "" >&2
fi
echo "" >&2

# Create output directories
mkdir -p "${COMPARE_OUTPUT_DIR}/figures"
mkdir -p "${COMPARE_OUTPUT_DIR}/tables"
mkdir -p "${COMPARE_OUTPUT_DIR}/data"
mkdir -p "${COMPARE_OUTPUT_DIR}/logs"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_compare"
    --time=0-12:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=64G
    --constraint=dgx
    --gres=gpu:1
    --output="${COMPARE_OUTPUT_DIR}/logs/compare_%j.out"
    --error="${COMPARE_OUTPUT_DIR}/logs/compare_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    # Support colon-separated multiple dependencies
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency:  afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "COMPARISON JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${COMPARE_OUTPUT_DIR}/logs/compare_${JOB_ID}.{out,err}" >&2
echo "Figures:   ${COMPARE_OUTPUT_DIR}/figures/" >&2
echo "Tables:    ${COMPARE_OUTPUT_DIR}/tables/" >&2
echo "Data:      ${COMPARE_OUTPUT_DIR}/data/" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
