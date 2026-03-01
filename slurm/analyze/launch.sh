#!/usr/bin/env bash
# =============================================================================
# POST-HOC ANALYSIS — SLURM LAUNCHER (atomic)
#
# Login-node script that submits the analysis job (bootstrap CIs, figures,
# tables, statistical tests). Requires evaluate to have completed first.
#
# Usage (from login node):
#   bash slurm/analyze/launch.sh
#   bash slurm/analyze/launch.sh --depends-on 142991
#   bash slurm/analyze/launch.sh --n-bootstrap 100 --skip-qualitative
#   bash slurm/analyze/launch.sh --results-dir /path/to/run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
export N_BOOTSTRAP=1000
export SKIP_QUALITATIVE=0
export SKIP_BOOTSTRAP=0
export NFE_LEVELS="1 10 50"
export RESULTS_SUBDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        --n-bootstrap)
            export N_BOOTSTRAP="$2"
            shift 2
            ;;
        --skip-qualitative)
            export SKIP_QUALITATIVE=1
            shift
            ;;
        --skip-bootstrap)
            export SKIP_BOOTSTRAP=1
            shift
            ;;
        --nfe)
            export NFE_LEVELS=""
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                export NFE_LEVELS="${NFE_LEVELS} $1"
                shift
            done
            ;;
        --results-dir)
            export RESULTS_SUBDIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/analyze/launch.sh [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --depends-on JOB_ID   Wait for evaluate job to finish" >&2
            echo "  --n-bootstrap N       Number of bootstrap samples (default: 1000)" >&2
            echo "  --skip-qualitative    Skip volume-dependent figures (fig08, fig09)" >&2
            echo "  --skip-bootstrap      Skip bootstrap CI computation" >&2
            echo "  --nfe 1 10 50         NFE levels to analyse" >&2
            echo "  --results-dir PATH    Override results directory" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "POST-HOC ANALYSIS LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_5_analysis"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

# Resolve the results directory
if [ -n "${RESULTS_SUBDIR}" ]; then
    export ANALYSIS_RESULTS_DIR="${RESULTS_SUBDIR}"
else
    export ANALYSIS_RESULTS_DIR="${RESULTS_DST}/phase_5"
fi

echo "Configuration:" >&2
echo "  Repo:             ${REPO_SRC}" >&2
echo "  Results dir:      ${ANALYSIS_RESULTS_DIR}" >&2
echo "  NFE levels:       ${NFE_LEVELS}" >&2
echo "  Bootstrap:        ${N_BOOTSTRAP} samples" >&2
echo "  Skip qualitative: ${SKIP_QUALITATIVE}" >&2
echo "  Skip bootstrap:   ${SKIP_BOOTSTRAP}" >&2
echo "" >&2

# Create output directories
mkdir -p "${ANALYSIS_RESULTS_DIR}/analysis/data"
mkdir -p "${ANALYSIS_RESULTS_DIR}/analysis/figures"
mkdir -p "${ANALYSIS_RESULTS_DIR}/analysis/tables"
mkdir -p "${ANALYSIS_RESULTS_DIR}/analysis/logs"

# ========================================================================
# PRE-FLIGHT: CHECK REQUIRED INPUTS
# ========================================================================
echo "Pre-flight checks:" >&2

MISSING=0
for nfe in ${NFE_LEVELS}; do
    nfe_padded=$(printf '%03d' $nfe)
    feat_file="${ANALYSIS_RESULTS_DIR}/features/gen_med3d_nfe${nfe_padded}.h5"
    json_file="${ANALYSIS_RESULTS_DIR}/metrics/metrics_nfe${nfe_padded}.json"

    if [ -f "${feat_file}" ]; then
        echo "  [OK]   features/gen_med3d_nfe${nfe_padded}.h5" >&2
    else
        echo "  [MISS] features/gen_med3d_nfe${nfe_padded}.h5" >&2
        MISSING=1
    fi

    if [ -f "${json_file}" ]; then
        echo "  [OK]   metrics/metrics_nfe${nfe_padded}.json" >&2
    else
        echo "  [MISS] metrics/metrics_nfe${nfe_padded}.json" >&2
        MISSING=1
    fi
done

if [ -f "${ANALYSIS_RESULTS_DIR}/features/real_med3d.h5" ]; then
    echo "  [OK]   features/real_med3d.h5" >&2
else
    echo "  [MISS] features/real_med3d.h5" >&2
    MISSING=1
fi

if [ "${MISSING}" -eq 1 ] && [ -z "${DEPENDS_ON}" ]; then
    echo "" >&2
    echo "WARNING: Some required files are missing." >&2
    echo "  If evaluate is still running, use --depends-on JOB_ID" >&2
    echo "  to chain the analysis job after it completes." >&2
    echo "" >&2
fi
echo "" >&2

# ========================================================================
# SUBMIT JOB
# ========================================================================
# Analysis is CPU-heavy (bootstrap) but also benefits from GPU for t-SNE
# on large feature sets. Request a single GPU to be safe.
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_p5_analysis"
    --time=0-06:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=64G
    --constraint=dgx
    --gres=gpu:1
    --output="${ANALYSIS_RESULTS_DIR}/analysis/logs/analyse_%j.out"
    --error="${ANALYSIS_RESULTS_DIR}/analysis/logs/analyse_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency=afterok:${DEPENDS_ON})
    echo "Dependency:  afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${ANALYSIS_RESULTS_DIR}/analysis/logs/analyse_${JOB_ID}.{out,err}" >&2
echo "Figures:   ${ANALYSIS_RESULTS_DIR}/analysis/figures/" >&2
echo "Tables:    ${ANALYSIS_RESULTS_DIR}/analysis/tables/" >&2
echo "Data:      ${ANALYSIS_RESULTS_DIR}/analysis/data/" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
