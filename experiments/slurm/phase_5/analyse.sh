#!/usr/bin/env bash
# =============================================================================
# PHASE 5: POST-HOC ANALYSIS — SLURM LAUNCHER
#
# Login-node script that submits the analysis job (bootstrap CIs, figures,
# tables, statistical tests). Requires evaluate.sh to have completed first.
#
# Usage (from login node):
#   bash experiments/slurm/phase_5/analyse.sh
#   bash experiments/slurm/phase_5/analyse.sh --depends-on 142991
#   bash experiments/slurm/phase_5/analyse.sh --n-bootstrap 100 --skip-qualitative
#   bash experiments/slurm/phase_5/analyse.sh --results-dir /path/to/run
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
            echo "Unknown argument: $1"
            echo "Usage: bash experiments/slurm/phase_5/analyse.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --depends-on JOB_ID   Wait for evaluate job to finish"
            echo "  --n-bootstrap N       Number of bootstrap samples (default: 1000)"
            echo "  --skip-qualitative    Skip volume-dependent figures (fig08, fig09)"
            echo "  --skip-bootstrap      Skip bootstrap CI computation"
            echo "  --nfe 1 10 50         NFE levels to analyse"
            echo "  --results-dir PATH    Override results directory"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "PHASE 5: POST-HOC ANALYSIS LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_5_analysis"
export CONDA_ENV_NAME="neuromf"

export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

# Resolve the results directory
if [ -n "${RESULTS_SUBDIR}" ]; then
    export ANALYSIS_RESULTS_DIR="${RESULTS_SUBDIR}"
else
    export ANALYSIS_RESULTS_DIR="${RESULTS_DST}/phase_5"
fi

echo "Configuration:"
echo "  Repo:             ${REPO_SRC}"
echo "  Results dir:      ${ANALYSIS_RESULTS_DIR}"
echo "  NFE levels:       ${NFE_LEVELS}"
echo "  Bootstrap:        ${N_BOOTSTRAP} samples"
echo "  Skip qualitative: ${SKIP_QUALITATIVE}"
echo "  Skip bootstrap:   ${SKIP_BOOTSTRAP}"
echo ""

# Create output directories
mkdir -p "${ANALYSIS_RESULTS_DIR}/analysis/data"
mkdir -p "${ANALYSIS_RESULTS_DIR}/analysis/figures"
mkdir -p "${ANALYSIS_RESULTS_DIR}/analysis/tables"
mkdir -p "${ANALYSIS_RESULTS_DIR}/analysis/logs"

# ========================================================================
# PRE-FLIGHT: CHECK REQUIRED INPUTS
# ========================================================================
echo "Pre-flight checks:"

MISSING=0
for nfe in ${NFE_LEVELS}; do
    nfe_padded=$(printf '%03d' $nfe)
    feat_file="${ANALYSIS_RESULTS_DIR}/features/gen_med3d_nfe${nfe_padded}.h5"
    json_file="${ANALYSIS_RESULTS_DIR}/metrics/metrics_nfe${nfe_padded}.json"

    if [ -f "${feat_file}" ]; then
        echo "  [OK]   features/gen_med3d_nfe${nfe_padded}.h5"
    else
        echo "  [MISS] features/gen_med3d_nfe${nfe_padded}.h5"
        MISSING=1
    fi

    if [ -f "${json_file}" ]; then
        echo "  [OK]   metrics/metrics_nfe${nfe_padded}.json"
    else
        echo "  [MISS] metrics/metrics_nfe${nfe_padded}.json"
        MISSING=1
    fi
done

if [ -f "${ANALYSIS_RESULTS_DIR}/features/real_med3d.h5" ]; then
    echo "  [OK]   features/real_med3d.h5"
else
    echo "  [MISS] features/real_med3d.h5"
    MISSING=1
fi

if [ "${MISSING}" -eq 1 ] && [ -z "${DEPENDS_ON}" ]; then
    echo ""
    echo "WARNING: Some required files are missing."
    echo "  If evaluate.sh is still running, use --depends-on JOB_ID"
    echo "  to chain the analysis job after it completes."
    echo ""
fi
echo ""

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
    echo "Dependency:  afterok:${DEPENDS_ON}"
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/analyse_worker.sh")

echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${ANALYSIS_RESULTS_DIR}/analysis/logs/analyse_${JOB_ID}.{out,err}"
echo "Figures:   ${ANALYSIS_RESULTS_DIR}/analysis/figures/"
echo "Tables:    ${ANALYSIS_RESULTS_DIR}/analysis/tables/"
echo "Data:      ${ANALYSIS_RESULTS_DIR}/analysis/data/"
