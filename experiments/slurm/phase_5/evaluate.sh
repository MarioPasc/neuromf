#!/usr/bin/env bash
# =============================================================================
# PHASE 5: EVALUATION — ORCHESTRATION
#
# Thin orchestration script that dispatches NeuroiMF and/or MOTFM evaluation
# jobs by calling the atomic launcher in slurm/evaluate/.
#
# Usage (from login node):
#   bash experiments/slurm/phase_5/evaluate.sh --results-dir .../NeuroiMF_01032026
#   bash experiments/slurm/phase_5/evaluate.sh --motfm \
#       --results-dir .../NeuroiMF_01032026 --motfm-results-dir .../MOTFM_01032026
#   bash experiments/slurm/phase_5/evaluate.sh --motfm --skip-neuromf \
#       --motfm-results-dir .../MOTFM_01032026
#   bash experiments/slurm/phase_5/evaluate.sh --depends-on 95484 --results-dir ...
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
ENABLE_MOTFM=0
SKIP_NEUROMF=0
NEUROMF_RESULTS_DIR=""
MOTFM_RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --motfm)
            ENABLE_MOTFM=1
            shift
            ;;
        --skip-neuromf)
            SKIP_NEUROMF=1
            shift
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        --results-dir)
            NEUROMF_RESULTS_DIR="$2"
            shift 2
            ;;
        --motfm-results-dir)
            MOTFM_RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash experiments/slurm/phase_5/evaluate.sh [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --results-dir PATH        NeuroiMF run directory" >&2
            echo "  --motfm                   Also evaluate MOTFM (separate job)" >&2
            echo "  --motfm-results-dir PATH  MOTFM run directory" >&2
            echo "  --skip-neuromf            Skip NeuroiMF evaluation" >&2
            echo "  --depends-on JOB_ID       Wait for job to finish" >&2
            exit 1
            ;;
    esac
done

# Validate flags
if [ "${SKIP_NEUROMF}" -eq 1 ] && [ "${ENABLE_MOTFM}" -eq 0 ]; then
    echo "ERROR: --skip-neuromf requires --motfm" >&2
    exit 1
fi

if [ "${SKIP_NEUROMF}" -eq 0 ] && [ -z "${NEUROMF_RESULTS_DIR}" ]; then
    echo "ERROR: --results-dir is required for NeuroiMF evaluation" >&2
    exit 1
fi

if [ "${ENABLE_MOTFM}" -eq 1 ] && [ -z "${MOTFM_RESULTS_DIR}" ]; then
    echo "ERROR: --motfm-results-dir is required when --motfm is set" >&2
    exit 1
fi

echo "==========================================" >&2
echo "PHASE 5: EVALUATION (orchestration)" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# DISPATCH NEUROMF EVALUATION
# ========================================================================
NEUROMF_JOB_ID=""
if [ "${SKIP_NEUROMF}" -eq 0 ]; then
    LAUNCH_ARGS=(--run-dir "${NEUROMF_RESULTS_DIR}")
    if [ -n "${DEPENDS_ON}" ]; then
        LAUNCH_ARGS+=(--depends-on "${DEPENDS_ON}")
    fi

    NEUROMF_JOB_ID=$(bash "${REPO_ROOT}/slurm/evaluate/launch.sh" "${LAUNCH_ARGS[@]}")
    echo "NeuroiMF evaluation dispatched: Job ${NEUROMF_JOB_ID} -> ${NEUROMF_RESULTS_DIR}" >&2
fi

# ========================================================================
# DISPATCH MOTFM EVALUATION
# ========================================================================
MOTFM_JOB_ID=""
if [ "${ENABLE_MOTFM}" -eq 1 ]; then
    LAUNCH_ARGS=(--run-dir "${MOTFM_RESULTS_DIR}")
    if [ -n "${DEPENDS_ON}" ]; then
        LAUNCH_ARGS+=(--depends-on "${DEPENDS_ON}")
    fi

    MOTFM_JOB_ID=$(bash "${REPO_ROOT}/slurm/evaluate/launch.sh" "${LAUNCH_ARGS[@]}")
    echo "MOTFM evaluation dispatched: Job ${MOTFM_JOB_ID} -> ${MOTFM_RESULTS_DIR}" >&2
fi

# ========================================================================
# SUMMARY
# ========================================================================
echo "" >&2
echo "==========================================" >&2
echo "SUMMARY" >&2
echo "==========================================" >&2
if [ -n "${NEUROMF_JOB_ID}" ]; then
    echo "  NeuroiMF:  Job ${NEUROMF_JOB_ID}  ->  ${NEUROMF_RESULTS_DIR}" >&2
fi
if [ -n "${MOTFM_JOB_ID}" ]; then
    echo "  MOTFM:     Job ${MOTFM_JOB_ID}  ->  ${MOTFM_RESULTS_DIR}" >&2
fi
echo "" >&2
echo "Next step: run analyse.sh with --depends-on and --results-dir" >&2
