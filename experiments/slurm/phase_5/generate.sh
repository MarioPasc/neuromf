#!/usr/bin/env bash
# =============================================================================
# PHASE 5: GENERATION PIPELINE — ORCHESTRATION
#
# Thin orchestration script that dispatches NeuroiMF and/or MOTFM generation
# jobs by calling the atomic launchers in slurm/generate/ and
# slurm/generate_motfm/.
#
# Usage (from login node):
#   bash experiments/slurm/phase_5/generate.sh
#   bash experiments/slurm/phase_5/generate.sh --motfm
#   bash experiments/slurm/phase_5/generate.sh --motfm --skip-neuromf
#   bash experiments/slurm/phase_5/generate.sh --depends-on <JOB_ID>
#   bash experiments/slurm/phase_5/generate.sh --n-samples 500
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ========================================================================
# PARSE ARGS
# ========================================================================
ENABLE_MOTFM=0
SKIP_NEUROMF=0
DEPENDS_ON=""
N_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash experiments/slurm/phase_5/generate.sh [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --motfm             Also generate MOTFM volumes (separate job)" >&2
            echo "  --skip-neuromf      Skip NeuroiMF generation (use with --motfm)" >&2
            echo "  --depends-on ID     Wait for job ID to finish" >&2
            echo "  --n-samples N       Override number of samples (default: from config)" >&2
            exit 1
            ;;
    esac
done

# Validate flags
if [ "${SKIP_NEUROMF}" -eq 1 ] && [ "${ENABLE_MOTFM}" -eq 0 ]; then
    echo "ERROR: --skip-neuromf requires --motfm" >&2
    exit 1
fi

echo "==========================================" >&2
echo "PHASE 5: GENERATION PIPELINE (orchestration)" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"
RUN_DATE=$(date +%d%m%Y)

# ========================================================================
# DISPATCH NEUROMF
# ========================================================================
NEUROMF_JOB_ID=""
if [ "${SKIP_NEUROMF}" -eq 0 ]; then
    NEUROMF_RUN_DIR="${RESULTS_DST}/phase_5/NeuroiMF_${RUN_DATE}"

    LAUNCH_ARGS=(--run-dir "${NEUROMF_RUN_DIR}")
    if [ -n "${DEPENDS_ON}" ]; then
        LAUNCH_ARGS+=(--depends-on "${DEPENDS_ON}")
    fi
    if [ -n "${N_SAMPLES}" ]; then
        LAUNCH_ARGS+=(--n-samples "${N_SAMPLES}")
    fi

    NEUROMF_JOB_ID=$(bash "${REPO_ROOT}/slurm/generate/launch.sh" "${LAUNCH_ARGS[@]}")
    echo "NeuroiMF dispatched: Job ${NEUROMF_JOB_ID} -> ${NEUROMF_RUN_DIR}" >&2
fi

# ========================================================================
# DISPATCH MOTFM
# ========================================================================
MOTFM_JOB_ID=""
if [ "${ENABLE_MOTFM}" -eq 1 ]; then
    MOTFM_RUN_DIR="${RESULTS_DST}/phase_5/MOTFM_${RUN_DATE}"

    LAUNCH_ARGS=(--run-dir "${MOTFM_RUN_DIR}")
    if [ -n "${DEPENDS_ON}" ]; then
        LAUNCH_ARGS+=(--depends-on "${DEPENDS_ON}")
    fi
    if [ -n "${N_SAMPLES}" ]; then
        LAUNCH_ARGS+=(--n-samples "${N_SAMPLES}")
    fi

    MOTFM_JOB_ID=$(bash "${REPO_ROOT}/slurm/generate_motfm/launch.sh" "${LAUNCH_ARGS[@]}")
    echo "MOTFM dispatched: Job ${MOTFM_JOB_ID} -> ${MOTFM_RUN_DIR}" >&2
fi

# ========================================================================
# SUMMARY
# ========================================================================
echo "" >&2
echo "==========================================" >&2
echo "SUMMARY" >&2
echo "==========================================" >&2
if [ -n "${NEUROMF_JOB_ID}" ]; then
    echo "  NeuroiMF:  Job ${NEUROMF_JOB_ID}  ->  ${NEUROMF_RUN_DIR}" >&2
fi
if [ -n "${MOTFM_JOB_ID}" ]; then
    echo "  MOTFM:     Job ${MOTFM_JOB_ID}  ->  ${MOTFM_RUN_DIR}" >&2
fi
echo "" >&2
echo "Next step: run evaluate.sh with --depends-on and --results-dir" >&2
