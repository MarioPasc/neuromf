#!/usr/bin/env bash
# =============================================================================
# LATENT PRE-COMPUTATION — SLURM LAUNCHER (atomic)
#
# Login-node script that submits the encoding pipeline on Picasso.
# Uses a single A100 GPU to encode all ~1,100 FOMO-60K volumes through
# the frozen MAISI VAE.
#
# Expected time: ~1-2h on A100 (vs ~6h on RTX 4060)
#
# Usage (from login node):
#   bash slurm/encode/launch.sh
#   bash slurm/encode/launch.sh --depends-on 12345
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/encode/launch.sh [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "LATENT PRE-COMPUTATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_1_latent_encoding"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Configs:     ${CONFIGS_DIR}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "  Conda env:   ${CONDA_ENV_NAME}" >&2
echo "" >&2

# Create output directories
mkdir -p "${RESULTS_DST}/latents"
mkdir -p "${RESULTS_DST}/phase_1/figures"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_p1_encode"
    --time=0-10:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=64G
    --constraint=dgx
    --gres=gpu:1
    --output="${RESULTS_DST}/phase_1/encode_%j.out"
    --error="${RESULTS_DST}/phase_1/encode_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency:  afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${RESULTS_DST}/phase_1/encode_${JOB_ID}.{out,err}" >&2
echo "Results:   ${RESULTS_DST}/latents/" >&2
echo "" >&2
echo "After completion, run Phase 1 tests:" >&2
echo "  python -m pytest tests/test_latent_dataset.py -v --tb=short" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
