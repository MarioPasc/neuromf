#!/usr/bin/env bash
# =============================================================================
# VAE VALIDATION — SLURM LAUNCHER (atomic)
#
# Login-node script that submits the VAE validation pipeline on Picasso.
# Uses a single A100 GPU to validate the frozen MAISI VAE on 20 FOMO-60K volumes
# at 192^3 resolution (requires ~15GB VRAM, too much for local RTX 4060 8GB).
#
# Expected time: ~15-30 min on A100
#
# Usage (from login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf
#   bash slurm/validate_vae/launch.sh
#   bash slurm/validate_vae/launch.sh --depends-on 12345
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
            echo "Usage: bash slurm/validate_vae/launch.sh [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "VAE VALIDATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="phase_0_vae_validation"
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
mkdir -p "${RESULTS_DST}/phase_0/vae_validation/metrics"
mkdir -p "${RESULTS_DST}/phase_0/vae_validation/reconstructions"
mkdir -p "${RESULTS_DST}/phase_0/vae_validation/figures"
mkdir -p "${RESULTS_DST}/phase_0/vae_validation/latent_stats"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_p0_vae"
    --time=0-01:00:00
    --ntasks=1
    --cpus-per-task=8
    --mem=32G
    --constraint=dgx
    --gres=gpu:1
    --output="${RESULTS_DST}/phase_0/vae_validation_%j.out"
    --error="${RESULTS_DST}/phase_0/vae_validation_%j.err"
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
echo "Logs:      ${RESULTS_DST}/phase_0/vae_validation_${JOB_ID}.{out,err}" >&2
echo "Results:   ${RESULTS_DST}/phase_0/" >&2
echo "" >&2
echo "After completion:" >&2
echo "  1. Check HTML report: ${RESULTS_DST}/phase_0/verification_report.html" >&2
echo "  2. Run Phase 0 tests:  python -m pytest tests/test_maisi_vae_wrapper.py -v --tb=short" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
