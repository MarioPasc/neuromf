#!/usr/bin/env bash
# =============================================================================
# AUGMENTATION VISUALISATION SUITE — SLURM LAUNCHER (atomic)
#
# Login-node script that submits the augmentation visualisation job on Picasso.
# Produces 18 PNGs (6 visualisations x 3 planes) in ~15 min on 1 A100.
#
# Usage (from login node):
#   bash slurm/augmentation_viz/launch.sh
#   VIZ_SEED=123 bash slurm/augmentation_viz/launch.sh
#   bash slurm/augmentation_viz/launch.sh --depends-on 12345
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
            echo "Usage: bash slurm/augmentation_viz/launch.sh [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "AUGMENTATION VISUALISATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"
export VIZ_SEED="${VIZ_SEED:-42}"

OUTPUT_DIR="${RESULTS_DST}/phase_4/augmentation_viz"

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Configs:     ${CONFIGS_DIR}" >&2
echo "  Output:      ${OUTPUT_DIR}" >&2
echo "  Conda env:   ${CONDA_ENV_NAME}" >&2
echo "  Seed:        ${VIZ_SEED}" >&2
echo "" >&2

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_aug_viz"
    --time=1:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=32G
    --constraint=dgx
    --gres=gpu:1
    --output="${RESULTS_DST}/phase_4/aug_viz_%j.out"
    --error="${RESULTS_DST}/phase_4/aug_viz_%j.err"
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
echo "Logs:      ${RESULTS_DST}/phase_4/aug_viz_${JOB_ID}.{out,err}" >&2
echo "Output:    ${OUTPUT_DIR}" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
