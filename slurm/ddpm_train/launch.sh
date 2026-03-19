#!/usr/bin/env bash
# =============================================================================
# DDPM BASELINE TRAINING — SLURM LAUNCHER
#
# Same UNet, same data, same budget as MOTFM. Only loss + sampler differ.
#
# Usage (from login node):
#   bash slurm/ddpm_train/launch.sh
#   bash slurm/ddpm_train/launch.sh --depends-on <JOB_ID>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPENDS_ON=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --depends-on) DEPENDS_ON="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"
export DDPM_CONFIG="${REPO_SRC}/configs/picasso/ddpm/fomo60k_unconditional_3d.yaml"

N_GPUS="${N_GPUS:-3}"

# Step-matching (same as MOTFM)
TRAIN_SAMPLES=5471
PAPER_STEPS=150000
STEPS_PER_EPOCH=$(( (TRAIN_SAMPLES + N_GPUS - 1) / N_GPUS ))
NUM_EPOCHS=$(( (PAPER_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))

echo "==========================================" >&2
echo "DDPM BASELINE TRAINING LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "Config:  ${DDPM_CONFIG}" >&2
echo "GPUs:    ${N_GPUS}" >&2
echo "Epochs:  ${NUM_EPOCHS} (step-matched to ${PAPER_STEPS})" >&2
echo "" >&2

TOTAL_CPUS=$(( N_GPUS * 16 ))
[ "$TOTAL_CPUS" -gt 128 ] && TOTAL_CPUS=128
TOTAL_MEM=$(( N_GPUS * 64 ))
[ "$TOTAL_MEM" -gt 480 ] && TOTAL_MEM=480

SBATCH_ARGS=(
    --parsable
    --job-name="ddpm_${N_GPUS}gpu"
    --time=5-00:00:00
    --ntasks=1
    --cpus-per-task="${TOTAL_CPUS}"
    --mem="${TOTAL_MEM}G"
    --constraint=dgx
    --gres="gpu:${N_GPUS}"
    --output="${RESULTS_DST}/ddpm/train_%j.out"
    --error="${RESULTS_DST}/ddpm/train_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency: afterok:${DEPENDS_ON}" >&2
fi

mkdir -p "${RESULTS_DST}/ddpm"

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "DDPM JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:  ${JOB_ID}" >&2
echo "Monitor: squeue -j ${JOB_ID}" >&2
echo "Logs:    ${RESULTS_DST}/ddpm/train_${JOB_ID}.{out,err}" >&2

echo "${JOB_ID}"
