#!/usr/bin/env bash
# =============================================================================
# DDPM GENERATION — SLURM LAUNCHER
#
# Usage:
#   bash slurm/generate_ddpm/launch.sh --run-dir /path/to/output
#   bash slurm/generate_ddpm/launch.sh --run-dir /path --checkpoint /path/to/best.ckpt
#   bash slurm/generate_ddpm/launch.sh --run-dir /path --depends-on 12345
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPENDS_ON=""
RUN_DIR_ARG=""
CHECKPOINT_ARG=""
N_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir) RUN_DIR_ARG="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT_ARG="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --depends-on) DEPENDS_ON="$2"; shift 2 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

export DDPM_RUN_DIR="${RUN_DIR_ARG:-${RESULTS_DST}/phase_5/DDPM_$(date +%d%m%Y)}"
export DDPM_CHECKPOINT="${CHECKPOINT_ARG:-${RESULTS_DST}/ddpm/checkpoints/fomo60k_unconditional_3d/last.ckpt}"
export DDPM_CONFIG="${REPO_SRC}/configs/picasso/ddpm/fomo60k_unconditional_3d.yaml"
export GEN_N_SAMPLES="${N_SAMPLES:-500}"

echo "==========================================" >&2
echo "DDPM GENERATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Run dir:    ${DDPM_RUN_DIR}" >&2
echo "Checkpoint: ${DDPM_CHECKPOINT}" >&2
echo "Config:     ${DDPM_CONFIG}" >&2
echo "n_samples:  ${GEN_N_SAMPLES}" >&2

mkdir -p "${DDPM_RUN_DIR}/generation/volumes"

SBATCH_ARGS=(
    --parsable
    --job-name="ddpm_gen"
    --time=1-00:00:00
    --ntasks=1 --cpus-per-task=16 --mem=128G
    --constraint=dgx --gres=gpu:1
    --output="${DDPM_RUN_DIR}/generate_%j.out"
    --error="${DDPM_RUN_DIR}/generate_%j.err"
    --export=ALL
)

[ -n "${DEPENDS_ON}" ] && SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "Job ID: ${JOB_ID}" >&2
echo "${JOB_ID}"
