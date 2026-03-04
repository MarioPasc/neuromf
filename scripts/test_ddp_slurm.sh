#!/usr/bin/env bash
# =============================================================================
# DDP SMOKE TEST — SLURM submission
#
# Submits a short job to a real compute node with 2 GPUs and runs the
# diagnostic test. This proves whether the SLURMEnvironment bug reproduces
# in the actual production environment (real SLURM_JOB_ID, SLURM_NTASKS=1).
#
# Usage (from Picasso login node):
#   bash scripts/test_ddp_slurm.sh
#   bash scripts/test_ddp_slurm.sh --gpus 3
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="${REPO_SRC:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_DIR="${CONDA_ENV_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/neuromf}"
N_GPUS="${1:-2}"

# Strip --gpus flag if passed
if [ "$N_GPUS" = "--gpus" ]; then
    N_GPUS="${2:-2}"
fi

echo "=========================================="
echo "DDP SMOKE TEST — SLURM SUBMISSION"
echo "=========================================="
echo "Repo:    ${REPO_SRC}"
echo "Env:     ${ENV_DIR}"
echo "GPUs:    ${N_GPUS}"
echo ""

JOB_ID=$(sbatch --parsable \
    --job-name="ddp_smoke_test" \
    --time=00:10:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --constraint=dgx \
    --gres="gpu:${N_GPUS}" \
    --output="${REPO_SRC}/ddp_test_%j.out" \
    --error="${REPO_SRC}/ddp_test_%j.err" \
    --wrap="
set -euo pipefail
echo '=== SLURM ENVIRONMENT ==='
echo \"SLURM_JOB_ID=\${SLURM_JOB_ID}\"
echo \"SLURM_NTASKS=\${SLURM_NTASKS}\"
echo \"SLURM_GPUS_ON_NODE=\${SLURM_GPUS_ON_NODE:-not set}\"
echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-not set}\"
echo \"Hostname: \$(hostname)\"
echo ''
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo ''

export PATH='${ENV_DIR}/bin:\${PATH}'
export CONDA_PREFIX='${ENV_DIR}'
cd '${REPO_SRC}'

echo '=== Running diagnostic (without fix then with fix) ==='
python scripts/test_ddp.py --diagnose --gpus ${N_GPUS}
")

echo "Submitted job: ${JOB_ID}"
echo ""
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Output:   ${REPO_SRC}/ddp_test_${JOB_ID}.out"
echo "Errors:   ${REPO_SRC}/ddp_test_${JOB_ID}.err"
echo ""
echo "After completion, check results:"
echo "  cat ${REPO_SRC}/ddp_test_${JOB_ID}.out"
