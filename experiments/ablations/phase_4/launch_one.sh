#!/usr/bin/env bash
# =============================================================================
# PHASE 4 ABLATIONS — SUBMIT A SINGLE JOB
#
# Submits one ablation from ablations.yaml by name.
#
# Usage (from Picasso login node, inside the repo):
#   bash experiments/ablations/phase_4/launch_one.sh v3_aug
#   bash experiments/ablations/phase_4/launch_one.sh v3_aug --resume
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <ablation_name> [--resume]"
    echo ""
    echo "Available ablations:"
    python -c "
import yaml
with open('${SCRIPT_DIR}/ablations.yaml') as f:
    data = yaml.safe_load(f)
for name, info in data['ablations'].items():
    print(f'  {name:20s} {info[\"description\"]}')
"
    exit 1
fi

NAME="$1"
RESUME_FLAG="${2:-}"

echo "=========================================="
echo "PHASE 4: ABLATION LAUNCHER (SINGLE)"
echo "=========================================="
echo "Time: $(date)"
echo "Ablation: ${NAME}"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="neuromf"
export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf"
export CONFIGS_DIR="${REPO_SRC}/configs/picasso"
export RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results"

WORKER_SCRIPT="${REPO_SRC}/experiments/slurm/phase_4/train_worker.sh"
MANIFEST="${SCRIPT_DIR}/ablations.yaml"

# ========================================================================
# PARSE MANIFEST — extract entry for the requested ablation
# ========================================================================
ENTRY=$(python -c "
import yaml, sys
with open('${MANIFEST}') as f:
    data = yaml.safe_load(f)
abl = data['ablations'].get('${NAME}')
if abl is None:
    print('ERROR: Ablation \"${NAME}\" not found in manifest.', file=sys.stderr)
    print('Available:', ', '.join(data['ablations'].keys()), file=sys.stderr)
    sys.exit(1)
print(f\"{abl['config']}|{abl['n_gpus']}|{abl['description']}\")
")

IFS='|' read -r CONFIG N_GPUS DESC <<< "${ENTRY}"

echo "Config: ${CONFIG}"
echo "GPUs:   ${N_GPUS}"
echo "Desc:   ${DESC}"
echo ""

# Resolve ablation config path relative to manifest directory
ABLATION_CONFIG="${SCRIPT_DIR}/${CONFIG}"
if [ ! -f "${ABLATION_CONFIG}" ]; then
    echo "ERROR: Config not found: ${ABLATION_CONFIG}"
    exit 1
fi

# TRAIN_CONFIG: Picasso overlay + ablation diff
export TRAIN_CONFIG="${CONFIGS_DIR}/train_meanflow.yaml ${ABLATION_CONFIG}"
export N_GPUS="${N_GPUS}"

# Create output directories
ABLATION_RESULTS="${RESULTS_DST}/ablations/${NAME}"
mkdir -p "${ABLATION_RESULTS}/checkpoints"
mkdir -p "${ABLATION_RESULTS}/logs"
mkdir -p "${ABLATION_RESULTS}/samples"
mkdir -p "${ABLATION_RESULTS}/diagnostics"

# Scale resources with GPU count
CPUS=$((16 * N_GPUS))
MEM=$((64 * N_GPUS))
if [ "$CPUS" -gt 128 ]; then CPUS=128; fi
if [ "$MEM" -gt 480 ]; then MEM=480; fi

# Optional resume from last checkpoint
export RESUME_CKPT=""
if [ "${RESUME_FLAG}" = "--resume" ]; then
    LAST_CKPT="${ABLATION_RESULTS}/checkpoints/last.ckpt"
    if [ -f "${LAST_CKPT}" ]; then
        export RESUME_CKPT="${LAST_CKPT}"
        echo "Resuming from: ${LAST_CKPT}"
    else
        echo "No last.ckpt found, starting fresh."
    fi
fi

# ========================================================================
# SUBMIT JOB
# ========================================================================
JOB_ID=$(sbatch --parsable \
    --job-name="nmf_abl_${NAME}" \
    --time=2-00:00:00 \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}G" \
    --constraint=dgx \
    --gres="gpu:${N_GPUS}" \
    --output="${ABLATION_RESULTS}/logs/train_%j.out" \
    --error="${ABLATION_RESULTS}/logs/train_%j.err" \
    --export=ALL \
    "${WORKER_SCRIPT}")

echo ""
echo "=========================================="
echo "JOB SUBMITTED"
echo "=========================================="
echo "Job ID:    ${JOB_ID}"
echo "Ablation:  ${NAME}"
echo "GPUs:      ${N_GPUS}"
echo "Monitor:   squeue -j ${JOB_ID}"
echo "Logs:      ${ABLATION_RESULTS}/logs/train_${JOB_ID}.{out,err}"
echo "Checkpts:  ${ABLATION_RESULTS}/checkpoints/"
echo ""
echo "TensorBoard: tensorboard --logdir ${ABLATION_RESULTS}/logs"
