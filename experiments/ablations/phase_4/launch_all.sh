#!/usr/bin/env bash
# =============================================================================
# PHASE 4 ABLATIONS — SUBMIT ALL JOBS
#
# Reads ablations.yaml and submits one SLURM job per ablation using the
# existing train_worker.sh. Each ablation gets its own results directory.
#
# Usage (from Picasso login node, inside the repo):
#   bash experiments/ablations/phase_4/launch_all.sh
#   bash experiments/ablations/phase_4/launch_all.sh --resume  # resume from last.ckpt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESUME_FLAG="${1:-}"

echo "=========================================="
echo "PHASE 4: ABLATION LAUNCHER (ALL)"
echo "=========================================="
echo "Time: $(date)"
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

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}"
    exit 1
fi

# ========================================================================
# PARSE MANIFEST — extract ablation entries via Python
# ========================================================================
# Outputs lines of: name|config_path|n_gpus|description
ABLATION_LIST=$(python -c "
import yaml, sys
with open('${MANIFEST}') as f:
    data = yaml.safe_load(f)
for name, info in data['ablations'].items():
    print(f\"{name}|{info['config']}|{info['n_gpus']}|{info['description']}\")
")

if [ -z "${ABLATION_LIST}" ]; then
    echo "ERROR: No ablations found in ${MANIFEST}"
    exit 1
fi

echo "Ablations to submit:"
echo "${ABLATION_LIST}" | while IFS='|' read -r name config n_gpus desc; do
    echo "  ${name}: ${desc} (${n_gpus} GPUs)"
done
echo ""

# ========================================================================
# SUBMIT JOBS
# ========================================================================
SUBMITTED=0

while IFS='|' read -r NAME CONFIG N_GPUS DESC; do
    echo "------------------------------------------"
    echo "Submitting: ${NAME}"
    echo "  Config: ${CONFIG}"
    echo "  GPUs:   ${N_GPUS}"
    echo "  Desc:   ${DESC}"

    # Resolve ablation config path relative to manifest directory
    ABLATION_CONFIG="${SCRIPT_DIR}/${CONFIG}"
    if [ ! -f "${ABLATION_CONFIG}" ]; then
        echo "  ERROR: Config not found: ${ABLATION_CONFIG}"
        echo "  SKIPPING."
        continue
    fi

    # TRAIN_CONFIG: Picasso overlay + ablation diff (space-separated for nargs="+")
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
            echo "  Resuming from: ${LAST_CKPT}"
        else
            echo "  No last.ckpt found, starting fresh."
        fi
    fi

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

    echo "  Job ID: ${JOB_ID}"
    echo "  Monitor: squeue -j ${JOB_ID}"
    echo "  Logs: ${ABLATION_RESULTS}/logs/"
    SUBMITTED=$((SUBMITTED + 1))

done <<< "${ABLATION_LIST}"

echo ""
echo "=========================================="
echo "SUBMISSION COMPLETE"
echo "=========================================="
echo "Jobs submitted: ${SUBMITTED}"
echo ""
echo "Monitor all:  squeue -u \$(whoami) -n nmf_abl"
echo "TensorBoard:  tensorboard --logdir ${RESULTS_DST}/ablations/"
