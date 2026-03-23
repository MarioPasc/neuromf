#!/usr/bin/env bash
#SBATCH -J neuromf_v2_bridge
#SBATCH --time=0-00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=bridge_%j.out
#SBATCH --error=bridge_%j.err

# =============================================================================
# NeuroMF v2 BRIDGE — finds Stage 1 best checkpoint and submits Stage 2
#
# This is a lightweight (no GPU) job that:
# 1. Waits for Stage 1 to complete (via SLURM --dependency=afterok)
# 2. Finds the best FID checkpoint from Stage 1
# 3. Submits Stage 2 with --resume pointing to that checkpoint
#
# Expected env vars (exported by launch_full.sh):
#   REPO_SRC, CONFIGS_DIR, RESULTS_DST, CONDA_ENV_NAME, N_GPUS,
#   STAGE2_CONFIG, STAGE2_WALL_TIME
# =============================================================================

set -euo pipefail

echo "==========================================="
echo "NeuroMF v2 BRIDGE"
echo "==========================================="
echo "Time:     $(date)"
echo "Hostname: $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo ""

# ========================================================================
# ENVIRONMENT SETUP (minimal — just need Python for checkpoint finder)
# ========================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_SRC}"

# ========================================================================
# FIND BEST CHECKPOINT FROM STAGE 1
# ========================================================================
echo "Searching for best FID checkpoint in ${RESULTS_DST}..."

BEST_CKPT=$(python slurm/train_v2/find_best_checkpoint.py "${RESULTS_DST}")
FIND_EXIT=$?

if [ "$FIND_EXIT" -ne 0 ] || [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: Could not find Stage 1 checkpoint. Stage 2 will NOT be submitted."
    echo "Check: ls ${RESULTS_DST}/runs/run_*/checkpoints/"
    exit 1
fi

echo "Found best checkpoint: ${BEST_CKPT}"
echo ""

# Verify file exists and has nonzero size
if [ ! -f "${BEST_CKPT}" ]; then
    echo "ERROR: Checkpoint file does not exist: ${BEST_CKPT}"
    exit 1
fi
CKPT_SIZE=$(stat -c%s "${BEST_CKPT}" 2>/dev/null || echo "0")
echo "Checkpoint size: ${CKPT_SIZE} bytes"
if [ "$CKPT_SIZE" -lt 1000 ]; then
    echo "ERROR: Checkpoint file suspiciously small (${CKPT_SIZE} bytes)"
    exit 1
fi

# ========================================================================
# SUBMIT STAGE 2
# ========================================================================
echo ""
echo "==========================================="
echo "SUBMITTING STAGE 2"
echo "==========================================="

export RESUME_CKPT="${BEST_CKPT}"
export TRAIN_CONFIG="${STAGE2_CONFIG}"

N_GPUS="${N_GPUS:-3}"
CPUS=$((16 * N_GPUS))
MEM=$((64 * N_GPUS))
if [ "$CPUS" -gt 128 ]; then CPUS=128; fi
if [ "$MEM" -gt 480 ]; then MEM=480; fi

SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_v2_s2"
    --time="${STAGE2_WALL_TIME:-7-00:00:00}"
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --mem="${MEM}G"
    --constraint=dgx
    --gres="gpu:${N_GPUS}"
    --output="${RESULTS_DST}/v2_stage2/train_%j.out"
    --error="${RESULTS_DST}/v2_stage2/train_%j.err"
    --export=ALL
)

# Capture sbatch output, separating stderr from stdout for clean ID parsing
STDERR_TMP=$(mktemp)
RAW_STDOUT=$(sbatch "${SBATCH_ARGS[@]}" "${REPO_SRC}/slurm/train_v2/worker.sh" 2>"${STDERR_TMP}") || {
    echo "ERROR: sbatch failed for Stage 2" >&2
    cat "${STDERR_TMP}"
    rm -f "${STDERR_TMP}"
    exit 1
}
if [ -s "${STDERR_TMP}" ]; then
    echo "sbatch warnings:" >&2
    cat "${STDERR_TMP}" >&2
fi
rm -f "${STDERR_TMP}"

# Extract numeric job ID (sbatch --parsable prints "JOBID" or "JOBID;cluster")
STAGE2_JOB_ID=$(echo "${RAW_STDOUT}" | grep -oE '^[0-9]+' | head -1)

if [ -z "${STAGE2_JOB_ID}" ]; then
    echo "ERROR: Could not parse Stage 2 job ID from: '${RAW_STDOUT}'"
    exit 1
fi

echo "Stage 2 submitted!"
echo "  Job ID:      ${STAGE2_JOB_ID}"
echo "  Checkpoint:  ${BEST_CKPT}"
echo "  Wall time:   ${STAGE2_WALL_TIME:-7-00:00:00}"
echo "  GPUs:        ${N_GPUS}"
echo "  Monitor:     squeue -j ${STAGE2_JOB_ID}"
echo "  Logs:        ${RESULTS_DST}/v2_stage2/train_${STAGE2_JOB_ID}.{out,err}"
echo ""
echo "Bridge complete at $(date)"
