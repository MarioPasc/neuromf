#!/usr/bin/env bash
# =============================================================================
# MOTFM TRAINING — SLURM LAUNCHER (atomic)
#
# Login-node script that submits the MOTFM training job on FOMO-60K.
# Requires data prep to be complete (fomo60k_3d.h5).
#
# Step-matched to paper budget:
#   Paper:  750 scans × 200 epochs × accum=1 = 150,000 optimizer steps
#           (1× RTX 4090, batch=1, no accumulation)
#   Ours:   5,471 scans, batch=1, accum=1
#           steps/epoch = ceil(5471 / N_GPUS)
#           110 epochs × ceil(5471/4) = 150,480 steps ≈ 150K
#
# Timing estimates (batch_size=1, accum=1, ~1.65s/sample/GPU):
#   1 GPU:  ~2.5h/epoch → 110 epochs = 275h (11.5 days)
#   2 GPUs: ~1.3h/epoch → 110 epochs = 138h (5.7 days)
#   4 GPUs: ~0.6h/epoch → 110 epochs =  69h (2.9 days)  ← default
#   8 GPUs: ~0.3h/epoch → 110 epochs =  35h (1.5 days)
#
# eff_batch = batch_size × accum × N_GPUS = 1 × 1 × N_GPUS.
# Paper used eff_batch=1. With N_GPUS>1, eff_batch grows linearly —
# this is standard DDP and commonly accepted for fair comparison.
#
# Usage (from login node):
#   bash slurm/motfm_train/launch.sh
#   N_GPUS=4 bash slurm/motfm_train/launch.sh
#   bash slurm/motfm_train/launch.sh --depends-on <JOB_ID>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGS
# ========================================================================
DEPENDS_ON=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/motfm_train/launch.sh [--depends-on JOB_ID]" >&2
            exit 1
            ;;
    esac
done

echo "==========================================" >&2
echo "MOTFM TRAINING LAUNCHER" >&2
echo "==========================================" >&2
echo "Time: $(date)" >&2
echo "" >&2

# ========================================================================
# CONFIGURATION
# ========================================================================
export EXPERIMENT_NAME="motfm_train"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"

export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"
export RESULTS_DST="${RESULTS_DST:-/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results}"

export MOTFM_CONFIG="${CONFIGS_DIR}/motfm/fomo60k_unconditional_3d.yaml"

# Number of GPUs — default 4 for ~3-day step-matched training
export N_GPUS="${N_GPUS:-4}"

# Scale resources with GPU count
CPUS=$((16 * N_GPUS))
MEM=$((64 * N_GPUS))

# Cap at DGX limits (128 cores, ~500GB)
if [ "$CPUS" -gt 128 ]; then CPUS=128; fi
if [ "$MEM" -gt 480 ]; then MEM=480; fi

# Step-matched training budget
TRAIN_SAMPLES=5471
PAPER_STEPS=150000
NUM_EPOCHS=110  # from config — step-matched for 4 GPUs

# Estimate timing
EPOCH_SECS=$((TRAIN_SAMPLES * 165 / (100 * N_GPUS)))  # ~1.65s/sample/GPU
STEPS_PER_EPOCH=$(( (TRAIN_SAMPLES + N_GPUS - 1) / N_GPUS ))  # ceil division
TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))
TOTAL_HOURS=$(( EPOCH_SECS * NUM_EPOCHS / 3600 ))

echo "Configuration:" >&2
echo "  Repo:        ${REPO_SRC}" >&2
echo "  Config:      ${MOTFM_CONFIG}" >&2
echo "  Results:     ${RESULTS_DST}" >&2
echo "  Conda env:   ${CONDA_ENV_NAME}" >&2
echo "  GPUs:        ${N_GPUS}" >&2
echo "  CPUs:        ${CPUS}" >&2
echo "  Memory:      ${MEM}G" >&2
echo "  Eff. batch:  $((1 * 1 * N_GPUS)) (batch=1 × accum=1 × ${N_GPUS} GPUs)" >&2
echo "  Epochs:      ${NUM_EPOCHS}" >&2
echo "  Steps/epoch: ~${STEPS_PER_EPOCH}" >&2
echo "  Total steps: ~${TOTAL_STEPS} (paper: ${PAPER_STEPS})" >&2
echo "  Est. time:   ~${TOTAL_HOURS}h (~$((TOTAL_HOURS / 24 + 1)) days)" >&2
echo "" >&2

# Create output directories
mkdir -p "${RESULTS_DST}/motfm"

# ========================================================================
# SUBMIT JOB
# ========================================================================
# Wall time with ~40% buffer over estimated time
EST_WALL_HOURS=$(( TOTAL_HOURS * 140 / 100 ))
EST_WALL_DAYS=$(( (EST_WALL_HOURS + 23) / 24 ))  # ceil to days
# Minimum 2 days, maximum 7 days
if [ "$EST_WALL_DAYS" -lt 2 ]; then EST_WALL_DAYS=2; fi
if [ "$EST_WALL_DAYS" -gt 7 ]; then EST_WALL_DAYS=7; fi
WALL_TIME="${EST_WALL_DAYS}-00:00:00"

SBATCH_ARGS=(
    --parsable
    --job-name="motfm_${N_GPUS}gpu"
    --time="${WALL_TIME}"
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --mem="${MEM}G"
    --constraint=dgx
    --gres="gpu:${N_GPUS}"
    --output="${RESULTS_DST}/motfm/train_%j.out"
    --error="${RESULTS_DST}/motfm/train_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency: afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:       ${JOB_ID}" >&2
echo "GPUs:         ${N_GPUS}" >&2
echo "Wall time:    ${WALL_TIME}" >&2
echo "Monitor:      squeue -j ${JOB_ID}" >&2
echo "Logs:         ${RESULTS_DST}/motfm/train_${JOB_ID}.{out,err}" >&2
echo "Checkpoints:  ${RESULTS_DST}/motfm/checkpoints/" >&2
echo "" >&2
echo "Chain with generation:" >&2
echo "  bash experiments/slurm/phase_5/generate.sh --motfm --depends-on ${JOB_ID}" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
