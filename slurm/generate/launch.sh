#!/usr/bin/env bash
# =============================================================================
# NEUROMF GENERATION — SLURM LAUNCHER (atomic)
#
# All paths are read from the merged config (configs/picasso/generate.yaml).
# CLI args override config values where specified.
#
# Usage (from login node):
#   bash slurm/generate/launch.sh
#   bash slurm/generate/launch.sh --n-samples 500
#   bash slurm/generate/launch.sh \
#       --norm-correction 1.65 --auto-calibrate --variance-rescale --comparison
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DEPENDS_ON=""
N_SAMPLES=""
RUN_DIR_ARG=""
CHECKPOINT_ARG=""
ENHANCEMENT_PARTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR_ARG="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_ARG="$2"
            shift 2
            ;;
        --norm-correction)
            ENHANCEMENT_PARTS+=("--norm-correction" "$2")
            shift 2
            ;;
        --sigma-inject)
            ENHANCEMENT_PARTS+=("--sigma-inject" "$2" "$3" "$4" "$5")
            shift 5
            ;;
        --auto-calibrate)
            ENHANCEMENT_PARTS+=("--auto-calibrate")
            shift
            ;;
        --variance-rescale)
            ENHANCEMENT_PARTS+=("--variance-rescale")
            shift
            ;;
        --comparison)
            ENHANCEMENT_PARTS+=("--comparison")
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash slurm/generate/launch.sh [OPTIONS]" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  --run-dir PATH       Override output directory (default: from config)" >&2
            echo "  --n-samples N        Override number of samples (default: from config)" >&2
            echo "  --depends-on ID      Wait for job ID to finish" >&2
            echo "  --checkpoint PATH    Override checkpoint path (default: from config)" >&2
            echo "  --norm-correction F  Norm correction gamma (1.0 = no correction)" >&2
            echo "  --auto-calibrate     Auto-compute sigma_inject from baseline" >&2
            echo "  --variance-rescale   Enable per-channel variance rescaling" >&2
            echo "  --comparison         Generate both baseline + enhanced" >&2
            exit 1
            ;;
    esac
done

# ========================================================================
# CONFIGURATION — read paths from merged config
# ========================================================================
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
export REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf}"
export CONFIGS_DIR="${CONFIGS_DIR:-${REPO_SRC}/configs/picasso}"

# Resolve Python: prefer neuromf conda env, fall back to system python
if command -v conda &>/dev/null; then
    NEUROMF_PYTHON="$(conda run -n "${CONDA_ENV_NAME}" which python 2>/dev/null || echo "")"
fi
NEUROMF_PYTHON="${NEUROMF_PYTHON:-python}"

# Read all paths from config in a single Python call
CONFIG_JSON=$(cd "${REPO_SRC}" && "${NEUROMF_PYTHON}" -c "
import json
from pathlib import Path
from neuromf.config import load_merged_config
cfg = load_merged_config(
    [Path('${CONFIGS_DIR}/generate.yaml')],
    configs_dir=Path('${CONFIGS_DIR}'),
    include_generate=True,
)
print(json.dumps({
    'generation_dir': str(cfg.paths.generation_dir),
    'checkpoint': str(cfg.paths.checkpoint),
    'latent_stats': str(cfg.paths.latent_stats),
}))
" 2>&1) || {
    echo "ERROR: Failed to read paths from config (${CONFIGS_DIR}/generate.yaml)" >&2
    echo "       Python used: ${NEUROMF_PYTHON}" >&2
    echo "       Output: ${CONFIG_JSON:-<empty>}" >&2
    exit 1
}

CFG_GENERATION_DIR=$(echo "${CONFIG_JSON}" | "${NEUROMF_PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['generation_dir'])")
CFG_CHECKPOINT=$(echo "${CONFIG_JSON}" | "${NEUROMF_PYTHON}" -c "import sys,json; print(json.load(sys.stdin)['checkpoint'])")

# Resolve RUN_DIR: CLI > config (generation_dir parent)
if [ -n "${RUN_DIR_ARG}" ]; then
    export RUN_DIR="${RUN_DIR_ARG}"
else
    # generation_dir from config is e.g. .../run_xxx/generation → parent is the run dir
    export RUN_DIR="$(dirname "${CFG_GENERATION_DIR}")"
fi

# Resolve checkpoint: CLI > config
if [ -n "${CHECKPOINT_ARG}" ]; then
    export CKPT_PATH="${CHECKPOINT_ARG}"
else
    export CKPT_PATH="${CFG_CHECKPOINT}"
fi

export EXPERIMENT_NAME="phase_5_gen_neuromf"
export GEN_N_SAMPLES="${N_SAMPLES}"

# Enhancement flags (passed through to generate_latents.py via worker.sh)
if [ ${#ENHANCEMENT_PARTS[@]} -gt 0 ]; then
    export ENHANCEMENT_FLAGS="${ENHANCEMENT_PARTS[*]}"
else
    export ENHANCEMENT_FLAGS=""
fi

# ========================================================================
# DISPLAY CONFIGURATION
# ========================================================================
echo "==========================================" >&2
echo "NEUROMF GENERATION LAUNCHER" >&2
echo "==========================================" >&2
echo "Time:        $(date)" >&2
echo "Run dir:     ${RUN_DIR}" >&2
echo "Checkpoint:  ${CKPT_PATH}" >&2
if [ -n "${N_SAMPLES}" ]; then
    echo "n_samples:   ${N_SAMPLES} (override)" >&2
fi
if [ -n "${ENHANCEMENT_FLAGS}" ]; then
    echo "Enhancements: ${ENHANCEMENT_FLAGS}" >&2
fi
echo "" >&2

# Verify the run directory is accessible
if ! mkdir -p "${RUN_DIR}" 2>/dev/null; then
    echo "ERROR: Cannot create run directory: ${RUN_DIR}" >&2
    echo "       Are you running on Picasso? Check configs/picasso/generate.yaml paths." >&2
    exit 1
fi

# Create output subdirectories
mkdir -p "${RUN_DIR}/generation/latents"
mkdir -p "${RUN_DIR}/generation/volumes"
mkdir -p "${RUN_DIR}/features"
mkdir -p "${RUN_DIR}/metrics"

# ========================================================================
# SUBMIT JOB
# ========================================================================
SBATCH_ARGS=(
    --parsable
    --job-name="neuromf_p5_gen"
    --time=1-00:00:00
    --ntasks=1
    --cpus-per-task=16
    --mem=128G
    --constraint=dgx
    --gres=gpu:1
    --output="${RUN_DIR}/generate_%j.out"
    --error="${RUN_DIR}/generate_%j.err"
    --export=ALL
)
if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
    echo "Dependency: afterok:${DEPENDS_ON}" >&2
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/worker.sh")

echo "==========================================" >&2
echo "NEUROMF JOB SUBMITTED" >&2
echo "==========================================" >&2
echo "Job ID:    ${JOB_ID}" >&2
echo "Monitor:   squeue -j ${JOB_ID}" >&2
echo "Logs:      ${RUN_DIR}/generate_${JOB_ID}.{out,err}" >&2
echo "Output:    ${RUN_DIR}/generation/" >&2

# Job ID to stdout for orchestration capture
echo "${JOB_ID}"
