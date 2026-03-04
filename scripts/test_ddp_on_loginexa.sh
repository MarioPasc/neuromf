#!/usr/bin/env bash
# =============================================================================
# DDP SMOKE TEST — for loginexa (GPU-enabled login node on Picasso)
#
# loginexa has GPUs but no 'module' system and conda is not in PATH.
# We source conda.sh directly from the installation directory.
#
# Usage (from picasso3):
#   ssh loginexa "cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf && bash scripts/test_ddp_on_loginexa.sh"
#
# Or interactively:
#   ssh loginexa
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf
#   bash scripts/test_ddp_on_loginexa.sh
#
# Optional:
#   bash scripts/test_ddp_on_loginexa.sh --gpus 3
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "DDP SMOKE TEST ON LOGINEXA"
echo "=========================================="
echo "Time:     $(date)"
echo "Host:     $(hostname)"
echo "User:     $(whoami)"
echo ""

# --- Activate conda env ---
# On loginexa there is no 'module' command and conda is not in PATH.
# Find the conda installation and source its init script directly.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-neuromf}"
echo "Activating conda environment '${CONDA_ENV_NAME}'..."

conda_activated=0

# 1. If conda is already in PATH (e.g. on compute nodes after module load)
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" && conda_activated=1
fi

# 2. Search common conda install locations under home
if [ "$conda_activated" -eq 0 ]; then
    HOME_DIR="${HOME:-/mnt/home/users/tic_163_uma/mpascual}"
    for candidate in \
        "${HOME_DIR}/miniconda3" \
        "${HOME_DIR}/miniforge3" \
        "${HOME_DIR}/mambaforge" \
        "${HOME_DIR}/anaconda3" \
        "${HOME_DIR}/.conda" \
        "/opt/conda" \
        "/opt/miniconda3"; do
        if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
            echo "  Found conda at: ${candidate}"
            source "${candidate}/etc/profile.d/conda.sh"
            conda activate "${CONDA_ENV_NAME}" && conda_activated=1
            break
        fi
    done
fi

# 3. Last resort: look for the environment's python directly
if [ "$conda_activated" -eq 0 ]; then
    echo "  WARNING: Could not find conda installation."
    echo "  Searching for environment python directly..."
    for candidate in \
        "${HOME_DIR:-$HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin" \
        "${HOME_DIR:-$HOME}/miniforge3/envs/${CONDA_ENV_NAME}/bin" \
        "${HOME_DIR:-$HOME}/mambaforge/envs/${CONDA_ENV_NAME}/bin" \
        "${HOME_DIR:-$HOME}/anaconda3/envs/${CONDA_ENV_NAME}/bin"; do
        if [ -x "${candidate}/python" ]; then
            echo "  Found env python at: ${candidate}/python"
            export PATH="${candidate}:${PATH}"
            conda_activated=1
            break
        fi
    done
fi

if [ "$conda_activated" -eq 0 ]; then
    echo "ERROR: Could not activate conda environment '${CONDA_ENV_NAME}'." >&2
    echo "Please set CONDA_PREFIX to your conda installation path and retry:" >&2
    echo "  CONDA_PREFIX=/path/to/miniconda3 bash scripts/test_ddp_on_loginexa.sh" >&2
    exit 1
fi

echo "Python:   $(which python)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo ""

# --- GPU info ---
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo ""
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-all}"
echo "torch.cuda.device_count(): $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# --- Simulate SLURM env to test the fix ---
# On loginexa SLURM_JOB_ID may or may not be set.
# To test our fix, we simulate the problematic SLURM environment:
echo "--- Test 1: With simulated SLURM env (the bug scenario) ---"
echo "Setting SLURM_JOB_ID=99999 SLURM_NTASKS=1 to simulate the bug..."
SLURM_JOB_ID=99999 SLURM_NTASKS=1 python scripts/test_ddp.py "$@"
echo ""

echo "--- Test 2: Without SLURM env (baseline) ---"
echo "Unsetting SLURM vars to verify DDP works natively..."
env -u SLURM_JOB_ID -u SLURM_NTASKS python scripts/test_ddp.py "$@"
echo ""

echo "=========================================="
echo "ALL DDP TESTS PASSED"
echo "=========================================="
echo ""
echo "The LightningEnvironment fix works correctly."
echo "You can now submit real training jobs with 3 GPUs."
