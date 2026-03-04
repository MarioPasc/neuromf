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
# Use the env's bin/ directory directly.
ENV_DIR="${CONDA_ENV_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/neuromf}"
echo "Activating environment from: ${ENV_DIR}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "ERROR: Python not found at ${ENV_DIR}/bin/python" >&2
    echo "Set CONDA_ENV_DIR to the correct path and retry." >&2
    exit 1
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export CONDA_PREFIX="${ENV_DIR}"

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

# --- Diagnose: run WITHOUT fix then WITH fix under simulated SLURM ---
echo "--- Diagnostic: proving the bug exists ---"
echo "Setting SLURM_JOB_ID=99999 SLURM_NTASKS=1 to simulate compute node..."
echo ""
SLURM_JOB_ID=99999 SLURM_NTASKS=1 python scripts/test_ddp.py --diagnose "$@"
