#!/usr/bin/env bash
# =============================================================================
# DDP SMOKE TEST — for loginexa (GPU-enabled login node on Picasso)
#
# loginexa has GPUs but no 'module' system for conda.
# Use 'source activate neuromf' instead of 'conda activate neuromf'.
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

# --- Activate conda env (no module system on loginexa) ---
echo "Activating conda environment 'neuromf'..."
# Try conda activate first, fall back to source activate
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate neuromf 2>/dev/null || source activate neuromf
else
    source activate neuromf
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
