#!/bin/bash
# NeuroMF environment activation script.
# Usage: source scripts/activate.sh

export NEUROMF_ROOT="/media/mpascual/Sandisk2TB/research/neuromf"
export NEUROMF_RESULTS="/media/mpascual/Sandisk2TB/research/neuromf/results"
export NEUROMF_DATA="/media/mpascual/Sandisk2TB/research/neuromf/datasets"
export NEUROMF_CKPT="/media/mpascual/Sandisk2TB/research/neuromf/checkpoints"
export NEUROMF_CODE="/home/mpascual/research/code/neuromf"

conda activate neuromf
echo "NeuroMF environment activated. GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
