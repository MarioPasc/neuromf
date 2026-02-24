#!/usr/bin/env bash
# =============================================================================
# SynthSeg Setup for Picasso (or any HPC without FreeSurfer)
#
# Run on the LOGIN NODE (internet required). Creates:
#   1. A separate conda env (synthseg) with TensorFlow + Keras
#   2. Clones the SynthSeg repo
#   3. Downloads pretrained model weights
#
# Usage:
#   bash scripts/setup_synthseg.sh --install-dir /path/to/tools
#
# After setup, configure configs/picasso/generate.yaml:
#   synthseg:
#     command:
#       - "/path/to/tools/synthseg_env/bin/python"
#       - "/path/to/tools/SynthSeg/scripts/commands/SynthSeg_predict.py"
# =============================================================================

set -euo pipefail

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
INSTALL_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/setup_synthseg.sh --install-dir /path/to/tools"
            exit 1
            ;;
    esac
done

if [ -z "${INSTALL_DIR}" ]; then
    echo "Error: --install-dir is required"
    echo "Usage: bash scripts/setup_synthseg.sh --install-dir /path/to/tools"
    exit 1
fi

SYNTHSEG_REPO="${INSTALL_DIR}/SynthSeg"
CONDA_ENV_DIR="${INSTALL_DIR}/synthseg_env"

echo "=========================================="
echo "SynthSeg Setup"
echo "=========================================="
echo "Install dir:  ${INSTALL_DIR}"
echo "Repo dir:     ${SYNTHSEG_REPO}"
echo "Conda env:    ${CONDA_ENV_DIR}"
echo ""

mkdir -p "${INSTALL_DIR}"

# ========================================================================
# STEP 1: CLONE SYNTHSEG REPO
# ========================================================================
echo "--- Step 1: Clone SynthSeg repository ---"
if [ -d "${SYNTHSEG_REPO}" ]; then
    echo "SynthSeg repo already exists at ${SYNTHSEG_REPO}, pulling latest..."
    cd "${SYNTHSEG_REPO}" && git pull && cd -
else
    git clone https://github.com/BBillot/SynthSeg.git "${SYNTHSEG_REPO}"
fi
echo ""

# ========================================================================
# STEP 2: CREATE CONDA ENVIRONMENT
# ========================================================================
echo "--- Step 2: Create conda environment (Python 3.8 + TensorFlow 2.2) ---"

if [ -d "${CONDA_ENV_DIR}" ]; then
    echo "Conda env already exists at ${CONDA_ENV_DIR}, skipping creation."
else
    conda create -p "${CONDA_ENV_DIR}" python=3.8 -y
    echo "Installing dependencies..."
    "${CONDA_ENV_DIR}/bin/pip" install \
        "tensorflow-gpu==2.2.0" \
        "keras==2.3.1" \
        "protobuf==3.20.3" \
        "numpy==1.23.5" \
        "nibabel==5.0.1" \
        "matplotlib==3.6.2" \
        "h5py==2.10.0"
fi

# Add SynthSeg to PYTHONPATH by creating a .pth file
SITE_PACKAGES=$("${CONDA_ENV_DIR}/bin/python" -c "import site; print(site.getsitepackages()[0])")
echo "${SYNTHSEG_REPO}" > "${SITE_PACKAGES}/synthseg.pth"
echo "Added SynthSeg to PYTHONPATH via ${SITE_PACKAGES}/synthseg.pth"
echo ""

# ========================================================================
# STEP 3: DOWNLOAD PRETRAINED MODELS
# ========================================================================
echo "--- Step 3: Check pretrained models ---"

MODELS_DIR="${SYNTHSEG_REPO}/models"
REQUIRED_MODELS=(
    "synthseg_2.0.h5"
    "synthseg_robust_2.0.h5"
    "synthseg_parc_2.0.h5"
    "synthseg_qc_2.0.h5"
)

ALL_PRESENT=true
for model in "${REQUIRED_MODELS[@]}"; do
    if [ -f "${MODELS_DIR}/${model}" ]; then
        SIZE=$(stat -c%s "${MODELS_DIR}/${model}" 2>/dev/null || echo "?")
        echo "[OK]   ${model} (${SIZE} bytes)"
    else
        echo "[MISS] ${model}"
        ALL_PRESENT=false
    fi
done

if [ "${ALL_PRESENT}" = false ]; then
    echo ""
    echo "WARNING: Some model files are missing!"
    echo "Download them from the UCL SharePoint link in the SynthSeg README:"
    echo "  https://github.com/BBillot/SynthSeg#installation"
    echo "Then copy them to: ${MODELS_DIR}/"
    echo ""
    echo "Alternatively, if you have access to the models elsewhere,"
    echo "copy them manually to ${MODELS_DIR}/"
fi
echo ""

# ========================================================================
# STEP 4: VERIFY INSTALLATION
# ========================================================================
echo "--- Step 4: Verify installation ---"

"${CONDA_ENV_DIR}/bin/python" -c "
import sys
print(f'Python: {sys.version}')
try:
    import tensorflow as tf
    print(f'TensorFlow: {tf.__version__}')
except ImportError as e:
    print(f'TensorFlow import failed: {e}')
try:
    import keras
    print(f'Keras: {keras.__version__}')
except ImportError as e:
    print(f'Keras import failed: {e}')
try:
    import nibabel
    print(f'nibabel: {nibabel.__version__}')
except ImportError as e:
    print(f'nibabel import failed: {e}')
try:
    from SynthSeg.predict_synthseg import predict
    print('SynthSeg predict: OK')
except ImportError as e:
    print(f'SynthSeg import failed: {e}')
"

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "To use SynthSeg in NeuroMF evaluation, set in configs/picasso/generate.yaml:"
echo ""
echo "  synthseg:"
echo "    command:"
echo "      - \"${CONDA_ENV_DIR}/bin/python\""
echo "      - \"${SYNTHSEG_REPO}/scripts/commands/SynthSeg_predict.py\""
echo ""
echo "Test manually:"
echo "  ${CONDA_ENV_DIR}/bin/python ${SYNTHSEG_REPO}/scripts/commands/SynthSeg_predict.py --help"
