#!/usr/bin/env bash
#SBATCH -J neuromf_ss_viz
#SBATCH --time=0-00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# SKULL-STRIP VISUALIZATION — SLURM DEPENDENCY JOB
#
# Runs after all Phase B array workers finish. Generates a 3×9 summary
# visualization and prints aggregate statistics from the shared log.
#
# Expected env vars (exported by skull_strip.sh launcher):
#   REPO_SRC, FOMO60K_ROOT, CONDA_ENV_NAME
# =============================================================================

set -euo pipefail

echo "Skull-strip visualization started at: $(date)"
echo "Hostname: $(hostname)"

# ========================================================================
# ENVIRONMENT SETUP
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
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_SRC}"

# ========================================================================
# RUN VISUALIZATION
# ========================================================================
python scripts/skull_strip_defaced.py \
    --phase visualize \
    --fomo60k-root "${FOMO60K_ROOT}"

echo ""
echo "Visualization complete at: $(date)"
echo "Output: ${FOMO60K_ROOT}/_skull_strip_summary.png"
