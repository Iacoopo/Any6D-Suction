#!/usr/bin/env bash
set -euo pipefail

source /home/iacopo/miniconda3/etc/profile.d/conda.sh
conda activate any6d_blackwell

export ANY6D_ROOT="/home/iacopo/cv_final/Any6D"
export TORCH_LIB_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
export FOUNDATIONPOSE_MYCPP_BUILD="$ANY6D_ROOT/foundationpose/mycpp/build"

# Prefer only a system CUDA toolkit that matches the PyTorch build.
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  else
    export CUDA_HOME="$CONDA_PREFIX"
  fi
fi

export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$ANY6D_ROOT:$ANY6D_ROOT/foundationpose:$ANY6D_ROOT/sam2:$FOUNDATIONPOSE_MYCPP_BUILD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CUDA_HOME/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/include:${C_INCLUDE_PATH:-}"

# Keep caches in the workspace.
export ANY6D_RUNTIME_DIR="${ANY6D_RUNTIME_DIR:-/home/iacopo/cv_final/Any6D/.runtime_blackwell}"
export HF_HOME="${HF_HOME:-$ANY6D_RUNTIME_DIR/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ANY6D_RUNTIME_DIR/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$ANY6D_RUNTIME_DIR/numba}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ANY6D_RUNTIME_DIR/xdg-cache}"

mkdir -p \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$MPLCONFIGDIR" \
  "$NUMBA_CACHE_DIR" \
  "$XDG_CACHE_HOME"

exec "$@"
