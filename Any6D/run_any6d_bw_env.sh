#!/usr/bin/env bash
set -euo pipefail

source /home/iacopo/miniconda3/etc/profile.d/conda.sh
conda activate any6d_bw

export ANY6D_ROOT="/home/iacopo/cv_final/Any6D"
export TORCH_LIB_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
export FOUNDATIONPOSE_MYCPP_BUILD="$ANY6D_ROOT/foundationpose/mycpp/build"
export NVIDIA_CUDA_RUNTIME_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime"
export NVIDIA_CUDA_NVCC_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvcc"
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda
  elif [ -x /usr/local/cuda-13.2/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda-13.2
  else
    export CUDA_HOME="$CONDA_PREFIX"
  fi
fi
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$NVIDIA_CUDA_NVCC_ROOT/bin:$PATH"
export PYTHONPATH="$ANY6D_ROOT:$ANY6D_ROOT/foundationpose:$ANY6D_ROOT/sam2:$FOUNDATIONPOSE_MYCPP_BUILD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$NVIDIA_CUDA_RUNTIME_ROOT/lib:$NVIDIA_CUDA_NVCC_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:$NVIDIA_CUDA_RUNTIME_ROOT/include:$NVIDIA_CUDA_NVCC_ROOT/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/include:$NVIDIA_CUDA_RUNTIME_ROOT/include:$NVIDIA_CUDA_NVCC_ROOT/include:${C_INCLUDE_PATH:-}"

# Keep caches inside the workspace to avoid read-only home/cache issues.
export ANY6D_RUNTIME_DIR="${ANY6D_RUNTIME_DIR:-/home/iacopo/cv_final/Any6D/.runtime_bw}"
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
