#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible env recipe for the QVQ nsys profile (docs/qvq_nsys_profile_llama32_1b.md).
# Box facts this was written against: no system torch, system Python 3.14, CUDA 13.3 toolkit at
# /usr/local/cuda (no nvcc on PATH), nsys at /usr/local/bin/nsys, uv at ~/.local/bin/uv,
# NVIDIA PG506-230 (Hopper-class, 96 GB).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${QVQ_PROFILE_VENV:-$REPO_ROOT/.venv-qvq-profile}"
PY_VERSION="${QVQ_PROFILE_PYTHON:-3.12}"
UV="${UV:-$HOME/.local/bin/uv}"
TORCH_INDEX="${QVQ_PROFILE_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

"$UV" python install "$PY_VERSION"
"$UV" venv --python "$PY_VERSION" "$VENV"
# torch CUDA wheels first so uv does not resolve the CPU wheel from PyPI.
"$UV" pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX" torch
"$UV" pip install --python "$VENV/bin/python" -r "$REPO_ROOT/requirements.txt" pytest
# Editable install without build isolation so the CUDA extensions are JIT-built against this torch.
"$UV" pip install --python "$VENV/bin/python" --no-build-isolation --no-deps -e "$REPO_ROOT"

# The box's /usr/local/cuda ships nvcc + core headers only: the math-library headers (cublas, cusparse,
# cusolver, curand, cufft ...) that torch's ATen/cuda/CUDAContext.h includes are missing.  Pointing CPATH at
# the whole nvidia/cu13 wheel include dir fails ("CUDA compiler and CUDA toolkit headers are incompatible"),
# so expose ONLY the headers the system toolkit lacks through a shim directory.
WHEEL_INC="$VENV/lib/python$PY_VERSION/site-packages/nvidia/cu13/include"
SHIM_INC="$VENV/cuda-shim-include"
mkdir -p "$SHIM_INC"
for h in "$WHEEL_INC"/*; do
  b="$(basename "$h")"
  [ -e "$CUDA_HOME/include/$b" ] || ln -sf "$h" "$SHIM_INC/$b"
done
export CPATH="$SHIM_INC${CPATH:+:$CPATH}"

# Pre-build the QVQ CUDA JIT extension (qvq_gemv/viterbi/hadamard/yaqa .cu) so profiling runs do not
# include the compile.  Parallel nvcc keeps this to a few minutes.
GPTQMODEL_QVQ_NVCC_THREADS="${GPTQMODEL_QVQ_NVCC_THREADS:-8}" \
  "$VENV/bin/python" -c "from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda, qvq_cuda_error; ok = prewarm_qvq_cuda(); print(\"qvq_cuda loaded:\", ok, qvq_cuda_error() if not ok else \"\")"

echo "venv ready: $VENV"
"$VENV/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
