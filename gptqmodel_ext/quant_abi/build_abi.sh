#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
abi_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
abi_out="${1:?usage: build_abi.sh OUTPUT_DIRECTORY}"
mkdir -p -- "$abi_out"
torch_root="$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')"
torch_abi="$(python -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')"
cuda_root="${CUDA_HOME:-/usr/local/cuda}"
"${CXX:-c++}" -std=c++17 -O2 -fPIC -shared -pthread \
  "-D_GLIBCXX_USE_CXX11_ABI=$torch_abi" \
  -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include" \
  -I"$cuda_root/include" "$abi_root/quant_abi.cpp" \
  -L"$torch_root/lib" -Wl,-rpath,"$torch_root/lib" \
  -L"$cuda_root/lib64" -Wl,-rpath,"$cuda_root/lib64" \
  -Wl,--no-undefined -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart \
  -o "$abi_out/libqvq_quant.so"
