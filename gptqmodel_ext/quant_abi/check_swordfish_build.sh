#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compile-only check of changed Swordfish TUs; does not link a backend library.
set -euo pipefail
abi_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
kernel_root="$abi_root/../swordfish"
cutlass_root="${1:?usage: check_swordfish_build.sh CUTLASS_4_7_1 OUTPUT_DIRECTORY [100a|103a|110a]}"
abi_out="${2:?output directory required}"
abi_arch="${3:-100a}"
case "$abi_arch" in 100a|103a|110a) ;; *) exit 2 ;; esac
mkdir -p -- "$abi_out"
torch_root="$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')"
cuda_root="${CUDA_HOME:-/usr/local/cuda}"
for unit in swordfish_mm swordfish_prefill swordfish_prefill_f16; do
  "$cuda_root/bin/nvcc" -std=c++17 -O2 --threads 2 \
    --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress=128,20012 \
    "-gencode=arch=compute_$abi_arch,code=sm_$abi_arch" \
    -DENABLE_BF16 -DUSE_CUDA -DTORCH_TARGET_VERSION=0x020a000000000000 \
    -Xcompiler=-fPIC -include "$kernel_root/swordfish_arch_macros.cuh" \
    -I"$torch_root/include" -I"$torch_root/include/torch/csrc/api/include" \
    -I"$kernel_root" -I"$cutlass_root/include" -I"$cutlass_root/tools/util/include" \
    -c "$kernel_root/libtorch_stable/quantization/swordfish/$unit.cu" \
    -o "$abi_out/${unit}_${abi_arch}.o" > "$abi_out/${unit}_${abi_arch}.log" 2>&1
done
