#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
build_dir=${1:?usage: bash build_abi.sh /absolute/build/directory}
mkdir -p -- "$build_dir"
build_dir=$(cd -- "$build_dir" && pwd)
nvcc=${QVQ_NVCC:-/usr/local/cuda/bin/nvcc}
jobs=${QVQ_BUILD_JOBS:-4}
python3 "$source_dir/generate_kernels.py" --check || python3 "$source_dir/generate_kernels.py"
export source_dir build_dir nvcc
compile_one() {
  local source=$1
  local object="$build_dir/$(basename "${source%.cu}").o"
  "$nvcc" -std=c++17 -O3 -lineinfo -DQVQ_MARLIN_STANDALONE \
    --expt-relaxed-constexpr --expt-extended-lambda \
    -static-global-template-stub=false -Xcompiler=-fPIC \
    -gencode arch=compute_80,code=sm_80 -I "$source_dir" \
    -c "$source" -o "$object" >"$object.log" 2>&1
}
export -f compile_one
sources=("$source_dir/marlin_abi.cu")
for source in "$source_dir"/kernel_*.cu; do
  case "$source" in *lora*) continue;; esac
  sources+=("$source")
done
printf '%s\0' "${sources[@]}" | xargs -0 -n 1 -P "$jobs" bash -c 'compile_one "$1"' _
objects=()
for source in "${sources[@]}"; do
  objects+=("$build_dir/$(basename "${source%.cu}").o")
done
"$nvcc" -shared -cudart=shared -Xlinker=--no-undefined \
  "${objects[@]}" -o "$build_dir/libqvq_marlin.so"
printf 'Built %s\n' "$build_dir/libqvq_marlin.so"
