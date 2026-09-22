"""Build macros for independently cached QVQ WGMMA rate families."""

def _wgmma_rate_object(bits):
    native.genrule(
        name = "qvq_wgmma_w%d_object" % bits,
        srcs = [
            "qvq_wgmma_cuda.cu",
            "qvq_wgmma_raw_abi.cu",
            "qvq_wgmma_raw_abi.h",
            "//gptqmodel_ext/qvq/p32:qvq_p32_abi.h",
            "@cutlass//:headers",
            "@cutlass//:include/cutlass/cutlass.h",
        ],
        outs = ["objects/w%d/qvq_wgmma_w%d.o" % (bits, bits)],
        cmd = """
set -eu
cuda_home="$${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "$${cuda_home}/bin/nvcc" ]; then cuda_home=/usr/local/cuda-13.3; fi
if [ ! -x "$${cuda_home}/bin/nvcc" ]; then
  echo "nvcc not found; set CUDA_HOME to a CUDA toolkit containing nvcc" >&2
  exit 1
fi
cutlass_header=$(location @cutlass//:include/cutlass/cutlass.h)
cutlass_include=$$(dirname "$$(dirname "$${cutlass_header}")")
"$${cuda_home}/bin/nvcc" \
  -std=c++17 -O3 -lineinfo -cudart=shared --expt-relaxed-constexpr \
  -diag-suppress=20013,20015 -Xcompiler=-fPIC,-fvisibility=hidden \
  -static-global-template-stub=false --split-compile=4 \
  -DQVQ_WGMMA_BITS_ONLY=%d \
  -gencode arch=compute_90a,code=sm_90a \
  -I$$(dirname $(location qvq_wgmma_raw_abi.h)) \
  -I"$${cutlass_include}" \
  -c -o "$@" $(location qvq_wgmma_raw_abi.cu)
""" % bits,
        tags = ["manual"],
    )

def wgmma_rate_objects():
    for bits in [4, 5, 6, 7]:
        _wgmma_rate_object(bits)
