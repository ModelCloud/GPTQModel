"""Bazel rules for parallel P32 CUDA template shards."""


def _p32_shard(name, bits, family):
    native.genrule(
        name = name,
        srcs = [
            "qvq_p32_abi.h",
            "qvq_p32_cuda.cu",
            "qvq_p32_internal.h",
        ],
        outs = ["objects/shards/%s.o" % name],
        cmd = """
set -eu
cuda_home="$${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "$${cuda_home}/bin/nvcc" ]; then cuda_home=/usr/local/cuda-13.3; fi
if [ ! -x "$${cuda_home}/bin/nvcc" ]; then
  echo "nvcc not found; set CUDA_HOME to a CUDA toolkit containing nvcc" >&2
  exit 1
fi
"$${cuda_home}/bin/nvcc" \\
  -std=c++17 -O3 -lineinfo -cudart=shared -Xcompiler=-fPIC \\
  --objdir-as-tempdir --split-compile=4 \\
  -DQVQ_P32_SHARD_BITS=%d -DQVQ_P32_SHARD_FAMILY=%d \\
  -gencode arch=compute_90,code=sm_90 \\
  -I$$(dirname $(location qvq_p32_abi.h)) \\
  -c -o "$@" $(location qvq_p32_cuda.cu)
""" % (bits, family),
        tags = ["manual"],
    )


def p32_shards():
    for bits in [4, 5, 6, 7]:
        _p32_shard("qvq_p32_standard_bits%d_object" % bits, bits, 1)
        _p32_shard("qvq_p32_grouped_bits%d_object" % bits, bits, 2)
