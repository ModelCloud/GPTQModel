"""Bazel rules for parallel P32 CUDA template shards."""


def _p32_shard(name, bits, family, kind, stage, split_compile):
    shard_srcs = [
        "qvq_p32_abi.h",
        "qvq_p32_cuda.cu",
        "qvq_p32_internal.h",
    ]
    if kind == 3 or kind == 4:
        shard_srcs += [
            "qvq_p32_large_m2_kernel.cuh",
            "qvq_p32_large_m2_launch.cuh",
        ]
    elif kind == 5 or kind == 6:
        shard_srcs += [
            "qvq_p32_large_m_grid_kernel.cuh",
            "qvq_p32_large_m_grid_launch.cuh",
        ]
    native.genrule(
        name = name,
        srcs = shard_srcs,
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
  --objdir-as-tempdir --split-compile=%d \\
  -DQVQ_P32_SHARD_BITS=%d -DQVQ_P32_SHARD_FAMILY=%d \\
  -DQVQ_P32_SHARD_KIND=%d \\
  -DQVQ_P32_SHARD_STAGE=%d \\
  -gencode arch=compute_90,code=sm_90 \\
  -I$$(dirname $(location qvq_p32_abi.h)) \\
  -c -o "$@" $(location qvq_p32_cuda.cu)
""" % (split_compile, bits, family, kind, stage),
        tags = ["manual"],
    )


def p32_shards():
    for bits in [4, 5, 6, 7]:
        _p32_shard(
            "qvq_p32_standard_scalar_bits%d_object" % bits,
            bits,
            1,
            1,
            0,
            1,
        )
        _p32_shard(
            "qvq_p32_standard_block_bits%d_object" % bits,
            bits,
            1,
            2,
            0,
            1,
        )
        _p32_shard(
            "qvq_p32_standard_large_m2_low_bits%d_object" % bits,
            bits,
            1,
            3,
            0,
            1,
        )
        for stage in [3, 4]:
            _p32_shard(
                "qvq_p32_standard_large_m2_stage%d_bits%d_object" % (stage, bits),
                bits,
                1,
                4,
                stage,
                1,
            )
        _p32_shard(
            "qvq_p32_standard_large_m_grid_low_bits%d_object" % bits,
            bits,
            1,
            5,
            0,
            1,
        )
        _p32_shard(
            "qvq_p32_standard_large_m_grid_high_bits%d_object" % bits,
            bits,
            1,
            6,
            0,
            1,
        )
        _p32_shard("qvq_p32_grouped_bits%d_object" % bits, bits, 2, 7, 0, 1)
