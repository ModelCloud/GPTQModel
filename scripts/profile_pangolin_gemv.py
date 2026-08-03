"""Minimal profiling target for ncu: runs the native Pangolin GEMV a few times.

Usage (under an allocator lease):
    ncu --section SpeedOfLight -o pangolin python scripts/profile_pangolin_gemv.py [bits] [K] [N] [M]
"""

import sys

import torch

from gptqmodel.utils.pangolin import pangolin_gemv
from gptqmodel.utils.planar_packing import planar_pack_cols, planar_pack_rows


def main():
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    m = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    group_size = 128
    groups = k // group_size

    torch.manual_seed(0)
    dev = torch.device("cuda")
    codes = torch.randint(0, 2**bits, (k, n), dtype=torch.int32)
    zeros = torch.randint(0, 2**bits, (groups, n), dtype=torch.int32)
    qweight = planar_pack_rows(codes, bits).to(dev)
    qzeros = planar_pack_cols(zeros, bits).to(dev)
    scales = (torch.rand(groups, n, dtype=torch.float16) * 0.01 + 0.005).to(dev)
    g_idx = (torch.arange(k, dtype=torch.int32) // group_size).to(dev)
    x = (torch.randn(m, k, dtype=torch.float16) * 0.5).to(dev)

    for _ in range(5):
        pangolin_gemv(x, qweight, scales, qzeros, g_idx, bits)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"pangolin_gemv_b{bits}_{k}x{n}_m{m}")
    for _ in range(3):
        pangolin_gemv(x, qweight, scales, qzeros, g_idx, bits)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"done bits={bits} K={k} N={n} M={m}")


if __name__ == "__main__":
    main()
