"""Minimal profiling target for ncu: runs the planar Triton dequant kernel a few times.

Usage (under an allocator lease):
    ncu --set full -o planar_dequant python scripts/profile_planar_dequant.py [bits] [K] [N]
"""

import sys

import torch

from gptqmodel.nn_modules.triton_utils.planar import planar_dequant
from gptqmodel.utils.planar_packing import planar_pack_cols, planar_pack_rows


def main():
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
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

    # Warmup (compilation + autotune) then a few profiled iterations.
    for _ in range(3):
        planar_dequant(torch.float16, qweight, scales, qzeros, g_idx, bits)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"planar_dequant_b{bits}_{k}x{n}")
    for _ in range(3):
        planar_dequant(torch.float16, qweight, scales, qzeros, g_idx, bits)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"done bits={bits} K={k} N={n}")


if __name__ == "__main__":
    main()
