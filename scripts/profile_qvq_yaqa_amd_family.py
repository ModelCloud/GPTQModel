#!/usr/bin/env python3
"""Execute exact family-batched YAQA kernels for rocprof/ISA capture."""

import argparse
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=float, nargs="+", default=[2, 2.5, 3, 3.5])
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--startup-delay", type=float, default=0)
    args = parser.parse_args()
    if args.families < 1 or args.count < 1 or args.families * args.count > 256:
        parser.error("Require 1-256 total family sequences")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch

    from gptqmodel.utils.qvq_yaqa_amd import (
        family_banked_viterbi_midpoint_trusted,
        family_banked_viterbi_trusted,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260914)
    sequences = torch.randn((args.families, args.count, 128, 2), device="cuda", generator=generator)
    codebooks = torch.randn((args.families, 2, 65536, 2), device="cuda", generator=generator)
    if args.startup_delay:
        time.sleep(args.startup_delay)
    for bits in args.bits:
        rotated = torch.roll(sequences, 64, 2).contiguous()
        provisional = family_banked_viterbi_trusted(rotated, codebooks, bits=bits)
        overlap = family_banked_viterbi_midpoint_trusted(
            rotated, codebooks, bits=bits
        )
        suffix_mask = (1 << (16 - int(2 * bits))) - 1
        if not torch.equal(overlap, provisional.states[:, :, 63] & suffix_mask):
            raise AssertionError(f"W{bits:g} midpoint mismatch")
        family_banked_viterbi_trusted(sequences, codebooks, bits=bits, overlap=overlap)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
