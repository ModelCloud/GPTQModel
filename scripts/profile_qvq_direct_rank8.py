"""Issue one direct Flash-Next gate/up rank-8 launch for Nsight Compute."""

from __future__ import annotations

import argparse

import torch

from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    repack_p32_planar_to_window,
)
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    pgc16_levels_for_version,
)
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
from gptqmodel.utils.qvq_ampere_cuda import (
    prewarm_qvq_ampere,
    qvq_p32_window_ampere,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--rank8", action="store_true")
    args = parser.parse_args()
    prewarm_qvq_ampere()
    device = torch.device("cuda", 0)
    bits = 3.0
    size_k, size_n = 2560, 640
    generator = torch.Generator(device=device).manual_seed(20261130 + args.m)
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    tile_count = (size_k // 16) * (size_n // 16)
    planar = torch.randint(
        0,
        1 << 32,
        (tile_count, qvq_words_per_tile(bits, weight_count=256, vector_size=2)),
        generator=generator,
        device=device,
        dtype=torch.int64,
    ).to(torch.int32)
    window = repack_p32_planar_to_window(planar, bits=bits)
    bank_ids = pack_qvq_binary_bank_ids(
        torch.randint(
            0,
            2,
            (tile_count * 8,),
            generator=generator,
            device=device,
            dtype=torch.uint8,
        )
    )
    input = (torch.randn((args.m, size_k), generator=generator, device=device) * 0.1).half()
    rank8_a = (torch.randn((size_k, 8), generator=generator, device=device) * 0.01).half()
    rank8_b = (torch.randn((8, size_n), generator=generator, device=device) * 0.01).half()
    for _ in range(5):
        qvq_p32_window_ampere(
            input, window, levels, bank_ids, bits, out_features=size_n,
            bank_alt_id=1, split_count=32,
            rank8_a=rank8_a if args.rank8 else None,
            rank8_b=rank8_b if args.rank8 else None, rank8_scale=0.75,
        )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(
        f"direct_rank8_M{args.m}" if args.rank8 else f"direct_base_M{args.m}"
    )
    qvq_p32_window_ampere(
        input, window, levels, bank_ids, bits, out_features=size_n,
        bank_alt_id=1, split_count=32,
        rank8_a=rank8_a if args.rank8 else None,
        rank8_b=rank8_b if args.rank8 else None, rank8_scale=0.75,
    )
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
