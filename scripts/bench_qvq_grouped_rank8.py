"""Benchmark the decode-sized grouped P32 QKV rank-8 path on SM80."""

from __future__ import annotations

import argparse
from statistics import median

import torch

from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    pgc16_levels_for_version,
)
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
from gptqmodel.utils.qvq_ampere_cuda import (
    prewarm_qvq_ampere_grouped,
    qvq_p32_window_ampere_group_plan,
    qvq_p32_window_ampere_grouped_packed,
    qvq_pack_p32_window_ampere_group,
)


def build_case(size_m: int):
    device = torch.device("cuda", 0)
    bits = 3.0
    size_k = 5120
    widths = (12288, 1024, 1024)
    alt_ids = (3, 1, 1)
    split_counts = (40, 56, 56)
    generator = torch.Generator(device=device).manual_seed(20260907 + size_m)
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    input = torch.randn((size_m, size_k), generator=generator, device=device).half()
    trellises = []
    bank_ids = []
    rank8_as = []
    rank8_bs = []
    for width in widths:
        tile_count = (size_k // 16) * (width // 16)
        words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
        trellises.append(
            torch.randint(
                -(1 << 31),
                1 << 31,
                (tile_count, words),
                generator=generator,
                device=device,
                dtype=torch.int32,
            )
        )
        bank_ids.append(
            pack_qvq_binary_bank_ids(
                torch.randint(
                    0,
                    2,
                    (tile_count * 8,),
                    generator=generator,
                    device=device,
                    dtype=torch.uint8,
                )
            )
        )
        rank8_as.append(
            (torch.randn((size_k, 8), generator=generator, device=device) * 0.01).half()
        )
        rank8_bs.append(
            (torch.randn((8, width), generator=generator, device=device) * 0.01).half()
        )
    plan = qvq_p32_window_ampere_group_plan(
        input,
        trellises,
        levels,
        bank_ids,
        bits,
        out_features=widths,
        bank_alt_ids=alt_ids,
        split_counts=split_counts,
    )
    payload = qvq_pack_p32_window_ampere_group(trellises, bank_ids, plan)
    packed_a = torch.cat(tuple(rank8_as), dim=1).contiguous()
    packed_b = torch.cat(tuple(rank8_bs), dim=1).contiguous()
    return input, payload, levels, packed_a, packed_b, rank8_as, rank8_bs, widths


def timed(
    input,
    payload,
    levels,
    packed_a,
    packed_b,
    rank8_as,
    rank8_bs,
    widths,
    rank8: bool,
):
    def replay():
        if rank8:
            qvq_p32_window_ampere_grouped_packed(
                input,
                payload,
                levels,
                rank8_as=rank8_as,
                rank8_bs=rank8_bs,
                rank8_packed_a=packed_a,
                rank8_packed_b=packed_b,
            )
        else:
            qvq_p32_window_ampere_grouped_packed(input, payload, levels)

    for _ in range(10):
        replay()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends, strict=True)]
    return median(samples), min(samples), max(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    prewarm_qvq_ampere_grouped()
    for size_m in args.m:
        case = build_case(size_m)
        input, payload, levels, packed_a, packed_b, rank8_as, rank8_bs, widths = case
        off = timed(
            input, payload, levels, packed_a, packed_b, rank8_as, rank8_bs, widths, False
        )
        on = timed(
            input, payload, levels, packed_a, packed_b, rank8_as, rank8_bs, widths, True
        )
        delta = (on[0] / off[0] - 1.0) * 100.0
        print(
            f"M={size_m:2d} off={off[0]:8.2f}us "
            f"rank8={on[0]:8.2f}us delta={delta:+6.1f}% "
            f"ranges=({on[1]:.2f},{on[2]:.2f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
