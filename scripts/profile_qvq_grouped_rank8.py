"""Issue a small, isolated grouped rank-8 launch for Nsight Compute."""

from __future__ import annotations

import torch

from gptqmodel.utils.qvq_ampere_cuda import (
    prewarm_qvq_ampere_grouped,
    qvq_p32_window_ampere_grouped_packed,
)
from scripts.bench_qvq_grouped_rank8 import build_case


def main() -> None:
    prewarm_qvq_ampere_grouped()
    input, payload, levels, packed_a, packed_b, rank8_as, rank8_bs, _ = build_case(1)
    for _ in range(5):
        qvq_p32_window_ampere_grouped_packed(
            input,
            payload,
            levels,
            rank8_as=rank8_as,
            rank8_bs=rank8_bs,
            rank8_packed_a=packed_a,
            rank8_packed_b=packed_b,
        )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("qkv_grouped_rank8_M1")
    qvq_p32_window_ampere_grouped_packed(
        input,
        payload,
        levels,
        rank8_as=rank8_as,
        rank8_bs=rank8_bs,
        rank8_packed_a=packed_a,
        rank8_packed_b=packed_b,
    )
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
