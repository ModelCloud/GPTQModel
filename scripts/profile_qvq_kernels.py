# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Isolated NVTX-marked harness for profiling every QVQ CUDA kernel with ncu/nsys."""

from __future__ import annotations

import sys

import torch


def main() -> None:
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.utils.qvq_cuda import (
        _qvq_cuda_yaqa_feedback_op,
        prewarm_qvq_cuda,
        qvq_cuda_gemv,
        qvq_cuda_hadamard,
        qvq_cuda_viterbi,
    )

    assert prewarm_qvq_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda")
    torch_profiler_start = torch.cuda.cudart().cudaProfilerStart
    torch_profiler_stop = torch.cuda.cudart().cudaProfilerStop

    def repeat(label: str, fn, warmup: int = 8, iters: int = 16) -> None:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch_profiler_start()
        torch.cuda.nvtx.range_push(label)
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch_profiler_stop()

    size_k, size_n = 4096, 4096

    # --- inference: gemv M=1 (splitk) and M=32 (rows=32), W2 and W4 ---
    for bits in (2.0, 4.0):
        rate_key = round(bits * 2)
        generator = torch.Generator(device="cpu").manual_seed(12000 + rate_key + size_k + size_n)
        tile_count = (size_k // 16) * (size_n // 16)
        words_per_tile = qvq_words_per_tile(bits, vector_size=2)
        trellis = torch.randint(
            -(2**31), 2**31 - 1, (tile_count, words_per_tile), generator=generator, dtype=torch.int32
        ).to(device)
        for m in (1, 32):
            x = torch.randn((m, size_k), generator=generator, dtype=torch.float32).half().to(device)

            def run_gemv(x=x, trellis=trellis, bits=bits, m=m):
                return qvq_cuda_gemv(x, trellis, bits, out_features=size_n, output_fp32=True)

            repeat(f"gemv_w{int(bits*2)}bit_m{m}", run_gemv)
        del trellis

    # --- inference: fused hadamard ---
    for n in (4096,):
        x = torch.randn((32, n), dtype=torch.float32).half().to(device)

        def run_had(x=x):
            return qvq_cuda_hadamard(x)

        repeat(f"hadamard_n{n}", run_had)

    # --- quantization process: viterbi + yaqa ---
    for bits in (2.0, 4.0):
        sequences = torch.randn((64, 256, 2), dtype=torch.float32, device=device)
        codebook = torch.randn((1 << 16, 2), dtype=torch.float32, device=device)

        def run_viterbi(sequences=sequences, codebook=codebook, bits=bits):
            return qvq_cuda_viterbi(sequences, codebook, bits)

        repeat(f"viterbi_w{int(bits*2)}bit_b64", run_viterbi)

    rows, cols = 4096, 4096
    source = torch.randn((rows, cols), dtype=torch.float32, device=device)
    left = torch.randn((rows * cols,), dtype=torch.float32, device=device).view(rows, cols)
    right = torch.randn((rows * cols,), dtype=torch.float32, device=device).view(rows, cols)
    feedback = torch.zeros((cols, cols), dtype=torch.float32, device=device)
    bias = None
    yaqa_feedback = _qvq_cuda_yaqa_feedback_op()

    def run_yaqa():
        return yaqa_feedback(source, left, right, feedback, 0, 7, 8, bias)

    repeat("yaqa_feedback", run_yaqa)

    print("harness done", file=sys.stderr)


if __name__ == "__main__":
    main()
