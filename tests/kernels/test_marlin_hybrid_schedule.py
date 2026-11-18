# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Marlin hybrid schedule (vllm-project/vllm#24722).

With more output tiles than threadblocks, each block first takes whole tiles
(all of K, no cross-block reduction: part 1), then the remaining tiles are
striped along K (part 2). The first part-2 slice of a block usually starts in
the middle of K, so every global read index (A, B, scales, zero points, bias)
is rebuilt from ``(slice_row, slice_col)`` at the transition. These shapes hit
that transition on any GPU with up to ``M/64 * N/256`` SMs, with g128 groups,
zero points, bias, and both the FP32 and FP16 global reductions; each launch is
checked against an FP64 oracle and for run-to-run determinism.
"""

from __future__ import annotations

import pytest
import torch

from .test_marlin_mxfp8 import _OPERANDS as _MX_OPERANDS
from .test_marlin_mxfp8 import _gemm as _mx_gemm
from .test_marlin_tile_determinism_4735 import _build_awq, _build_gptq, _skip_reason


def _part1_rounds(sms: int, size_m: int, size_n: int) -> int:
    # {64, 256, 256} with 64-row M tiles for these M > 16 shapes.
    tiles = (size_m // 64) * (size_n // 256)
    if tiles <= sms:
        return 0
    part2 = tiles % sms
    if part2 * 3 <= sms:
        part2 += sms
    return (tiles - part2) // sms


# (M, K, N): 2048 x 4096 is Llama-3-8B o_proj prefill; 1000 adds an M tail.
_SHAPES = ((2048, 4096, 4096), (1024, 2048, 8192), (1000, 4096, 4096))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp32_reduce", [True, False], ids=["fp32-reduce", "fp16-reduce"])
@pytest.mark.parametrize("kind", ["gptq", "awq"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("size_m,size_k,size_n", _SHAPES)
def test_part1_to_part2_transition_matches_oracle(size_m, size_k, size_n, dtype, kind, fp32_reduce):
    reason = _skip_reason(dtype)
    if reason is not None:
        pytest.skip(reason)
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if _part1_rounds(sms, size_m, size_n) == 0:
        pytest.skip(f"{sms} SMs: no part-1 round for this shape")

    module, weight = (_build_gptq if kind == "gptq" else _build_awq)(size_k, size_n, dtype)
    module.fp32 = fp32_reduce
    torch.manual_seed(24722)
    x = torch.randn(size_m, size_k, device="cuda", dtype=dtype)
    bias = torch.randn(size_n, device="cuda", dtype=torch.float32)
    module.bias = bias.to(dtype)
    module._marlin_bias_padding = (size_n, size_n)

    with torch.inference_mode():
        out = module(x)
        again = module(x)
    torch.cuda.synchronize()

    assert out.isfinite().all()
    assert torch.equal(out, again), "repeated launches differ"
    oracle = x.double() @ weight.double() + module.bias.double()
    error = (out.double() - oracle).abs().max().item()
    # The #4735 determinism bounds for these builders (output magnitudes ~10-60).
    bound = {(torch.float16, True): 0.05, (torch.float16, False): 0.1}.get((dtype, fp32_reduce), 0.5)
    assert error < bound, error


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4"])
@pytest.mark.parametrize("size_m,size_k,size_n", _SHAPES)
def test_part2_restart_rebuilds_8bit_scale_rows(size_m, size_k, size_n, fmt):
    """The part-2 restart must rebuild the paired 8-bit scale row like the initial ``s_gl_rd``.

    8-bit scales (MXFP8 and MXFP4) pack two groups per 16-bit scale row, so the
    row offset is divided by 2 for them. Keying that divisor on the FP4 weight
    type instead of the scale width sent MXFP8 restarts to twice the right row.
    """

    reason = _skip_reason(torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if _part1_rounds(sms, size_m, size_n) == 0:
        pytest.skip(f"{sms} SMs: no part-1 round for this shape")

    device = torch.device("cuda")
    qweight, scales, b_q_type, dequant = _MX_OPERANDS[fmt](size_n, size_k, device, seed=size_k + size_n)
    torch.manual_seed(size_m)
    x = (torch.randn(size_m, size_k) * 0.25).to(torch.bfloat16)
    oracle = x.double() @ dequant.T
    out = _mx_gemm(x.to(device), qweight, scales, b_q_type, size_n, size_k).double().cpu()

    err = (out - oracle).abs()
    tol = 2.0 ** -7 * oracle.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    assert not (err > tol).any(), (
        f"{fmt} M={size_m} K={size_k} N={size_n}: max err {err.max().item():.4g} "
        f"vs |oracle| max {oracle.abs().max().item():.4g}"
    )
