# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Marlin small-grid tile override (vllm-project/vllm#24722).

When the auto-selected thread tile leaves fewer than four output tiles per SM,
``marlin_mm`` switches to the narrow ``{thread_k=128, thread_n=64, threads=128}``
tile. The old planner kept the wide tile, so most SMs split K for a handful of
output tiles and then queued on the reduction locks (for example Llama-3-8B
``k_proj``/``v_proj`` at M=64 took 72.6 us on an RTX 5090, 16.9 us after).

The launched kernel's template arguments are read back from the profiler, so
the test pins the planner decision itself, and every launch is checked against
an independent FP64 oracle.
"""

from __future__ import annotations

import re

import pytest
import torch

from .test_marlin_tile_determinism_4735 import _build_awq, _build_gptq, _skip_reason


# Marlin<scalar_t, w_type, s_type, threads, thread_m_blocks, thread_n_blocks,
#        thread_k_blocks, m_block_size_8, stages, group_blocks, is_zp_float>
_MARLIN_ARGS = re.compile(r"Marlin<([^<>]*)>")


def _launched_tiles(module, x) -> set[tuple[int, int, int]]:
    """Return ``(threads, thread_n, thread_k)`` of every Marlin GEMM launch."""

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            module(x)
        torch.cuda.synchronize()
    tiles = set()
    for event in prof.events():
        match = _MARLIN_ARGS.search(event.name)
        if match is None:
            continue
        args = [part.strip() for part in match.group(1).split(",")]
        threads, n_blocks, k_blocks = int(args[3]), int(args[5]), int(args[6])
        tiles.add((threads, 16 * n_blocks, 16 * k_blocks))
    assert tiles, "no Marlin GEMM launch was recorded"
    return tiles


def _small_grid_expected(sms: int, size_m: int, size_n: int) -> bool:
    # Auto config before the override: {128, 128, 256} for M <= 16, else
    # {64, 256, 256} with 64-row M tiles (these shapes keep thread_m_blocks).
    if size_m <= 16:
        output_tiles = size_n // 128
    else:
        output_tiles = -(-size_m // 64) * (size_n // 256)
    return output_tiles * 4 <= sms


def _full_sm80(props) -> bool:
    return (props.major, props.minor) == (8, 0) and props.multi_processor_count >= 124


# (M, K, N): Llama-3-8B k/v projection and a Qwen2.5 kv projection.
_SHAPES = ((1, 4096, 1024), (16, 4096, 1024), (64, 4096, 1024), (64, 3584, 512))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("size_m,size_k,size_n", _SHAPES)
def test_small_grid_uses_narrow_tiles_and_matches_oracle(size_m, size_k, size_n, dtype):
    reason = _skip_reason(dtype)
    if reason is not None:
        pytest.skip(reason)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if _full_sm80(props):
        pytest.skip("full-SM sm_80 boards keep their own tuned tile tables")
    if not _small_grid_expected(props.multi_processor_count, size_m, size_n):
        pytest.skip(f"{props.multi_processor_count} SMs: shape is not a small grid")

    # GPTQ: the sm_89 narrow FP32 AWQ schedule never applies to it.
    module, weight = _build_gptq(size_k, size_n, dtype)
    torch.manual_seed(24722)
    x = torch.randn(size_m, size_k, device="cuda", dtype=dtype)

    assert _launched_tiles(module, x) == {(128, 64, 128)}

    with torch.inference_mode():
        out = module(x)
    error = (out.double() - x.double() @ weight.double()).abs().max().item()
    # Same bounds as the #4735 determinism tests for these GPTQ builders.
    assert error < (0.05 if dtype == torch.float16 else 0.5), error


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_awq_small_grid_matches_oracle(dtype):
    reason = _skip_reason(dtype)
    if reason is not None:
        pytest.skip(reason)
    module, weight = _build_awq(4096, 1024, dtype)
    torch.manual_seed(24722)
    for size_m in (1, 7, 16, 64, 128):
        x = torch.randn(size_m, 4096, device="cuda", dtype=dtype)
        with torch.inference_mode():
            out = module(x)
        error = (out.double() - x.double() @ weight.double()).abs().max().item()
        assert error < (0.05 if dtype == torch.float16 else 0.5), (size_m, error)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_grid_keeps_wide_tiles():
    reason = _skip_reason(torch.float16)
    if reason is not None:
        pytest.skip(reason)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    size_m, size_k, size_n = 512, 4096, 4096
    if _small_grid_expected(props.multi_processor_count, size_m, size_n):
        pytest.skip(f"{props.multi_processor_count} SMs: shape is a small grid")
    module, _ = _build_gptq(size_k, size_n, torch.float16)
    x = torch.randn(size_m, size_k, device="cuda", dtype=torch.float16)
    assert (128, 64, 128) not in _launched_tiles(module, x)
