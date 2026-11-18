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

#5281: the tile-count rule ignored K and the 256-thread tile it replaced. On
128-SM Ada parts N=4096 sits exactly on the boundary, so large-K down
projections took the narrow tile although the wide tile already had enough
split-K work per SM (1.1-1.35x slower). The selection contract below pins the
wide tile for those shapes and keeps the narrow tile where it still splits K.
"""

from __future__ import annotations

import re

import pytest
import torch

from gptqmodel.utils.marlin import gptq_marlin_gemm, marlin_stream_workspace

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

# Large-K projections where the tile-count rule alone chose the narrow tile on
# 128-SM parts although the wide tile already filled the SMs with split-K work.
_LARGE_K_WIDE_SHAPES = (
    (4, 14336, 4096),
    (16, 14336, 4096),
    (128, 14336, 4096),
    (16, 11008, 4096),
    (128, 11008, 4096),
    (128, 8192, 4096),
    (32, 28672, 8192),
    (64, 28672, 8192),
)

# Large-K shapes where the narrow tile still splits K over enough threadblocks
# (or K is short), so the override stays.
_LARGE_K_NARROW_SHAPES = (
    (32, 14336, 4096),
    (64, 14336, 4096),
    (128, 14336, 2048),
    (16, 8192, 2048),
    (16, 4096, 4096),
)


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


_NARROW_TILE = (128, 64, 128)


def _wide_tile(size_m: int) -> tuple[int, int, int]:
    """The auto table's tile before the small-grid override."""

    # ``{128, 128, 256}`` for M <= 16, else ``{64, 256, 256}`` with 64-row M
    # tiles. Both are returned as (threads, thread_n, thread_k).
    return (256, 128, 128) if size_m <= 16 else (256, 256, 64)


def _gemm_with_tile(module, x, tile):
    """Launch the module's operands with an explicit ``(threads, n, k)`` tile."""

    threads, thread_n, thread_k = tile
    return gptq_marlin_gemm(
        x,
        None,
        module.qweight,
        None,
        module.scales,
        None,
        module.qzeros,
        module.g_idx,
        module.g_idx_sort_indices,
        marlin_stream_workspace(module),
        module.weight_type,
        size_m=x.shape[0],
        size_n=module.out_features,
        size_k=module.in_features,
        is_k_full=module.is_k_full,
        use_fp32_reduce=module.fp32,
        thread_k=thread_k,
        thread_n=thread_n,
        num_threads=threads,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size_m,size_k,size_n", _LARGE_K_WIDE_SHAPES)
def test_large_k_keeps_wide_tiles_on_128_sm_parts(size_m, size_k, size_n):
    """#5281: the wide tile already splits K over the SMs for large K."""

    reason = _skip_reason(torch.float16)
    if reason is not None:
        pytest.skip(reason)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if _full_sm80(props):
        pytest.skip("full-SM sm_80 boards keep their own tuned tile tables")
    if props.multi_processor_count > 128:
        pytest.skip(">128-SM parts keep #5216's measured narrow tile")

    module, _ = _build_gptq(size_k, size_n, torch.float16)
    torch.manual_seed(5281)
    x = torch.randn(size_m, size_k, device="cuda", dtype=torch.float16)
    assert _launched_tiles(module, x) == {_wide_tile(size_m)}


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size_m,size_k,size_n", _LARGE_K_NARROW_SHAPES)
def test_large_k_keeps_narrow_when_it_still_splits_k(size_m, size_k, size_n):
    """Large K alone does not veto the narrow tile: it must still split K."""

    reason = _skip_reason(torch.float16)
    if reason is not None:
        pytest.skip(reason)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if _full_sm80(props):
        pytest.skip("full-SM sm_80 boards keep their own tuned tile tables")

    module, _ = _build_gptq(size_k, size_n, torch.float16)
    torch.manual_seed(5281)
    x = torch.randn(size_m, size_k, device="cuda", dtype=torch.float16)
    assert _launched_tiles(module, x) == {_NARROW_TILE}


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "size_m,size_k,size_n",
    ((16, 14336, 4096), (128, 14336, 4096), (64, 4096, 1024)),
)
def test_tile_choice_only_selects_and_both_tiles_agree(size_m, size_k, size_n, dtype):
    """The rule must not change arithmetic; the two tiles differ only in the
    FP32 reduction tree (K is split across warps and threadblocks)."""

    reason = _skip_reason(dtype)
    if reason is not None:
        pytest.skip(reason)
    module, weight = _build_gptq(size_k, size_n, dtype)
    torch.manual_seed(5281)
    x = torch.randn(size_m, size_k, device="cuda", dtype=dtype)

    wide, narrow = _wide_tile(size_m), _NARROW_TILE
    with torch.inference_mode():
        out_wide = _gemm_with_tile(module, x, wide)
        out_narrow = _gemm_with_tile(module, x, narrow)
        out_wide_again = _gemm_with_tile(module, x, wide)
        out_narrow_again = _gemm_with_tile(module, x, narrow)
        auto = module(x)
    torch.cuda.synchronize()

    assert torch.equal(out_wide, out_wide_again), "wide tile is not deterministic"
    assert torch.equal(out_narrow, out_narrow_again), "narrow tile is not deterministic"

    oracle = x.double() @ weight.double()
    bound = 0.05 if dtype == torch.float16 else 0.5
    for name, out in (("wide", out_wide), ("narrow", out_narrow)):
        error = (out.double() - oracle).abs().max().item()
        assert error < bound, (name, error)

    rounded = oracle.to(dtype)
    assert torch.isfinite(rounded).all()
    ulp = (
        torch.nextafter(rounded, torch.full_like(rounded, float("inf"))) - rounded
    ).abs().max().item()
    diff = (out_wide.float() - out_narrow.float()).abs().max().item()
    assert diff <= ulp, (diff, ulp)

    launched = _launched_tiles(module, x)
    assert len(launched) == 1
    selected = _gemm_with_tile(module, x, next(iter(launched)))
    assert torch.equal(auto, selected), "auto selection changed the tile's numerics"
