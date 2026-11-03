# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The AllSpark W8A16 GEMM has to stay usable inside a captured decode graph.

A capture fails on anything that round-trips to the host: a ``.item()``, a
``.cpu()``, an ``synchronize()``, or a hidden ``cudaMalloc`` through a path the
capture allocator pool does not cover. It does *not* fail on a plain
``torch::empty`` - the pool absorbs that - so these tests guard the host-sync
and shape-dispatch behaviour, not allocation.

The shape dispatch matters because ``allspark_w8a16_gemm`` picks the MMA kernel
or the cuBLAS fallback from ``M`` at call time, and the fallback allocates its
dequantized weights. That decision has to stay in the captured region.
"""

from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AllSpark needs a CUDA device"
)


def _quantize_per_column(weight: torch.Tensor):
    """Symmetric int8 per output column (group_size = -1) in GPTQ uint8b128 form."""
    scale = (weight.abs().amax(dim=0) / 127).clamp_min(1e-8)
    qweight = (weight / scale).round().add(128).clamp(0, 255).to(torch.uint8)
    return qweight.contiguous(), scale.to(weight.dtype).contiguous()


def _prepare(dtype, m, k=256, n=512):
    from gptqmodel.utils.allspark import (
        allspark_repack,
        allspark_supported,
        prewarm_allspark_extension,
    )

    if not allspark_supported():
        pytest.skip("AllSpark requires compute capability >= 8.0")
    if not prewarm_allspark_extension():
        pytest.fail("AllSpark JIT extension failed to build on this device")

    generator = torch.Generator(device="cuda").manual_seed(3515 + m)
    weight = torch.randn(k, n, dtype=dtype, device="cuda", generator=generator) * 0.02
    qweight, scales = _quantize_per_column(weight)
    packed, packed_scales, _ = allspark_repack(qweight, scales, n, k)
    x = torch.randn(m, k, dtype=dtype, device="cuda", generator=generator) * 0.02
    dense = ((qweight.float() - 128) * scales.float()).to(dtype)
    return x, packed, packed_scales, dense


def _capture(fn, warmup=3):
    """Warm up outside capture so lazy handles (cuBLAS, allocator) are built first."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def _assert_matches_reference(output, x, dense):
    # Accumulate independently in FP64 over the dtype-rounded W8A16 weights.
    # Keep the returned dtype check: casting output to float would hide a
    # regression in the kernel's output contract.
    reference = x.double() @ dense.double()
    assert output.dtype == x.dtype
    torch.testing.assert_close(output.double(), reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(output, reference.to(x.dtype), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 16])
def test_allspark_gemm_captures_and_replays(m, dtype):
    from gptqmodel.utils.allspark import allspark_gemm

    x, packed, packed_scales, dense = _prepare(dtype, m, n=512)
    n = packed.shape[0]

    graph, output = _capture(lambda: allspark_gemm(x, packed, packed_scales, n))
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(output, x, dense)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_allspark_graph_replay_reads_current_input(dtype):
    """The captured kernel must read whatever is in the input buffer at replay time."""
    from gptqmodel.utils.allspark import allspark_gemm

    x, packed, packed_scales, dense = _prepare(dtype, m=8, n=512)
    n = packed.shape[0]

    graph, output = _capture(lambda: allspark_gemm(x, packed, packed_scales, n))
    for seed in range(4):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x.copy_(torch.randn(x.shape, dtype=dtype, device="cuda", generator=generator) * 0.02)
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches_reference(output, x, dense)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("corruption", ["zero", "stale"])
def test_replay_check_rejects_incorrect_output(monkeypatch, dtype, corruption):
    """Pin sensitivity to zero output and a graph that stops reading its input."""
    capture = _capture

    def broken_capture(fn, warmup=3):
        graph, output = capture(fn, warmup)
        graph.replay()
        torch.cuda.synchronize()
        frozen = output.clone() if corruption == "stale" else torch.zeros_like(output)
        return SimpleNamespace(replay=lambda: None), frozen

    monkeypatch.setitem(globals(), "_capture", broken_capture)
    with pytest.raises(AssertionError):
        test_allspark_graph_replay_reads_current_input(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_allspark_cublas_fallback_captures(dtype):
    """M above the threshold takes the fallback path, which builds dense scratch per call."""
    from gptqmodel.utils.allspark import ALLSPARK_M_CUBLAS_THRESHOLD, allspark_gemm

    x, packed, packed_scales, dense = _prepare(dtype, m=ALLSPARK_M_CUBLAS_THRESHOLD + 8, n=512)
    n = packed.shape[0]

    graph, output = _capture(lambda: allspark_gemm(x, packed, packed_scales, n))
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(output, x, dense)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_allspark_module_forward_captures(dtype):
    """AllSparkLinear.forward must not sync either - it is what a real model calls."""
    from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear
    from gptqmodel.utils.allspark import allspark_supported, prewarm_allspark_extension

    if not allspark_supported():
        pytest.skip("AllSpark requires compute capability >= 8.0")
    if not prewarm_allspark_extension():
        pytest.fail("AllSpark JIT extension failed to build on this device")

    k, n = 256, 512
    weight = torch.randn(k, n, dtype=dtype, device="cuda") * 0.02
    qweight, scales = _quantize_per_column(weight)
    shifts = (torch.arange(4, device="cuda", dtype=torch.int32) * 8).view(1, 4, 1)
    codes = (weight / scales).round().add(128).clamp(0, 255).to(torch.int32)
    packed_gptq = (codes.view(k // 4, 4, n) << shifts).sum(dim=1).to(torch.int32)

    module = AllSparkLinear(
        bits=8, group_size=-1, desc_act=False, sym=True,
        in_features=k, out_features=n, bias=False, dtype=dtype,
    ).cuda().to(dtype)
    module.qweight.data.copy_(packed_gptq)
    module.scales.data.copy_(scales.view(1, n))
    module.post_init()

    x = torch.randn(2, 4, k, dtype=dtype, device="cuda") * 0.02
    dense = ((codes.float() - 128) * scales.float()).to(dtype)

    graph, output = _capture(lambda: module(x))
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(output, x, dense)
    assert output.shape == (2, 4, n)
