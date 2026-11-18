# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""K-quant ``GGUFTorchLinear`` layers must be CUDA-graph capturable (#5014).

``GGUFTorchLinear`` declares ``SUPPORTS_GRAPH_CAPTURE = True`` and the compile
matrix lists it as capture-safe, but the Q4_K/Q5_K/Q6_K (and Q2_K/Q3_K) block
dequantizers built their right-shift table with
``torch.tensor(values, dtype=torch.uint8, device=blocks.device)`` on every
forward. A pageable host-to-device copy is illegal during capture, so an
explicit ``torch.cuda.CUDAGraph`` capture of every K-quant layer failed with

    RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
    capture unless the CPU tensor is pinned.

which covers the default ``GGUFConfig(bits=4, format="q_k_m")`` output. Outside
capture the same code issued one to three small copies per layer per forward.

The tables are now immutable device constants cached per module, so a warm
layer captures with a stable allocation and issues no host copy at all. The
host-dequantized formats (Q2_K/Q3_K and the remaining codebook types) still
cannot be captured; the per-instance ``supports_graph_capture()`` says so, and
those nodes only assert the declaration -- a *failed* capture poisons the
caching allocator's graph pool for the rest of the process, so it is never
attempted here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear._compile_support import graph_capture_safe
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


pytestmark = [pytest.mark.gpu]

IN_FEATURES = 512
OUT_FEATURES = 256
CAPTURABLE_BITS = ("q4_0", "q8_0", "q4_k", "q5_k", "q6_k")
HOST_DEQUANT_BITS = ("q2_k", "q3_k")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GGUF graph capture needs a CUDA device")


def _build(bits: str) -> GGUFTorchLinear:
    torch.manual_seed(7)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.02)
    module = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
    )
    GGUFTorchLinear.pack_original(module, linear, scales=None, zeros=None)
    return module.cuda().eval()


def _activations(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        rows, IN_FEATURES, device="cuda", dtype=torch.float16, generator=generator
    )


@pytest.mark.parametrize("bits", CAPTURABLE_BITS)
@pytest.mark.parametrize("rows", [1, 64])
def test_layer_captures_and_replays_bitwise(bits, rows):
    """Capture, then replay both the same and a new input, bitwise vs eager."""

    _require_cuda()
    module = _build(bits)
    x = _activations(rows, seed=11)
    x_other = _activations(rows, seed=12)

    with torch.inference_mode():
        eager_same = module(x).clone()
        eager_other = module(x_other).clone()
        # Warm up so every lazily built device constant exists before capture.
        module(x)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(x)
        torch.cuda.synchronize()

        # Capture records work instead of running it, so the captured buffer is
        # only meaningful after a replay.
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager_same), (
            f"{bits} rows={rows}: the captured replay does not reproduce the eager output"
        )

        x.copy_(x_other)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager_other), (
            f"{bits} rows={rows}: the captured replay does not follow a new input"
        )


@pytest.mark.parametrize("bits", ("q4_k", "q5_k", "q6_k"))
def test_warm_forward_issues_no_pageable_host_copy(bits, monkeypatch):
    """A warm forward must not build a CUDA tensor from host data (#5014)."""

    _require_cuda()
    module = _build(bits)
    x = _activations(8, seed=21)
    with torch.inference_mode():
        module(x)
        torch.cuda.synchronize()

        host_constructions = []
        real_tensor = torch.tensor

        def counting_tensor(*args, **kwargs):
            device = kwargs.get("device")
            if device is not None and torch.device(device).type == "cuda":
                host_constructions.append((args[0] if args else None, device))
            return real_tensor(*args, **kwargs)

        monkeypatch.setattr(torch, "tensor", counting_tensor)
        for _ in range(4):
            module(x)
        torch.cuda.synchronize()

    assert host_constructions == [], (
        f"{bits}: a warm forward still builds {len(host_constructions)} CUDA "
        f"tensor(s) from host data: {host_constructions[:3]}"
    )


@pytest.mark.parametrize("bits", ("q4_0", "q8_0", "q4_k", "q5_k", "q6_k"))
def test_shift_tables_are_reused_device_constants(bits):
    """The dequantizer shift tables are cached per device, not rebuilt."""

    _require_cuda()
    module = _build(bits)
    device = module.qweight.device

    first = module._u8_shift((0, 4), device)
    assert first.device.type == "cuda"
    assert first.dtype == torch.uint8
    assert module._u8_shift((0, 4), device) is first, (
        f"{bits}: the shift table is rebuilt on every call"
    )
    # Distinct tuples get distinct entries; a device move invalidates them.
    assert module._u8_shift((0, 2, 4, 6), device) is not first
    module.clear_weight_cache()
    assert module._u8_shift((0, 4), device) is not first


def test_graph_capture_declaration_is_per_format():
    """The class declares the device formats; the helper answers per instance."""

    _require_cuda()
    assert GGUFTorchLinear.SUPPORTS_GRAPH_CAPTURE is True
    # A class is answered by its declaration: the protocol test needs a bool.
    assert graph_capture_safe(GGUFTorchLinear) is True

    for bits in CAPTURABLE_BITS:
        module = _build(bits)
        assert module.supports_graph_capture() is True, bits
        assert graph_capture_safe(module) is True, bits

    for bits in HOST_DEQUANT_BITS:
        module = _build(bits)
        assert module.supports_graph_capture() is False, (
            f"{bits} dequantizes on the host, so it cannot be graph captured"
        )
        assert graph_capture_safe(module) is False, bits
