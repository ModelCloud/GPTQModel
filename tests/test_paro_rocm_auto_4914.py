# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""AUTO selection and graph capture for ParoQuant on ROCm (#4914).

``ParoLinear`` declares ``SUPPORTS_FORMATS = {FORMAT.PARO: 55}`` and the native
AWQ GEMM it dispatches to is a CUDA-only extension. On ROCm that path can never
run, so every forward falls through to ``_forward_dense`` and re-dequantizes
the whole packed weight. ``ParoTritonLinear`` is the native ROCm path but used
to keep priority 0, which opts a kernel out of AUTO selection entirely.

The dense fallback also built its extraction offsets with
``torch.tensor([...], device=...)``, a host-to-device copy that an explicit
``torch.cuda.CUDAGraph`` capture rejects. The declaration
``SUPPORTS_GRAPH_CAPTURE = True`` was therefore false on every ROCm host.

These tests pin both behaviours: AUTO must pick the native Triton kernel on
ROCm (and keep ``ParoLinear`` first on CUDA), and a dense ``ParoLinear``
forward must survive a CUDA-graph capture with output matching eager.
"""

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import paro as paro_module
from gptqmodel.nn_modules.qlinear import paro_triton as paro_triton_module
from gptqmodel.nn_modules.qlinear._compile_support import graph_capture_safe
from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.nn_modules.qlinear.paro_triton import ParoTritonLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import (
    get_auto_backend_kernel_mapping,
    select_quant_linear,
)
from gptqmodel.utils.paro import build_identity_rotation_buffers


def _patch_rocm(monkeypatch, rocm: bool):
    # ``raising=False`` keeps the helper usable against a tree that predates the
    # per-device priority hook, so the assertions below fail instead of the
    # whole module failing to collect.
    monkeypatch.setattr(paro_module, "IS_ROCM", rocm, raising=False)
    monkeypatch.setattr(paro_triton_module, "IS_ROCM", rocm, raising=False)


def test_paro_triton_is_auto_eligible():
    # Priority 0 opts a kernel out of AUTO selection; a positive rank keeps the
    # Triton kernel reachable from the auto map.
    assert ParoTritonLinear.SUPPORTS_FORMATS[FORMAT.PARO] > 0
    auto_map = get_auto_backend_kernel_mapping()[METHOD.PARO][FORMAT.PARO]
    assert BACKEND.PARO_TRITON in auto_map


def test_rocm_auto_priority_prefers_triton(monkeypatch):
    _patch_rocm(monkeypatch, True)

    triton_rank = ParoTritonLinear.auto_select_priority(FORMAT.PARO, torch.float16)
    dense_rank = ParoLinear.auto_select_priority(FORMAT.PARO, torch.float16)

    assert triton_rank > dense_rank, (
        "on ROCm the native Triton kernel must outrank the per-forward dense "
        f"dequantize fallback: triton={triton_rank} dense={dense_rank}"
    )


def test_cuda_auto_priority_keeps_paro_linear(monkeypatch):
    _patch_rocm(monkeypatch, False)

    triton_rank = ParoTritonLinear.auto_select_priority(FORMAT.PARO, torch.float16)
    dense_rank = ParoLinear.auto_select_priority(FORMAT.PARO, torch.float16)

    assert dense_rank > triton_rank, (
        "on CUDA the measured ``PARO_CUDA`` kernel must stay first: "
        f"dense={dense_rank} triton={triton_rank}"
    )


def _select(device):
    return select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=device,
        backend=BACKEND.AUTO,
        format=FORMAT.PARO,
        quant_method=METHOD.PARO,
        dtype=torch.float16,
        pack_dtype=torch.int32,
    )


def test_rocm_auto_selects_triton_kernel(monkeypatch):
    _patch_rocm(monkeypatch, True)
    # Validation is environment-dependent (Triton import, GPU presence); this
    # test pins the ordering the selector applies after validation succeeds.
    monkeypatch.setattr(
        ParoLinear, "validate", classmethod(lambda cls, **kwargs: (True, None))
    )
    monkeypatch.setattr(
        ParoTritonLinear, "validate", classmethod(lambda cls, **kwargs: (True, None))
    )

    assert _select(DEVICE.ROCM) is ParoTritonLinear


def test_cuda_auto_selects_paro_linear(monkeypatch):
    _patch_rocm(monkeypatch, False)
    monkeypatch.setattr(
        ParoLinear, "validate", classmethod(lambda cls, **kwargs: (True, None))
    )
    monkeypatch.setattr(
        ParoTritonLinear, "validate", classmethod(lambda cls, **kwargs: (True, None))
    )

    assert _select(DEVICE.CUDA) is ParoLinear


def test_awq_shifts_match_the_host_constant():
    # The cached device constant must be exactly the old host-built tensor.
    from gptqmodel.quantization.awq.utils.packing_utils import (
        AWQ_REVERSE_ORDER,
        _awq_shifts,
    )

    expected = torch.tensor(
        [lane * 4 for lane in AWQ_REVERSE_ORDER], dtype=torch.int64
    )
    torch.testing.assert_close(_awq_shifts(torch.device("cpu"), 4), expected)


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor), dtype=torch.int32
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


def _dense_paro_module(device: torch.device, dtype: torch.dtype) -> ParoLinear:
    bits = 4
    in_features = 256
    out_features = 128
    group_size = 128
    groups = in_features // group_size

    torch.manual_seed(0)
    int_weight = torch.randint(0, 2**bits, (in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, (groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 2.0) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)

    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    )
    module.qweight.copy_(_pack_awq_tensor(int_weight, bits))
    module.qzeros.copy_(_pack_awq_tensor(zero_points, bits))
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        dtype=torch.float16,
    )
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()
    return module.to(device=device, dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA/ROCm GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_paro_forward_survives_cuda_graph_capture(monkeypatch, dtype):
    # Force the portable dense path: this is the only path a HIP build can
    # take, and the one whose host-to-device offset copy used to break capture.
    monkeypatch.setattr(
        ParoLinear, "_cuda_awq_kernel_supported", lambda self, compute_dtype: False
    )
    device = torch.device("cuda", 0)
    module = _dense_paro_module(device, dtype)
    x = torch.randn(1, module.in_features, device=device, dtype=dtype)

    assert graph_capture_safe(module) is True

    with torch.no_grad():
        eager = module(x)
        torch.cuda.synchronize()
        # Warm the lazy rotation/runtime caches on a side stream, as a caller
        # must before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                module(x)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(x)
        graph.replay()
        torch.cuda.synchronize()
        replayed = captured.clone()

    torch.testing.assert_close(replayed, eager, atol=2e-2, rtol=2e-2)
