# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A non-identity Paro Triton layer must trace and compile without graph breaks.

``ParoTritonLinear`` declares ``COMPILE_LEVEL.CUDAGRAPH``, but every layer with
a learned rotation used to break the graph twice (#5225):

* ``_resolve_rotation_launch`` -> ``_ensure_rotation_autotune_cache_loaded``
  entered ``_ROTATION_AUTOTUNE_DISK_LOCK`` before its "already loaded" check,
  so an eager warmup could not stop dynamo from hitting an unsupported lock
  context manager.
* The fused rotation op (``gptqmodel_paro.rotate``) had no fake/meta kernel, so
  tracing with fake tensors raised ``Operator does not support running with
  fake tensors``.

The compile-level tests missed both because their Paro builder used an identity
rotation (``_rotate_inputs`` returns early) and ``group_size=32`` (the fused
kernel only accepts 128). These tests build a non-identity layer and assert the
whole forward traces, compiles and captures, and that the fused op has a Meta
implementation. They are red on unmodified ``main``.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.paro_triton import ParoTritonLinear
from gptqmodel.utils.paro import build_identity_rotation_buffers


pytestmark = [pytest.mark.gpu]


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unpacked integers into the AWQ bit layout used by the kernels."""
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


def _build_nonidentity_layer(*, out_features: int = 64) -> tuple[ParoTritonLinear, torch.Tensor]:
    """Build a packed Paro Triton layer whose rotation is not the identity."""
    in_features, group_size = 256, 128
    torch.manual_seed(7)
    module = ParoTritonLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**4, (in_features, out_features), dtype=torch.int32)
    module.qweight.copy_(_pack_awq_tensor(int_weight, 4))
    module.qzeros.copy_(
        _pack_awq_tensor(torch.full((groups, out_features), 8, dtype=torch.int32), 4)
    )
    module.scales.copy_(torch.rand(groups, out_features, dtype=torch.float16) * 0.03 + 0.01)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=module.krot,
        dtype=torch.float16,
    )
    theta.fill_(0.2)
    channel_scales.mul_(0.75)
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module = module.cuda().eval()
    assert module._rotation_identity is False
    x = torch.randn(4, in_features, device="cuda", dtype=torch.float16)
    return module, x


def _reset_dynamo() -> None:
    torch._dynamo.reset()


def test_nonidentity_layer_traces_with_zero_graph_breaks():
    """A warmed eager forward must make the whole traced forward one graph."""
    module, x = _build_nonidentity_layer()
    with torch.inference_mode():
        reference = module(x).clone()

    _reset_dynamo()
    explained = torch._dynamo.explain(module.forward)(x)
    assert explained.graph_break_count == 0, [
        str(reason.reason)[:300] for reason in explained.break_reasons[:5]
    ]

    for backend in ("eager", "aot_eager", "inductor"):
        _reset_dynamo()
        compiled = torch.compile(module.forward, fullgraph=True, backend=backend)
        with torch.inference_mode():
            out = compiled(x)
        torch.testing.assert_close(out, reference, atol=2e-2, rtol=1e-2)


def test_nonidentity_layer_reaches_declared_cudagraph_level():
    """The declared ``CUDAGRAPH`` level must be reachable with a real rotation."""
    from tests.kernels.compile_probe import probe

    module, x = _build_nonidentity_layer()
    result = probe(module, (x,))
    assert result.eager, result.eager_error
    assert result.graph_breaks == 0, result.break_reasons[:5]
    assert result.reached(type(module).SUPPORTS_COMPILE_LEVEL), result.summary()
    assert result.graph_capture is True, result.graph_capture_error


def test_fused_rotate_ops_have_fake_tensor_implementations():
    """The CUDA-only fused ops need Meta kernels so dynamo can trace them."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    module, x = _build_nonidentity_layer()
    # Loading the JIT extension happens on the first eager call, outside any
    # trace, exactly like a normal model load.
    with torch.inference_mode():
        module(x)

    with FakeTensorMode():
        fx = torch.empty((4, 256), dtype=torch.float16, device="cuda")
        fpairs = torch.empty((8, 256), dtype=torch.int16, device="cuda")
        ftheta = torch.empty((8, 128), dtype=torch.float16, device="cuda")
        fscales = torch.empty((1, 256), dtype=torch.float16, device="cuda")
        rotated = torch.ops.gptqmodel_paro.rotate(fx, fpairs, ftheta, fscales, 128, 8, 2, 0)
        assert rotated.shape == fx.shape
        assert rotated.dtype == torch.float16

        rotated2, grad2, grad_theta = torch.ops.gptqmodel_paro.rotate_pair_grad(
            fx, fx, fpairs, ftheta, 128, 8, 2
        )
        assert rotated2.shape == fx.shape
        assert grad2.shape == fx.shape
        assert grad_theta.shape == (8, 128)
        assert grad_theta.dtype == torch.float32


def test_compile_level_probe_builder_uses_a_real_rotation():
    """The level test's builder must not take the identity early return."""
    from tests.kernels.test_kernel_compile_levels_quant import _build_paro

    module, _, _ = _build_paro(ParoTritonLinear)
    assert module._rotation_identity is False
    assert module.group_size == 128
    assert float(module.theta.abs().max()) > 0.0


def test_optimize_compiles_a_tiny_paro_model_without_breaks():
    """Model-level ``optimize()`` must install a graph-break-free forward."""
    import torch.nn as nn

    from gptqmodel.models.base import BaseQModel

    class Tiny(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class Harness(BaseQModel):
        def __init__(self, model):
            nn.Module.__init__(self)
            self.model = model
            self.quantized = True
            self.compiled = False

    layers = [_build_nonidentity_layer(out_features=256)[0] for _ in range(2)]
    model = Tiny(layers).cuda().eval()
    x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        reference = model(x).clone()

    harness = Harness(model)
    harness.optimize(backend="inductor")
    if not harness.compiled:
        pytest.skip("torch.compile is unavailable on this build")

    from torch._dynamo.utils import counters

    counters.clear()
    with torch.inference_mode():
        out = harness.model(x)
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=1e-2)
