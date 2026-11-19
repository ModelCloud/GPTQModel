# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A compiled grad forward must keep activation gradients after a fused warmup.

The CPU int4pack kernels latch ``linear_mode="inference"`` on the first
no-grad forward. The compile branch only read that latch, so a grad-requiring
``torch.compile`` call took the ``@torch.no_grad`` fused op and returned a
detached tensor. The eager path restored the checkpoint layout and kept the
gradient (#5314, #5392); the compiled path must do the same without mutating
the packed runtime state inside the trace (#5369).
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
    TorchAtenLinear,
    _has_local_int4pack_cpu_ops,
)
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.torch import TORCH_HAS_FUSED_OPS
from tests.test_cpu_int4_snapshot_5013 import _desc_act_gptq_module
from tests.test_cpu_trainable_kernels_grad_after_inference_5314 import (
    IN_FEATURES,
    OUT_FEATURES,
    _aten_awq,
    _aten_gptq,
    _awq_layer,
    _dequantized,
    _fused_awq,
    _fused_gptq,
    _gptq_layer,
    _skip_unless,
    _skip_unless_valid,
)


BUILDERS = [
    ("gptq-fused", _fused_gptq),
    ("gptq-aten", _aten_gptq),
    ("awq-fused", _fused_awq),
    ("awq-aten", _aten_awq),
]


def _fused_gptq_bias():
    _skip_unless(TORCH_HAS_FUSED_OPS, "Torch fused ops are unavailable")
    _skip_unless_valid(TorchFusedLinear)
    return _gptq_layer(TorchFusedLinear, bias=True)


def _aten_gptq_bias():
    _skip_unless(_has_local_int4pack_cpu_ops(), "ATen CPU int4pack ops are unavailable")
    _skip_unless_valid(TorchAtenLinear)
    return _gptq_layer(TorchAtenLinear, bias=True)


def _fused_awq_bias():
    _skip_unless(TORCH_HAS_FUSED_OPS, "Torch fused ops are unavailable")
    _skip_unless_valid(TorchFusedAwqLinear)
    return _awq_layer(TorchFusedAwqLinear, bias=True)


def _aten_awq_bias():
    _skip_unless(_has_local_int4pack_cpu_ops(), "ATen CPU int4pack ops are unavailable")
    _skip_unless_valid(TorchAtenAwqLinear)
    return _awq_layer(TorchAtenAwqLinear, bias=True)


BIAS_BUILDERS = [
    ("gptq-fused", _fused_gptq_bias),
    ("gptq-aten", _aten_gptq_bias),
    ("awq-fused", _fused_awq_bias),
    ("awq-aten", _aten_awq_bias),
]

DESC_ACT_BUILDERS = [
    ("gptq-fused", TorchFusedLinear, BACKEND.GPTQ_TORCH_FUSED),
    ("gptq-aten", TorchAtenLinear, BACKEND.GPTQ_TORCH_ATEN),
]


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _mode(layer):
    mode = getattr(layer, "linear_mode", None)
    return getattr(mode, "value", mode)


def _warm(layer):
    layer.eval()
    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    with torch.inference_mode():
        layer(x)
    assert _mode(layer) == "inference"
    return x


def _reference(builder):
    layer = builder()
    layer.eval()
    weight = _dequantized(layer).float()
    return layer, weight


def _assert_compiled_grad(layer, weight):
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16, requires_grad=True)
    out = compiled(x)
    assert out.requires_grad, "the compiled forward detached the grad-mode output"
    assert out.grad_fn is not None

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    assert x.grad is not None

    expected = grad_out.float().reshape(-1, OUT_FEATURES) @ weight.T
    torch.testing.assert_close(
        x.grad.float().reshape(-1, IN_FEATURES),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    "builder", [builder for _, builder in BUILDERS], ids=[name for name, _ in BUILDERS]
)
def test_compiled_grad_forward_after_eager_warm_keeps_gradient(builder):
    reference, weight = _reference(builder)
    layer = builder()
    _warm(layer)

    _assert_compiled_grad(layer, weight)

    # The dense fallback must not disturb the runtime layout: the reference
    # checkpoint-layout forward still matches the compiled output.
    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    with torch.inference_mode():
        compiled_out = compiled(x)
        expected_out = reference(x)
    torch.testing.assert_close(
        compiled_out.float(), expected_out.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    "builder", [builder for _, builder in BUILDERS], ids=[name for name, _ in BUILDERS]
)
def test_compiled_inference_then_grad_forward_keeps_gradient(builder):
    _, weight = _reference(builder)
    layer = builder()
    x_eval = _warm(layer)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)

    with torch.inference_mode():
        first = compiled(x_eval)
    assert first.shape == (4, OUT_FEATURES)
    assert torch.isfinite(first).all()

    _assert_compiled_grad(layer, weight)


@pytest.mark.parametrize(
    "builder",
    [builder for _, builder in BIAS_BUILDERS],
    ids=[name for name, _ in BIAS_BUILDERS],
)
def test_compiled_grad_forward_with_bias_keeps_gradient(builder):
    reference, weight = _reference(builder)
    layer = builder()
    _warm(layer)

    _assert_compiled_grad(layer, weight)

    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    with torch.inference_mode():
        compiled_out = compiled(x)
        expected_out = reference(x)
    torch.testing.assert_close(
        compiled_out.float(), expected_out.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    "cls,backend",
    [(cls, backend) for _, cls, backend in DESC_ACT_BUILDERS],
    ids=[name for name, _, _ in DESC_ACT_BUILDERS],
)
def test_compiled_grad_forward_gathers_desc_act_activations(cls, backend):
    in_features = 128
    out_features = 64
    reference = _desc_act_gptq_module(
        cls, backend, in_features=in_features, group_size=32
    )
    weight = reference.dequantize_weight().float()
    layer = _desc_act_gptq_module(
        cls, backend, in_features=in_features, group_size=32
    )
    assert not layer._ret_idx_is_identity()

    layer.eval()
    with torch.inference_mode():
        layer(torch.randn(4, in_features, dtype=torch.bfloat16))

    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    x = torch.randn(4, in_features, dtype=torch.bfloat16, requires_grad=True)
    out = compiled(x)
    assert out.requires_grad, "the compiled forward detached the grad-mode output"
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    expected = grad_out.float().reshape(-1, out_features) @ weight.T
    torch.testing.assert_close(
        x.grad.float().reshape(-1, in_features),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


def _issue_fixture_layer(cls):
    if cls is TorchFusedLinear:
        _skip_unless(TORCH_HAS_FUSED_OPS, "Torch fused ops are unavailable")
    else:
        _skip_unless(
            _has_local_int4pack_cpu_ops(), "ATen CPU int4pack ops are unavailable"
        )
    _skip_unless_valid(cls)

    layer = cls(
        bits=4,
        group_size=32,
        in_features=32,
        out_features=16,
        sym=True,
        desc_act=False,
        register_buffers=True,
    ).eval()
    layer.qzero_format(2)
    with torch.no_grad():
        layer.qweight.fill_(0x77777777)
        layer.qzeros.fill_(-2004318072)  # packed zero code 8
        layer.scales.fill_(0.5)
    layer.post_init()
    with torch.no_grad():
        layer(torch.ones(1, 32, dtype=torch.bfloat16))
    return layer


@pytest.mark.parametrize(
    "cls", [TorchFusedLinear, TorchAtenLinear], ids=["gptq-fused", "gptq-aten"]
)
def test_issue_5369_fixture_compiled_grad_matches_eager(cls):
    compiled = torch.compile(_issue_fixture_layer(cls), backend="eager", fullgraph=True)
    x = torch.ones(1, 32, dtype=torch.bfloat16, requires_grad=True)
    out = compiled(x)
    assert out.requires_grad, "the compiled forward detached the grad-mode output"
    grad = torch.autograd.grad(out.sum(), x)[0][0, 0]
    torch.testing.assert_close(grad.float(), torch.tensor(-8.0), rtol=0, atol=0)
