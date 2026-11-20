# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A compiled grad forward must stay differentiable on the AVX512 CPU kernels.

#5499 gave the CPU int4pack forwards a differentiable branch for grad-requiring
calls after the fused inference warm-up, but the two AVX512 kernels were left
out (#5569):

* ``TorchCpuAvx512Linear`` inherits ``TorchAtenLinear.forward``, whose compile
  branch decoded the live ATen int4pack layout directly. The AVX512 transform
  builds the HF int4 tile layout instead, so that decoder raised
  ``RuntimeError: ... has no verified CPU int4pack block size`` even though the
  inference forward had run.
* ``TorchCpuAvx512AwqLinear`` never consulted the autograd need in its compile
  branch, so a traced grad forward took the ``@torch.no_grad`` fused op and
  returned a detached tensor.

The compile branch now asks the module for its dense operands through
``_runtime_dense_operands()``. The first node exercises that contract on the
ATen int4pack kernel, which runs on any CPU host; the AVX512 nodes need an
AVX512-BF16 host and skip otherwise.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
    TorchAtenLinear,
    _has_local_int4pack_cpu_ops,
)
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from tests.test_cpu_int4_snapshot_5013 import _desc_act_gptq_module
from tests.test_cpu_trainable_kernels_grad_after_inference_5314 import (
    IN_FEATURES,
    OUT_FEATURES,
    _awq_layer,
    _dequantized,
    _gptq_layer,
)


pytestmark = [pytest.mark.cpu]


def _skip_unless_avx512(cls):
    if not has_avx512_bf16():
        pytest.skip(avx512_bf16_error())
    ok, err = cls.cached_validate_once()
    if not ok:
        pytest.skip(str(err))


def _skip_unless_aten():
    if not _has_local_int4pack_cpu_ops():
        pytest.skip("ATen CPU int4pack ops are unavailable")
    ok, err = TorchAtenLinear.cached_validate_once()
    if not ok:
        pytest.skip(str(err))


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _mode(layer):
    mode = getattr(layer, "linear_mode", None)
    return getattr(mode, "value", mode)


def _latch_inference(layer):
    """Reproduce the state a fused eager warm-up leaves on the module.

    ``transform_cpu`` needs the AVX512-BF16 extension; the autograd dispatch
    under test only reads the latched ``linear_mode``.
    """

    layer.eval()
    layer.linear_mode = "inference"
    assert _mode(layer) == "inference"
    return layer


def _grad_forward(layer):
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16, requires_grad=True)
    out = compiled(x)
    assert out.requires_grad, "the compiled forward detached the grad-mode output"
    assert out.grad_fn is not None

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    return x, out, grad_out


def _assert_matches_dense_reference(x, grad_out, weight):
    expected = grad_out.float().reshape(-1, OUT_FEATURES) @ weight.T
    torch.testing.assert_close(
        x.grad.float().reshape(-1, IN_FEATURES),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


def test_compile_branch_uses_the_runtime_dense_hook(monkeypatch):
    _skip_unless_aten()

    calls = []

    def checkpoint_order_operands(module):
        # ``TorchCpuAvx512Linear`` has the same contract: its runtime layout is
        # not the ATen int4pack permutation, so it decodes with
        # ``dequantize_weight()`` and must not gather ``ret_idx`` again.
        calls.append(1)
        return module.dequantize_weight(), False

    monkeypatch.setattr(
        TorchAtenLinear, "_runtime_dense_operands", checkpoint_order_operands
    )

    layer = _latch_inference(_gptq_layer(TorchAtenLinear))
    weight = _dequantized(layer)

    x, _out, grad_out = _grad_forward(layer)

    assert calls, "the compile branch bypassed _runtime_dense_operands()"
    _assert_matches_dense_reference(x, grad_out, weight)


def test_avx512_gptq_dense_operands_skip_the_int4pack_decoder():
    _skip_unless_avx512(TorchCpuAvx512Linear)

    layer = _gptq_layer(TorchCpuAvx512Linear)

    weight, gather = layer._runtime_dense_operands()

    assert gather is False, "dequantize_weight() already restores checkpoint order"
    torch.testing.assert_close(weight, layer.dequantize_weight())


def test_avx512_gptq_compiled_grad_forward_after_inference_latch_keeps_gradient():
    _skip_unless_avx512(TorchCpuAvx512Linear)

    layer = _latch_inference(_gptq_layer(TorchCpuAvx512Linear))
    weight = _dequantized(layer)

    x, _out, grad_out = _grad_forward(layer)

    _assert_matches_dense_reference(x, grad_out, weight)


def test_avx512_gptq_compiled_grad_forward_keeps_the_runtime_latch():
    _skip_unless_avx512(TorchCpuAvx512Linear)

    layer = _latch_inference(_gptq_layer(TorchCpuAvx512Linear))

    _grad_forward(layer)

    assert _mode(layer) == "inference", "the dense fallback mutated the runtime latch"
    dense_shapes = {
        (IN_FEATURES, OUT_FEATURES),
        (OUT_FEATURES, IN_FEATURES),
    }
    for name, value in list(layer.named_parameters()) + list(layer.named_buffers()):
        assert tuple(value.shape) not in dense_shapes, f"{name} retained a dense weight"


def test_avx512_gptq_compiled_grad_forward_with_bias_keeps_gradient():
    _skip_unless_avx512(TorchCpuAvx512Linear)

    layer = _latch_inference(_gptq_layer(TorchCpuAvx512Linear, bias=True))
    weight = _dequantized(layer)

    x, _out, grad_out = _grad_forward(layer)

    _assert_matches_dense_reference(x, grad_out, weight)


def test_avx512_gptq_desc_act_compiled_grad_forward_keeps_checkpoint_order():
    _skip_unless_avx512(TorchCpuAvx512Linear)

    layer = _desc_act_gptq_module(
        TorchCpuAvx512Linear,
        BACKEND.GPTQ_TORCH_CPU_AVX512,
        in_features=IN_FEATURES,
        group_size=128,
    )
    assert not layer._ret_idx_is_identity(), "fixture must reorder the input rows"
    weight, gather = layer._runtime_dense_operands()
    assert gather is False, "checkpoint order must not be gathered again"
    torch.testing.assert_close(weight, layer.dequantize_weight())

    _latch_inference(layer)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16, requires_grad=True)
    out = compiled(x)
    assert out.requires_grad
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_avx512_awq_compiled_grad_forward_after_inference_latch_keeps_gradient():
    _skip_unless_avx512(TorchCpuAvx512AwqLinear)

    layer = _latch_inference(_awq_layer(TorchCpuAvx512AwqLinear))
    weight = _dequantized(layer)

    x, _out, grad_out = _grad_forward(layer)

    _assert_matches_dense_reference(x, grad_out, weight)


def test_avx512_awq_compiled_grad_forward_keeps_the_runtime_latch():
    _skip_unless_avx512(TorchCpuAvx512AwqLinear)

    layer = _latch_inference(_awq_layer(TorchCpuAvx512AwqLinear))

    _grad_forward(layer)

    assert _mode(layer) == "inference", "the dense fallback mutated the runtime latch"


def test_avx512_awq_compiled_grad_forward_with_bias_keeps_gradient():
    _skip_unless_avx512(TorchCpuAvx512AwqLinear)

    layer = _latch_inference(_awq_layer(TorchCpuAvx512AwqLinear, bias=True))
    weight = _dequantized(layer)

    x, _out, grad_out = _grad_forward(layer)

    _assert_matches_dense_reference(x, grad_out, weight)


@pytest.mark.parametrize("cls", [TorchCpuAvx512Linear, TorchCpuAvx512AwqLinear])
def test_avx512_compiled_grad_forward_after_fused_warmup(cls):
    _skip_unless_avx512(cls)

    builder = _gptq_layer if cls is TorchCpuAvx512Linear else _awq_layer
    layer = builder(cls)
    layer.eval()
    x_eval = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    with torch.inference_mode():
        eager = layer(x_eval)
    assert _mode(layer) == "inference"
    assert layer._cpu_transformed

    weight = _dequantized(layer)
    x, _out, grad_out = _grad_forward(layer)
    _assert_matches_dense_reference(x, grad_out, weight)

    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    with torch.inference_mode():
        traced = compiled(x_eval)
    torch.testing.assert_close(traced.float(), eager.float(), rtol=2e-2, atol=2e-2)
