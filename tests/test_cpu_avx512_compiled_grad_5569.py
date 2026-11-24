# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A compiled grad forward must stay differentiable on the AVX512 CPU kernels.

#5499 gave the CPU int4pack forwards a differentiable branch for grad-requiring
calls after the fused inference warm-up, but the two AVX512 kernels were left
out (#5569). ``TorchCpuAvx512AwqLinear`` never consulted the autograd need in
its compile branch, so a traced grad forward took the ``@torch.no_grad`` fused
op and returned a detached tensor. (The GPTQ half, ``TorchCpuAvx512Linear``,
was fixed by the ``_runtime_dense_weight_is_checkpoint_order`` contract that
main gained afterwards; the nodes below keep it covered.)

The AVX512-BF16 extension is only needed for ``transform_cpu`` itself, so the
AWQ nodes build the post-warm-up state by hand and run on any CPU host. The
end-to-end nodes skip unless the host provides the kernel.
"""

import threading

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import (
    TorchCpuAvx512Linear,
    _fake_gemm_int4_qzero_group_scaled,
)
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.cpu_features import avx512_bf16_error, has_avx512_bf16
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


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _mode(layer):
    mode = getattr(layer, "linear_mode", None)
    return getattr(mode, "value", mode)


def _untransformed_awq_module():
    """An AVX512 AWQ module in the state a fused inference warm-up leaves.

    The kernel's ``__init__`` validates AVX512-BF16, but the dense fall-back
    this test targets never touches the extension. Buffers are filled the same
    way ``_awq_layer`` fills them, so ``dequantize_weight`` is the ground truth.
    """

    groups = IN_FEATURES // 128
    codes = torch.randint(0, 16, (IN_FEATURES, OUT_FEATURES), dtype=torch.uint8)
    zeros = torch.randint(0, 16, (groups, OUT_FEATURES), dtype=torch.uint8)
    qweight, qzeros = pack_awq(codes, zeros, 4)
    scales = torch.rand((groups, OUT_FEATURES), dtype=torch.float16) * 0.04 + 0.01

    layer = TorchCpuAvx512AwqLinear.__new__(TorchCpuAvx512AwqLinear)
    torch.nn.Module.__init__(layer)
    layer.bits = 4
    layer.group_size = 128
    layer.in_features = IN_FEATURES
    layer.out_features = OUT_FEATURES
    layer.bias = None
    layer.qweight = qweight
    layer.qzeros = qzeros
    layer.scales = scales
    layer.linear_mode = "inference"
    layer._cpu_transformed = False
    layer._cpu_transform_lock = threading.RLock()
    layer.adapter = None
    layer.training = False
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


def test_fake_int4_gemm_reports_the_output_width_for_both_layouts():
    """The traced shape must be ``[rows, out_features]`` for GPTQ and AWQ.

    ``weight.shape[0]`` is ``out_features`` for the GPTQ packed layout but
    ``in_features`` for AWQ's ``[in_features, out_features // 2]`` layout, so
    the fake implementation used to hand dynamo a width the real kernel never
    returns and the traced reshape failed.
    """

    x = torch.zeros(4, IN_FEATURES, dtype=torch.bfloat16)
    absmax = torch.zeros(IN_FEATURES // 128, OUT_FEATURES, dtype=torch.float32)
    gptq_weight = torch.zeros(OUT_FEATURES, IN_FEATURES // 2, dtype=torch.int32)
    awq_weight = torch.zeros(IN_FEATURES, OUT_FEATURES // 2, dtype=torch.int32)

    for weight in (gptq_weight, awq_weight):
        out = _fake_gemm_int4_qzero_group_scaled(x, weight, None, absmax, 128)
        assert out.shape == (4, OUT_FEATURES), tuple(weight.shape)


def test_avx512_awq_compiled_grad_forward_keeps_gradient():
    layer = _untransformed_awq_module()
    weight = _dequantized(layer)

    x, _out, grad_out = _grad_forward(layer)

    _assert_matches_dense_reference(x, grad_out, weight)


def test_avx512_awq_compiled_grad_forward_keeps_the_runtime_latch():
    layer = _untransformed_awq_module()

    _grad_forward(layer)

    assert _mode(layer) == "inference", "the dense fall-back mutated the runtime latch"
    dense_shapes = {
        (IN_FEATURES, OUT_FEATURES),
        (OUT_FEATURES, IN_FEATURES),
    }
    for name, value in list(layer.named_parameters()) + list(layer.named_buffers()):
        assert tuple(value.shape) not in dense_shapes, f"{name} retained a dense weight"


def test_avx512_awq_compiled_inference_forward_matches_eager():
    _skip_unless_avx512(TorchCpuAvx512AwqLinear)

    # The fused kernel consumes the runtime layout produced by
    # ``transform_cpu``; a hand-built ``linear_mode="inference"`` module that
    # was never transformed is not a reachable state. Use the real checkpoint
    # fixture and let the eager warm-up install the layout.
    layer = _awq_layer(TorchCpuAvx512AwqLinear)
    layer.eval()
    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)

    with torch.inference_mode():
        eager = layer(x)

    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    with torch.inference_mode():
        traced = compiled(x)

    torch.testing.assert_close(traced.float(), eager.float(), rtol=2e-2, atol=2e-2)


def test_avx512_gptq_runtime_dense_weight_is_checkpoint_order():
    _skip_unless_avx512(TorchCpuAvx512Linear)

    layer = _gptq_layer(TorchCpuAvx512Linear)

    assert TorchCpuAvx512Linear._runtime_dense_weight_is_checkpoint_order is True
    torch.testing.assert_close(layer._runtime_dense_weight(), layer.dequantize_weight())


def test_avx512_gptq_compiled_grad_forward_after_fused_warm_up():
    _skip_unless_avx512(TorchCpuAvx512Linear)

    layer = _gptq_layer(TorchCpuAvx512Linear)
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


def test_avx512_awq_compiled_grad_forward_after_fused_warm_up():
    _skip_unless_avx512(TorchCpuAvx512AwqLinear)

    layer = _awq_layer(TorchCpuAvx512AwqLinear)
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
