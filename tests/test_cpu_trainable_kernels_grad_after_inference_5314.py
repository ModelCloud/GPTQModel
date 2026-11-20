# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A fused eval forward must not detach later grad-mode forwards (#5314).

The CPU int4pack kernels latch ``linear_mode=inference`` on the first
``no_grad`` forward and then keep using the ``@torch.no_grad`` fused op. A
later grad-mode forward silently returned a detached tensor. The forward now
checks the call's autograd need and takes the dense-dequant path when a
gradient is required.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
    TorchAtenLinear,
    _has_local_int4pack_cpu_ops,
)
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from gptqmodel.utils.torch import TORCH_HAS_FUSED_OPS
from tests.test_cpu_int4_snapshot_5013 import _desc_act_gptq_module


IN_FEATURES = 256
OUT_FEATURES = 32
GROUP_SIZE = 128


def _skip_unless(condition, reason):
    if not condition:
        pytest.skip(reason)


def _skip_unless_valid(cls):
    ok, err = cls.cached_validate_once()
    if not ok:
        pytest.skip(str(err))


def _gptq_layer(cls, bias: bool = False):
    layer = cls(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
        format=FORMAT.GPTQ,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    torch.manual_seed(5314)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.1
    groups = IN_FEATURES // GROUP_SIZE
    scales = weight.reshape(OUT_FEATURES, groups, GROUP_SIZE).abs().amax(dim=-1).clamp_min(1e-4) / 7.0
    zeros = torch.full((OUT_FEATURES, groups), 8.0)
    groups_idx = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    source = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias)
    with torch.no_grad():
        source.weight.copy_(weight)
        if bias:
            source.bias.copy_(torch.randn(OUT_FEATURES))
    layer.pack_original(source, scales, zeros, groups_idx)
    layer.post_init()
    return layer


def _awq_layer(cls, bias: bool = False):
    layer = cls(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
        format=FORMAT.GEMM,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    groups = IN_FEATURES // GROUP_SIZE
    torch.manual_seed(5314)
    codes = torch.randint(0, 16, (IN_FEATURES, OUT_FEATURES), dtype=torch.uint8)
    zeros = torch.randint(0, 16, (groups, OUT_FEATURES), dtype=torch.uint8)
    qweight, qzeros = pack_awq(codes, zeros, 4)
    scales = torch.rand((groups, OUT_FEATURES), dtype=torch.float16) * 0.04 + 0.01
    layer.qweight.copy_(qweight)
    layer.qzeros.copy_(qzeros)
    layer.scales.copy_(scales)
    layer.post_init()
    if bias:
        with torch.no_grad():
            layer.bias.copy_(torch.randn(OUT_FEATURES))
    return layer


def _fused_gptq():
    _skip_unless(TORCH_HAS_FUSED_OPS, "Torch fused ops are unavailable")
    _skip_unless_valid(TorchFusedLinear)
    return _gptq_layer(TorchFusedLinear)


def _aten_gptq():
    _skip_unless(_has_local_int4pack_cpu_ops(), "ATen CPU int4pack ops are unavailable")
    _skip_unless_valid(TorchAtenLinear)
    return _gptq_layer(TorchAtenLinear)


def _avx512_gptq():
    _skip_unless(has_avx512_bf16(), avx512_bf16_error())
    _skip_unless_valid(TorchCpuAvx512Linear)
    return _gptq_layer(TorchCpuAvx512Linear)


def _fused_awq():
    _skip_unless(TORCH_HAS_FUSED_OPS, "Torch fused ops are unavailable")
    _skip_unless_valid(TorchFusedAwqLinear)
    return _awq_layer(TorchFusedAwqLinear)


def _aten_awq():
    _skip_unless(_has_local_int4pack_cpu_ops(), "ATen CPU int4pack ops are unavailable")
    _skip_unless_valid(TorchAtenAwqLinear)
    return _awq_layer(TorchAtenAwqLinear)


def _avx512_awq():
    _skip_unless(has_avx512_bf16(), avx512_bf16_error())
    _skip_unless_valid(TorchCpuAvx512AwqLinear)
    return _awq_layer(TorchCpuAvx512AwqLinear)


BUILDERS = [
    ("gptq-fused", _fused_gptq),
    ("gptq-aten", _aten_gptq),
    ("gptq-avx512", _avx512_gptq),
    ("awq-fused", _fused_awq),
    ("awq-aten", _aten_awq),
    ("awq-avx512", _avx512_awq),
]


def _dequantized(layer) -> torch.Tensor:
    if hasattr(layer, "awq_weight_dequantize"):
        return layer.awq_weight_dequantize(device=torch.device("cpu"), dtype=torch.float32)
    return layer.dequantize_weight().float()


@pytest.mark.parametrize("builder", [builder for _, builder in BUILDERS], ids=[name for name, _ in BUILDERS])
def test_grad_forward_after_inference_keeps_the_gradient(builder):
    layer = builder()
    layer.eval()
    # Decode the checkpoint layout before the first fused forward. A grad call
    # keeps the runtime layout installed instead of restoring the checkpoint
    # layout, so the module's own dequantizer no longer sees checkpoint words
    # after that call (#5509).
    weight = _dequantized(layer)

    x_eval = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    with torch.inference_mode():
        first = layer(x_eval)
    assert getattr(layer, "linear_mode", None) == "inference"
    assert first.shape == (4, OUT_FEATURES)

    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16, requires_grad=True)
    out = layer(x)
    assert out.requires_grad, "the latched fused path detached the grad-mode output"
    assert out.grad_fn is not None
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    assert x.grad is not None

    expected = grad_out.float().reshape(-1, OUT_FEATURES) @ weight.T
    torch.testing.assert_close(x.grad.float().reshape(-1, IN_FEATURES), expected, rtol=2e-2, atol=2e-2)

    # A later inference forward must still work after the grad call.
    with torch.inference_mode():
        again = layer(x_eval)
    assert again.shape == (4, OUT_FEATURES)
    assert torch.isfinite(again).all()


@pytest.mark.parametrize(
    "cls,backend",
    [
        (TorchFusedLinear, BACKEND.GPTQ_TORCH_FUSED),
        (TorchAtenLinear, BACKEND.GPTQ_TORCH_ATEN),
    ],
    ids=["fused", "aten"],
)
def test_eval_grad_forward_gathers_desc_act_runtime_layout(cls, backend):
    """The eager runtime-dense path must gather ``desc_act`` activations.

    A grad call decodes the live int4pack layout, which is in runtime input
    order, so the activation rows must be gathered with ``ret_idx`` exactly as
    the fused op does (#5509).
    """

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

    x = torch.randn(4, in_features, dtype=torch.bfloat16, requires_grad=True)
    out = layer(x)
    assert out.requires_grad, "the runtime-dense grad path detached the output"
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    expected = grad_out.float().reshape(-1, out_features) @ weight.T
    torch.testing.assert_close(
        x.grad.float().reshape(-1, in_features), expected, rtol=2e-2, atol=2e-2
    )
