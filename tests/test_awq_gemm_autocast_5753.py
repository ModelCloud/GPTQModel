# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ_TORCH GEMM must keep its bias epilogue wide under ambient CPU autocast (#5753).

The portable GEMM promises one reduction, one bias add and a single output
rounding. An ambient autocast context disables the fused ``addmm`` epilogue
inside ``matmul_fp32_reduce`` (autocast narrows its operands), which left a
plain matmul: the product was rounded *before* the stored bias cancelled it, so
an exact ``257 - 256`` came out as 0 instead of 1. The forward now selects the
call-local wide epilogue for a biased FP16/BF16 reduction under autocast and
restores the caller's context on exit.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear import torch_awq
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear


IN_FEATURES = 128
OUT_FEATURES = 32
GROUP_SIZE = 128


def _unit_weight_layer(dtype: torch.dtype, bias: float | None = -256.0):
    """A packed layer whose dequantized weight is all ones.

    ``qweight`` stores nibble ``1`` in every slot, ``qzeros`` is zero and every
    scale is ``1.0``, so asymmetric dequantization yields exactly ``1.0``.
    """

    module = AwqTorchLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=False,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias is not None,
        dtype=dtype,
        register_buffers=True,
    ).eval()
    module.qweight.fill_(0x11111111)
    module.qzeros.zero_()
    module.scales.fill_(1.0)
    if bias is not None:
        module.bias.fill_(bias)
    module.post_init()
    return module


def _cancelling_input(dtype: torch.dtype, shape=(1, IN_FEATURES)):
    """Input whose exact product is ``256 + 1 == 257`` before a ``-256`` bias."""

    x = torch.zeros(shape, dtype=dtype)
    x.reshape(-1, IN_FEATURES)[:, 0] = 256
    x.reshape(-1, IN_FEATURES)[:, 1] = 1
    return x


def _oracle(x: torch.Tensor, bias: float | None, dtype: torch.dtype):
    weight = torch.ones((IN_FEATURES, OUT_FEATURES), dtype=torch.float64)
    out = x.double() @ weight
    if bias is not None:
        out = out + torch.full((OUT_FEATURES,), bias, dtype=torch.float64)
    return out.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_exact_cancellation_survives_autocast(dtype):
    module = _unit_weight_layer(dtype)
    x = _cancelling_input(dtype)
    expected = _oracle(x, -256.0, dtype)

    with torch.inference_mode():
        eager = module(x)
        with torch.autocast("cpu", dtype=dtype):
            assert torch.is_autocast_enabled("cpu")
            autocast = module(x)

    # 257 is not representable in BF16 (8 mantissa bits) and FP16 is exact for
    # it, so the BF16 case is the one that used to collapse to 0.
    assert expected[0, 0].item() == 1.0
    assert eager[0, 0].item() == 1.0
    assert autocast[0, 0].item() == 1.0
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    torch.testing.assert_close(autocast, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_checkpoint_bias_matching_the_activation_dtype_is_exact(dtype):
    # The wide-bias branch never fired here: the stored bias is already the
    # activation dtype, so this is the case #5753 reported.
    module = _unit_weight_layer(dtype)
    assert module.bias.dtype is dtype

    x = _cancelling_input(dtype)
    with torch.inference_mode(), torch.autocast("cpu", dtype=dtype):
        autocast = module(x)

    assert torch.equal(autocast, _oracle(x, -256.0, dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_wide_checkpoint_bias_keeps_working(dtype):
    # Control: a bias stored wider than the activation already selected the
    # wide epilogue before #5753. The new autocast branch must not disturb it.
    module = _unit_weight_layer(dtype)
    module.bias = torch.nn.Parameter(module.bias.detach().float())
    assert module.bias.dtype is torch.float32

    x = _cancelling_input(dtype)
    with torch.inference_mode():
        eager = module(x)
        with torch.autocast("cpu", dtype=dtype):
            autocast = module(x)

    assert torch.equal(eager, _oracle(x, -256.0, dtype))
    assert torch.equal(autocast, _oracle(x, -256.0, dtype))


def _track_wide_epilogue(monkeypatch):
    calls = []
    original = torch_awq.linear_wide_bias

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_awq, "linear_wide_bias", tracked)
    return calls


def test_unbiased_layer_skips_the_wide_epilogue(monkeypatch):
    module = _unit_weight_layer(torch.bfloat16, bias=None)
    assert module.bias is None
    calls = _track_wide_epilogue(monkeypatch)
    x = _cancelling_input(torch.bfloat16)

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(x)

    assert calls == [], "an unbiased reduction needs no wide bias epilogue"
    assert out.dtype is torch.bfloat16
    assert torch.equal(out, _oracle(x, None, torch.bfloat16))


def test_biased_autocast_layer_uses_the_wide_epilogue(monkeypatch):
    module = _unit_weight_layer(torch.bfloat16)
    calls = _track_wide_epilogue(monkeypatch)
    x = _cancelling_input(torch.bfloat16)

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(x)

    assert len(calls) == 1, "a biased autocast reduction must round once"
    assert out[0, 0].item() == 1.0


def test_biased_eager_layer_keeps_the_fused_epilogue(monkeypatch):
    module = _unit_weight_layer(torch.bfloat16)
    calls = _track_wide_epilogue(monkeypatch)
    x = _cancelling_input(torch.bfloat16)

    with torch.inference_mode():
        out = module(x)

    assert calls == [], "outside autocast the fused addmm epilogue is unchanged"
    assert out[0, 0].item() == 1.0


def test_autocast_matches_eager_for_random_packed_weights():
    torch.manual_seed(5753)
    module = _unit_weight_layer(torch.bfloat16)
    module.qweight.copy_(torch.randint(-(2**31), 2**31 - 1, module.qweight.shape))
    module.qzeros.copy_(torch.randint(0, 1 << 4, module.qzeros.shape))
    module.scales.copy_(torch.rand(module.scales.shape, dtype=torch.bfloat16) * 0.02)
    module.bias.copy_(torch.randn(module.bias.shape, dtype=torch.bfloat16))
    module.post_init()

    x = torch.randn((4, IN_FEATURES), dtype=torch.bfloat16) * 4
    with torch.inference_mode():
        eager = module(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            autocast = module(x)

    torch.testing.assert_close(autocast, eager, rtol=0, atol=0)


def test_caller_autocast_context_is_restored():
    module = _unit_weight_layer(torch.bfloat16)
    x = _cancelling_input(torch.bfloat16)

    assert not torch.is_autocast_enabled("cpu")
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        module(x)
        # The local disable must not leak out of the epilogue.
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") is torch.bfloat16

    assert not torch.is_autocast_enabled("cpu")


def test_three_dimensional_input_keeps_its_shape_and_value():
    module = _unit_weight_layer(torch.bfloat16)
    x = _cancelling_input(torch.bfloat16, shape=(2, 3, IN_FEATURES))

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(x)

    assert out.shape == (2, 3, OUT_FEATURES)
    assert torch.equal(out, _oracle(x, -256.0, torch.bfloat16))
