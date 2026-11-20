# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ GEMV/GEMV_FAST must keep their FP32 epilogue out of ambient CPU autocast (#5613).

Both portable readers promise one wide reduction, one bias add and a single
output rounding. Under ``torch.autocast("cpu", bfloat16)`` the explicit FP32
matmul used to be narrowed to BF16, so the product was rounded *before* the
stored bias cancelled it: an exact ``512 + 1 - 512`` came out as 0 instead of
1. The epilogue now disables autocast locally and the caller's context is
restored on exit.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTorchLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastTorchLinear


IN_FEATURES = 128
OUT_FEATURES = 128
GROUP_SIZE = 128
BIAS = -512.0

PORTABLE_GEMV = (
    pytest.param(AwqGEMVTorchLinear, torch.int32, id="gemv"),
    pytest.param(AwqGEMVFastTorchLinear, torch.int16, id="gemv_fast"),
)


def _pack(linear: torch.nn.Linear, cls, pack_dtype: torch.dtype):
    packed = cls(
        bits=4,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        pack_dtype=pack_dtype,
        bias=True,
    )
    packed.pack(
        linear,
        torch.ones(IN_FEATURES, 1),
        torch.full((IN_FEATURES, 1), 8.0),
    )
    return packed


def _cancelling_layer(cls, pack_dtype):
    """A dense layer whose exact result is ``512 + 1 - 512 == 1``.

    All weights, activations and the bias are exactly representable in FP16
    and BF16. The FP32 product ``513`` is not representable in BF16, so any
    narrowing before the bias add rounds it to ``512`` and the layer returns 0.
    """

    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[:, :2] = 1
        linear.bias.fill_(BIAS)

    x = torch.zeros(1, IN_FEATURES, dtype=torch.float16)
    x[:, 0] = 512
    x[:, 1] = 1
    return linear, _pack(linear, cls, pack_dtype), x


def _random_layer(cls, pack_dtype, seed: int):
    generator = torch.Generator().manual_seed(seed)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator))
        linear.bias.copy_(torch.randn(linear.bias.shape, generator=generator))

    x = torch.randn((4, IN_FEATURES), generator=generator).to(torch.float16)
    return linear, _pack(linear, cls, pack_dtype), x


@pytest.mark.parametrize(("cls", "pack_dtype"), PORTABLE_GEMV)
def test_exact_result_is_one_with_and_without_autocast(cls, pack_dtype):
    linear, packed, x = _cancelling_layer(cls, pack_dtype)

    with torch.no_grad():
        eager = packed(x)
        dense = linear(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            autocast = packed(x)

    assert dense[0, 0].item() == 1.0
    assert eager[0, 0].item() == 1.0
    assert autocast[0, 0].item() == 1.0
    # The bias must not change anything outside the first output element.
    assert torch.equal(autocast, eager)
    assert autocast.dtype == torch.float16


@pytest.mark.parametrize(("cls", "pack_dtype"), PORTABLE_GEMV)
@pytest.mark.parametrize("seed", (0, 1))
def test_autocast_matches_eager_for_random_weights(cls, pack_dtype, seed):
    _linear, packed, x = _random_layer(cls, pack_dtype, seed)

    with torch.no_grad():
        eager = packed(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            autocast = packed(x)

    # Both contexts promise the same FP32 reduction and single rounding, so
    # they must agree bitwise even though the packed weight is quantized.
    assert torch.equal(autocast, eager)


@pytest.mark.parametrize(("cls", "pack_dtype"), PORTABLE_GEMV)
def test_caller_autocast_context_is_restored(cls, pack_dtype):
    _, packed, x = _cancelling_layer(cls, pack_dtype)

    assert not torch.is_autocast_enabled("cpu")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") is torch.bfloat16

        packed(x)

        # The local disable must not leak out of the epilogue.
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") is torch.bfloat16

    assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize(("cls", "pack_dtype"), PORTABLE_GEMV)
def test_nested_disabled_autocast_still_rounds_once(cls, pack_dtype):
    _, packed, x = _cancelling_layer(cls, pack_dtype)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=False):
            assert not torch.is_autocast_enabled("cpu")
            output = packed(x)
            assert not torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_enabled("cpu")

    assert output[0, 0].item() == 1.0
    assert not torch.is_autocast_enabled("cpu")
