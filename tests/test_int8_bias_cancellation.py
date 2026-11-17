# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_int8 import (
    INT8_GROUP_OFFSET_BUFFER_NAME,
    INT8_GROUP_SCALE_BUFFER_NAME,
    INT8_INPUT_PERM_BUFFER_NAME,
    INT8_WEIGHT_BUFFER_NAME,
    Int8PackedModule,
)


def test_torch_int8_bias_is_added_to_wide_product():
    # A single group of 512 channels whose scale is 1.0 and whose residual
    # codes are all +1.
    module = Int8PackedModule(
        torch.ones((1, 1, 512), dtype=torch.int8),
        torch.ones((1, 1)),
        group_lengths=(512,),
    )
    x = torch.full((1, 512), 128.0, dtype=torch.float16)
    bias = torch.tensor([-65504.0], dtype=torch.float16)

    output = module(x, bias=bias)

    assert output.dtype == torch.float16
    assert output[0, 0].item() == 32.0


# ---------------------------------------------------------------------------
# The asymmetric path decodes weights in FP32 and cancels them against the bias
# inside the same reduction. An outer autocast context must not narrow that
# matmul: assigning into the FP32 output tile would only widen the value that
# autocast already rounded.
# ---------------------------------------------------------------------------

_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_AUTOCAST_DTYPES = (torch.bfloat16, torch.float16)


def _folded_offset_module(lengths, out_features=1):
    """Group whose only nonzero weight column comes from the folded zero point."""
    width = max(lengths)
    if width % 8:
        width += 8 - width % 8
    codes = torch.zeros((len(lengths), out_features, width), dtype=torch.int8)
    scale = torch.full((len(lengths), out_features), 10000.0)
    offset = torch.full((len(lengths), out_features), 10000.0)
    return Int8PackedModule(codes, scale, offset, group_lengths=tuple(lengths))


def _decode_oracle(module, x, bias=None):
    """Independent FP64 group decode/reduction over the stored buffers."""
    lengths = module.group_lengths or (module.group_width,) * module.int8_weight_gnk.shape[0]
    ordered = x if module.int8_input_perm is None else x.index_select(-1, module.int8_input_perm)
    total = torch.zeros((x.shape[0], module.out_features), dtype=torch.float64)
    start = 0
    for group, length in enumerate(lengths):
        piece = ordered[:, start:start + length].double()
        start += length
        decoded = (
            module.int8_weight_gnk[group, :, :length].double()
            * module.int8_group_scale[group].double().unsqueeze(-1)
        )
        if module.int8_group_offset is not None:
            decoded = decoded + module.int8_group_offset[group].double().unsqueeze(-1)
        total = total + piece @ decoded.t()
    return total if bias is None else total + bias.double()


def _multi_group_asymmetric_module(lengths, out_features=4, seed=20261117):
    generator = torch.Generator().manual_seed(seed)
    width = max(lengths)
    if width % 8:
        width += 8 - width % 8
    codes = torch.randint(-8, 8, (len(lengths), out_features, width), generator=generator).to(torch.int8)
    scale = torch.rand((len(lengths), out_features), generator=generator) * 0.5 + 0.01
    offset = scale * 37.0
    return Int8PackedModule(codes, scale, offset, group_lengths=tuple(lengths))


def test_reported_asymmetric_bias_cancellation_under_cpu_autocast():
    module = _folded_offset_module((16,))
    x = torch.zeros(1, 16, dtype=torch.float16)
    x[0, 0] = 10
    bias = torch.tensor([-100000.0])

    normal = module(x, bias=bias)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        under_autocast = module(x, bias=bias)
        assert torch.is_autocast_enabled("cpu")

    # The checkpoint group is code=128, zero=127, scale=10000 and the bias
    # cancels the whole product.
    assert not torch.is_autocast_enabled("cpu")
    assert normal[0, 0].item() == 0.0
    assert under_autocast[0, 0].item() == 0.0
    assert under_autocast.dtype == torch.float16


@pytest.mark.parametrize("dtype", _ACTIVATION_DTYPES)
@pytest.mark.parametrize("autocast_dtype", _AUTOCAST_DTYPES)
@pytest.mark.parametrize(
    "lengths",
    [(16, 16, 16, 16), (5, 16, 3)],
    ids=["equal-groups", "ragged-groups"],
)
def test_asymmetric_group_reduction_ignores_outer_autocast(dtype, autocast_dtype, lengths):
    module = _multi_group_asymmetric_module(lengths)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((3, sum(lengths)), generator=generator).to(dtype)
    bias = torch.randn(module.out_features, generator=generator)
    expected = _decode_oracle(module, x, bias).to(dtype)

    normal = module(x, bias=bias)
    with torch.autocast("cpu", dtype=autocast_dtype):
        under_autocast = module(x, bias=bias)
    assert not torch.is_autocast_enabled("cpu")

    # The FP32 decode/reduction must round once, at the output, for both paths.
    assert normal.dtype == dtype
    assert under_autocast.dtype == dtype
    torch.testing.assert_close(normal, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(under_autocast, expected, rtol=2e-3, atol=2e-3)
    assert torch.equal(normal, under_autocast)


@pytest.mark.parametrize("dtype", _ACTIVATION_DTYPES)
@pytest.mark.parametrize("autocast_dtype", _AUTOCAST_DTYPES)
def test_symmetric_group_reduction_ignores_outer_autocast(dtype, autocast_dtype):
    generator = torch.Generator().manual_seed(11)
    codes = torch.randint(1, 8, (2, 4, 16), generator=generator).to(torch.int8)
    scale = torch.rand((2, 4), generator=generator) * 0.5 + 0.01
    module = Int8PackedModule(codes, scale, group_lengths=(16, 16))
    x = torch.randn((3, 32), generator=generator).to(dtype)
    expected = _decode_oracle(module, x).to(dtype)

    with torch.autocast("cpu", dtype=autocast_dtype):
        under_autocast = module(x)
    assert under_autocast.dtype == dtype
    torch.testing.assert_close(under_autocast, expected, rtol=2e-3, atol=2e-3)


def test_asymmetric_forward_keeps_quantized_storage_only():
    module = _multi_group_asymmetric_module((5, 16, 3))
    x = torch.randn((2, 24))
    codes = module.int8_weight_gnk.clone()

    module(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        module(x)

    assert module.state_dict() == {}
    assert set(module._buffers) == {
        INT8_WEIGHT_BUFFER_NAME,
        INT8_GROUP_SCALE_BUFFER_NAME,
        INT8_GROUP_OFFSET_BUFFER_NAME,
        INT8_INPUT_PERM_BUFFER_NAME,
    }
    assert module.int8_weight_gnk.dtype == torch.int8
    assert torch.equal(module.int8_weight_gnk, codes)
    dense_shape = (module.out_features, module.group_width)
    for tensor in vars(module).values():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            assert tuple(tensor.shape) != dense_shape
