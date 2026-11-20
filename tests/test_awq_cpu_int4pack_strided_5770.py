# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ CPU int4pack must accept noncontiguous FP16 activations (#5770).

``aten::_weight_int4pack_mm_cpu`` requires a contiguous activation. The AWQ
ATen and Fused wrappers handed it the caller's view unchanged, so a row-gap
view (``src[:, 1, :]``) or a column stride (``x[:, ::2]``) raised ``expect A to
be contiguous`` while the identical values were exact after ``.contiguous()``.
GPTQ's wrappers already normalized the layout; AWQ's did not.

The shared ``_cpu_int4pack_mm`` boundary now normalizes the activation, and
the casting custom op normalizes the operands it casts, so every caller gets
one contract. These cases build the packed buffers directly (no checkpoint)
and compare against an independent sum oracle.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear


IN_FEATURES = 128
OUT_FEATURES = 32
GROUP_SIZE = 128

NATIVE_AWQ_KERNELS = (TorchAtenAwqLinear, TorchFusedAwqLinear)
ALL_AWQ_KERNELS = (AwqTorchLinear, *NATIVE_AWQ_KERNELS)


def _module(cls, dtype, *, bias=False):
    """A packed AWQ layer whose codes all decode to weight ``1.0``.

    ``qweight`` holds nibble code 1 everywhere, ``qzeros`` is zero and every
    scale is 1.0, so each output row is exactly ``sum(x)`` (plus a zero bias
    when requested). That makes the sum oracle independent of the packing
    layout.
    """

    module = cls(
        bits=4,
        group_size=GROUP_SIZE,
        sym=False,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
        dtype=dtype,
        register_buffers=True,
    ).eval()
    module.qweight.fill_(0x11111111)
    module.qzeros.zero_()
    module.scales.fill_(1.0)
    if bias:
        module.bias.zero_()
    module.post_init()
    return module


def _values(count, dtype):
    return torch.arange(count).remainder(7).to(dtype) / 8


def _oracle(x, out_features):
    return x.double().sum(-1, keepdim=True).expand(x.shape[0], out_features).to(x.dtype)


def _layouts(rows, dtype):
    """The strided views reported in the issue, plus an offset-contiguous view."""

    base = _values(rows * 2 * IN_FEATURES, dtype).reshape(rows, 2, IN_FEATURES)
    yield "row_gap", base[:, 1, :]
    yield "column_stride", base.reshape(rows, 2 * IN_FEATURES)[:, ::2]

    pad = 7
    storage = _values(pad + rows * IN_FEATURES, dtype)
    offset_view = storage[pad:].view(rows, IN_FEATURES)
    assert offset_view.storage_offset() == pad
    yield "offset_contiguous", offset_view


def _assert_native_layout(module):
    """The contiguous control forward installed the native inference layout."""

    if isinstance(module, TorchAtenAwqLinear):
        assert module._cpu_transformed
    elif isinstance(module, TorchFusedAwqLinear):
        assert module.torch_fused_op is not None
    else:  # AwqTorchLinear dequantizes per forward; it has no native layout.
        return
    assert module.scales_and_zeros is not None


@pytest.mark.cpu
@pytest.mark.parametrize("cls", ALL_AWQ_KERNELS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_noncontiguous_activations_match_contiguous(cls, dtype):
    module = _module(cls, dtype)

    for rows in (1, 3, 16):
        for layout, x in _layouts(rows, dtype):
            assert not x.is_contiguous() or x.storage_offset() > 0, layout
            expected = _oracle(x, OUT_FEATURES)

            with torch.inference_mode():
                torch.testing.assert_close(
                    module(x.contiguous()), expected, atol=0, rtol=0
                )
                torch.testing.assert_close(module(x), expected, atol=0, rtol=0)

    _assert_native_layout(module)


@pytest.mark.cpu
@pytest.mark.parametrize("cls", NATIVE_AWQ_KERNELS, ids=lambda cls: cls.__name__)
def test_biased_fp16_cast_path_accepts_transposed_activation(cls):
    """A biased FP16 layer takes the casting op; transposed strides survive it."""

    module = _module(cls, torch.float16, bias=True)
    base = _values(IN_FEATURES * 3, torch.float16).reshape(IN_FEATURES, 3)
    x = base.t()
    assert not x.is_contiguous()
    expected = _oracle(x, OUT_FEATURES)

    with torch.inference_mode():
        torch.testing.assert_close(module(x), expected, atol=0, rtol=0)

    _assert_native_layout(module)
