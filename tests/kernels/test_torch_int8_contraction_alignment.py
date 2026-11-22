# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The INT8 group layout must satisfy ``_weight_int8pack_mm``'s K alignment.

``aten::_weight_int8pack_mm`` reduces the contraction axis in 8-wide chunks. On
torch 2.11 a width of 8/16/24/32 is exact while 4/12/15/17/33 reads past each
weight row and silently returns wrong values, so a group whose widest length is
not a multiple of 8 produced incorrect logits with no error at all:

* ``group_size=-1`` (one group spanning ``in_features``) with ``in_features % 8``;
* ragged ``g_idx`` groups, e.g. the ``(30, 32, 34)`` layout that negative group
  indices produce.

The group codes are now zero padded to the kernel alignment. The padded
activation columns are already zero, so the extra products contribute nothing
and the result stays exact.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_int8 import (
    Int8PackedModule,
    TorchInt8Linear,
    _align_int8_group_width,
)


def _pack_words(words: torch.Tensor) -> torch.Tensor:
    return (
        (words[..., 0] << 0)
        | (words[..., 1] << 8)
        | (words[..., 2] << 16)
        | (words[..., 3] << 24)
    ).to(torch.int32).contiguous()


def _group_oracle(
    activation: torch.Tensor,
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    lengths: tuple[int, ...],
) -> torch.Tensor:
    """Exact ``sum_k x_k * scale_g * (code_k - zero_g)`` in float64."""

    output = torch.zeros(
        activation.shape[0], codes.shape[1], dtype=torch.float64
    )
    start = 0
    for group, length in enumerate(lengths):
        rows = codes[start:start + length].double() - zeros[group].double()
        output += activation[:, start:start + length].double() @ (rows * scales[group].double())
        start += length
    assert start == codes.shape[0]
    return output


@pytest.mark.parametrize(
    "width,expected",
    [(1, 8), (4, 8), (7, 8), (8, 8), (9, 16), (16, 16), (17, 24), (30, 32), (34, 40)],
)
def test_group_width_alignment_rounds_up_to_the_kernel_contraction_step(width, expected):
    assert _align_int8_group_width(width) == expected


@pytest.mark.parametrize("width", [0, -1])
def test_group_width_alignment_rejects_non_positive_widths(width):
    with pytest.raises(ValueError, match="must be positive"):
        _align_int8_group_width(width)


@pytest.mark.parametrize("width", [4, 12, 34])
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_packed_module_is_exact_for_unaligned_group_widths(width, asymmetric):
    """A directly constructed module must pad, not miscompute."""

    generator = torch.Generator().manual_seed(4479 + width)
    num_groups = 2
    out_features = 8
    lengths = (width, width)
    codes = torch.randint(0, 256, (num_groups * width, out_features), generator=generator)
    zeros = (
        torch.full((num_groups, out_features), 128, dtype=torch.int64)
        if not asymmetric
        else torch.randint(0, 256, (num_groups, out_features), generator=generator)
    )
    scales = (torch.rand(num_groups, out_features, generator=generator) * 0.01 + 1e-3)

    residual = (codes - 128).to(torch.int8).reshape(num_groups, width, out_features).permute(0, 2, 1)
    offset = None if not asymmetric else ((128.0 - zeros.double()) * scales.double()).float()

    module = Int8PackedModule(
        residual.contiguous(),
        scales.clone(),
        offset,
        None,
        lengths,
    ).eval()

    # The runtime contraction width is padded even though the logical groups are not.
    assert module.source_group_width == width
    assert module.group_width == _align_int8_group_width(width)

    activation = torch.randn(3, num_groups * width)
    with torch.inference_mode():
        output = module(activation)

    expected = _group_oracle(activation, codes, zeros, scales, lengths)
    torch.testing.assert_close(output.double(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("in_features", [17, 20, 34])
def test_torch_int8_linear_single_group_with_unaligned_in_features(in_features):
    """``group_size=-1`` makes the group width equal to ``in_features``."""

    out_features = 8
    generator = torch.Generator().manual_seed(9000 + in_features)
    codes = torch.randint(0, 256, (in_features, out_features), generator=generator)
    zeros = torch.full((1, out_features), 128, dtype=torch.int64)
    scales = (torch.rand(1, out_features, generator=generator) * 0.01 + 1e-3).half()
    g_idx = torch.zeros(in_features, dtype=torch.int32)
    weight = (codes - zeros[0]).double() * scales.double()[0]
    # The GPTQ container pads the input-channel axis to the pack factor.
    padded = -(-in_features // 4) * 4
    packed_codes = torch.cat(
        (codes, codes.new_zeros(padded - in_features, out_features)), dim=0
    )

    module = TorchInt8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
    )
    module.load_state_dict(
        {
            "qweight": _pack_words(
                packed_codes.reshape(padded // 4, 4, out_features).permute(0, 2, 1)
            ),
            "qzeros": _pack_words(zeros.reshape(1, out_features // 4, 4)),
            "scales": scales,
            "g_idx": g_idx,
        },
        strict=True,
    )
    module.post_init()
    module.eval()

    assert module._int8_group_lengths == (in_features,)
    torch.testing.assert_close(module.dequantize_weight().double(), weight, rtol=0, atol=0)

    activation = torch.randn(3, in_features)
    with torch.inference_mode():
        output = module(activation)
    torch.testing.assert_close(
        output.double(), activation.double() @ weight, rtol=2e-3, atol=2e-3
    )
