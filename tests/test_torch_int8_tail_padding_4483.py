# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Cold INT8 source decode must crop the padded input-channel rows (#4483).

The packed GPTQ layout pads the input-channel axis up to the pack factor, so
``in_features=17`` stores 20 decoded rows. ``transform_cpu`` cropped them to
``in_features`` when it built the runtime view, but
``_dequantize_gptq_source_weight`` did not. Once the runtime view was dropped
(any ``.to(...)``/``_apply``), ``dequantize_weight()`` decoded all 20 rows and
then subtracted per-channel zeros with 17 rows, raising a row mismatch. Explicit
dequantization/export therefore depended on call order.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear


def _pack_words(words: torch.Tensor) -> torch.Tensor:
    return (
        (words[..., 0] << 0)
        | (words[..., 1] << 8)
        | (words[..., 2] << 16)
        | (words[..., 3] << 24)
    ).to(torch.int32).contiguous()


def _build(in_features: int, out_features: int, group_size: int, seed: int) -> tuple[TorchInt8Linear, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pack_factor = 4
    padded = -(-in_features // pack_factor) * pack_factor
    num_groups = 1 if group_size == -1 else -(-in_features // group_size)

    # Padding rows are real packed data; only the first `in_features` are logical.
    codes = torch.randint(0, 256, (padded, out_features), generator=generator)
    zeros = torch.randint(0, 256, (num_groups, out_features), generator=generator)
    scales = (torch.rand(num_groups, out_features, generator=generator) * 0.01 + 1e-3).half()
    groups = (
        torch.zeros(in_features, dtype=torch.int32)
        if group_size == -1
        else (torch.arange(in_features) // group_size).int()
    )

    weight = (codes[:in_features] - zeros[groups].long()).double() * scales.double()[groups]

    module = TorchInt8Linear(
        bits=8,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
    )
    module.load_state_dict(
        {
            "qweight": _pack_words(codes.reshape(padded // pack_factor, pack_factor, out_features).permute(0, 2, 1)),
            "qzeros": _pack_words(zeros.reshape(num_groups, out_features // pack_factor, pack_factor)),
            "scales": scales,
            "g_idx": groups,
        },
        strict=True,
    )
    module.post_init()
    module.eval()
    return module, weight


@pytest.mark.parametrize("in_features", [17, 18, 19])
def test_cold_source_decode_matches_the_runtime_view(in_features):
    module, expected = _build(in_features, out_features=8, group_size=-1, seed=4483)

    # Warm the runtime view, then drop it the way any dtype/device apply does.
    warm = module.dequantize_weight().clone()
    assert module.int8_module is not None

    module.to(dtype=torch.float32)
    assert module.int8_module is None, "the runtime view must be invalidated for this regression"

    cold = module.dequantize_weight()

    assert tuple(cold.shape) == (in_features, 8)
    torch.testing.assert_close(cold, warm, rtol=0, atol=0)
    torch.testing.assert_close(cold.double(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("in_features", [17, 18, 19])
def test_cold_source_decode_keeps_the_forward_oracle(in_features):
    module, expected = _build(in_features, out_features=8, group_size=32, seed=4483 + in_features)
    activation = torch.randn(3, in_features)

    module.to(dtype=torch.float32)
    assert module.int8_module is None

    with torch.inference_mode():
        output = module(activation)

    torch.testing.assert_close(output.double(), activation.double() @ expected, rtol=2e-3, atol=2e-3)


def test_aligned_width_is_unchanged():
    """K divisible by the pack factor must keep decoding every row."""

    module, expected = _build(16, out_features=8, group_size=-1, seed=7)
    module.to(dtype=torch.float32)

    torch.testing.assert_close(module.dequantize_weight().double(), expected, rtol=0, atol=0)
