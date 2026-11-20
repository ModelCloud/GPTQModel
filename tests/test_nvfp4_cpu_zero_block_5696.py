# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #5696: an all-zero block's scale must not change CPU output.

``TorchFP4Linear``'s CPU fallback used to choose the narrow decode from the
global scale range.  Two packs whose decoded weights are identical, differing
only in the block scale of a block whose E2M1 codes are all zero, selected
different decode precisions: with the unused block scale at ``1`` the active
weight ``64.03125`` was rounded to FP16 ``64`` before the cancelling ``-2048``
bias, while at ``256`` the range gate selected the wide path.  The exact
projection is ``32 * 64.03125 - 2048 = 1`` and is representable in FP16.

The CPU gate must ignore blocks that decode to zeros and must keep the narrow
decode only when every ``code * scale`` is exactly representable in the
activation dtype; otherwise the wide path preserves the effective weight until
the final output rounding.
"""

from __future__ import annotations

import torch

from gptqmodel.nn_modules.qlinear.fp4 import TorchFP4Linear
from gptqmodel.quantization.dtype import dequantize_f4_e2m1


E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
# 32 * 64.03125 - 2048 == 1.  The scale itself is not representable in FP16
# (it needs 12 significant bits), which is what the old gate missed.
GLOBAL_SCALE = 64.03125
ZERO_BLOCK_SCALES = (1.0, 256.0)
BIAS = -2048.0


def _oracle_weight(
    packed: torch.Tensor, scale: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Decode the packed nibbles and the effective scale independently in FP64."""

    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    values = torch.tensor(E2M1_VALUES, dtype=torch.float64)[codes]
    expanded = scale.to(torch.float64).repeat_interleave(block_size, dim=-1)
    return values * expanded


def _zero_block_linear(unused_scale: float) -> TorchFP4Linear:
    """One output: block 0 is a single E2M1 +1, block 1 is all zeros."""

    packed = torch.zeros(1, 16, dtype=torch.uint8)
    packed[0, 0] = 2  # low nibble is the first logical value: E2M1 +1
    return TorchFP4Linear(
        in_features=32,
        out_features=1,
        weight=packed,
        weight_scale=torch.tensor([[1.0, unused_scale]], dtype=torch.float32) * GLOBAL_SCALE,
        weight_block_size=16,
        orig_dtype=torch.float16,
        bias=torch.tensor([BIAS], dtype=torch.float32),
    ).eval()


def _issue_input() -> torch.Tensor:
    x = torch.zeros(1, 32, dtype=torch.float16)
    x[0, 0] = 32
    return x


def test_zero_block_scale_leaves_decoded_weights_identical():
    """Changing only the all-zero block's scale cannot change the weights."""

    modules = [_zero_block_linear(unused) for unused in ZERO_BLOCK_SCALES]
    decoded = [
        dequantize_f4_e2m1(
            module.weight,
            scale=module.weight_scale,
            axis=None,
            target_dtype=torch.float64,
        )
        for module in modules
    ]
    assert torch.equal(decoded[0], decoded[1])
    expected = _oracle_weight(modules[0].weight, modules[0].weight_scale, 16)
    assert torch.equal(decoded[0], expected)
    assert float(expected[0, 0]) == GLOBAL_SCALE
    assert torch.equal(expected[0, 16:], torch.zeros(16, dtype=torch.float64))


def test_zero_block_scale_does_not_change_cpu_output():
    """Both encodings must produce the exact projection instead of 0 and 1."""

    x = _issue_input()
    oracle_weight = _oracle_weight(
        _zero_block_linear(ZERO_BLOCK_SCALES[0]).weight,
        _zero_block_linear(ZERO_BLOCK_SCALES[0]).weight_scale,
        16,
    )
    oracle = x.to(torch.float64) @ oracle_weight.t() + torch.tensor([[BIAS]], dtype=torch.float64)
    assert float(oracle.item()) == 1.0

    outputs = []
    for unused in ZERO_BLOCK_SCALES:
        module = _zero_block_linear(unused)
        with torch.inference_mode():
            actual = module(x)
        assert actual.dtype is torch.float16
        assert actual.shape == (1, 1)
        assert float(actual.item()) == 1.0
        assert torch.equal(actual, oracle.to(torch.float16))
        outputs.append(actual)
    assert torch.equal(outputs[0], outputs[1])


def test_cpu_gate_ignores_an_all_zero_block_scale():
    """An out-of-range scale on an all-zero block must not force the wide path."""

    packed = torch.zeros(1, 16, dtype=torch.uint8)
    packed[0, 0] = 6  # E2M1 4.0, exactly representable with the power-of-two scale
    scale = torch.tensor([[64.0, 1.0e9]], dtype=torch.float32)
    module = TorchFP4Linear(
        in_features=32,
        out_features=1,
        weight=packed,
        weight_scale=scale,
        weight_block_size=16,
        orig_dtype=torch.float16,
    ).eval()
    assert module._cpu_narrow_decode_decidable(torch.float16)
    x = torch.zeros(1, 32, dtype=torch.float16)
    x[0, 0] = 1.0
    with torch.inference_mode():
        actual = module(x)
    assert float(actual.item()) == 256.0


def test_non_degenerate_block_still_decodes_as_before():
    """Mixed nonzero codes keep their decoded values and narrow fast path."""

    packed = torch.tensor(
        [[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0] * 2],
        dtype=torch.uint8,
    )
    scale = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    module = TorchFP4Linear(
        in_features=32,
        out_features=1,
        weight=packed,
        weight_scale=scale,
        weight_block_size=16,
        orig_dtype=torch.float16,
    ).eval()
    expected = _oracle_weight(packed, scale, 16)
    decoded = dequantize_f4_e2m1(
        module.weight,
        scale=module.weight_scale,
        axis=None,
        target_dtype=torch.float64,
    )
    assert torch.equal(decoded, expected)
    # Power-of-two scales are exact in FP16, so the narrow decode is retained.
    assert module._cpu_narrow_decode_decidable(torch.float16)
    x = torch.zeros(1, 32, dtype=torch.float16)
    x[0, 0] = 1.0
    with torch.inference_mode():
        actual = module(x)
    assert torch.equal(actual, (x.to(torch.float64) @ expected.t()).to(torch.float16))
