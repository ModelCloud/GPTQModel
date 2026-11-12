# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Quantization scales must fail loudly when they cannot be stored.

Scales are stored as fp16. A 4-bit scale is ``amax / 15``, so any channel whose
largest weight passes ``65504 * 15`` (about 9.8e5) casts to ``inf``; the channel
then dequantizes to NaN and nothing raises. bf16 checkpoints carry that
magnitude legitimately, so this is reachable from a valid file.
"""

import torch

from gptqmodel.nn_modules.qlinear import _scales_for_storage


def test_representable_scales_are_cast_to_fp16():
    scales = torch.tensor([[1e-3], [1.0], [4.0e4]], dtype=torch.float32)
    out = _scales_for_storage(scales, name="layer")
    assert out.dtype == torch.float16
    assert torch.allclose(out.float(), scales, rtol=1e-3), out


def test_largest_representable_scale_is_accepted():
    out = _scales_for_storage(torch.tensor([[65504.0]]), name="edge")
    assert out.item() == 65504.0


def test_scale_above_fp16_range_is_rejected():
    # amax = 2e6 with 4-bit scales is 2e6 / 15 = 1.33e5.
    with pytest_raises_value_error("do not fit fp16"):
        _scales_for_storage(torch.tensor([[2e6 / 15.0]]), name="layer.q_proj.scales")


def test_infinite_scale_is_rejected():
    with pytest_raises_value_error("do not fit fp16"):
        _scales_for_storage(torch.tensor([[float("inf")]]), name="layer")


def test_nan_scale_is_rejected():
    with pytest_raises_value_error("do not fit fp16"):
        _scales_for_storage(torch.tensor([[float("nan")]]), name="layer")


def test_error_names_the_offending_tensor_and_magnitude():
    with pytest_raises_value_error("layer.q_proj.scales") as info:
        _scales_for_storage(torch.tensor([[2e6 / 15.0]]), name="layer.q_proj.scales")
    message = str(info.value)
    assert "1.333e+05" in message
    assert "65504" in message


class pytest_raises_value_error:
    """Tiny local stand-in so the assertions above read in one line each."""

    def __init__(self, needle: str):
        self.needle = needle
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert exc_type is not None, "expected a ValueError"
        assert issubclass(exc_type, ValueError), f"expected ValueError, got {exc_type}"
        assert self.needle in str(exc), f"{self.needle!r} missing from {exc}"
        self.value = exc
        return True
