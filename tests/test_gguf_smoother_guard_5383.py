# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A direct packer must not advertise a smoother it cannot apply (#5383).

``GGUFConfig`` accepted ``smoother=SmoothMSE()`` and recorded it in
``meta.preprocessors``, but ``smooth_block()`` has no MSE branch -- its search
needs the quantizer's code grid. The GGUF and BitsAndBytes packers therefore
saved byte-identical weights to the no-smoother run while the metadata claimed a
preprocessing step. The shared helper now fails closed instead.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import _apply_optional_smoother
from gptqmodel.quantization.config import Fallback, FallbackStrategy, SmoothMAD, SmoothMSE
from gptqmodel.quantization.fallback_smooth import smooth_block


def _fixture() -> torch.Tensor:
    """Seeded weight with one outlier per row, as in the report."""
    generator = torch.Generator().manual_seed(5383)
    weight = torch.randn(16, 256, generator=generator)
    weight[:, 3] = 10.0
    return weight


def test_direct_packer_rejects_smooth_mse():
    with pytest.raises(ValueError, match="SmoothMSE"):
        _apply_optional_smoother(_fixture(), smooth=SmoothMSE(), group_size=128)


def test_smooth_block_rejects_smooth_mse():
    fallback = Fallback(strategy=FallbackStrategy.RTN, smooth=SmoothMSE())
    with pytest.raises(ValueError, match="MSE search"):
        smooth_block(_fixture(), fallback, group_size=128)


def test_direct_packer_still_applies_a_supported_smoother():
    weight = _fixture()
    smoothed = _apply_optional_smoother(weight, smooth=SmoothMAD(k=1.0), group_size=128)
    assert not torch.equal(smoothed, weight)


def test_no_smoother_leaves_the_weight_untouched():
    weight = _fixture()
    assert torch.equal(_apply_optional_smoother(weight, smooth=None, group_size=128), weight)


def test_small_groups_still_skip_a_smoother_before_the_support_check():
    """The ``group_size_threshold`` short-circuit is a documented no-op, not a bug."""
    weight = _fixture()
    # SmoothMAD defaults to a 128-token threshold; a 32-wide block must not raise
    # and must not be clipped.
    smoothed = _apply_optional_smoother(weight, smooth=SmoothMAD(k=1.0), group_size=32)
    assert torch.equal(smoothed, weight)
