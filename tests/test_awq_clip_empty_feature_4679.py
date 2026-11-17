# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ clip search must tolerate a projection with no calibration samples (#4679).

When a subset's early stop starved a sibling (Falcon-H1 ``gate_proj``), the AWQ
clip search received a ``(0,)`` feature and crashed inside the reshape with
``cannot reshape tensor of 0 elements into shape [-1, 0]``. The module now logs
and skips the clip search instead, and the low-level helper refuses an empty
feature with an explicit error.
"""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from torch import nn

from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig
from tests.test_awq_clip_consistency import _ClipTestAWQProcessor


pytestmark = [pytest.mark.cpu]


def _processor():
    return _ClipTestAWQProcessor(
        QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32)
    )


def test_clip_search_skips_a_module_with_no_calibration_samples():
    layer = nn.Module()
    layer.proj = nn.Linear(32, 16, bias=False)

    clip_list = _processor()._search_best_clip(
        layer, {"proj": layer.proj}, {"proj": torch.zeros((0, 32))}
    )

    assert clip_list == []


def test_clip_search_refuses_an_empty_feature_with_an_explicit_error():
    weight = torch.randn(16, 32, generator=torch.Generator().manual_seed(4679))

    with pytest.raises(ValueError, match="at least one calibration token row"):
        _processor()._compute_best_clip(weight, torch.zeros((0, 32)))
