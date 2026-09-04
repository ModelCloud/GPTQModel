# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.models.definitions.ernie4_5_moe import Ernie4_5_MoeQModel
from gptqmodel.models.definitions.ernie4_5_vl_moe import Ernie4_5_VLMoeQModel


def _collect_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _collect_strings(key)
            yield from _collect_strings(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _collect_strings(item)


@pytest.mark.parametrize("model_cls", [Ernie4_5_MoeQModel, Ernie4_5_VLMoeQModel])
def test_expert_projections_use_up_proj(model_cls):
    tokens = {entry.split(":", 1)[0] for entry in _collect_strings(model_cls.module_tree)}
    # Regression: routed-expert entries carried a "upe_proj" typo while the
    # shared-expert entries correctly used "up_proj" (ERNIE 4.5 MoE modeling
    # names the expert projections gate_proj/up_proj/down_proj).
    assert "upe_proj" not in tokens
    assert "up_proj" in tokens
