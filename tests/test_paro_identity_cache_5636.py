# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The Paro identity shortcut must observe optimizer updates (#5636).

``_rotate_inputs`` used to skip the rotation whenever grad was disabled, even
though the rotation operands were still trainable. An optimizer step changes
them without clearing the cached identity flag, so a no-grad forward kept
returning the pre-step projection.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch


_FIXTURE = runpy.run_path(
    str(Path(__file__).with_name("test_paro_identity_gradients_5419.py"))
)
_make_identity_module = _FIXTURE["_make_identity_module"]


def _one_hot() -> torch.Tensor:
    x = torch.zeros(1, 16)
    x[0, 0] = 1.0
    return x


def test_no_grad_forward_observes_the_optimizer_step() -> None:
    module = _make_identity_module(requires_grad=True).train()
    x = _one_hot()
    optimizer = torch.optim.SGD([module.channel_scales], lr=0.5)

    module(x)[0, 0].backward()
    optimizer.step()

    assert module.channel_scales[0, 0].item() == pytest.approx(0.5)
    assert module(x)[0, 0].item() == pytest.approx(0.5)
    with torch.no_grad():
        assert module(x)[0, 0].item() == pytest.approx(0.5)
        # Clearing the cache must not change the value again.
        module.clear_weight_cache()
        assert module(x)[0, 0].item() == pytest.approx(0.5)


def test_no_grad_forward_before_any_step_stays_identity() -> None:
    module = _make_identity_module(requires_grad=True).train()
    with torch.no_grad():
        assert module(_one_hot())[0, 0].item() == pytest.approx(1.0)


def test_frozen_identity_module_keeps_the_call_free_shortcut() -> None:
    module = _make_identity_module(requires_grad=False).train()
    x = _one_hot()
    with torch.no_grad():
        assert module._rotate_inputs(x) is x
        assert module(x)[0, 0].item() == pytest.approx(1.0)
