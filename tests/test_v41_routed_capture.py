# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for V4.1 routed capture without the optional model fork."""

import pytest
import torch
from torch import nn

from gptqmodel.utils.v41_capture import V41Capture
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)
        self.limit = 0


def _block():
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.experts = nn.ModuleList([_Expert(), _Expert()])
    return block


def test_direct_capture_rejects_duplicate_routes_before_accumulating():
    batch = V41RoutedBatch(
        hidden=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        logits=torch.tensor([[0.9, 0.1]]),
        weights=torch.tensor([[0.5, 0.5]]),
        indices=torch.tensor([[0, 0]]),
    )
    capture = V41Capture(_block(), [0], device="cpu", phase="gate_up")

    with pytest.raises(ValueError, match="duplicate expert assignments"):
        capture.capture_routed(batch)

    assert capture.failed
    with pytest.raises(RuntimeError, match="failed capture"):
        capture.projection(0, "w1")


def test_direct_capture_matches_independent_raw_hessian():
    batch = V41RoutedBatch(
        hidden=torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 1.0, 1.0]]),
        logits=torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
        weights=torch.ones(2, 1),
        indices=torch.tensor([[0], [1]]),
    )
    capture = V41Capture(_block(), [0], device="cpu", phase="gate_up")
    capture.capture_routed(batch)

    result, evidence = capture.projection(0, "w1")
    expected = batch.hidden[:1].T @ batch.hidden[:1]
    torch.testing.assert_close(result["H"], expected, rtol=0, atol=0)
    assert result["count"] == evidence["natural_rows"] == 1
