# SPDX-License-Identifier: Apache-2.0
"""Device-arrival order must not change the Hessian used after a restart."""

from itertools import permutations

import pytest
import torch

from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


@pytest.mark.parametrize("arrival", list(permutations(range(3))))
def test_hessian_reduction_uses_device_order_not_arrival(arrival):
    # Each matrix is positive semidefinite and can be an actual X.T @ X.
    # Off-diagonal cancellation exposes non-associative floating-point sums.
    # CPU payloads with CUDA identity keys simulate arrival order without
    # pretending this is a physical multi-GPU integration test.
    partials = [
        torch.tensor([[1e20, 1e20], [1e20, 1e20]], dtype=torch.float32),
        torch.tensor([[1e20, -1e20], [-1e20, 1e20]], dtype=torch.float32),
        torch.ones(2, 2, dtype=torch.float32),
    ]
    task = GPTQ(
        torch.nn.Linear(2, 2, bias=False),
        QuantizeConfig(group_size=-1, act_group_aware=False),
    )
    task._device_hessian_partials = {
        torch.device(f"cuda:{index}"): partials[index] for index in arrival
    }
    task._device_sample_counts = {torch.device(f"cuda:{index}"): 1 for index in arrival}
    task._hessian_dirty = True
    task.materialize_global_hessian(torch.device("cpu"))
    expected = torch.zeros(2, 2)
    for partial in partials:
        expected.add_(partial)
    expected.mul_(2.0 / 3.0)
    assert task.nsamples == 3
    assert torch.equal(task.H.view(torch.uint8), expected.view(torch.uint8))
