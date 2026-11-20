# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""#5693: the reported GSQ baseline must not depend on candidate chunking.

A zero-energy output row used to be normalized on its own when the candidate
bank split it into a separate chunk, which added a fictitious ``+1`` to the
aggregate teacher energy.  The near-zero convention belongs to the layer
total, not to each chunk.
"""

import pytest
import torch

from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.gsq_scalar import refine_gptq_scalar


def _refine(weight, target, budget):
    rows, columns = weight.shape
    scales = torch.ones((rows, 1))
    zeros = scales * 8
    config = GSQConfig(
        enabled=True,
        steps=1,
        candidates=4,
        learn_scales=False,
        max_candidate_bytes=budget,
    )
    return refine_gptq_scalar(
        weight,
        scales,
        zeros,
        torch.zeros(columns, dtype=torch.int32),
        target=target,
        hessian=torch.eye(columns),
        bits=4,
        config=config,
    )


def _fp64_normalized(weight, target):
    error = weight.double() - target.double()
    return (error.square().sum() / target.double().square().sum()).item()


def _fp64_energy(weight, target):
    return (weight.double() - target.double()).square().sum().item()


def test_baseline_is_chunk_invariant_with_a_zero_energy_row():
    weight = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    # 4 candidates * 2 columns * 4 bytes = 32 bytes per row: the 32-byte
    # budget puts each row in its own chunk, 64 bytes keeps them together.
    chunked = _refine(weight, target, 32)
    whole = _refine(weight, target, 64)

    reference = _fp64_normalized(weight, target)
    assert reference == pytest.approx(5.0)
    for result in (chunked, whole):
        assert result.before == pytest.approx(reference, rel=1e-6)
        assert result.after == pytest.approx(
            _fp64_normalized(result.weight, target), rel=1e-6
        )
    assert chunked.before == whole.before
    assert chunked.after == whole.after
    assert chunked.history == whole.history
    assert torch.equal(chunked.weight, whole.weight)


def test_zero_teacher_stays_chunk_invariant():
    weight = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.zeros_like(weight)
    chunked = _refine(weight, target, 32)
    whole = _refine(weight, target, 64)

    # With no teacher energy the objective is the unnormalized quadratic, and
    # the near-zero convention applies once to the layer total.
    reference = _fp64_energy(weight, target)
    assert reference == pytest.approx(2.0)
    for result in (chunked, whole):
        assert result.before == pytest.approx(reference, rel=1e-6)
        assert result.after == pytest.approx(
            _fp64_energy(result.weight, target), rel=1e-6
        )
    assert chunked.before == whole.before
    assert chunked.after == whole.after
    assert chunked.history == whole.history
    assert torch.equal(chunked.weight, whole.weight)
