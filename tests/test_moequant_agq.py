# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Affinity-Guided Quantization: Hessian math and routed-row pairing."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.moe_affinity import MoEAffinityCapture


pytestmark = [pytest.mark.cpu]


ROWS = 11
COLS = 16
OUT_FEATURES = 8


def _fixture(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    activations = torch.randn(ROWS, COLS, generator=generator)
    weights = torch.rand(ROWS, generator=generator) + 0.1
    return activations, weights


def _task(*, chunk_size=None) -> GPTQ:
    module = nn.Linear(COLS, OUT_FEATURES, bias=False)
    hessian = HessianConfig(sink_row_cap_median_multiple=None)
    if chunk_size is not None:
        hessian.chunk_size = chunk_size
    task = GPTQ(
        module,
        QuantizeConfig(bits=4, group_size=8, desc_act=False, sym=True, hessian=hessian),
    )
    task.quantizer.configure(perchannel=True)
    return task


def test_agq_hessian_matches_float64_oracle():
    activations, affinity = _fixture()
    task = _task()
    actual = task.compute_hessian_xtx(activations, row_weights=affinity)
    oracle = activations.T.to(torch.float64) @ (affinity.double()[:, None] * activations.double())
    relative = (actual.double() - oracle).norm() / oracle.norm()
    assert float(relative) < 1e-6


def test_agq_hessian_matches_oracle_in_chunked_path():
    activations, affinity = _fixture(seed=3)
    task = _task(chunk_size=3)
    actual = task.compute_hessian_xtx(activations, row_weights=affinity)
    oracle = activations.T.to(torch.float64) @ (affinity.double()[:, None] * activations.double())
    relative = (actual.double() - oracle).norm() / oracle.norm()
    assert float(relative) < 1e-6


def test_agq_finalize_matches_documented_normalisation():
    activations, affinity = _fixture(seed=5)
    task = _task()
    task.add_batch(activations, None, row_weights=affinity)
    actual = task.finalize_hessian()
    oracle = activations.T.to(torch.float64) @ (affinity.double()[:, None] * activations.double())
    oracle = oracle * (2.0 / ROWS)
    relative = (actual.double() - oracle).norm() / oracle.norm()
    assert task.nsamples == ROWS
    assert float(relative) < 1e-6


def test_agq_hessian_composes_with_sink_row_cap():
    generator = torch.Generator().manual_seed(11)
    activations = torch.randn(ROWS, COLS, generator=generator)
    activations[0] *= 100.0
    affinity = torch.rand(ROWS, generator=generator) + 0.1
    task = _cap_task()
    cap = task.resolve_sink_row_cap_scale(activations, None)
    assert cap is not None
    assert float(cap.min()) < 1.0

    actual = task.compute_hessian_xtx(activations, row_weights=affinity)
    scaled = activations * cap.unsqueeze(1)
    oracle = scaled.T.to(torch.float64) @ (
        affinity.double()[:, None] * scaled.double()
    )
    relative = (actual.double() - oracle).norm() / oracle.norm()
    assert float(relative) < 1e-6


def test_agq_all_zero_affinity_skips_batch():
    activations, _ = _fixture()
    task = _task()
    task.add_batch(activations, None, row_weights=torch.zeros(ROWS))
    assert task.nsamples == 0


def test_agq_rejects_bad_row_weights():
    activations, affinity = _fixture()
    task = _task()
    with pytest.raises(ValueError, match="one entry per"):
        task.add_batch(activations, None, row_weights=affinity[:-1])
    with pytest.raises(ValueError, match="non-negative"):
        task.add_batch(activations, None, row_weights=-affinity)
    bad = affinity.clone()
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        task.add_batch(activations, None, row_weights=bad)


def _cap_task() -> GPTQ:
    module = nn.Linear(COLS, OUT_FEATURES, bias=False)
    return GPTQ(
        module,
        QuantizeConfig(
            bits=4,
            group_size=8,
            desc_act=True,
            act_group_aware=False,
            sym=True,
            hessian=HessianConfig(sink_row_cap_median_multiple=4.0),
        ),
    )


def _routing_fixture():
    torch.manual_seed(0)
    hidden = torch.randn(9, 6)
    index = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 3],
            [0, 2],
            [3, 1],
            [2, 2],
            [1, 0],
            [3, 3],
        ]
    )
    weights = torch.rand(9, 2) + 0.1
    return hidden, index, weights


def _expert_rows(hidden, index, weights, expert):
    mask = torch.nn.functional.one_hot(index, num_classes=4).permute(2, 1, 0)
    positions, tokens = torch.where(mask[expert])
    return hidden[tokens], weights[tokens, positions], tokens, positions


class _Container(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts


def test_agq_capture_pairs_slot_major_rows():
    hidden, index, weights = _routing_fixture()
    capture = MoEAffinityCapture()
    capture.register_expert("blk.2.gate_proj", "blk", 2)
    capture.record("blk", _Container(4), (hidden, index, weights), {})

    inp, expected, _, _ = _expert_rows(hidden, index, weights, 2)
    actual = capture.weights_for("blk.2.gate_proj", inp)
    torch.testing.assert_close(actual, expected)


def test_agq_capture_accepts_singleton_expert_axis():
    hidden, index, weights = _routing_fixture()
    capture = MoEAffinityCapture()
    capture.register_expert("blk.2.gate_proj", "blk", 2)
    capture.record(
        "blk",
        _Container(4),
        (hidden, index.unsqueeze(-1), weights.unsqueeze(-1)),
        {},
    )

    inp, expected, _, _ = _expert_rows(hidden, index, weights, 2)
    actual = capture.weights_for("blk.2.gate_proj", inp)
    torch.testing.assert_close(actual, expected)


def test_agq_capture_accepts_token_major_rows():
    hidden, index, weights = _routing_fixture()
    capture = MoEAffinityCapture()
    capture.register_expert("blk.2.gate_proj", "blk", 2)
    capture.record("blk", _Container(4), (hidden, index, weights), {})

    tokens, positions = torch.where(index == 2)
    inp = hidden[tokens]
    expected = weights[tokens, positions]
    actual = capture.weights_for("blk.2.gate_proj", inp)
    torch.testing.assert_close(actual, expected)


def test_agq_capture_down_proj_checks_only_row_count():
    hidden, index, weights = _routing_fixture()
    capture = MoEAffinityCapture()
    capture.register_expert("blk.2.down_proj", "blk", 2)
    capture.record("blk", _Container(4), (hidden, index, weights), {})

    _, expected, _, _ = _expert_rows(hidden, index, weights, 2)
    intermediate = torch.randn(expected.numel(), 3)
    actual = capture.weights_for("blk.2.down_proj", intermediate)
    torch.testing.assert_close(actual, expected)


def test_agq_capture_rejects_wrong_row_order():
    hidden, index, weights = _routing_fixture()
    capture = MoEAffinityCapture()
    capture.register_expert("blk.2.gate_proj", "blk", 2)
    capture.record("blk", _Container(4), (hidden, index, weights), {})

    inp, _, _, _ = _expert_rows(hidden, index, weights, 2)
    with pytest.raises(RuntimeError, match="routing order"):
        capture.weights_for("blk.2.gate_proj", inp + 1.0)


def test_agq_capture_fails_closed_without_routing():
    hidden, index, weights = _routing_fixture()
    capture = MoEAffinityCapture()
    capture.register_expert("blk.2.gate_proj", "blk", 2)
    inp, _, _, _ = _expert_rows(hidden, index, weights, 2)
    with pytest.raises(RuntimeError, match="routing capture"):
        capture.weights_for("blk.2.gate_proj", inp)

    capture.record("blk", _Container(4), (hidden, index, weights), {})
    capture.clear("blk")
    with pytest.raises(RuntimeError, match="routing capture"):
        capture.weights_for("blk.2.gate_proj", inp)


def test_agq_capture_rejects_negative_or_nonfinite_affinity():
    hidden, index, weights = _routing_fixture()
    capture = MoEAffinityCapture()
    capture.register_expert("blk.2.gate_proj", "blk", 2)
    inp, _, tokens, positions = _expert_rows(hidden, index, weights, 2)

    negative = weights.clone()
    negative[tokens[0], positions[0]] = -1.0
    capture.record("blk", _Container(4), (hidden, index, negative), {})
    with pytest.raises(RuntimeError, match="negative"):
        capture.weights_for("blk.2.gate_proj", inp)

    nonfinite = weights.clone()
    nonfinite[tokens[0], positions[0]] = float("inf")
    capture.record("blk", _Container(4), (hidden, index, nonfinite), {})
    with pytest.raises(RuntimeError, match="non-finite"):
        capture.weights_for("blk.2.gate_proj", inp)
