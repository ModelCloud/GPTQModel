# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fused AVX-512 CPU Viterbi kernel versus the torch oracle and baseline kernel."""

import math

import pytest
import torch

import gptqmodel.quantization.qvq as qvq_module
import gptqmodel.utils.qvq_cpu as qvq_cpu_module
from gptqmodel.utils.qvq_cpu import (
    qvq_cpu_supported,
    qvq_cpu_viterbi,
    qvq_cpu_viterbi_opt,
)


pytestmark = pytest.mark.skipif(not qvq_cpu_supported(), reason="QVQ CPU kernels unavailable")


def _torch_oracle_viterbi(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    shift: int,
    overlap: torch.Tensor | None,
    step_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch port of the ``batched_viterbi_quantize`` fallback oracle."""

    batch_size, step_count, _ = sequences.shape
    state_count = codebook.shape[0]
    trellis_window = int(math.log2(state_count))
    overlap_bits = trellis_window - shift
    work_sequence = sequences.to(torch.float32)
    work_codebook = codebook.to(torch.float32)
    work_step_weights = None if step_weights is None else step_weights.to(torch.float32)
    state_ids = torch.arange(state_count, dtype=torch.long)
    codebook_norm = work_codebook.square().sum(dim=-1)

    def emission(step: int) -> torch.Tensor:
        target = work_sequence[:, step]
        distance = (
            target.square().sum(dim=-1, keepdim=True)
            + codebook_norm.unsqueeze(0)
            - 2 * target @ work_codebook.transpose(0, 1)
        ).clamp_min_(0)
        if work_step_weights is not None:
            distance = distance * work_step_weights[:, step].unsqueeze(1)
        return distance

    costs = emission(0)
    backpointers: list[torch.Tensor] = []
    predecessor_prefix_count = 1 << shift
    predecessor_suffix_count = 1 << (trellis_window - shift)

    if overlap is not None:
        allowed_start = (state_ids.unsqueeze(0) >> shift) == overlap.unsqueeze(1)
        costs = costs.masked_fill(~allowed_start, torch.inf)

    for step in range(1, step_count):
        predecessor_costs = costs.reshape(batch_size, predecessor_prefix_count, predecessor_suffix_count)
        best_cost, best_prefix = predecessor_costs.min(dim=1)
        transitioned = best_cost[:, state_ids >> shift]
        costs = transitioned + emission(step)
        traceback_dtype = torch.int16 if shift <= 15 else torch.int32
        backpointers.append(best_prefix.to(traceback_dtype))

    if overlap is not None and overlap_bits:
        overlap_mask = (1 << overlap_bits) - 1
        allowed_end = (state_ids.unsqueeze(0) & overlap_mask) == overlap.unsqueeze(1)
        costs = costs.masked_fill(~allowed_end, torch.inf)

    end_state = costs.argmin(dim=1)
    path = torch.empty((batch_size, step_count), dtype=torch.long)
    path[:, -1] = end_state
    batch_ids = torch.arange(batch_size, dtype=torch.long)
    for step in range(step_count - 1, 0, -1):
        suffix = path[:, step] >> shift
        prefix = backpointers[step - 1][batch_ids, suffix].to(torch.long)
        path[:, step - 1] = prefix * predecessor_suffix_count + suffix

    return path, costs[batch_ids, end_state]


def _run_case(
    batch_size: int,
    steps: int,
    state_count: int,
    vector_size: int,
    shift: int,
    *,
    seed: int,
    overlap: bool = False,
    step_weights: bool = False,
    scale: float = 1.0,
):
    generator = torch.Generator().manual_seed(seed)
    sequences = torch.randn((batch_size, steps, vector_size), generator=generator, dtype=torch.float32) * scale
    codebook = torch.randn((state_count, vector_size), generator=generator, dtype=torch.float32) * scale

    overlap_i64 = None
    if overlap:
        overlap_bits = int(math.log2(state_count)) - shift
        limit = 1 << overlap_bits if overlap_bits else 1
        overlap_i64 = torch.randint(0, limit, (batch_size,), generator=generator, dtype=torch.int64)

    weights = None
    if step_weights:
        weights = torch.rand((batch_size, steps), generator=generator, dtype=torch.float32)

    expected_states, expected_se = _torch_oracle_viterbi(sequences, codebook, shift, overlap_i64, weights)
    actual_states, actual_se = qvq_cpu_viterbi_opt(
        sequences,
        codebook,
        shift,
        overlap=overlap_i64,
        step_weights=weights,
    )

    assert actual_states.shape == expected_states.shape
    assert actual_states.dtype == torch.int64
    assert torch.equal(actual_states, expected_states), f"states diverged for seed={seed}"
    torch.testing.assert_close(actual_se, expected_se, atol=2e-4, rtol=2e-5)

    baseline_states, baseline_se = qvq_cpu_viterbi(
        sequences.contiguous(),
        codebook.contiguous(),
        shift,
        overlap=overlap_i64,
        step_weights=weights,
    )
    assert torch.equal(actual_states, baseline_states), f"states differ from baseline for seed={seed}"
    # MEASURED matrix-calibrated, not universal: a standardized 96-config matrix
    # had max delta 1.1205673217773438e-5, so 1.25e-5 provides about 11.6%
    # headroom. A fixed absolute bound cannot remain valid as step count,
    # weights, or input magnitude changes.
    torch.testing.assert_close(actual_se, baseline_se, atol=1.25e-5, rtol=0)


@pytest.mark.parametrize("vector_size", (2, 4))
@pytest.mark.parametrize("shift", (2, 3, 8))
def test_qvq_viterbi_opt_matches_torch_oracle(vector_size, shift):
    _run_case(3, 9, 1 << 10, vector_size, shift, seed=100 + shift * 10 + vector_size)


def test_qvq_viterbi_opt_full_window_matches_oracle():
    _run_case(2, 16, 1 << 16, 2, 14, seed=7)


def test_qvq_viterbi_opt_w4_int32_backpointers():
    shift = 16
    _run_case(1, 5, 1 << 16, 4, shift, seed=11)


def test_qvq_viterbi_opt_overlap_and_weights():
    _run_case(4, 33, 1 << 12, 2, 5, seed=23, overlap=True)
    _run_case(4, 33, 1 << 12, 2, 5, seed=24, step_weights=True)
    _run_case(4, 33, 1 << 12, 2, 5, seed=25, overlap=True, step_weights=True)


def test_qvq_viterbi_opt_overlap_full_shift_noop():
    # shift == trellis window leaves zero overlap bits; masks must be no-ops.
    _run_case(2, 8, 1 << 8, 2, 8, seed=31, overlap=True)


def test_qvq_viterbi_opt_single_batch_long_steps():
    _run_case(1, 128, 1 << 16, 2, 4, seed=41)


def test_qvq_viterbi_opt_large_magnitudes():
    _run_case(2, 9, 1 << 10, 2, 4, seed=51, scale=512.0)


def test_qvq_viterbi_opt_tied_codebook_rows_pick_first_index():
    generator = torch.Generator().manual_seed(61)
    sequences = torch.randn((2, 7, 2), generator=generator, dtype=torch.float32)
    base = torch.randn((1 << 8, 2), generator=generator, dtype=torch.float32)
    codebook = base[torch.randint(0, base.shape[0], (1 << 8,), generator=generator)].contiguous()

    expected_states, expected_se = _torch_oracle_viterbi(sequences, codebook, 4, None, None)
    actual_states, actual_se = qvq_cpu_viterbi_opt(sequences, codebook, 4)
    assert torch.equal(actual_states, expected_states)
    torch.testing.assert_close(actual_se, expected_se, atol=2e-4, rtol=2e-5)


def test_qvq_viterbi_opt_repeated_calls_are_deterministic():
    generator = torch.Generator().manual_seed(71)
    sequences = torch.randn((3, 12, 2), generator=generator, dtype=torch.float32)
    codebook = torch.randn((1 << 12, 2), generator=generator, dtype=torch.float32)
    first = qvq_cpu_viterbi_opt(sequences, codebook, 6)
    for _ in range(3):
        repeat = qvq_cpu_viterbi_opt(sequences, codebook, 6)
        assert torch.equal(first[0], repeat[0])
        assert torch.equal(first[1], repeat[1])


@pytest.mark.xfail(
    strict=False,
    reason="Known qvq_cpu_viterbi_opt discrete divergence from qvq_cpu_viterbi at V=2, transition_bits=16, "
    "steps=32, batch=128; see the 2026-08-24 opt-in Viterbi discrete-divergence defect record in "
    "CPU_KERNEL_LOG.md",
)
def test_qvq_viterbi_opt_known_v2_rate8_large_batch_divergence():
    vector_size = 2
    steps = 32
    generator = torch.Generator().manual_seed(2026082400 + 100 * vector_size + steps)
    sequences = torch.randn((128, steps, vector_size), generator=generator, dtype=torch.float32)
    codebook = torch.randn((1 << 16, vector_size), generator=generator, dtype=torch.float32)

    production_states, _ = qvq_cpu_viterbi(sequences, codebook, transition_bits=16)
    opt_states, _ = qvq_cpu_viterbi_opt(sequences, codebook, transition_bits=16)

    assert torch.equal(opt_states, production_states)


def test_qvq_viterbi_production_cpu_dispatch_excludes_opt(monkeypatch):
    calls = []

    def production_viterbi(sequences, codebook, transition_bits, overlap=None, step_weights=None):
        calls.append((transition_bits, overlap, step_weights))
        return torch.zeros(sequences.shape[:2], dtype=torch.long), torch.zeros(sequences.shape[0])

    def forbidden_opt_viterbi(*args, **kwargs):
        pytest.fail("production CPU dispatch called qvq_cpu_viterbi_opt")

    monkeypatch.setattr(qvq_cpu_module, "qvq_cpu_viterbi", production_viterbi)
    monkeypatch.setattr(qvq_cpu_module, "qvq_cpu_viterbi_opt", forbidden_opt_viterbi)
    sequences = torch.zeros((1, 2, 2), dtype=torch.float32)
    codebook = torch.zeros((1 << 16, 2), dtype=torch.float32)

    result = qvq_module.batched_viterbi_quantize(sequences, codebook, bits=8)

    assert len(calls) == 1
    assert calls[0] == (16, None, None)
    assert torch.equal(result.states, torch.zeros((1, 2), dtype=torch.long))


def test_qvq_viterbi_opt_rejects_out_of_range_transition_bits():
    sequences = torch.zeros((2, 4, 2))
    codebook = torch.zeros((1 << 8, 2))
    with pytest.raises(RuntimeError, match="transition_bits out of range"):
        qvq_cpu_viterbi_opt(sequences, codebook, 99)


def test_qvq_viterbi_opt_rejects_bad_shapes():
    sequences = torch.zeros((2, 4, 3))
    codebook = torch.zeros((1 << 8, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        qvq_cpu_viterbi_opt(sequences, codebook, 4)
    with pytest.raises(ValueError, match="requires CPU tensors"):
        qvq_cpu_viterbi_opt(sequences.to("meta"), codebook, 4)


def test_qvq_viterbi_opt_tied_end_state_picks_lowest_index():
    """Regression: the vectorized final argmin must break exact cost ties by
    lowest global state id.

    The fixture below produces two final states (18554 and 18656) whose costs
    are bit-identical; the winner must be 18554, matching the baseline kernel
    and torch.argmin.
    """

    generator = torch.Generator().manual_seed(90_002)
    sequences = torch.randn((64, 128, 2), generator=generator)[54:55].contiguous()
    codebook = torch.randn((1 << 16, 2), generator=generator).contiguous()

    base_states, base_se = qvq_cpu_viterbi(sequences, codebook, 14)
    opt_states, opt_se = qvq_cpu_viterbi_opt(sequences, codebook, 14)

    assert torch.equal(opt_states, base_states)
    torch.testing.assert_close(opt_se, base_se, atol=1.25e-5, rtol=0)
    assert base_states[0, -1].item() == 18554
    assert opt_states[0, -1].item() == 18554
