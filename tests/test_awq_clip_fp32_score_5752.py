# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ clip search must score in FP32 for a half-precision checkpoint (#5752).

The search forms a dot product, a residual, its square and a mean before it
compares candidates. In FP16 several distinct positive losses round to the same
value (usually zero), so the first-minimum tie break kept an arbitrarily early
grid step: on the fixture below it selected the unclipped bound with a 52.8x
larger reconstruction MSE than the true minimum. Only the score is widened —
the clamped weights, the quantized candidates and the stored bounds stay in the
checkpoint dtype.

The oracle here is deliberately independent of the scoring code: it reuses the
module's own candidate quantizer (so a quantizer difference cannot mask a
scoring bug) and evaluates the reconstruction error in float64.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import FORMAT, AWQConfig, CalibrationConfig


BITS = 4
GROUP_SIZE = 32
N_GRID = 20
MAX_SHRINK = 0.5


def _processor(sym: bool) -> AWQProcessor:
    return AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(bits=BITS, group_size=GROUP_SIZE, format=FORMAT.GEMM, sym=sym),
        calibration=CalibrationConfig(),
        gptq_model=None,
        model=None,
        calculate_w_wq_diff=False,
    )


def _bounds(w: torch.Tensor) -> torch.Tensor:
    """The clip grid ``_compute_best_clip`` searches, per (row, group).

    The search takes the absolute maximum inside each group, not across the
    whole row, so a row with two groups gets two independent grids.
    """

    steps = int(MAX_SHRINK * N_GRID)
    co, ci = w.shape
    n_group = ci // GROUP_SIZE
    top = w.reshape(co, n_group, GROUP_SIZE).abs().amax(dim=-1).float()
    factors = torch.tensor([1 - i / N_GRID for i in range(steps)])
    return (top.view(1, co, n_group) * factors.view(-1, 1, 1)).to(w.dtype)  # [steps, co, ng]


def _wide_errors(processor, w, x, bounds, sym):
    """Reconstruction MSE per (candidate, output row, group) in float64.

    The candidate weights come from the module's own quantizer at the same
    width, group size and symmetry, so this isolates the *scoring* precision
    from any quantizer behavior.
    """

    co, ci = w.shape
    n_group = ci // GROUP_SIZE
    wq = w.reshape(co, n_group, GROUP_SIZE)
    xd = x.reshape(-1, n_group, GROUP_SIZE).double()
    wd = wq.double()
    baseline = torch.einsum("ngk,rgk->rng", xd, wd)
    errors = []
    for step in range(bounds.shape[0]):
        bound = bounds[step].reshape(co, n_group, 1)
        clipped = wq.clamp(-bound, bound).reshape(co, ci)
        quantized, _, _ = processor.pseudo_quantize_tensor(
            clipped,
            bits=BITS,
            group_size=GROUP_SIZE,
            format=FORMAT.GEMM,
            sym=sym,
        )
        qd = quantized.double().reshape(co, n_group, GROUP_SIZE)
        errors.append((torch.einsum("ngk,rgk->rng", xd, qd) - baseline).square().mean(dim=1))
    return torch.stack(errors)  # [steps, co, n_group]


def _selected_steps(got: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Index of the chosen bound per (output row, group)."""

    co, n_group = got.shape[0], got.shape[1]
    flat = got.reshape(co, n_group)
    matches = bounds == flat.unsqueeze(0)  # [steps, co, ng]
    assert matches.sum(dim=0).min() == 1, "every row must select one grid bound"
    return matches.float().argmax(dim=0)


def _oracle_steps(processor, w, x, bounds, sym):
    errors = _wide_errors(processor, w, x, bounds, sym)
    return errors.argmin(dim=0), errors


def _cancellation_fixture(dtype=torch.float16):
    """The #5752 fixture: a wide bound is not the minimum reconstruction error."""

    w = torch.zeros(8, 32, dtype=dtype)
    w[:, :2] = torch.tensor([0.01, 0.0013])
    x = torch.zeros(3, 32, dtype=dtype)
    x[:, :2] = torch.tensor([[0.001, 1.0], [0.002, 0.8], [-0.001, 1.25]])
    return w, x


@pytest.mark.parametrize("sym", [False, True])
def test_fixture_selects_the_oracle_bound(sym):
    processor = _processor(sym)
    w, x = _cancellation_fixture()
    bounds = _bounds(w)

    got = processor._compute_best_clip(w, x)
    expected_steps, errors = _oracle_steps(processor, w, x, bounds, sym)
    actual_steps = _selected_steps(got, bounds)

    assert torch.equal(actual_steps, expected_steps)
    # The selected bound must reach the oracle's minimum, not merely a
    # plausible step: this is the 52.8x error the issue reported.
    chosen = errors.gather(0, actual_steps.unsqueeze(0)).squeeze(0)
    best = errors.min(dim=0).values
    torch.testing.assert_close(chosen, best, rtol=0, atol=0)


@pytest.mark.parametrize("sym", [False, True])
def test_unclipped_bound_is_not_kept_when_it_is_not_the_minimum(sym):
    # Regression shape: the buggy scoring rounded every candidate's loss to
    # zero and the first-minimum tie break kept step 0, the unclipped bound.
    processor = _processor(sym)
    w, x = _cancellation_fixture()
    bounds = _bounds(w)

    got = processor._compute_best_clip(w, x)

    assert not torch.equal(got[0, 0, 0], bounds[0][0])


@pytest.mark.parametrize("multiplier", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("sym", [False, True])
def test_activation_scale_above_and_below_the_fp16_error_floor(sym, multiplier):
    processor = _processor(sym)
    w, x = _cancellation_fixture()
    x = x * multiplier
    bounds = _bounds(w)

    got = processor._compute_best_clip(w, x)
    expected_steps, _ = _oracle_steps(processor, w, x, bounds, sym)

    assert torch.equal(_selected_steps(got, bounds), expected_steps)


@pytest.mark.parametrize("sym", [False, True])
def test_batched_rows_and_groups_match_the_oracle(sym):
    torch.manual_seed(5752)
    processor = _processor(sym)
    w = (torch.randn(8, 64, dtype=torch.float16) * 0.02).contiguous()
    w[:, :2] *= 8  # give the first group a distinct scale from the second
    x = torch.randn(6, 64, dtype=torch.float16)
    bounds = _bounds(w)

    got = processor._compute_best_clip(w, x)
    expected_steps, _ = _oracle_steps(processor, w, x, bounds, sym)

    assert got.shape == (8, 2, 1)
    assert torch.equal(_selected_steps(got, bounds), expected_steps)


@pytest.mark.parametrize("sym", [False, True])
def test_bfloat16_weights_match_the_oracle(sym):
    torch.manual_seed(5753)
    processor = _processor(sym)
    w = (torch.randn(4, 32, dtype=torch.bfloat16) * 0.05).contiguous()
    w[:, :2] *= 6
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    bounds = _bounds(w)

    got = processor._compute_best_clip(w, x)
    expected_steps, _ = _oracle_steps(processor, w, x, bounds, sym)

    assert torch.equal(_selected_steps(got, bounds), expected_steps)


@pytest.mark.parametrize("sym", [False, True])
def test_float32_weights_keep_the_existing_precision(sym):
    # Guard: FP32 scoring already ran in FP32, so the selection must be the
    # same as the wide oracle rather than merely "close".
    torch.manual_seed(5754)
    processor = _processor(sym)
    w = torch.randn(4, 32) * 0.05
    w[:, :2] *= 6
    x = torch.randn(5, 32)
    bounds = _bounds(w)

    got = processor._compute_best_clip(w, x)
    expected_steps, _ = _oracle_steps(processor, w, x, bounds, sym)

    assert got.dtype == torch.float32
    assert torch.equal(_selected_steps(got, bounds), expected_steps)


@pytest.mark.parametrize("sym", [False, True])
def test_returned_bounds_stay_in_the_checkpoint_dtype(sym):
    processor = _processor(sym)
    w, x = _cancellation_fixture()

    got = processor._compute_best_clip(w, x)

    assert got.dtype is torch.float16
    # Every returned bound must still be a grid value, not an FP32 score.
    assert torch.equal(got, got.to(torch.float16))
