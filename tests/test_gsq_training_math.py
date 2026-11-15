# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Independent gradient checks for the staged GSQ scalar relaxation."""

import pytest
import torch

from gptqmodel.quantization.gsq_training import relaxed_scalar_weights


@pytest.mark.parametrize("with_initial", [False, True])
def test_scalar_relaxation_matches_autograd(with_initial):
    torch.manual_seed(19)
    logits = torch.randn(5, 3, 8, dtype=torch.float64, requires_grad=True)
    scales = torch.rand(3, 2, dtype=torch.float64, requires_grad=True) + 0.3
    candidates = torch.arange(-2, 3, dtype=torch.float64)[:, None, None].expand(5, 3, 8)
    groups = torch.arange(8) // 4
    uniform = torch.rand(5, 3, 8, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
    initial = torch.randn(3, 8, dtype=torch.float64) if with_initial else None
    gradient = torch.randn(3, 8, dtype=torch.float64)
    actual = relaxed_scalar_weights(logits, scales, candidates, groups,
                                    uniform=uniform, temperature=0.7,
                                    multiplier=1.8, initial=initial)
    actual_grads = torch.autograd.grad((actual * gradient).sum(), (logits, scales))

    reference_logits = logits.detach().clone().requires_grad_()
    reference_scales = scales.detach().clone().requires_grad_()
    noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
    probabilities = ((reference_logits * 1.8 + noise) / 0.7).softmax(0)
    expected = (probabilities * candidates).sum(0)
    if initial is not None:
        expected = expected + initial
    reference = expected * reference_scales[:, groups]
    reference_grads = torch.autograd.grad((reference * gradient).sum(),
                                          (reference_logits, reference_scales))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-12)


def test_reconstruction_stage_loss_reduces_in_fp32():
    """An FP16 residual of 300 already squares to `inf`; the reduction must not."""

    from gptqmodel.quantization.gsq_training import reconstruction_stage_loss

    class _Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float16))

        def forward(self):
            return self.weight

    module = _Scalar()
    student = {"weight": torch.tensor(300.0, dtype=torch.float16)}
    teacher = {"weight": torch.tensor(0.0, dtype=torch.float16)}
    loss = reconstruction_stage_loss(
        module, (), {}, student_weights=student, teacher_weights=teacher
    )
    assert torch.isfinite(loss), loss
    torch.testing.assert_close(loss, torch.tensor(90000.0), rtol=1e-3, atol=1e-3)


def test_relaxed_scalar_weights_wide_multiplier_stays_finite_in_half_precision():
    """`logits * multiplier` overflows FP16; the exponent must be evaluated wide."""

    from gptqmodel.quantization.gsq_training import relaxed_scalar_weights

    logits = torch.full((5, 3, 8), 200.0, dtype=torch.float16)
    scales = torch.ones(3, 2, dtype=torch.float16)
    candidates = torch.arange(-2, 3, dtype=torch.float16)[:, None, None].expand(5, 3, 8)
    groups = torch.arange(8) // 4
    uniform = torch.full((5, 3, 8), 0.5, dtype=torch.float16)
    result = relaxed_scalar_weights(
        logits, scales, candidates, groups, uniform=uniform,
        temperature=1.0, multiplier=500.0,
    )
    assert torch.isfinite(result).all(), result


def test_train_stage_update_raises_recoverable_nonfinite_error(monkeypatch):
    """A nonfinite stage must raise a Python error, not poison the device context."""

    from gptqmodel.quantization.gsq_training import train_stage_update

    def _device_side_assert(*_args, **_kwargs):
        raise AssertionError("train_stage_update must not call torch._assert_async")

    # The device-side primitive poisons a CUDA context when it fires.
    monkeypatch.setattr(torch, "_assert_async", _device_side_assert)

    class _Quantizer:
        def __init__(self):
            self.logits = torch.zeros(1, requires_grad=True)

        def __call__(self, **_kwargs):
            return self.logits

    quantizer = _Quantizer()
    optimizer = torch.optim.SGD([quantizer.logits], lr=0.1)

    def nonfinite_objective(_batch, weights):
        return weights["w"].sum() * float("nan")

    with pytest.raises(RuntimeError, match="nonfinite"):
        train_stage_update(
            {"w": quantizer},
            optimizer,
            [(None, 4)],
            nonfinite_objective,
            generator=torch.Generator(),
            temperature=1.0,
            multiplier=1.0,
        )

    # A device-side assert would have made the next operation fail instead.
    assert torch.isfinite(torch.ones(1)).all()

    def finite_objective(_batch, weights):
        return weights["w"].sum()

    reported = train_stage_update(
        {"w": quantizer},
        optimizer,
        [(None, 4)],
        finite_objective,
        generator=torch.Generator(),
        temperature=1.0,
        multiplier=1.0,
    )
    assert torch.isfinite(reported).all()


def _stage_quantizer(bits=2, rows=4, columns=8, group_size=8):
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule

    torch.manual_seed(3)
    dense = torch.randn(rows, columns)
    # Scales are per (row, group), so the second dimension is columns // group_size.
    # Sizing them off each group's own maximum keeps the rounded initializer inside
    # the signed grid for every bit-width.
    groups = dense.reshape(rows, columns // group_size, group_size)
    scales = groups.abs().amax(-1) / 2 + .1
    extended = scales.repeat_interleave(group_size, 1)
    weight = (dense / extended).round() * extended
    noise = torch.randn(4 if bits == 2 else 5, rows, columns)
    return GSQScalarTrainingModule(weight, scales, group_size, bits=bits, noise=noise,
                                   logits_dtype=torch.float32)


def test_restore_best_restores_held_out_minimum_not_last_epoch():
    """Best-epoch selection must export an earlier checkpoint, not the last epoch.

    Train and held-out batches carry different targets, so the held-out objective
    is U-shaped: it improves while the weights move toward the held-out target and
    degrades once the training target pulls them past it. Without restoration the
    final epoch is exported regardless.
    """
    from gptqmodel.quantization.gsq_training import fit_reconstruction_stage

    train_target = torch.full((4, 8), 1.0)
    heldout_target = torch.zeros(4, 8)
    train_batches = [[(train_target, 1)]]
    heldout_batches = [[(heldout_target, 1)]]

    def objective(batch, weights):
        return (weights['weight'] - batch).square().mean()

    def run(restore_best):
        quantizer = _stage_quantizer()
        return fit_reconstruction_stage({'weight': quantizer}, train_batches, objective,
                                        epochs=40, assignment_lr=5e-2, scale_lr=5e-2,
                                        validation_batches=heldout_batches,
                                        restore_best=restore_best, seed=11)

    kept = run(False)
    best = run(True)

    # The held-out curve must actually be U-shaped, otherwise the test is vacuous.
    heldout_history = [entry['hard_loss'] for entry in best['validation_history']]
    assert len(heldout_history) == 40
    assert min(heldout_history) < heldout_history[-1], 'held-out loss never degrades'

    assert kept['restored_best_validation_checkpoint'] is False
    assert best['restored_best_validation_checkpoint'] is True
    # The selector must keep the minimum, not the final epoch.
    assert best['best_validation_epoch'] == int(min(range(len(heldout_history)),
                                                    key=lambda i: heldout_history[i]))
    assert best['validation_hard_loss_after'] == pytest.approx(min(heldout_history), rel=0, abs=1e-12)
    assert best['validation_hard_loss_after'] < kept['validation_hard_loss_after']

    # The exported payload must be the restored checkpoint, not the final epoch.
    def exported_heldout_loss(stage):
        return float(objective(heldout_target, stage['weights']))

    assert exported_heldout_loss(best) == pytest.approx(min(heldout_history), rel=0, abs=1e-12)
    assert exported_heldout_loss(kept) > exported_heldout_loss(best)


def test_restore_best_requires_validation_batches():
    from gptqmodel.quantization.gsq_training import fit_reconstruction_stage

    with pytest.raises(ValueError, match='restore_best requires nonempty held-out'):
        fit_reconstruction_stage({'weight': _stage_quantizer()}, [[(None, 1)]],
                                 lambda _b, w: w['weight'].square().mean(),
                                 epochs=2, restore_best=True)


def test_training_config_forwards_restore_best():
    from gptqmodel.quantization.gsq_training_config import GSQTrainingConfig

    kwargs = GSQTrainingConfig(enabled=True, restore_best=True).training_kwargs()
    assert kwargs['restore_best'] is True
    # Default is automatic: restore the best held-out epoch whenever validation
    # batches reach the stage, instead of silently keeping the last epoch.
    assert GSQTrainingConfig(enabled=True).training_kwargs()['restore_best'] is None
    assert GSQTrainingConfig(enabled=True, restore_best=False).training_kwargs()['restore_best'] is False
    with pytest.raises(TypeError, match='restore_best must be boolean or None'):
        GSQTrainingConfig(restore_best='yes')


def test_default_restore_best_uses_held_out_batches_without_explicit_flag():
    """The reported bug: the caller never passed restore_best, so the last epoch won."""
    from gptqmodel.quantization.gsq_training import fit_reconstruction_stage

    train_target = torch.full((4, 8), 1.0)
    heldout_target = torch.zeros(4, 8)

    def objective(batch, weights):
        return (weights['weight'] - batch).square().mean()

    result = fit_reconstruction_stage(
        {'weight': _stage_quantizer()}, [[(train_target, 1)]], objective,
        epochs=40, assignment_lr=5e-2, scale_lr=5e-2,
        validation_batches=[[(heldout_target, 1)]], seed=11)

    heldout_history = [entry['hard_loss'] for entry in result['validation_history']]
    assert min(heldout_history) < heldout_history[-1], 'held-out loss never degrades'
    assert result['restored_best_validation_checkpoint'] is True
    assert result['best_validation_epoch'] == int(min(range(len(heldout_history)),
                                                    key=lambda i: heldout_history[i]))
    assert result['validation_hard_loss_after'] == pytest.approx(min(heldout_history), rel=0, abs=1e-12)


def test_default_restore_best_stays_off_without_held_out_batches():
    from gptqmodel.quantization.gsq_training import fit_reconstruction_stage

    result = fit_reconstruction_stage({'weight': _stage_quantizer()}, [[(torch.ones(4, 8), 1)]],
                                      lambda batch, weights: (weights['weight']-batch).square().mean(),
                                      epochs=2)
    assert result['restored_best_validation_checkpoint'] is False
    assert result['best_validation_epoch'] == -1
