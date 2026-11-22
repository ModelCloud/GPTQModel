# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""QQQ must honour ``damp_auto_increment`` when the Cholesky fails (#5858).

``QQQConfig`` accepts and serializes ``damp_auto_increment``, but the solve
made exactly one damping attempt: a rank-deficient Hessian raised from
``torch.linalg.cholesky`` instead of retrying at the next candidate. QQQ now
routes through the shared GPTQ recovery policy while keeping its own factor
convention, so the applied damping is the one the config asked for.
"""

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.qqq import QQQ


ROWS, COLUMNS = 32, 64
# Rank one: positive semidefinite and singular, so the undamped Cholesky fails
# and only an added diagonal can recover it.
RANK_ONE = torch.ones(COLUMNS, COLUMNS, dtype=torch.float32)
# Eigenvalues 3 and -1 with a nonzero diagonal, so dead-column neutralization
# leaves it alone. No damping below 1.0 (and no diagonal floor up to 0.1) can
# make it positive definite: this is the fixture where recovery must fail.
INDEFINITE = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32)


def _task(hessian, *, damp_percent, damp_auto_increment):
    generator = torch.Generator().manual_seed(5858)
    columns = hessian.shape[0]
    weight = torch.randn(ROWS, columns, generator=generator) * 0.05
    layer = torch.nn.Linear(columns, ROWS, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(weight)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=16,
        desc_act=False,
        sym=True,
        damp_percent=damp_percent,
        damp_auto_increment=damp_auto_increment,
        device="cpu",
        offload_to_disk=False,
    )
    task = QQQ(layer, qcfg=qcfg)
    task.quantizer.configure(4, perchannel=True, sym=True, groupsize=16)
    task.H = hessian.clone()
    task._materialized_samples = 32
    task.nsamples = 32
    return task


def _quantize(task):
    return task.quantize(blocksize=16)


def test_rank_deficient_hessian_recovers_at_the_configured_increment():
    """The recovered weights must match the explicit control damping exactly."""

    increment = 0.01
    initial = 1e-10
    recovered = _quantize(
        _task(RANK_ONE, damp_percent=initial, damp_auto_increment=increment)
    )
    control = _quantize(
        _task(RANK_ONE, damp_percent=initial + increment, damp_auto_increment=0.0)
    )

    # The control run is the same damping the recovery selects, so every
    # serialized tensor has to agree bit for bit.
    for left, right in zip(recovered, control):
        if not torch.is_tensor(left):
            continue
        assert left.dtype == right.dtype
        assert torch.equal(left, right)


def test_successful_first_attempt_does_not_start_recovery(monkeypatch):
    messages = []
    monkeypatch.setattr(
        "gptqmodel.quantization.gptq.log.warn",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )

    _quantize(_task(torch.eye(COLUMNS) * float(COLUMNS), damp_percent=0.01, damp_auto_increment=0.01))

    assert not [message for message in messages if "damp recovery" in message.lower()]


def test_recovery_reports_the_applied_damping(monkeypatch):
    messages = []
    monkeypatch.setattr(
        "gptqmodel.quantization.gptq.log.warn",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )

    _quantize(_task(RANK_ONE, damp_percent=1e-10, damp_auto_increment=0.01))

    applied = [message for message in messages if "Damp recovery succeeded at" in message]
    assert applied, messages
    assert "0.01000" in applied[-1]


def test_zero_increment_is_the_explicit_no_recovery_control():
    """A zero increment must not silently search a different damping grid."""

    task = _task(INDEFINITE, damp_percent=1e-10, damp_auto_increment=0.0)

    with pytest.raises(torch._C._LinAlgError):
        _quantize(task)


def test_the_configured_increment_is_what_recovers_the_indefinite_fixture():
    """The same fixture succeeds once the grid can reach a large enough damp."""

    recovered = _quantize(_task(INDEFINITE, damp_percent=1e-10, damp_auto_increment=0.01))

    assert recovered[0].shape == (ROWS, INDEFINITE.shape[0])
