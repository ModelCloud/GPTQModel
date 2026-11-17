# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #4538: QQQ's CPU Hessian retry must carry the act-order permutation.

QQQ rejects ``desc_act=True`` in the public config because its packed runtime
has no ``g_idx`` buffer, but ``_quantize_impl`` retains the act-order branch and
the CUDA-OOM retry added by #4464/#4468. That retry moved the Hessian, the
weight and the static-group quantizers to the host but left ``perm``/``invperm``
on CUDA, so the epilogue indexed the host ``Q`` with a CUDA index tensor:

    RuntimeError: indices should be either on cpu or on the same device as the
    indexed tensor (cpu)

The retry now moves both permutations, and the epilogue re-homes them on
``Q.device``. These tests force a one-shot CUDA allocation failure in the
Cholesky step and compare against a run that never leaves the host.
"""

import pytest
import torch

from gptqmodel.quantization.config import QQQConfig
from gptqmodel.quantization.qqq import QQQ


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the fallback only engages for accelerator allocations",
)

ROWS, COLUMNS = 32, 64
GROUP_SIZE = 16
SAMPLES = 64
BLOCKSIZE = 16
REAL_CHOLESKY = torch.linalg.cholesky


def _weight() -> torch.Tensor:
    generator = torch.Generator().manual_seed(4538)
    return torch.randn(ROWS, COLUMNS, generator=generator) * 0.05


def _hessian() -> torch.Tensor:
    """Positive-definite Hessian with a scrambled diagonal.

    A scrambled diagonal makes ``argsort(diag(H), descending=True)`` a
    non-identity permutation, so a dropped reorder or a stale-device index
    tensor cannot pass unnoticed.
    """

    generator = torch.Generator().manual_seed(4538)
    skew = torch.randn(COLUMNS, COLUMNS, generator=generator)
    skew = (skew - skew.T) / 2
    diagonal = torch.linspace(1.0, 2.0, COLUMNS)[
        torch.randperm(COLUMNS, generator=generator)
    ]
    return torch.diag(diagonal) + 1e-3 * skew


def _expected_g_idx(static_groups: bool) -> torch.Tensor:
    """Recompute the epilogue's ``g_idx`` from the same Hessian."""

    perm = torch.argsort(torch.diag(_hessian()), descending=True)
    invperm = torch.argsort(perm)
    if static_groups:
        # ``(perm // group)[invperm]`` reduces to ``column // group`` because
        # the static groups were fitted in original column order.
        return (perm // GROUP_SIZE).to(dtype=torch.int32)[invperm]
    return (torch.arange(COLUMNS, dtype=torch.int32) // GROUP_SIZE)[invperm]


def _qqq_task(device, *, static_groups: bool, desc_act: bool = True) -> QQQ:
    layer = torch.nn.Linear(
        COLUMNS, ROWS, bias=False, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        layer.weight.copy_(_weight().to(device))

    qcfg = QQQConfig(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        damp_percent=0.125,
        device=str(device),
        static_groups=static_groups,
    )
    qcfg.desc_act = desc_act

    task = QQQ(layer, qcfg=qcfg)
    if desc_act:
        # The public config rejects desc_act; drive the retained act-order path.
        def _allow_desc_act() -> None:
            return None

        task._reject_unsupported_activation_order = _allow_desc_act
    task.quantizer.configure(4, perchannel=True, sym=True, groupsize=GROUP_SIZE)
    task.H = _hessian().to(device)
    task._materialized_samples = SAMPLES
    task.nsamples = SAMPLES
    return task


def _fail_first_cuda_cholesky(monkeypatch) -> dict:
    state = {"failed": False}

    def flaky(hessian, *args, **kwargs):
        if hessian.device.type == "cuda" and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return REAL_CHOLESKY(hessian, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", flaky)
    return state


@pytest.mark.parametrize(
    "static_groups",
    [False, True],
    ids=["dynamic_groups", "static_groups"],
)
def test_qqq_desc_act_cpu_fallback_keeps_permutation_in_sync(
    monkeypatch, static_groups
):
    reference = _qqq_task("cpu", static_groups=static_groups).quantize(
        blocksize=BLOCKSIZE
    )
    contiguous = _qqq_task(
        "cpu", static_groups=static_groups, desc_act=False
    ).quantize(blocksize=BLOCKSIZE)
    # The reorder must change the result; otherwise the round-trip comparison
    # below could pass on a silently skipped permutation.
    assert not torch.equal(reference[0], contiguous[0])

    task = _qqq_task("cuda", static_groups=static_groups)
    state = _fail_first_cuda_cholesky(monkeypatch)
    messages = []
    monkeypatch.setattr(
        "gptqmodel.quantization.qqq.log.warn",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )

    result = task.quantize(blocksize=BLOCKSIZE)

    assert state["failed"] is True
    assert "falling back to CPU" in "\n".join(messages)

    expected_g_idx = _expected_g_idx(static_groups)
    # The module's device is restored after the host retry, and both the
    # quantized weight and the act-order map match a run that never left the
    # host. Before the fix this raised the device-mismatch RuntimeError.
    assert result[0].device.type == "cuda"
    assert torch.equal(result[0].cpu(), reference[0].cpu())
    assert torch.equal(result[3].cpu(), reference[3].cpu())
    assert torch.equal(result[3].cpu(), expected_g_idx)
    if not static_groups:
        # The dynamic-group map exposes the permutation directly: it must not
        # collapse to the contiguous ``column // group`` order.
        assert not torch.equal(
            expected_g_idx,
            torch.arange(COLUMNS, dtype=torch.int32) // GROUP_SIZE,
        )
