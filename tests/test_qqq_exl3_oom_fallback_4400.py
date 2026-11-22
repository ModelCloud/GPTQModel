# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4400 item 3: QQQ and EXL3 must retry solver allocations on CPU.

QQQ's Cholesky/inverse sequence used to re-raise every failure that was not the
sample-count fallback, and EXL3's ``block_ldl`` only matched literal
``OutOfMemoryError`` / "CUDA|HIP out of memory" text. cuSOLVER and hipSOLVER
report a failed workspace/handle allocation as
``*SOLVER_STATUS_ALLOC_FAILED``; both paths now reuse GPTQ's
``_is_memory_allocation_failure`` classifier and finish on the host.
"""

import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import block_ldl
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.qqq import QQQ


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the fallback only engages for accelerator allocations",
)

ALLOCATION_FAILURE = RuntimeError(
    "hipsolver error: HIPSOLVER_STATUS_ALLOC_FAILED, "
    "when calling hipsolverDnCreate(handle)"
)

ROWS, COLUMNS = 32, 64


def _constant_diagonal_hessian(columns, generator):
    """Positive-definite matrix with an exactly constant diagonal.

    A constant diagonal makes ``mean(diag(H))`` bitwise identical on CPU and
    CUDA, so the CPU retry sees the same damped Hessian as the reference run.
    """

    skew = torch.randn(columns, columns, generator=generator)
    skew = (skew - skew.T) / 2
    return torch.eye(columns) * float(columns) + 1e-3 * skew


def _qqq_task(device):
    # A dedicated generator keeps the CPU and CUDA fixtures identical: the
    # layer's own init draws from the global RNG, which differs per device.
    generator = torch.Generator().manual_seed(4400)
    weight = torch.randn(ROWS, COLUMNS, generator=generator) * 0.05
    hessian = _constant_diagonal_hessian(COLUMNS, generator)
    layer = torch.nn.Linear(COLUMNS, ROWS, bias=False, dtype=torch.float32, device=device)
    with torch.no_grad():
        layer.weight.copy_(weight.to(device))

    qcfg = QuantizeConfig(
        bits=4,
        group_size=16,
        desc_act=False,
        sym=True,
        damp_percent=0.125,
        device=str(device),
    )
    task = QQQ(layer, qcfg=qcfg)
    task.quantizer.configure(4, perchannel=True, sym=True, groupsize=16)
    task.H = hessian.to(device)
    task._materialized_samples = 64
    task.nsamples = 64
    return task


def _fail_first_cuda_cholesky(monkeypatch, error):
    real_cholesky = torch.linalg.cholesky
    state = {"failed": False}

    def flaky(hessian, *args, **kwargs):
        if hessian.device.type == "cuda" and not state["failed"]:
            state["failed"] = True
            raise error
        return real_cholesky(hessian, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", flaky)
    return state


def test_qqq_solver_allocation_failure_retries_on_cpu(monkeypatch):
    reference = _qqq_task("cpu").quantize(blocksize=16)

    task = _qqq_task("cuda")
    state = _fail_first_cuda_cholesky(monkeypatch, ALLOCATION_FAILURE)
    messages = []
    monkeypatch.setattr(
        "gptqmodel.quantization.qqq.log.warn",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )

    retried = task.quantize(blocksize=16)

    assert state["failed"] is True
    assert "falling back to CPU" in "\n".join(messages)
    # The retry finishes the whole module on the host and moves the packed
    # weight back to the module's device.
    assert retried[0].device.type == "cuda"
    assert torch.equal(reference[0].cpu(), retried[0].cpu())
    assert torch.equal(reference[1].cpu(), retried[1].cpu())


def test_qqq_numerical_failure_does_not_retry_on_cpu(monkeypatch):
    """An indefinite Hessian exhausts damp recovery without leaving CUDA.

    Recovery intentionally retries the additive damp grid and the diagonal
    floor (issue #5858), so the failure must be a matrix that stays
    non-positive-definite for every candidate: ``cholesky_ex`` reports
    ``info != 0`` on CUDA for the whole run.
    """

    task = _qqq_task("cuda")
    real_cholesky_ex = torch.linalg.cholesky_ex
    calls = {"cuda": 0}

    def always_indefinite(hessian, *args, **kwargs):
        if hessian.device.type == "cuda":
            calls["cuda"] += 1
            return torch.empty_like(hessian), torch.tensor(1, device=hessian.device)
        return real_cholesky_ex(hessian, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", always_indefinite)
    messages = []
    monkeypatch.setattr(
        "gptqmodel.quantization.qqq.log.warn",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )

    with pytest.raises(torch._C._LinAlgError):
        task.quantize(blocksize=16)

    assert calls["cuda"] > 1
    assert "falling back to CPU" not in "\n".join(messages)


def test_exl3_block_ldl_solver_allocation_failure_retries_on_cpu(monkeypatch):
    torch.manual_seed(4400)
    size = 64
    dense = torch.randn(size, size, dtype=torch.float64)
    hessian = (dense @ dense.T + torch.eye(size, dtype=torch.float64) * size).to(
        torch.float32
    )

    reference, _ = block_ldl(hessian.clone(), 16, False)

    state = _fail_first_cuda_cholesky(monkeypatch, ALLOCATION_FAILURE)
    retried, _ = block_ldl(hessian.clone().cuda(), 16, False)

    assert state["failed"] is True
    assert retried.device.type == "cuda"
    torch.testing.assert_close(reference, retried.cpu(), rtol=1e-6, atol=1e-6)
