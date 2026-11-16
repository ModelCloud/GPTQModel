# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4350: solver allocation failures must use the GPTQ CPU fallback.

The Hessian-inverse fallback used to re-raise every ``RuntimeError`` whose text
did not contain ``"out of memory"``. cuSOLVER/hipSOLVER report a failed
workspace allocation as ``CUSOLVER_STATUS_ALLOC_FAILED`` /
``HIPSOLVER_STATUS_ALLOC_FAILED``, so the whole run aborted instead of finishing
the module on CPU. A non-positive-definite Cholesky must still propagate.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


ALLOCATION_FAILURES = [
    pytest.param(
        torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        id="torch-out-of-memory",
    ),
    pytest.param(MemoryError("host allocation failed"), id="memory-error"),
    pytest.param(
        RuntimeError(
            "hipsolver error: HIPSOLVER_STATUS_ALLOC_FAILED, "
            "when calling hipsolverDnSpotrs(...)"
        ),
        id="hipsolver-alloc-failed",
    ),
    pytest.param(
        RuntimeError(
            "cuSOLVER error: CUSOLVER_STATUS_ALLOC_FAILED when calling "
            "cusolverDnSpotrf(...)"
        ),
        id="cusolver-alloc-failed",
    ),
    pytest.param(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"), id="cublas-alloc-failed"),
    pytest.param(RuntimeError("hipErrorOutOfMemory"), id="hip-out-of-memory"),
    pytest.param(
        RuntimeError("CUDA error: cudaErrorMemoryAllocation"),
        id="cuda-memory-allocation",
    ),
    pytest.param(RuntimeError("CUDA out of memory (legacy text)"), id="legacy-phrase"),
]

NUMERICAL_FAILURES = [
    pytest.param(
        torch._C._LinAlgError(
            "The leading minor of order 1 is not positive definite"
        ),
        id="linalg-not-positive-definite",
    ),
    pytest.param(
        RuntimeError("cholesky failed: input is not positive-definite"),
        id="runtime-not-positive-definite",
    ),
    pytest.param(ValueError("shape mismatch"), id="unrelated-value-error"),
]


def _classify(exc):
    # Imported lazily so the behavioral GPU tests still collect on a tree that
    # predates the structural classifier (their fail-before is the point).
    from gptqmodel.quantization.gptq import _is_memory_allocation_failure

    return _is_memory_allocation_failure(exc)


@pytest.mark.parametrize("exc", ALLOCATION_FAILURES)
def test_allocation_failures_are_classified_as_memory_failures(exc):
    assert _classify(exc)


@pytest.mark.parametrize("exc", NUMERICAL_FAILURES)
def test_numerical_failures_are_not_classified_as_memory_failures(exc):
    assert not _classify(exc)


def _make_module(hidden_dim, device):
    layer = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float16)
    return layer.to(device).eval()


def _generate_input(batch_size, seq_len, hidden_dim, device):
    return torch.randn(
        batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16
    )


def _prepare_quantizer(monkeypatch, exc):
    """Build a GPU GPTQ task whose Hessian inverse raises ``exc`` on CUDA."""

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    layer = _make_module(hidden_dim=8, device=device)
    qcfg = QuantizeConfig(bits=4, group_size=2, act_group_aware=True)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(_generate_input(1, 4, 8, device), None)

    calls = {"cuda": 0, "cpu": 0}

    def _patched_hessian_inverse(self, hessian):
        if hessian.device.type == "cuda":
            calls["cuda"] += 1
            raise exc
        calls["cpu"] += 1
        identity = torch.eye(
            hessian.shape[0], dtype=torch.float32, device=hessian.device
        )
        return identity, self.qcfg.damp_percent

    monkeypatch.setattr(GPTQ, "hessian_inverse", _patched_hessian_inverse)
    return device, gptq, calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the fallback path")
def test_solver_alloc_failure_engages_the_cpu_fallback(monkeypatch):
    device, gptq, calls = _prepare_quantizer(
        monkeypatch,
        RuntimeError(
            "hipsolver error: HIPSOLVER_STATUS_ALLOC_FAILED, "
            "when calling hipsolverDnSpotrs(...)"
        ),
    )
    log_messages = []
    monkeypatch.setattr(
        gptq_mod.log,
        "warn",
        lambda message, *args, **kwargs: log_messages.append(
            message % args if args else message
        ),
    )

    qweight, *_ = gptq.quantize(blocksize=4)

    assert calls == {"cuda": 1, "cpu": 1}
    assert qweight.device == device
    assert "falling back to CPU" in "\n".join(log_messages)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the fallback path")
def test_numerical_cholesky_failure_does_not_engage_the_cpu_fallback(monkeypatch):
    error = torch._C._LinAlgError(
        "The leading minor of order 1 is not positive definite"
    )
    _device, gptq, calls = _prepare_quantizer(monkeypatch, error)

    with pytest.raises(torch._C._LinAlgError):
        gptq.quantize(blocksize=4)

    assert calls == {"cuda": 1, "cpu": 0}
