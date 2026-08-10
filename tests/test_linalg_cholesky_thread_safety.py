# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for free-threaded CUDA linalg lazy-dispatch initialization."""

import concurrent.futures
import os
import subprocess
import sys
import textwrap
import threading
from unittest.mock import patch

import pytest
import torch

import gptqmodel.utils.torch as torch_utils
from gptqmodel.utils.linalg_warmup import run_torch_linalg_warmup
from gptqmodel.utils.threadx import WarmUpCtx
from gptqmodel.utils.torch import (
    cholesky_inverse,
    linalg_cholesky,
    linalg_cholesky_ex,
    linalg_eigh,
    linalg_inv,
    linalg_qr,
    linalg_svd,
)


def _spd_matrix(n: int, device: torch.device) -> torch.Tensor:
    a = torch.randn(n, n, device=device, dtype=torch.float32)
    h = a @ a.T
    h.diagonal().add_(0.1)
    return h


def _device_for_worker(idx: int) -> torch.device:
    return torch.device("cuda", idx % torch.cuda.device_count())


def test_linalg_dispatch_bypasses_guard_for_cpu_and_non_tensor_calls():
    calls = []

    def operation(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    with patch.object(torch_utils, "_LINALG_DISPATCH_READY", False):
        assert torch_utils._linalg_dispatch_call(operation, torch.ones(1), value=1) == "result"
        assert torch_utils._linalg_dispatch_call(operation, value=2) == "result"
        assert torch_utils._LINALG_DISPATCH_READY is False
    assert len(calls) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pre_fix_race_control_and_guarded_fix_are_deterministic():
    """The control reproduces overlapping first use; the guarded path prevents it."""

    tensor = torch.ones(1, device="cuda")

    def run_pair(guarded: bool):
        entered = threading.Event()
        release = threading.Event()
        state_lock = threading.Lock()
        active = False

        def unsafe_lazy_operation(_tensor):
            nonlocal active
            with state_lock:
                if active:
                    raise RuntimeError("lazy wrapper should be called at most once")
                active = True
                entered.set()
            assert release.wait(timeout=5)
            with state_lock:
                active = False
            return _tensor

        call = (
            (lambda: torch_utils._linalg_dispatch_call(unsafe_lazy_operation, tensor))
            if guarded
            else (lambda: unsafe_lazy_operation(tensor))
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(call)
            assert entered.wait(timeout=5)
            second = executor.submit(call)
            if guarded:
                release.set()
                assert first.result(timeout=5) is tensor
                assert second.result(timeout=5) is tensor
            else:
                with pytest.raises(RuntimeError, match="lazy wrapper should be called at most once"):
                    second.result(timeout=5)
                release.set()
                assert first.result(timeout=5) is tensor

    run_pair(guarded=False)
    with patch.object(torch_utils, "_LINALG_DISPATCH_READY", False):
        run_pair(guarded=True)
        assert torch_utils._LINALG_DISPATCH_READY is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_linalg_dispatch_failure_does_not_mark_initialization_ready():
    tensor = torch.ones(1, device="cuda")

    def fail(_tensor):
        raise ValueError("expected")

    with patch.object(torch_utils, "_LINALG_DISPATCH_READY", False):
        with pytest.raises(ValueError, match="expected"):
            torch_utils._linalg_dispatch_call(fail, tensor)
        assert torch_utils._LINALG_DISPATCH_READY is False


def test_public_linalg_wrappers_delegate_to_dispatch_guard():
    cases = [
        (linalg_cholesky_ex, torch.linalg.cholesky_ex),
        (linalg_cholesky, torch.linalg.cholesky),
        (cholesky_inverse, torch.cholesky_inverse),
        (linalg_inv, torch.linalg.inv),
        (linalg_eigh, torch.linalg.eigh),
        (linalg_svd, torch.linalg.svd),
        (linalg_qr, torch.linalg.qr),
    ]
    sentinel = object()
    with patch.object(torch_utils, "_linalg_dispatch_call", return_value=sentinel) as dispatch:
        for wrapper, operation in cases:
            assert wrapper("input", option=True) is sentinel
            dispatch.assert_called_with(operation, "input", option=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_linalg_mixed_ops_are_thread_safe_and_accurate():
    """Concurrent first use on multiple GPUs must match dense numerical identities."""

    def run(idx: int):
        device = _device_for_worker(idx)
        torch.cuda.set_device(device)
        h = _spd_matrix(64, device)
        lower = linalg_cholesky(h)
        assert torch.allclose(lower @ lower.T, h, rtol=2e-4, atol=2e-4)
        lower_ex, info = linalg_cholesky_ex(h)
        assert info.item() == 0
        inverse = cholesky_inverse(lower_ex)
        eye = torch.eye(64, device=device)
        assert torch.allclose(h @ inverse, eye, rtol=2e-3, atol=2e-3)
        values, vectors = linalg_eigh(h)
        assert torch.allclose((vectors * values) @ vectors.T, h, rtol=3e-4, atol=3e-4)
        matrix = torch.randn(32, 16, device=device)
        u, s, vh = linalg_svd(matrix, full_matrices=False)
        assert torch.allclose((u * s) @ vh, matrix, rtol=3e-4, atol=3e-4)
        q, r = linalg_qr(matrix, mode="reduced")
        assert torch.allclose(q @ r, matrix, rtol=3e-4, atol=3e-4)
        torch.cuda.synchronize(device)

    with patch.object(torch_utils, "_LINALG_DISPATCH_READY", False):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run, i) for i in range(16)]
            for future in concurrent.futures.as_completed(futures):
                future.result()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_linalg_warmup_can_overlap_hessian_factorization():
    def run(idx: int):
        device = _device_for_worker(idx)
        torch.cuda.set_device(device)
        if idx % 2 == 0:
            run_torch_linalg_warmup(device, WarmUpCtx.THREAD)
        else:
            lower, info = linalg_cholesky_ex(_spd_matrix(128, device))
            assert info.item() == 0
            assert torch.isfinite(cholesky_inverse(lower)).all().item()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run, i) for i in range(16)]
        for future in concurrent.futures.as_completed(futures):
            future.result()


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("GPTQMODEL_RAW_LINALG_RACE_PROBE") != "1",
    reason="opt-in upstream PyTorch race probe requiring CUDA",
)
def test_raw_linalg_race_probe():
    """Diagnostic proof against an affected PyTorch build, not a permanent CI contract."""

    script = textwrap.dedent(
        """
        import concurrent.futures
        import sys
        import threading
        import torch

        assert not sys._is_gil_enabled()
        barrier = threading.Barrier(8)
        def run(_):
            barrier.wait()
            a = torch.randn(64, 64, device="cuda")
            return torch.linalg.cholesky(a @ a.T + 0.1 * torch.eye(64, device="cuda"))
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for future in [executor.submit(run, i) for i in range(8)]:
                try:
                    future.result()
                except Exception as error:
                    errors.append(str(error))
        if any("lazy wrapper should be called at most once" in error for error in errors):
            print("REPRODUCED")
            raise SystemExit(0)
        print(errors or ["NO_RACE"])
        raise SystemExit(1)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-X", "gil=0", "-c", script],
        env={**os.environ, "PYTHON_GIL": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0 and "REPRODUCED" in proc.stdout, (
        f"returncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
