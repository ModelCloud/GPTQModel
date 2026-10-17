# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple, Union

import torch

from .env import env_flag
from .logger import setup_logger
from .torch import TORCH_GTE_210


if TYPE_CHECKING:
    from .threadx import WarmUpCtx


log = setup_logger()

_GLOBAL_WARMUP_LOCK = threading.Lock()

_LINEAR_PREWARM_LOCK = threading.Lock()
_LINEAR_PREWARM_RUNNING: Set[Tuple[str, str]] = set()
_LINEAR_PREWARM_DONE: Set[Tuple[str, str]] = set()
_LINEAR_PREWARM_THREADS: Dict[Tuple[str, str], threading.Thread] = {}
_LINEAR_PREWARM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_LINEAR_PREWARM_DEVICE_TYPES = ("cuda", "xpu", "npu")
# A prewarm is a one-time library init that takes seconds; the bound only exists
# so that a wedged init can never stall the caller indefinitely.
_LINEAR_PREWARM_JOIN_TIMEOUT = 120.0


def _get_cuda_preferred_linalg_library():
    preferred = getattr(torch.backends.cuda, "preferred_linalg_library", None)
    if preferred is None:
        return None
    if callable(preferred):
        return preferred()
    return preferred


def _set_cuda_preferred_linalg_library(backend) -> bool:
    preferred = getattr(torch.backends.cuda, "preferred_linalg_library", None)
    if preferred is None:
        return False
    if callable(preferred):
        preferred(backend=backend)
        return True
    setattr(torch.backends.cuda, "preferred_linalg_library", backend)
    return True


def _make_spd(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate a small symmetric positive definite matrix without a BLAS GEMM.

    ``A @ A.T`` would dispatch a GEMM that is not itself the warmup target: on
    ROCm the default BLAS preference routes an fp32 GEMM to hipBLASLt, which
    lazily loads a per-dtype Tensile library (~1 s) just to build this 4x4
    warmup input.  An elementwise product is symmetric by construction, and
    shifting the diagonal by the largest off-diagonal row sum keeps it strictly
    diagonally dominant, hence positive definite.
    """
    base = torch.randn((size, size), device=device, dtype=dtype)
    spd = base * base.transpose(-1, -2)
    off_diagonal = spd.abs().sum(dim=-1) - spd.diagonal().abs()
    spd.diagonal().add_(off_diagonal.max() + 1.0)
    return spd


def _run_cholesky_and_eigh(device: torch.device, dtype: torch.dtype) -> None:
    spd = _make_spd(4, device, dtype)
    torch.linalg.cholesky(spd)

    # mps has no aten.eigh implementation
    if device.type == "mps":
        return

    torch.linalg.eigh(spd)


def _run_svd(device: torch.device, dtype: torch.dtype) -> None:
    mat = torch.randn((4, 3), device=device, dtype=dtype)
    torch.linalg.svd(mat, full_matrices=False)


def _run_qr(device: torch.device, dtype: torch.dtype) -> None:
    # mps has no aten.qr implementation
    if device.type == "mps":
        return

    square = torch.randn((4, 4), device=device, dtype=dtype)
    torch.linalg.qr(square)


def run_torch_linalg_warmup(device: torch.device, warmup_ctx: "WarmUpCtx") -> None:
    """
    Execute the torch.linalg operators used across the project once on the worker thread.

    Serialized under a global lock to avoid races inside PyTorch's lazy wrappers.
    The caller decides whether this invocation is running in device-wide or
    thread-local warmup context via `warmup_ctx`.
    """
    with _GLOBAL_WARMUP_LOCK:
        if device.type == "mps":
            dtypes = (torch.float32,)  # MPS backend does not implement float64.
        else:
            dtypes = (torch.float32, torch.float64)

        for dtype in dtypes:
            _run_cholesky_and_eigh(device, dtype)
            _run_svd(device, dtype)
            _run_qr(device, dtype)

        if device.type == "cuda" and hasattr(torch.backends, "cuda"):
            current = _get_cuda_preferred_linalg_library()
            if current is not None and not TORCH_GTE_210:
                # Core warmup already ran using the currently preferred backend above.
                # Some installations fall back to MAGMA when the primary solver is unavailable,
                # so we pre-initialize MAGMA as well when it differs from the preferred backend.
                if current and current != "magma":
                    with contextlib.suppress(Exception):
                        _set_cuda_preferred_linalg_library("magma")
                        _run_cholesky_and_eigh(device, torch.float32)
                if current:
                    with contextlib.suppress(Exception):
                        _set_cuda_preferred_linalg_library(current)


def _run_linear_prewarm(device: torch.device, dtype: torch.dtype) -> None:
    """Touch the `linear` dispatch path once so the one-time library init lands here."""
    activations = torch.zeros((8, 64), device=device, dtype=dtype)
    weight = torch.zeros((64, 64), device=device, dtype=dtype)
    torch.nn.functional.linear(activations, weight)


def _linear_prewarm_device(device: Optional[Union[str, torch.device]]) -> Optional[torch.device]:
    """Resolve the accelerator a prewarm applies to, or None when unsupported."""
    if device is None:
        device = "cuda"
    if isinstance(device, int):
        device = f"cuda:{device}"
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return None
    if resolved.type not in _LINEAR_PREWARM_DEVICE_TYPES:
        return None
    return resolved


def kick_torch_linear_prewarm(
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float16,
) -> Optional[threading.Thread]:
    """
    Start the one-time `linear` accelerator-library init on a background thread.

    The first `linear` forward of a given dtype pays a one-time library init
    (measured ~1.2 s on ROCm) that is unrelated to the matmul itself. Kicking it
    while the caller still has CPU-bound work to do (e.g. loading model weights)
    hides that init instead of stalling the first real forward pass.

    The init is process-global, so at most one warmup per device type/dtype pair
    is started per process. Best-effort by design: unsupported devices/dtypes and
    runtime failures are ignored (debug-logged) so callers can invoke this
    unconditionally. Set `GPTQMODEL_LINEAR_PREWARM=0` to disable.
    """
    if not env_flag("GPTQMODEL_LINEAR_PREWARM", default=True):
        return None

    if dtype not in _LINEAR_PREWARM_DTYPES:
        return None

    target = _linear_prewarm_device(device)
    if target is None:
        return None

    key = (target.type, str(dtype))

    def _worker() -> None:
        ok = False
        try:
            _run_linear_prewarm(target, dtype)
            ok = True
        except Exception as exc:
            log.debug(f"linear prewarm skipped for {target} {dtype}: {exc}")
        finally:
            with _LINEAR_PREWARM_LOCK:
                _LINEAR_PREWARM_RUNNING.discard(key)
                if ok:
                    _LINEAR_PREWARM_DONE.add(key)

    with _LINEAR_PREWARM_LOCK:
        if key in _LINEAR_PREWARM_DONE or key in _LINEAR_PREWARM_RUNNING:
            return None
        _LINEAR_PREWARM_RUNNING.add(key)
        thread = threading.Thread(
            target=_worker,
            name=f"GPTPrewarm-{target.type}-{dtype}",
            daemon=True,
        )
        _LINEAR_PREWARM_THREADS[key] = thread
        try:
            # Started under the lock: a concurrent `await_torch_linear_prewarm` must
            # never observe a registered thread that cannot be joined yet.
            thread.start()
        except Exception as exc:
            _LINEAR_PREWARM_RUNNING.discard(key)
            _LINEAR_PREWARM_THREADS.pop(key, None)
            log.debug(f"linear prewarm could not start for {target} {dtype}: {exc}")
            return None

    return thread


def await_torch_linear_prewarm(
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    timeout: Optional[float] = _LINEAR_PREWARM_JOIN_TIMEOUT,
) -> None:
    """
    Join an in-flight `linear` prewarm so the first real forward does not pay the init again.

    A completed accelerator-library init is shared process-wide, but a *concurrent*
    first touch is not: a second caller arriving while the init is still running
    pays the full init again on its own thread. Waiting for the remaining prewarm
    time is therefore cheaper than racing it, and the call is a no-op once the
    prewarm has finished (or was never started).

    Best-effort by design: disabled prewarm, unsupported devices, and threads that
    already finished all return immediately. `dtype=None` matches any prewarm for
    the device family; pass the dtype the caller is about to use to avoid waiting
    on an unrelated init.
    """
    if not env_flag("GPTQMODEL_LINEAR_PREWARM", default=True):
        return

    target = _linear_prewarm_device(device)
    if target is None:
        return

    device_type = target.type
    dtype_key = str(dtype) if dtype is not None else None
    with _LINEAR_PREWARM_LOCK:
        threads = [
            thread
            for (prewarm_device_type, prewarm_dtype), thread in _LINEAR_PREWARM_THREADS.items()
            if prewarm_device_type == device_type and (dtype_key is None or prewarm_dtype == dtype_key)
        ]

    current = threading.current_thread()
    for thread in threads:
        if thread is current or not thread.is_alive():
            continue
        thread.join(timeout)


__all__ = [
    "await_torch_linear_prewarm",
    "kick_torch_linear_prewarm",
    "run_torch_linalg_warmup",
]
