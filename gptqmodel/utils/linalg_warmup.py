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
from .rocm import quantize_tuning
from .torch import TORCH_GTE_210


if TYPE_CHECKING:
    from .threadx import WarmUpCtx


log = setup_logger()

_GLOBAL_WARMUP_LOCK = threading.Lock()

_LINEAR_PREWARM_LOCK = threading.Lock()
_LINEAR_PREWARM_RUNNING: Set[Tuple[str, str, str]] = set()
_LINEAR_PREWARM_DONE: Set[Tuple[str, str, str]] = set()
_LINEAR_PREWARM_THREADS: Dict[Tuple[str, str, str], threading.Thread] = {}
_LINEAR_PREWARM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_LINEAR_PREWARM_DEVICE_TYPES = ("cuda", "xpu", "npu")
# A prewarm is a one-time library init that takes seconds; the bound only exists
# so that a wedged init can never stall the caller indefinitely.
_LINEAR_PREWARM_JOIN_TIMEOUT = 120.0

# Prewarm payloads.  Each name keys its own one-per-process registry, so the
# payloads dedupe independently while sharing the join machinery.
_PREWARM_LINEAR = "linear"
_PREWARM_QUANTIZE = "quantize"

# Quantize prewarm op groups, in execution order.  ``_PREWARM_QUANTIZE_GROUPS``
# is the full historical payload; ``prewarm_ops`` from
# ``gptqmodel.utils.rocm.quantize_tuning`` selects a per-architecture subset
# (RDNA4 currently drops ``shape``).  ``None`` means "all groups".
_PREWARM_GROUP_ORDER = ("elementwise", "fp32", "reduce", "gather", "shape", "dtype", "bitwise", "softmax")


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


def _run_quantize_prewarm(
    device: torch.device,
    dtype: torch.dtype,
    groups: Optional[tuple[str, ...]] = None,
) -> None:
    """
    Launch each accelerator kernel the quantization loop uses once, off the critical path.

    ``groups`` selects a subset of the payload; ``None`` resolves it from the
    architecture via ``quantize_tuning(device)["prewarm_ops"]`` (which is
    ``None`` itself for the full historical payload).

    The launch-cost figures below were measured on CDNA4 (MI350X, gfx950) unless
    stated otherwise; RDNA4 (gfx120*) has not been measured yet, so its payload
    is deliberately a conservative subset and must be validated by the RDNA4
    benchmark script.

    An accelerator runtime resolves a kernel's code object lazily: the first launch
    of each distinct kernel pays a module load, measured at 3-28 ms per kernel on
    MI350X (gfx950), where 36 of the ~70 kernels a 4-bit GPTQ pass touches exceed
    1 ms on their first launch and ~2-4 us on every launch after that.  The load is
    process-global -- a fresh thread launches the same kernel in under 0.2 ms once
    any thread has paid it -- but it is paid on whichever thread launches first,
    and the quantization loop runs on four device-pool workers that then queue
    behind it.  Measured on the 6-layer Qwen2 probe, the first-launch cost inside
    `quantize()` is 1.5 s of a 2.25 s cold pass, while a warm pass of the same
    model is 0.65 s.

    Running one launch of each kernel here, on the background prewarm thread while
    the CPU-bound model load is still in flight, moves that cost off the
    quantization critical path.

    Kernel selection depends on shape and dtype, not on values, so the payload
    deliberately builds every tensor without the RNG: a prewarm must never perturb
    a caller's random stream.  Failures are per-op and swallowed, because a
    warmup is best-effort by construction and some ops are absent or unsupported
    on some backends.
    """
    if dtype not in _LINEAR_PREWARM_DTYPES:
        return

    if groups is None:
        groups = quantize_tuning(device)["prewarm_ops"]

    dev = device
    half = dtype
    single = torch.float32

    if dev.type == "cuda" and torch.version.hip:
        with contextlib.suppress(Exception):
            matrix = torch.eye(1024, device=dev, dtype=single)
            torch.linalg.cholesky_ex(matrix, check_errors=False)

    with contextlib.suppress(Exception):
        values = torch.ones((7, 128), device=dev, dtype=single)
        torch.topk(values, k=128, dim=1, largest=True, sorted=True)

    a = torch.full((256, 128), 0.25, device=dev, dtype=half)
    b = torch.full((256, 128), 0.5, device=dev, dtype=half)
    a32 = a.to(single)
    b32 = b.to(single)
    positive = a.abs() + 0.5
    index32 = torch.ones((256, 128), device=dev, dtype=torch.int32)
    index32b = torch.full((256, 128), 3, device=dev, dtype=torch.int32)
    index8 = torch.full((256, 128), 7, device=dev, dtype=torch.uint8)
    rows = torch.arange(128, device=dev)

    # Deliberately no `matmul` / `addmm` / `bmm` / `linear` / `torch.linalg.*` here.
    # Those dispatch to a BLAS backend whose per-(dtype, layout) library load is
    # already paid by the `linear` prewarm and by the device-thread-pool linalg
    # warmup, and each extra flavor costs ~1 s on ROCm.  Measured, a payload that
    # included them spent 4.0 s of its 4.9 s in `aten.mm` alone, which is worse
    # than the launch latency this warmup exists to remove.
    ops_by_group = {
        "elementwise": (
        # elementwise, the bulk of the per-block update
        lambda: torch.add(a, b),
        lambda: torch.sub(a, b),
        lambda: torch.mul(a, b),
        lambda: torch.div(a, b + 1.0),
        lambda: torch.neg(a),
        lambda: torch.abs(a),
        lambda: torch.relu(a),
        lambda: torch.nn.functional.silu(a),
        lambda: torch.nn.functional.gelu(a),
        lambda: torch.nn.functional.softplus(a),
        lambda: torch.sigmoid(a),
        lambda: torch.tanh(a),
        lambda: torch.exp(a),
        lambda: torch.log(positive),
        lambda: torch.sqrt(positive),
        lambda: torch.rsqrt(positive),
        lambda: torch.erf(a),
        lambda: torch.pow(positive, 2.0),
        lambda: torch.maximum(a, b),
        lambda: torch.minimum(a, b),
        lambda: torch.where(a > b, a, b),
        lambda: torch.clamp(a, -1.0, 1.0),
        lambda: torch.round(a),
        lambda: torch.floor(a),
        lambda: torch.lerp(a, b, 0.25),
        lambda: torch.addcmul(a, a, b),
        lambda: torch.addcdiv(a, a, b + 1.0),
        ),
        # fp32 mirrors: the Hessian and its inverse run in fp32
        "fp32": (
        lambda: torch.add(a32, b32),
        lambda: torch.mul(a32, b32),
        lambda: torch.div(a32, b32 + 1.0),
        lambda: torch.neg(a32),
        lambda: torch.abs(a32),
        lambda: torch.pow(a32.abs() + 0.5, 2.0),
        lambda: torch.where(a32 > b32, a32, b32),
        lambda: torch.clamp(a32, -1.0, 1.0),
        ),
        # reductions used by the range search and the damped inverse
        "reduce": (
        lambda: torch.aminmax(a, dim=1),
        lambda: torch.amax(a),
        lambda: torch.amin(a),
        lambda: torch.argmax(a),
        lambda: torch.argmin(a),
        lambda: torch.sum(a32, dim=1),
        lambda: torch.mean(a32, dim=1),
        lambda: a32.std(dim=1),
        lambda: torch.var(a32, dim=1),
        lambda: torch.cumsum(a32, dim=1),
        lambda: torch.sort(a32, dim=1),
        lambda: torch.argsort(a32, dim=1),
        ),
        # gather / scatter / masking in the block loop
        "gather": (
        lambda: torch.nonzero(a > b),
        lambda: torch.gather(a, 1, rows.unsqueeze(0).expand(256, 128)),
        lambda: torch.index_select(a, 0, rows),
        lambda: torch.zeros_like(a).index_add_(0, rows, a[:128]),
        lambda: torch.zeros_like(a).index_copy_(0, rows, a[:128]),
        lambda: a.masked_fill(a > b, 0.0),
        lambda: torch.zeros_like(a).scatter_(1, rows.unsqueeze(0).expand(256, 128), a),
        ),
        # shape and movement: plain copy kernels, the cheapest to warm
        "shape": (
        lambda: torch.cat((a, b), dim=0),
        lambda: torch.stack((a, b)),
        lambda: torch.roll(a, 3, dims=1),
        lambda: a.transpose(0, 1).contiguous(),
        lambda: a.reshape(-1),
        lambda: a.repeat_interleave(2, dim=0),
        lambda: torch.nn.functional.pad(a, (1, 1, 1, 1)),
        ),
        # dtype conversions across the quantize path
        "dtype": (
        lambda: a.to(single),
        lambda: a32.to(half),
        lambda: a32.to(torch.int32),
        lambda: index32.to(torch.uint8),
        lambda: index8.to(half),
        lambda: index32.to(torch.float32),
        ),
        # bitwise work used when packing
        "bitwise": (
        lambda: torch.bitwise_and(index32, index32b),
        lambda: torch.bitwise_or(index32, index32b),
        lambda: torch.bitwise_xor(index32, index32b),
        lambda: torch.bitwise_left_shift(index32, index32b),
        lambda: torch.bitwise_right_shift(index32, index32b),
        lambda: torch.bitwise_and(index8, index8),
        ),
        "softmax": (
        lambda: torch.nn.functional.softmax(a32, dim=1),
        ),
    }

    if groups is None:
        selected = _PREWARM_GROUP_ORDER
    else:
        selected = tuple(group for group in _PREWARM_GROUP_ORDER if group in set(groups))

    for group in selected:
        for op in ops_by_group[group]:
            with contextlib.suppress(Exception):
                op()

    if dev.type == "cuda":
        torch.cuda.synchronize(device=dev)


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


def _kick_prewarm(
    payload: str,
    runner,
    env_name: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float16,
) -> Optional[threading.Thread]:
    """
    Start one named prewarm payload on a background thread.

    The work a payload does is a one-time, process-global accelerator-library
    init, so at most one thread per (payload, device type, dtype) is started per
    process. Best-effort by design: unsupported devices/dtypes and runtime
    failures are ignored (debug-logged) so callers can invoke unconditionally.
    """
    if not env_flag(env_name, default=True):
        return None

    if dtype not in _LINEAR_PREWARM_DTYPES:
        return None

    target = _linear_prewarm_device(device)
    if target is None:
        return None

    key = (payload, target.type, str(dtype))

    def _worker() -> None:
        ok = False
        try:
            runner(target, dtype)
            ok = True
        except Exception as exc:
            log.debug(f"{payload} prewarm skipped for {target} {dtype}: {exc}")
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
            name=f"GPTPrewarm-{payload}-{target.type}-{dtype}",
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
            log.debug(f"{payload} prewarm could not start for {target} {dtype}: {exc}")
            return None

    return thread


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

    Set `GPTQMODEL_LINEAR_PREWARM=0` to disable.
    """
    return _kick_prewarm(_PREWARM_LINEAR, _run_linear_prewarm, "GPTQMODEL_LINEAR_PREWARM", device, dtype)


def kick_torch_quantize_prewarm(
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float16,
) -> Optional[threading.Thread]:
    """
    Start the one-time quantization-kernel warmup on a background thread.

    The first launch of each distinct accelerator kernel pays a module load that
    is process-global but lands on whichever thread launches first; see
    `_run_quantize_prewarm`. Kicking this while the caller still has CPU-bound
    work to do (loading model weights) keeps that load off the quantization
    critical path.

    Set `GPTQMODEL_QUANTIZE_PREWARM=0` to disable.
    """
    return _kick_prewarm(_PREWARM_QUANTIZE, _run_quantize_prewarm, "GPTQMODEL_QUANTIZE_PREWARM", device, dtype)


def await_torch_linear_prewarm(
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    timeout: Optional[float] = _LINEAR_PREWARM_JOIN_TIMEOUT,
    payload: Optional[str] = None,
) -> None:
    """
    Join in-flight accelerator prewarms so the first real work does not pay the init again.

    A completed accelerator-library init is shared process-wide, but a *concurrent*
    first touch is not: a second caller arriving while the init is still running
    pays the full init again on its own thread. Waiting for the remaining prewarm
    time is therefore cheaper than racing it, and the call is a no-op once the
    prewarm has finished (or was never started).

    Best-effort by design: disabled prewarm, unsupported devices, and threads that
    already finished all return immediately. `dtype=None` matches any prewarm for
    the device family and `payload=None` matches every payload; pass the dtype the
    caller is about to use, and a payload name, to avoid waiting on an unrelated
    init.
    """
    target = _linear_prewarm_device(device)
    if target is None:
        return

    device_type = target.type
    dtype_key = str(dtype) if dtype is not None else None
    with _LINEAR_PREWARM_LOCK:
        threads = [
            thread
            for (prewarm_payload, prewarm_device_type, prewarm_dtype), thread in _LINEAR_PREWARM_THREADS.items()
            if prewarm_device_type == device_type
            and (dtype_key is None or prewarm_dtype == dtype_key)
            and (payload is None or prewarm_payload == payload)
        ]

    current = threading.current_thread()
    for thread in threads:
        if thread is current or not thread.is_alive():
            continue
        thread.join(timeout)


__all__ = [
    "await_torch_linear_prewarm",
    "kick_torch_linear_prewarm",
    "kick_torch_quantize_prewarm",
    "run_torch_linalg_warmup",
]
