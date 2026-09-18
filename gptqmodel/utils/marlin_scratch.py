# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Caller-owned temporary buffers for the Marlin GEMM operator.

The CUDA operator allocates temporary tensors when callers do not provide
them.  ``MarlinScratchContext`` lets a caller retain those allocations for a
bounded, eager execution scope without putting temporary tensors on a model.
"""

from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar, Token
import threading
from typing import Any, Optional

import torch


_ACTIVE_CONTEXT: ContextVar[Optional["MarlinScratchContext"]] = ContextVar(
    "gptqmodel_marlin_scratch_context", default=None
)
_MAX_METADATA_ENTRIES = 64


def active_marlin_scratch_context() -> Optional["MarlinScratchContext"]:
    """Return the context active in the current logical execution context."""

    return _ACTIVE_CONTEXT.get()


def _device_index(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    return int(torch.cuda.current_device())


def _stream_identity(stream: Any) -> Any:
    """Get a stable identity for a CUDA stream, including mocked streams."""

    if stream is None:
        return None
    raw = getattr(stream, "cuda_stream", None)
    if raw is not None:
        try:
            hash(raw)
            return ("cuda_stream", raw)
        except Exception:
            pass
    try:
        hash(stream)
        return ("stream", stream)
    except Exception:
        return ("stream_id", id(stream))


def _dtype_itemsize(dtype: torch.dtype) -> int:
    # Marlin's activation dtypes are float16 and bfloat16.  Keep this helper
    # allocation-free on the hot path and retain a generic fallback for test
    # doubles and future operator dtypes.
    try:
        return torch.finfo(dtype).bits // 8
    except (TypeError, ValueError):
        try:
            return torch.iinfo(dtype).bits // 8
        except (TypeError, ValueError):
            return torch.empty(0, dtype=dtype).element_size()


def _record_stream(tensor: Optional[torch.Tensor], stream: Any) -> None:
    if tensor is None or stream is None:
        return
    record = getattr(tensor, "record_stream", None)
    if record is None:
        return
    try:
        record(stream)
    except (RuntimeError, AssertionError):
        # CPU tensors and CPU test doubles do not support CUDA stream
        # recording.  CUDA tensors still use the real record_stream path.
        if getattr(tensor, "is_cuda", False):
            raise


class MarlinScratchContext:
    """Own and reuse Marlin GEMM scratch tensors for one eager caller.

    The context is deliberately thread and stream affine.  Only one Marlin
    invocation may borrow it at a time, and CUDA graph capture is rejected at
    entry because this manager may grow or replace its tensors.  Explicit
    ``c_tmp``/``a_tmp`` arguments to :func:`gptq_marlin_gemm` remain available
    for graph-safe callers outside a context.
    """

    def __init__(
        self,
        device: torch.device | str | int,
        *,
        max_cached_bytes: int = 64 << 20,
    ) -> None:
        if max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be nonnegative")
        self.device = torch.device(device)
        self.max_cached_bytes = int(max_cached_bytes)

        self._metadata: "OrderedDict[tuple[Any, ...], tuple[int, int]]" = OrderedDict()
        self._c_tmp: Optional[torch.Tensor] = None
        self._a_tmp: Optional[torch.Tensor] = None
        self._workspace: Optional[torch.Tensor] = None
        self._workspace_bytes = 0
        self._buffer_dtype: Optional[torch.dtype] = None
        self._last_stream: Any = None
        self._workspace_last_stream: Any = None
        self._borrowed_workspace: Optional[torch.Tensor] = None
        self._owner_thread: Optional[int] = None
        self._owner_device: Optional[int] = None
        self._owner_stream_identity: Any = None
        self._owner_stream: Any = None
        self._active_token: Optional[Token] = None
        self._borrowed = False
        self._lock = threading.Lock()

    @property
    def c_tmp(self) -> Optional[torch.Tensor]:
        return self._c_tmp

    @property
    def a_tmp(self) -> Optional[torch.Tensor]:
        return self._a_tmp

    @property
    def workspace(self) -> Optional[torch.Tensor]:
        return self._workspace

    @property
    def metadata_cache_size(self) -> int:
        return len(self._metadata)

    def __enter__(self) -> "MarlinScratchContext":
        active = _ACTIVE_CONTEXT.get()
        if active is not None:
            raise RuntimeError("MarlinScratchContext cannot be nested or re-entered")
        if self._active_token is not None:
            raise RuntimeError("MarlinScratchContext is already active")

        if self.device.type == "cuda" and self._is_capturing():
            raise RuntimeError(
                "MarlinScratchContext supports eager execution only and cannot be entered during CUDA graph capture"
            )

        owner_thread = threading.get_ident()
        owner_device = _device_index(self.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", owner_device)
        owner_stream = self._current_stream()
        owner_stream_identity = _stream_identity(owner_stream)
        if self._owner_thread is not None and owner_thread != self._owner_thread:
            raise RuntimeError("MarlinScratchContext cannot be reused from another thread")
        if (
            self._owner_device is not None
            and owner_device != self._owner_device
        ):
            raise RuntimeError("MarlinScratchContext CUDA device changed between entries")
        if (
            self._owner_stream_identity is not None
            and owner_stream_identity != self._owner_stream_identity
        ):
            raise RuntimeError("MarlinScratchContext CUDA stream changed between entries")
        if self.device.type == "cuda" and self.device.index is not None:
            if owner_device != self.device.index:
                raise ValueError(
                    f"MarlinScratchContext device {self.device} is not current CUDA device"
                )
        self._owner_thread = owner_thread
        self._owner_device = owner_device
        self._owner_stream = owner_stream
        self._owner_stream_identity = owner_stream_identity
        self._active_token = _ACTIVE_CONTEXT.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._active_token is None:
            raise RuntimeError("MarlinScratchContext is not active")
        token = self._active_token
        error: Optional[BaseException] = None
        try:
            self._check_owner()
            if self._borrowed:
                raise RuntimeError("MarlinScratchContext cannot exit while scratch is borrowed")
        except BaseException as exc:
            error = exc
        finally:
            self._active_token = None
            _ACTIVE_CONTEXT.reset(token)
        if error is not None:
            raise error

    def clear(self) -> None:
        """Release cached tensors after recording their last-use streams.

        ``clear`` never synchronizes.  It is safe to call while the context is
        active after a GEMM has returned, and it can also be used between
        sequential context entries.
        """

        if self._borrowed:
            raise RuntimeError("MarlinScratchContext cannot clear while scratch is borrowed")
        if self._active_token is not None:
            self._check_owner()
        elif self._owner_thread is not None and threading.get_ident() != self._owner_thread:
            raise RuntimeError("MarlinScratchContext cannot be cleared from another thread")
        self._release_cached()
        self._metadata.clear()

    def acquire(
        self,
        a: torch.Tensor,
        *,
        size_m: int,
        size_k: int,
        use_fp32_reduce: bool,
        has_act_order: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Borrow ``(c_tmp, a_tmp, workspace)`` for one GEMM invocation."""

        self._check_owner()
        if self._borrowed:
            raise RuntimeError("MarlinScratchContext scratch is already borrowed")
        if a.device != self.device and not (
            a.device.type == self.device.type == "cuda"
            and (a.device.index is None or self.device.index is None
                 or a.device.index == self.device.index)
        ):
            raise ValueError(
                f"MarlinScratchContext is for {self.device}, got activation on {a.device}"
            )
        if a.ndim < 2:
            raise ValueError("Marlin activation must have at least two dimensions")
        if a.shape[0] == 0 or a.shape[1] == 0:
            # The backend owns its empty-output short circuit.  In particular,
            # do not query capacities or allocate large buffers for M == 0.
            with self._lock:
                if self._borrowed:
                    raise RuntimeError("MarlinScratchContext scratch is already borrowed")
                self._borrowed = True
            try:
                return None, None, self._workspace_for_call()
            except BaseException:
                self._borrowed = False
                raise

        with self._lock:
            if self._borrowed:
                raise RuntimeError("MarlinScratchContext scratch is already borrowed")
            self._borrowed = True
        try:
            # The tensor shape is authoritative: padded callers pass their
            # post-padding A here, even if a logical K was retained elsewhere.
            actual_m = int(a.shape[0])
            actual_k = int(a.shape[1])
            key = (
                self.device.type,
                self.device.index,
                a.dtype,
                actual_m,
                actual_k,
                bool(use_fp32_reduce),
                bool(has_act_order),
            )
            sizes = self._metadata.get(key)
            if sizes is None:
                from .marlin import marlin_scratch_sizes

                sizes = tuple(
                    int(value)
                    for value in marlin_scratch_sizes(
                        a,
                        actual_m,
                        actual_k,
                        bool(use_fp32_reduce),
                        bool(has_act_order),
                    )
                )
                if len(sizes) != 2 or any(value < 0 for value in sizes):
                    raise RuntimeError(f"Invalid Marlin scratch sizes: {sizes!r}")
                self._metadata[key] = sizes
                while len(self._metadata) > _MAX_METADATA_ENTRIES:
                    self._metadata.popitem(last=False)
            else:
                self._metadata.move_to_end(key)

            c_count, a_count = sizes
            c_tmp, a_tmp = self._prepare_buffers(
                a.dtype,
                c_count if use_fp32_reduce else 0,
                a_count if has_act_order else 0,
            )
            workspace = self._workspace_for_call()
            return c_tmp, a_tmp, workspace
        except BaseException:
            self._borrowed = False
            raise

    def release(self) -> None:
        """Return a previously borrowed scratch lease to the context."""

        if not self._borrowed:
            raise RuntimeError("MarlinScratchContext scratch is not borrowed")
        if self._owner_thread is not None and threading.get_ident() != self._owner_thread:
            raise RuntimeError("MarlinScratchContext cannot be released from another thread")
        transient = self._borrowed_workspace
        self._borrowed_workspace = None
        if transient is not None:
            stream = self._last_stream if self._last_stream is not None else self._current_stream()
            _record_stream(transient, stream)
        self._borrowed = False

    def _check_owner(self) -> None:
        if self._active_token is None:
            raise RuntimeError("MarlinScratchContext must be active to invoke Marlin")
        current_thread = threading.get_ident()
        if current_thread != self._owner_thread:
            raise RuntimeError("MarlinScratchContext cannot be used from another thread")
        if self.device.type == "cuda":
            if self._is_capturing():
                raise RuntimeError(
                    "MarlinScratchContext supports eager execution only and cannot be used during CUDA graph capture"
                )
            current_device = _device_index(self.device)
            if self._owner_device is not None and current_device != self._owner_device:
                raise RuntimeError("MarlinScratchContext CUDA device changed while active")
            current_stream = self._current_stream()
            current_identity = _stream_identity(current_stream)
            if (
                self._owner_stream_identity is not None
                and current_identity != self._owner_stream_identity
            ):
                raise RuntimeError("MarlinScratchContext CUDA stream changed while active")

    def _is_capturing(self) -> bool:
        checker = getattr(torch.cuda, "is_current_stream_capturing", None)
        if checker is None:
            return False
        return bool(checker())

    def _current_stream(self) -> Any:
        if self.device.type != "cuda":
            return None
        return torch.cuda.current_stream(self.device)

    def _workspace_for_call(self) -> torch.Tensor:
        if self._workspace is not None:
            return self._workspace
        if self.device.type == "cuda":
            sms = int(torch.cuda.get_device_properties(self.device).multi_processor_count)
        else:
            sms = 1
        sms = max(sms, 1)
        self._workspace_bytes = sms * 4
        stream = self._current_stream()
        # A zero/tiny cache budget must still isolate module locks.  The
        # resulting workspace is transient and is released in ``release``.
        if self._workspace_bytes <= self.max_cached_bytes:
            self._workspace = torch.zeros(sms, dtype=torch.int32, device=self.device)
            self._workspace_last_stream = stream
            return self._workspace
        workspace = torch.zeros(sms, dtype=torch.int32, device=self.device)
        self._borrowed_workspace = workspace
        self._last_stream = stream
        return workspace

    def _prepare_buffers(
        self,
        dtype: torch.dtype,
        c_count: int,
        a_count: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._buffer_dtype is not None and self._buffer_dtype != dtype:
            self._release_temporary_buffers()
        self._buffer_dtype = dtype

        c_capacity = (
            self._c_tmp.numel()
            if self._c_tmp is not None and self._c_tmp.dtype == torch.float32
            else 0
        )
        a_capacity = (
            self._a_tmp.numel()
            if self._a_tmp is not None and self._a_tmp.dtype == dtype
            else 0
        )
        c_bytes = max(c_count, c_capacity) * 4
        a_bytes = max(a_count, a_capacity) * _dtype_itemsize(dtype)
        required_bytes = c_bytes + a_bytes
        lock_bytes = self._workspace_bytes or self._lock_workspace_bytes()
        # Lock storage is always part of the cache budget calculation.  For a
        # zero/tiny budget it is transient, but it still prevents caching the
        # other temporaries beyond the configured bound.
        if lock_bytes + required_bytes > self.max_cached_bytes:
            # Retain the bounded cache on an over-budget growth request and
            # still use whichever existing allocation already covers this
            # request.  Missing/undersized buffers remain None so the backend
            # allocates those portions temporarily.
            return (
                self._c_tmp if c_count > 0 and c_capacity >= c_count else None,
                self._a_tmp if a_count > 0 and a_capacity >= a_count else None,
            )

        if c_count > 0 and (
            self._c_tmp is None
            or self._c_tmp.dtype != torch.float32
            or self._c_tmp.numel() < c_count
        ):
            self._release_tensor("_c_tmp", self._last_stream)
            self._c_tmp = torch.empty(c_count, dtype=torch.float32, device=self.device)
        if a_count > 0 and (
            self._a_tmp is None
            or self._a_tmp.dtype != dtype
            or self._a_tmp.numel() < a_count
        ):
            self._release_tensor("_a_tmp", self._last_stream)
            self._a_tmp = torch.empty(a_count, dtype=dtype, device=self.device)
        stream = self._current_stream()
        self._last_stream = stream
        return (self._c_tmp if c_count > 0 else None,
                self._a_tmp if a_count > 0 else None)

    def _lock_workspace_bytes(self) -> int:
        if self.device.type == "cuda":
            sms = int(torch.cuda.get_device_properties(self.device).multi_processor_count)
        else:
            sms = 1
        return max(sms, 1) * 4

    def _release_tensor(self, attr: str, stream: Any) -> None:
        tensor = getattr(self, attr)
        if tensor is not None:
            _record_stream(tensor, stream)
            setattr(self, attr, None)

    def _release_temporary_buffers(self) -> None:
        self._release_tensor("_c_tmp", self._last_stream)
        self._release_tensor("_a_tmp", self._last_stream)
        self._last_stream = None

    def _release_cached(self) -> None:
        self._release_temporary_buffers()
        self._release_tensor("_workspace", self._workspace_last_stream)
        transient = self._borrowed_workspace
        self._borrowed_workspace = None
        if transient is not None:
            stream = self._last_stream if self._last_stream is not None else self._current_stream()
            _record_stream(transient, stream)
        self._workspace_last_stream = None
        self._workspace_bytes = 0
        self._buffer_dtype = None


__all__ = ["MarlinScratchContext", "active_marlin_scratch_context"]
