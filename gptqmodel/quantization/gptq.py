# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Based on original gptq algorithm and code from https://github.com/IST-DASLab/gptq

import contextlib
import math
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn.modules.conv import _ConvNd

from .._alloc_conf import large_hessian_alloc_hint
from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..quantization.config import FallbackStrategy, SmoothMSE
from ..utils.device import get_device
from ..utils.env import env_flag
from ..utils.kernel import resolve_kernel_config
from ..utils.logger import setup_logger
from ..utils.p2p import p2p_all_enabled, safe_move_to
from ..utils.rocm import quant_blas_backend
from .conv_padding import conv_padding_pairs, pad_mode
from .fallback_smooth import (
    affine_range_with_zero,
    element_saliency_weights,
    hessian_column_weights,
    mse_optimal_quant,
    smooth_block,
    symmetric_range,
)
from .gar import (
    compose_final_perm,
    compute_global_perm,
    compute_local_perms,
    extend_perm_with_tail,
    invert_perm,
)
from .gsq_scalar import gsq_enabled_for, refine_gptq_scalar
from .npu_linalg import npu_inverse_cholesky_factor
from .quantizer import (
    HF_OPTIMUM,
    SCALE_STORAGE_FLOOR,
    Quantizer,
    _floor_scale_for_storage,
    _neutralize_dead_columns,
    _replace_zero_scale,
)


log = setup_logger()


# Accelerator solver/allocator status codes that mean "allocation failed" even
# though the message does not contain the legacy "out of memory" phrase.
_MEMORY_ALLOCATION_ERROR_MARKERS = (
    "cusolver_status_alloc_failed",
    "hipsolver_status_alloc_failed",
    "cublas_status_alloc_failed",
    "hipblas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "miopen_status_out_of_resources",
    "cuda_error_out_of_memory",
    "cudaerroroutofmemory",
    "cudaerrormemoryallocation",
    "cuda_error_memory_allocation",
    "cudaerror_memoryallocation",
    "hiperroroutofmemory",
    "hiperrormemoryallocation",
    "hiperror_out_of_memory",
)


def _is_memory_allocation_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is an accelerator memory/allocation failure.

    The GPTQ CPU fallbacks used to match only ``"out of memory"``. cuSOLVER and
    hipSOLVER report a failed workspace allocation as
    ``CUSOLVER_STATUS_ALLOC_FAILED`` / ``HIPSOLVER_STATUS_ALLOC_FAILED``, so the
    exception type and solver status code are the primary discriminator and the
    historical phrase is only a last-resort supplement (issue #4350).

    A non-positive-definite factorization is a numerical failure, not a memory
    failure. ``torch._C._LinAlgError`` is excluded first so it can never be
    downgraded to the CPU retry.
    """

    if isinstance(exc, torch._C._LinAlgError):
        return False
    if isinstance(exc, (torch.OutOfMemoryError, MemoryError)):
        return True
    text = str(exc).lower()
    if any(marker in text for marker in _MEMORY_ALLOCATION_ERROR_MARKERS):
        return True
    return "out of memory" in text


def _log_hessian_verbose() -> bool:
    """Verbose per-module hessian-inverse markers are opt-in.

    The QuantizationRegionTimer measurement is always recorded and flushed
    periodically, which provides aggregate stall isolation without per-module
    line noise.
    """
    return env_flag("GPTQMODEL_LOG_HESSIAN")


# ``desc_act`` orders the columns by Hessian diagonal, so a handful of
# massive-activation rows can own the ordering and saturate the columns
# correlated with them (issue #4742). Rows above this percentile of the
# row-norm distribution are clamped to it before ``X^T X``.
SINK_ROW_CAP_PERCENTILE = 99.9


_WORKSPACE_LOCKS_GUARD = threading.Lock()


class _HessianAccumulatorState:
    """Reference-counted capture state shared by identical projection inputs."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.hessian: Optional[torch.Tensor] = None
        self.nsamples = 0
        self.device_partials: Dict[torch.device, torch.Tensor] = {}
        self.device_partial_ready: Dict[torch.device, Optional[torch.cuda.Event]] = {}
        self.ready_event: Optional[torch.cuda.Event] = None
        self.deferred_sources: list[Tuple[torch.Tensor, Optional[torch.cuda.Event]]] = []
        self.device_sample_counts: Dict[torch.device, int] = {}
        self.dirty = False
        self.final_device_hint: Optional[torch.device] = None
        self.users = 1

    def release(self) -> None:
        with self.lock:
            self.users -= 1
            if self.users < 0:
                raise RuntimeError("Hessian accumulator state was released twice")
            if self.users == 0:
                # Cross-device copies may still consume source tensors after
                # their producing projection has released its capture handle.
                for _, event in self.deferred_sources:
                    if event is not None:
                        event.synchronize()
                self.deferred_sources.clear()
                self.device_partial_ready.clear()
                self.ready_event = None
                self.hessian = None
                self.device_partials.clear()
                self.device_sample_counts.clear()
                self.dirty = False

# Shared workspaces are cached globally per device and dtype so concurrent
# GPTQ instances reuse temporary buffers instead of repeatedly allocating
# large tensors during Hessian accumulation. Each pool is bounded below.
# ``_WORKSPACE_CACHE`` remains a compatibility view of the most recently
# released tensor. The actual reusable slots live in ``_WORKSPACE_SLOTS``;
# a slot is reusable from another CUDA stream only after that stream waits
# on the release event.
_WORKSPACE_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_WORKSPACE_SLOTS: Dict[Tuple[Tuple[str, Optional[int]], torch.dtype], list["_WorkspaceSlot"]] = {}
_WORKSPACE_LOCKS: Dict[Tuple[str, Optional[int]], threading.Lock] = {}
_WORKSPACE_GENERATION = 0
_BF16_SUPPORT_CACHE: Dict[Tuple[str, Optional[int]], bool] = {}


@dataclass
class _WorkspaceSlot:
    tensor: torch.Tensor
    ready_event: Optional[torch.cuda.Event] = None


class SharedHessianFactorCache:
    """Single-flight, group-scoped cache for inverse Hessian factors.

    A cache is created for one shared-input capture cohort by
    :class:`GPTQProcessor`; it is intentionally never global.  Producers put
    tensors in the cache only after the computation succeeds.  CUDA consumers
    use the recorded event rather than synchronizing the host.
    """

    def __init__(self, generation: int, cohort: str, leader: Optional[str] = None):
        self.generation = int(generation)
        self.cohort = str(cohort)
        self.leader = str(leader if leader is not None else cohort)
        self._lock = threading.Lock()
        self._values: Dict[Tuple[Any, ...], Future] = {}

    def _publish(self, future: Future, value: Any) -> None:
        factor = value[0]
        device = factor.device if isinstance(factor, torch.Tensor) else None
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            event = torch.cuda.Event(blocking=False, interprocess=False)
            event.record(torch.cuda.current_stream(device))
            # Keep the event adjacent to the immutable result without wrapping
            # the public artifact in a mutable holder.
            future._gptq_ready_event = event  # type: ignore[attr-defined]
        future.set_result(value)

    def get_or_compute(
        self,
        key: Tuple[Any, ...],
        producer: Callable[[], Any],
    ) -> Any:
        with self._lock:
            future = self._values.get(key)
            if future is None:
                future = Future()
                self._values[key] = future
                owner = True
            else:
                owner = False

        if owner:
            try:
                self._publish(future, producer())
            except BaseException as exc:
                # Failed/OOM computations are never published as artifacts;
                # remove the failed single-flight so callers can fall back or
                # retry independently without poisoning the cohort cache.
                with self._lock:
                    if self._values.get(key) is future:
                        del self._values[key]
                future.set_exception(exc)
                raise
        value = future.result()
        factor = value[0]
        event = getattr(future, "_gptq_ready_event", None)
        if event is not None and isinstance(factor, torch.Tensor) and factor.device.type == "cuda":
            torch.cuda.current_stream(factor.device).wait_event(event)
        return value


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    dev = torch.device(device)
    return dev.type, dev.index


def _device_reduction_key(device: torch.device) -> Tuple[str, int]:
    """Order reductions by device identity, never worker arrival order."""
    dev = torch.device(device)
    return dev.type, -1 if dev.index is None else dev.index


def _workspace_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    return _device_cache_key(device)


def _workspace_lock(key: Tuple[str, Optional[int]]) -> threading.Lock:
    lock = _WORKSPACE_LOCKS.get(key)
    if lock is not None:
        return lock

    with _WORKSPACE_LOCKS_GUARD:
        lock = _WORKSPACE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _WORKSPACE_LOCKS[key] = lock
    return lock


def _needs_workspace_resize(
    workspace: Optional[torch.Tensor],
    dtype: torch.dtype,
    required_rows: int,
    cols: int,
) -> bool:
    if workspace is None:
        return True
    if workspace.ndim != 2:
        return True
    if workspace.dtype != dtype:
        return True
    if workspace.shape[1] != cols:
        return True
    if workspace.shape[0] < required_rows:
        return True
    return False


@contextlib.contextmanager
def _lease_workspace(
    device: torch.device,
    dtype: torch.dtype,
    cols: int,
    required_rows: int,
) -> Tuple[torch.Tensor, bool]:
    key = _workspace_cache_key(device)
    lock = _workspace_lock(key)
    slot_key = (key, dtype)
    with _WORKSPACE_LOCKS_GUARD:
        generation = _WORKSPACE_GENERATION
    with lock:
        # Test fixtures and callers historically clear ``_WORKSPACE_CACHE``;
        # treat that as clearing the event-gated pool too.
        if key not in _WORKSPACE_CACHE:
            _WORKSPACE_SLOTS.pop(slot_key, None)
        slots = _WORKSPACE_SLOTS.setdefault(slot_key, [])
        slot_index = next(
            (
                idx
                for idx, candidate in enumerate(slots)
                if not _needs_workspace_resize(candidate.tensor, dtype, required_rows, cols)
            ),
            None,
        )
        if slot_index is None:
            rows = max(required_rows, 1)
            workspace = torch.empty((rows, cols), dtype=dtype, device=device)
            slot = _WorkspaceSlot(workspace)
            reused = False
        else:
            slot = slots.pop(slot_index)
            workspace = slot.tensor
            reused = True
            if slot.ready_event is not None and torch.device(device).type == "cuda":
                # Wait for the stream that released this slot before reusing it.
                torch.cuda.current_stream(device).wait_event(slot.ready_event)
    try:
        if workspace.device.type == "cuda":
            # ``record_stream`` only extends allocator lifetime.  The event
            # below is the ordering primitive used for cross-stream reuse.
            workspace.record_stream(torch.cuda.current_stream(device))
        yield workspace, reused
    finally:
        with lock:
            if workspace.device.type == "cuda":
                stream = torch.cuda.current_stream(device)
                workspace.record_stream(stream)
                # Publish completion so a later stream can safely reuse the slot.
                slot.ready_event = torch.cuda.Event(blocking=False, interprocess=False)
                slot.ready_event.record(stream)
            else:
                slot.ready_event = None
            with _WORKSPACE_LOCKS_GUARD:
                released_while_leased = _WORKSPACE_GENERATION != generation
            if not released_while_leased:
                slots = _WORKSPACE_SLOTS.setdefault(slot_key, [])
                slots.append(slot)
                # Keep a small bounded pool (the default CUDA worker pool is four
                # workers/GPU) while allowing concurrent leases without a host
                # synchronization.  Extra slots are safe to release.
                if len(slots) > 8:
                    del slots[:-8]
                _WORKSPACE_CACHE[key] = workspace


def release_workspace_cache() -> None:
    """Drop every cached Hessian workspace slot.

    The slots are process-wide tensors, so they survive ``quantize()`` returning
    and the model being deleted, and ``torch.cuda.empty_cache()`` cannot
    reclaim them. A lease that is still running observes the generation change
    and drops its slot instead of repopulating the pool.
    """

    global _WORKSPACE_GENERATION
    with _WORKSPACE_LOCKS_GUARD:
        _WORKSPACE_GENERATION += 1
        keys = list(_WORKSPACE_LOCKS)
    for key in keys:
        with _workspace_lock(key):
            _WORKSPACE_CACHE.pop(key, None)
            for slot_key in [slot_key for slot_key in _WORKSPACE_SLOTS if slot_key[0] == key]:
                _WORKSPACE_SLOTS.pop(slot_key, None)


def _device_supports_bfloat16(device: torch.device) -> bool:
    cache_key = _device_cache_key(device)
    cached = _BF16_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dev = torch.device(device)
    if dev.type == "meta":
        _BF16_SUPPORT_CACHE[cache_key] = False
        return False

    try:
        a = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        b = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        _ = torch.matmul(a, b)
        support = True
    except Exception:
        support = False

    _BF16_SUPPORT_CACHE[cache_key] = support
    return support


def get_number_of_rows_and_cols(layer: nn.Module):
    # return layer.weight.shape[0], np.prod(layer.weight.shape[1:])
    if isinstance(layer, NamedModule):
        layer = layer.module

    if isinstance(layer, nn.Embedding):
        return layer.weight.shape[1], layer.weight.shape[0]

    if isinstance(layer, transformers.Conv1D):
        # transformers.Conv1D: weight shape is (n_in, n_out)
        return layer.weight.shape[1], layer.weight.shape[0]
    else:
        # weight shape is (n_out, n_in); math.prod keeps `columns` a plain int
        return layer.weight.shape[0], math.prod(layer.weight.shape[1:])


def _uniform_column_group_stream(device: torch.device):
    """Identify the stream whose ordering makes a cached zero-fill visible.

    The cached tensor's fill is queued on the stream that created it. A second
    worker on another stream could otherwise observe the tensor before that
    fill completes. Keying the cache by the stream object keeps one tensor per
    stream and keeps the stream alive for as long as the entry is cached.
    """

    if device.type == "cuda":
        return torch.cuda.current_stream(device)
    return None


@lru_cache(maxsize=32)
def _uniform_column_group_cached(
    count: int,
    device: torch.device,
    stream,
) -> torch.Tensor:
    del stream
    return torch.zeros(count, device=device, dtype=torch.int32)


def _uniform_column_group(count: int, device: torch.device) -> torch.Tensor:
    """Constant group index for blocks whose columns share one quantizer group."""

    return _uniform_column_group_cached(count, device, _uniform_column_group_stream(device))


def _move_static_groups(groups, device: torch.device):
    """Move static-group quantizers to ``device`` and return their parameters.

    Static groups measure ``scale``/``zero`` once, before activation ordering
    and the block pass. A CPU fallback that moves ``W``/``H`` must carry those
    quantizers along, otherwise the block pass mixes host weights with
    accelerator parameters (issue #4464). The returned lists replace the
    caller's ``scale``/``zero`` lists, whose entries still point at the old
    device tensors.
    """

    for group_quantizer in groups:
        group_quantizer.to(device=device)
    return (
        [group_quantizer.scale for group_quantizer in groups],
        [group_quantizer.zero for group_quantizer in groups],
    )


class GPTQ:
    @property
    def H(self) -> Optional[torch.Tensor]:
        return self._hessian_state.hessian

    @H.setter
    def H(self, value: Optional[torch.Tensor]) -> None:
        self._hessian_state.hessian = value

    @H.deleter
    def H(self) -> None:
        self._hessian_state.hessian = None

    @property
    def nsamples(self) -> int:
        return self._hessian_state.nsamples

    @nsamples.setter
    def nsamples(self, value: int) -> None:
        self._hessian_state.nsamples = int(value)

    @property
    def _device_hessian_partials(self) -> Dict[torch.device, torch.Tensor]:
        return self._hessian_state.device_partials

    @_device_hessian_partials.setter
    def _device_hessian_partials(
        self, value: Dict[torch.device, torch.Tensor]
    ) -> None:
        self._hessian_state.device_partials = value

    @property
    def _device_hessian_partial_ready(self) -> Dict[torch.device, Optional[torch.cuda.Event]]:
        return self._hessian_state.device_partial_ready

    @_device_hessian_partial_ready.setter
    def _device_hessian_partial_ready(self, value: Dict[torch.device, Optional[torch.cuda.Event]]) -> None:
        self._hessian_state.device_partial_ready = value

    @property
    def _hessian_ready_event(self) -> Optional[torch.cuda.Event]:
        return self._hessian_state.ready_event

    @_hessian_ready_event.setter
    def _hessian_ready_event(self, value: Optional[torch.cuda.Event]) -> None:
        self._hessian_state.ready_event = value

    @property
    def _deferred_hessian_sources(self) -> list[Tuple[torch.Tensor, Optional[torch.cuda.Event]]]:
        return self._hessian_state.deferred_sources

    @_deferred_hessian_sources.setter
    def _deferred_hessian_sources(self, value: list[Tuple[torch.Tensor, Optional[torch.cuda.Event]]]) -> None:
        self._hessian_state.deferred_sources = value

    @property
    def _device_sample_counts(self) -> Dict[torch.device, int]:
        return self._hessian_state.device_sample_counts

    @_device_sample_counts.setter
    def _device_sample_counts(self, value: Dict[torch.device, int]) -> None:
        self._hessian_state.device_sample_counts = value

    @property
    def _hessian_dirty(self) -> bool:
        return self._hessian_state.dirty

    @_hessian_dirty.setter
    def _hessian_dirty(self, value: bool) -> None:
        self._hessian_state.dirty = bool(value)

    @property
    def _final_hessian_device_hint(self) -> Optional[torch.device]:
        return self._hessian_state.final_device_hint

    @_final_hessian_device_hint.setter
    def _final_hessian_device_hint(
        self, value: Optional[torch.device]
    ) -> None:
        self._hessian_state.final_device_hint = (
            None if value is None else torch.device(value)
        )

    @staticmethod
    def resolve_module_source(module: nn.Module) -> nn.Module:
        """Resolve the dense module view GPTQ should quantize for one wrapper."""

        if isinstance(module, NamedModule):
            quant_source = module.state.get("quant_source_module")
            if isinstance(quant_source, nn.Module):
                return quant_source
            return module.module
        return module

    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig] = None, region_timer=None):
        self._hessian_state = _HessianAccumulatorState()
        self.lock = self._hessian_state.lock
        self._hessian_capture_enabled = True
        self._hessian_state_released = False
        self._hessian_accumulator_device: Optional[torch.device] = None
        self.region_timer = region_timer

        # self.num_tied_handles = 0
        # if qcfg.tied_gptq_handle is not None:
        #     qcfg.tied_gptq_handle.num_tied_handles += 1

        # Flags indicating issues
        # self.issue_zero_samples = False
        # self.issue_nan_hessian = False
        # self.issue_non_invertible = False

        # self.W = module.weight
        resolved_module = self.resolve_module_source(module)
        self.rows, self.columns = get_number_of_rows_and_cols(resolved_module)
        if isinstance(module, NamedModule):
            self.module = resolved_module
            self.name = module.name
            self._named_module = module
        else:
            self.name = HF_OPTIMUM
            self.module = resolved_module
            self._named_module = None

        self._original_rows = self.rows
        self._original_columns = self.columns
        if self._named_module is not None:
            pad_info = self._named_module.state.get("tp_pad_info")
        else:
            pad_info = getattr(self.module, "_tp_pad_info", None)
        if isinstance(pad_info, dict):
            pad_cols = int(pad_info.get("pad_cols", 0) or 0)
            pad_cols = max(pad_cols, 0)
        else:
            pad_info = None
            pad_cols = 0

        self._tp_pad_info = pad_info
        self._tp_pad_cols = pad_cols
        if self._tp_pad_cols:
            self.columns += self._tp_pad_cols

        module_device = get_device(self.module)
        setattr(self.module, "target_device", module_device)

        if module_device.type == "meta":
            self._final_hessian_device_hint = torch.device("cpu")
        else:
            self._final_hessian_device_hint = torch.device(module_device)

        self.validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig()  # HF compat will not pass qcfg
        self._validate_act_group_aware_shape()

        self.module_copy = None

        self.H = None
        self.nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        # fwd counter
        self.fwd_counter = 0

        self.fallback = self.qcfg.fallback
        self.expected_nsamples: Optional[float] = None

        self.H: Optional[torch.Tensor] = None
        self._H_diag: Optional[torch.Tensor] = None

        # Store per-device Hessian contributions so multi-GPU calibration can
        # keep local accumulators and merge only once when quantization begins.
        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        # Readiness events order stream-local writes before partial or final-H reads.
        self._device_hessian_partial_ready: Dict[torch.device, Optional[torch.cuda.Event]] = {}
        self._device_embedding_counts: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}
        self._hessian_total_samples: int = 0
        self._hessian_dirty: bool = False
        self._hessian_ready_event: Optional[torch.cuda.Event] = None
        # Keep asynchronous source tensors alive until their destination
        # stream has consumed them.  This is required for cross-device partial
        # reductions and leader->follower Hessian copies.
        self._deferred_hessian_sources: list[Tuple[torch.Tensor, Optional[torch.cuda.Event]]] = []
        self._shared_hessian_cache: Optional[SharedHessianFactorCache] = None
        self._shared_hessian_cache_disabled = False

        self._borrow_workspace_stats = {
            "requests": 0,
            "staging_requests": 0,
            "staging_hits": 0,
            "staging_misses": 0,
            "materialized_requests": 0,
            "materialized_hits": 0,
            "materialized_misses": 0,
        }
        self._borrow_workspace_totals = {
            "requests": 0,
            "materialized_hits": 0,
            "materialized_misses": 0,
            "staging_hits": 0,
            "staging_misses": 0,
        }
        self._borrow_workspace_last_summary: Optional[Dict[str, object]] = None
        self._borrow_workspace_stage_dtype: Optional[torch.dtype] = None
        self._borrow_workspace_last_chunk_rows: Optional[int] = None

    def share_hessian_state_from(self, source: "GPTQ") -> None:
        """Use one accumulator for projections fed by exactly the same rows."""

        if not isinstance(source, GPTQ) or source is self:
            raise TypeError("shared Hessian source must be another GPTQ capture")
        if self.columns != source.columns:
            raise ValueError("shared Hessian projections have different input widths")
        if (
            self.nsamples
            or self.H is not None
            or self._device_hessian_partials
            or self._device_sample_counts
        ):
            raise RuntimeError("cannot share a Hessian after capture has started")
        self._hessian_state.release()
        self._hessian_state = source._hessian_state
        self._hessian_state.users += 1
        self.lock = self._hessian_state.lock
        self._hessian_capture_enabled = False
        self._hessian_accumulator_device = source._hessian_accumulator_device

    def set_hessian_accumulator_device(
        self, device: torch.device | str | None
    ) -> None:
        target = None if device is None else torch.device(device)
        if (
            target is not None
            and target.type != "cpu"
            and target.index is None
        ):
            raise ValueError(
                f"GPTQ module `{getattr(self, 'name', '<unknown>')}`: the "
                "Hessian accumulator device "
                f"must name a concrete ordinal, got `{target}`. An index-less "
                "accelerator device resolves to the calling thread's current "
                "device, so data-parallel workers would key different GPUs "
                "under the same accumulator entry (#5093)."
            )
        current = self._hessian_accumulator_device
        if current is not None and target is not None and current != target:
            raise RuntimeError("shared Hessian accumulator ownership changed")
        self._hessian_accumulator_device = target
        if target is not None:
            self._final_hessian_device_hint = target

    @property
    def hessian_accumulator_device(self) -> Optional[torch.device]:
        return self._hessian_accumulator_device

    @property
    def hessian_is_shared(self) -> bool:
        return self._hessian_state.users > 1

    def _validate_act_group_aware_shape(self) -> None:
        if not getattr(self.qcfg, "act_group_aware", False):
            return

        group_size = int(getattr(self.qcfg, "group_size", -1) or -1)
        if group_size <= 0:
            raise ValueError(
                f"Quantization: Module `{self.name}` -> `act_group_aware=True` requires `group_size > 0`, "
                f"got `{group_size}`."
            )

    @staticmethod
    def validate_module(module):
        assert isinstance(module, (nn.Embedding, nn.Linear, nn.Conv1d, nn.Conv2d,
                                   transformers.Conv1D)), f"We supports only linear and convolutional layers. actual = `{module}`"
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            # Resolve the padding contract before calibration: a setting whose
            # im2col geometry cannot be reproduced must not silently capture a
            # Hessian for a different operator.
            conv_padding_pairs(module)

    # def has_hessian_issues(self) -> bool:
    #     return any([self.issue_zero_samples, self.issue_nan_hessian, self.issue_non_invertible])

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name)

    def shape(self):
        if hasattr(self, "module"):
            return self.module.weight.shape
        else:
            return (0, 0)

    def mock_hessian_inverse(self, H: torch.Tensor):
        """Mock hessian inverse for fast testing"""
        damp = self.qcfg.damp_percent
        # Return identity matrix instead of complex inversion
        identity = torch.eye(H.shape[0], dtype=torch.float32, device=H.device)
        return identity, damp

    def log_cpu_fallback(self, stage: str, source_device: torch.device) -> None:
        """Explain when a memory-heavy GPTQ step moves from CUDA to CPU."""

        log.warn(
            "Quantization: Module `%s` -> CUDA OOM during %s on %s; falling back to CPU. "
            "Due to this fallback, the calculation may take much longer than normal.",
            self.name,
            stage,
            source_device,
        )
        # A wide projection (e.g. a 40960-wide `down_proj`) needs three
        # `columns ** 2 * 4` buffers at once: the Hessian, the Cholesky factor
        # and the cuSOLVER workspace. The native caching allocator fails that
        # request on a busy multi-device run even with plenty of free memory.
        hint = large_hessian_alloc_hint()
        if hint is not None:
            log.warn("Quantization: Module `%s` -> %s", self.name, hint)

    def _run_with_accelerator_oom_cpu_fallback(
        self,
        compute: Callable[[], Any],
        *,
        source_device: torch.device,
        stage: str,
        prepare_cpu: Callable[[torch.device], None],
    ) -> Tuple[Any, bool]:
        """Run ``compute`` on the accelerator, retrying once on the host after an OOM.

        GPTAQ and FOEM re-implement the block pass but used to call
        ``hessian_inverse`` (and GPTAQ's native-delta product) without GPTQ's
        allocation guard, so a cuSOLVER/hipSOLVER workspace failure aborted a
        run GPTQ would have finished on CPU (issue #4604).

        ``prepare_cpu`` must move every tensor the retry reads (Hessian,
        weight, native delta, act-order permutations, static-group quantizers)
        to the host and rebind the caller's variables through ``nonlocal``.
        ``compute`` closes over those same variables, so its second invocation
        runs entirely on the host.
        """

        try:
            return compute(), False
        except RuntimeError as exc:
            if (
                source_device.type != "cuda"
                or not _is_memory_allocation_failure(exc)
            ):
                raise
            self.log_cpu_fallback(stage, source_device)
            cdu_device = torch.device("cpu")
            prepare_cpu(cdu_device)
            return compute(), True

    def _sanitize_scale_and_log(self, scale: torch.Tensor, zero: torch.Tensor):
        """Bypass zero scales and report the unstable module once per pass."""

        zero_mask = scale == 0
        if bool(zero_mask.any()):
            log.error(
                "Quantization: Module `%s` -> zero scale found and bypassed. "
                "This module may not be properly trained or received enough calibration data, "
                "causing unmanageable outliers. If this is a MoE module, please check with the "
                "model publisher and/or provide more calibration data.",
                self.name,
            )
        return _replace_zero_scale(scale, zero, zero_mask=zero_mask)

    def clone_module(self, copy=True, device: torch.device = None):
        if not device:
            device = self.module.weight.data.device

        # A cross-device clone is a peer copy; on an asymmetric ROCm link the
        # direct path segfaults, so go through the direction-aware helper.
        clone = safe_move_to(self.module.weight.data, device, copy=copy)

        if isinstance(self.module, nn.Embedding):
            clone = clone.t()

        if isinstance(self.module, _ConvNd):
            clone = clone.flatten(1)

        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        if self._tp_pad_cols:
            pad = torch.zeros(
                (clone.shape[0], self._tp_pad_cols),
                dtype=clone.dtype,
                device=clone.device,
            )
            clone = torch.cat((clone, pad), dim=1)

        return clone.float()

    @staticmethod
    def truncate_last_dim(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor

        trim = min(length, tensor.shape[-1])
        if trim == tensor.shape[-1]:
            return tensor

        return tensor.narrow(tensor.dim() - 1, 0, trim).contiguous()

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor, batch_index: Optional[int] = None):
        if not self._hessian_capture_enabled:
            return
        if isinstance(self.module, nn.Embedding):
            del out
            token_count, counts, device = self.process_batch(inp, batch_index=batch_index)
            if token_count == 0 or counts is None:
                return

            dev = torch.device(device)
            with self.lock:
                self.fwd_counter += 1
                existing = self._device_embedding_counts.get(dev)
                if existing is None:
                    self._device_embedding_counts[dev] = counts
                else:
                    self._wait_partial_ready(dev)
                    existing.add_(counts)
                self._record_partial_ready(dev)
                self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + token_count
                self.nsamples += token_count
                self._hessian_dirty = True
            return

        batch_token_size, xtx, device = self.process_batch(inp, batch_index=batch_index)
        if batch_token_size == 0 or xtx is None:
            return

        dev = torch.device(device)

        with self.lock:
            self.fwd_counter += 1

            existing = self._device_hessian_partials.get(dev)
            if existing is None:
                self._device_hessian_partials[dev] = xtx
            else:
                self._wait_partial_ready(dev)
                existing.add_(xtx)
                del xtx

            self._record_partial_ready(dev)

            self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + batch_token_size
            self.nsamples += batch_token_size
            self._hessian_dirty = True

    @staticmethod
    def _cuda_event_for(device: torch.device) -> Optional[torch.cuda.Event]:
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            return None
        return torch.cuda.Event(blocking=False, interprocess=False)

    def _wait_partial_ready(self, device: torch.device) -> None:
        event = self._device_hessian_partial_ready.get(torch.device(device))
        if event is not None:
            torch.cuda.current_stream(device).wait_event(event)

    def _record_partial_ready(self, device: torch.device) -> None:
        device = torch.device(device)
        event = self._cuda_event_for(device)
        if event is not None:
            event.record(torch.cuda.current_stream(device))
        # A later update or reduction waits for this partial's completion.
        self._device_hessian_partial_ready[device] = event

    def _wait_hessian_ready(self, consumer_device: torch.device) -> None:
        event = self._hessian_ready_event
        consumer_device = torch.device(consumer_device)
        if event is None:
            return
        if consumer_device.type == "cuda":
            # The event may have been recorded on another device.  Enqueue
            # the dependency on the stream that will consume the Hessian,
            # never on the producer/ready device's stream.
            torch.cuda.current_stream(consumer_device).wait_event(event)
        else:
            # A CPU consumer cannot enqueue a CUDA-stream dependency.  This
            # is an explicit cross-device consumption boundary, so wait only
            # for the producer event (never the whole device).
            event.synchronize()

    def _record_hessian_ready(self, device: torch.device) -> None:
        device = torch.device(device)
        event = self._cuda_event_for(device)
        if event is not None:
            event.record(torch.cuda.current_stream(device))
        # Consumers use this marker before reading the merged Hessian elsewhere.
        self._hessian_ready_event = event

    def _defer_hessian_sources(
        self,
        sources: list[torch.Tensor],
        event: Optional[torch.cuda.Event],
    ) -> None:
        if event is None:
            return
        # Keep sources alive until the consuming stream has finished reading them.
        self._deferred_hessian_sources.extend((source, event) for source in sources)

    def _reap_deferred_hessian_sources(self) -> None:
        if not self._deferred_hessian_sources:
            return
        pending = []
        for source, event in self._deferred_hessian_sources:
            # Poll completion so the hot path can release sources without syncing.
            if event is None or event.query():
                del source
            else:
                pending.append((source, event))
        self._deferred_hessian_sources = pending

    def _release_deferred_hessian_sources(self) -> None:
        """Synchronize and release sources before memory-heavy quantization."""

        if not self._deferred_hessian_sources:
            return
        for _, event in self._deferred_hessian_sources:
            if event is not None:
                event.synchronize()
        self._deferred_hessian_sources.clear()

    def preferred_staging_dtype(self, input_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        device = torch.device(device)

        staging_dtype = self.qcfg.hessian.staging_dtype
        if staging_dtype == torch.float32:
            return torch.float32

        if input_dtype not in (torch.float16, torch.bfloat16):
            return torch.float32

        if staging_dtype == torch.bfloat16:
            if not _device_supports_bfloat16(device):
                return torch.float32
            return torch.bfloat16

        if staging_dtype == torch.float16:
            # float16 has a much narrower exponent range than bfloat16, so a
            # bf16 activation outside +/-65504 becomes inf when staged. Never
            # stage to a narrower-range dtype than the input.
            if input_dtype == torch.bfloat16:
                if not _device_supports_bfloat16(device):
                    return torch.float32
                return torch.bfloat16
            return torch.float16

        return torch.float32

    def resolve_hessian_chunk_size(self, rows: int, stage_dtype: torch.dtype) -> Optional[int]:
        if rows == 0:
            return None

        cfg_chunk = self.qcfg.hessian.chunk_size
        if cfg_chunk is not None:
            return max(1, min(cfg_chunk, rows))

        bytes_budget = self.qcfg.hessian.chunk_bytes
        if bytes_budget is not None:
            bytes_per_row = self.columns * torch.tensor([], dtype=stage_dtype).element_size()
            if bytes_per_row > 0:
                chunk_rows = bytes_budget // bytes_per_row
                if chunk_rows > 0:
                    return max(1, min(int(chunk_rows), rows))
            return 1

        return None

    @contextlib.contextmanager
    def borrow_materialized_chunk_fp32(
        self,
        chunk: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        if rows == 0:
            yield chunk.new_zeros((0, self.columns), dtype=torch.float32)
            return

        device = chunk.device
        stage_dtype = self.preferred_staging_dtype(chunk.dtype, device)

        stats = self._borrow_workspace_stats
        stats["requests"] += 1

        with _lease_workspace(device, stage_dtype, self.columns, rows) as (
            staging_workspace,
            staging_reused,
        ):
            stats["staging_requests"] += 1
            if staging_reused:
                stats["staging_hits"] += 1
            else:
                stats["staging_misses"] += 1

            staging_view = staging_workspace[:rows, :]
            staging_view.copy_(chunk.to(dtype=stage_dtype))

            if stage_dtype == torch.float32:
                stats["materialized_requests"] += 1
                if staging_reused:
                    stats["materialized_hits"] += 1
                else:
                    stats["materialized_misses"] += 1

                try:
                    yield staging_view
                finally:
                    # The lease records a completion event; no host wait is
                    # needed before returning the slot to the pool.
                    pass
            else:
                with _lease_workspace(
                    device,
                    torch.float32,
                    self.columns,
                    rows,
                ) as (
                    fp32_workspace,
                    fp32_reused,
                ):
                    stats["materialized_requests"] += 1
                    if fp32_reused:
                        stats["materialized_hits"] += 1
                    else:
                        stats["materialized_misses"] += 1

                    try:
                        fp32_view = fp32_workspace[:rows, :]
                        fp32_view.copy_(staging_view.to(torch.float32))
                        yield fp32_view
                    finally:
                        pass

    def sink_row_cap_median_multiple(self) -> Optional[float]:
        """Return the sink-row cap multiplier, or ``None`` when it does not apply.

        ``desc_act=True`` disables ``act_group_aware``, and with it the
        per-group scale protection that keeps a few massive-activation
        ("attention sink") rows from dominating ``X^T X``. Those rows then own
        the largest Hessian diagonal entries, so ``desc_act`` quantizes their
        columns first with a tiny ``Hinv[i, i]``; the resulting error is
        amplified into every column correlated with the sink and saturates it
        (issue #4742). Capping each row's norm bounds that contribution while
        keeping the sink direction in the Hessian.

        The cap is a no-op whenever ``act_group_aware`` is active (the default
        GPTQ path) so that configuration stays bit-identical.
        """

        qcfg = getattr(self, "qcfg", None)
        if qcfg is None or not qcfg.desc_act or qcfg.act_group_aware:
            return None

        multiple = getattr(qcfg.hessian, "sink_row_cap_median_multiple", None)
        if multiple is None:
            return None
        multiple = float(multiple)
        if multiple <= 0.0:
            return None
        return multiple

    def _sink_row_norms(self, matrix: torch.Tensor, chunk_size: Optional[int]) -> torch.Tensor:
        """Return per-row fp32 L2 norms, staged chunk-by-chunk (device-only)."""

        rows = matrix.shape[0]
        norms = torch.empty(rows, dtype=torch.float32, device=matrix.device)
        # Keep the fp32 staging bounded even when the caller accumulates the
        # Hessian in one chunk.
        step = chunk_size if chunk_size is not None else max(1, (1 << 20) // max(1, matrix.shape[1]))
        step = min(step, rows)
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            norms[start:stop] = torch.linalg.vector_norm(
                matrix[start:stop], ord=2, dim=1, dtype=torch.float32
            )
        return norms

    def resolve_sink_row_cap_threshold(
        self, matrix: torch.Tensor, chunk_size: Optional[int]
    ) -> Optional[torch.Tensor]:
        """Return the scalar sink-row norm cap threshold, or ``None``.

        The threshold is ``min(p99.9 of the row norms, multiple x median)``.
        The percentile alone is not enough on a small batch, where it *is* the
        largest row (two sink rows in a 64-row batch sit above p99.9); the
        median multiple then supplies the bound. On a large calibration window
        the percentile is the binding term and the cap only touches the few
        extreme rows.

        The threshold is derived from the whole batch, not per chunk, so the
        result does not depend on ``HessianConfig.chunk_size``. Row norms are
        computed chunk-by-chunk in fp32, so the whole matrix is never staged in
        fp32 just to take the norms.

        Everything is a tensor op: ``compute_hessian_xtx`` runs inside the layer
        forward hooks, so a ``bool()``/``float()`` here would drain the CUDA
        queue on every ``add_batch`` (issue #4950). A non-finite or non-positive
        threshold becomes ``+inf`` (which makes every scale 1) instead of an
        early ``None``, so the decision never reads the host.
        """

        multiple = self.sink_row_cap_median_multiple()
        if multiple is None:
            return None

        rows = matrix.shape[0]
        if rows == 0:
            return None

        norms = self._sink_row_norms(matrix, chunk_size)
        # One device sort serves both quantiles: two ``kthvalue`` calls cost
        # ~1.5x more and are the only per-batch work the cap still adds.
        ordered = torch.sort(norms).values
        median = ordered[(rows + 1) // 2 - 1]
        rank = max(1, min(rows, math.ceil(SINK_ROW_CAP_PERCENTILE / 100.0 * rows)))
        percentile = ordered[rank - 1]
        threshold = torch.minimum(percentile, median * multiple)
        # ``main`` returned ``None`` (no cap) on any non-finite row norm or a
        # non-positive threshold. Folding that into the tensor keeps the
        # decision identical without a host read.
        usable = (
            torch.isfinite(norms).all()
            & torch.isfinite(threshold)
            & (threshold > 0.0)
        )
        return torch.where(usable, threshold, torch.full_like(threshold, float("inf")))

    def _sink_row_cap_scale_from_threshold(
        self,
        matrix: torch.Tensor,
        threshold: torch.Tensor,
        chunk_size: Optional[int],
    ) -> torch.Tensor:
        """Turn a scalar cap threshold into a per-row clamp scale (device-only)."""

        norms = self._sink_row_norms(matrix, chunk_size)
        usable_rows = torch.isfinite(norms) & (norms > 0.0)
        ratio = torch.clamp(
            threshold / norms.clamp_min(torch.finfo(torch.float32).tiny),
            max=1.0,
        )
        return torch.where(usable_rows, ratio, torch.ones_like(norms))

    def resolve_sink_row_cap_scale(self, matrix: torch.Tensor, chunk_size: Optional[int]) -> Optional[torch.Tensor]:
        """Return a per-row clamp scale that caps sink-row norms, or ``None``.

        This is a pure function of ``matrix`` (used by the #4742 regression
        tests). ``compute_hessian_xtx`` resolves the threshold and the scale
        separately so the threshold never depends on the chunk layout.
        """

        threshold = self.resolve_sink_row_cap_threshold(matrix, chunk_size)
        if threshold is None:
            return None
        return self._sink_row_cap_scale_from_threshold(matrix, threshold, chunk_size)

    def compute_hessian_xtx(self, matrix: torch.Tensor) -> torch.Tensor:
        rows = matrix.shape[0]
        if rows == 0:
            return torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)

        stage_dtype = self.preferred_staging_dtype(matrix.dtype, matrix.device)
        chunk_size = self.resolve_hessian_chunk_size(rows, stage_dtype)
        self._borrow_workspace_stage_dtype = stage_dtype
        self._borrow_workspace_last_chunk_rows = chunk_size if chunk_size is not None else rows

        threshold = self.resolve_sink_row_cap_threshold(matrix, chunk_size)
        cap_scale = (
            self._sink_row_cap_scale_from_threshold(matrix, threshold, chunk_size)
            if threshold is not None
            else None
        )

        if chunk_size is None:
            needs_copy = matrix.dtype != torch.float32
            mat32 = matrix.to(dtype=torch.float32)
            if cap_scale is not None:
                # Fold the cap into the fp32 GEMM input instead of switching
                # to the chunk-workspace lease: an all-ones scale is a
                # bitwise no-op, and a capped batch keeps the single-GEMM
                # fast path (issue #4950).
                # ``matrix.to(float32)`` returns the caller's tensor unchanged
                # when it is already fp32 (and activations may be shared by
                # several projections), so this must be out of place.
                if needs_copy:
                    mat32.mul_(cap_scale.unsqueeze(1))
                else:
                    mat32 = mat32 * cap_scale.unsqueeze(1)
            # Determinism across processes comes from this fp32 X^T X GEMM
            # running on rocBLAS (see ``utils/rocm.py``). The quantization run
            # already holds the override for Hessian methods; this per-call
            # context keeps a direct Hessian computation on the deterministic
            # path too, and is a reference-counted no-op when nested.
            with quant_blas_backend(matrix.device):
                xtx = torch.matmul(mat32.T, mat32)
            del mat32
            return xtx

        xtx_accum = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)

        for start in range(0, rows, chunk_size):
            rows_this = min(chunk_size, rows - start)
            source = matrix[start:start + rows_this]
            with self.borrow_materialized_chunk_fp32(source, rows_this) as materialized:
                materialized32 = materialized
                if cap_scale is not None:
                    # ``materialized`` is a private workspace copy, so the
                    # in-place clamp never touches the caller's activations.
                    materialized32.mul_(cap_scale[start:start + rows_this].unsqueeze(1))
                with quant_blas_backend(matrix.device):
                    xtx_accum.add_(torch.matmul(materialized32.T, materialized32))

        return xtx_accum

    def process_batch(
        self, inp: torch.Tensor, batch_index: Optional[int] = None
    ) -> Tuple[int, Optional[torch.Tensor], torch.device]:
        """Return ``(token_count, xtx_or_counts, device)`` for one calibration batch.

        ``batch_index`` is the native-input pairing hook used by GPTAQ/FOEM;
        plain GPTQ does not use it.
        """
        del batch_index
        inp_device = get_device(inp)

        if isinstance(self.module, nn.Embedding):
            token_ids = inp.reshape(-1).to(torch.long)
            token_count = token_ids.numel()
            device = torch.device(inp_device)
            if token_count == 0:
                return 0, None, device

            min_id, max_id = torch.aminmax(token_ids)
            if min_id.item() < 0 or max_id.item() >= self.columns:
                raise ValueError(
                    f"Embedding calibration token IDs must be in [0, {self.columns}); "
                    f"observed range [{min_id.item()}, {max_id.item()}]."
                )
            counts = torch.bincount(token_ids, minlength=self.columns).to(dtype=torch.float32)
            return token_count, counts, device

        #inp = inp.to(device=self.module.target_device, dtype=torch.float32)

        # input reshaping
        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            reshaped_inp = inp.reshape(-1, inp.shape[-1])
        else:
            if isinstance(self.module, nn.Conv1d):
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    1,
                )
                # `nn.Unfold` only accepts integer zero padding, so apply the
                # layer's own padding (including its mode) and unfold the
                # already-padded input. `padding="same"` is resolved to
                # PyTorch's asymmetric split by `conv_padding_pairs`.
                (left, right), = conv_padding_pairs(self.module)
                reshaped_inp = F.pad(
                    reshaped_inp,
                    (0, 0, left, right),
                    mode=pad_mode(self.module.padding_mode),
                )
                unfold = nn.Unfold(
                    self.module.kernel_size + (1,),
                    dilation=self.module.dilation + (1,),
                    padding=0,
                    stride=self.module.stride + (1,),
                )
                # output size (batch_size, channels * \prod kernel_size, num_patches)
                reshaped_inp = unfold(reshaped_inp)
            else:
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    inp.shape[3],
                )
                # `conv_padding_pairs` reports pairs in the module's spatial
                # order (height, width); `F.pad` wants the last dimension first.
                (top, bottom), (left, right) = conv_padding_pairs(self.module)
                reshaped_inp = F.pad(
                    reshaped_inp,
                    (left, right, top, bottom),
                    mode=pad_mode(self.module.padding_mode),
                )
                unfold = nn.Unfold(
                    self.module.kernel_size,
                    dilation=self.module.dilation,
                    padding=0,
                    stride=self.module.stride,
                )
                # output size (batch_size, channels * \prod kernel_size, num_patches)
                reshaped_inp = unfold(reshaped_inp)
            reshaped_inp = reshaped_inp.transpose(1, 2).flatten(0, 1)

        # Delay dtype conversion until we materialize Hessian chunks to avoid unnecessary temporaries
        reshaped_inp = reshaped_inp.contiguous()
        if self._tp_pad_cols:
            pad = reshaped_inp.new_zeros((reshaped_inp.shape[0], self._tp_pad_cols))
            reshaped_inp = torch.cat((reshaped_inp, pad), dim=1)
            del pad
        canonical_device = torch.device(inp_device)

        accumulator_device = self._hessian_accumulator_device
        if accumulator_device is not None and canonical_device != accumulator_device:
            reshaped_inp = reshaped_inp.to(
                device=accumulator_device,
                non_blocking=False,
            )
            canonical_device = accumulator_device

        batch_token_size = reshaped_inp.shape[0]

        if batch_token_size == 0:
            del reshaped_inp
            return 0, None, canonical_device

        try:
            xtx = self.compute_hessian_xtx(reshaped_inp).to(dtype=torch.float32)
        except RuntimeError as exc:
            if (
                torch.device(inp_device).type == "cuda"
                and _is_memory_allocation_failure(exc)
            ):
                log.warn(
                    "GPTQ module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
                    getattr(self, "name", "<unknown>"),
                )
                reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
                del reshaped_inp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                canonical_device = torch.device("cpu")
                xtx = self.compute_hessian_xtx(reshaped_inp_cpu).to(dtype=torch.float32)
                xtx = xtx.detach()
                del reshaped_inp_cpu
            else:
                del reshaped_inp
                raise
        else:
            xtx = xtx.detach()
            del reshaped_inp

        self._snapshot_borrow_workspace_stats(context="process_batch")
        return batch_token_size, xtx, canonical_device

    def _select_hessian_target_device(self, requested: Optional[torch.device]) -> torch.device:
        with self.lock:
            if requested is not None:
                return torch.device(requested)

            hint = getattr(self, "_final_hessian_device_hint", None)
            if hint is not None:
                return torch.device(hint)

            if self._device_hessian_partials:
                partial_device = next(iter(self._device_hessian_partials.keys()))
                return torch.device(partial_device)

            if self._device_embedding_counts:
                partial_device = next(iter(self._device_embedding_counts.keys()))
                return torch.device(partial_device)

            return torch.device("cpu")

    def _captured_hessian_diagonal(
        self, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Fold the captured Hessian state into its diagonal.

        The low-sample fallback only needs the per-channel input energy, so
        reduce the per-device ``XtX`` partials straight to a ``columns``
        vector instead of materializing the ``columns ** 2`` matrix that
        ``finalize_hessian`` would build (#5300). The result follows the same
        weighting as ``finalize_hessian``: an existing materialized ``H``
        contributes with its own sample count and each partial contributes
        ``2 / total_samples``.
        """

        with self.lock:
            self._reap_deferred_hessian_sources()
            previous_samples = self._hessian_total_samples
            new_samples = sum(self._device_sample_counts.values())
            total_samples = previous_samples + new_samples
            if total_samples == 0:
                return None

            diagonal = torch.zeros(self.columns, dtype=torch.float32, device=device)
            if (
                previous_samples
                and self.H is not None
                and self.H.dim() == 2
                and self.H.shape[0] == self.columns
            ):
                source = self.H
                if source.device != device or source.dtype != torch.float32:
                    source = safe_move_to(source, device, dtype=torch.float32)
                diagonal.add_(
                    source.diagonal(),
                    alpha=float(previous_samples) / float(total_samples),
                )

            for partial_device in sorted(
                self._device_hessian_partials, key=_device_reduction_key
            ):
                partial = self._device_hessian_partials[partial_device]
                event = self._device_hessian_partial_ready.get(
                    torch.device(partial_device)
                )
                if event is not None:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).wait_event(event)
                    else:
                        event.synchronize()
                if partial.device != device or partial.dtype != torch.float32:
                    partial = safe_move_to(partial, device, dtype=torch.float32)
                diagonal.add_(partial.diagonal(), alpha=2.0 / float(total_samples))
            return diagonal

    def materialize_global_hessian(self, target_device: Optional[torch.device] = None) -> None:
        with self.lock:
            self._reap_deferred_hessian_sources()
            # Select the destination under the same lock as partial-state reads;
            # this closes the GIL=0 window between device selection and merge.
            device = self._select_hessian_target_device(target_device)
            if isinstance(self.module, nn.Embedding):
                if not self._hessian_dirty and self._H_diag is not None:
                    self._wait_hessian_ready(device)
                    if self._H_diag.device != device:
                        source_diag = self._H_diag
                        self._H_diag = safe_move_to(self._H_diag, device)
                        self._record_hessian_ready(device)
                        self._defer_hessian_sources([source_diag], self._hessian_ready_event)
                    self.H = None
                    self.nsamples = self._hessian_total_samples
                    self._final_hessian_device_hint = device
                    return

                previous_samples = self._hessian_total_samples
                new_samples = sum(self._device_sample_counts.values())
                total_samples = previous_samples + new_samples
                reduction_sources = list(self._device_embedding_counts.values())
                source_diag: Optional[torch.Tensor] = None

                if self._H_diag is None or self._H_diag.shape != (self.columns,):
                    diag = torch.zeros(self.columns, dtype=torch.float32, device=device)
                    previous_samples = 0
                else:
                    existing_diag = self._H_diag
                    self._wait_hessian_ready(device)
                    diag = safe_move_to(existing_diag, device, dtype=torch.float32)
                    if diag is not existing_diag:
                        source_diag = existing_diag

                if total_samples == 0:
                    diag.zero_()
                else:
                    if previous_samples:
                        diag.mul_(float(previous_samples) / float(total_samples))
                    else:
                        diag.zero_()
                    for partial_device in sorted(self._device_embedding_counts, key=_device_reduction_key):
                        counts = self._device_embedding_counts[partial_device]
                        partial_event = self._device_hessian_partial_ready.get(torch.device(partial_device))
                        if partial_event is not None:
                            if device.type == "cuda":
                                torch.cuda.current_stream(device).wait_event(partial_event)
                            else:
                                partial_event.synchronize()
                        diag.add_(
                            safe_move_to(counts, device, dtype=torch.float32),
                            alpha=2.0 / float(total_samples),
                        )

                self._H_diag = diag
                self.H = None
                self.nsamples = total_samples
                self._hessian_total_samples = total_samples
                self._hessian_dirty = False
                self._final_hessian_device_hint = device
                self._device_embedding_counts.clear()
                self._device_sample_counts.clear()
                self._device_hessian_partial_ready.clear()
                self._record_hessian_ready(device)
                if source_diag is not None:
                    reduction_sources.insert(0, source_diag)
                self._defer_hessian_sources(reduction_sources, self._hessian_ready_event)
                return

            if not self._hessian_dirty and self.H is not None:
                self._wait_hessian_ready(device)
                if self.H.device != device:
                    source_hessian = self.H
                    self.H = safe_move_to(self.H, device)
                    self._record_hessian_ready(device)
                    self._defer_hessian_sources([source_hessian], self._hessian_ready_event)
                return

            previous_samples = self._hessian_total_samples
            new_samples = sum(self._device_sample_counts.values())
            total_samples = previous_samples + new_samples

            self._wait_hessian_ready(device)

            # Reuse the existing tensor when possible to avoid an extra allocation.
            reuse_buffer = (
                self.H is not None
                and self.H.shape == (self.columns, self.columns)
                and self.H.device == device
            )

            result_accum: torch.Tensor
            if reuse_buffer and self.H.dtype == torch.float32:
                result_accum = self.H
                if previous_samples:
                    result_accum.mul_(float(previous_samples) / float(total_samples))
                else:
                    result_accum.zero_()
            else:
                result_accum = torch.zeros(
                    (self.columns, self.columns),
                    dtype=torch.float32,
                    device=device,
                )
                if previous_samples and self.H is not None:
                    source_hessian = self.H
                    previous = safe_move_to(
                        source_hessian, device, dtype=torch.float32
                    )
                    result_accum.add_(
                        previous,
                        alpha=float(previous_samples) / float(total_samples),
                    )
                    if previous is not source_hessian:
                        self._defer_hessian_sources(
                            [source_hessian], self._hessian_ready_event
                        )

            if total_samples == 0:
                self.H = result_accum
                self.nsamples = 0
                self._hessian_total_samples = 0
                self._hessian_dirty = False
                self._final_hessian_device_hint = device
                self._device_hessian_partials.clear()
                self._device_sample_counts.clear()
                self._device_hessian_partial_ready.clear()
                self._record_hessian_ready(device)
                return

            # GPU workers can finish in a different order after restart. A
            # stable reduction order avoids changing floating-point rounding.
            reduction_sources = list(self._device_hessian_partials.values())
            # Reuse one staging buffer instead of allocating a temporary per
            # partial. With P2P enabled the copy is direct GPU-to-GPU; without
            # it the regular ``to(device)`` fallback below still applies.
            reduction_staging = None
            if result_accum.device.type == "cuda" and len(reduction_sources) > 1:
                try:
                    if p2p_all_enabled(
                        [
                            *[partial.device for partial in reduction_sources],
                            result_accum.device,
                        ]
                    ):
                        # Avoid turning a memory-tight layer into an avoidable
                        # OOM: only allocate the staging buffer when the device
                        # has room for it.
                        staging_bytes = result_accum.numel() * result_accum.element_size()
                        free_bytes, _ = torch.cuda.mem_get_info(result_accum.device)
                        if free_bytes > staging_bytes * 1.1:
                            reduction_staging = torch.empty_like(result_accum)
                except Exception:
                    reduction_staging = None
            for partial_device in sorted(self._device_hessian_partials, key=_device_reduction_key):
                partial = self._device_hessian_partials[partial_device]
                partial_device = torch.device(partial_device)
                partial_event = self._device_hessian_partial_ready.get(partial_device)
                if partial_event is not None:
                    if result_accum.device.type == "cuda":
                        torch.cuda.current_stream(result_accum.device).wait_event(partial_event)
                    else:
                        partial_event.synchronize()
                if partial.device != result_accum.device or partial.dtype != torch.float32:
                    if reduction_staging is not None and partial.device.type == "cuda":
                        reduction_staging.copy_(partial, non_blocking=True)
                        result_accum.add_(
                            reduction_staging,
                            alpha=2.0 / float(total_samples),
                        )
                    else:
                        result_accum.add_(
                            safe_move_to(partial, result_accum.device, dtype=torch.float32),
                            alpha=2.0 / float(total_samples),
                        )
                else:
                    result_accum.add_(partial, alpha=2.0 / float(total_samples))
            del reduction_staging

            self.H = result_accum
            self.nsamples = total_samples
            self._hessian_total_samples = total_samples
            self._hessian_dirty = False
            self._final_hessian_device_hint = result_accum.device
            self._device_hessian_partials.clear()
            self._device_hessian_partial_ready.clear()
            self._device_sample_counts.clear()
            self._record_hessian_ready(result_accum.device)
            self._defer_hessian_sources(reduction_sources, self._hessian_ready_event)
            del result_accum

    def adopt_hessian_from(self, leader: "GPTQ") -> None:
        """Replace this task's Hessian statistics with a private copy of ``leader``'s.

        Both modules must consume the identical input tensor (see ``:in=<tag>`` in
        ``module_tree``). The copy is independent because :meth:`quantize` mutates
        ``self.H`` in place and :meth:`free` drops it.
        """
        if leader is self:
            return
        if leader.columns != self.columns:
            raise ValueError(
                f"GPTQ: cannot share Hessian from `{leader.name}` ({leader.columns} columns) "
                f"with `{self.name}` ({self.columns} columns)."
            )

        leader.materialize_global_hessian()
        with leader.lock:
            source = leader.H
            nsamples = leader.nsamples
            fwd_counter = leader.fwd_counter
            if source is None:
                source = leader.create_H(None)

            target_device = self._select_hessian_target_device(getattr(self.module, "target_device", None))
            # The follower's destination stream consumes ``source``.  Queue
            # the leader readiness dependency there before an asynchronous
            # cross-device copy, rather than waiting on the leader stream.
            leader._wait_hessian_ready(target_device)
            copy = safe_move_to(
                source.detach(),
                target_device,
                dtype=torch.float32,
                copy=True,
                non_blocking=(target_device.type == "cuda" and source.device.type == "cuda"),
            )

        with self.lock:
            self.H = copy
            self.nsamples = nsamples
            self._hessian_total_samples = nsamples
            self.fwd_counter = fwd_counter
            self._device_hessian_partials.clear()
            self._device_hessian_partial_ready.clear()
            self._device_sample_counts.clear()
            self._hessian_dirty = False
            self._final_hessian_device_hint = copy.device
            self._record_hessian_ready(copy.device)
            self._defer_hessian_sources([source], self._hessian_ready_event)

        # Adoption is the last consumer of the leader's reduction partials and
        # of the follower's reference to ``leader.H``.  Release both now rather
        # than at each task's own ``quantize()``: otherwise every leader/
        # follower pair keeps leader H + leader partial + follower copy alive
        # simultaneously until quantization begins.  Both registries only hold
        # sources that a consuming stream may still be reading, so the
        # synchronizing release is safe here and drops no live Hessian.
        with leader.lock:
            leader._release_deferred_hessian_sources()
        with self.lock:
            self._release_deferred_hessian_sources()

    def attach_shared_hessian_cache(self, cache: Optional[SharedHessianFactorCache]) -> None:
        """Attach the current layer/group cache to this task."""

        with self.lock:
            self._shared_hessian_cache = cache
            self._shared_hessian_cache_disabled = False

    def clear_shared_hessian_cache(self) -> None:
        with self.lock:
            self._shared_hessian_cache = None
            self._shared_hessian_cache_disabled = False

    def _shared_factor_key(self, ordering: str) -> Tuple[Any, ...]:
        """Return a complete same-device compatibility key for a factor."""

        hessian = self.qcfg.hessian
        return (
            self._shared_hessian_cache.generation if self._shared_hessian_cache else None,
            self._shared_hessian_cache.cohort if self._shared_hessian_cache else None,
            self._shared_hessian_cache.leader if self._shared_hessian_cache else None,
            tuple(self.H.shape) if self.H is not None else None,
            self.H.dtype if self.H is not None else None,
            # Include device identity so a factor is reused only on its device.
            self.H.device.type if self.H is not None else None,
            self.H.device.index if self.H is not None else None,
            ordering,
            int(self.qcfg.group_size),
            bool(self.qcfg.desc_act),
            bool(self.qcfg.act_group_aware),
            float(self.qcfg.damp_percent),
            float(self.qcfg.damp_auto_increment),
            bool(self.qcfg.mock_quantization),
            str(hessian.staging_dtype),
            hessian.chunk_size,
            hessian.chunk_bytes,
        )

    def _shared_factor_or_local(self, ordering: str, producer: Callable[[], Any]) -> Any:
        cache = self._shared_hessian_cache
        if cache is None or self._shared_hessian_cache_disabled:
            return producer()
        try:
            return cache.get_or_compute(
                self._shared_factor_key(ordering),
                producer,
            )
        except RuntimeError as exc:
            # Memory/CPU fallback paths must not leave a partially computed
            # artifact visible to sibling tasks.
            if _is_memory_allocation_failure(exc):
                self._shared_hessian_cache_disabled = True
            raise

    def _disable_shared_hessian_cache(self) -> None:
        self._shared_hessian_cache_disabled = True

    def finalize_hessian(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        self.materialize_global_hessian(target_device=target_device)
        if isinstance(self.module, nn.Embedding):
            assert self._H_diag is not None
            return self._H_diag
        if self.H is None:
            self.H = self.create_H(target_device)
        return self.H

    def snapshot_hessian(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        """Return a normalized Hessian without changing the live capture.

        Durable capture-frontier writers use this to stream one Hessian at a
        time to host storage.  Unlike :meth:`finalize_hessian`, this method
        deliberately leaves ``H`` and the per-device partials in place so the
        subsequent quantization pass can consume the original device-local
        capture without retaining every Hessian in host memory.

        Both the already-finalized contribution (``self.H`` or ``self._H_diag``)
        and any partials collected after it are merged by their own sample
        counts, mirroring :meth:`materialize_global_hessian`.  Embedding
        captures use the same count-derived diagonal normalization as
        :meth:`finalize_hessian`.
        """

        with self.lock:
            device = self._select_hessian_target_device(target_device)

            if isinstance(self.module, nn.Embedding):
                if not self._hessian_dirty and self._H_diag is not None:
                    self._wait_hessian_ready(device)
                    source_diag = self._H_diag
                    result = source_diag.detach().to(device=device, dtype=torch.float32)
                    if result.data_ptr() == source_diag.data_ptr():
                        # `.to()` is a no-op that returns the live diagonal
                        # itself when the device and dtype already match. The
                        # snapshot outlives the capture, so detach it from the
                        # storage that later finalization mutates in place.
                        result = result.clone()
                    elif result.device != source_diag.device or result.dtype != source_diag.dtype:
                        event = self._cuda_event_for(device)
                        if event is not None:
                            event.record(torch.cuda.current_stream(device))
                        self._defer_hessian_sources([source_diag], event)
                    return result

                finalized_samples, pending_samples = self._snapshot_sample_counts()
                total_samples = finalized_samples + pending_samples
                diag = torch.zeros(self.columns, dtype=torch.float32, device=device)
                if total_samples == 0:
                    return diag

                sources: list[torch.Tensor] = []
                if finalized_samples and self._H_diag is not None:
                    self._wait_hessian_ready(device)
                    source_diag = self._H_diag
                    previous = safe_move_to(source_diag, device, dtype=torch.float32)
                    diag.add_(
                        previous,
                        alpha=float(finalized_samples) / float(total_samples),
                    )
                    if previous is not source_diag:
                        sources.append(source_diag)

                for partial_device in sorted(
                    self._device_embedding_counts, key=_device_reduction_key
                ):
                    counts = self._device_embedding_counts[partial_device]
                    partial_event = self._device_hessian_partial_ready.get(
                        torch.device(partial_device)
                    )
                    if partial_event is not None:
                        if device.type == "cuda":
                            torch.cuda.current_stream(device).wait_event(partial_event)
                        else:
                            partial_event.synchronize()
                    diag.add_(
                        safe_move_to(counts, device, dtype=torch.float32),
                        alpha=2.0 / float(total_samples),
                    )
                event = self._cuda_event_for(device)
                if event is not None:
                    event.record(torch.cuda.current_stream(device))
                self._defer_hessian_sources(sources, event)
                return diag

            if not self._hessian_dirty and self.H is not None:
                self._wait_hessian_ready(device)
                source = self.H
                result = source.detach().to(device=device, dtype=torch.float32)
                if result.data_ptr() == source.data_ptr():
                    # Same device and dtype: `.to()` returned the live Hessian,
                    # which `materialize_global_hessian` mutates in place on the
                    # next finalization. The snapshot must be independent.
                    result = result.clone()
                elif result.device != source.device:
                    event = self._cuda_event_for(device)
                    if event is not None:
                        event.record(torch.cuda.current_stream(device))
                    self._defer_hessian_sources([source], event)
                return result

            finalized_samples, pending_samples = self._snapshot_sample_counts()
            total_samples = finalized_samples + pending_samples
            result = torch.zeros(
                (self.columns, self.columns),
                dtype=torch.float32,
                device=device,
            )
            if total_samples == 0:
                return result

            sources = list(self._device_hessian_partials.values())
            if finalized_samples and self.H is not None:
                self._wait_hessian_ready(device)
                source_hessian = self.H
                previous = safe_move_to(source_hessian, device, dtype=torch.float32)
                result.add_(
                    previous,
                    alpha=float(finalized_samples) / float(total_samples),
                )
                if previous is not source_hessian:
                    sources.insert(0, source_hessian)
            for partial_device in sorted(self._device_hessian_partials, key=_device_reduction_key):
                partial = self._device_hessian_partials[partial_device]
                event = self._device_hessian_partial_ready.get(torch.device(partial_device))
                if event is not None:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).wait_event(event)
                    else:
                        event.synchronize()
                result.add_(
                    safe_move_to(partial, device, dtype=torch.float32),
                    alpha=2.0 / float(total_samples),
                )
            event = self._cuda_event_for(device)
            if event is not None:
                event.record(torch.cuda.current_stream(device))
            self._defer_hessian_sources(sources, event)
            return result

    def _snapshot_sample_counts(self) -> Tuple[int, int]:
        """Return the finalized and pending sample counts of the live capture."""

        finalized_samples = self._hessian_total_samples
        pending_samples = sum(self._device_sample_counts.values())
        if finalized_samples + pending_samples != self.nsamples:
            raise RuntimeError(
                f"Hessian snapshot sample-count drift: finalized={finalized_samples}, "
                f"partials={pending_samples}, capture={self.nsamples}"
            )
        return finalized_samples, pending_samples

    def create_H(self, target_device):
        return torch.zeros((self.columns, self.columns), dtype=torch.float32,
                           device=self._select_hessian_target_device(target_device))

    def _fallback_quantize(
        self,
        strategy: FallbackStrategy,
        blocksize: int,
        target_device: Optional[torch.device] = None,
        hessian_diag: Optional[torch.Tensor] = None,
    ):
        """Apply a lightweight quantization fallback using the requested strategy.

        ``target_device`` is the device the weight clone is quantized on; when
        None it is taken from ``self.H`` if present, else the weight.
        ``hessian_diag`` supplies the captured Hessian diagonal when the full
        matrix has already been released (the low-sample fallback path). When
        present it is preferred over ``self.H``: the diagonal is reduced from
        the live capture immediately before this call, so it also includes
        batches captured after an earlier ``finalize_hessian()``, while
        ``self.H`` is only the last finalized snapshot.
        """
        maxq = 2 ** self.qcfg.bits - 1
        sigma = 3.0
        effective_group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        start_time = time.time()
        smooth_method = getattr(self.fallback, "smooth", None)
        mse_steps = 32
        mse_maxshrink = 0.8
        if isinstance(smooth_method, SmoothMSE):
            mse_steps = smooth_method.steps
            mse_maxshrink = smooth_method.maxshrink

        if target_device is None:
            target_device = self.H.device if self.H is not None else self.module.weight.device
        W = self.clone_module(device=target_device)
        # When the Hessian exists its diagonal is the exact input-channel
        # importance for the layer's output error, so weight the SmoothMSE
        # objective with it instead of averaging every channel equally. A
        # fallback without captured activations uses the weight-magnitude
        # proxy instead of the harmful uniform mean.
        col_weights = None
        if isinstance(smooth_method, SmoothMSE):
            col_weights = hessian_column_weights(
                hessian_diag, self.columns, device=target_device
            )
            if col_weights is None:
                col_weights = hessian_column_weights(
                    self.H, self.columns, device=target_device
                )
        Q = torch.empty_like(W)
        scale_chunks = []
        zero_chunks = []

        def _affine_quantize(block_mod, lo, hi, scale_min):
            """Quantize one group against a range that includes zero.

            ``lo``/``hi`` are the strategy's desired endpoints. Extending them
            to include zero keeps the derived integer zero point inside
            ``[0, maxq]`` instead of clamping a positive-only group to zero and
            shifting every value by roughly the group minimum.
            """

            if self.qcfg.sym:
                # Every symmetric kernel assumes zero = 2 ** (bits - 1);
                # Marlin does not even read the stored qzeros. Fit the common
                # symmetric range so all strategies store that fixed zero.
                lo, hi = symmetric_range(block_mod)
            else:
                lo, hi = affine_range_with_zero(lo, hi)
            scale = _floor_scale_for_storage(
                _replace_zero_scale(torch.clamp((hi - lo) / maxq, min=1e-8)),
                scale_min,
            )
            if self.qcfg.sym:
                zero = torch.full_like(scale, (maxq + 1) // 2)
            else:
                zero = torch.clamp(torch.round(-lo / scale), 0, maxq)
            q = torch.clamp(torch.round(block_mod / scale + zero), 0, maxq)
            return (q - zero) * scale, scale, zero

        for start in range(0, self.columns, effective_group_size):
            end = min(start + effective_group_size, self.columns)
            block = W[:, start:end]

            if isinstance(smooth_method, SmoothMSE):
                weights = element_saliency_weights(block) if col_weights is None else col_weights[start:end]
                dequant, scale, zero = mse_optimal_quant(
                    block,
                    self.qcfg,
                    maxq,
                    steps=mse_steps,
                    maxshrink=mse_maxshrink,
                    eps=SCALE_STORAGE_FLOOR,
                    weights=weights,
                )
            else:
                block_mod, scale_factor = smooth_block(
                    block,
                    self.fallback,
                    group_size=effective_group_size,
                )
                # Row/column smoothing restores this factor after quantization.
                # Bound scales in that final weight domain before selecting codes.
                scale_min = (
                    SCALE_STORAGE_FLOOR if scale_factor is None
                    else SCALE_STORAGE_FLOOR / scale_factor
                )
                if strategy == FallbackStrategy.MIDPOINT:
                    w_min = block_mod.min(dim=1, keepdim=True).values
                    w_max = block_mod.max(dim=1, keepdim=True).values
                    dequant, scale, zero = _affine_quantize(
                        block_mod, w_min, w_max, scale_min
                    )
                elif strategy == FallbackStrategy.MEAN:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    max_dev = torch.max((block_mod - mean).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    dequant, scale, zero = _affine_quantize(
                        block_mod, mean - max_dev, mean + max_dev, scale_min
                    )
                elif strategy == FallbackStrategy.MEDIAN:
                    median = block_mod.median(dim=1, keepdim=True).values
                    max_dev = torch.max((block_mod - median).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    dequant, scale, zero = _affine_quantize(
                        block_mod, median - max_dev, median + max_dev, scale_min
                    )
                elif strategy == FallbackStrategy.STDCLIP:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    std = block_mod.std(dim=1, keepdim=True, unbiased=False)
                    std = torch.clamp(std, min=1e-8)
                    lo = mean - sigma * std
                    hi = mean + sigma * std
                    dequant, scale, zero = _affine_quantize(
                        block_mod, lo, hi, scale_min
                    )
                elif strategy == FallbackStrategy.RTN:
                    self.quantizer.find_params(
                        block_mod, weight=True,
                        scale_min=scale_min.flatten() if isinstance(scale_min, torch.Tensor) else scale_min,
                    )
                    dequant = self.quantizer.quantize(block_mod)
                    scale, zero = _replace_zero_scale(
                        self.quantizer.scale, self.quantizer.zero
                    )
                else:
                    raise ValueError(f"Unsupported fallback strategy: {strategy}")

                if scale_factor is not None:
                    scale = scale * scale_factor
                    dequant = dequant * scale_factor

            Q[:, start:end] = dequant

            scale_block = scale if scale.dim() > 1 else scale.unsqueeze(1)
            zero_block = zero if zero.dim() > 1 else zero.unsqueeze(1)
            if scale_block.shape[1] > 1:
                scale_block = scale_block.mean(dim=1, keepdim=True)
            if zero_block.shape[1] > 1:
                zero_block = zero_block.mean(dim=1, keepdim=True)
            scale_chunks.append(scale_block)
            zero_chunks.append(zero_block)

        scale = torch.cat(scale_chunks, dim=1)
        zero = torch.cat(zero_chunks, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)
        else:
            valid_cols = self.columns

        scale, zero = self._sanitize_scale_and_log(scale, zero)

        group_size = effective_group_size if effective_group_size != -1 else self.columns
        g_idx = torch.arange(valid_cols, device=Q.device, dtype=torch.int32) // group_size

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).to(self.module.weight.dtype)
        else:
            Q = Q.to(self.module.weight.dtype)

        Q = safe_move_to(Q, self.module.weight.data.device)
        mean_abs_err = (Q - self.module.weight.data).abs().mean().item()
        duration = time.time() - start_time
        avg_loss = f"fallback({strategy.value}): {mean_abs_err:.7f}"
        damp = 0.0

        self.H = None
        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def _rtn_fallback_after_hessian_failure(self, blocksize: int):
        """Quantize with GPTQ's data-independent RTN grid.

        ``hessian_inverse`` returns ``(None, 1.0)`` when every damp and
        diagonal-floor attempt fails. GPTAQ and FOEM call this helper so a
        non-invertible Hessian degrades to the same weight-only grid GPTQ uses
        for low-sample modules instead of dereferencing the missing inverse.
        It delegates to ``_fallback_quantize``; there is no second fallback
        implementation.
        """

        log.warn(
            "Quantization: Module `%s` -> Hessian inverse unavailable; using `%s` fallback quantization.",
            self.name,
            FallbackStrategy.RTN.value,
        )
        self.H = None
        return self._fallback_quantize(FallbackStrategy.RTN, blocksize)

    # FIXME, optimum needs fasterquant, we need to remove it
    def fasterquant(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        return self.hf_quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    # public api exposed to hf
    def hf_quantize(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
            act_group_aware: Optional[bool] = None,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.desc_act = actorder
        if act_group_aware is not None:
            self.qcfg.act_group_aware = act_group_aware
        self.qcfg._resolve_activation_ordering(actorder, act_group_aware)
        self.qcfg.static_groups = static_groups
        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)
        self.module.weight.data = Q
        return scale, zero, g_idx, duration, avg_loss, damp_percent

    def hessian_inverse(self, H: torch.Tensor):
        timer = getattr(self, "region_timer", None)
        timer_cm = (
            timer.measure("hessian_inverse")
            if timer is not None
            else contextlib.nullcontext()
        )
        # Keep the deterministic rocBLAS selection on the Hessian/linalg path
        # (rocSOLVER GEMMs included) for direct callers; the run-scoped override
        # in ``models/base.py`` already covers the quantization path.
        with timer_cm, quant_blas_backend(H.device):
            if _log_hessian_verbose():
                log.info(f"GPTQ: hessian_inverse begin {self.name} shape={tuple(H.shape)}")
            try:
                # Capture a writable view of the Hessian diagonal so we can restore it between attempts.
                diag_view = H.diagonal()
                orig_diag = diag_view.clone()
                max_floor_attempts = 6
                last_error = None
                damp_candidates = None
                floor_base = None

                def try_damp(current_diag: torch.Tensor, mean: torch.Tensor, damp: float):
                    """Return a Cholesky result for one grid value, restoring ``H``."""

                    nonlocal last_error
                    diag_view.add_(damp * mean)
                    try:
                        if H.device.type == "npu":
                            return npu_inverse_cholesky_factor(
                                H,
                                block_size=resolve_kernel_config(self.qcfg).npu_cholesky_block_size,
                            )

                        # ``cholesky_ex`` reports a non positive-definite input through
                        # ``info`` instead of raising, so failed probes stay cheap.
                        H2, info = torch.linalg.cholesky_ex(H, check_errors=False)
                        info_value = int(info.item())
                        if info_value != 0:
                            last_error = f"torch.linalg.cholesky_ex info={info_value}"
                            del H2
                            return None
                        return H2
                    except torch._C._LinAlgError as exc:
                        last_error = exc
                        return None
                    finally:
                        diag_view.copy_(current_diag)

                def inverse_factor(cholesky_result: torch.Tensor) -> torch.Tensor:
                    """Finish only the selected successful candidate."""

                    if H.device.type == "npu":
                        return cholesky_result
                    if H.device.type in ("cpu", "cuda"):
                        # cholesky_inverse permits the factor as its output.
                        # Reuse that buffer while keeping H intact for damp retries.
                        torch.cholesky_inverse(cholesky_result, out=cholesky_result)
                        return torch.linalg.cholesky(cholesky_result, upper=True)
                    return torch.linalg.cholesky(
                        torch.cholesky_inverse(cholesky_result),
                        upper=True,
                    )

                def try_factor(current_diag: torch.Tensor, mean: torch.Tensor, damp: float):
                    """Return the finished inverse factor for one grid value, or ``None``.

                    The cheap probe only proves the Cholesky factor exists. The legacy
                    linear walk also required the inverse step to succeed, which can
                    fail for Hessians with an extreme dynamic range even when the
                    factorization itself reports success, so both steps must agree.
                    """

                    nonlocal last_error
                    cholesky_result = try_damp(current_diag, mean, damp)
                    if cholesky_result is None:
                        return None
                    try:
                        return inverse_factor(cholesky_result)
                    except torch._C._LinAlgError as exc:
                        last_error = exc
                        return None

                def legacy_damp_candidates() -> list[float]:
                    """Materialize the old additive grid without changing float rounding."""

                    # Repeated addition (not one multiply-add) keeps the selected damp
                    # bit-identical to the former linear walk.
                    values = [float(self.qcfg.damp_percent)]
                    increment = float(self.qcfg.damp_auto_increment)
                    if increment == 0.0:
                        return values
                    while True:
                        next_damp = values[-1] + increment
                        # The progress guard also makes malformed/non-finite increments
                        # terminate instead of spinning forever.
                        if not (values[-1] < next_damp < 1.0):
                            return values
                        values.append(next_damp)

                def finish_damped_candidate(first_success: int, first_cholesky: torch.Tensor):
                    """Finish the first grid value whose Cholesky factor also inverts.

                    Bisection only checks the cheap factorization probe. If the selected
                    factor still fails the inverse step, keep walking the legacy grid so
                    the chosen ``damp`` stays identical to the former linear scan.
                    """

                    nonlocal last_error
                    index = first_success
                    cholesky_result = first_cholesky
                    while index < len(damp_candidates):
                        if cholesky_result is None:
                            cholesky_result = try_damp(current_diag, mean, damp_candidates[index])
                        if cholesky_result is None:
                            index += 1
                            continue
                        try:
                            return inverse_factor(cholesky_result), damp_candidates[index]
                        except torch._C._LinAlgError as exc:
                            last_error = exc
                            cholesky_result = None
                        index += 1
                    return None

                def log_failure_diagnostics() -> float:
                    """Run full-matrix diagnostics only after the healthy path failed.

                    Healthy layers therefore never pay for these matrix-wide scans.
                    """

                    finite = bool(torch.isfinite(H).all().item())
                    if not finite:
                        log.error(
                            f"Quantization: Module `{self.name}` -> Hessian contains non-finite values."
                        )
                    elif not torch.equal(H, H.mT):
                        log.warn(
                            f"Quantization: Module `{self.name}` -> Hessian is not exactly symmetric; "
                            "recovery will not modify or symmetrize it."
                        )

                    base_abs_max = torch.max(orig_diag.abs()).item()
                    if not math.isfinite(base_abs_max) or base_abs_max == 0.0:
                        log.warn(
                            f"Quantization: Module `{self.name}` -> Hessian diagonal is degenerate; "
                            "using the minimum diagonal-floor scale."
                        )
                        return 1.0
                    return base_abs_max

                attempt = 0
                while attempt <= max_floor_attempts:
                    if attempt == 0:
                        current_diag = orig_diag
                    else:
                        assert floor_base is not None
                        floor_increment = floor_base * math.pow(10.0, attempt - 1)
                        current_diag = torch.clamp(orig_diag + floor_increment, min=floor_increment)
                        if attempt == 1:
                            log.warn(
                                f"Quantization: Module `{self.name}` -> Applying Hessian diagonal floor (+{floor_increment:.2e}) to recover positive definiteness.")
                        else:
                            log.warn(
                                f"Quantization: Module `{self.name}` -> Increasing Hessian diagonal floor to +{floor_increment:.2e}.")

                    diag_view.copy_(current_diag)
                    mean = torch.mean(current_diag)
                    initial_damp = float(self.qcfg.damp_percent)
                    # Probe the configured damp first, on the untouched diagonal.
                    initial_factor = try_factor(current_diag, mean, initial_damp)
                    if initial_factor is not None:
                        return initial_factor, initial_damp

                    if attempt == 0:
                        # Defer matrix-wide scans and their host decisions until the
                        # first Cholesky failure. The healthy path stays untouched.
                        floor_base = log_failure_diagnostics() * 1e-6

                    if self.qcfg.damp_auto_increment == 0:
                        log.warn(
                            f"Quantization: Module `{self.name}` -> Hessian Cholesky failed with "
                            f"`damp_percent={initial_damp:.5f}` and no auto increment configured."
                        )
                    else:
                        if damp_candidates is None:
                            damp_candidates = legacy_damp_candidates()

                        log.warn(
                            f"Quantization: Module `{self.name}` -> Starting compatible damp recovery at "
                            f"`damp_percent={initial_damp:.5f}`, increment step "
                            f"`{self.qcfg.damp_auto_increment:.5f}`."
                        )

                        if len(damp_candidates) > 1:
                            # Cholesky success is monotone on this grid when the
                            # diagonal mean is positive. Test the endpoint first,
                            # then find the first successful legacy candidate.
                            high = len(damp_candidates) - 1
                            high_cholesky = try_damp(current_diag, mean, damp_candidates[high])
                            if high_cholesky is not None:
                                low = 0  # The initial candidate already failed.
                                while high - low > 1:
                                    middle = (low + high) // 2
                                    middle_cholesky = try_damp(current_diag, mean, damp_candidates[middle])
                                    if middle_cholesky is None:
                                        low = middle
                                    else:
                                        high = middle
                                        high_cholesky = middle_cholesky

                                selected = finish_damped_candidate(high, high_cholesky)
                                if selected is not None:
                                    factor, used_damp = selected
                                    log.warn(
                                        f"Quantization: Module `{self.name}` -> Damp recovery succeeded at "
                                        f"`damp_percent={used_damp:.5f}` (started at {initial_damp:.5f})."
                                    )
                                    return factor, used_damp

                        log.warn(
                            f"Quantization: Module `{self.name}` -> Damp recovery failed after reaching "
                            f"`damp_percent={damp_candidates[-1]:.5f}`."
                        )

                    attempt += 1

                log.error(
                    f"Quantization: Module `{self.name}` -> Hessian remained non positive-definite after diagonal floor attempts. "
                    f"Last `damp_percent` tried = "
                    f"{(damp_candidates[-1] if damp_candidates else self.qcfg.damp_percent):.5f}."
                )
                if last_error is not None:
                    log.debug(f"Hessian failure detail: {last_error}")
                return None, 1.0
            finally:
                if _log_hessian_verbose():
                    log.info(f"GPTQ: hessian_inverse end {self.name}")

    @torch.inference_mode()
    def _quantize_embedding(self, blocksize=128):
        """Quantize an embedding using its token-frequency diagonal Hessian."""

        del blocksize
        start = time.time()
        target_device = getattr(self.module, "target_device", None)
        if target_device is None:
            target_device = self.module.weight.device
        target_device = torch.device(target_device)

        diag = self.finalize_hessian(target_device=target_device)
        self._release_deferred_hessian_sources()
        original_weight = self.clone_module(device=target_device)
        weight = original_weight
        inverse_permutation = None
        group_permutation = None

        if self.qcfg.static_groups:
            pass
        elif self.qcfg.desc_act:
            permutation = torch.argsort(diag, descending=True)
            inverse_permutation = torch.argsort(permutation)
            weight = weight[:, permutation]
        elif self.qcfg.act_group_aware:
            local_perms, local_values = compute_local_perms(
                diag, self.qcfg.group_size, return_values=True
            )
            group_permutation = compute_global_perm(
                diag,
                self.qcfg.group_size,
                precomputed_values=local_values,
            )
            permutation = compose_final_perm(local_perms, group_permutation, self.qcfg.group_size)
            permutation = extend_perm_with_tail(permutation, self.columns)
            inverse_permutation = invert_perm(permutation)
            weight = weight[:, permutation]

        quantized = torch.empty_like(weight)
        scales = []
        zeros = []
        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        for start_col in range(0, self.columns, group_size):
            end_col = min(start_col + group_size, self.columns)
            block = weight[:, start_col:end_col]
            self.quantizer.find_params(block, weight=True)
            quantized[:, start_col:end_col] = self.quantizer.quantize(block)
            scales.append(self.quantizer.scale)
            zeros.append(self.quantizer.zero)

        scale = torch.cat(scales, dim=1)
        zero = torch.cat(zeros, dim=1)
        base_group_index = torch.arange(self.columns, dtype=torch.int32, device=quantized.device) // group_size
        if self.qcfg.desc_act and not self.qcfg.static_groups:
            g_idx = base_group_index[inverse_permutation]
        else:
            g_idx = base_group_index

        if inverse_permutation is not None:
            quantized = quantized[:, inverse_permutation]

        if group_permutation is not None:
            full_group_count = self.columns // group_size
            if full_group_count:
                inverse_groups = invert_perm(group_permutation).to(device=scale.device)
                full_scale = scale[:, :full_group_count].index_select(1, inverse_groups)
                full_zero = zero[:, :full_group_count].index_select(1, inverse_groups.to(device=zero.device))
                scale = torch.cat((full_scale, scale[:, full_group_count:]), dim=1)
                zero = torch.cat((full_zero, zero[:, full_group_count:]), dim=1)

        scale, zero = self._sanitize_scale_and_log(scale, zero)

        # Keep the floor selection on-device: the following loss read already
        # synchronizes the stream, so reading `diag.max()` here only adds a
        # second host round trip per module. `max(x, 1) * 1e-6` is
        # `max(x * 1e-6, 1e-6)`; computing it in the diagonal dtype can differ
        # from the former double-precision product by at most one ULP.
        floor = torch.clamp(diag.max() * 1e-6, min=1e-6)
        effective_diag = torch.clamp(diag, min=floor) + self.qcfg.damp_percent * diag.mean()
        column_error = (original_weight - quantized).square().sum(dim=0)
        if self.nsamples:
            avg_loss = (0.5 * torch.sum(effective_diag * column_error) / self.nsamples).item()
        else:
            avg_loss = 999999999

        if self._tp_pad_cols:
            quantized = quantized[:, :self._original_columns]
            g_idx = g_idx[:self._original_columns]

        quantized = quantized.t().reshape(self.module.weight.shape).to(self.module.weight.dtype)
        quantized = safe_move_to(quantized, self.module.weight.device)

        self.H = None
        self._device_hessian_partials.clear()
        self._device_hessian_partial_ready.clear()
        return (
            quantized,
            scale,
            zero,
            g_idx,
            time.time() - start,
            avg_loss,
            self.qcfg.damp_percent,
            self.nsamples,
        )

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        config = getattr(self.qcfg, "gsq", None)
        module_name = self._named_module.full_name if self._named_module is not None else self.name
        if getattr(self, "_gsq_active", False) or not gsq_enabled_for(config, module_name):
            return self._quantize_impl(blocksize=blocksize)

        if not isinstance(self.module, nn.Linear):
            raise ValueError("GSQ currently supports GPTQ linear projections only")
        if self._tp_pad_cols:
            raise ValueError("GSQ does not support tensor-parallel padded projections")
        from ..utils.fallback import should_use_fallback

        if should_use_fallback(self.fallback, float(self.nsamples), self.expected_nsamples):
            self.gsq_diagnostics = {"status": "skipped", "reason": "data_independent_fallback"}
            return self._quantize_impl(blocksize=blocksize)
        if self.nsamples == 0:
            raise ValueError("GSQ requires calibration activations")

        start = time.time()
        target = self.clone_module().detach()
        hessian = self.finalize_hessian(target_device=target.device).detach().clone()
        self._gsq_active = True
        try:
            result = self._quantize_impl(blocksize=blocksize)
        finally:
            self._gsq_active = False
        weight, scales, zeros, groups, duration, avg_loss, damp, samples = result
        if self.qcfg.mock_quantization or not isinstance(avg_loss, (int, float)) or not math.isfinite(avg_loss):
            self.gsq_diagnostics = {"status": "skipped", "reason": "initializer_fallback"}
            return result

        device = weight.device
        fitted = refine_gptq_scalar(
            weight,
            scales.to(device),
            zeros.to(device),
            groups.to(device),
            target=target.to(device),
            bits=int(self.qcfg.bits),
            config=config,
            hessian=hessian.to(device),
        )
        self.gsq_diagnostics = {
            "objective": "calibration_hessian",
            "before": fitted.before,
            "after": fitted.after,
            "learn_scales": config.learn_scales,
        }
        return (
            # The looper assigns this tensor into an inference-mode Parameter.
            # Match the tensor kind of the original GPTQ result so the next
            # calibration replay can use the updated weight.
            fitted.weight.contiguous().clone(),
            fitted.scales.to(scales.device),
            fitted.zeros.to(zeros.device),
            fitted.g_idx.to(groups.device),
            time.time() - start,
            avg_loss,
            damp,
            samples,
        )

    @torch.inference_mode()
    def _quantize_impl(
            self,
            blocksize=128,
    ):
        if isinstance(self.module, nn.Embedding):
            return self._quantize_embedding(blocksize=blocksize)

        # self.H = self.H.to(device=CUDA_0)
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        target_device = getattr(self.module, "target_device", None)
        result_device = torch.device(self.module.weight.data.device)
        cpu_fallback_used = False
        from ..utils.fallback import resolve_fallback_strategy, resolve_threshold, should_use_fallback

        zero_samples = self.nsamples == 0
        resolved_strategy = resolve_fallback_strategy(self.fallback)
        if zero_samples:
            # A zero Hessian makes every column look calibration-dead, and
            # ``_neutralize_dead_columns`` would then zero the entire weight.
            # RTN is the only data-independent grid, so use it even when the
            # threshold fallback is disabled rather than ship an all-zero module.
            resolved_strategy = FallbackStrategy.RTN
        fallback_requested = zero_samples or should_use_fallback(
            self.fallback,
            float(self.nsamples),
            self.expected_nsamples,
        )
        threshold_raw, is_percent = resolve_threshold(self.fallback, self.expected_nsamples)
        fallback_configured = threshold_raw is not None

        if fallback_requested:
            use_hessian = False
            self._disable_shared_hessian_cache()
            if zero_samples:
                log.warn(
                    "Quantization: Module `%s` -> no calibration samples reached this module; "
                    "forcing `%s` fallback quantization instead of quantizing a zero Hessian.",
                    self.name,
                    FallbackStrategy.RTN.value,
                )
            threshold_text = str(getattr(self.fallback, "threshold", None))
            threshold_info = f", threshold_raw={threshold_raw}" if threshold_raw is not None and is_percent else ""
            log.warn(
                f"Quantization: Module `{self.name}` -> "
                f"Using `{resolved_strategy.value}` fallback quantization (observed {self.nsamples} samples, threshold={threshold_text}{threshold_info}, max_total={self.expected_nsamples})."
            )
            # The fallback does not need the full Hessian: release the fp32
            # XtX partials (columns^2 x 4 B each) instead of folding them into
            # a throwaway zero H. Resolve the compute device first, since the
            # clear changes what _select_hessian_target_device returns.
            with self.lock:
                fallback_device = self._select_hessian_target_device(target_device)
            # SmoothMSE still needs the per-channel input energy. Reduce the
            # partials to the diagonal before releasing them; this keeps the
            # memory win while letting the documented Hessian weighting reach
            # singleton modules (o_proj/down_proj) as well (#5300).
            fallback_diag = (
                self._captured_hessian_diagonal(fallback_device)
                if isinstance(getattr(self.fallback, "smooth", None), SmoothMSE)
                else None
            )
            with self.lock:
                self._device_hessian_partials.clear()
                self._device_hessian_partial_ready.clear()
                self._device_sample_counts.clear()
                self._hessian_dirty = False

            # Shared-input followers may retain a leader Hessian source until
            # the destination copy completes.  Drain it before the fallback
            # clones weights and allocates its output buffers.
            self._release_deferred_hessian_sources()

            return self._fallback_quantize(
                resolved_strategy,
                blocksize,
                target_device=fallback_device,
                hessian_diag=fallback_diag,
            )
        else:
            use_hessian = True
            self.finalize_hessian(target_device=target_device)
            self._release_deferred_hessian_sources()

        if self.qcfg.mock_quantization:
            # Use simplified hessian inverse (identity matrix)
            self.hessian_inverse = self.mock_hessian_inverse

        # The block-pass CPU retry must restart from the pre-pass weight; the
        # failed pass error-propagates into ``W`` in place. A module-sourced
        # ``W`` is rebuilt from the untouched module weight on retry, so the
        # common path keeps no extra copy. An explicit ``module_copy`` has no
        # such source and is snapshotted to the host before the block pass.
        weight_from_module = self.module_copy is None
        if weight_from_module:
            # log.info("copy W to cuda_1")
            W = self.clone_module(device=self.H.device)
        else:
            W = safe_move_to(self.module_copy, self.H.device)
            del self.module_copy

        self.quantizer.find_params(W, weight=True)

        # H = self.H.to(device=self.H.device)

        # Capture the NaN-retry diagonal before any activation ordering. The
        # retry clones ``W`` in natural column order, so a diagonal captured
        # after desc_act/act_group_aware permutation would make ``argsort`` the
        # identity and silently drop act-order from the retried ``g_idx``.
        retry_hessian_diag = (
            self.H.diagonal().clone()
            if use_hessian and fallback_configured and not self.qcfg.mock_quantization
            else None
        )

        dead_columns = None
        if use_hessian:
            dead_columns = _neutralize_dead_columns(self.H, W)

        # g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if self.qcfg.static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, self.qcfg.group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i: (i + self.qcfg.group_size)], weight=True)

                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if self.qcfg.desc_act and use_hessian:
            perm = torch.argsort(torch.diag(self.H), descending=True)
            try:
                # Gather H before W: H needs two ``cols x cols`` temporaries
                # and is the allocation most likely to exhaust the device.
                # Both results stay in locals until both gathers succeed, so
                # the fallback below always sees an unpermuted ``W`` and
                # ``self.H``.  Assigning ``W`` first would leave it permuted
                # when the H gather raises and make the fallback permute it
                # a second time.
                H_permuted = self.H[perm][:, perm]
                W_permuted = W[:, perm]
            except RuntimeError as exc:
                if self.H.device.type != "cuda" or not _is_memory_allocation_failure(exc):
                    raise

                self.log_cpu_fallback("Hessian permutation", self.H.device)
                self._disable_shared_hessian_cache()
                cpu_fallback_used = True
                cpu_device = torch.device("cpu")
                perm = perm.to(device=cpu_device)
                W = W.to(device=cpu_device)[:, perm]
                self.H = self.H.to(device=cpu_device)[perm][:, perm]
                self.quantizer.find_params(W, weight=True)
                if self.qcfg.static_groups:
                    scale, zero = _move_static_groups(groups, cpu_device)
            else:
                W = W_permuted
                self.H = H_permuted
            invperm = torch.argsort(perm)

        elif self.qcfg.act_group_aware and use_hessian:
            diag_h = torch.diag(self.H)
            local_perms, local_values = compute_local_perms(
                diag_h, self.qcfg.group_size, return_values=True
            )
            global_perm = compute_global_perm(
                diag_h,
                self.qcfg.group_size,
                precomputed_values=local_values,
            )
            del local_values
            final_perm = compose_final_perm(local_perms, global_perm, self.qcfg.group_size)
            final_perm = extend_perm_with_tail(final_perm, self.columns)
            try:
                # See the desc_act branch above: keep both permutations in
                # locals so a failed H gather leaves W unpermuted for the
                # CPU fallback.
                H_permuted = self.H[final_perm][:, final_perm]
                W_permuted = W[:, final_perm]
            except RuntimeError as exc:
                if self.H.device.type != "cuda" or not _is_memory_allocation_failure(exc):
                    raise

                self.log_cpu_fallback("act-group Hessian permutation", self.H.device)
                self._disable_shared_hessian_cache()
                cpu_fallback_used = True
                cpu_device = torch.device("cpu")
                final_perm = final_perm.to(device=cpu_device)
                W = W.to(device=cpu_device)[:, final_perm]
                self.H = self.H.to(device=cpu_device)[final_perm][:, final_perm]
                self.quantizer.find_params(W, weight=True)
                if self.qcfg.static_groups:
                    scale, zero = _move_static_groups(groups, cpu_device)
            else:
                W = W_permuted
                self.H = H_permuted

        # Column order the block pass sees, for rebuilding ``W`` on retry.
        block_column_perm = None
        if self.qcfg.desc_act and use_hessian:
            block_column_perm = perm
        elif self.qcfg.act_group_aware and use_hessian:
            block_column_perm = final_perm

        static_group_perm = block_column_perm if self.qcfg.static_groups else None

        if use_hessian:
            try:
                ordering = "desc_act" if self.qcfg.desc_act else (
                    "act_group_aware" if self.qcfg.act_group_aware else "natural"
                )
                cache = self._shared_hessian_cache
                if cache is None or self._shared_hessian_cache_disabled:
                    Hinv, damp = self.hessian_inverse(self.H)
                else:
                    def _compute_shared_inverse():
                        factor, used_damp = self.hessian_inverse(self.H)
                        effective_diag = self.H.diagonal().detach().clone() if factor is not None else None
                        return factor, used_damp, effective_diag

                    shared_inverse = self._shared_factor_or_local(ordering, _compute_shared_inverse)
                    Hinv, damp, effective_diag = shared_inverse
                    if effective_diag is not None:
                        # Each task owns a private ordered H.  Apply the
                        # producer's post-floor diagonal before capturing the
                        # retry/mock snapshot so NaN recovery remains exact.
                        self.H.diagonal().copy_(effective_diag)
            except RuntimeError as exc:
                if self.H.device.type != "cuda" or not _is_memory_allocation_failure(exc):
                    raise

                # Full-attention blocks on very large models can exceed GPU memory during the
                # dense Hessian inverse; finish that module on CPU instead of aborting the run.
                self.log_cpu_fallback("Hessian inverse", self.H.device)
                cpu_fallback_used = True
                cpu_device = torch.device("cpu")
                self.H = self.H.to(device=cpu_device)
                W = W.to(device=cpu_device)
                self.quantizer.find_params(W, weight=True)
                if self.qcfg.static_groups:
                    scale, zero = _move_static_groups(groups, cpu_device)
                self._disable_shared_hessian_cache()
                Hinv, damp = self.hessian_inverse(self.H)
        else:
            Hinv, damp = None, 0.0

        # The block update uses Hinv, but never reads the Hessian again. The
        # NaN -> mock-quantization retry rebuilds a diagonal matrix from the
        # natural-order snapshot captured above, then re-runs neutralization and
        # activation ordering so it reproduces the primary column order.
        if Hinv is not None:
            self.H = None

        # The block update and its W-sized scratch run after the inverse
        # fallback decision, so a CUDA OOM here used to abort the module even
        # though the run could finish on the host. Retry the whole block pass
        # on CPU. The failed pass has already error-propagated finished blocks
        # into ``W[:, i2:]`` in place, so the retry must restart from the
        # pre-pass weight rather than move the partially updated one
        # (issue #4464). It may also have appended partial scale/zero entries
        # and advanced the quantizer, so reset that state before re-running.
        block_input_host = None
        if not weight_from_module and W.device.type == "cuda":
            block_input_host = W.to(device="cpu")
        block_retry = False
        while True:
            if block_retry:
                cpu_device = torch.device("cpu")
                del W
                if weight_from_module:
                    # Replay the pre-pass preparation on the host: clone,
                    # natural-order quantizer measurement, dead-column
                    # neutralization, then activation ordering. Each step is
                    # exact, so this reproduces the pre-pass ``W`` bit for bit.
                    W = self.clone_module(device=cpu_device)
                    if not self.qcfg.static_groups:
                        self.quantizer.find_params(W, weight=True)
                    if dead_columns is not None:
                        W.masked_fill_(dead_columns.to(device=cpu_device).unsqueeze(0), 0.0)
                    if block_column_perm is not None:
                        W = W[:, block_column_perm.to(device=cpu_device)]
                else:
                    W = block_input_host
                    block_input_host = None
                    if not self.qcfg.static_groups:
                        self.quantizer.find_params(W, weight=True)
                if Hinv is not None:
                    Hinv = Hinv.to(device=cpu_device)
                if self.qcfg.static_groups:
                    # Static groups measured their parameters before the block
                    # pass; keep those entries but on the retry device.
                    scale, zero = _move_static_groups(groups, cpu_device)
                else:
                    scale = []
                    zero = []
                now_idx = 1

            try:
                Losses = torch.zeros_like(W)
                Q = torch.zeros_like(W)

                fused_block_update = None
                if (
                    Hinv is not None
                    and W.device.type == "cuda"
                    and blocksize == 128
                    and self.qcfg.bits in (2, 3, 4, 5, 6, 7, 8)
                    and self.quantizer._maxq_value == (1 << self.qcfg.bits) - 1
                    and not self.qcfg.mock_quantization
                    and type(self.quantizer) is Quantizer
                    and resolve_kernel_config(self.qcfg).gptq_fused_block_update
                ):
                    from .gptq_cuda import block_update_available, gptq_block_update

                    if block_update_available():
                        fused_block_update = gptq_block_update

                # Use simplified loop when mock_quantization is active
                if self.qcfg.mock_quantization:
                    # Keep each dynamic group within a block. A group that straddles
                    # Q1 would write beyond its bounds and append duplicate metadata.
                    mock_blocksize = (
                        self.qcfg.group_size
                        if not self.qcfg.static_groups and self.qcfg.group_size > 0
                        else blocksize
                    )
                    for i1 in range(0, self.columns, mock_blocksize):
                        i2 = min(i1 + mock_blocksize, self.columns)
                        count = i2 - i1

                        W1 = W[:, i1:i2]
                        Q1 = torch.zeros_like(W1)

                        # Handle group quantization parameters efficiently (similar to original)
                        if self.qcfg.group_size != -1:
                            if not self.qcfg.static_groups:
                                for group_start in range(i1, i2, self.qcfg.group_size):
                                    group_end = min(
                                        group_start + self.qcfg.group_size,
                                        i2,
                                    )
                                    self.quantizer.find_params(
                                        W[:, group_start:group_end],
                                        weight=True,
                                    )
                                    scale.append(self.quantizer.scale)
                                    zero.append(self.quantizer.zero)
                                    now_idx += 1
                                    for column in range(group_start, group_end):
                                        q = self.quantizer.quantize(
                                            W[:, column : column + 1]
                                        ).flatten()
                                        Q1[:, column - i1] = q
                            else:
                                # A static block can cross several reordered groups.
                                for i in range(count):
                                    column = i1 + i
                                    idx = i1 + i
                                    if self.qcfg.desc_act:
                                        idx = perm[idx]
                                    elif self.qcfg.act_group_aware and use_hessian:
                                        idx = final_perm[idx]
                                    self.quantizer = groups[idx // self.qcfg.group_size]
                                    q = self.quantizer.quantize(
                                        W[:, column : column + 1]
                                    ).flatten()
                                    Q1[:, column - i1] = q
                        else:
                            # No grouping - vectorized quantization for entire block
                            maxq_val = 2 ** self.qcfg.bits - 1
                            if hasattr(self.quantizer, 'scale') and hasattr(self.quantizer, 'zero'):
                                latest_scale, latest_zero = _replace_zero_scale(
                                    self.quantizer.scale, self.quantizer.zero
                                )

                                if latest_scale.dim() == 1:
                                    latest_scale = latest_scale.view(-1, 1)
                                if latest_zero.dim() == 1:
                                    latest_zero = latest_zero.view(-1, 1)

                                if self.qcfg.sym:
                                    Q1 = latest_scale * torch.clamp(
                                        torch.round(W1 / latest_scale),
                                        -(maxq_val // 2),
                                        maxq_val // 2
                                    )
                                else:
                                    quantized = torch.clamp(
                                        torch.round(W1 / latest_scale) + latest_zero,
                                        0,
                                        maxq_val
                                    )
                                    Q1 = latest_scale * (quantized - latest_zero)
                            else:
                                # Fallback to individual quantization
                                for i in range(count):
                                    w = W1[:, i]
                                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                                    Q1[:, i] = q

                        Q[:, i1:i2] = Q1
                else:
                    # Original heavy loop for normal quantization
                    effective_block = blocksize
                    if Hinv is None and self.qcfg.group_size and self.qcfg.group_size > 0:
                        # Align RTN fallback work chunks to group boundaries to avoid
                        # redundant quantizer reconfiguration across partial groups.
                        effective_block = self.qcfg.group_size

                    for i1 in range(0, self.columns, effective_block):
                        i2 = min(i1 + effective_block, self.columns)
                        count = i2 - i1

                        W1 = W[:, i1:i2].clone()
                        Q1 = Q[:, i1:i2]
                        # ``Err1`` is fully overwritten per block by the fused kernel (or by
                        # ``torch.sub(..., out=delta)`` in the eager loop) before it is read,
                        # so skip the redundant zero-fill of a 128-column block.
                        Err1 = torch.empty_like(W1) if Hinv is not None else None
                        Losses1 = Losses[:, i1:i2] if Hinv is not None else None

                        if Hinv is not None:
                            Hinv1 = Hinv[i1:i2, i1:i2]

                        if fused_block_update is not None and count == 128:
                            # Group parameters are based on W, which the eager column
                            # loop does not modify until after the block. Thus every
                            # boundary inside this block can be prepared before launch.
                            block_group_size = self.qcfg.group_size
                            uniform_group = block_group_size == -1 or (
                                i1 // block_group_size == (i2 - 1) // block_group_size
                                and (
                                    not self.qcfg.static_groups
                                    or static_group_perm is None
                                )
                            )
                            if uniform_group:
                                # Every column of this block resolves to the same
                                # quantizer group, so the per-column bookkeeping
                                # collapses to one lookup and a constant group index.
                                if block_group_size != -1:
                                    if not self.qcfg.static_groups:
                                        if i1 % block_group_size == 0:
                                            self.quantizer.find_params(
                                                W[:, i1 : i1 + block_group_size], weight=True,
                                            )
                                        if (i1 // block_group_size) - now_idx == -1:
                                            scale.append(self.quantizer.scale)
                                            zero.append(self.quantizer.zero)
                                            now_idx += 1
                                    else:
                                        self.quantizer = groups[i1 // block_group_size]
                                block_scale, block_zero = _replace_zero_scale(
                                    self.quantizer.scale, self.quantizer.zero
                                )
                                block_scale = block_scale.flatten().unsqueeze(0)
                                block_zero = block_zero.flatten().unsqueeze(0)
                                column_group = _uniform_column_group(count, W.device)
                            else:
                                block_scales = []
                                block_zeros = []
                                column_groups = []
                                parameter_indices = {}
                                for i in range(count):
                                    column = i1 + i
                                    if self.qcfg.group_size != -1:
                                        if not self.qcfg.static_groups:
                                            if column % self.qcfg.group_size == 0:
                                                self.quantizer.find_params(
                                                    W[:, column : column + self.qcfg.group_size], weight=True,
                                                )
                                            if (column // self.qcfg.group_size) - now_idx == -1:
                                                scale.append(self.quantizer.scale)
                                                zero.append(self.quantizer.zero)
                                                now_idx += 1
                                        else:
                                            source_column = (
                                                static_group_perm[column].item()
                                                if static_group_perm is not None
                                                else column
                                            )
                                            self.quantizer = groups[
                                                source_column // self.qcfg.group_size
                                            ]

                                    key = id(self.quantizer.scale)
                                    group_index = parameter_indices.get(key)
                                    if group_index is None:
                                        group_index = len(block_scales)
                                        parameter_indices[key] = group_index
                                        block_scale, block_zero = _replace_zero_scale(
                                            self.quantizer.scale, self.quantizer.zero
                                        )
                                        block_scales.append(block_scale.flatten())
                                        block_zeros.append(block_zero.flatten())
                                    column_groups.append(group_index)

                                block_scale = torch.stack(block_scales)
                                block_zero = torch.stack(block_zeros)
                                column_group = torch.tensor(column_groups, device=W.device, dtype=torch.int32)
                            fused_block_update(
                                W1, Hinv1, block_scale, block_zero, column_group,
                                Q1, Err1, Losses1, (1 << self.qcfg.bits) - 1,
                            )
                        else:
                            for i in range(count):
                                w = W1[:, i]
                                if Hinv is not None:
                                    d = Hinv1[i, i]

                                if self.qcfg.group_size != -1:
                                    if not self.qcfg.static_groups:
                                        if (i1 + i) % self.qcfg.group_size == 0:
                                            self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + self.qcfg.group_size)], weight=True)

                                        if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
                                            scale.append(self.quantizer.scale)
                                            zero.append(self.quantizer.zero)
                                            now_idx += 1
                                    else:
                                        idx = i1 + i
                                        if static_group_perm is not None:
                                            idx = static_group_perm[idx].item()

                                        self.quantizer = groups[idx // self.qcfg.group_size]

                                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                                Q1[:, i] = q
                                if Hinv is not None:
                                    delta = Err1[:, i]
                                    torch.sub(w, q, out=delta)
                                    Losses1[:, i] = delta.square() / d.square()
                                    delta.div_(d)
                                    W1[:, i:] -= delta.unsqueeze(1) * Hinv1[i, i:]

                        if Hinv is not None:
                            Losses1.div_(2)
                            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

                        del W1, Q1, Err1, Losses1
                        if Hinv is not None:
                            del Hinv1
                break
            except RuntimeError as exc:
                if (
                    block_retry
                    or W.device.type != "cuda"
                    or not _is_memory_allocation_failure(exc)
                ):
                    raise
                self.log_cpu_fallback("block update", W.device)
                cpu_fallback_used = True
                block_retry = True

        if Hinv is not None:
            del Hinv
            if self.nsamples != 0:
                avg_loss = torch.sum(Losses).item() / self.nsamples

                if math.isnan(avg_loss):
                    print("Losses sum item:", torch.sum(Losses).item())
                    if fallback_configured:
                        log.info(f"Quantization: Failed due to `NaN` loss for `{self.name}`, use mock quantization retry for `{self.name}`")
                        del Losses, Q, W
                        if retry_hessian_diag is not None:
                            self.H = torch.diag(retry_hessian_diag)
                        self.qcfg.mock_quantization = True
                        return self.quantize(blocksize=blocksize)
                    else:
                        raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`, please try increasing calibration data samples or enable fallback=True")
            else:
                if fallback_configured:
                    log.warn(f"Quantization: Module `{self.name}` -> using fail safe mode. Please check if calibration data is sufficient.")
                else:
                    log.warn(f"Quantization: `{self.name}` is not activated due to model inference logic (MoE)")
                avg_loss = f"{resolved_strategy.value} fallback" if fallback_configured else 999999999
        else:
            avg_loss = f"{resolved_strategy.value} fallback" if fallback_configured else 999999999

        del Losses
        del self.H
        del W

        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = (perm // group_size).to(dtype=torch.int32, device=Q.device)
        else:
            g_idx = torch.arange(self.columns, dtype=torch.int32, device=Q.device) // group_size

        if self.qcfg.desc_act and use_hessian:
            invperm = invperm.to(device=Q.device)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]
            del perm, invperm

        elif self.qcfg.act_group_aware and use_hessian:
            inv_final = invert_perm(final_perm).to(device=Q.device)
            Q = Q[:, inv_final]
            if not self.qcfg.static_groups:
                # Dynamic parameters were measured in reordered group order.
                # Static parameters were measured before reordering and already
                # match the original-column g_idx returned below.
                inv_global_perm = invert_perm(global_perm)
                inv_global_perm_list = inv_global_perm.tolist()
                reordered_group_count = len(inv_global_perm_list)
                temp_scale = [scale[i] for i in inv_global_perm_list]
                temp_scale.extend(scale[reordered_group_count:])
                scale = temp_scale
                temp_zero = [zero[i] for i in inv_global_perm_list]
                temp_zero.extend(zero[reordered_group_count:])
                zero = temp_zero
            del final_perm, inv_final, global_perm, local_perms

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            g_idx = g_idx[:valid_cols]

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).to(self.module.weight.dtype)
        else:
            Q = Q.to(self.module.weight.dtype)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)

        scale, zero = self._sanitize_scale_and_log(scale, zero)

        if cpu_fallback_used and Q.device != result_device:
            log.info(
                "Quantization: Module `%s` -> CPU fallback complete; moving final quantized weights back to %s.",
                self.name,
                result_device,
            )

        # ``result_device`` is the module's device, which can differ from the
        # Hessian/compute device when a follower adopted a leader's Hessian.
        # That makes this a cross-device move, so route it through the
        # direction-aware helper; it returns ``Q`` unchanged when the devices
        # already match.
        Q = safe_move_to(Q, result_device)

        duration = time.time() - start

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def borrow_materialized_chunk_stats(self, reset: bool = False) -> Dict[str, int]:
        stats = dict(self._borrow_workspace_stats)
        if reset:
            for key in self._borrow_workspace_stats:
                self._borrow_workspace_stats[key] = 0
        return stats

    def _snapshot_borrow_workspace_stats(self, *, context: str) -> None:
        stats = self.borrow_materialized_chunk_stats(reset=True)
        total_requests = int(stats.get("requests", 0) or 0)
        if total_requests == 0:
            return

        materialized_hits = int(stats.get("materialized_hits", 0) or 0)
        materialized_misses = int(stats.get("materialized_misses", 0) or 0)
        staging_hits = int(stats.get("staging_hits", 0) or 0)
        staging_misses = int(stats.get("staging_misses", 0) or 0)
        chunk_rows = self._borrow_workspace_last_chunk_rows
        stage_dtype = self._borrow_workspace_stage_dtype
        stage_dtype_str = str(stage_dtype) if stage_dtype is not None else "n/a"
        hit_rate = materialized_hits / total_requests if total_requests else 0.0

        summary = {
            "context": context,
            "requests": total_requests,
            "materialized_hits": materialized_hits,
            "materialized_misses": materialized_misses,
            "staging_hits": staging_hits,
            "staging_misses": staging_misses,
            "chunk_rows": chunk_rows,
            "staging_dtype": stage_dtype_str,
            "hit_rate": hit_rate,
        }
        self._borrow_workspace_last_summary = summary

        totals = self._borrow_workspace_totals
        totals["requests"] += total_requests
        totals["materialized_hits"] += materialized_hits
        totals["materialized_misses"] += materialized_misses
        totals["staging_hits"] += staging_hits
        totals["staging_misses"] += staging_misses

    def log_workspace_stats(self, *, context: str, reset: bool = True) -> None:
        totals = self._borrow_workspace_totals
        total_requests = int(totals.get("requests", 0) or 0)
        if total_requests == 0:
            if reset:
                self.reset_workspace_stats()
            return

        total_hits = int(totals.get("materialized_hits", 0) or 0)
        total_misses = int(totals.get("materialized_misses", 0) or 0)
        total_hit_rate = total_hits / total_requests if total_requests else 0.0

        last = self._borrow_workspace_last_summary or {}
        last_requests = int(last.get("requests", 0) or 0)
        last_hits = int(last.get("materialized_hits", 0) or 0)
        last_misses = int(last.get("materialized_misses", 0) or 0)
        last_hit_rate = float(last.get("hit_rate", 0.0) or 0.0)
        rows_label = last.get("chunk_rows", "n/a")
        stage_dtype = last.get("staging_dtype", "n/a")

        log.info(
            "GPTQ workspace cache [%s]: module=%s rows=%s staging_dtype=%s "
            "requests=%d hits=%d misses=%d hit_rate=%.2f total_requests=%d "
            "total_hits=%d total_misses=%d total_hit_rate=%.2f",
            context,
            getattr(self, "name", "<unknown>"),
            rows_label,
            stage_dtype,
            last_requests,
            last_hits,
            last_misses,
            last_hit_rate,
            total_requests,
            total_hits,
            total_misses,
            total_hit_rate,
        )

        if reset:
            self.reset_workspace_stats()

    def reset_workspace_stats(self) -> None:
        for key in self._borrow_workspace_stats:
            self._borrow_workspace_stats[key] = 0
        for key in self._borrow_workspace_totals:
            self._borrow_workspace_totals[key] = 0
        self._borrow_workspace_last_summary = None
        self._borrow_workspace_stage_dtype = None
        self._borrow_workspace_last_chunk_rows = None

    def release_hessian_state(self) -> None:
        """Drop the accumulated Hessian and its staging state (idempotent).

        The accumulator is the largest per-module allocation (one fp32
        ``[columns, columns]`` tensor). :meth:`free` calls this as part of the
        full success-path teardown, and a failing :meth:`quantize` calls it
        directly so a failure does not pin the accumulator until the processor
        is destroyed (#5026). A shared capture stays live until the final
        projection releases it; factor-cache ownership remains
        projection-local.
        """

        with self.lock:
            self._shared_hessian_cache = None
            self._shared_hessian_cache_disabled = False
        if not self._hessian_state_released:
            self._hessian_state.release()
            self._hessian_state_released = True

    def free(self):
        self.release_hessian_state()
        if hasattr(self, "_H_diag"):
            del self._H_diag
        self._device_embedding_counts.clear()
        del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy

        if self._named_module is not None:
            self._named_module.state.pop("tp_pad_info", None)

        target = getattr(self, "module", None)
        if target is not None:
            del self.module

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]
