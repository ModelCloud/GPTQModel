# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utilities for orchestrating the quantisation loop across multiple devices.

ModuleLooper is the high-level coordinator that fans calibration batches across
the available accelerators, runs each processing stage, and keeps the shell and
turtle model state coherent. The implementation mixes synchronous orchestration
with asynchronous workers, so the helpers below focus on keeping device context
consistent and ensuring data dependencies survive the roundtrips through the
thread pool.
"""

from __future__ import annotations

import math
import threading
import time
import logging
import os
from concurrent.futures import as_completed
from typing import Dict, List, NamedTuple, Optional, TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ..looper.dequantize_processor import DequantizeProcessor
from ..looper.eora_processor import EoraProcessor
from ..looper.gptq_processor import GPTQProcessor
from ..looper.input_cache import InputCache
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import SUPPORTS_MODULE_TYPES
from ..models.base import CAPTURE_ONLY_FLAG
from ..nn_modules.hooked_linear import HookedLinear, replace_module_with_hooked_legacy
from ..quantization.config import METHOD, VramStrategy
from ..utils.attn_mask import apply_keep_mask_bt
from ..utils.ctx import ctx
from ..utils.device_telemetry import emit_device_telemetry
from ..utils.device import get_device, get_device_new
from ..utils.disk import estimate_disk_io_speed
from ..utils.logger import setup_logger, log_time_block
from ..utils.looper_helpers import (
    clone_module_for_devices,
    device_ctx,
    forward_batch_worker,
    normalize_device_like,
    rehome_module_to_device,
    select_forward_devices,
)
from ..utils.model import find_modules, get_module, get_module_by_name_prefix, move_to, MoETopKState, set_moe_topk, restore_moe_topk
from ..utils.offload import offload_to_disk
from ..utils.python import has_gil_control, has_gil_disabled
from ..utils.torch import (CPU, META, timed_gc_collect, torch_sync, tf32_high_precision_guard)
from .. import DEVICE_THREAD_POOL
from .awq_processor import AWQProcessor
from .forward_executor import ForwardExecutor
from .paroquant_processor import ParoQuantProcessor
from .qqq_processor import QQQProcessor
from .stage_inputs_capture import StageInputsCapture
from .stage_layer import run_layer_stage

log = setup_logger()

_IO_WRITE_SPEED_MB: Optional[float] = None
_IO_WRITE_SPEED_LOCK = threading.Lock()

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from logbar.progress import ProgressBar


class FinalizeProgressInfo(NamedTuple):
    """Progress payload for processor finalization reporting."""

    module_label: Optional[str]
    process_name: str
    layer_idx: Optional[int]


def _restrict_quant_devices_for_method(method: Any, quant_devices: List[torch.device]) -> List[torch.device]:
    """Apply method-specific device constraints for quantization workers."""

    try:
        normalized_method = METHOD(method) if method is not None else None
    except (TypeError, ValueError):
        normalized_method = None

    if normalized_method != METHOD.PARO or not quant_devices:
        return quant_devices

    non_cpu_devices = [device for device in quant_devices if getattr(device, "type", None) != "cpu"]
    if non_cpu_devices:
        return [non_cpu_devices[0]]

    return quant_devices[:1]


def _resolve_strategy_device_pool(
    configured_devices: Optional[List[str]],
    available_devices: List[torch.device],
    *,
    label: str,
) -> List[torch.device]:
    """Resolve one strategy device pool as a validated subset of available devices."""

    if not configured_devices:
        return list(available_devices)

    available_by_name = {
        str(normalize_device_like(device)): normalize_device_like(device)
        for device in available_devices
        if normalize_device_like(device) is not None
    }
    resolved: List[torch.device] = []
    for device_name in configured_devices:
        normalized = normalize_device_like(device_name)
        if normalized is None:
            raise ValueError(f"ModuleLooper: {label} device pool contains an unsupported device value: {device_name!r}.")
        matched = available_by_name.get(str(normalized))
        if matched is None:
            raise ValueError(
                f"ModuleLooper: {label} device pool {configured_devices} must be a subset of the visible compute devices "
                f"{list(available_by_name.keys())}."
            )
        if matched not in resolved:
            resolved.append(matched)

    if not resolved:
        raise ValueError(f"ModuleLooper: {label} device pool is empty after normalization.")

    return resolved


class StopMainLoop(Exception):
    """Signal that the module loop should abort immediately."""


def io_write_performance() -> Optional[float]:
    """Estimate and cache sustained disk write throughput in MB/s."""

    global _IO_WRITE_SPEED_MB
    if _IO_WRITE_SPEED_MB is not None:
        return _IO_WRITE_SPEED_MB
    with _IO_WRITE_SPEED_LOCK:
        if _IO_WRITE_SPEED_MB is not None:
            return _IO_WRITE_SPEED_MB
        disk_speed = estimate_disk_io_speed()
        _IO_WRITE_SPEED_MB = disk_speed / (1024 * 1024)
    return _IO_WRITE_SPEED_MB


class ModuleLooper():
    """Drive the per-layer quantisation workflow over one or more devices.

    The looper executes work on the shared global :class:`DeviceThreadPool`
    instance so tasks such as module reloading, forward passes, and finalisation
    reuse the same worker threads.
    """
    def __init__(self, model: BaseQModel, processors: List[LoopProcessor]):
        """Initialize loop state, device policy, and callback wiring."""

        self.processors = processors
        self.gptq_model = model

        self.support_batch_quantize = model.support_batch_quantize
        self.lock = threading.Lock()
        self._forward_executor = ForwardExecutor(self, logger=log)
        self._layer_callback = getattr(model, "layer_callback", None)
        self._loop_stop_event = threading.Event()
        self._loop_stop_exc: Optional[BaseException] = None
        self._loop_stop_waited = False
        self._dangling_threads: List[threading.Thread] = []
        self._dangling_threads_lock = threading.Lock()

        io_write_speed = io_write_performance()
        if io_write_speed is not None:
            if io_write_speed < 100:
                log.error(
                    "Disk subsystem write throughput detected at "
                    f"{io_write_speed:.1f} MB/s; quantization may be severely slowed by IO."
                )
            elif io_write_speed < 200:
                log.warn(
                    "Disk subsystem write throughput detected at "
                    f"{io_write_speed:.1f} MB/s; quantization may be slowed by IO."
                )
            else:
                log.info(
                    "Disk subsystem write throughput detected at "
                    f"{io_write_speed:.1f} MB/s."
                )

        quant_device_hint = getattr(self.gptq_model.quantize_config, "device", None)
        normalized_quant_device = normalize_device_like(quant_device_hint)
        quant_devices = select_forward_devices(normalized_quant_device) if normalized_quant_device else [CPU]
        if not quant_devices:
            quant_devices = [CPU]

        # Apply compute device filter if provided to determine which devices to use for quantization
        compute_device_filter = getattr(self.gptq_model.quantize_config, "compute_device_filter", None)
        if compute_device_filter is not None:
            quant_devices_filtered = compute_device_filter(quant_devices)
            if len(quant_devices_filtered) >= 1:
                quant_devices = quant_devices_filtered
            else:
                log.warn(
                    "compute_device_filter returned empty device list. "
                    "Using all devices for quantization."
                )

        restricted_quant_devices = _restrict_quant_devices_for_method(
            getattr(self.gptq_model.quantize_config, "method", None),
            quant_devices,
        )
        if restricted_quant_devices != quant_devices:
            log.warn(
                "ModuleLooper: METHOD.PARO forcing single-device quantization on `%s`; "
                "ignoring additional devices %s to avoid multi-GPU sync issues.",
                restricted_quant_devices[0],
                [str(device) for device in quant_devices if device != restricted_quant_devices[0]],
            )
            quant_devices = restricted_quant_devices

        self._quant_devices = quant_devices
        self._quant_device_rr = 0
        self._module_device_map: Dict[str, torch.device] = {}
        self._quant_device_lock = threading.Lock()

        # Resolve the user-facing split dense/MoE placement settings once at
        # looper construction time so subset planning can reuse stable pools.
        dense_vram_strategy = getattr(self.gptq_model.quantize_config, "dense_vram_strategy", VramStrategy.EXCLUSIVE)
        if isinstance(dense_vram_strategy, str):
            try:
                dense_vram_strategy = VramStrategy(dense_vram_strategy.lower())
            except ValueError:
                dense_vram_strategy = VramStrategy.EXCLUSIVE
        supported_dense_strategies = getattr(
            self.gptq_model,
            "supported_dense_vram_strategies",
            [
                VramStrategy.EXCLUSIVE,
                VramStrategy.BALANCED,
            ],
        )
        if isinstance(supported_dense_strategies, VramStrategy):
            supported_dense_strategies = [supported_dense_strategies]
        if dense_vram_strategy not in supported_dense_strategies:
            log.debug(
                "ModuleLooper: Model %s does not support dense VRAM strategy %s; falling back to exclusive.",
                getattr(self.gptq_model, "__class__", type(self.gptq_model)).__name__,
                dense_vram_strategy,
            )
            dense_vram_strategy = VramStrategy.EXCLUSIVE

        moe_vram_strategy = getattr(
            self.gptq_model.quantize_config,
            "moe_vram_strategy",
            VramStrategy.EXCLUSIVE,
        )
        if isinstance(moe_vram_strategy, str):
            try:
                moe_vram_strategy = VramStrategy(moe_vram_strategy.lower())
            except ValueError:
                moe_vram_strategy = VramStrategy.EXCLUSIVE
        supported_moe_strategies = getattr(
            self.gptq_model,
            "supported_moe_vram_strategies",
            [
                VramStrategy.EXCLUSIVE,
                VramStrategy.BALANCED,
            ],
        )
        if isinstance(supported_moe_strategies, VramStrategy):
            supported_moe_strategies = [supported_moe_strategies]
        if moe_vram_strategy not in supported_moe_strategies:
            log.debug(
                "ModuleLooper: Model %s does not support MoE VRAM strategy %s; falling back to exclusive.",
                getattr(self.gptq_model, "__class__", type(self.gptq_model)).__name__,
                moe_vram_strategy,
            )
            moe_vram_strategy = VramStrategy.EXCLUSIVE

        self._dense_vram_strategy = dense_vram_strategy
        self._moe_vram_strategy = moe_vram_strategy
        dense_strategy_devices = getattr(self.gptq_model.quantize_config, "dense_vram_strategy_devices", None)
        moe_strategy_devices = getattr(self.gptq_model.quantize_config, "moe_vram_strategy_devices", None)
        self._dense_quant_devices = _resolve_strategy_device_pool(
            dense_strategy_devices,
            quant_devices,
            label="dense_vram_strategy_devices",
        )
        self._moe_quant_devices = _resolve_strategy_device_pool(
            moe_strategy_devices,
            quant_devices,
            label="moe_vram_strategy_devices",
        )
        # Keep a cheap flag so the planner can skip split-pool logic entirely
        # when the user leaves a pool on the default exclusive behavior.
        self._dense_vram_strategy_explicit = bool(dense_strategy_devices) or self._dense_vram_strategy != VramStrategy.EXCLUSIVE
        self._moe_vram_strategy_explicit = bool(moe_strategy_devices) or self._moe_vram_strategy != VramStrategy.EXCLUSIVE

        self._moe_subset_threshold = 16
        self._subset_callback = getattr(self.gptq_model, "subset_callback", None)

        # Track current subset for MoE lifecycle hooks
        self._current_subset: Optional[Dict[str, Any]] = None

        # moe_routing_override is only required for MoE models (i.e., models with dynamic_expert_index).
        if getattr(self.gptq_model, "dynamic_expert_index", None):
            num_experts = self.gptq_model.get_num_experts(self.gptq_model.model.config)
            self.moe_routing_override = self.gptq_model.quantize_config.moe_routing_override(num_experts)
        else:
            self.moe_routing_override = None
        self.moe_routing_bypass = self.gptq_model.quantize_config.moe_routing_bypass()
        self._emit_moe_parallel_quant_runtime()

        for processor in self.processors:
            self._processor_mask_tls(processor)

    def _emit_moe_parallel_quant_runtime(self) -> None:
        """Log the runtime knobs that decide whether MoE quant can fan out efficiently."""

        if not getattr(self.gptq_model, "dynamic_expert_index", None):
            return

        dense_devices = [str(device) for device in self._dense_quant_devices]
        moe_devices = [str(device) for device in self._moe_quant_devices]
        gil_env = os.environ.get("PYTHON_GIL")
        gil_disabled = has_gil_disabled()
        gil_controllable = has_gil_control()
        routing_mode = (
            "override"
            if self.moe_routing_override is not None
            else "bypass"
            if self.moe_routing_bypass
            else "native"
        )
        free_threaded_parallel_quant_eligible = bool(gil_disabled and len(self._moe_quant_devices) > 0)

        log.info(
            "ModuleLooper: MoE quant runtime dense_pool=%s moe_pool=%s routing_mode=%s routing_override=%s "
            "PYTHON_GIL=%s gil_disabled=%s free_threaded_parallel_quant_eligible=%s",
            dense_devices,
            moe_devices,
            routing_mode,
            self.moe_routing_override,
            gil_env,
            gil_disabled,
            free_threaded_parallel_quant_eligible,
        )
        if moe_devices and gil_controllable and not gil_disabled:
            log.warn(
                "ModuleLooper: MoE quant is configured for device fan-out across %s but Python GIL is still enabled; "
                "rerun with PYTHON_GIL=0 to unlock the free-threaded parallel quant path.",
                moe_devices,
            )

        emit_device_telemetry(
            "moe_parallel_quant_runtime",
            dense_devices=dense_devices,
            moe_devices=moe_devices,
            dense_strategy=self._dense_vram_strategy,
            moe_strategy=self._moe_vram_strategy,
            routing_mode=routing_mode,
            routing_override=self.moe_routing_override,
            routing_bypass=self.moe_routing_bypass,
            python_gil_env=gil_env,
            python_gil_controllable=gil_controllable,
            python_gil_disabled=gil_disabled,
            free_threaded_parallel_quant_eligible=free_threaded_parallel_quant_eligible,
        )

    class MoERoutingOverrideContext:
        """
        Context manager that temporarily overrides MoE routing top-k.

        On entry, applies the specified top-k override to all MoE routing modules.
        On exit, restores the original routing configuration.
        """

        def __init__(self, model, moe_routing_override: int):
            """Capture the model and temporary top-k override to apply."""

            # Model containing MoE routing modules
            self.model = model
            # Target top-k value for per-token expert routing
            self.moe_routing_override = moe_routing_override
            # Saved state for restoring original top-k values
            self._state: MoETopKState | None = None

        def __enter__(self):
            """Apply the temporary routing override before the forward pass."""

            # Apply routing override if specified
            if self.moe_routing_override:
                self._state = set_moe_topk(self.model, self.moe_routing_override)
            return self

        def __exit__(self, exc_type, exc, tb):
            """Restore the original routing state when leaving the context."""

            # Restore original routing configuration
            if self.moe_routing_override:
                restore_moe_topk(self._state)
            return False  # Do not suppress exceptions

    class MoELifecycleContext:
        """Context manager for MoE lifecycle hooks integration."""

        def __init__(self, module_looper, module, processor, current_subset, ordered_module_names):
            """Capture the replica state needed to patch the MoE block."""

            self.module_looper = module_looper
            self.module = module
            self.processor = processor
            self.current_subset = current_subset
            self.ordered_module_names = ordered_module_names
            self.moe_hooks_active = False
            self.moe_block = None
            self.moe_forward_original = None

        def __enter__(self):
            """Set up MoE lifecycle hooks if applicable."""
            if self.module_looper._should_use_moe_lifecycle(self.module, self.processor):
                hooks = self.module_looper.gptq_model.moe_lifecycle_hooks
                self.moe_block = hooks.get_moe_block(self.module, self.module_looper.gptq_model.__class__)

                if self.moe_block is not None:
                    # Save original forward method
                    self.moe_forward_original = self.moe_block.forward

                    # Create wrapper that forwards to all experts
                    moe_block_prefix = hooks._extract_moe_block_prefix(self.current_subset, self.moe_block)

                    def moe_forward_wrapper(hidden_states, **kwargs):
                        """Route the replica forward through the all-experts hook."""

                        return hooks.forward_to_all_experts(
                            moe_block=self.moe_block,
                            hidden_states=hidden_states,
                            processor=self.processor,
                            subset=self.current_subset,
                            ordered_module_names=self.ordered_module_names,
                            original_forward=self.moe_forward_original,
                            model_class=self.module_looper.gptq_model.__class__,
                            module_looper=self.module_looper,  # Pass for TLS-based hooks pausing
                            moe_block_prefix=moe_block_prefix,
                            replica_module=self.module,  # Pass replica for device-correct module resolution
                            **kwargs
                        )

                    # Temporarily replace forward method
                    self.moe_block.forward = moe_forward_wrapper
                    self.moe_hooks_active = True

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Restore original MoE forward method if it was patched."""
            if self.moe_hooks_active and self.moe_forward_original is not None and self.moe_block is not None:
                self.moe_block.forward = self.moe_forward_original
            return False  # Don't suppress exceptions

    def register_layer_callback(self, callback) -> None:
        """Register or replace the layer-complete callback target."""
        self._layer_callback = callback

    def register_subset_callback(self, callback) -> None:
        """Register or replace the subset event callback target."""
        self._subset_callback = callback

    def register_dangling_thread(self, watcher: threading.Thread) -> None:
        """Track a watcher thread that should be joined before exit."""

        with self._dangling_threads_lock:
            if self._dangling_threads:
                self._dangling_threads = [
                    thread for thread in self._dangling_threads if thread.is_alive()
                ]
            self._dangling_threads.append(watcher)

    def wait_dangling_threads(self) -> None:
        """Join any still-running watcher threads and clear the registry."""

        with self._dangling_threads_lock:
            threads = list(self._dangling_threads)
            self._dangling_threads.clear()
        alive_threads = [thread for thread in threads if thread.is_alive()]
        for thread in alive_threads:
            thread.join()

    def _resolve_layer_callback(self):
        """Resolve the active layer-complete callback using legacy fallbacks."""

        for candidate in (
            getattr(self, "_layer_callback", None),
            getattr(self, "layer_callback", None),
            getattr(self.gptq_model, "layer_callback", None),
            getattr(self.gptq_model, "callbackup", None),
            getattr(self.gptq_model, "callback", None),
        ):
            if candidate is not None:
                return candidate
        return None

    def _resolve_subset_callback(self):
        """Resolve the active subset callback from looper or model state."""

        for candidate in (
            getattr(self, "_subset_callback", None),
            getattr(self, "subset_callback", None),
            getattr(self.gptq_model, "subset_callback", None),
        ):
            if candidate is not None:
                return candidate
        return None

    def callbackup(self, layer_idx: int, submodule_finalized: bool):
        """Invoke the layer callback and normalize stop-loop responses."""

        callback = self._resolve_layer_callback()
        if callback is None:
            return None

        handler = getattr(callback, "layer_complete", None)
        if handler is None and callable(callback):
            handler = callback
        if handler is None:
            return None

        try:
            result = handler(layer_idx=layer_idx, submodule_finalized=submodule_finalized)
        except StopMainLoop:
            raise
        if result is StopMainLoop:
            raise StopMainLoop(f"Layer callback requested stop at layer {layer_idx}")
        if isinstance(result, StopMainLoop):
            raise result
        return result

    def _subset_event_dispatch(
        self,
        *,
        stage: str,
        layer_idx: int,
        subset_index: int,
        subset_total: int,
        module_names: List[str],
        processor: str,
    ) -> None:
        """Emit a subset event immediately and surface callback failures."""

        self._emit_subset_event(
            stage=stage,
            layer_idx=layer_idx,
            subset_index=subset_index,
            subset_total=subset_total,
            module_names=module_names,
            processor=processor,
            raise_in_place=True,
        )

    def _request_loop_stop(self, exc: Optional[BaseException]) -> None:
        """Record the first stop reason and signal loop shutdown."""

        with self.lock:
            if self._loop_stop_exc is None and exc is not None:
                self._loop_stop_exc = exc
        self._loop_stop_event.set()

    def _check_loop_stop(self) -> bool:
        """Drain outstanding work and re-raise any recorded stop signal."""

        if not self._loop_stop_event.is_set():
            return False
        if not self._loop_stop_waited:
            DEVICE_THREAD_POOL.wait()
            self._loop_stop_waited = True
        if self._loop_stop_exc is not None:
            raise self._loop_stop_exc
        return True

    def _emit_subset_event(
        self,
        *,
        stage: str,
        layer_idx: int,
        subset_index: int,
        subset_total: int,
        module_names: List[str],
        processor: str,
        raise_in_place: bool,
    ) -> None:
        """Forward a subset lifecycle event to the configured callback."""

        callback = self._resolve_subset_callback()
        if callback is None:
            return

        handler = getattr(callback, "subset_event", None)
        if handler is None and callable(callback):
            handler = callback
        if handler is None:
            return

        try:
            result = handler(
                stage=stage,
                layer_idx=layer_idx,
                subset_index=subset_index,
                subset_total=subset_total,
                module_names=module_names,
                processor=processor,
            )
        except StopMainLoop as exc:
            self._request_loop_stop(exc)
            if raise_in_place:
                raise
            return
        except BaseException as exc:
            self._request_loop_stop(exc)
            if raise_in_place:
                raise
            return

        if result is StopMainLoop:
            exc = StopMainLoop(f"Subset callback requested stop at layer {layer_idx} subset {subset_index}")
            self._request_loop_stop(exc)
            if raise_in_place:
                raise exc
            return

        if isinstance(result, StopMainLoop):
            self._request_loop_stop(result)
            if raise_in_place:
                raise result
            return

    def _emit_layer_complete(
        self,
        layer_idx: int,
        submodule_finalized: bool,
        *,
        raise_in_place: bool,
    ) -> None:
        """Notify listeners that a layer finished and handle stop requests."""

        try:
            self.callbackup(layer_idx=layer_idx, submodule_finalized=submodule_finalized)
        except StopMainLoop:
            self._request_loop_stop(None)
            return
        except BaseException as exc:
            if raise_in_place:
                raise
            log.exception(
                "Layer completion callback raised an exception (layer=%s, submodule_finalized=%s)",
                layer_idx,
                submodule_finalized,
            )
            self._request_loop_stop(exc)

    # Processors capture activations through hooks that need thread-local state
    # so masks survive the roundtrip to worker threads.
    def _processor_mask_tls(self, processor: LoopProcessor) -> threading.local:
        """Get or create thread-local storage for the active keep mask."""

        tls = getattr(processor, "_mask_tls", None)
        if tls is None:
            tls = threading.local()
            setattr(processor, "_mask_tls", tls)
        return tls

    def _processor_hooks_paused_tls(self, processor):
        """Get or create thread-local storage for hooks_paused flag."""
        if not hasattr(processor, "_hooks_paused_tls"):
            processor._hooks_paused_tls = threading.local()
        return processor._hooks_paused_tls

    def _set_processor_hooks_paused(self, processor: LoopProcessor, paused: bool):
        """Set hooks paused state for current thread."""
        tls = self._processor_hooks_paused_tls(processor)
        tls.value = paused

    def _get_processor_hooks_paused(self, processor: LoopProcessor) -> bool:
        """Get hooks paused state for current thread (thread-safe)."""
        tls = getattr(processor, "_hooks_paused_tls", None)
        return getattr(tls, "value", False) if tls else False

    def _set_processor_mask(self, processor: LoopProcessor, mask):
        """Store the active sequence mask for the current worker thread."""

        tls = self._processor_mask_tls(processor)
        tls.value = mask

    def _get_processor_mask(self, processor: LoopProcessor):
        """Return the sequence mask bound to the current worker thread."""

        tls = getattr(processor, "_mask_tls", None)
        return getattr(tls, "value", None) if tls else None

    def _safe_len(self, sequence) -> Optional[int]:
        """Return ``len(sequence)`` when the object exposes a safe length."""

        if sequence is None:
            return None
        try:
            return len(sequence)
        except (TypeError, AttributeError):
            return None

    def _coerce_to_int(self, value) -> Optional[int]:
        """Best-effort conversion for scalar-like values used in counters."""

        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value

        indexer = getattr(value, "__index__", None)
        if callable(indexer):
            try:
                return indexer()
            except Exception:
                pass

        if torch.is_tensor(value):
            if value.numel() == 1:
                try:
                    return int(value.item())
                except Exception:
                    return None
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _resolve_batch_total(self, raw_count, fallback_sequence) -> int:
        """Resolve a non-negative batch count from explicit or inferred input."""

        count = self._coerce_to_int(raw_count)
        fallback_len = self._safe_len(fallback_sequence)
        fallback = self._coerce_to_int(fallback_len)

        if count is not None and count > 0:
            if fallback is not None and fallback >= 0:
                return min(count, fallback)
            return count

        if fallback is not None:
            return max(fallback, 0)

        if count is not None:
            return max(count, 0)

        return 0

    def _batch_row_count(self, batch_inputs: Optional[List[torch.Tensor]]) -> int:
        """Infer how many rows a cached batch contributes to progress."""

        if not batch_inputs:
            return 0

        primary = batch_inputs[0]

        if isinstance(primary, torch.Tensor):
            if primary.ndim > 0:
                return max(int(primary.shape[0]), 0)
            return max(int(primary.numel()), 0)

        if isinstance(primary, (list, tuple)) and primary:
            nested_first = primary[0]
            if isinstance(nested_first, torch.Tensor) and nested_first.ndim > 0:
                return max(int(nested_first.shape[0]), 0)

        return 0

    def _collect_row_counts(self, layer_inputs: Optional[List[List[torch.Tensor]]]) -> List[int]:
        """Collect per-batch row counts for progress tracking."""

        if not layer_inputs:
            return []

        counts: List[int] = []
        for batch_inputs in layer_inputs:
            count = self._batch_row_count(batch_inputs)
            counts.append(count if count > 0 else 0)
        return counts

    def _extract_moe_group_key(self, module_name: Optional[str]) -> Optional[str]:
        """Collapse expert module names into a stable MoE routing group key."""

        if not module_name:
            return None

        if ".experts." in module_name:
            prefix, remainder = module_name.split(".experts.", 1)
            expert_id = remainder.split(".", 1)[0]
            if expert_id:
                return f"{prefix}.experts.{expert_id}"
            return None

        if ".shared_experts." in module_name:
            prefix, _ = module_name.split(".shared_experts.", 1)
            return f"{prefix}.shared_experts"

        if ".shared_expert." in module_name:
            prefix, _ = module_name.split(".shared_expert.", 1)
            return f"{prefix}.shared_expert"

        return None

    def _is_attention_module_name(self, module_name: str) -> bool:
        """Heuristically detect attention modules from their qualified name."""

        if not module_name:
            return False

        lowered = module_name.lower()
        if ".attn" in lowered or "attn." in lowered:
            return True
        if ".attention" in lowered or "attention." in lowered:
            return True
        if lowered.endswith("attn") or lowered.endswith("attention"):
            return True
        return False

    def _should_use_moe_lifecycle(self, module: nn.Module, processor: LoopProcessor) -> bool:
        """
        Check if MoE lifecycle hooks should be used for this module.

        Returns True if:
        - pass_whole_dataset_to_each_expert flag is enabled
        - Model has lifecycle hooks configured
        - Module contains an MoE block
        """
        # Check if feature is enabled
        flag_enabled = self.gptq_model.quantize_config.moe_routing_bypass()
        if not flag_enabled:
            return False

        # Check if model has lifecycle hooks
        hooks = getattr(self.gptq_model, 'moe_lifecycle_hooks', None)
        if hooks is None:
            log.warn(
                f"pass_whole_dataset_to_each_expert is enabled but {self.gptq_model.__class__.__name__} "
                f"model does not have 'moe_lifecycle_hooks' configured. MoE optimization will be disabled. "
                f"Please ensure your model definition has proper MoE lifecycle hooks configured."
            )
            return False

        # Check if this module contains an MoE block
        moe_block = hooks.get_moe_block(module, self.gptq_model.__class__)
        if moe_block is None:
            log.warn(
                f"pass_whole_dataset_to_each_expert is enabled but no MoE block found in module "
                f"{module.__class__.__name__}. MoE optimization will be disabled for this module. "
                f"This may indicate an issue with the model's MoE configuration or module structure."
            )
            return False

        return True

    def _assign_quant_device_for_module(
        self,
        named_module: NamedModule,
        fallback_device: torch.device,
    ) -> torch.device:
        """Pick and memoize the quantization device for one named module."""

        key = getattr(named_module, "full_name", None) or named_module.name
        with self._quant_device_lock:
            cached = self._module_device_map.get(key)
            if cached is not None:
                emit_device_telemetry(
                    "quant_device_cache_hit",
                    module=key,
                    target_device=cached,
                )
                return cached
            device: Optional[torch.device]
            preferred_device = normalize_device_like(named_module.state.get("preferred_quant_device"))
            if preferred_device is not None and any(dev == preferred_device for dev in self._quant_devices if dev is not None):
                device = preferred_device
                emit_device_telemetry(
                    "quant_device_preferred_hint",
                    module=key,
                    target_device=device,
                )
            elif len(self._quant_devices) <= 1:
                device = self._quant_devices[0]
            else:
                device = self._quant_devices[self._quant_device_rr % len(self._quant_devices)]
                self._quant_device_rr += 1

            if device is None:
                device = fallback_device

            self._module_device_map[key] = device
            emit_device_telemetry(
                "quant_device_assign",
                module=key,
                target_device=device,
                fallback_device=fallback_device,
            )
            return device

    def _apply_forward_device_overrides(
        self,
        subset: Dict[str, NamedModule],
        device_map: Dict[str, torch.device],
        *,
        fallback_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ) -> Dict[str, torch.device]:
        """Move selected modules to temporary forward devices and record prior placement."""

        previous: Dict[str, torch.device] = {}
        if not device_map:
            return previous

        for name, target in device_map.items():
            named_module = subset.get(name)
            module_ref = None
            if named_module is None and fallback_modules is not None:
                module_ref = fallback_modules.get(name)
                named_module = module_ref
            elif named_module is not None:
                module_ref = named_module.module if isinstance(named_module, NamedModule) else named_module
            if module_ref is None:
                continue
            try:
                current = get_device(module_ref)
            except Exception:
                current = None

            if target is None or (current is not None and current == target):
                continue

            if current is not None:
                previous[name] = current

            emit_device_telemetry(
                "forward_override_apply",
                module=getattr(named_module, "full_name", name) if named_module is not None else name,
                current_device=current,
                target_device=target,
            )
            move_to(module_ref, device=target)
            rehome_module_to_device(module_ref, target, move_parameters=True, move_buffers=True)
            if isinstance(named_module, NamedModule):
                setattr(named_module, "target_device", target)
            setattr(module_ref, "target_device", target)

        return previous

    def _restore_forward_device_overrides(
        self,
        subset: Dict[str, NamedModule],
        previous_devices: Dict[str, torch.device],
        *,
        fallback_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ) -> None:
        """Restore module placements saved by ``_apply_forward_device_overrides``."""

        if not previous_devices:
            return

        for name, revert_device in previous_devices.items():
            named_module = subset.get(name)
            module_ref = None
            if named_module is None and fallback_modules is not None:
                module_ref = fallback_modules.get(name)
                named_module = module_ref
            elif named_module is not None:
                module_ref = named_module.module if isinstance(named_module, NamedModule) else named_module
            if module_ref is None:
                continue
            emit_device_telemetry(
                "forward_override_restore",
                module=getattr(named_module, "full_name", name) if named_module is not None else name,
                target_device=revert_device,
            )
            move_to(module_ref, device=revert_device)
            rehome_module_to_device(module_ref, revert_device, move_parameters=True, move_buffers=True)
            if isinstance(named_module, NamedModule):
                setattr(named_module, "target_device", revert_device)
            setattr(module_ref, "target_device", revert_device)

    def _rehome_processor_task(
        self,
        processor: LoopProcessor,
        named_module: NamedModule,
        target_device: torch.device,
    ) -> None:
        """Move processor-owned task state alongside the module it quantizes."""

        task_map = getattr(processor, "tasks", None)
        if not task_map:
            return

        task = task_map.get(named_module.name)
        if task is None:
            return

        quant_source = named_module.state.get("quant_source_module")
        if isinstance(quant_source, torch.nn.Module) and hasattr(task, "module"):
            task.module = quant_source

        to_device_fn = getattr(task, "to_device", None)
        if callable(to_device_fn):
            to_device_fn(target_device)
            return

        module_attr = getattr(task, "module", None)
        if isinstance(module_attr, torch.nn.Module):
            move_to(module_attr, device=target_device)
            rehome_module_to_device(module_attr, target_device, move_parameters=True, move_buffers=True)
            setattr(module_attr, "target_device", target_device)

        layer_attr = getattr(task, "layer", None)
        if isinstance(layer_attr, torch.nn.Module):
            move_to(layer_attr, device=target_device)
            rehome_module_to_device(layer_attr, target_device, move_parameters=True, move_buffers=True)
            setattr(layer_attr, "target_device", target_device)

        quantizer = getattr(task, "quantizer", None)
        if quantizer is not None and hasattr(quantizer, "to"):
            try:
                quantizer.to(target_device)
            except Exception:
                pass

        tensor_attrs = ("H", "module_copy")
        for attr_name in tensor_attrs:
            tensor_value = getattr(task, attr_name, None)
            if isinstance(tensor_value, torch.Tensor):
                setattr(task, attr_name, tensor_value.to(device=target_device, non_blocking=True))

        if hasattr(task, "dev"):
            task.dev = target_device

    def _prepare_named_module_for_forward(
        self,
        named_module: NamedModule,
        fallback_device: torch.device,
    ) -> torch.nn.Module:
        """Prepare one named module for the forward role before replay starts."""

        target_device = get_device(named_module.module)
        if target_device == META:
            target_device = fallback_device

        prepared = self.gptq_model.shell_module_materialize(
            target_submodule=named_module.module,
            device=target_device,
            role="forward",
            named_module=named_module,
        )
        if prepared is not named_module.module:
            named_module.module = prepared

        setattr(named_module, "target_device", target_device)
        setattr(named_module.module, "target_device", target_device)
        return prepared

    def _prepare_named_module_for_quantization(
        self,
        processor: LoopProcessor,
        named_module: NamedModule,
        fallback_device: torch.device,
    ) -> torch.device:
        """Place a named module and its processor task on the chosen device."""

        try:
            previous_device = get_device(named_module.module)
        except Exception:
            previous_device = None

        target_device = self._assign_quant_device_for_module(
            named_module,
            fallback_device=fallback_device,
        )

        if isinstance(named_module.state.get("quant_source_module"), torch.nn.Module):
            prepared = self.gptq_model.shell_module_materialize(
                target_submodule=named_module.module,
                device=target_device,
                role="quant_source",
                named_module=named_module,
            )
            if prepared is not named_module.module:
                named_module.module = prepared
        else:
            move_to(named_module.module, device=target_device)
        rehome_module_to_device(named_module.module, target_device, move_parameters=True, move_buffers=True)

        setattr(named_module, "target_device", target_device)
        setattr(named_module.module, "target_device", target_device)
        emit_device_telemetry(
            "quant_prepare",
            module=getattr(named_module, "full_name", named_module.name),
            previous_device=previous_device,
            target_device=target_device,
        )

        self._rehome_processor_task(processor, named_module, target_device)

        return target_device

    def _run_forward_batches(
        self,
        *,
        module: torch.nn.Module,
        processor: LoopProcessor,
        current_subset: Optional[Dict[str, Any]] = None,
        ordered_module_names: Optional[List[str]] = None,
        layer_inputs: List[List[torch.Tensor]],
        layer_input_kwargs: List[Dict[str, torch.Tensor]],
        position_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        cur_layer_device: torch.device,
        is_lm_head_module: bool,
        shared_kv_cache_dict: Dict[int, torch.Tensor],
        layer_index: int,
        need_outputs: bool,
        reuse_kv: bool,
        progress_pb: ProgressBar = None,
        progress_title: Optional[str] = None,
        progress_stage: Optional[str] = None,
        progress_rows_per_batch: Optional[List[int]] = None,
        progress_total_rows: Optional[int] = None,
        force_serial: bool = False,
        preserve_module_devices: bool = False,
        apply_moe_config: bool = True,
    ) -> List[List[torch.Tensor]]:
        """Run cached batches through the module using serial or parallel execution."""

        return self._forward_executor.run(
            module=module,
            processor=processor,
            current_subset=current_subset,
            ordered_module_names=ordered_module_names,
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
            cur_layer_device=cur_layer_device,
            is_lm_head_module=is_lm_head_module,
            shared_kv_cache_dict=shared_kv_cache_dict,
            layer_index=layer_index,
            need_outputs=need_outputs,
            reuse_kv=reuse_kv,
            progress_pb=progress_pb,
            progress_title=progress_title,
            progress_stage=progress_stage,
            progress_rows_per_batch=progress_rows_per_batch,
            progress_total_rows=progress_total_rows,
            force_serial=force_serial,
            preserve_module_devices=preserve_module_devices,
            apply_moe_config=apply_moe_config,
            select_forward_devices_fn=select_forward_devices,
        )

    def _run_forward_batches_single(
        self,
        *,
        module: torch.nn.Module,
        processor: LoopProcessor,
        layer_inputs: List[List[torch.Tensor]],
        layer_input_kwargs: List[Dict[str, torch.Tensor]],
        position_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        cur_layer_device: torch.device,
        is_lm_head_module: bool,
        shared_kv_cache_dict: Dict[int, torch.Tensor],
        layer_index: int,
        need_outputs: bool,
        reuse_kv: bool,
        progress_pb: "ProgressBar" | None = None,
        progress_title: Optional[str] = None,
        progress_stage: Optional[str] = None,
        progress_rows_per_batch: Optional[List[int]] = None,
        progress_total_rows: Optional[int] = None,
        preserve_module_devices: bool = False,
        apply_moe_config: bool = True,
    ) -> List[List[torch.Tensor]]:
        """Run cached batches on a single device and return ordered outputs when requested."""

        return self._forward_executor.run_single(
            module=module,
            processor=processor,
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
            cur_layer_device=cur_layer_device,
            is_lm_head_module=is_lm_head_module,
            shared_kv_cache_dict=shared_kv_cache_dict,
            layer_index=layer_index,
            need_outputs=need_outputs,
            reuse_kv=reuse_kv,
            progress_pb=progress_pb,
            progress_title=progress_title,
            progress_stage=progress_stage,
            progress_rows_per_batch=progress_rows_per_batch,
            progress_total_rows=progress_total_rows,
            preserve_module_devices=preserve_module_devices,
            apply_moe_config=apply_moe_config,
        )

    def _run_forward_batches_parallel(
        self,
        *,
        module: torch.nn.Module,
        processor: LoopProcessor,
        layer_inputs: List[List[torch.Tensor]],
        layer_input_kwargs: List[Dict[str, torch.Tensor]],
        position_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        cur_layer_device: torch.device,
        is_lm_head_module: bool,
        shared_kv_cache_dict: Dict[int, torch.Tensor],
        layer_index: int,
        need_outputs: bool,
        reuse_kv: bool,
        devices: List[torch.device],
        progress_pb: "ProgressBar" | None = None,
        progress_title: Optional[str] = None,
        progress_stage: Optional[str] = None,
        progress_rows_per_batch: Optional[List[int]] = None,
        progress_total_rows: Optional[int] = None,
        apply_moe_config: bool = True,
    ) -> List[List[torch.Tensor]]:
        """Run cached batches across device replicas and preserve batch ordering in the result."""

        return self._forward_executor.run_parallel(
            module=module,
            processor=processor,
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
            cur_layer_device=cur_layer_device,
            is_lm_head_module=is_lm_head_module,
            shared_kv_cache_dict=shared_kv_cache_dict,
            layer_index=layer_index,
            need_outputs=need_outputs,
            reuse_kv=reuse_kv,
            devices=devices,
            progress_pb=progress_pb,
            progress_title=progress_title,
            progress_stage=progress_stage,
            progress_rows_per_batch=progress_rows_per_batch,
            progress_total_rows=progress_total_rows,
            apply_moe_config=apply_moe_config,
            clone_module_for_devices_fn=clone_module_for_devices,
            forward_batch_worker_fn=forward_batch_worker,
            device_thread_pool=DEVICE_THREAD_POOL,
        )

    def _masked_hook_wrapper(self, processor: LoopProcessor, inner_hook, hook_source: str):
        """Wrap a forward hook so it sees masked activations for the current batch."""

        def hook(module, inputs, output):
            """Apply the thread-local keep mask before delegating to ``inner_hook``."""

            # Thread-safe check if hooks are paused (TLS-based, per-thread)
            if self._get_processor_hooks_paused(processor):
                return

            keep = self._get_processor_mask(processor)

            timer = getattr(self.gptq_model, "quant_region_timer", None)
            start = time.perf_counter() if timer else None

            # Mask first tensor-like input if it's [B, S, ...]
            new_inputs = inputs
            try:
                if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                    x = inputs[0]
                    if keep is not None and x.dim() >= 3:
                        xk = apply_keep_mask_bt(x, keep)
                        if isinstance(inputs, tuple):
                            new_inputs = (xk,) + tuple(inputs[1:])
                        else:
                            new_inputs = [xk] + list(inputs[1:])
            except Exception:
                # Never break the forward due to masking; fall back to original
                new_inputs = inputs

            # Mask primary tensor output if it's [B, S, ...]
            new_output = output
            try:
                if isinstance(output, (tuple, list)) and len(output) > 0:
                    y0 = output[0]
                    if torch.is_tensor(y0) and keep is not None and y0.dim() >= 3:
                        yk = apply_keep_mask_bt(y0, keep)
                        if isinstance(output, tuple):
                            new_output = (yk,) + tuple(output[1:])
                        else:
                            new_output = [yk] + list(output[1:] )
                elif torch.is_tensor(output) and keep is not None and output.dim() >= 3:
                    new_output = apply_keep_mask_bt(output, keep)
            except Exception:
                new_output = output
            try:
                return inner_hook(module, new_inputs, new_output)
            finally:
                if timer is not None and start is not None:
                    timer.record(
                        "forward_hook",
                        time.perf_counter() - start,
                        source=hook_source,
                    )
        return hook

    def _masked_pre_hook_wrapper(self, processor: LoopProcessor, inner_hook, hook_source: str):
        """
        Pre-forward hook wrapper for MoE expert modules.
        This is called BEFORE forward executes (when used with HookedLinear.forward_hook).
        Respects hooks_paused state to avoid double-counting during intermediate calculations.
        """
        def pre_hook(module, inputs, output):
            """Apply the current keep mask before invoking the wrapped pre-hook."""

            # Thread-safe check if hooks are paused (TLS-based, per-thread)
            if self._get_processor_hooks_paused(processor):
                return

            # Get mask using TLS (thread-safe)
            keep = self._get_processor_mask(processor)

            timer = getattr(self.gptq_model, "quant_region_timer", None)
            start = time.perf_counter() if timer else None

            # Mask first tensor-like input if it's [B, S, ...]
            new_inputs = inputs
            try:
                if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                    x = inputs[0]
                    if keep is not None and x.dim() >= 3:
                        xk = apply_keep_mask_bt(x, keep)
                        if isinstance(inputs, tuple):
                            new_inputs = (xk,) + tuple(inputs[1:])
                        else:
                            new_inputs = [xk] + list(inputs[1:])
            except Exception:
                # Never break the forward due to masking
                new_inputs = inputs

            # Call inner hook with inputs and output (GPTQ ignores output anyway)
            try:
                inner_hook(module, new_inputs, output)
            finally:
                if timer is not None and start is not None:
                    timer.record(
                        "forward_pre_hook",
                        time.perf_counter() - start,
                        source=hook_source,
                    )

        return pre_hook

    def cache_inputs(self, layers, calibration_data, use_cache):
        """Capture and cache per-layer calibration inputs for later replay."""

        capture_stage = StageInputsCapture(self, logger=log)
        return capture_stage.cache_inputs(
            layers=layers,
            calibration_data=calibration_data,
            use_cache=use_cache,
        )

    def loop(self, fallback=None, **kwargs):
        """Run the quantization loop under the TF32 guard."""

        with tf32_high_precision_guard():
            return self._loop_impl(fallback=fallback, **kwargs)

    @torch.inference_mode()
    def _loop_impl(self, fallback=None, **kwargs):
        """Execute the full layer-by-layer quantization workflow."""

        if fallback is None:
            fallback = getattr(self.gptq_model.quantize_config, "fallback", None)

        if self.gptq_model.quantize_config.lm_head:
            if self.gptq_model.model.config.tie_word_embeddings and hasattr(self.gptq_model.model.model, "_tied_weights_keys"):
                tied_keys = self.gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if self.gptq_model.lm_head in item:
                        raise NotImplementedError("quantization of `lm_head` layer with `tied_weights=True` model state is not supported. Please check model has `tied_weights=False`.")

            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if get_module(self.gptq_model.model, key=self.gptq_model.lm_head) is None:
                raise ValueError(f"could not find layer {self.gptq_model.lm_head} in the model, exit...")

            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                                          f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}")

            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if self.gptq_model.quantize_config.dynamic is None:
                self.gptq_model.quantize_config.dynamic = {self.gptq_model.lm_head: lm_head_quant_config}
            elif self.gptq_model.quantize_config.dynamic_get(self.gptq_model.lm_head, default=None) is None:
                self.gptq_model.quantize_config.dynamic[self.gptq_model.lm_head] = lm_head_quant_config

        forward_pass_use_cache = self.gptq_model.model.config.use_cache if hasattr(self.gptq_model.model.config, "use_cache") else False
        self.gptq_model.model.config.use_cache = False
        layers, layers_prefix = get_module_by_name_prefix(self.gptq_model.model, self.gptq_model.extract_layers_node())
        region_timer = getattr(self.gptq_model, "quant_region_timer", None)

        for p_index, processor in enumerate(self.processors):
            if not processor.verify_calibration_dataset(p_index):
                if isinstance(processor, EoraProcessor) or\
                        (isinstance(processor, GPTQProcessor) and getattr(self.gptq_model.quantize_config, "gptaq", None) is not None) or\
                        (isinstance(processor, GPTQProcessor) and getattr(self.gptq_model.quantize_config, "foem", None) is not None):
                    prev_processor = self.processors[p_index - 1]
                    processor.set_calibration_dataset(prev_processor.calibration_dataset)
                    # If calibration_dataset is None or Empty, the input_cache of the previous processor is used.
                    processor.receive_input_cache(prev_processor.inputs_cache)
                elif isinstance(processor, DequantizeProcessor):
                    # DequantizeProcessor does not perform any operations on dataset.
                    processor.set_calibration_dataset([])
                    processor.receive_input_cache(InputCache([], [], [], []))

                continue

            input_cache = self.cache_inputs(layers=layers,
                                            calibration_data=processor.calibration_dataset,
                                            use_cache=False)
            processor.receive_input_cache(input_cache)

        # release calibration_dataset
        for processor in self.processors:
            processor.release_calibration_dataset()

        if self.gptq_model.quantize_config.offload_to_disk:
            log.info("Offloading base modules to disk...")
            offload_to_disk(
                model=self.gptq_model.model,
                module=self.gptq_model.get_base_modules(model=self.gptq_model.model),
                disk_path=self.gptq_model.quantize_config.offload_to_disk_path
            )

        for processor in self.processors:
            # Pre-build ParoQuant's optional fused rotation extension before the
            # first timed layer so layer 0 does not absorb a one-time JIT cost.
            if isinstance(processor, ParoQuantProcessor):
                processor.prewarm_runtime()

        if region_timer is not None:
            region_timer.flush()

        is_awq_quantize = any(isinstance(proc, (AWQProcessor, ParoQuantProcessor)) for proc in self.processors)
        # Capture-only layer groups are driven by processor execution config,
        # not by ad-hoc processor attributes.
        requires_activation_capture = any(
            getattr(getattr(proc, "execution_config", None), "enable_activation_capture", False)
            for proc in self.processors
        )
        layer_modules = self.gptq_model.simple_layer_modules(
            model_config=self.gptq_model.model.config,
            quantize_config=self.gptq_model.quantize_config,
            is_awq_quantize=is_awq_quantize,
            include_capture_only=requires_activation_capture,
        )
        planning_layer_modules = self.gptq_model.full_layer_modules(
            model_config=self.gptq_model.model.config,
            is_awq_quantize=is_awq_quantize,
            include_capture_only=requires_activation_capture,
        )

        # true-sequential will replay the quantized activations after each subset has been quantized to be used for next subset quantization
        # this should always be true for gptq unless you want lower but misleading error_loss that is misleading and will lead to lower post-quantized model
        if not self.gptq_model.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        layer_count = len(layers)
        pb = (log.pb(layer_count + 1 if self.gptq_model.quantize_config.lm_head else layer_count)
                            .manual()
                            .set(left_steps_offset=1))

        for processor in self.processors:
            processor.layer_count = layer_count
            processor.pb = pb

        shared_kv_cache_dict = {}

        if self.gptq_model.quantize_config.lm_head:
            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if lm_head_module and isinstance(lm_head_module, torch.nn.Linear):
                hooked_lm_head = HookedLinear.from_linear(lm_head_module)
                module_path = self.gptq_model.lm_head.split('.')
                parent = self.gptq_model.model
                for part in module_path[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, module_path[-1], hooked_lm_head)

        run_layer_stage(
            self,
            layers=layers,
            layer_modules=layer_modules,
            planning_layer_modules=planning_layer_modules,
            layers_prefix=layers_prefix,
            fallback=fallback,
            shared_kv_cache_dict=shared_kv_cache_dict,
            pb=pb,
            layer_count=layer_count,
            region_timer=region_timer,
            finalize_progress_cls=FinalizeProgressInfo,
            logger=log,
        )

        # LifeCycle: All sub-modules have finalized meaning quantization work is complete
        self._check_loop_stop()
        # Ensure ANY remaining tasks the looper submitted have drained
        DEVICE_THREAD_POOL.wait()  # same as wait('all')
        # Drain any watcher threads tracking submodule finalization progress.
        self.wait_dangling_threads()
        # Ensure any background stream sync tasks complete before returning.
        from ..utils.stream import STREAM_DEVICE_POOL
        STREAM_DEVICE_POOL.wait()
        self._check_loop_stop()

        # paranoid safety check
        # torch_sync()
        # torch_sync(device=CPU)

        total_log = {}
        reversed_processors = list(reversed(self.processors))

        process_finalize_total = len(reversed_processors)

        process_finalize_pb = (
                log.pb(range(process_finalize_total))
                   .manual()
                   .set(show_left_steps=False)
                   .title("Processor finalization")
                   .subtitle("")
        )
        process_finalize_pb.draw()

        try:
            for index, reverse_p in enumerate(reversed_processors, start=1):
                # Finalize processors in reverse order
                self._check_loop_stop()
                if isinstance(reverse_p, GPTQProcessor):
                    pass
                elif isinstance(reverse_p, EoraProcessor):
                    pass
                elif isinstance(reverse_p, DequantizeProcessor):
                    pass
                else:
                    log.info(f"{reverse_p.name()} summary:\n{reverse_p.log}")

                processor_name = reverse_p.name()
                total_log[processor_name] = reverse_p.log
                if processor_name in ["gptq", "gptq v2", "awq"]:
                    self.gptq_model.quant_log = reverse_p.log

                for module_log in reverse_p.log:
                    log.info(module_log)
                reverse_p.log_plotly()

                finalize_start = time.perf_counter() if region_timer is not None else None
                try:
                    reverse_p.finalize(model=self.gptq_model, **kwargs)
                finally:
                    if region_timer is not None and finalize_start is not None:
                        region_timer.record(
                            "process_finalize",
                            time.perf_counter() - finalize_start,
                            source=processor_name,
                        )

                process_finalize_pb.title(
                    f"Processor finalization {index}/{process_finalize_total}"
                ).subtitle(reverse_p.name()).next().draw()
        finally:
            process_finalize_pb.close()

        if region_timer is not None:
            region_timer.flush()

        self.gptq_model.model.config.use_cache = forward_pass_use_cache

        return total_log

    def create_named_modules(self, module, full, is_lm_head_module, layer_index, layers_prefix, names, processor, fallback, layer_module=None) -> Dict[str, NamedModule]:
        """Build the named-module subset a processor will quantize for one layer."""

        subset = {}
        capture_only_flags: Dict[str, bool] = {}
        for n in names:
            capture_only = False
            if n.endswith(CAPTURE_ONLY_FLAG):
                capture_only = True
                n = n.split(CAPTURE_ONLY_FLAG, 1)[0]
            if n in full:
                subset[n] = full[n]
            elif capture_only:
                # Obtain the CAPTURE_ONLY_FLAG Module separately
                subset[n], _ = get_module_by_name_prefix(module, module_name=n)
            # some modules have layer_modules that are dynamic based on config
            # ref: deepseek v2/v3/r1
            elif self.gptq_model.layer_modules_strict:
                raise ValueError(f"layer module item `{n}` not found in model, please check your model config.")
            if capture_only:
                capture_only_flags[n] = True  # forward-only modules should not be finalized
        skipped_modules = []
        for name in subset:
            layer_name = self.gptq_model.lm_head if is_lm_head_module else f"{layers_prefix}.{layer_index}.{name}"

            # gptq task is created and stored inside processor
            if not isinstance(subset[name], NamedModule):
                named_module = NamedModule(subset[name], name=name, full_name=layer_name,
                                           layer_index=layer_index)
                if capture_only_flags.get(name, False):
                    named_module.state["capture_only"] = True
                if isinstance(processor, EoraProcessor):
                    named_module.state.update({
                        "wq": processor.quantized_weights[layer_name],
                    })

                subset[name] = named_module
                full[name] = named_module
                if layer_module is not None:
                    named_module.state.setdefault("layer_module", layer_module)
            elif capture_only_flags.get(name, False):
                subset[name].state["capture_only"] = True

            if isinstance(processor, GPTQProcessor):
                processor.preprocess(subset[name], fallback=fallback)
            else:
                processor.preprocess(subset[name])
            # some modules are skipped
            if processor.is_skipped(subset[name]):
                skipped_modules.append(name)

        for name in skipped_modules:
            subset.pop(name)
            task_map = getattr(processor, "tasks", None)
            if task_map is not None:
                task_map.pop(name, None)
        return subset
