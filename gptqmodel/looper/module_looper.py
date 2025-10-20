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

import threading
import time
from concurrent.futures import as_completed
from contextlib import nullcontext
from typing import Dict, List, NamedTuple, Optional, TYPE_CHECKING

import torch

from ..looper.dequantize_processor import DequantizeProcessor
from ..looper.eora_processor import EoraProcessor
from ..looper.gptq_processor import GPTQProcessor
from ..looper.input_cache import InputCache
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import SUPPORTS_MODULE_TYPES
from ..nn_modules.hooked_linear import (STOP_FORWARD_EXCEPTION, HookedLinear,
                                        StopForward, replace_module_with_hooked_legacy)
from ..utils.attn_mask import apply_keep_mask_bt, normalize_seq_mask
from ..utils.ctx import ctx
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
from ..utils.model import find_modules, get_module, get_module_by_name_prefix, move_to, nested_move_to
from ..utils.offload import offload_to_disk
from ..utils.torch import (CPU, META, timed_gc_collect, torch_sync, tf32_high_precision_guard)
from .. import DEVICE_THREAD_POOL
from .awq_processor import AWQProcessor
from .qqq_processor import QQQProcessor

log = setup_logger()

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from logbar.progress import ProgressBar


class FinalizeProgressInfo(NamedTuple):
    module_label: Optional[str]
    process_name: str
    layer_idx: Optional[int]


class StopMainLoop(Exception):
    """Signal that the module loop should abort immediately."""


class ModuleLooper():
    """Drive the per-layer quantisation workflow over one or more devices.

    The looper executes work on the shared global :class:`DeviceThreadPool`
    instance so tasks such as module reloading, forward passes, and finalisation
    reuse the same worker threads.
    """
    def __init__(self, model: BaseQModel, processors: List[LoopProcessor]):
        self.processors = processors
        self.gptq_model = model
        self.support_batch_quantize = model.support_batch_quantize
        self.lock = threading.Lock()
        self._layer_callback = getattr(model, "layer_callback", None)
        self._loop_stop_event = threading.Event()
        self._loop_stop_exc: Optional[BaseException] = None
        self._loop_stop_waited = False

        disk_speed = estimate_disk_io_speed()
        disk_speed_mb = disk_speed / (1024 * 1024)
        if disk_speed < 200 * 1024 * 1024:
            log.warn(
                "Disk subsystem write throughput detected at "
                f"{disk_speed_mb:.1f} MB/s; quantization may be slowed by IO."
            )
        else:
            log.info(
                "Disk subsystem write throughput detected at "
                f"{disk_speed_mb:.1f} MB/s."
            )

        quant_device_hint = getattr(self.gptq_model.quantize_config, "device", None)
        normalized_quant_device = normalize_device_like(quant_device_hint)
        quant_devices = select_forward_devices(normalized_quant_device) if normalized_quant_device else [CPU]
        if not quant_devices:
            quant_devices = [CPU]

        self._quant_devices = quant_devices
        self._quant_device_rr = 0
        self._module_device_map: Dict[str, torch.device] = {}
        self._quant_device_lock = threading.Lock()

        for processor in self.processors:
            self._processor_mask_tls(processor)

    def register_layer_callback(self, callback) -> None:
        """Register or replace the layer-complete callback target."""
        self._layer_callback = callback

    def _resolve_layer_callback(self):
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

    def callbackup(self, layer_idx: int, submodule_finalized: bool):
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

    def _request_loop_stop(self, exc: Optional[BaseException]) -> None:
        with self.lock:
            if self._loop_stop_exc is None and exc is not None:
                self._loop_stop_exc = exc
        self._loop_stop_event.set()

    def _check_loop_stop(self) -> bool:
        if not self._loop_stop_event.is_set():
            return False
        if not self._loop_stop_waited:
            DEVICE_THREAD_POOL.wait()
            self._loop_stop_waited = True
        if self._loop_stop_exc is not None:
            raise self._loop_stop_exc
        return True

    def _emit_layer_complete(
        self,
        layer_idx: int,
        submodule_finalized: bool,
        *,
        raise_in_place: bool,
    ) -> None:
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
        tls = getattr(processor, "_mask_tls", None)
        if tls is None:
            tls = threading.local()
            setattr(processor, "_mask_tls", tls)
        return tls

    def _set_processor_mask(self, processor: LoopProcessor, mask):
        tls = self._processor_mask_tls(processor)
        tls.value = mask

    def _get_processor_mask(self, processor: LoopProcessor):
        tls = getattr(processor, "_mask_tls", None)
        return getattr(tls, "value", None) if tls else None

    def _safe_len(self, sequence) -> Optional[int]:
        if sequence is None:
            return None
        try:
            return len(sequence)
        except (TypeError, AttributeError):
            return None

    def _coerce_to_int(self, value) -> Optional[int]:
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
        count = self._coerce_to_int(raw_count)
        if count is not None and count > 0:
            return count

        fallback_len = self._safe_len(fallback_sequence)
        fallback = self._coerce_to_int(fallback_len)
        if fallback is not None:
            return max(fallback, 0)

        if count is not None:
            return max(count, 0)

        return 0

    def _batch_row_count(self, batch_inputs: Optional[List[torch.Tensor]]) -> int:
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
        if not layer_inputs:
            return []

        counts: List[int] = []
        for batch_inputs in layer_inputs:
            count = self._batch_row_count(batch_inputs)
            counts.append(count if count > 0 else 0)
        return counts

    def _assign_quant_device_for_module(
        self,
        named_module: NamedModule,
        fallback_device: torch.device,
    ) -> torch.device:
        key = getattr(named_module, "full_name", None) or named_module.name
        with self._quant_device_lock:
            cached = self._module_device_map.get(key)
            if cached is not None:
                return cached

            if len(self._quant_devices) <= 1:
                device = self._quant_devices[0]
            else:
                device = self._quant_devices[self._quant_device_rr % len(self._quant_devices)]
                self._quant_device_rr += 1

            if device is None:
                device = fallback_device

            self._module_device_map[key] = device
            return device

    def _rehome_processor_task(
        self,
        processor: LoopProcessor,
        named_module: NamedModule,
        target_device: torch.device,
    ) -> None:
        task_map = getattr(processor, "tasks", None)
        if not task_map:
            return

        task = task_map.get(named_module.name)
        if task is None:
            return

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

    def _prepare_named_module_for_quantization(
        self,
        processor: LoopProcessor,
        named_module: NamedModule,
        fallback_device: torch.device,
    ) -> torch.device:
        target_device = self._assign_quant_device_for_module(named_module, fallback_device=fallback_device)

        move_to(named_module.module, device=target_device)
        rehome_module_to_device(named_module.module, target_device, move_parameters=True, move_buffers=True)

        setattr(named_module, "target_device", target_device)
        setattr(named_module.module, "target_device", target_device)

        self._rehome_processor_task(processor, named_module, target_device)

        return target_device

    def _run_forward_batches(
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
    ) -> List[List[torch.Tensor]]:
        """Dispatch the captured layer inputs through the module.

        When multiple accelerators of the same type are available we clone the
        module and execute batches in parallel, otherwise we fall back to a
        single threaded path. The helper returns the ordered outputs that feed
        the next processor stage when ``need_outputs`` is set.
        """
        devices = select_forward_devices(cur_layer_device)

        if len(devices) <= 1:
            return self._run_forward_batches_single(
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
            )

        return self._run_forward_batches_parallel(
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
    ) -> List[List[torch.Tensor]]:
        """Sequential fallback when only one forward device is in use."""
        outputs: List[List[torch.Tensor]] = []
        prev_kv = shared_kv_cache_dict.get(layer_index - 1) if reuse_kv else None
        total_batches = self._resolve_batch_total(processor.num_batches, layer_inputs)
        batch_row_counts = progress_rows_per_batch or self._collect_row_counts(layer_inputs)
        batch_row_counts = list(batch_row_counts)
        if len(batch_row_counts) > total_batches:
            batch_row_counts = batch_row_counts[:total_batches]
        elif len(batch_row_counts) < total_batches:
            batch_row_counts.extend([0] * (total_batches - len(batch_row_counts)))
        total_rows = progress_total_rows if progress_total_rows is not None else sum(batch_row_counts)
        if total_rows <= 0 and total_batches > 0:
            total_rows = total_batches
        total_rows = max(total_rows, 1)
        processed_rows = 0
        stage_label = progress_stage or "Forward"

        for batch_idx in range(total_batches):
            processor._set_current_batch_index(batch_idx)
            try:
                layer_input = [move_to(inp, device=cur_layer_device) for inp in layer_inputs[batch_idx]]

                raw_mask = attention_masks[batch_idx]
                attn_tensor = raw_mask if raw_mask is None else move_to(raw_mask, device=cur_layer_device)

                keep_mask = None
                if attn_tensor is not None:
                    seq_len = layer_input[0].shape[1] if (len(layer_input) > 0 and layer_input[0].dim() >= 2) else None
                    keep_mask = normalize_seq_mask(attn_tensor, seq_len=seq_len)
                self._set_processor_mask(processor, keep_mask)

                additional_inputs: Dict[str, torch.Tensor] = {}
                if self.support_batch_quantize and attn_tensor is not None:
                    additional_inputs["attention_mask"] = attn_tensor

                if position_ids:
                    pos = position_ids[batch_idx]
                    if pos is not None:
                        additional_inputs["position_ids"] = move_to(pos, device=cur_layer_device)

                for key, value in layer_input_kwargs[batch_idx].items():
                    additional_inputs[key] = nested_move_to(value, device=cur_layer_device)

                if reuse_kv and prev_kv is not None:
                    additional_inputs["kv_last_layer"] = nested_move_to(prev_kv, device=cur_layer_device)

                rehome_module_to_device(module, cur_layer_device, move_parameters=True, move_buffers=True)

                module_output = None
                try:
                    if is_lm_head_module:
                        module_output = module(*layer_input)
                    else:
                        module_output = module(*layer_input, **additional_inputs)
                except StopForward:
                    module_output = None
                finally:
                    self._set_processor_mask(processor, None)

                if (
                    reuse_kv
                    and module_output is not None
                    and isinstance(module_output, tuple)
                    and len(module_output) > 0
                    and shared_kv_cache_dict.get(layer_index) is None
                ):
                    shared_kv_cache_dict[layer_index] = module_output[-1]

                if need_outputs and module_output is not None:
                    primary = module_output[0] if isinstance(module_output, tuple) else module_output
                    primary = move_to(primary, device=cur_layer_device)
                    outputs.append([primary])

                rows_for_batch = batch_row_counts[batch_idx] if batch_idx < len(batch_row_counts) else 0
                if rows_for_batch <= 0:
                    rows_for_batch = self._batch_row_count(layer_inputs[batch_idx]) if layer_inputs and batch_idx < len(layer_inputs) else 1
                    rows_for_batch = max(rows_for_batch, 1)

                processed_rows = min(processed_rows + rows_for_batch, total_rows)
                if progress_pb is not None:
                    if progress_title:
                        progress_pb.title(progress_title)
                    progress_pb.current_iter_step = processed_rows
                    progress_pb.subtitle(
                        f"{stage_label} rows {processed_rows}/{total_rows}"
                    ).draw()
            finally:
                processor._set_current_batch_index(None)

        return outputs

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
    ) -> List[List[torch.Tensor]]:
        """Fan batches across device clones and preserve result ordering."""
        effective_title = progress_title or (progress_stage or "Forward")

        total_batches = self._resolve_batch_total(processor.num_batches, layer_inputs)
        batch_row_counts = progress_rows_per_batch or self._collect_row_counts(layer_inputs)
        batch_row_counts = list(batch_row_counts)
        if len(batch_row_counts) > total_batches:
            batch_row_counts = batch_row_counts[:total_batches]
        elif len(batch_row_counts) < total_batches:
            batch_row_counts.extend([0] * (total_batches - len(batch_row_counts)))
        total_rows = progress_total_rows if progress_total_rows is not None else sum(batch_row_counts)
        if total_rows <= 0 and total_batches > 0:
            total_rows = total_batches
        total_rows = max(total_rows, 1)
        stage_label = progress_stage or "Forward"

        replica_pb: "ProgressBar" | None = None
        replica_title = ""
        replica_completed = 0

        if progress_pb is not None:
            progress_pb.title(effective_title)
            if len(devices) > 1:
                replica_title = f"{stage_label}: replicate to {len(devices)} devices"
                replica_pb = (
                    log.pb(range(len(devices)))
                       .manual()
                       .set(show_left_steps=False)
                )
                replica_pb.title(replica_title).subtitle("Staging module...").draw()
            else:
                device_label = str(devices[0]) if devices else "<device>"
                progress_pb.subtitle(f"{stage_label}: staging on {device_label}").draw()

        def _replica_progress(idx: int, total: int, device: torch.device, step: str) -> None:
            nonlocal replica_completed
            device_label = str(device)
            if replica_pb is not None:
                if step == "stage":
                    replica_pb.title(replica_title).subtitle(f"Stage {device_label}").draw()
                    return
                if idx > replica_completed:
                    replica_completed = idx
                    replica_pb.title(replica_title).subtitle(
                        f"{device_label} {idx}/{total}"
                    ).next().draw()
                else:
                    replica_pb.title(replica_title).subtitle(
                        f"{device_label} {idx}/{total}"
                    ).draw()
            elif progress_pb is not None:
                stage_msg = (
                    f"{stage_label}: staging on {device_label}"
                    if step == "stage"
                    else f"{stage_label}: {step} {idx}/{total} on {device_label}"
                )
                progress_pb.title(effective_title).subtitle(stage_msg).draw()

        progress_cb = _replica_progress if progress_pb is not None else None

        # Ensure any async replication/memcpy ops are complete before threads start fanning out.
        torch_sync()

        try:
            module_replicas = clone_module_for_devices(
                module,
                devices,
                progress_callback=progress_cb,
            )
        finally:
            if replica_pb is not None:
                replica_pb.close()
            if progress_pb is not None:
                progress_pb.title(effective_title).subtitle(
                    f"{stage_label} rows 0/{total_rows}"
                ).draw()

        prev_kv = shared_kv_cache_dict.get(layer_index - 1) if reuse_kv else None

        results: Dict[int, torch.Tensor | tuple | None] = {}

        processed_rows = 0
        device_segments: Dict[torch.device, List[int]] = {}
        segment_start = 0
        num_devices = len(devices)

        for index, device in enumerate(devices):
            remaining_batches = max(total_batches - segment_start, 0)
            remaining_devices = max(num_devices - index, 1)
            segment_length = remaining_batches // remaining_devices
            remainder = remaining_batches % remaining_devices
            if remainder > 0:
                segment_length += 1

            if segment_length <= 0:
                device_segments[device] = []
                continue

            segment_end = min(segment_start + segment_length, total_batches)
            device_segments[device] = list(range(segment_start, segment_end))
            segment_start = segment_end

        max_segment_length = 0
        for indices in device_segments.values():
            if len(indices) > max_segment_length:
                max_segment_length = len(indices)

        for position in range(max_segment_length):
            futures = []
            for device in devices:
                segment_indices = device_segments.get(device, [])
                if position >= len(segment_indices):
                    continue
                batch_idx = segment_indices[position]
                replica = module_replicas[device]
                submitter = (
                    DEVICE_THREAD_POOL.submit_serial
                    if device.type in ("cuda", "xpu", "mps")
                    else DEVICE_THREAD_POOL.submit
                )

                futures.append(
                    submitter(
                        device,
                        forward_batch_worker,
                        replica,
                        processor,
                        batch_idx,
                        layer_inputs[batch_idx],
                        layer_input_kwargs[batch_idx],
                        attention_masks[batch_idx],
                        position_ids[batch_idx] if position_ids else None,
                        support_batch_quantize=self.support_batch_quantize,
                        is_lm_head_module=is_lm_head_module,
                        need_output=need_outputs,
                        reuse_kv=reuse_kv,
                        prev_kv=prev_kv,
                    )
                )

            for fut in futures:
                batch_idx, module_output, kv_next = fut.result()
                if need_outputs and module_output is not None:
                    results[batch_idx] = module_output
                if reuse_kv and kv_next is not None and shared_kv_cache_dict.get(layer_index) is None:
                    shared_kv_cache_dict[layer_index] = nested_move_to(kv_next, device=cur_layer_device)

                rows_for_batch = batch_row_counts[batch_idx] if batch_idx < len(batch_row_counts) else 0
                if rows_for_batch <= 0:
                    rows_for_batch = self._batch_row_count(layer_inputs[batch_idx]) if layer_inputs and batch_idx < len(layer_inputs) else 1
                    rows_for_batch = max(rows_for_batch, 1)

                processed_rows = min(processed_rows + rows_for_batch, total_rows)
                if progress_pb is not None:
                    if progress_title:
                        progress_pb.title(progress_title)
                    progress_pb.current_iter_step = processed_rows
                    progress_pb.subtitle(
                        f"{stage_label} rows {processed_rows}/{total_rows}"
                    ).draw()

        # ensure replicas release promptly and free GPU memory
        for dev in list(module_replicas.keys()):
            del module_replicas[dev]

        if not need_outputs:
            return []

        ordered_outputs: List[List[torch.Tensor]] = []
        for idx in range(total_batches):
            module_output = results.get(idx)
            if module_output is None:
                raise RuntimeError("Forward batch returned no output; data-parallel execution produced empty result.")
            if isinstance(module_output, tuple):
                primary = module_output[0]
            else:
                primary = module_output
            primary = move_to(primary, device=cur_layer_device)
            ordered_outputs.append([primary])

        return ordered_outputs

    def _masked_hook_wrapper(self, processor: LoopProcessor, inner_hook, hook_source: str):
        def hook(module, inputs, output):
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

    def cache_inputs(self, layers, calibration_data, use_cache):
        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []

        timer = getattr(self.gptq_model, "quant_region_timer", None)
        if layers:
            first_layer = layers[0]
            layer_label = getattr(first_layer, "full_name", None)
            if layer_label is None:
                layer_label = getattr(getattr(first_layer, "__class__", None), "__name__", None)
            if layer_label is None:
                layer_label = type(first_layer).__name__
            capture_source = f"cache_inputs:{layer_label}"
        else:
            capture_source = "cache_inputs"
        start_time = time.perf_counter() if timer else None

        try:
            calibration_batches = len(calibration_data)
        except (TypeError, AttributeError):
            calibration_batches = None

        if calibration_batches is None:
            log.info("ModuleLooper: capturing layer inputs (batch count unknown)")
        else:
            log.info(
                f"ModuleLooper: capturing layer inputs from {calibration_batches} calibration batches"
            )

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device

        # TODO HookLinear add register_forward_pre_hook()
        def store_input_hook(module, args, kwargs):
            # Positional arguments.
            layer_input = []
            if kwargs.get("hidden_states") is not None:
                layer_input.append(move_to(kwargs["hidden_states"], device=data_device))
            else:
                # If hidden_states is not in kwargs, get it from the first positional argument
                # If error occurs here, check the model's modeling code
                layer_input.append(move_to(args[0], device=data_device))

            layer_inputs.append(layer_input)

            # Keyword arguments.
            # Always capture attention_mask so downstream masking can drop padded tokens
            if kwargs.get("attention_mask") is not None:
                attention_masks.append(kwargs["attention_mask"].to(device=data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, device=data_device))
            one_kwargs = {}
            for (k, v) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=data_device)
            layer_input_kwargs.append(one_kwargs)

            raise STOP_FORWARD_EXCEPTION

        # move layer to target device
        if cur_layer_device == META:
            layers[0] = self.gptq_model.shell_module_materialize(
                target_submodule=layers[0],
                device=self.gptq_model.quantize_config.device,
            )
            cur_layer_device = self.gptq_model.quantize_config.device
        else:
            layers[0] = layers[0].to(self.gptq_model.quantize_config.device)

        ori_outside_layer_module_devices = {}
        for module_name in self.gptq_model.get_base_modules(self.gptq_model.model):
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])

            if module is None:
                continue

            m_device = get_device(module)
            ori_outside_layer_module_devices[module_name] = CPU if m_device == META else m_device
            if module is not None:
                self.gptq_model.shell_module_materialize(
                    target_submodule=module,
                    device=cur_layer_device,
                )

        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)

        # TODO FIX ME.. remove hard coded Ovis code
        is_ovis = self.gptq_model.__class__.__name__ == "OvisGPTQ"

        # LifeCycle: start pre-first layer embedding hook
        self.gptq_model.pre_quantize_generate_hook_start()

        for example in calibration_data:
            for k, v in example.items():
                if self.gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT:
                    data_device = self.gptq_model.quantize_config.device
                else:
                    data_device = self.gptq_model.quantize_config.device if k == "pixel_values" else cur_layer_device
                if isinstance(v, list):
                    for index in range(len(v)):
                        if len(v[index].shape) == 1:
                            v[index] = v[index].unsqueeze(0)
                        v[index] = move_to(v[index].to(self.gptq_model.model.visual_tokenizer.dtype) if is_ovis else v[index],
                                                  device=data_device)
                else:
                    if len(v.shape) == 1:
                        v = v.unsqueeze(0)
                    example[k] = move_to(v, device=data_device)
            try:
                if self.gptq_model.ATTENTION_MASKS_DTYPE is torch.long:
                    example["attention_mask"] = example["attention_mask"].long()

                # Ensure initial caches (like RoPE) are created on the quant device
                with ctx(
                    DEVICE_THREAD_POOL.read_lock(self.gptq_model.quantize_config.device),
                    device_ctx(self.gptq_model.quantize_config.device),
                ):
                    if self.gptq_model.INPUT_EMBEDDING_EXTRA_ARGS:
                        self.gptq_model.model.generate(**example, **self.gptq_model.INPUT_EMBEDDING_EXTRA_ARGS)
                    else:
                        self.gptq_model.model(**example, use_cache=use_cache)
            except StopForward:
                pass

        # LifeCycle: pre-first layer embedding hook
        self.gptq_model.pre_quantize_generate_hook_end()
        handle.remove()

        result = InputCache(
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
        )

        if timer is not None and start_time is not None:
            timer.record(
                "capture_inputs",
                time.perf_counter() - start_time,
                source=capture_source,
            )

        return result

    def loop(self, fail_safe: bool = False, **kwargs):
        with tf32_high_precision_guard():
            return self._loop_impl(fail_safe=fail_safe, **kwargs)

    @torch.inference_mode()
    def _loop_impl(self, fail_safe: bool = False, **kwargs):
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
                        (isinstance(processor, GPTQProcessor) and self.gptq_model.quantize_config.v2):
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

        if region_timer is not None:
            region_timer.flush()

        layer_modules = self.gptq_model.simple_layer_modules(model_config=self.gptq_model.model.config, quantize_config=self.gptq_model.quantize_config)

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

        replace_module_with_hooked_legacy(self.gptq_model.model, quant_lm_head=self.gptq_model.quantize_config.lm_head)

        if self.gptq_model.quantize_config.lm_head:
            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if lm_head_module and isinstance(lm_head_module, torch.nn.Linear):
                hooked_lm_head = HookedLinear.from_linear(lm_head_module)
                module_path = self.gptq_model.lm_head.split('.')
                parent = self.gptq_model.model
                for part in module_path[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, module_path[-1], hooked_lm_head)

        for layer_index in pb:
            if self._check_loop_stop():
                break
            is_lm_head_module = layer_index >= layer_count

            if is_lm_head_module:
                layer_title = "Quantizing lm_head"
                module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            else:
                layer_title = f"Quantizing layer {layer_index} of {layer_count - 1}"
                module = layers[layer_index]

            pb.title(layer_title).subtitle("").draw()

            if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
                continue

            module = self.gptq_model.pre_quantize(module)

            if is_lm_head_module:
                layer_descriptor = self.gptq_model.lm_head
            elif layers_prefix:
                layer_descriptor = f"{layers_prefix}.{layer_index}"
            else:
                layer_descriptor = str(layer_index)

            cur_layer_device = get_device(module)
            full = find_modules(module, name=self.gptq_model.lm_head if is_lm_head_module else "")

            for p_index, processor in enumerate(self.processors):
                processor.log_call_count = 0  # reset
                processor.collect_memory_info(layer_index)

                modules = [[self.gptq_model.lm_head]] if is_lm_head_module else layer_modules

                # for NativeProcessor we process one time forward on all grouped module subsets
                if processor.fwd_all_modules_in_single_pass:
                    # merge all subsets into one
                    modules = [sum(modules, [])]

                # AWQ does per-layer itself; skip here
                if isinstance(processor, AWQProcessor):
                    named_childs = dict()
                    for index, names in enumerate(modules):
                        named_modules = self.crate_named_modules(full=full,
                                                                 is_lm_head_module=is_lm_head_module,
                                                                 layer_index=layer_index, layers_prefix=layers_prefix,
                                                                 names=names,
                                                                 processor=processor,
                                                                 fail_safe=fail_safe)
                        named_childs.update(named_modules)

                    lock_ctx = nullcontext()
                    device_for_ctx = cur_layer_device if getattr(cur_layer_device, 'type', None) != 'meta' else None
                    if device_for_ctx is not None:
                        lock_ctx = DEVICE_THREAD_POOL.read_lock(cur_layer_device)
                    with ctx(lock_ctx, device_ctx(device_for_ctx)):
                        processor.layer_quantize(module, cur_layer_device, named_childs)
                    if p_index == len(self.processors) - 1:
                        self._emit_layer_complete(
                            layer_idx=layer_index,
                            submodule_finalized=False,
                            raise_in_place=True,
                        )
                        self._emit_layer_complete(
                            layer_idx=layer_index,
                            submodule_finalized=True,
                            raise_in_place=True,
                        )
                    continue

                layer_inputs = processor.inputs_cache.layer_inputs
                if is_lm_head_module:
                    layer_inputs = self.gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                position_ids = processor.inputs_cache.position_ids
                attention_masks = processor.inputs_cache.attention_masks

                processed_subset = {}

                for index, names in enumerate(modules):
                    subset = self.crate_named_modules(full=full, is_lm_head_module=is_lm_head_module,
                                                      layer_index=layer_index, layers_prefix=layers_prefix,
                                                      names=names,
                                                      processor=processor,
                                                      fail_safe=fail_safe)

                    if len(subset) == 0:
                        continue

                    handle = []
                    subset_total = len(modules)
                    batch_count = self._resolve_batch_total(
                        getattr(processor, "num_batches", None),
                        layer_inputs,
                    )
                    forward_row_counts = list(self._collect_row_counts(layer_inputs))
                    if not forward_row_counts and batch_count > 0:
                        forward_row_counts = [1] * batch_count
                    if len(forward_row_counts) > batch_count:
                        forward_row_counts = forward_row_counts[:batch_count]
                    forward_total_rows = sum(forward_row_counts) if forward_row_counts else batch_count
                    forward_total_rows = max(forward_total_rows, 1)
                    if len(forward_row_counts) < batch_count:
                        forward_row_counts.extend([1] * (batch_count - len(forward_row_counts)))

                    subset_size = len(subset)
                    for idx, (name, m) in enumerate(subset.items()):
                        is_last = (idx == subset_size - 1)
                        hook_source = getattr(m, "full_name", None)
                        if hook_source is None:
                            hook_source = getattr(m, "name", name)
                        if hook_source is None:
                            hook_source = str(name)

                        # Wrap the processor hook with masking
                        if hasattr(subset[name], 'forward_hook'):
                            original_hook = processor.pre_process_fwd_hook(name)
                            subset[name].forward_hook = self._masked_hook_wrapper(processor, original_hook, hook_source)
                            if is_last and processor.fwd_after_process:
                                subset[name].forward_hook_last = True
                        else:
                            # Older registration path
                            original_hook = processor.pre_process_fwd_hook(name)
                            handle.append(subset[name].register_forward_hook(
                                self._masked_hook_wrapper(processor, original_hook, hook_source)
                            ))

                    # ---- Start Pre-Quantized Forward ----
                    fwd_start = time.perf_counter()
                    forward_source = f"{layer_descriptor}:subset{index + 1}/{subset_total}"

                    need_outputs = not processor.fwd_after_process
                    reuse_kv = bool(getattr(module, "reuse_kv", False))
                    forward_msg = (
                        "Forward: "
                        f"Layer=`{layer_descriptor}`, subset={index + 1}/{subset_total}, "
                        f"batches={batch_count}"
                    )
                    forward_pb = (
                        log.pb(range(forward_total_rows))
                           .manual()
                           .set(show_left_steps=False)
                    )
                    forward_pb.title(forward_msg).subtitle(
                        f"Row 0/{forward_total_rows}"
                    ).draw()
                    # Drain any background work so the forward spike does not race pooled tasks.
                    # DEVICE_THREAD_POOL.wait()
                    # try to cleanup recent objects before forward
                    #timed_gc_collect(1)

                    try:
                        forward_outputs = self._run_forward_batches(
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
                            progress_pb=forward_pb,
                            progress_title=forward_msg,
                            progress_stage="Forward",
                            progress_rows_per_batch=forward_row_counts,
                            progress_total_rows=forward_total_rows,
                        )
                    finally:
                        if forward_pb is not None:
                            forward_pb.close()
                    if need_outputs:
                        processor.receive_layer_inputs(forward_outputs)
                        layer_inputs = processor.inputs_cache.layer_inputs
                        del forward_outputs

                    fwd_time = time.perf_counter() - fwd_start
                    processor.set_fwd_time(fwd_time)
                    if region_timer is not None:
                        region_timer.record(
                            "pre_quant_forward",
                            fwd_time,
                            source=forward_source,
                        )

                    pb.title(layer_title).subtitle("").draw()

                    for h in handle:
                        h.remove()

                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = None
                            subset[name].forward_hook_last = False

                    # MoE coverage check for GPTQ
                    moe_skip_modules = []
                    if isinstance(processor, GPTQProcessor):
                        for name in subset:
                            if processor.tasks[name].fwd_counter == 0:
                                log.error(f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it.")
                                moe_skip_modules.append(name)

                        if not fail_safe:
                            for name in moe_skip_modules:
                                subset.pop(name)
                                task_map = getattr(processor, "tasks", None)
                                if task_map is not None:
                                    task_map.pop(name, None)

                    # ---- Start Process Hook (via DeviceThreadPool) ----
                    quant_target_devices: Dict[str, torch.device] = {}
                    for name, named_module in subset.items():
                        task_map = getattr(processor, "tasks", None)
                        has_task = bool(task_map and task_map.get(name) is not None)

                        if has_task:
                            target_device = self._prepare_named_module_for_quantization(
                                processor=processor,
                                named_module=named_module,
                                fallback_device=cur_layer_device,
                            )
                        else:
                            target_device = get_device(named_module.module)
                            setattr(named_module, "target_device", target_device)
                            setattr(named_module.module, "target_device", target_device)

                        quant_target_devices[name] = target_device

                    futures = []

                    @torch.inference_mode()
                    def _process_on_worker(
                        proc: LoopProcessor,
                        nm: NamedModule,
                        expected_device: torch.device,
                    ):
                        module_label = getattr(nm, "full_name", getattr(nm, "name", repr(nm)))
                        module_ref = nm.module if isinstance(nm, NamedModule) else nm
                        module_weight = getattr(module_ref, "weight", None)
                        if module_weight is not None and expected_device is not None:
                            target_device = expected_device if isinstance(expected_device, torch.device) else torch.device(expected_device)
                            actual_device = get_device(module_weight)
                            assert actual_device == target_device, (
                                f"Device mismatch for '{module_label}' process task: "
                                f"module weight on {actual_device}, thread target {target_device}."
                            )

                        # Run processor.process for this NamedModule
                        timer = getattr(self.gptq_model, "quant_region_timer", None)
                        start = time.perf_counter() if timer else None
                        try:
                            proc.process(module=nm)
                        finally:
                            if timer is not None and start is not None:
                                timer.record(
                                    "process_quant",
                                    time.perf_counter() - start,
                                    source=module_label,
                                )
                        return nm.name, nm

                    for name, m in subset.items():
                        tgt_dev = quant_target_devices.get(name, cur_layer_device)
                        futures.append(
                            DEVICE_THREAD_POOL.submit(tgt_dev, _process_on_worker, processor, m, tgt_dev)
                        )

                    for fut in futures:
                        name, m = fut.result()
                        processed_subset[name] = m
                    torch_sync()
                    # ---- End Process Hook ----

                is_last_module = layer_index == len(pb) - 1
                layer_outputs: List[List[torch.Tensor]] = []
                # second forward after process()
                if not is_last_module and processor.fwd_after_process:
                    replay_batch_count = self._resolve_batch_total(
                        getattr(processor, "num_batches", None),
                        layer_inputs,
                    )
                    replay_row_counts = list(self._collect_row_counts(layer_inputs))
                    if not replay_row_counts and replay_batch_count > 0:
                        replay_row_counts = [1] * replay_batch_count
                    if len(replay_row_counts) > replay_batch_count:
                        replay_row_counts = replay_row_counts[:replay_batch_count]
                    replay_total_rows = sum(replay_row_counts) if replay_row_counts else replay_batch_count
                    replay_total_rows = max(replay_total_rows, 1)
                    if len(replay_row_counts) < replay_batch_count:
                        replay_row_counts.extend([1] * (replay_batch_count - len(replay_row_counts)))
                    replay_msg = (
                        "Forward replay "
                        f"(layer=`{layer_descriptor}`, batches={replay_batch_count}, rows={replay_total_rows})"
                    )
                    replay_pb = (
                        log.pb(range(replay_total_rows))
                           .manual()
                           .set(show_left_steps=False)
                    )
                    replay_pb.title(replay_msg).subtitle(
                        f"Forward replay Row 0/{replay_total_rows}"
                    ).draw()
                    # Forward replay shares the same VRAM spike; block until the pool drains first.
                    # DEVICE_THREAD_POOL.wait()
                    # try to cleanup recent objects before forward
                    #timed_gc_collect(1)

                    replay_start = time.perf_counter()
                    replay_source = f"{layer_descriptor}:subset{index + 1}/{subset_total}"

                    try:
                        layer_outputs = self._run_forward_batches(
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
                            need_outputs=True,
                            reuse_kv=False,
                            progress_pb=replay_pb,
                            progress_title=replay_msg,
                            progress_stage="Forward replay",
                            progress_rows_per_batch=replay_row_counts,
                            progress_total_rows=replay_total_rows,
                        )
                    finally:
                        if replay_pb is not None:
                            replay_pb.close()
                    if region_timer is not None:
                        region_timer.record(
                            "post_quant_forward",
                            time.perf_counter() - replay_start,
                            source=replay_source,
                        )

                # Finalize module after last processor
                if p_index == len(self.processors) - 1:
                    torch_sync()

                    if not is_lm_head_module:
                        layers[layer_index] = self.gptq_model.post_quantize(module)
                    else:
                        self.gptq_model.post_quantize(module)

                    for finalized in processed_subset.values():
                        if isinstance(finalized, NamedModule):
                            setattr(finalized, "target_device", CPU)
                            inner_module = getattr(finalized, "module", None)
                        else:
                            inner_module = finalized

                        if inner_module is not None and hasattr(inner_module, "target_device"):
                            setattr(inner_module, "target_device", CPU)

                    if region_timer is not None:
                        region_timer.flush()

                if processor.fwd_after_process:
                    processor.clear_cache_data()
                    processor.receive_layer_inputs(layer_outputs)
                    layer_inputs = processor.inputs_cache.layer_inputs

                    pb.title(layer_title).subtitle("").draw()

                if p_index == len(self.processors) - 1:
                    torch_sync()

                    # Gather finalize tasks (can offload to disk); run them via the pool
                    finalize_tasks = []

                    for reverse_p in reversed(self.processors):
                        for module in processed_subset.values():
                            actual_module = module.module if isinstance(module, NamedModule) else module

                            get_device_new(
                                actual_module,
                                recursive=True,
                                assert_mode=True,
                                expected=CPU,
                            )
                            with self._quant_device_lock:
                                key = getattr(module, "full_name", getattr(module, "name", None))
                                if key is not None:
                                    self._module_device_map[key] = CPU

                            target_dev = CPU
                            module_label = getattr(module, "full_name", getattr(module, "name", ""))
                            layer_idx = getattr(module, "layer_index", None)
                            finalize_tasks.append((reverse_p, module, module_label, target_dev, layer_idx))

                    finalize_count = len(finalize_tasks)
                    finalize_futures = []
                    finalize_pb = log.pb(range(finalize_count)).manual().set(show_left_steps=False)

                    @torch.inference_mode()
                    def _finalize_on_worker(process, module, idx, total, module_label, layer_idx):
                        resolved_label = module_label or getattr(module, "full_name", getattr(module, "name", ""))
                        start = time.perf_counter() if region_timer is not None else None
                        try:
                            with log_time_block(
                                "submodule_finalize",
                                logger=log,
                                module_name=resolved_label,
                            ):
                                process.submodule_finalize(module, self.gptq_model)

                            # Disk offload (lifecycle TODO note preserved)
                            if isinstance(process, (GPTQProcessor, QQQProcessor, AWQProcessor)):
                                quant_config = getattr(self.gptq_model, "quantize_config", None)
                                if quant_config and getattr(quant_config, "offload_to_disk", False):
                                    offload_path = getattr(quant_config, "offload_to_disk_path", None)
                                    if offload_path:
                                        module_full_name = getattr(module, "full_name", None)
                                        target_module = (
                                            self.gptq_model.model.get_submodule(module_full_name)
                                            if module_full_name
                                            else module
                                        )
                                        offload_start = time.perf_counter() if region_timer is not None else None
                                        with log_time_block(
                                            "disk_offload",
                                            logger=log,
                                            module_name=resolved_label,
                                        ):
                                            offload_to_disk(
                                                model=self.gptq_model.model,
                                                module=target_module,
                                                disk_path=offload_path,
                                            )
                                        if region_timer is not None and offload_start is not None:
                                            region_timer.record(
                                                "submodule_finalize_offload",
                                                time.perf_counter() - offload_start,
                                                source=resolved_label,
                                            )
                                    else:
                                        log.warning(
                                            "Skipping disk offload for %s: no offload path configured",
                                            module_label,
                                        )
                        finally:
                            if region_timer is not None and start is not None:
                                region_timer.record(
                                    "submodule_finalize",
                                    time.perf_counter() - start,
                                    source=resolved_label,
                                )
                        process_name = process.name() if process is not None else "<processor>"
                        return FinalizeProgressInfo(module_label, process_name, layer_idx)

                        # pb.subtitle(
                        #     f"{process.name()}: layer:{layer_idx} Finalized {idx}/{total} {module_label}"
                        # ).draw()

                    for index, (process, module, module_label, target_dev, layer_idx) in enumerate(finalize_tasks, start=1):
                        future = DEVICE_THREAD_POOL.submit(
                            target_dev,
                            _finalize_on_worker,
                            process,
                            module,
                            index,
                            finalize_count,
                            module_label,
                            layer_idx,
                        )
                        finalize_futures.append((future, index, module_label, process, layer_idx))

                    finalize_futures_snapshot = list(finalize_futures)

                    self._emit_layer_complete(
                        layer_idx=layer_index,
                        submodule_finalized=False,
                        raise_in_place=True,
                    )

                    if finalize_futures_snapshot:
                        known_layers = sorted(
                            {
                                layer_idx
                                for _, _, _, _, layer_idx in finalize_futures_snapshot
                                if layer_idx is not None
                            }
                        )
                        includes_unknown = any(
                            layer_idx is None
                            for _, _, _, _, layer_idx in finalize_futures_snapshot
                        )

                        layer_heading = "Layer ?"
                        if known_layers:
                            sample_layers = ", ".join(str(idx) for idx in known_layers[:3])
                            if len(known_layers) > 3:
                                sample_layers += ", …"
                            suffix = ", ?" if includes_unknown else ""
                            prefix = "Layer" if len(known_layers) == 1 else "Layers"
                            layer_heading = f"{prefix} {sample_layers}{suffix}"
                        elif includes_unknown:
                            layer_heading = "Layer ?"

                        finalize_pb.title(
                            f"{layer_heading} Submodule finalize 0/{finalize_count}"
                        ).subtitle("Waiting for completions...").draw()

                    def _drain_finalize_futures(
                        futures,
                        finalize_pb_local,
                        finalize_count_local,
                        layer_idx_for_callback,
                    ):
                        completed_local = 0
                        try:
                            for future in as_completed(futures):
                                try:
                                    result = future.result()
                                except BaseException as exc:
                                    log.exception("Submodule finalize task raised an exception")
                                    self._request_loop_stop(exc)
                                    return

                                if isinstance(result, FinalizeProgressInfo):
                                    module_label = result.module_label
                                    process_name = result.process_name
                                    layer_idx = result.layer_idx
                                elif isinstance(result, tuple) and len(result) == 3:
                                    module_label, process_name, layer_idx = result
                                else:
                                    module_label = None
                                    process_name = "<processor>"
                                    layer_idx = None

                                layer_label = f"Layer {layer_idx}" if layer_idx is not None else "Layer ?"
                                display_module = module_label or "<unnamed>"
                                subtitle = f"{process_name}: {display_module}"

                                completed_local += 1
                                finalize_pb_local.next()
                                finalize_pb_local.title(
                                    f"{layer_label} Finalize {completed_local}/{finalize_count_local}"
                                ).subtitle(subtitle).draw()
                        finally:
                            finalize_pb_local.close()
                            self._emit_layer_complete(
                                layer_idx=layer_idx_for_callback,
                                submodule_finalized=True,
                                raise_in_place=False,
                            )

                    if finalize_futures_snapshot:
                        # Drain finalize futures asynchronously so the main loop can continue scheduling work.
                        threading.Thread(
                            target=_drain_finalize_futures,
                            args=(
                                [future for future, *_ in finalize_futures_snapshot],
                                finalize_pb,
                                finalize_count,
                                layer_index,
                            ),
                            name="SubmoduleFinalizeWatcher",
                            daemon=True,
                        ).start()
                    else:
                        self._emit_layer_complete(
                            layer_idx=layer_index,
                            submodule_finalized=True,
                            raise_in_place=True,
                        )

        # LifeCycle: All sub-modules have finalized meaning quantization work is complete
        self._check_loop_stop()
        # Ensure ANY remaining tasks the looper submitted have drained
        DEVICE_THREAD_POOL.wait()  # same as wait('all')
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
                if processor_name in ["gptq", "gptq v2"]:
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

    def crate_named_modules(self, full, is_lm_head_module, layer_index, layers_prefix, names, processor, fail_safe) -> Dict[str, NamedModule]:
        is_awq_quant = isinstance(processor, AWQProcessor)
        subset = {}
        for n in names:
            if n in full:
                subset[n] = full[n]
            # some modules have layer_modules that are dynamic based on config
            # ref: deepseek v2/v3/r1
            elif self.gptq_model.layer_modules_strict:
                raise ValueError(f"layer module item `{n}` not found in model, please check your model config.")
        skipped_modules = []
        for name in subset:
            layer_name = self.gptq_model.lm_head if is_lm_head_module else f"{layers_prefix}.{layer_index}.{name}"

            # gptq task is created and stored inside processor
            if not isinstance(subset[name], NamedModule):
                named_module = NamedModule(subset[name], name=name, full_name=layer_name,
                                           layer_index=layer_index)
                if isinstance(processor, EoraProcessor):
                    named_module.state.update({
                        "wq": processor.quantized_weights[layer_name],
                    })

                subset[name] = named_module
                full[name] = named_module

            if not is_awq_quant:
                if isinstance(processor, GPTQProcessor):
                    processor.preprocess(subset[name], fail_safe=fail_safe)
                else:
                    processor.preprocess(subset[name])
                # some modules are skipped
                if processor.is_skipped(subset[name]):
                    skipped_modules.append(name)

        if not is_awq_quant:
            for name in skipped_modules:
                subset.pop(name)
                task_map = getattr(processor, "tasks", None)
                if task_map is not None:
                    task_map.pop(name, None)
        return subset
