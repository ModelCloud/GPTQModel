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
from ..models.base import CAPTURE_ONLY_FLAG
from ..nn_modules.hooked_linear import (STOP_FORWARD_EXCEPTION, HookedLinear,
                                        StopForward, replace_module_with_hooked_legacy)
from ..quantization.config import VRAMStrategy
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
from .stage_inputs_capture import StageInputsCapture
from .stage_layer import run_layer_stage

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
        vram_strategy = getattr(self.gptq_model.quantize_config, "vram_strategy", VRAMStrategy.EXCLUSIVE)
        if isinstance(vram_strategy, str):
            try:
                vram_strategy = VRAMStrategy(vram_strategy.lower())
            except ValueError:
                vram_strategy = VRAMStrategy.EXCLUSIVE
        supported_strategies = getattr(self.gptq_model, "supported_vram_strategies", [VRAMStrategy.EXCLUSIVE, VRAMStrategy.BALANCED])
        if isinstance(supported_strategies, VRAMStrategy):
            supported_strategies = [supported_strategies]
        if vram_strategy not in supported_strategies:
            log.debug(
                "ModuleLooper: Model %s does not support VRAM strategy %s; falling back to exclusive.",
                getattr(self.gptq_model, "__class__", type(self.gptq_model)).__name__,
                vram_strategy,
            )
            vram_strategy = VRAMStrategy.EXCLUSIVE
        self._vram_strategy = vram_strategy
        self._moe_subset_threshold = 16
        self._subset_callback = getattr(self.gptq_model, "subset_callback", None)

        for processor in self.processors:
            self._processor_mask_tls(processor)

    def register_layer_callback(self, callback) -> None:
        """Register or replace the layer-complete callback target."""
        self._layer_callback = callback

    def register_subset_callback(self, callback) -> None:
        """Register or replace the subset event callback target."""
        self._subset_callback = callback

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

    def _resolve_subset_callback(self):
        for candidate in (
            getattr(self, "_subset_callback", None),
            getattr(self, "subset_callback", None),
            getattr(self.gptq_model, "subset_callback", None),
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

    def _extract_moe_group_key(self, module_name: Optional[str]) -> Optional[str]:
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

        return None

    def _is_attention_module_name(self, module_name: str) -> bool:
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
            device: Optional[torch.device]
            if len(self._quant_devices) <= 1:
                device = self._quant_devices[0]
            else:
                device = self._quant_devices[self._quant_device_rr % len(self._quant_devices)]
                self._quant_device_rr += 1

            if device is None:
                device = fallback_device

            self._module_device_map[key] = device
            return device

    def _apply_forward_device_overrides(
        self,
        subset: Dict[str, NamedModule],
        device_map: Dict[str, torch.device],
        *,
        fallback_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ) -> Dict[str, torch.device]:
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
        target_device = self._assign_quant_device_for_module(
            named_module,
            fallback_device=fallback_device,
        )

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
        progress_pb: ProgressBar = None,
        progress_title: Optional[str] = None,
        progress_stage: Optional[str] = None,
        progress_rows_per_batch: Optional[List[int]] = None,
        progress_total_rows: Optional[int] = None,
        force_serial: bool = False,
        preserve_module_devices: bool = False,
    ) -> List[List[torch.Tensor]]:
        """Dispatch the captured layer inputs through the module.

        When multiple accelerators of the same type are available we clone the
        module and execute batches in parallel, otherwise we fall back to a
        single threaded path. The helper returns the ordered outputs that feed
        the next processor stage when ``need_outputs`` is set.
        """
        if force_serial:
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
                preserve_module_devices=preserve_module_devices,
            )

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
                preserve_module_devices=preserve_module_devices,
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
        preserve_module_devices: bool = False,
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
                exec_device = cur_layer_device
                if preserve_module_devices:
                    module_target = getattr(module, "target_device", None)
                    if module_target is not None:
                        exec_device = module_target

                layer_input = [move_to(inp, device=exec_device) for inp in layer_inputs[batch_idx]]

                raw_mask = attention_masks[batch_idx]
                attn_tensor = raw_mask if raw_mask is None else move_to(raw_mask, device=exec_device)

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
                        additional_inputs["position_ids"] = move_to(pos, device=exec_device)

                for key, value in layer_input_kwargs[batch_idx].items():
                    additional_inputs[key] = nested_move_to(value, device=exec_device)

                if reuse_kv and prev_kv is not None:
                    additional_inputs["kv_last_layer"] = nested_move_to(prev_kv, device=exec_device)

                if not preserve_module_devices:
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
            # Split the outstanding batches across devices so that each accelerator
            # receives a contiguous slice.
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
            # Submit one batch per device
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
                # Preserve the original batch order
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
            # Rebuild the ordered list of batch outputs expected by the next
            # stage.
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
        capture_stage = StageInputsCapture(self, logger=log)
        return capture_stage.cache_inputs(
            layers=layers,
            calibration_data=calibration_data,
            use_cache=use_cache,
        )

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
                        (isinstance(processor, GPTQProcessor) and self.gptq_model.quantize_config.gptaq):
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

        if region_timer is not None:
            region_timer.flush()

        is_awq_quantize = any(isinstance(proc, AWQProcessor) for proc in self.processors)
        requires_activation_capture = any(
            getattr(proc, "enable_activation_capture", False) for proc in self.processors
        )
        layer_modules = self.gptq_model.simple_layer_modules(
            model_config=self.gptq_model.model.config,
            quantize_config=self.gptq_model.quantize_config,
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
            layers_prefix=layers_prefix,
            fail_safe=fail_safe,
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

    def crate_named_modules(self, module, full, is_lm_head_module, layer_index, layers_prefix, names, processor, fail_safe, layer_module=None) -> Dict[str, NamedModule]:
        subset = {}
        for n in names:
            if n in full:
                subset[n] = full[n]
            elif n.endswith(CAPTURE_ONLY_FLAG):
                # Obtain the CAPTURE_ONLY_FLAG Module separately
                n = n.split(CAPTURE_ONLY_FLAG, 1)[0]
                subset[n], _ = get_module_by_name_prefix(module, module_name=n)
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
                if layer_module is not None:
                    named_module.state.setdefault("layer_module", layer_module)

            if isinstance(processor, GPTQProcessor):
                processor.preprocess(subset[name], fail_safe=fail_safe)
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
