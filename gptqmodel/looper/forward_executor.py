# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Forward execution logic for cached layer and subset batches."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from .. import DEVICE_THREAD_POOL
from ..nn_modules.hooked_linear import StopForward
from ..utils.attn_mask import normalize_seq_mask
from ..utils.logger import setup_logger
from ..utils.looper_helpers import (
    clone_module_for_devices,
    forward_batch_worker,
    rehome_module_to_device,
    select_forward_devices,
)
from ..utils.model import move_to, nested_move_to
from ..utils.torch import torch_sync


if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from logbar.progress import ProgressBar

    from .loop_processor import LoopProcessor
    from .module_looper import ModuleLooper


class ForwardExecutor:
    """Own the layer/subset forward execution logic for ModuleLooper."""

    def __init__(self, looper: "ModuleLooper", logger=None) -> None:
        """Bind the executor to the looper that owns device state and helpers."""

        self.looper = looper
        self.log = logger or setup_logger()

    def _output_collection(
        self,
        *,
        layer_index: int,
        expected_batches: int,
        need_outputs: bool,
        progress_stage: Optional[str],
        apply_moe_config: bool,
    ):
        if not need_outputs:
            return []
        factory = getattr(
            self.looper.gptq_model,
            "create_quantization_layer_output_writer",
            None,
        )
        if callable(factory):
            result = factory(
                layer_index=layer_index,
                expected_batches=expected_batches,
                progress_stage=progress_stage,
                apply_moe_config=apply_moe_config,
            )
            if result is not None:
                if not callable(getattr(result, "put", None)) or not callable(
                    getattr(result, "finalize", None)
                ):
                    raise TypeError(
                        "quantization layer-output writer lacks put/finalize"
                    )
                return result
        return []

    @staticmethod
    def _store_output(collection, index: int, tensor: torch.Tensor) -> None:
        value = [tensor]
        put = getattr(collection, "put", None)
        if callable(put):
            put(index, value)
        else:
            collection.append(value)

    @staticmethod
    def _finish_outputs(collection):
        finalize = getattr(collection, "finalize", None)
        return finalize() if callable(finalize) else collection

    def _filter_forward_devices(self, devices: List[torch.device]) -> List[torch.device]:
        """Apply `compute_device_filter` to the forward device pool, preserving order."""

        quant_config = getattr(self.looper.gptq_model, "quantize_config", None)
        compute_device_filter = getattr(quant_config, "compute_device_filter", None)
        if compute_device_filter is None:
            return list(devices)

        filtered = compute_device_filter(devices)
        if len(filtered) < 1:
            self.log.warn(
                "compute_device_filter returned empty device list. "
                "Using all devices for forward execution."
            )
            return list(devices)
        return list(filtered)

    def _resolve_batch_progress(
        self,
        processor: "LoopProcessor",
        layer_inputs: List[List[torch.Tensor]],
        progress_rows_per_batch: Optional[List[int]] = None,
        progress_total_rows: Optional[int] = None,
    ) -> Tuple[int, List[int], int]:
        """Normalize batch and row progress accounting for a forward pass."""

        total_batches = self.looper._resolve_batch_total(processor.num_batches, layer_inputs)
        batch_row_counts = progress_rows_per_batch or self.looper._collect_row_counts(layer_inputs)
        batch_row_counts = list(batch_row_counts)
        if len(batch_row_counts) > total_batches:
            batch_row_counts = batch_row_counts[:total_batches]
        elif len(batch_row_counts) < total_batches:
            batch_row_counts.extend([0] * (total_batches - len(batch_row_counts)))

        total_rows = progress_total_rows if progress_total_rows is not None else sum(batch_row_counts)
        if total_rows <= 0 and total_batches > 0:
            total_rows = total_batches
        total_rows = max(total_rows, 1)
        return total_batches, batch_row_counts, total_rows

    def _moe_forward_context(
        self,
        *,
        module: torch.nn.Module,
        processor: "LoopProcessor",
        current_subset: Optional[Dict[str, Any]],
        ordered_module_names: Optional[List[str]],
        apply_moe_config: bool,
    ):
        """Pick the MoE routing context for a forward pass."""

        if not apply_moe_config:
            # Replay forwards opt out of quant-time MoE overrides and bypass hooks.
            return nullcontext()
        if self.looper.moe_routing_override:
            return self.looper.MoERoutingOverrideContext(module, self.looper.moe_routing_override)
        if not getattr(self.looper, "moe_routing_bypass", False):
            return nullcontext()

        should_use_lifecycle = getattr(self.looper, "_should_use_moe_lifecycle", None)
        if callable(should_use_lifecycle) and not should_use_lifecycle(
            module,
            processor,
            current_subset=current_subset,
        ):
            return nullcontext()

        return self.looper.MoELifecycleContext(
            self.looper,
            module,
            processor,
            current_subset,
            ordered_module_names,
        )

    def run(
        self,
        *,
        module: torch.nn.Module,
        processor: "LoopProcessor",
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
        progress_pb: "ProgressBar" | None = None,
        progress_title: Optional[str] = None,
        progress_stage: Optional[str] = None,
        progress_rows_per_batch: Optional[List[int]] = None,
        progress_total_rows: Optional[int] = None,
        force_serial: bool = False,
        preserve_module_devices: bool = False,
        apply_moe_config: bool = True,
        select_forward_devices_fn: Callable[[Optional[torch.device]], List[torch.device]] = select_forward_devices,
    ) -> List[List[torch.Tensor]]:
        """Dispatch the cached batches through the most appropriate forward path."""

        if not force_serial:
            quant_config = getattr(self.looper.gptq_model, "quantize_config", None)
            if quant_config is not None and not getattr(quant_config, "auto_forward_data_parallel", True):
                force_serial = True

        if force_serial:
            return self.run_single(
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
                preserve_module_devices=preserve_module_devices,
                apply_moe_config=apply_moe_config,
            )

        devices = select_forward_devices_fn(cur_layer_device)
        # Apply `compute_device_filter` before staging replicas. Filtering only in
        # `run_parallel` leaves a full copy of the module on every excluded GPU,
        # which both defeats the caller's device restriction and can OOM the
        # device it was meant to avoid.
        devices = self._filter_forward_devices(devices)
        if len(devices) <= 1:
            return self.run_single(
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
                preserve_module_devices=preserve_module_devices,
                apply_moe_config=apply_moe_config,
            )

        return self.run_parallel(
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
            devices=devices,
            progress_pb=progress_pb,
            progress_title=progress_title,
            progress_stage=progress_stage,
            progress_rows_per_batch=progress_rows_per_batch,
            progress_total_rows=progress_total_rows,
            apply_moe_config=apply_moe_config,
        )

    def run_single(
        self,
        *,
        module: torch.nn.Module,
        processor: "LoopProcessor",
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
        progress_pb: "ProgressBar" | None = None,
        progress_title: Optional[str] = None,
        progress_stage: Optional[str] = None,
        progress_rows_per_batch: Optional[List[int]] = None,
        progress_total_rows: Optional[int] = None,
        preserve_module_devices: bool = False,
        apply_moe_config: bool = True,
    ) -> List[List[torch.Tensor]]:
        """Run the forward pass sequentially on the current device."""

        outputs = self._output_collection(
            layer_index=layer_index,
            expected_batches=self.looper._resolve_batch_total(
                processor.num_batches, layer_inputs
            ),
            need_outputs=need_outputs,
            progress_stage=progress_stage,
            apply_moe_config=apply_moe_config,
        )
        write_shared_kv_cache = bool(getattr(self.looper.gptq_model, "write_shared_kv_cache", False))
        prev_kv = shared_kv_cache_dict.get(layer_index - 1) if reuse_kv else None
        total_batches, batch_row_counts, total_rows = self._resolve_batch_progress(
            processor,
            layer_inputs,
            progress_rows_per_batch=progress_rows_per_batch,
            progress_total_rows=progress_total_rows,
        )
        processed_rows = 0
        stage_label = progress_stage or "Forward"
        committed_indices = frozenset(
            getattr(outputs, "committed_indices", frozenset())
        )
        capture_committed = getattr(
            processor, "forward_committed_batch_indices", None
        )
        if callable(capture_committed):
            committed_indices = committed_indices.union(
                capture_committed(layer_index=layer_index)
            )

        for batch_idx in range(total_batches):
            if batch_idx in committed_indices:
                rows_for_batch = batch_row_counts[batch_idx]
                processed_rows = min(
                    processed_rows + max(rows_for_batch, 1), total_rows
                )
                continue
            processor._set_current_batch_index(batch_idx)
            try:
                exec_device = cur_layer_device
                if preserve_module_devices:
                    module_target = getattr(module, "target_device", None)
                    if module_target is not None:
                        exec_device = module_target

                # Capture input device before moving - used for output placement
                cached_layer_inputs = layer_inputs[batch_idx]
                cached_layer_kwargs = layer_input_kwargs[batch_idx]
                input_device = cached_layer_inputs[0].device if cached_layer_inputs else cur_layer_device

                layer_input = [nested_move_to(inp, device=exec_device) for inp in cached_layer_inputs]

                raw_mask = attention_masks[batch_idx]
                attn_tensor = raw_mask if raw_mask is None else move_to(raw_mask, device=exec_device)

                keep_mask = None
                if attn_tensor is not None:
                    seq_len = layer_input[0].shape[1] if (len(layer_input) > 0 and layer_input[0].dim() >= 2) else None
                    keep_mask = normalize_seq_mask(attn_tensor, seq_len=seq_len)

                additional_inputs: Dict[str, Optional[torch.Tensor]] = {}
                if self.looper.support_batch_quantize and attn_tensor is not None:
                    additional_inputs["attention_mask"] = attn_tensor
                else:
                    additional_inputs["attention_mask"] = None

                if position_ids:
                    pos = position_ids[batch_idx]
                    if pos is not None:
                        additional_inputs["position_ids"] = move_to(pos, device=exec_device)

                for key, value in cached_layer_kwargs.items():
                    if key in ["past_key_values", "past_key_value"]:
                        continue
                    additional_inputs[key] = nested_move_to(value, device=exec_device)

                if reuse_kv and prev_kv is not None:
                    additional_inputs["kv_last_layer"] = nested_move_to(prev_kv, device=exec_device)

                additional_inputs["use_cache"] = False
                additional_inputs = self.looper.gptq_model.prepare_layer_replay_kwargs(
                    layer=module,
                    layer_input=layer_input,
                    additional_inputs=additional_inputs,
                    target_device=exec_device,
                )
                keep_mask = self.looper.gptq_model.prepare_layer_replay_keep_mask(
                    layer=module,
                    layer_input=layer_input,
                    additional_inputs=additional_inputs,
                    keep_mask=keep_mask,
                    target_device=exec_device,
                )
                self.looper._set_processor_mask(processor, keep_mask)

                if not preserve_module_devices:
                    rehome_module_to_device(
                        module,
                        cur_layer_device,
                        move_parameters=True,
                        move_buffers=True,
                        root_module=getattr(self.looper.gptq_model, "model", None),
                    )

                with self._moe_forward_context(
                    module=module,
                    processor=processor,
                    current_subset=current_subset,
                    ordered_module_names=ordered_module_names,
                    apply_moe_config=apply_moe_config,
                ):
                    module_output = None
                    try:
                        with torch.no_grad():
                            if is_lm_head_module:
                                module_output = module(*layer_input)
                            else:
                                module_output = module(*layer_input, **additional_inputs)
                    except StopForward:
                        module_output = None
                    finally:
                        self.looper._set_processor_mask(processor, None)

                del layer_input
                del attn_tensor
                del keep_mask
                del additional_inputs

                if (
                    (reuse_kv or write_shared_kv_cache)
                    and module_output is not None
                    and isinstance(module_output, tuple)
                    and len(module_output) > 0
                    and shared_kv_cache_dict.get(layer_index) is None
                ):
                    shared_kv_cache_dict[layer_index] = module_output[-1]

                if module_output is not None:
                    calib_device_cfg = self.looper.gptq_model.quantize_config.calibration_data_device
                    target_device = input_device if calib_device_cfg is not None else cur_layer_device
                    update_replay_kwargs = getattr(
                        self.looper.gptq_model,
                        "update_layer_replay_kwargs_from_output",
                        None,
                    )
                    if callable(update_replay_kwargs):
                        update_replay_kwargs(
                            layer=module,
                            layer_output=module_output,
                            layer_input_kwargs=cached_layer_kwargs,
                            target_device=target_device,
                        )

                if need_outputs and module_output is not None:
                    primary = module_output[0] if isinstance(module_output, tuple) else module_output
                    # Move output back to the same device where input was stored
                    # This preserves calibration data placement when calibration_data_device is set
                    calib_device_cfg = self.looper.gptq_model.quantize_config.calibration_data_device
                    target_device = input_device if calib_device_cfg is not None else cur_layer_device
                    primary = move_to(primary, device=target_device)
                    self._store_output(outputs, batch_idx, primary)

                if module_output is not None:
                    del module_output

                batch_complete = getattr(processor, "forward_batch_completed", None)
                if callable(batch_complete):
                    batch_complete(layer_index=layer_index, batch_index=batch_idx)

                rows_for_batch = batch_row_counts[batch_idx] if batch_idx < len(batch_row_counts) else 0
                if rows_for_batch <= 0:
                    rows_for_batch = self.looper._batch_row_count(cached_layer_inputs) if layer_inputs and batch_idx < len(layer_inputs) else 1
                    rows_for_batch = max(rows_for_batch, 1)

                processed_rows = min(processed_rows + rows_for_batch, total_rows)
                if progress_pb is not None:
                    if progress_title:
                        progress_pb.title(progress_title)
                    progress_pb.current_iter_step = processed_rows
                    progress_pb.subtitle(f"{stage_label} rows {processed_rows}/{total_rows}").draw()
            finally:
                processor._set_current_batch_index(None)

        return self._finish_outputs(outputs)

    def run_parallel(
        self,
        *,
        module: torch.nn.Module,
        processor: "LoopProcessor",
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
        devices: List[torch.device],
        progress_pb: "ProgressBar" | None = None,
        progress_title: Optional[str] = None,
        progress_stage: Optional[str] = None,
        progress_rows_per_batch: Optional[List[int]] = None,
        progress_total_rows: Optional[int] = None,
        apply_moe_config: bool = True,
        clone_module_for_devices_fn=clone_module_for_devices,
        forward_batch_worker_fn=forward_batch_worker,
        device_thread_pool=DEVICE_THREAD_POOL,
    ) -> List[List[torch.Tensor]]:
        """Fan batches across device replicas and preserve result ordering."""

        effective_title = progress_title or (progress_stage or "Forward")
        total_batches, batch_row_counts, total_rows = self._resolve_batch_progress(
            processor,
            layer_inputs,
            progress_rows_per_batch=progress_rows_per_batch,
            progress_total_rows=progress_total_rows,
        )
        stage_label = progress_stage or "Forward"
        output_collection = self._output_collection(
            layer_index=layer_index,
            expected_batches=total_batches,
            need_outputs=need_outputs,
            progress_stage=progress_stage,
            apply_moe_config=apply_moe_config,
        )
        committed_indices = frozenset(
            getattr(output_collection, "committed_indices", frozenset())
        )
        capture_committed = getattr(
            processor, "forward_committed_batch_indices", None
        )
        if callable(capture_committed):
            committed_indices = committed_indices.union(
                capture_committed(layer_index=layer_index)
            )

        replica_pb: "ProgressBar" | None = None
        replica_title = ""
        replica_completed = 0

        if progress_pb is not None:
            progress_pb.title(effective_title)
            if len(devices) > 1:
                replica_title = f"{stage_label}: replicate to {len(devices)} devices"
                replica_pb = self.log.pb(range(len(devices))).manual().set(show_left_steps=False)
                replica_pb.title(replica_title).subtitle("Staging module...").draw()
            else:
                device_label = str(devices[0]) if devices else "<device>"
                progress_pb.subtitle(f"{stage_label}: staging on {device_label}").draw()

        def _replica_progress(idx: int, total: int, device: torch.device, step: str) -> None:
            """Update the progress bar while replicas are materialized."""

            nonlocal replica_completed
            device_label = str(device)
            if replica_pb is not None:
                if step == "stage":
                    replica_pb.title(replica_title).subtitle(f"Stage {device_label}").draw()
                    return
                if idx > replica_completed:
                    replica_completed = idx
                    replica_pb.title(replica_title).subtitle(f"{device_label} {idx}/{total}").next().draw()
                else:
                    replica_pb.title(replica_title).subtitle(f"{device_label} {idx}/{total}").draw()
            elif progress_pb is not None:
                stage_msg = (
                    f"{stage_label}: staging on {device_label}"
                    if step == "stage"
                    else f"{stage_label}: {step} {idx}/{total} on {device_label}"
                )
                progress_pb.title(effective_title).subtitle(stage_msg).draw()

        progress_cb = _replica_progress if progress_pb is not None else None

        torch_sync()

        try:
            module_replicas = clone_module_for_devices_fn(
                module,
                devices,
                progress_callback=progress_cb,
                root_module=getattr(self.looper.gptq_model, "model", None),
            )
        finally:
            if replica_pb is not None:
                replica_pb.close()
            if progress_pb is not None:
                progress_pb.title(effective_title).subtitle(f"{stage_label} rows 0/{total_rows}").draw()

        moe_contexts = []
        try:
            for _device, replica in module_replicas.items():
                ctx = self._moe_forward_context(
                    module=replica,
                    processor=processor,
                    current_subset=current_subset,
                    ordered_module_names=ordered_module_names,
                    apply_moe_config=apply_moe_config,
                )

                if not isinstance(ctx, nullcontext):
                    ctx.__enter__()
                    moe_contexts.append(ctx)

            write_shared_kv_cache = bool(getattr(self.looper.gptq_model, "write_shared_kv_cache", False))
            prev_kv = shared_kv_cache_dict.get(layer_index - 1) if reuse_kv else None
            results: Dict[int, torch.Tensor | None] = {}
            input_devices: Dict[int, torch.device] = {}
            processed_rows = 0

            forward_devices = self._filter_forward_devices(devices)

            device_segments: Dict[torch.device, List[int]] = {}
            segment_start = 0
            num_devices = len(forward_devices)

            # Check if balanced mode is active - if so, assign batches where data already resides
            calib_device_cfg = self.looper.gptq_model.quantize_config.calibration_data_device
            is_balanced_mode = calib_device_cfg == "balanced"

            if is_balanced_mode:
                # In balanced mode, assign each batch to the device where its input resides
                for device in forward_devices:
                    device_segments[device] = []
                for batch_idx in range(total_batches):
                    if batch_idx in committed_indices:
                        continue
                    if layer_inputs[batch_idx]:
                        batch_device = layer_inputs[batch_idx][0].device
                        # Check if this device is in our forward_devices, otherwise use first one
                        if batch_device in device_segments:
                            device_segments[batch_device].append(batch_idx)
                        else:
                            # Fallback: data is on a device not in forward_devices, use round-robin
                            fallback_device = forward_devices[batch_idx % num_devices]
                            device_segments[fallback_device].append(batch_idx)
            else:
                # Default behavior: split batches contiguously across devices
                pending_indices = [
                    index
                    for index in range(total_batches)
                    if index not in committed_indices
                ]
                pending_start = 0
                for index, device in enumerate(forward_devices):
                    remaining_batches = max(
                        len(pending_indices) - pending_start, 0
                    )
                    remaining_devices = max(num_devices - index, 1)
                    segment_length = remaining_batches // remaining_devices
                    remainder = remaining_batches % remaining_devices
                    if remainder > 0:
                        segment_length += 1

                    if segment_length <= 0:
                        device_segments[device] = []
                        continue

                    segment_end = min(
                        pending_start + segment_length, len(pending_indices)
                    )
                    device_segments[device] = pending_indices[
                        pending_start:segment_end
                    ]
                    pending_start = segment_end

            max_segment_length = 0
            for indices in device_segments.values():
                if len(indices) > max_segment_length:
                    max_segment_length = len(indices)

            for position in range(max_segment_length):
                futures = []
                for device in forward_devices:
                    segment_indices = device_segments.get(device, [])
                    if position >= len(segment_indices):
                        continue
                    batch_idx = segment_indices[position]
                    replica = module_replicas[device]
                    cached_layer_inputs = layer_inputs[batch_idx]
                    cached_layer_kwargs = layer_input_kwargs[batch_idx]
                    cached_attention_mask = attention_masks[batch_idx]
                    cached_position_ids = (
                        position_ids[batch_idx] if position_ids else None
                    )
                    input_devices[batch_idx] = (
                        cached_layer_inputs[0].device
                        if cached_layer_inputs
                        else cur_layer_device
                    )
                    submitter = (
                        device_thread_pool.submit_serial
                        if device.type in ("cuda", "xpu", "npu", "mps")
                        else device_thread_pool.submit
                    )

                    futures.append(
                        submitter(
                            device,
                            forward_batch_worker_fn,
                            replica,
                            processor,
                            batch_idx,
                            cached_layer_inputs,
                            cached_layer_kwargs,
                            cached_attention_mask,
                            cached_position_ids,
                            gptq_model=self.looper.gptq_model,
                            support_batch_quantize=self.looper.support_batch_quantize,
                            is_lm_head_module=is_lm_head_module,
                            need_output=need_outputs,
                            reuse_kv=reuse_kv,
                            prev_kv=prev_kv,
                            write_shared_kv_cache=write_shared_kv_cache,
                        )
                    )

                for fut in futures:
                    batch_idx, module_output, kv_next = fut.result()
                    if need_outputs and module_output is not None:
                        input_device = input_devices.pop(batch_idx, cur_layer_device)
                        target_device = input_device if calib_device_cfg is not None else cur_layer_device
                        # Move each batch result to its final target device as
                        # soon as the worker finishes.
                        primary = module_output[0] if isinstance(module_output, tuple) else module_output
                        primary = move_to(primary, device=target_device)
                        if callable(getattr(output_collection, "put", None)):
                            self._store_output(output_collection, batch_idx, primary)
                        else:
                            results[batch_idx] = primary
                        del module_output
                    if (reuse_kv or write_shared_kv_cache) and kv_next is not None and shared_kv_cache_dict.get(layer_index) is None:
                        shared_kv_cache_dict[layer_index] = nested_move_to(kv_next, device=cur_layer_device)

                    batch_complete = getattr(
                        processor, "forward_batch_completed", None
                    )
                    if callable(batch_complete):
                        batch_complete(
                            layer_index=layer_index,
                            batch_index=batch_idx,
                        )

                    rows_for_batch = batch_row_counts[batch_idx] if batch_idx < len(batch_row_counts) else 0
                    if rows_for_batch <= 0:
                        rows_for_batch = self.looper._batch_row_count(layer_inputs[batch_idx]) if layer_inputs and batch_idx < len(layer_inputs) else 1
                        rows_for_batch = max(rows_for_batch, 1)

                    processed_rows = min(processed_rows + rows_for_batch, total_rows)
                    if progress_pb is not None:
                        if progress_title:
                            progress_pb.title(progress_title)
                        progress_pb.current_iter_step = processed_rows
                        progress_pb.subtitle(f"{stage_label} rows {processed_rows}/{total_rows}").draw()
        finally:
            for ctx in moe_contexts:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
            moe_contexts.clear()

        for dev in list(module_replicas.keys()):
            del module_replicas[dev]

        if not need_outputs:
            return []

        if callable(getattr(output_collection, "put", None)):
            return self._finish_outputs(output_collection)

        ordered_outputs: List[List[torch.Tensor]] = []
        for idx in range(total_batches):
            primary = results.get(idx)
            if primary is None:
                # `StopForward` (MoE early-stop) is a normal outcome and yields no
                # output; `run_single` already tolerates it by skipping the batch.
                # Mirror that contract instead of failing the whole data-parallel
                # forward, otherwise multi-device quantization crashes as soon as
                # an early-stopping subset is reached.
                continue
            ordered_outputs.append([primary])

        return ordered_outputs
