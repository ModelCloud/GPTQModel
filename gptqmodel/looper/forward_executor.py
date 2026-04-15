# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Forward execution logic for cached layer and subset batches."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

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
        if callable(should_use_lifecycle) and not should_use_lifecycle(module, processor):
            return nullcontext()

        return self.looper.MoELifecycleContext(
            self.looper,
            module,
            processor,
            self.looper._current_subset,
        )

    def run(
        self,
        *,
        module: torch.nn.Module,
        processor: "LoopProcessor",
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
        if len(devices) <= 1:
            return self.run_single(
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

        return self.run_parallel(
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
        )

    def run_single(
        self,
        *,
        module: torch.nn.Module,
        processor: "LoopProcessor",
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

        outputs: List[List[torch.Tensor]] = []
        prev_kv = shared_kv_cache_dict.get(layer_index - 1) if reuse_kv else None
        total_batches, batch_row_counts, total_rows = self._resolve_batch_progress(
            processor,
            layer_inputs,
            progress_rows_per_batch=progress_rows_per_batch,
            progress_total_rows=progress_total_rows,
        )
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

                # Capture input device before moving - used for output placement
                input_device = layer_inputs[batch_idx][0].device if layer_inputs[batch_idx] else cur_layer_device

                layer_input = [move_to(inp, device=exec_device) for inp in layer_inputs[batch_idx]]

                raw_mask = attention_masks[batch_idx]
                attn_tensor = raw_mask if raw_mask is None else move_to(raw_mask, device=exec_device)

                keep_mask = None
                if attn_tensor is not None:
                    seq_len = layer_input[0].shape[1] if (len(layer_input) > 0 and layer_input[0].dim() >= 2) else None
                    keep_mask = normalize_seq_mask(attn_tensor, seq_len=seq_len)

                self.looper._set_processor_mask(processor, keep_mask)
                additional_inputs: Dict[str, Optional[torch.Tensor]] = {}
                if self.looper.support_batch_quantize and attn_tensor is not None:
                    additional_inputs["attention_mask"] = attn_tensor
                else:
                    additional_inputs["attention_mask"] = None

                if position_ids:
                    pos = position_ids[batch_idx]
                    if pos is not None:
                        additional_inputs["position_ids"] = move_to(pos, device=exec_device)

                for key, value in layer_input_kwargs[batch_idx].items():
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

                if not preserve_module_devices:
                    rehome_module_to_device(module, cur_layer_device, move_parameters=True, move_buffers=True)

                with self._moe_forward_context(
                    module=module,
                    processor=processor,
                    apply_moe_config=apply_moe_config,
                ):
                    module_output = None
                    try:
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
                    reuse_kv
                    and module_output is not None
                    and isinstance(module_output, tuple)
                    and len(module_output) > 0
                    and shared_kv_cache_dict.get(layer_index) is None
                ):
                    shared_kv_cache_dict[layer_index] = module_output[-1]

                if need_outputs and module_output is not None:
                    primary = module_output[0] if isinstance(module_output, tuple) else module_output
                    # Move output back to the same device where input was stored
                    # This preserves calibration data placement when calibration_data_device is set
                    calib_device_cfg = self.looper.gptq_model.quantize_config.calibration_data_device
                    target_device = input_device if calib_device_cfg is not None else cur_layer_device
                    primary = move_to(primary, device=target_device)
                    outputs.append([primary])

                if module_output is not None:
                    del module_output

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
                processor._set_current_batch_index(None)

        return outputs

    def run_parallel(
        self,
        *,
        module: torch.nn.Module,
        processor: "LoopProcessor",
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
                    apply_moe_config=apply_moe_config,
                )

                if not isinstance(ctx, nullcontext):
                    ctx.__enter__()
                    moe_contexts.append(ctx)

            prev_kv = shared_kv_cache_dict.get(layer_index - 1) if reuse_kv else None
            results: Dict[int, torch.Tensor | None] = {}
            processed_rows = 0

            if self.looper.gptq_model.quantize_config.compute_device_filter is not None:
                forward_devices = self.looper.gptq_model.quantize_config.compute_device_filter(devices)
                if len(forward_devices) < 1:
                    self.log.warn(
                        "compute_device_filter returned empty device list. "
                        "Using all devices for forward execution."
                    )
                    forward_devices = devices
            else:
                forward_devices = devices

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
                for index, device in enumerate(forward_devices):
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
                for device in forward_devices:
                    segment_indices = device_segments.get(device, [])
                    if position >= len(segment_indices):
                        continue
                    batch_idx = segment_indices[position]
                    replica = module_replicas[device]
                    submitter = (
                        device_thread_pool.submit_serial
                        if device.type in ("cuda", "xpu", "mps")
                        else device_thread_pool.submit
                    )

                    futures.append(
                        submitter(
                            device,
                            forward_batch_worker_fn,
                            replica,
                            processor,
                            batch_idx,
                            layer_inputs[batch_idx],
                            layer_input_kwargs[batch_idx],
                            attention_masks[batch_idx],
                            position_ids[batch_idx] if position_ids else None,
                            gptq_model=self.looper.gptq_model,
                            support_batch_quantize=self.looper.support_batch_quantize,
                            is_lm_head_module=is_lm_head_module,
                            need_output=need_outputs,
                            reuse_kv=reuse_kv,
                            prev_kv=prev_kv,
                        )
                    )

                for fut in futures:
                    batch_idx, module_output, kv_next = fut.result()
                    if need_outputs and module_output is not None:
                        input_device = layer_inputs[batch_idx][0].device if layer_inputs[batch_idx] else cur_layer_device
                        target_device = input_device if calib_device_cfg is not None else cur_layer_device
                        # Move each batch result to its final target device as
                        # soon as the worker finishes.
                        primary = module_output[0] if isinstance(module_output, tuple) else module_output
                        results[batch_idx] = move_to(primary, device=target_device)
                        del module_output
                    if reuse_kv and kv_next is not None and shared_kv_cache_dict.get(layer_index) is None:
                        shared_kv_cache_dict[layer_index] = nested_move_to(kv_next, device=cur_layer_device)

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

        ordered_outputs: List[List[torch.Tensor]] = []
        for idx in range(total_batches):
            primary = results.get(idx)
            if primary is None:
                raise RuntimeError("Forward batch returned no output; data-parallel execution produced empty result.")
            ordered_outputs.append([primary])

        return ordered_outputs
