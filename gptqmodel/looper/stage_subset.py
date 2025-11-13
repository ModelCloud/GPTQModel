# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Subset-level processing stage extracted from ModuleLooper."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch

from .. import DEBUG_ON, DEVICE_THREAD_POOL
from ..looper.gptq_processor import GPTQProcessor
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..quantization.config import VRAMStrategy
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.torch import torch_sync

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .module_looper import ModuleLooper


@dataclass
class SubsetForwardContext:
    subset: Dict[str, NamedModule]
    forward_device_map: Dict[str, torch.device]
    subset_forward_serial: bool
    subset_total: int
    subset_index: int


@dataclass
class SubsetStageResult:
    processed_subset: Dict[str, NamedModule]
    layer_inputs: List[List[torch.Tensor]]
    forward_context: Optional[SubsetForwardContext]


def run_subset_stage(
    looper: 'ModuleLooper',
    *,
    processor: LoopProcessor,
    module: torch.nn.Module,
    layer_inputs: List[List[torch.Tensor]],
    layer_input_kwargs: List[Dict[str, torch.Tensor]],
    position_ids: List[torch.Tensor],
    attention_masks: List[torch.Tensor],
    cur_layer_device: torch.device,
    is_lm_head_module: bool,
    layer_descriptor: str,
    layer_title: str,
    layer_index: int,
    layers_prefix: Optional[str],
    subset_names: List[str],
    subset_index: int,
    subset_total: int,
    full,
    fail_safe: bool,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    pb,
    log=None,
    region_timer=None,
    previous_processed_subset: Optional[Dict[str, NamedModule]] = None,
    subset_event_cb: Optional[Callable[..., None]] = None,
) -> SubsetStageResult:
    """Process a single subset of modules within the layer quantization loop."""
    logger = log or setup_logger()

    processor_name = processor.name() if hasattr(processor, "name") else type(processor).__name__
    processor_name_lower = processor_name.lower()
    is_awq_processor = processor_name_lower.startswith("awq")

    subset = looper.crate_named_modules(
        module=module,
        full=full,
        is_lm_head_module=is_lm_head_module,
        layer_index=layer_index,
        layers_prefix=layers_prefix,
        names=subset_names,
        processor=processor,
        fail_safe=fail_safe,
        layer_module=module,
    )

    def emit_subset_event(stage: str) -> None:
        if subset_event_cb is None:
            return
        subset_event_cb(
            stage=stage,
            layer_idx=layer_index,
            subset_index=subset_index,
            subset_total=subset_total,
            module_names=list(subset.keys()),
            processor=processor_name,
        )

    # TODO FIXME: If a full layer has no module to quantize a simple forward() is enough and output is captured 
    # to be used as next layer's input. So one pass forward (entire layer simple forward wihout need of dealing 
    # with subset loops and micro forward loops, just full layer, usally XXXDecodeLayer.forward(). 
    # So output = current_layer.forward() is enough or sometimes just calling the layer callable like layer() 
    # which same as layer.forward().
    #
    # Assume layer 2 has no modules to quantize. At beginniing loop for layer 2, we have layer_output 
    # from completed forward_replay() of layer 1. Then pass this to layer 2 (as a whole) as layer_input 
    # and store ouput, then immediately loop to layer 3 without any further subset work that is only necessary 
    # if we need to quantize part of a layer.
    #
    # if len(subset) == 0:
    #     if logger.isEnabledFor(logging.DEBUG):
    #         logger.debug(
    #             "StageSubset: layer=%s subset=%s/%s processor=%s produced empty subset (names=%s)",
    #             layer_index,
    #             subset_index + 1,
    #             subset_total,
    #             processor_name,
    #             subset_names,
    #         )
    #     return SubsetStageResult(processed_subset={}, layer_inputs=layer_inputs, forward_context=None)

    if DEBUG_ON and logger.isEnabledFor(logging.DEBUG):
        if is_awq_processor:
            logger.debug(
                "StageSubset[awq]: layer=%s subset=%s/%s modules=%s sample=%s",
                layer_index,
                subset_index + 1,
                subset_total,
                len(subset),
                list(subset.keys())[:8],
            )
        else:
            logger.debug(
                "StageSubset: layer=%s subset=%s/%s processor=%s created %s modules (sample=%s)",
                layer_index,
                subset_index + 1,
                subset_total,
                processor_name,
                len(subset),
                list(subset.keys())[:8],
            )

    moe_group_keys_all: List[str] = []
    forward_device_map: Dict[str, torch.device] = {}
    subset_forward_serial = False

    attention_subset = bool(subset) and all(
        looper._is_attention_module_name(name) for name in subset
    )

    moe_group_key_by_name: Dict[str, Optional[str]] = {
        name: looper._extract_moe_group_key(name)
        for name in subset
    }
    moe_module_names = [
        name for name, group_key in moe_group_key_by_name.items()
        if group_key is not None
    ]
    moe_modules_set = set(moe_module_names)
    is_moe_subset = len(moe_module_names) >= looper._moe_subset_threshold

    if is_moe_subset:
        expert_groups: Dict[str, List[str]] = {}
        combined_names: List[str] = list(subset.keys())
        if full is not None:
            for candidate in full.keys():
                if candidate not in subset:
                    combined_names.append(candidate)

        for sub_name in combined_names:
            # Group every expert (including ones outside the current subset) so
            # load balancing decisions can span the full MoE family.
            group_key = looper._extract_moe_group_key(sub_name)
            if group_key is None:
                continue
            expert_groups.setdefault(group_key, []).append(sub_name)

        moe_group_keys_all = list(expert_groups.keys())

        for name, named_module in subset.items():
            setattr(named_module, "moe_enabled", name in moe_modules_set)

        if looper._vram_strategy == VRAMStrategy.BALANCED:
            devices = [
                dev for dev in looper._quant_devices
                if dev is not None and getattr(dev, "type", None) != "cpu"
            ]
            if len(devices) > 1 and expert_groups:
                assignable_group_keys: List[str] = []
                for group_key, module_names in expert_groups.items():
                    suffixes = {name.rsplit(".", 1)[-1] for name in module_names}
                    if {"gate_proj", "up_proj"}.issubset(suffixes):
                        assignable_group_keys.append(group_key)

                if assignable_group_keys:
                    groups_per_device = max(
                        math.ceil(len(assignable_group_keys) / len(devices)), 1
                    )
                    for group_index, group_key in enumerate(assignable_group_keys):
                        device_idx = min(group_index // groups_per_device, len(devices) - 1)
                        target_device = devices[device_idx]
                        for module_name in expert_groups[group_key]:
                            forward_device_map[module_name] = target_device

        subset_forward_serial = looper._vram_strategy == VRAMStrategy.BALANCED
        if subset_forward_serial:
            active_group_count = len(moe_group_keys_all)
            if active_group_count == 0:
                subset_forward_serial = False
            elif attention_subset and active_group_count <= looper._moe_subset_threshold:
                subset_forward_serial = False
    else:
        for named_module in subset.values():
            setattr(named_module, "moe_enabled", False)

    handle = []

    # some processes are simple and not require forward captures
    if processor.require_fwd:
        batch_count = looper._resolve_batch_total(
            getattr(processor, "num_batches", None),
            layer_inputs,
        )
        forward_row_counts = list(looper._collect_row_counts(layer_inputs))
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
            # Register the forward hook that captures activations for quantization.
            # The final module optionally flips a flag so processors can trigger
            # once-per-subset logic after the forward pass.
            is_last = (idx == subset_size - 1)
            hook_source = getattr(m, "full_name", None)
            if hook_source is None:
                hook_source = getattr(m, "name", name)
            if hook_source is None:
                hook_source = str(name)

            if hasattr(subset[name], 'forward_hook'):
                original_hook = processor.pre_process_fwd_hook(name)
                subset[name].forward_hook = looper._masked_hook_wrapper(processor, original_hook, hook_source)
                enable_stop = processor.fwd_after_process or getattr(processor, "subset_forward_early_stop", False)
                if is_last and enable_stop:
                    subset[name].forward_hook_last = True
            else:
                original_hook = processor.pre_process_fwd_hook(name)
                handle.append(subset[name].register_forward_hook(
                    looper._masked_hook_wrapper(processor, original_hook, hook_source)
                ))

        if DEBUG_ON and logger.isEnabledFor(logging.DEBUG):
            if is_awq_processor:
                logger.debug(
                    "StageSubset[awq]: layer=%s subset=%s/%s processor=%s registering hooks for %s modules",
                    layer_index,
                    subset_index + 1,
                    subset_total,
                    len(subset),
                )
            else:
                logger.debug(
                    "StageSubset: layer=%s subset=%s/%s processor=%s registering hooks for %s modules",
                    layer_index,
                    subset_index + 1,
                    subset_total,
                    processor_name,
                    len(subset),
                )

        emit_subset_event("forward_start")

        fwd_start = time.perf_counter()
        forward_source = f"{layer_descriptor}:subset{subset_index + 1}/{subset_total}"

        need_outputs = not processor.fwd_after_process
        reuse_kv = bool(getattr(module, "reuse_kv", False))
        forward_msg = (
            "Forward: "
            f"Layer=`{layer_descriptor}`, subset={subset_index + 1}/{subset_total}, "
            f"batches={batch_count}"
        )
        forward_pb = (
            logger.pb(range(forward_total_rows))
               .manual()
               .set(show_left_steps=False)
        )
        forward_pb.title(forward_msg).subtitle(
            f"Row 0/{forward_total_rows}"
        ).draw()

        previous_forward_devices: Dict[str, torch.device] = {}
        preserve_devices = bool(forward_device_map)
        if forward_device_map:
            previous_forward_devices = looper._apply_forward_device_overrides(
                subset,
                forward_device_map,
                fallback_modules=full,
            )

        try:
            forward_outputs = looper._run_forward_batches(
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
                force_serial=subset_forward_serial,
                preserve_module_devices=preserve_devices,
            )
        finally:
            if forward_device_map:
                looper._restore_forward_device_overrides(
                    subset,
                    previous_forward_devices,
                    fallback_modules=full,
                )
            if forward_pb is not None:
                forward_pb.close()
        if need_outputs:
            processor.receive_layer_inputs(forward_outputs)
            layer_inputs = processor.inputs_cache.layer_inputs
            del forward_outputs
        emit_subset_event("forward_end")

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
            # Detach temporary hooks to avoid leaking state into future passes.
            h.remove()

        for name in subset:
            # Reset inline hook attributes on NamedModule wrappers so future passes
            # do not reuse state from this subset run.
            if hasattr(subset[name], 'forward_hook'):
                subset[name].forward_hook = None
                subset[name].forward_hook_last = False
    else:
        if DEBUG_ON:
            logger.debug(
                "StageSubset: processor=%s layer=%s subset=%s/%s skipping forward (require_fwd=False)",
                processor_name,
                layer_index,
                subset_index + 1,
                subset_total,
            )
        emit_subset_event("forward_start")
        emit_subset_event("forward_end")

    moe_skip_modules = []
    if isinstance(processor, GPTQProcessor):
        for name in subset:
            # Skip MoE experts that never fired; they likely lacked calibration
            # traffic and would produce invalid statistics.
            if processor.tasks[name].fwd_counter == 0:
                logger.error(f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it.")
                moe_skip_modules.append(name)

        if not fail_safe:
            for name in moe_skip_modules:
                subset.pop(name)
                task_map = getattr(processor, "tasks", None)
                if task_map is not None:
                    task_map.pop(name, None)

    quant_target_devices: Dict[str, torch.device] = {}
    for name, named_module in subset.items():
        # Ensure each module has a matching processor task before sending it to
        # the worker pool; otherwise freeze it on the current device.
        task_map = getattr(processor, "tasks", None)
        has_task = bool(task_map and task_map.get(name) is not None)

        if has_task:
            target_device = looper._prepare_named_module_for_quantization(
                processor=processor,
                named_module=named_module,
                fallback_device=cur_layer_device,
            )
        else:
            target_device = get_device(named_module.module)
            setattr(named_module, "target_device", target_device)
            setattr(named_module.module, "target_device", target_device)

        quant_target_devices[name] = target_device

    processed_subset: Dict[str, NamedModule] = {}
    futures = []

    emit_subset_event("quant_start")

    @torch.inference_mode()
    def _process_on_worker(
        proc: LoopProcessor,
        nm: NamedModule,
        expected_device: torch.device,
        subset_ref: Dict[str, NamedModule],
        previous_subset_ref: Optional[Dict[str, NamedModule]],
        subset_idx: int,
        subset_total_count: int,
    ):
        module_label = getattr(nm, "full_name", getattr(nm, "name", repr(nm)))
        proc_name = proc.name() if hasattr(proc, "name") else type(proc).__name__
        module_ref = nm.module if isinstance(nm, NamedModule) else nm
        module_weight = getattr(module_ref, "weight", None)
        if module_weight is not None and expected_device is not None:
            target_device = expected_device if isinstance(expected_device, torch.device) else torch.device(expected_device)
            actual_device = get_device(module_weight)
            assert actual_device == target_device, (
                f"Device mismatch for '{module_label}' process task: "
                f"module weight on {actual_device}, thread target {target_device}."
            )

        timer = getattr(looper.gptq_model, "quant_region_timer", None)
        start = time.perf_counter() if timer else None
        try:
            if DEBUG_ON and logger.isEnabledFor(logging.DEBUG):
                if is_awq_processor:
                    logger.debug(
                        "StageSubsetWorker[awq]: layer=%s subset=%s/%s module=%s previous_subset=%s",
                        getattr(nm, "layer_index", None),
                        subset_idx + 1,
                        subset_total_count,
                        module_label,
                        bool(previous_subset_ref),
                    )
                else:
                    logger.debug(
                        "StageSubsetWorker: processor=%s layer=%s subset=%s/%s module=%s running on %s (previous_subset=%s)",
                        proc_name,
                        getattr(nm, "layer_index", None),
                        subset_idx + 1,
                        subset_total_count,
                        module_label,
                        expected_device,
                        bool(previous_subset_ref),
                    )
            proc.process(
                module=nm,
                subset=subset_ref,
                previous_subset=previous_subset_ref,
                subset_index=subset_idx,
                subset_total=subset_total_count,
            )
        finally:
            if timer is not None and start is not None:
                timer.record(
                    "process_quant",
                    time.perf_counter() - start,
                    source=module_label,
                )
        return nm.name, nm

    for name, named_module in subset.items():
        # Launch processing for every module in the subset; tasks may run in
        # parallel as allowed by the device thread pool.
        tgt_dev = quant_target_devices.get(name, cur_layer_device)
        futures.append(
            DEVICE_THREAD_POOL.submit(
                tgt_dev,
                _process_on_worker,
                processor,
                named_module,
                tgt_dev,
                subset,
                previous_processed_subset,
                subset_index,
                subset_total,
            )
        )

    for fut in futures:
        # Collect results in submission order so the final subset map preserves
        # deterministic iteration for downstream consumers.
        name, named_module = fut.result()
        processed_subset[name] = named_module
    torch_sync()

    emit_subset_event("quant_complete")

    context = SubsetForwardContext(
        subset=subset,
        forward_device_map=forward_device_map,
        subset_forward_serial=subset_forward_serial,
        subset_total=subset_total,
        subset_index=subset_index,
    )

    return SubsetStageResult(
        processed_subset=processed_subset,
        layer_inputs=layer_inputs,
        forward_context=context,
    )
