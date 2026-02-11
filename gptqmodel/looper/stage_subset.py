# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Subset-level processing stage extracted from ModuleLooper."""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from .awq_processor import AWQProcessor
from .qqq_processor import QQQProcessor
from .. import DEBUG_ON, DEVICE_THREAD_POOL
from ..looper.gptq_processor import GPTQProcessor
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..quantization.config import VramStrategy, GcMode, ExpertsRoutingBypass
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.torch import torch_empty_cache, torch_sync

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


def _run_single_subset_pass(
    looper: 'ModuleLooper',
    processor: LoopProcessor,
    module: torch.nn.Module,
    subset: Dict[str, NamedModule],
    layer_inputs: List[List[torch.Tensor]],
    layer_input_kwargs: List[Dict[str, torch.Tensor]],
    position_ids: List[torch.Tensor],
    attention_masks: List[torch.Tensor],
    cur_layer_device: torch.device,
    is_lm_head_module: bool,
    layer_descriptor: str,
    layer_title: str,
    layer_index: int,
    subset_index: int,
    subset_total: int,
    full,
    failsafe,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    pb,
    logger,
    is_awq_processor: bool,
    forward_total_rows: int,
    forward_row_counts: List[int],
    batch_count: int,
    forward_device_map: Dict[str, torch.device],
    subset_forward_serial: bool,
    region_timer=None,
    previous_processed_subset: Optional[Dict[str, NamedModule]] = None,
    subset_event_cb: Optional[Callable[..., None]] = None,
    return_outputs: bool = False,
    disable_moe_hooks: bool = False,
) -> Tuple[Dict[str, NamedModule], Optional[List[List[torch.Tensor]]]]:
    """Execute forward and quantization for a specific subset/chunk."""
    
    handle = []
    subset_size = len(subset)
    
    # Determine MoE block name for hook selection
    moe_block_name = None
    if looper.gptq_model and hasattr(looper.gptq_model, 'moe_lifecycle_hooks'):
        hooks = looper.gptq_model.moe_lifecycle_hooks
        if hooks is not None:
            moe_block = hooks.get_moe_block(module, looper.gptq_model.__class__)
            if moe_block is not None:
                # Get the full name/path of the MoE block
                for mod_name, mod in module.named_modules():
                    if mod is moe_block:
                        moe_block_name = mod_name
                        break

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

        # Determine if this module is part of MoE block (needs pre-hook to avoid StopForward)
        is_moe_module = moe_block_name and name.startswith(moe_block_name + ".")

        if hasattr(subset[name], 'forward_hook'):
            original_hook = processor.pre_process_fwd_hook(name)
            # Use pre-hook for MoE modules to fire before StopForward
            if is_moe_module:
                subset[name].forward_hook = looper._masked_pre_hook_wrapper(processor, original_hook, hook_source)
            else:
                subset[name].forward_hook = looper._masked_hook_wrapper(processor, original_hook, hook_source)
            enable_stop = processor.fwd_after_process or getattr(processor, "subset_forward_early_stop", False)
            if is_last and enable_stop:
                subset[name].forward_hook_last = True
        else:
            original_hook = processor.pre_process_fwd_hook(name)
            # Use pre-hook registration for MoE modules
            if is_moe_module:
                handle.append(subset[name].register_forward_hook(
                    looper._masked_pre_hook_wrapper(processor, original_hook, hook_source)
                ))
            else:
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
                getattr(processor, "name", type(processor).__name__),
                len(subset),
            )

    if subset_event_cb:
        subset_event_cb(stage="forward_start", layer_idx=layer_index, subset_index=subset_index, subset_total=subset_total, module_names=list(subset.keys()), processor=getattr(processor, "name", type(processor).__name__))

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
        # Set the current subset for MoE lifecycle hooks
        if disable_moe_hooks:
            looper._current_subset = None
        else:
            looper._current_subset = subset
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
    
    returned_outputs = None
    if need_outputs:
        processor.receive_layer_inputs(forward_outputs)
        if return_outputs:
             returned_outputs = processor.inputs_cache.layer_inputs
        del forward_outputs
    
    if subset_event_cb:
        subset_event_cb(stage="forward_end", layer_idx=layer_index, subset_index=subset_index, subset_total=subset_total, module_names=list(subset.keys()), processor=getattr(processor, "name", type(processor).__name__))


    fwd_time = time.perf_counter() - fwd_start
    processor.set_fwd_time(fwd_time)
    if region_timer is not None:
        region_timer.record(
            "pre_quant_forward",
            fwd_time,
            source=forward_source,
        )

    for h in handle:
        # Detach temporary hooks to avoid leaking state into future passes.
        h.remove()

    for name in subset:
        # Reset inline hook attributes on NamedModule wrappers so future passes
        # do not reuse state from this subset run.
        if hasattr(subset[name], 'forward_hook'):
            subset[name].forward_hook = None
            subset[name].forward_hook_last = False

    if looper.gptq_model.quantize_config.gc_mode == GcMode.ON_STAGE_END:
        torch_sync()
        torch_empty_cache()
    moe_skip_modules = []
    failsafe_enabled = failsafe is not None
    if isinstance(processor, GPTQProcessor) or isinstance(processor, QQQProcessor) or isinstance(processor, AWQProcessor):
        for name in subset:
            # Skip MoE experts that never fired; they likely lacked calibration
            # traffic and would produce invalid statistics.
            if not processor.has_captured_input_ids(name):
                # only log for moe if `failsafe` is not enabled
                if not failsafe_enabled:
                    logger.error(
                        f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it. "
                        f"Please enable and use `failsafe` config option."
                    )
                moe_skip_modules.append(name)

        if not failsafe_enabled:
            for name in moe_skip_modules:
                skipped_module = subset.pop(name)
                task_map = getattr(processor, "tasks", None)
                if task_map is not None:
                    task_map.pop(name, None)

                # No calibration data was routed to these MoE expert modules.
                # We skip quantization them and record them in `qcfg.dynamic` as dynamically excluded modules.
                if processor.qcfg.dynamic is None:
                    processor.qcfg.dynamic = {}
                processor.qcfg.dynamic[f"-:{re.escape(skipped_module.full_name)}"] = {}

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

    if subset_event_cb:
        subset_event_cb(stage="quant_start", layer_idx=layer_index, subset_index=subset_index, subset_total=subset_total, module_names=list(subset.keys()), processor=getattr(processor, "name", type(processor).__name__))


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
        if isinstance(named_module, NamedModule) and named_module.state.get("capture_only"):
            # Capture-only modules should not be finalized or offloaded.
            continue
        processed_subset[name] = named_module
    torch_sync()

    if looper.gptq_model.quantize_config.gc_mode == GcMode.ON_STAGE_END:
        torch_empty_cache()

    if subset_event_cb:
        subset_event_cb(stage="quant_complete", layer_idx=layer_index, subset_index=subset_index, subset_total=subset_total, module_names=list(subset.keys()), processor=getattr(processor, "name", type(processor).__name__))

    return processed_subset, returned_outputs


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
    subset: Dict[str, NamedModule],
    subset_index: int,
    subset_total: int,
    full,
    failsafe: bool,
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

        if looper._vram_strategy == VramStrategy.BALANCED:
            devices = [
                dev for dev in looper._quant_devices
                if dev is not None and getattr(dev, "type", None) != "cpu"
            ]
            if len(devices) > 1 and expert_groups:
                assignable_group_keys: List[str] = []
                for group_key, module_names in expert_groups.items():
                    suffixes = {name.rsplit(".", 1)[-1] for name in module_names}
                    # TODO: Need to make this configuratble and not static string based. Some moe use wN naming.
                    if {"gate_proj", "up_proj"}.issubset(suffixes) or {"w1", "w3"}.issubset(suffixes):
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

        subset_forward_serial = looper._vram_strategy == VramStrategy.BALANCED
        if subset_forward_serial:
            active_group_count = len(moe_group_keys_all)
            if active_group_count == 0:
                subset_forward_serial = False
            elif attention_subset and active_group_count <= looper._moe_subset_threshold:
                subset_forward_serial = False
    else:
        for named_module in subset.values():
            setattr(named_module, "moe_enabled", False)

    auto_forward_data_parallel = getattr(
        looper.gptq_model.quantize_config,
        "auto_forward_data_parallel",
        True,
    )
    subset_forward_serial = subset_forward_serial or not auto_forward_data_parallel

    # Prepare Loop Parameters

    forward_total_rows = 1
    forward_row_counts = []
    batch_count = 0
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
    
    # Check for MoE batching
    # batch_size is only available when using ExpertsRoutingBypass routing strategy
    moe_routing = looper.gptq_model.quantize_config.moe
    batch_size = None
    if moe_routing is not None and isinstance(moe_routing.routing, ExpertsRoutingBypass):
        batch_size = moe_routing.routing.batch_size
    batching_enabled = is_moe_subset and batch_size is not None and batch_size > 0
    
    processed_results = {}
    
    if batching_enabled and processor.require_fwd:
        # Simply sort all module names and chunk them by batch_size
        # This processes exactly batch_size MODULES per batch, not batch_size experts
        sorted_module_names = sorted(subset.keys())

        # Chunk module names directly by batch_size
        module_chunks = [sorted_module_names[i:i + batch_size] for i in range(0, len(sorted_module_names), batch_size)]

        if DEBUG_ON and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"MoE Expert Batching Enabled: Processing {len(sorted_module_names)} modules in {len(module_chunks)} batches "
                f"(batch_size={batch_size} modules per batch)."
            )

        # Create progress bar for MOE chunks
        moe_chunk_pb = logger.pb(range(len(module_chunks))).manual()
        moe_chunk_pb.title(f"MoE Chunk")

        for chunk_idx in moe_chunk_pb:
            chunk_keys = module_chunks[chunk_idx]
            # Create subset for this chunk
            chunk_subset = {k: subset[k] for k in chunk_keys}

            moe_chunk_pb.subtitle(f"({len(chunk_subset)} modules)").draw()
            if DEBUG_ON and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Processing MoE Chunk {chunk_idx+1}/{len(module_chunks)} ({len(chunk_subset)} modules)...")

            # Run pass
            chunk_result, _ = _run_single_subset_pass(
                looper=looper,
                processor=processor,
                module=module,
                subset=chunk_subset,
                layer_inputs=layer_inputs,
                layer_input_kwargs=layer_input_kwargs,
                position_ids=position_ids,
                attention_masks=attention_masks,
                cur_layer_device=cur_layer_device,
                is_lm_head_module=is_lm_head_module,
                layer_descriptor=layer_descriptor,
                layer_title=layer_title,
                layer_index=layer_index,
                subset_index=subset_index,
                subset_total=subset_total,
                full=full,
                failsafe=failsafe,
                shared_kv_cache_dict=shared_kv_cache_dict,
                pb=pb,
                logger=logger,
                is_awq_processor=is_awq_processor,
                forward_total_rows=forward_total_rows,
                forward_row_counts=forward_row_counts,
                batch_count=batch_count,
                forward_device_map=forward_device_map,
                subset_forward_serial=subset_forward_serial,
                region_timer=region_timer,
                previous_processed_subset=previous_processed_subset,
                subset_event_cb=None,
                return_outputs=False, 
            )
            processed_results.update(chunk_result)
            
            # Force cleanup between chunks
            if looper.gptq_model.quantize_config.gc_mode == GcMode.ON_STAGE_END:
                 torch_empty_cache()

        # Close MOE chunks progress bar
        moe_chunk_pb.close()

        # If processor.fwd_after_process is False, stage_layer won't run replay.
        # But we haven't collected proper full outputs yet (we ignored them or they were partial).
        # So we MUST run a replay here to get valid layer_inputs for the next layer.
        if not processor.fwd_after_process:
             # Final Replay to collect layer outputs
             _, new_layer_inputs = _run_single_subset_pass(
                looper=looper,
                processor=processor,
                module=module,
                subset={}, # Empty subset prevents quantization/hooks
                layer_inputs=layer_inputs,
                layer_input_kwargs=layer_input_kwargs,
                position_ids=position_ids,
                attention_masks=attention_masks,
                cur_layer_device=cur_layer_device,
                is_lm_head_module=is_lm_head_module,
                layer_descriptor=layer_descriptor,
                layer_title=layer_title,
                layer_index=layer_index,
                subset_index=subset_index,
                subset_total=subset_total,
                full=full,
                failsafe=failsafe,
                shared_kv_cache_dict=shared_kv_cache_dict,
                pb=pb,
                logger=logger,
                is_awq_processor=is_awq_processor,
                forward_total_rows=forward_total_rows,
                forward_row_counts=forward_row_counts,
                batch_count=batch_count,
                forward_device_map=forward_device_map,
                subset_forward_serial=subset_forward_serial,
                region_timer=region_timer,
                previous_processed_subset=previous_processed_subset,
                subset_event_cb=None,
                return_outputs=True,
                disable_moe_hooks=True,
            )
             if new_layer_inputs is not None:
                 layer_inputs = new_layer_inputs
    
    elif processor.require_fwd:
        # Single pass
        processed_results, new_layer_inputs = _run_single_subset_pass(
            looper=looper,
            processor=processor,
            module=module,
            subset=subset,
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
            cur_layer_device=cur_layer_device,
            is_lm_head_module=is_lm_head_module,
            layer_descriptor=layer_descriptor,
            layer_title=layer_title,
            layer_index=layer_index,
            subset_index=subset_index,
            subset_total=subset_total,
            full=full,
            failsafe=failsafe,
            shared_kv_cache_dict=shared_kv_cache_dict,
            pb=pb,
            logger=logger,
            is_awq_processor=is_awq_processor,
            forward_total_rows=forward_total_rows,
            forward_row_counts=forward_row_counts,
            batch_count=batch_count,
            forward_device_map=forward_device_map,
            subset_forward_serial=subset_forward_serial,
            region_timer=region_timer,
            previous_processed_subset=previous_processed_subset,
            subset_event_cb=subset_event_cb,
            return_outputs=True,
        )
        if new_layer_inputs is not None:
             layer_inputs = new_layer_inputs
    else:
        # No forward required
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
        emit_subset_event("quant_start")
        emit_subset_event("quant_complete")

    context = SubsetForwardContext(
        subset=subset,
        forward_device_map=forward_device_map,
        subset_forward_serial=subset_forward_serial,
        subset_total=subset_total,
        subset_index=subset_index,
    )

    return SubsetStageResult(
        processed_subset=processed_results,
        layer_inputs=layer_inputs,
        forward_context=context,
    )
