# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Layer-level orchestration for subset execution, replay, and finalization.

For each processor and layer, this stage:
- builds all subset plans up front
- executes subsets using those plans
- replays forward once when the processor needs post-process outputs
- finalizes processed modules after the processor pipeline completes
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from concurrent.futures import as_completed
from typing import TYPE_CHECKING, Dict, List, Optional

from defuser.modeling.replace_modules import materialize_model
from ..nn_modules.hooked_linear import replace_module_with_hooked_legacy
from ..nn_modules.converter import MODULE_CONVERTER_MAP
from ..quantization.config import GcMode
import torch

from .. import DEBUG_ON, DEVICE_THREAD_POOL
from ..looper.awq_processor import AWQProcessor
from ..looper.gptq_processor import GPTQProcessor
from ..looper.named_module import NamedModule
from ..looper.paroquant_processor import ParoQuantProcessor
from ..looper.qqq_processor import QQQProcessor
from ..utils.device import get_device, get_device_new
from ..utils.looper_helpers import normalize_device_like
from ..utils.logger import live_renderables_suppressed, log_time_block, setup_logger
from ..utils.model import find_modules, get_module
from ..utils.offload import offload_to_disk
from ..utils.torch import CPU, torch_empty_cache, torch_sync
from .stage_subset import SubsetPlan, build_layer_subset_plans, run_subset_stage

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .module_looper import ModuleLooper


def _find_last_quantized_layer_index(
    looper: "ModuleLooper",
    *,
    layer_modules: List[List[str]],
    layers_prefix: Optional[str],
    layer_count: int,
) -> Optional[int]:
    """Return the highest layer index whose tracked modules are not all dynamically skipped."""
    if looper.gptq_model.quantize_config.lm_head or not layers_prefix:
        return None

    layer_module_names = {
        name.split("#", 1)[0]
        for module_group in layer_modules
        for name in module_group
        if name
    }
    if not layer_module_names:
        return None

    last_quantized_layer_index = -1
    for candidate_layer_index in range(layer_count):
        for module_name in layer_module_names:
            module_full_name = f"{layers_prefix}.{candidate_layer_index}.{module_name}"
            # If at least one module in this layer is not dynamically excluded,
            # the layer still needs forward/quantization work.
            if looper.gptq_model.quantize_config.dynamic_get(layer_name=module_full_name) != False:
                last_quantized_layer_index = candidate_layer_index
                break

    return last_quantized_layer_index


def _should_drain_finalize_futures_synchronously(
    looper: "ModuleLooper",
    *,
    finalize_tasks,
) -> bool:
    """Decide whether one layer must finish finalization before the next begins.

    ParoQuant layer/group optimization holds substantially more live CUDA state
    than the weight-only paths. Letting its finalizers overlap the next layer
    can visibly ratchet active VRAM upward from layer N to N+1, so ParoQuant
    always drains per-layer finalizers synchronously.
    """
    if looper.gptq_model.quantize_config.wait_for_submodule_finalizers:
        return True
    return any(isinstance(process, ParoQuantProcessor) for process, *_ in finalize_tasks)


def _should_empty_cache_after_sync_finalize(
    looper: "ModuleLooper",
    *,
    finalize_tasks,
) -> bool:
    """Release CUDA cache after synchronous ParoQuant finalization when offload is active.

    Disk offload correctly moves finalized modules out of the live model path,
    but CUDA's allocator can still hold onto the just-freed pools across layer
    boundaries. That shows up as a steady nvidia-smi climb even though the
    previous layer no longer needs those weights on device. A cache release at
    the synchronous boundary keeps layer-scope memory flat without changing the
    quantization objective.
    """
    if not getattr(looper.gptq_model.quantize_config, "offload_to_disk", False):
        return False
    return any(isinstance(process, ParoQuantProcessor) for process, *_ in finalize_tasks)


def _processor_needs_pristine_group_clone(processor) -> bool:
    """Whether grouped capture needs a dedicated pristine layer clone for this processor."""
    needs_clone = getattr(processor, "needs_pristine_layer_clone", None)
    if callable(needs_clone):
        return bool(needs_clone())
    uses_grouped_optimization = getattr(processor, "uses_grouped_optimization", None)
    return callable(uses_grouped_optimization) and bool(uses_grouped_optimization())


def _collect_layer_forward_progress(
    looper: "ModuleLooper",
    *,
    processor,
    layer_inputs: List[List[torch.Tensor]],
) -> tuple[int, List[int], int]:
    """Compute replay progress metadata for a whole-layer lifecycle forward.

    Subset-driven replay normally reuses progress data that was already planned
    inside :class:`SubsetPlan`. When an entire layer is dynamically excluded,
    no subset plan exists, but the layer stage may still need one untouched
    forward pass so the next layer receives the correct activations.

    This helper mirrors the subset planner's batch/row normalization so the
    fallback layer replay uses the same progress accounting contract:
    - `batch_count`: number of cached calibration batches to replay
    - `forward_row_counts`: per-batch row counts for progress updates
    - `forward_total_rows`: normalized total rows shown by the replay progress
    """

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

    return batch_count, forward_row_counts, forward_total_rows


def _replay_layer_outputs(
    looper: "ModuleLooper",
    *,
    module: torch.nn.Module,
    processor,
    layer_inputs: List[List[torch.Tensor]],
    layer_input_kwargs: List[Dict[str, torch.Tensor]],
    position_ids: List[torch.Tensor],
    attention_masks: List[torch.Tensor],
    cur_layer_device: torch.device,
    is_lm_head_module: bool,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    layer_index: int,
    layer_descriptor: str,
    full,
    log,
    region_timer,
    replay_plan: Optional[SubsetPlan] = None,
) -> List[List[torch.Tensor]]:
    """Replay one layer forward to materialize outputs for the next layer."""

    if replay_plan is None:
        replay_batch_count, replay_row_counts, replay_total_rows = _collect_layer_forward_progress(
            looper,
            processor=processor,
            layer_inputs=layer_inputs,
        )
        replay_source = f"{layer_descriptor}:untouched"
        replay_modules = None
        replay_forward_device_map: Dict[str, torch.device] = {}
        replay_force_serial = False
        replay_preserve_module_devices = False
    else:
        replay_batch_count = replay_plan.batch_count
        replay_row_counts = replay_plan.forward_row_counts
        replay_total_rows = replay_plan.forward_total_rows
        replay_source = (
            f"{layer_descriptor}:subset"
            f"{replay_plan.subset_index + 1}/{replay_plan.subset_total}"
        )
        replay_modules = replay_plan.modules
        replay_forward_device_map = replay_plan.forward_device_map
        replay_force_serial = replay_plan.subset_forward_serial
        replay_preserve_module_devices = replay_plan.preserve_module_devices

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

    replay_prev_devices: Dict[str, torch.device] = {}
    if replay_modules is not None and replay_forward_device_map:
        replay_prev_devices = looper._apply_forward_device_overrides(
            replay_modules,
            replay_forward_device_map,
            fallback_modules=full,
        )

    replay_start = time.perf_counter()
    try:
        looper._current_subset = None
        layer_outputs = looper._run_forward_batches(
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
            force_serial=replay_force_serial,
            preserve_module_devices=replay_preserve_module_devices,
            # Replay should emit next-layer activations under the model's native router.
            # And reduce the execution time of `forward()`.
            apply_moe_config=False,
        )
    finally:
        if (
            replay_modules is not None
            and replay_forward_device_map
            and (replay_plan is None or replay_plan.restore_forward_device_overrides)
        ):
            looper._restore_forward_device_overrides(
                replay_modules,
                replay_prev_devices,
                fallback_modules=full,
            )
        replay_pb.close()

    if region_timer is not None:
        region_timer.record(
            "post_quant_forward",
            time.perf_counter() - replay_start,
            source=replay_source,
        )

    return layer_outputs


def _capture_pristine_group_context(
    looper: "ModuleLooper",
    *,
    processor,
    module: torch.nn.Module,
    pristine_module: Optional[torch.nn.Module],
    subset_plans: List[SubsetPlan],
    layer_inputs: List[List[torch.Tensor]],
    layer_input_kwargs: List[Dict[str, torch.Tensor]],
    position_ids: List[torch.Tensor],
    attention_masks: List[torch.Tensor],
    cur_layer_device: torch.device,
    is_lm_head_module: bool,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    layer_index: int,
    layer_descriptor: str,
    full,
    log,
    region_timer,
) -> None:
    """Capture clean grouped targets while the main layer cache keeps the noisy stream."""
    uses_grouped_optimization = getattr(processor, "uses_grouped_optimization", None)
    if not callable(uses_grouped_optimization) or not uses_grouped_optimization():
        return
    clean_layer_inputs = layer_inputs
    resolve_clean_inputs = getattr(processor, "clean_group_layer_inputs", None)
    if callable(resolve_clean_inputs):
        clean_layer_inputs = resolve_clean_inputs(
            layer_index=layer_index,
            layer_inputs=layer_inputs,
        )
    capture_pristine_layer_module = getattr(processor, "receive_pristine_layer_module", None)
    if subset_plans and callable(capture_pristine_layer_module):
        capture_pristine_layer_module(
            layer_index=layer_index,
            layer_module=pristine_module if pristine_module is not None else module,
        )

    pristine_replay_module = pristine_module if pristine_module is not None else module
    pristine_outputs = _replay_layer_outputs(
        looper,
        module=pristine_replay_module,
        processor=processor,
        layer_inputs=clean_layer_inputs,
        layer_input_kwargs=layer_input_kwargs,
        position_ids=position_ids,
        attention_masks=attention_masks,
        cur_layer_device=cur_layer_device,
        is_lm_head_module=is_lm_head_module,
        shared_kv_cache_dict=shared_kv_cache_dict,
        layer_index=layer_index,
        layer_descriptor=layer_descriptor,
        full=full,
        log=log,
        region_timer=region_timer,
        replay_plan=None,
    )
    receive_clean_layer_inputs = getattr(processor, "receive_clean_layer_inputs", None)
    if callable(receive_clean_layer_inputs):
        receive_clean_layer_inputs(
            layer_index=layer_index,
            layer_inputs=pristine_outputs,
        )
    if subset_plans:
        processor.receive_layer_forward_context(
            layer_index=layer_index,
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            layer_outputs=pristine_outputs,
            subset_index=None,
            subset_total=len(subset_plans),
        )


def run_layer_stage(
    looper: 'ModuleLooper',
    *,
    layers: List[torch.nn.Module],
    layer_modules: List[List[str]],
    planning_layer_modules: List[List[str]],
    layers_prefix: Optional[str],
    fallback,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    pb,
    layer_count: int,
    region_timer,
    finalize_progress_cls,
    logger=None,
) -> None:
    """Execute the main per-layer quantization loop."""
    # Trailing layers whose tracked modules are all dynamically excluded never
    # need another forward or finalize pass, so the loop can stop once the
    # final eligible layer has been processed.
    last_quantized_layer_index = _find_last_quantized_layer_index(
        looper,
        layer_modules=layer_modules,
        layers_prefix=layers_prefix,
        layer_count=layer_count,
    )

    log = logger or setup_logger()
    durable_progress_logs = live_renderables_suppressed()
    for layer_index in pb:
        # Iterate over every transformer layer (plus lm_head when enabled) as
        # progress-bar controlled units of work.
        if looper._check_loop_stop():
            break
        is_lm_head_module = layer_index >= layer_count

        if (
            not is_lm_head_module
            and last_quantized_layer_index is not None
            and layer_index > last_quantized_layer_index
        ):
            # The remaining layers are fully skipped by dynamic config, so
            # avoid entering another layer-level quantization cycle.
            log.debug(
                "StageLayer: early stop at layer=%s, last_quantized_layer=%s",
                layer_index,
                last_quantized_layer_index,
            )
            pb.close()
            break

        if is_lm_head_module:
            layer_title = "Quantizing lm_head"
            module = get_module(looper.gptq_model.model, key=looper.gptq_model.lm_head)
            pristine_group_module = None
        else:
            layer_title = f"Quantizing layer {layer_index} of {layer_count - 1}"
            module = layers[layer_index]
            pristine_group_module = None

        pb.title(layer_title).subtitle("").draw()
        if durable_progress_logs:
            log.info(
                "StageLayer: start layer=%s/%s title=`%s`",
                layer_index if not is_lm_head_module else "lm_head",
                layer_count - 1 if not is_lm_head_module else "lm_head",
                layer_title,
            )

        if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
            # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
            continue

        module = looper.gptq_model.pre_quantize(module)

        if is_lm_head_module:
            layer_descriptor = looper.gptq_model.lm_head
        else:
            model_type = looper.gptq_model.model.config.model_type
            if model_type in MODULE_CONVERTER_MAP:
                converter = MODULE_CONVERTER_MAP[model_type]
                module = converter(module, looper.gptq_model.model.config)

            needs_group_pristine = any(
                callable(getattr(processor, "uses_grouped_optimization", None)) and processor.uses_grouped_optimization()
                for processor in looper.processors
            )
            needs_pristine_group_clone = any(
                _processor_needs_pristine_group_clone(processor)
                for processor in looper.processors
            )
            if needs_group_pristine:
                pristine_group_module = copy.deepcopy(module) if needs_pristine_group_clone else None

            replace_module_with_hooked_legacy(module, quant_lm_head=looper.gptq_model.quantize_config.lm_head)

            layers[layer_index] = module

            if layers_prefix:
                layer_descriptor = f"{layers_prefix}.{layer_index}"
            else:
                layer_descriptor = str(layer_index)

        materialize_model(module)

        cur_layer_device = get_device(module)
        if getattr(cur_layer_device, "type", None) == "meta":
            # Lazy shell layers can stay meta until a later subset stage materializes them.
            cur_layer_device = normalize_device_like(looper.gptq_model.quantize_config.device) or CPU
        full = find_modules(module, name=looper.gptq_model.lm_head if is_lm_head_module else "")

        for p_index, processor in enumerate(looper.processors):
            # Each processor contributes a quantization phase; walk them in
            # order so their caches and side effects line up with the pipeline.
            processor.log_call_count = 0  # reset
            processor.collect_memory_info(layer_index)
            # Read the replay policy once per processor so the layer stage uses
            # one execution config instead of a group of unrelated flags.
            execution_config = processor.execution_config

            layer_inputs = processor.inputs_cache.layer_inputs
            if is_lm_head_module and layer_inputs:
                layer_inputs = looper.gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
            layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
            position_ids = processor.inputs_cache.position_ids
            attention_masks = processor.inputs_cache.attention_masks

            processed_subset: Dict[str, NamedModule] = {}
            last_subset_plan: Optional[SubsetPlan] = None
            previous_subset_processed: Optional[Dict[str, NamedModule]] = None

            # Freeze all subset-level execution decisions before the processor
            # starts running this layer. The rest of the layer stage can then
            # iterate plans instead of repeatedly re-deriving replay, batching,
            # and device-routing state inside the execution loop.
            subset_plans = build_layer_subset_plans(
                looper,
                processor=processor,
                module=module,
                layer_modules=layer_modules,
                planning_layer_modules=planning_layer_modules,
                layer_inputs=layer_inputs,
                full=full,
                is_lm_head_module=is_lm_head_module,
                layer_index=layer_index,
                layers_prefix=layers_prefix,
                fallback=fallback,
            )
            if durable_progress_logs:
                log.info(
                    "StageLayer: layer=%s processor=%s begin subsets=%s",
                    layer_index if not is_lm_head_module else "lm_head",
                    processor.name(),
                    len(subset_plans),
                )

            _capture_pristine_group_context(
                looper,
                processor=processor,
                module=module,
                pristine_module=pristine_group_module,
                subset_plans=subset_plans,
                layer_inputs=layer_inputs,
                layer_input_kwargs=layer_input_kwargs,
                position_ids=position_ids,
                attention_masks=attention_masks,
                cur_layer_device=cur_layer_device,
                is_lm_head_module=is_lm_head_module,
                shared_kv_cache_dict=shared_kv_cache_dict,
                layer_index=layer_index,
                layer_descriptor=layer_descriptor,
                full=full,
                log=log,
                region_timer=region_timer,
            )
            pristine_group_module = None

            is_last_module = layer_index == len(pb) - 1
            for subset_plan in subset_plans:
                # Process the layer in smaller subsets so attention groups or
                # MoE experts can be quantized independently within a layer.
                if DEBUG_ON and log.isEnabledFor(logging.DEBUG):
                    if isinstance(processor, (AWQProcessor, ParoQuantProcessor)):
                        log.debug(
                            "StageLayer[%s]: layer=%s subset=%s/%s size=%s names=%s",
                            processor.name(),
                            layer_index,
                            subset_plan.subset_index + 1,
                            subset_plan.subset_total,
                            len(subset_plan.modules),
                            list(subset_plan.modules.keys())[:5],
                        )
                    else:
                        log.debug(
                            "StageLayer: layer=%s subset=%s/%s processor=%s size=%s names=%s",
                            layer_index,
                            subset_plan.subset_index + 1,
                            subset_plan.subset_total,
                            processor.name(),
                            len(subset_plan.modules),
                            list(subset_plan.modules.keys())[:8],
                        )
                subset_result = run_subset_stage(
                    looper=looper,
                    plan=subset_plan,
                    processor=processor,
                    module=module,
                    layer_inputs=layer_inputs,
                    layer_input_kwargs=layer_input_kwargs,
                    position_ids=position_ids,
                    attention_masks=attention_masks,
                    cur_layer_device=cur_layer_device,
                    is_lm_head_module=is_lm_head_module,
                    layer_descriptor=layer_descriptor,
                    layer_title=layer_title,
                    layer_index=layer_index,
                    full=full,
                    fallback=fallback,
                    shared_kv_cache_dict=shared_kv_cache_dict,
                    pb=pb,
                    log=log,
                    region_timer=region_timer,
                    previous_processed_subset=previous_subset_processed,
                    subset_event_cb=looper._subset_event_dispatch,
                )

                layer_inputs = subset_result.layer_inputs
                processed_subset.update(subset_result.processed_subset)
                previous_subset_processed = subset_result.processed_subset
                if subset_result.plan is not None:
                    # The most recent subset plan defines the replay contract
                    # for the outputs that flow into the next layer.
                    last_subset_plan = subset_result.plan
                if durable_progress_logs:
                    log.info(
                        "StageLayer: layer=%s processor=%s subset=%s/%s complete modules=%s",
                        layer_index if not is_lm_head_module else "lm_head",
                        processor.name(),
                        subset_plan.subset_index + 1,
                        subset_plan.subset_total,
                        len(subset_plan.modules),
                    )

            layer_outputs: List[List[torch.Tensor]] = []
            replay_plan = last_subset_plan

            # When dynamic exclusions remove every tracked module from a layer,
            # no subset stage runs, so nothing materializes that layer's
            # outputs. Processors that enable post-process forward replay
            # (`fwd_replay_after_process`) still need one forward of the untouched
            # layer so the next layer receives the correct activations.
            replay_skipped_layer = (
                not is_last_module
                and not subset_plans
                and execution_config.require_fwd
                and execution_config.fwd_replay_after_process
            )

            # Some processors consume outputs only after `process()` updates the
            # current layer. In that case, replay the layer once using the
            # metadata already computed by the final subset plan.
            replay_after_process = (
                not is_last_module
                and replay_plan is not None
                and replay_plan.replay_after_process
            )

            if replay_skipped_layer or replay_after_process:
                # Pass `replay_plan` through unconditionally: the helper uses
                # subset metadata when available and falls back to generic
                # untouched-layer replay when it is `None`.
                layer_outputs = _replay_layer_outputs(
                    looper,
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
                    layer_descriptor=layer_descriptor,
                    full=full,
                    log=log,
                    region_timer=region_timer,
                    replay_plan=replay_plan,
                )

            # Finalize module after last processor
            if p_index == len(looper.processors) - 1:
                torch_sync()

                if not is_lm_head_module:
                    layers[layer_index] = looper.gptq_model.post_quantize(module)
                else:
                    looper.gptq_model.post_quantize(module)

                for finalized in processed_subset.values():
                    # Reset finalized modules to CPU to guarantee deterministic
                    # ownership before the next processor touches the layer.
                    if isinstance(finalized, NamedModule):
                        setattr(finalized, "target_device", CPU)
                        inner_module = getattr(finalized, "module", None)
                    else:
                        inner_module = finalized

                    if inner_module is not None and hasattr(inner_module, "target_device"):
                        setattr(inner_module, "target_device", CPU)

                if region_timer is not None:
                    region_timer.flush()

            if execution_config.fwd_replay_after_process:
                processor.clear_cache_data()
                processor.receive_layer_inputs(layer_outputs)
                layer_inputs = processor.inputs_cache.layer_inputs
                pb.title(layer_title).subtitle("").draw()

            if p_index == len(looper.processors) - 1:
                torch_sync()

                # Gather finalize tasks (can offload to disk); run them via the pool
                finalize_tasks = []

                for reverse_p in reversed(looper.processors):
                    # Collect finalize tasks in reverse to mirror the processor
                    # execution order and honor downstream dependencies.
                    for module in processed_subset.values():
                        actual_module = module.module if isinstance(module, NamedModule) else module

                        get_device_new(
                            actual_module,
                            recursive=True,
                            assert_mode=True,
                            expected=CPU,
                        )
                        with looper._quant_device_lock:
                            key = getattr(module, "full_name", getattr(module, "name", None))
                            if key is not None:
                                looper._module_device_map[key] = CPU

                        target_dev = CPU
                        module_label = getattr(module, "full_name", getattr(module, "name", ""))
                        layer_idx = getattr(module, "layer_index", None)
                        finalize_tasks.append((reverse_p, module, module_label, target_dev, layer_idx))

                finalize_count = len(finalize_tasks)
                finalize_futures = []
                finalize_pb = log.pb(range(finalize_count)).manual().set(show_left_steps=False)

                @torch.inference_mode()
                def _finalize_on_worker(process, module, idx, total, module_label, layer_idx):
                    """Runs processor finalization and optional disk offload for one module."""

                    resolved_label = module_label or getattr(module, "full_name", getattr(module, "name", ""))
                    start = time.perf_counter() if region_timer is not None else None
                    try:
                        with log_time_block(
                            "submodule_finalize",
                            logger=log,
                            module_name=resolved_label,
                        ):
                            process.submodule_finalize(module, looper.gptq_model)

                        # Disk offload (lifecycle TODO note preserved)
                        if isinstance(process, (GPTQProcessor, QQQProcessor, AWQProcessor, ParoQuantProcessor)):
                            quant_config = getattr(looper.gptq_model, "quantize_config", None)
                            if quant_config and getattr(quant_config, "offload_to_disk", False):
                                offload_path = getattr(quant_config, "offload_to_disk_path", None)
                                if offload_path:
                                    module_full_name = getattr(module, "full_name", None)
                                    target_module = (
                                        looper.gptq_model.model.get_submodule(module_full_name)
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
                                            model=looper.gptq_model.model,
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
                    return finalize_progress_cls(module_label, process_name, layer_idx)

                    # pb.subtitle(
                    #     f"{process.name()}: layer:{layer_idx} Finalized {idx}/{total} {module_label}"
                    # ).draw()

                for index, (process, module, module_label, target_dev, layer_idx) in enumerate(finalize_tasks, start=1):
                    # Schedule finalize work on the device thread pool so CPU
                    # bound tasks do not stall the main orchestration loop.
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

                looper._emit_layer_complete(
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
                    """Consumes finalize futures, updating progress and surfacing errors."""

                    completed_local = 0
                    try:
                        for future in as_completed(futures):
                            # Drain futures as they complete to surface errors
                            # quickly and keep the progress bar in sync.
                            try:
                                result = future.result()
                            except BaseException as exc:
                                log.exception("Submodule finalize task raised an exception")
                                looper._request_loop_stop(exc)
                                return

                            if isinstance(result, finalize_progress_cls):
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
                        looper._emit_layer_complete(
                            layer_idx=layer_idx_for_callback,
                            submodule_finalized=True,
                            raise_in_place=False,
                        )

                if finalize_futures_snapshot:
                    drain_sync = _should_drain_finalize_futures_synchronously(
                        looper,
                        finalize_tasks=finalize_tasks,
                    )
                    if durable_progress_logs:
                        log.info(
                            "StageLayer: layer=%s finalize queued modules=%s mode=%s",
                            layer_index if not is_lm_head_module else "lm_head",
                            finalize_count,
                            "sync" if drain_sync else "async",
                        )
                    if drain_sync:
                        # Synchronous: wait for all finalization to complete before proceeding to next layer
                        # This ensures all packing and writing tasks are done
                        _drain_finalize_futures(
                            [future for future, *_ in finalize_futures_snapshot],
                            finalize_pb,
                            finalize_count,
                            layer_index,
                        )
                        if looper.gptq_model.quantize_config.gc_mode == GcMode.ON_STAGE_END:
                            torch_empty_cache(device=cur_layer_device, sync=True)
                        elif _should_empty_cache_after_sync_finalize(
                            looper,
                            finalize_tasks=finalize_tasks,
                        ):
                            torch_empty_cache(device=cur_layer_device, gc=False, sync=True)
                    else:
                        # Asynchronous (current/default behavior): drain in background thread
                        # This allows next layer to start while current layer finalizes
                        finalizer_thread = threading.Thread(
                            target=_drain_finalize_futures,
                            args=(
                                [future for future, *_ in finalize_futures_snapshot],
                                finalize_pb,
                                finalize_count,
                                layer_index,
                            ),
                            name="SubmoduleFinalizeWatcher",
                            daemon=True,
                        )
                        looper.register_dangling_thread(finalizer_thread)
                        finalizer_thread.start()
                else:
                    looper._emit_layer_complete(
                        layer_idx=layer_index,
                        submodule_finalized=True,
                        raise_in_place=True,
                    )
                    if durable_progress_logs:
                        log.info(
                            "StageLayer: layer=%s complete (no finalize tasks)",
                            layer_index if not is_lm_head_module else "lm_head",
                        )

        if durable_progress_logs:
            log.info(
                "StageLayer: handoff complete for layer=%s",
                layer_index if not is_lm_head_module else "lm_head",
            )
