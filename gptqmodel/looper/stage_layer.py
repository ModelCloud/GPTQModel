# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Layer execution stage extracted from ModuleLooper."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import as_completed
from typing import TYPE_CHECKING, Dict, List, Optional
from ..nn_modules.hooked_linear import replace_module_with_hooked_legacy
from ..nn_modules.converter import MODULE_CONVERTER_MAP
import torch

from .. import DEBUG_ON, DEVICE_THREAD_POOL
from ..looper.awq_processor import AWQProcessor
from ..looper.gptq_processor import GPTQProcessor
from ..looper.named_module import NamedModule
from ..looper.qqq_processor import QQQProcessor
from ..utils.device import get_device, get_device_new
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import find_modules, get_module
from ..utils.offload import offload_to_disk
from ..utils.torch import CPU, torch_sync
from .stage_subset import SubsetForwardContext, run_subset_stage

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .module_looper import ModuleLooper


def run_layer_stage(
    looper: 'ModuleLooper',
    *,
    layers: List[torch.nn.Module],
    layer_modules: List[List[str]],
    layers_prefix: Optional[str],
    fail_safe: bool,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    pb,
    layer_count: int,
    region_timer,
    finalize_progress_cls,
    logger=None,
) -> None:
    """Execute the main per-layer quantization loop."""
    log = logger or setup_logger()
    for layer_index in pb:
        # Iterate over every transformer layer (plus lm_head when enabled) as
        # progress-bar controlled units of work.
        if looper._check_loop_stop():
            break
        is_lm_head_module = layer_index >= layer_count

        if is_lm_head_module:
            layer_title = "Quantizing lm_head"
            module = get_module(looper.gptq_model.model, key=looper.gptq_model.lm_head)
        else:
            layer_title = f"Quantizing layer {layer_index} of {layer_count - 1}"
            module = layers[layer_index]

        pb.title(layer_title).subtitle("").draw()

        if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
            # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
            continue

        module = looper.gptq_model.pre_quantize(module)

        model_type = looper.gptq_model.model.config.model_type
        if model_type in MODULE_CONVERTER_MAP:
            converter = MODULE_CONVERTER_MAP[model_type]
            module = converter(module, looper.gptq_model.model.config)

        replace_module_with_hooked_legacy(module, quant_lm_head=looper.gptq_model.quantize_config.lm_head)

        layers[layer_index] = module
        if is_lm_head_module:
            layer_descriptor = looper.gptq_model.lm_head
        elif layers_prefix:
            layer_descriptor = f"{layers_prefix}.{layer_index}"
        else:
            layer_descriptor = str(layer_index)

        cur_layer_device = get_device(module)
        full = find_modules(module, name=looper.gptq_model.lm_head if is_lm_head_module else "")

        for p_index, processor in enumerate(looper.processors):
            # Each processor contributes a quantization phase; walk them in
            # order so their caches and side effects line up with the pipeline.
            processor.log_call_count = 0  # reset
            processor.collect_memory_info(layer_index)

            modules = [[looper.gptq_model.lm_head]] if is_lm_head_module else layer_modules

            # for NativeProcessor we process one time forward on all grouped module subsets
            if processor.fwd_all_modules_in_single_pass:
                # merge all subsets into one
                modules = [sum(modules, [])]

            layer_inputs = processor.inputs_cache.layer_inputs
            if is_lm_head_module:
                layer_inputs = looper.gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
            layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
            position_ids = processor.inputs_cache.position_ids
            attention_masks = processor.inputs_cache.attention_masks

            processed_subset: Dict[str, NamedModule] = {}
            last_subset_context: Optional[SubsetForwardContext] = None
            subset_total = len(modules)
            previous_subset_processed: Optional[Dict[str, NamedModule]] = None

            for index, names in enumerate(modules):
                # Process the layer in smaller subsets so attention groups or
                # MoE experts can be quantized independently within a layer.
                if DEBUG_ON and log.isEnabledFor(logging.DEBUG):
                    if isinstance(processor, AWQProcessor):
                        log.debug(
                            "StageLayer[awq]: layer=%s subset=%s/%s size=%s names=%s",
                            layer_index,
                            index + 1,
                            subset_total,
                            len(names),
                            names[:5],
                        )
                    else:
                        log.debug(
                            "StageLayer: layer=%s subset=%s/%s processor=%s size=%s names=%s",
                            layer_index,
                            index + 1,
                            subset_total,
                            processor.name(),
                            len(names),
                            names[:8],
                        )
                subset_result = run_subset_stage(
                    looper=looper,
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
                    layers_prefix=layers_prefix,
                    subset_names=names,
                    subset_index=index,
                    subset_total=subset_total,
                    full=full,
                    fail_safe=fail_safe,
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
                if subset_result.forward_context is not None:
                    last_subset_context = subset_result.forward_context

            is_last_module = layer_index == len(pb) - 1
            layer_outputs: List[List[torch.Tensor]] = []
            subset_context = last_subset_context
            forward_device_map = subset_context.forward_device_map if subset_context else {}
            subset_forward_serial = subset_context.subset_forward_serial if subset_context else False
            subset_reference_total = subset_context.subset_total if subset_context else subset_total
            subset_reference_index = subset_context.subset_index if subset_context else max(subset_total - 1, 0)
            subset_for_overrides = subset_context.subset if subset_context else {}
            preserve_devices = bool(forward_device_map)

            # second forward after process()
            if not is_last_module and processor.fwd_after_process and subset_context is not None:
                replay_batch_count = looper._resolve_batch_total(
                    getattr(processor, "num_batches", None),
                    layer_inputs,
                )
                replay_row_counts = list(looper._collect_row_counts(layer_inputs))
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
                replay_source = f"{layer_descriptor}:subset{subset_reference_index + 1}/{subset_reference_total}"

                replay_prev_devices: Dict[str, torch.device] = {}
                if forward_device_map:
                    replay_prev_devices = looper._apply_forward_device_overrides(
                        subset_for_overrides,
                        forward_device_map,
                        fallback_modules=full,
                    )

                # if log.isEnabledFor(logging.DEBUG):
                #     replay_snapshot = []
                #     for name, named_module in subset.items():
                #         target_device = getattr(named_module, "target_device", None)
                #         if target_device is None:
                #             try:
                #                 target_device = get_device(named_module.module)
                #             except Exception:
                #                 target_device = None
                #         target_device_str = str(target_device) if target_device is not None else "unknown"
                #         replay_snapshot.append(f"{name}:{target_device_str}")
                #     log.debug(
                #         "ModuleLooper: Forward replay device snapshot (layer=`%s`, subset=%d/%d, serial=%s) %s",
                #         layer_descriptor,
                #         index + 1,
                #         subset_total,
                #         subset_forward_serial,
                #         ", ".join(replay_snapshot),
                #     )

                try:
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
                        force_serial=subset_forward_serial,
                        preserve_module_devices=preserve_devices,
                    )
                finally:
                    if forward_device_map:
                        looper._restore_forward_device_overrides(
                            subset_for_overrides,
                            replay_prev_devices,
                            fallback_modules=full,
                        )
                    if replay_pb is not None:
                        replay_pb.close()
                if region_timer is not None:
                    region_timer.record(
                        "post_quant_forward",
                        time.perf_counter() - replay_start,
                        source=replay_source,
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

            if processor.fwd_after_process:
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
                        if isinstance(process, (GPTQProcessor, QQQProcessor, AWQProcessor)):
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
                    looper._emit_layer_complete(
                        layer_idx=layer_index,
                        submodule_finalized=True,
                        raise_in_place=True,
                    )
