from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import as_completed
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from ... import DEVICE_THREAD_POOL
from ...utils.ctx import ctx
from ...utils.device import get_device, get_device_new
from ...utils.logger import log_time_block, setup_logger
from ...utils.model import find_modules, get_module, get_module_by_name_prefix
from ...utils.offload import offload_to_disk
from ...utils.torch import CPU, torch_sync
from ...nn_modules.hooked_linear import HookedLinear, replace_module_with_hooked_legacy
from ...quantization.config import VRAMStrategy
from ...models._const import SUPPORTS_MODULE_TYPES
from ...utils.looper_helpers import device_ctx
from ..awq_processor import AWQProcessor
from ..dequantize_processor import DequantizeProcessor
from ..eora_processor import EoraProcessor
from ..gptq_processor import GPTQProcessor
from ..qqq_processor import QQQProcessor
from ..named_module import NamedModule
from ..lifecycle.types import FinalizeProgressInfo
from ..input_cache import InputCache

LOOP_DEBUG_ENABLED = os.environ.get("GPTQ_LOOP_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _loop_debug(log: logging.Logger, message: str) -> None:
    if LOOP_DEBUG_ENABLED:
        log.info("[LoopDebug] %s", message)

if TYPE_CHECKING:  # pragma: no cover
    from ..module_looper import ModuleLooper


class LayerLoopStage:
    """Encapsulate the per-layer lifecycle."""

    def __init__(self, looper: "ModuleLooper", logger: Optional[logging.Logger] = None) -> None:
        self.looper = looper
        self.log = logger or setup_logger()

    def run(self, fail_safe: bool = False, **kwargs):
        looper = self.looper
        gptq_model = looper.gptq_model
        log = self.log

        if gptq_model.quantize_config.lm_head:
            if gptq_model.model.config.tie_word_embeddings and hasattr(gptq_model.model.model, "_tied_weights_keys"):
                tied_keys = gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if gptq_model.lm_head in item:
                        raise NotImplementedError("quantization of `lm_head` layer with `tied_weights=True` model state is not supported. Please check model has `tied_weights=False`.")

            lm_head_module = get_module(gptq_model.model, key=gptq_model.lm_head)
            if get_module(gptq_model.model, key=gptq_model.lm_head) is None:
                raise ValueError(f"could not find layer {gptq_model.lm_head} in the model, exit...")

            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                                          f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}")

            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if gptq_model.quantize_config.dynamic is None:
                gptq_model.quantize_config.dynamic = {gptq_model.lm_head: lm_head_quant_config}
            elif gptq_model.quantize_config.dynamic_get(gptq_model.lm_head, default=None) is None:
                gptq_model.quantize_config.dynamic[gptq_model.lm_head] = lm_head_quant_config

        forward_pass_use_cache = gptq_model.model.config.use_cache if hasattr(gptq_model.model.config, "use_cache") else False
        gptq_model.model.config.use_cache = False
        layers, layers_prefix = get_module_by_name_prefix(gptq_model.model, gptq_model.extract_layers_node())
        region_timer = getattr(gptq_model, "quant_region_timer", None)

        for p_index, processor in enumerate(looper.processors):
            if not processor.verify_calibration_dataset(p_index):
                if isinstance(processor, EoraProcessor) or\
                        (isinstance(processor, GPTQProcessor) and gptq_model.quantize_config.v2):
                    prev_processor = looper.processors[p_index - 1]
                    processor.set_calibration_dataset(prev_processor.calibration_dataset)
                    processor.receive_input_cache(prev_processor.inputs_cache)
                elif isinstance(processor, DequantizeProcessor):
                    processor.set_calibration_dataset([])
                    processor.receive_input_cache(InputCache([], [], [], []))

                continue

            input_cache = looper.cache_inputs(layers=layers,
                                            calibration_data=processor.calibration_dataset,
                                            use_cache=False)
            processor.receive_input_cache(input_cache)

        for processor in looper.processors:
            processor.release_calibration_dataset()

        if region_timer is not None:
            region_timer.flush()

        layer_modules = gptq_model.simple_layer_modules(model_config=gptq_model.model.config, quantize_config=gptq_model.quantize_config)

        if not gptq_model.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        layer_count = len(layers)
        pb = (log.pb(layer_count + 1 if gptq_model.quantize_config.lm_head else layer_count)
                            .manual()
                            .set(left_steps_offset=1))

        for processor in looper.processors:
            processor.layer_count = layer_count
            processor.pb = pb

        shared_kv_cache_dict = {}

        replace_module_with_hooked_legacy(gptq_model.model, quant_lm_head=gptq_model.quantize_config.lm_head)

        if gptq_model.quantize_config.lm_head:
            lm_head_module = get_module(gptq_model.model, key=gptq_model.lm_head)
            if lm_head_module and isinstance(lm_head_module, torch.nn.Linear):
                hooked_lm_head = HookedLinear.from_linear(lm_head_module)
                module_path = gptq_model.lm_head.split('.')
                parent = gptq_model.model
                for part in module_path[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, module_path[-1], hooked_lm_head)

        for layer_index in pb:
            if looper._check_loop_stop():
                break
            is_lm_head_module = layer_index >= layer_count

            if is_lm_head_module:
                layer_title = "Quantizing lm_head"
                module = get_module(gptq_model.model, key=gptq_model.lm_head)
            else:
                layer_title = f"Quantizing layer {layer_index} of {layer_count - 1}"
                module = layers[layer_index]

            pb.title(layer_title).subtitle("").draw()

            if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                continue

            module = gptq_model.pre_quantize(module)

            if is_lm_head_module:
                layer_descriptor = gptq_model.lm_head
            elif layers_prefix:
                layer_descriptor = f"{layers_prefix}.{layer_index}"
            else:
                layer_descriptor = str(layer_index)

            _loop_debug(
                log,
                f"Entering layer {layer_index} (descriptor={layer_descriptor}, lm_head={is_lm_head_module})",
            )

            cur_layer_device = get_device(module)
            full = find_modules(module, name=gptq_model.lm_head if is_lm_head_module else "")

            for p_index, processor in enumerate(looper.processors):
                _loop_debug(
                    log,
                    f"Layer {layer_index}: starting processor {p_index + 1}/{len(looper.processors)} ({processor.name()})",
                )
                processor.log_call_count = 0
                processor.collect_memory_info(layer_index)

                modules = [[gptq_model.lm_head]] if is_lm_head_module else layer_modules

                if processor.fwd_all_modules_in_single_pass:
                    modules = [sum(modules, [])]

                if isinstance(processor, AWQProcessor):
                    named_childs = dict()
                    for index, names in enumerate(modules):
                        named_modules = looper.crate_named_modules(full=full,
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
                    if p_index == len(looper.processors) - 1:
                        looper._emit_layer_complete(
                            layer_idx=layer_index,
                            submodule_finalized=False,
                            raise_in_place=True,
                        )
                        looper._emit_layer_complete(
                            layer_idx=layer_index,
                            submodule_finalized=True,
                            raise_in_place=True,
                        )
                    continue

                layer_inputs = processor.inputs_cache.layer_inputs
                if is_lm_head_module:
                    layer_inputs = gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                position_ids = processor.inputs_cache.position_ids
                attention_masks = processor.inputs_cache.attention_masks

                processed_subset = {}

                for index, names in enumerate(modules):
                    subset = looper.crate_named_modules(full=full, is_lm_head_module=is_lm_head_module,
                                                      layer_index=layer_index, layers_prefix=layers_prefix,
                                                      names=names,
                                                      processor=processor,
                                                      fail_safe=fail_safe)

                    if len(subset) == 0:
                        continue

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

                        subset_forward_serial = looper._vram_strategy == VRAMStrategy.EXCLUSIVE
                    else:
                        for named_module in subset.values():
                            setattr(named_module, "moe_enabled", False)

                    handle = []
                    subset_total = len(modules)
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
                        is_last = (idx == subset_size - 1)
                        hook_source = getattr(m, "full_name", None)
                        if hook_source is None:
                            hook_source = getattr(m, "name", name)
                        if hook_source is None:
                            hook_source = str(name)

                        if hasattr(subset[name], 'forward_hook'):
                            original_hook = processor.pre_process_fwd_hook(name)
                            subset[name].forward_hook = looper._masked_hook_wrapper(processor, original_hook, hook_source)
                            if is_last and processor.fwd_after_process:
                                subset[name].forward_hook_last = True
                        else:
                            original_hook = processor.pre_process_fwd_hook(name)
                            handle.append(subset[name].register_forward_hook(
                                looper._masked_hook_wrapper(processor, original_hook, hook_source)
                            ))

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

                    quant_target_devices: Dict[str, torch.device] = {}
                    for name, named_module in subset.items():
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

                    futures = []

                    @torch.inference_mode()
                    def _process_on_worker(
                        proc,
                        nm,
                        expected_device,
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

                        timer = getattr(gptq_model, "quant_region_timer", None)
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

                is_last_module = layer_index == len(pb) - 1
                layer_outputs: List[List[torch.Tensor]] = []
                if not is_last_module and processor.fwd_after_process:
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

                    replay_start = time.perf_counter()
                    replay_source = f"{layer_descriptor}:subset{index + 1}/{subset_total}"

                    replay_prev_devices: Dict[str, torch.device] = {}
                    if forward_device_map:
                        replay_prev_devices = looper._apply_forward_device_overrides(
                            subset,
                            forward_device_map,
                            fallback_modules=full,
                        )

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
                            reuse_kv=reuse_kv,
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
                                subset,
                                replay_prev_devices,
                                fallback_modules=full,
                            )
                        if replay_pb is not None:
                            replay_pb.close()
                    processor.receive_layer_inputs(layer_outputs)
                    replay_time = time.perf_counter() - replay_start
                    processor.set_replay_time(replay_time)
                    if region_timer is not None:
                        region_timer.record(
                            "post_quant_forward",
                            replay_time,
                            source=replay_source,
                        )

                if isinstance(processor, GPTQProcessor):
                    processor.recall_zp_snr_stats(layer_index)

                if p_index == len(looper.processors) - 1:
                    torch_sync()

                    finalize_tasks = []

                    for reverse_p in reversed(looper.processors):
                        for module in processed_subset.values():
                            # actual_module = module.module if isinstance(module, NamedModule) else module

                            # get_device_new(
                            #     actual_module,
                            #     recursive=True,
                            #     assert_mode=True,
                            #     expected=CPU,
                            # )
                            with looper._quant_device_lock:
                                key = getattr(module, "full_name", getattr(module, "name", None))
                                if key is not None:
                                    looper._module_device_map[key] = CPU

                            target_dev = CPU
                            module_label = getattr(module, "full_name", getattr(module, "name", ""))
                            layer_idx = getattr(module, "layer_index", None)
                            finalize_tasks.append((reverse_p, module, module_label, target_dev, layer_idx))

                    if LOOP_DEBUG_ENABLED:
                        debug_summary = [
                            f"{getattr(proc, 'name', lambda: '<processor>')()}:{label or '<unnamed>'}"
                            for proc, _, label, _, _ in finalize_tasks
                        ]
                        _loop_debug(
                            log,
                            f"Layer {layer_index}: scheduled {len(finalize_tasks)} finalize tasks -> {debug_summary}",
                        )

                    finalize_count = len(finalize_tasks)
                    finalize_futures = []
                    finalize_pb = log.pb(range(finalize_count)).manual().set(show_left_steps=False)

                    @torch.inference_mode()
                    def _finalize_on_worker(process, module, idx, total, module_label, layer_idx):
                        resolved_label = module_label or getattr(module, "full_name", getattr(module, "name", ""))
                        start = time.perf_counter() if region_timer is not None else None
                        _loop_debug(
                            log,
                            f"Layer {layer_idx}: finalize start ({idx}/{total}) proc={getattr(process, 'name', lambda: '<processor>')()} module={resolved_label}",
                        )
                        try:
                            with log_time_block(
                                "submodule_finalize",
                                logger=log,
                                module_name=resolved_label,
                            ):
                                process.submodule_finalize(module, gptq_model)

                            if isinstance(process, (GPTQProcessor, QQQProcessor, AWQProcessor)):
                                quant_config = getattr(gptq_model, "quantize_config", None)
                                if quant_config and getattr(quant_config, "offload_to_disk", False):
                                    offload_path = getattr(quant_config, "offload_to_disk_path", None)
                                    if offload_path:
                                        module_full_name = getattr(module, "full_name", None)
                                        target_module = (
                                            gptq_model.model.get_submodule(module_full_name)
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
                                                model=gptq_model.model,
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
                            _loop_debug(
                                log,
                                f"Layer {layer_idx}: finalize end proc={getattr(process, 'name', lambda: '<processor>')()} module={resolved_label}",
                            )
                            if region_timer is not None and start is not None:
                                region_timer.record(
                                    "submodule_finalize",
                                    time.perf_counter() - start,
                                    source=resolved_label,
                                )
                        process_name = process.name() if process is not None else "<processor>"
                        return FinalizeProgressInfo(module_label, process_name, layer_idx)

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
                        includes_unknown = any(layer_idx is None for _, _, _, _, layer_idx in finalize_futures_snapshot)
                        if known_layers:
                            sample_layers = ", ".join(str(idx) for idx in known_layers[:3])
                            if len(known_layers) > 3:
                                sample_layers += ", …"
                            suffix = ", ?" if includes_unknown else ""
                            prefix = "Layer" if len(known_layers) == 1 else "Layers"
                            layer_heading = f"{prefix} {sample_layers}{suffix}"
                        elif includes_unknown:
                            layer_heading = "Layer ?"
                        else:
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
                        _loop_debug(
                            log,
                            f"Layer {layer_idx_for_callback}: draining {len(futures)} finalize futures",
                        )
                        completed_local = 0
                        try:
                            for future in as_completed(futures):
                                try:
                                    result = future.result()
                                    _loop_debug(
                                        log,
                                        f"Layer {layer_idx_for_callback}: finalize future completed ({completed_local + 1}/{finalize_count_local})",
                                    )
                                except BaseException as exc:
                                    log.exception("Submodule finalize task raised an exception")
                                    looper._request_loop_stop(exc)
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
                            looper._emit_layer_complete(
                                layer_idx=layer_idx_for_callback,
                                submodule_finalized=True,
                                raise_in_place=False,
                            )

                    if finalize_futures_snapshot:
                        _loop_debug(
                            log,
                            f"Layer {layer_index}: launching finalize watcher for {len(finalize_futures_snapshot)} futures",
                        )
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
                        _loop_debug(
                            log,
                            f"Layer {layer_index}: no finalize futures, emitting completion directly",
                        )
                        looper._emit_layer_complete(
                            layer_idx=layer_index,
                            submodule_finalized=True,
                            raise_in_place=True,
                        )

                _loop_debug(
                    log,
                    f"Layer {layer_index}: completed processor {p_index + 1}/{len(looper.processors)} ({processor.name()})",
                )

        looper._check_loop_stop()
        DEVICE_THREAD_POOL.wait()
        looper._check_loop_stop()

        total_log = {}
        reversed_processors = list(reversed(looper.processors))

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
                looper._check_loop_stop()
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
                    gptq_model.quant_log = reverse_p.log

                for module_log in reverse_p.log:
                    log.info(module_log)
                reverse_p.log_plotly()

                finalize_start = time.perf_counter() if region_timer is not None else None
                try:
                    reverse_p.finalize(model=gptq_model, **kwargs)
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

        gptq_model.model.config.use_cache = forward_pass_use_cache

        return total_log
