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

import torch
from defuser.modeling.replace_modules import materialize_model

from .. import DEBUG_ON, DEVICE_THREAD_POOL
from ..looper.awq_processor import AWQProcessor
from ..looper.gptq_processor import GPTQProcessor
from ..looper.named_module import NamedModule
from ..looper.paroquant_processor import ParoQuantProcessor
from ..looper.qqq_processor import QQQProcessor
from ..nn_modules.converter import MODULE_CONVERTER_MAP
from ..nn_modules.hooked_linear import replace_module_with_hooked_legacy
from ..quantization.config import GcMode, QuantizeEmbed
from ..utils.device import get_device, get_device_new
from ..utils.logger import live_renderables_suppressed, log_time_block, setup_logger
from ..utils.looper_helpers import (
    find_last_quantized_layer_index,
    normalize_device_like,
)
from ..utils.model import find_modules, get_layer_name, get_module
from ..utils.offload import offload_to_disk
from ..utils.torch import CPU, torch_empty_cache, torch_sync
from .extension import LoopStep
from .resume import (
    load_activation_cache,
    marker_layer_finalized_count,
    read_resume_target,
    restore_completed_layer,
    restore_completed_layer_class_only,
    resume_state_path,
    save_activation_cache,
    write_resume_marker,
)
from .stage_subset import SubsetPlan, build_layer_subset_plans, run_subset_stage

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .module_looper import ModuleLooper


def _log_cuda_memory_diagnostics(log, layer_index) -> None:
    """One line per visible CUDA device: allocator-active vs allocator-reserved bytes.

    Diagnostic for layer-over-layer GPU memory growth: `nvidia-smi`/process RSS
    only show *reserved* memory (what the CUDA caching allocator has claimed
    from the driver and is holding, whether or not anything is using it right
    now), not what is actually *active* (live tensors). If `active` stays flat
    while `reserved` keeps climbing layer over layer, the cause is allocator
    fragmentation/churn, not a genuine reference-counting leak of some tensor
    nobody is freeing. If both climb together, that points back to a real leak.
    """
    if not torch.cuda.is_available():
        return
    try:
        for idx in range(torch.cuda.device_count()):
            stats = torch.cuda.memory_stats(idx)
            active = stats.get("active_bytes.all.current", 0) / (1024 ** 3)
            reserved = stats.get("reserved_bytes.all.current", 0) / (1024 ** 3)
            retries = stats.get("num_alloc_retries", 0)
            log.info(
                "MemDiag: layer=%s cuda:%s active=%.2fGiB reserved=%.2fGiB gap=%.2fGiB alloc_retries=%s",
                layer_index,
                idx,
                active,
                reserved,
                reserved - active,
                retries,
            )
    except Exception as exc:  # pragma: no cover - diagnostics must never break the run
        log.debug("MemDiag: failed to collect CUDA memory stats: %s", exc)


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

    Any multi-accelerator quantization flow can overlap layer N finalizers with
    layer N+1 materialization/replay if we keep the default async drain. That
    saves some wall time, but it also broadens the lifetime of device-resident
    weights, activations, and packing state across layer boundaries. In
    practice, the overlap is not worth the allocator pressure risk, so
    multi-device runs drain per-layer finalizers synchronously.

    Same trade when this run opted into resume (GPTQMODEL_RESUME=1 with
    offload_to_disk configured, see resume_state_path): the marker can only
    be written once finalization is known durable, so the async-drain
    speedup is given up in exchange for that run being resumable if it later
    crashes. Offload users who never set that env var keep the plain async
    drain unchanged.
    """
    if looper.gptq_model.quantize_config.wait_for_submodule_finalizers:
        return True

    # A resume marker can only be written once this layer's finalizers are
    # known to have durably completed, so resume-capable runs must drain
    # synchronously regardless of the flag above -- otherwise the marker
    # simply never gets written and GPTQMODEL_RESUME=1 has nothing to
    # resume from.
    if resume_state_path(looper.gptq_model.quantize_config) is not None:
        return True

    quant_devices = getattr(looper, "_quant_devices", None) or []
    active_accelerators = {
        (device.type, device.index)
        for device_like in quant_devices
        if (device := normalize_device_like(device_like)) is not None and device.type != "cpu"
    }
    if len(active_accelerators) > 1:
        return True
    return any(isinstance(process, ParoQuantProcessor) for process, *_ in finalize_tasks)


def _is_resume_fastforward_candidate(
    *,
    is_embeddings_module: bool,
    layer_index: int,
    resume_target: Optional[int],
) -> bool:
    """Whether this loop iteration is a completed transformer layer to fast-forward past.

    Input/output embeddings and lm_head share layer_index=0's numeric value
    with a real transformer layer once layer_index_offset is applied (see
    is_embeddings_module at its call site), so they must stay out of this
    path entirely -- otherwise it would fast-forward using the wrong module.
    """
    return resume_target is not None and not is_embeddings_module and layer_index <= resume_target


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


def _collect_hook_skip_modules(planning_layer_modules: List[List[str]]) -> set[str]:
    """Collect module paths flagged as non-quantized in module-tree planning blocks."""

    skip_modules: set[str] = set()
    for block in planning_layer_modules:
        for module_name in block:
            if ":!" in module_name:
                path = module_name.split(":", 1)[0]
                if path:
                    skip_modules.add(path)
    return skip_modules


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
    force_serial: bool = False,
) -> List[List[torch.Tensor]]:
    """Replay one layer forward to materialize outputs for the next layer.

    ``force_serial`` overrides data-parallel forward dispatch. Resume replay
    passes this: running several batches through torch's dynamo-instrumented
    eval-frame hook from multiple free-threading workers at once has been
    observed to livelock (every worker spins inside the frame-evaluation
    shim, 0% GPU util, no forward progress). Serial execution avoids the
    concurrent entry entirely; the cost is bounded to the handful of already-
    quantized layers being fast-forwarded, not the run's main quant loop.
    """

    if replay_plan is None:
        replay_batch_count, replay_row_counts, replay_total_rows = _collect_layer_forward_progress(
            looper,
            processor=processor,
            layer_inputs=layer_inputs,
        )
        replay_source = f"{layer_descriptor}:untouched"
        replay_modules = None
        replay_forward_device_map: Dict[str, torch.device] = {}
        replay_force_serial = force_serial
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
        replay_force_serial = replay_plan.subset_forward_serial or force_serial
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
            reuse_kv=getattr(module, "reuse_kv", False),
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


def _resume_replay_layer(
    looper: "ModuleLooper",
    *,
    layers: List[torch.nn.Module],
    layer_index: int,
    layer_name: str,
    layer_title: str,
    layer_count: int,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    pb,
    log,
    region_timer,
    preloaded_cache=None,
) -> None:
    """Fast-forward one already-quantized layer during a resumed run.

    `preloaded_cache`, when given, is this layer's already-loaded
    `load_activation_cache` result (the caller already had to load it once
    to decide the fast-forward strategy -- see `resume_target_cache` in
    `run_layer_stage` -- so this avoids re-reading the same safetensors
    file for the same layer a second time).

    Replays the layer using its ORIGINAL (pre-quantization) checkpoint
    weights, then swaps in the packed quant modules from the offload
    directory afterward. This mirrors the normal (non-resumed) quantization
    loop exactly: `run_layer_stage`'s per-subset loop always computes the
    next layer's input from the still-unquantized modules and only finalizes
    (replaces with quant modules, packs, offloads) once every subset for the
    layer is done -- the quantized weights never re-enter the forward path.
    Replaying with the already-quantized weights instead silently produces
    wrong activations, so this must use the pristine (pre-quant) weights.
    """
    model = looper.gptq_model
    module = layers[layer_index]

    pb.title(f"{layer_title} (resume replay)").subtitle("").draw()

    # Use the same CPU-resident weights as the normal replay path.
    module = model.shell_module_materialize(target_submodule=module, device=CPU)
    model_type = model.model.config.model_type
    if model_type in MODULE_CONVERTER_MAP:
        module = MODULE_CONVERTER_MAP[model_type](module, model.model.config)
        layers[layer_index] = module
    materialize_model(module)

    layer_prefix = layer_name if layer_name else f"{model.extract_layers_node()}.{layer_index}"

    module = model.pre_quantize(module)
    layers[layer_index] = module
    cur_layer_device = get_device(module)

    processor = looper.processors[-1]
    cached = preloaded_cache if preloaded_cache is not None else load_activation_cache(looper, layer_index, layer_count)
    if cached is not None:
        cached_layer_outputs, cached_shared_kv = cached
        # Reuse cached outputs and restore the paired shared state.
        calib_device_cfg = model.quantize_config.calibration_data_device
        target_device = cur_layer_device if calib_device_cfg is None else CPU
        layer_outputs = [
            [item.to(target_device) if torch.is_tensor(item) else item for item in batch]
            for batch in cached_layer_outputs
        ]
        if cached_shared_kv is not None:
            # Restore the shared state consumed by the next layer.
            shared_kv_cache_dict[layer_index] = cached_shared_kv.to(cur_layer_device)
        log.info(
            "Resume: layer %s used the cached activation output; skipped forward replay.",
            layer_index,
        )
    else:
        layer_outputs = _replay_layer_outputs(
            looper,
            module=module,
            processor=processor,
            layer_inputs=processor.inputs_cache.layer_inputs,
            layer_input_kwargs=processor.inputs_cache.layer_input_kwargs,
            position_ids=processor.inputs_cache.position_ids,
            attention_masks=processor.inputs_cache.attention_masks,
            cur_layer_device=cur_layer_device,
            is_lm_head_module=False,
            shared_kv_cache_dict=shared_kv_cache_dict,
            layer_index=layer_index,
            layer_descriptor=f"{layer_prefix}:resume",
            full=find_modules(module, name=""),
            log=log,
            region_timer=region_timer,
            force_serial=True,
            replay_plan=None,
        )
    processor.clear_cache_data()
    processor.receive_layer_inputs(layer_outputs)

    # Replay with original weights, then restore the packed modules.
    module = model.post_quantize(module)
    layers[layer_index] = module
    restored = restore_completed_layer(looper, layer_prefix)
    if not restored:
        raise RuntimeError(
            f"Resume: marker claims layer {layer_index} is complete but no offloaded "
            f"quant modules were found under prefix `{layer_prefix}`. Delete "
            "the resume marker (or unset GPTQMODEL_RESUME) to requantize from scratch."
        )
    expected_count = marker_layer_finalized_count(model.quantize_config, layer_index)
    if expected_count is not None and len(restored) != expected_count:
        # Missing modules would otherwise remain unquantized in the saved model.
        raise RuntimeError(
            f"Resume: layer {layer_index} restored {len(restored)} quant modules but the "
            f"marker records {expected_count} finalized modules. The offload directory is "
            "incomplete or damaged; delete the resume marker (or unset GPTQMODEL_RESUME) "
            "to requantize from scratch."
        )
    if expected_count is None:
        log.warn(
            "Resume: marker has no finalized-module count for layer %s (written by an "
            "older build?); restored set cannot be validated.",
            layer_index,
        )
    log.info(
        "Resume: layer %s replayed with original weights and restored %s quant "
        "modules from offload for the saved model.",
        layer_index,
        len(restored),
    )

    layers[layer_index] = model.post_quantize(module)
    # Re-offload restored modules to keep memory flat during replay.
    if model.quantize_config.offload_to_disk:
        offload_to_disk(
            model=model.model,
            module=restored,
            disk_path=model.quantize_config.offload_to_disk_path,
            force=True,
        )
    # Evict per-layer shared state to prevent unbounded growth.
    shared_kv_cache_dict.pop(layer_index - 1, None)
    torch_empty_cache(device=cur_layer_device)


def _resume_restore_only_layer(
    looper: "ModuleLooper",
    *,
    layers: List[torch.nn.Module],
    layer_index: int,
    layer_name: str,
    layer_title: str,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    pb,
    log,
) -> None:
    """Restore one already-quantized layer's packed modules without replaying its forward.

    Used when a downstream (higher-index) layer in this same resume already has
    a valid activation cache: that layer's cached output short-circuits the
    whole 0..resume_target chain, so every layer strictly before it becomes a
    pure pass-through -- none of their forward outputs are consumed by
    anything. The only work that still matters for them is restoring their
    packed quant modules from the offload directory into the live model so
    the eventual save() has correctly quantized weights.
    """
    model = looper.gptq_model
    module = layers[layer_index]

    restore_title = layer_title.replace("Quantizing", "Restoring", 1)
    pb.title(f"{restore_title} (no replay)").subtitle("").draw()

    # CPU materialization is needed for restore; no GPU compute is required.
    module = model.shell_module_materialize(target_submodule=module, device=CPU)
    model_type = model.model.config.model_type
    if model_type in MODULE_CONVERTER_MAP:
        module = MODULE_CONVERTER_MAP[model_type](module, model.model.config)
    layers[layer_index] = module
    materialize_model(module)

    layer_prefix = layer_name if layer_name else f"{model.extract_layers_node()}.{layer_index}"

    # Restore only module classes; save() reads weights from the offload bundle.
    restored = restore_completed_layer_class_only(looper, layer_prefix)
    if not restored:
        raise RuntimeError(
            f"Resume: marker claims layer {layer_index} is complete but no offloaded "
            f"quant modules were found under prefix `{layer_prefix}`. Delete "
            "the resume marker (or unset GPTQMODEL_RESUME) to requantize from scratch."
        )
    expected_count = marker_layer_finalized_count(model.quantize_config, layer_index)
    if expected_count is not None and len(restored) != expected_count:
        # Missing modules would otherwise remain unquantized in the saved model.
        raise RuntimeError(
            f"Resume: layer {layer_index} restored {len(restored)} quant modules but the "
            f"marker records {expected_count} finalized modules. The offload directory is "
            "incomplete or damaged; delete the resume marker (or unset GPTQMODEL_RESUME) "
            "to requantize from scratch."
        )
    if expected_count is None:
        log.warn(
            "Resume: marker has no finalized-module count for layer %s (written by an "
            "older build?); restored set cannot be validated.",
            layer_index,
        )
    log.info(
        "Resume: layer %s class-restored %s quant modules (meta, no data read/rewrite); "
        "forward replay skipped (a downstream layer's activation cache makes this layer's "
        "own output unnecessary).",
        layer_index,
        len(restored),
    )

    # No re-offload: restore_completed_layer_class_only never materialized or
    # modified any tensor data, so there is nothing to write back -- the
    # original offload bundle already has everything save() will need.
    shared_kv_cache_dict.pop(layer_index - 1, None)
    torch_empty_cache()


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
    layer_names: Optional[List[str]],
    fallback,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    pb,
    layer_count: int,
    region_timer,
    finalize_progress_cls,
    embed_quant_mode: Optional[QuantizeEmbed] = None,
    embed_only: Optional[bool] = None,
    logger=None,
) -> None:
    """Execute the main per-layer quantization loop."""
    # Trailing layers whose tracked modules are all dynamically excluded never
    # need another forward or finalize pass, so the loop can stop once the
    # final eligible layer has been processed.
    last_quantized_layer_index = find_last_quantized_layer_index(
        looper.gptq_model.quantize_config,
        layer_modules=layer_modules,
        layer_names=layer_names,
        layer_count=layer_count,
    )

    log = logger or setup_logger()
    durable_progress_logs = live_renderables_suppressed()
    hook_skip_modules = _collect_hook_skip_modules(planning_layer_modules)
    quant_input_embeddings = embed_quant_mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH)
    quant_output_embeddings = embed_quant_mode in (QuantizeEmbed.OUTPUT, QuantizeEmbed.BOTH)
    input_embeddings_name = getattr(looper, "input_embeddings_name", None)
    output_embeddings_name = getattr(looper, "output_embeddings_name", None)
    requant_endpoint_names = {
        name
        for enabled, name in (
            (quant_input_embeddings, input_embeddings_name),
            (quant_output_embeddings, output_embeddings_name),
        )
        if enabled and name
    }
    layer_index_offset = 1 if quant_input_embeddings else 0

    resume_target = read_resume_target(looper, layer_count)
    extensions = getattr(looper, "extensions", None)
    if extensions and resume_target is not None:
        raise ValueError("Loop extensions cannot yet be combined with legacy resume replay")
    # If the resume_target layer itself already has a valid activation cache,
    # its cached output short-circuits the whole 0..resume_target chain: none
    # of the earlier layers' forward outputs are consumed by anything, so
    # they can skip forward replay entirely (see `_resume_restore_only_layer`).
    # Loaded once here (not just checked) so _resume_replay_layer below can
    # reuse the same result instead of re-reading the same file.
    resume_target_cache = (
        load_activation_cache(looper, resume_target, layer_count) if resume_target is not None else None
    )
    resume_target_cache_hit = resume_target_cache is not None
    if resume_target is not None:
        log.info(
            "Resume: layers 0..%s are already quantized on disk; "
            "fast-forwarding them with forward-only replay.",
            resume_target,
        )
        if resume_target_cache_hit:
            log.info(
                "Resume: layer %s has a cached activation output; layers 0..%s will skip "
                "forward replay entirely (restore-only).",
                resume_target,
                resume_target - 1,
            )

    for layer_index in pb:
        boundary_futures = []
        # Iterate over every transformer layer (plus lm_head when enabled) as
        # progress-bar controlled units of work.
        if looper._check_loop_stop():
            break
        progress_index = layer_index
        is_input_embeddings_module = quant_input_embeddings and progress_index == 0
        model_layer_index = progress_index - layer_index_offset
        is_output_embeddings_module = quant_output_embeddings and model_layer_index >= layer_count
        is_lm_head_module = (
            not is_output_embeddings_module
            and looper.gptq_model.quantize_config.lm_head
            and model_layer_index >= layer_count
        )
        is_embeddings_module = (
            is_input_embeddings_module or is_output_embeddings_module or is_lm_head_module
        )

        if (
            embed_quant_mode is None
            and not is_embeddings_module
            and last_quantized_layer_index is not None
            and model_layer_index > last_quantized_layer_index
        ):
            # The remaining layers are fully skipped by dynamic config, so
            # avoid entering another layer-level quantization cycle.
            log.debug(
                "StageLayer: early stop at layer=%s, last_quantized_layer=%s",
                model_layer_index,
                last_quantized_layer_index,
            )
            pb.close()
            break

        if is_input_embeddings_module:
            layer_title = "Quantizing input embeddings"
            module = looper.gptq_model.get_input_embeddings()
            pristine_group_module = None
            layer_name = ""
        elif is_output_embeddings_module:
            layer_title = "Quantizing output embeddings"
            module = looper.gptq_model.get_output_embeddings()
            pristine_group_module = None
            layer_name = ""
        elif is_lm_head_module:
            layer_title = "Quantizing lm_head"
            module = get_module(looper.gptq_model.model, key=looper.gptq_model.lm_head)
            pristine_group_module = None
            layer_name = ""
        else:
            layer_index = model_layer_index
            layer_title = f"Quantizing layer {layer_index} of {layer_count - 1}"
            module = layers[layer_index]
            pristine_group_module = None
            layer_name = get_layer_name(layer_names, layer_index)

        pb.title(layer_title).subtitle("").draw()
        if durable_progress_logs:
            log.info(
                "StageLayer: start layer=%s/%s title=`%s`",
                layer_index if not is_embeddings_module else layer_title.replace("Quantizing ", ""),
                layer_count - 1 if not is_embeddings_module else layer_title.replace("Quantizing ", ""),
                layer_title,
            )
        # Emit diagnostics even when live progress logs are disabled.
        _log_cuda_memory_diagnostics(log, layer_index if not is_lm_head_module else "lm_head")

        should_quantize_layer = getattr(looper.gptq_model, "should_quantize_layer", None)
        if callable(should_quantize_layer) and not should_quantize_layer(
            module,
            layer_name,
            layer_index,
            looper.gptq_model.quantize_config,
        ):
            # Excluded layers have no bundles and remain skipped on resume.
            continue

        if _is_resume_fastforward_candidate(
            is_embeddings_module=is_embeddings_module,
            layer_index=layer_index,
            resume_target=resume_target,
        ):
            if resume_target_cache_hit and layer_index < resume_target:
                _resume_restore_only_layer(
                    looper,
                    layers=layers,
                    layer_index=layer_index,
                    layer_name=layer_name,
                    layer_title=layer_title,
                    shared_kv_cache_dict=shared_kv_cache_dict,
                    pb=pb,
                    log=log,
                )
            else:
                _resume_replay_layer(
                    looper,
                    layers=layers,
                    layer_index=layer_index,
                    layer_name=layer_name,
                    layer_title=layer_title,
                    layer_count=layer_count,
                    shared_kv_cache_dict=shared_kv_cache_dict,
                    pb=pb,
                    log=log,
                    region_timer=region_timer,
                    preloaded_cache=resume_target_cache if layer_index == resume_target else None,
                )
            continue

        module = looper.gptq_model.pre_quantize(module)

        embedding_module_name = None
        if is_input_embeddings_module:
            embedding_module_name = looper.gptq_model.get_input_embeddings_name()
            layer_descriptor = embedding_module_name
        elif is_output_embeddings_module:
            embedding_module_name = looper.gptq_model.get_output_embeddings_name()
            layer_descriptor = embedding_module_name
        elif is_lm_head_module:
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

            replace_module_with_hooked_legacy(
                module,
                quant_lm_head=looper.gptq_model.quantize_config.lm_head,
                skip_module_paths=hook_skip_modules,
            )

            layers[layer_index] = module

            if layer_name:
                layer_descriptor = layer_name
            else:
                layer_descriptor = str(layer_index)

        materialize_model(module)

        cur_layer_device = get_device(module)
        if getattr(cur_layer_device, "type", None) == "meta":
            # Lazy shell layers can stay meta until a later subset stage materializes them.
            cur_layer_device = normalize_device_like(looper.gptq_model.quantize_config.device) or CPU
        full = find_modules(
            module,
            name=embedding_module_name or (looper.gptq_model.lm_head if is_lm_head_module else ""),
        )

        for p_index, processor in enumerate(looper.processors):
            # Each processor contributes a quantization phase; walk them in
            # order so their caches and side effects line up with the pipeline.
            processor.log_call_count = 0  # reset
            processor.collect_memory_info(layer_index)
            # Read the replay policy once per processor so the layer stage uses
            # one execution config instead of a group of unrelated flags.
            execution_config = processor.execution_config

            if is_input_embeddings_module:
                layer_inputs = processor.inputs_cache.src_inputs
            else:
                layer_inputs = processor.inputs_cache.layer_inputs
            if (is_output_embeddings_module or is_lm_head_module) and layer_inputs:
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
            if embed_quant_mode is not None and embed_only is not False and not is_embeddings_module:
                # Embedding-only operations replay loaded decoder blocks only
                # to propagate calibration activations to the output endpoint.
                subset_plans = []
            else:
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
                    layers_prefix=layer_name,
                    fallback=fallback,
                    embedding_module_name=embedding_module_name,
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

            is_last_module = progress_index == len(pb) - 1
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
                    is_lm_head_module=is_embeddings_module,
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
                    is_lm_head_module=is_embeddings_module,
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

                if is_embeddings_module:
                    looper.gptq_model.post_quantize(module)
                else:
                    layers[layer_index] = looper.gptq_model.post_quantize(module)

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
                                    if module_full_name not in requant_endpoint_names:
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
                                                # The on-disk bundle is the only record a
                                                # resumed run can restore from, so never
                                                # skip it for small modules.
                                                force=True,
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
                    boundary_futures.append(future)

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
                    """Consumes finalize futures, updating progress and surfacing errors.

                    Returns True only if every future completed successfully —
                    callers must not treat the layer as a durable resume point
                    otherwise (a failed future means some module was never
                    packed/offloaded even though the loop-stop is deferred).
                    """

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
                                return False

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
                    return True

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
                        finalize_ok = _drain_finalize_futures(
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
                        if finalize_ok and not is_lm_head_module:
                            # All of this layer's modules are finalized and on
                            # disk (synchronous drain above succeeded), so the
                            # layer is a durable resume point. On a failed
                            # drain the marker must NOT advance: the layer
                            # looks complete on disk except for the failed
                            # module, and resuming past it would silently keep
                            # that module's original weights.
                            write_resume_marker(
                                looper,
                                layer_index,
                                layer_count,
                                finalized_count=finalize_count,
                            )
                            # `processor.inputs_cache.layer_inputs` was just set
                            # to this layer's output above, i.e. exactly what the
                            # next layer consumes -- cache it so a future resume
                            # can skip replaying this layer's forward pass with
                            # its original weights. Also carry along
                            # `shared_kv_cache_dict[layer_index]` (e.g. this
                            # model's DSA-style indexer top-k indices) -- it's
                            # populated by this same forward and the next layer
                            # depends on it via `reuse_kv`.
                            save_activation_cache(
                                looper,
                                layer_index,
                                layer_count,
                                processor.inputs_cache.layer_inputs,
                                shared_kv_value=shared_kv_cache_dict.get(layer_index),
                            )
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

        if extensions:
            step_kind = (
                "input_embeddings" if is_input_embeddings_module else
                "output_embeddings" if is_output_embeddings_module else
                "lm_head" if is_lm_head_module else "layer"
            )
            extensions.publish(
                LoopStep(step_kind, model_layer_index, layer_name or step_kind),
                boundary_futures,
            )

        if durable_progress_logs:
            log.info(
                "StageLayer: handoff complete for layer=%s",
                layer_index if not is_lm_head_module else "lm_head",
            )

        # `layer_index - 1`'s cached output (e.g. this model's DSA-style
        # indexer top-k indices, threaded via reuse_kv) is only ever read
        # during layer_index's own forward/quantize/replay work above, which
        # has just finished. Evict it now -- otherwise shared_kv_cache_dict
        # grows by one entry per layer for the whole run and is never
        # reclaimed (harmless per-entry, but unbounded over many layers).
        shared_kv_cache_dict.pop(layer_index - 1, None)
