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
import time
from concurrent.futures import as_completed
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch
from defuser.modeling.replace_modules import materialize_model

from .. import DEBUG_ON, DEVICE_THREAD_POOL
from ..looper.awq_processor import AWQProcessor
from ..looper.named_module import NamedModule
from ..looper.paroquant_processor import ParoQuantProcessor
from ..models.base import MODULE_TREE_FLAG_ROUTED
from ..nn_modules.converter import MODULE_CONVERTER_MAP
from ..nn_modules.fused_group_forward import (
    clear_fused_group_forward_caches,
    install_fused_group_forward,
)
from ..nn_modules.hooked_linear import replace_module_with_hooked_legacy
from ..quantization.config import GcMode, QuantizeEmbed
from ..utils.device import get_device, get_device_new
from ..utils.disk_telemetry import disk_telemetry
from ..utils.device_telemetry import capture_device_telemetry
from ..utils.logger import live_renderables_suppressed, log_time_block, setup_logger
from ..utils.looper_helpers import (
    find_last_quantized_layer_index,
    normalize_device_like,
)
from ..utils.model import find_modules, get_layer_name, get_module
from ..utils.offload import offload_to_disk
from ..utils.torch import CPU, torch_empty_cache, torch_sync
from .output_replay import resolve_output_replay_execution
from .extension import LoopStep
from .stage_subset import SubsetPlan, build_layer_subset_plans, run_subset_stage

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .module_looper import ModuleLooper


def _build_pre_quantize_skip_plan(layer_modules: List[List[str]]) -> Set[str]:
    """Build the relative leaf-module names that pre_quantize should skip for batch loading.

    Note: this function assumes `layer_modules` has already had MoE expert placeholders
    (e.g. `EXPERT_INDEX_PLACEHOLDER`) expanded into concrete indices by
    `simple_layer_modules` -> `build_moe_modules_if_need` before `run_layer_stage` is
    invoked. If a future caller passes unexpanded module trees containing `#`, the skip
    names will not match real submodule paths and those leaves will fall back to the
    per-submodule materialize path instead of the grouped batch load.
    """

    skip_module_names: Set[str] = set()
    for block in layer_modules:
        for raw_name in block:
            if not raw_name:
                continue
            # layer_modules may still contain colon-delimited flags; strip them for the actual path.
            clean_name = raw_name.split(":", 1)[0]
            if not clean_name:
                continue
            skip_module_names.add(clean_name)
    return skip_module_names


def _build_pre_quantize_defer_plan(
    looper: "ModuleLooper",
    layer_modules: List[List[str]],
) -> Set[str]:
    """Return routed leaves whose weights are unnecessary during direct capture.

    GPTQ's MoE bypass invokes input hooks directly for routed gate/up leaves and
    reconstructs down inputs from the already-quantized gate/up pair. Those
    routed shells can therefore stay meta until the subset quantization batch
    loads them directly onto their assigned device. Classification comes only
    from module-tree metadata; runtime path spelling is not semantic.
    """

    if not looper.gptq_model.quantize_config.moe_routing_bypass():
        return set()
    if not looper.processors or not all(
        getattr(processor, "moe_input_capture_without_forward", False)
        for processor in looper.processors
    ):
        return set()

    deferred: Set[str] = set()
    for block in layer_modules:
        for raw_name in block:
            clean_name = raw_name.split(":", 1)[0]
            if not clean_name:
                continue
            flags = looper.gptq_model.get_module_tree_flags(clean_name)
            if MODULE_TREE_FLAG_ROUTED in flags:
                deferred.add(clean_name)
    return deferred


def _materialize_unclaimed_deferred_modules(
    looper: "ModuleLooper",
    *,
    module: torch.nn.Module,
    layer_name: str,
    defer_module_names: Set[str],
    device: torch.device,
) -> None:
    """Load deferred leaves that no subset claimed before replay/finalization.

    Dynamic exclusions and calibration-coverage pruning can remove a routed
    leaf after the initial batch-load plan deliberately left it meta. Keep the
    fast path for claimed leaves, but reconcile any remaining shells in one
    grouped LazyTurtle load so native replay and ``post_quantize`` always see a
    complete layer.
    """

    pending = []
    for relative_name in sorted(defer_module_names):
        try:
            deferred_module = module.get_submodule(relative_name)
        except AttributeError:
            continue
        has_direct_meta = any(
            tensor.device.type == "meta"
            for tensor in (
                *deferred_module.parameters(recurse=False),
                *deferred_module.buffers(recurse=False),
            )
        )
        if has_direct_meta:
            full_name = f"{layer_name}.{relative_name}" if layer_name else relative_name
            pending.append((deferred_module, full_name, device))

    if pending:
        looper.gptq_model.lazy_turtle_batch_materialize_submodules(pending)
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

    This is the lifecycle control for serial vs. parallel layer finalization.
    The default (False) lets a background watcher drain finalizer futures while
    the main loop proceeds to the next layer, so module packing and disk offload
    can overlap the next layer's work. The tree-race safety work guarantees
    that per-leaf replacement and `state_dict()` access in those workers do not
    corrupt the module tree.

    AWQ scale search consumes activations produced by already-processed
    previous layers. Letting AWQ finalizers overlap the next layer makes those
    activations timing-sensitive: a faster capture path can change whether
    replay sees dense or packed previous modules. Drain AWQ finalizers
    synchronously so cache optimizations cannot change quantization math.

    ParoQuant layer/group optimization holds substantially more live CUDA state
    than the weight-only paths. Letting its finalizers overlap the next layer
    can visibly ratchet active VRAM upward from layer N to N+1, so ParoQuant
    always drains per-layer finalizers synchronously.

    GPTQ finalization packs CPU-owned leaves and writes them to the configured
    offload target. It does not participate in the next layer's forward, so it
    may overlap that layer even when quantization uses multiple accelerators.
    Any multi-accelerator quantization flow can overlap layer N finalizers with
    layer N+1 materialization/replay if we keep the default async drain. That
    saves some wall time, but it also broadens the lifetime of device-resident
    weights, activations, and packing state across layer boundaries. In
    practice, the overlap is not worth the allocator pressure risk, so
    multi-device runs drain per-layer finalizers synchronously.

    """
    if looper.gptq_model.quantize_config.wait_for_submodule_finalizers:
        return True

    # AWQ/ParoQuant finalizers are drained synchronously for correctness/VRAM
    # reasons noted above; everything else defaults to async parallel drain.
    return any(isinstance(process, (AWQProcessor, ParoQuantProcessor)) for process, *_ in finalize_tasks)


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
    is_embeddings_module: Optional[bool] = None,
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

    if is_embeddings_module is None:
        is_embeddings_module = is_lm_head_module

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
        replay_install_device_overrides = False
    else:
        replay_batch_count = replay_plan.batch_count
        replay_row_counts = replay_plan.forward_row_counts
        replay_total_rows = replay_plan.forward_total_rows
        replay_source = (
            f"{layer_descriptor}:subset"
            f"{replay_plan.subset_index + 1}/{replay_plan.subset_total}"
        )
        replay_modules = replay_plan.modules
        quantize_config = getattr(getattr(looper, "gptq_model", None), "quantize_config", None)
        moe_execution = getattr(getattr(quantize_config, "moe", None), "execution", None)
        replay_execution = resolve_output_replay_execution(
            replay_plan,
            parallel_moe_replay=bool(getattr(moe_execution, "parallel_output_replay", True)),
        )
        replay_forward_device_map = replay_execution.forward_device_map
        replay_force_serial = replay_execution.force_serial or force_serial
        replay_preserve_module_devices = replay_execution.preserve_module_devices
        replay_install_device_overrides = replay_execution.install_device_overrides

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
    if replay_modules is not None and replay_install_device_overrides:
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
            is_embeddings_module=is_embeddings_module,
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
        cleanup_native_replay = getattr(processor, "cleanup_native_replay", None)
        if callable(cleanup_native_replay) and full is not None:
            cleanup_native_replay(full)
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
    pristine_capture = getattr(processor, "pristine_quant_input_capture", None)
    pristine_capture_context = (
        pristine_capture(layer_index=layer_index)
        if callable(pristine_capture)
        else nullcontext()
    )
    with pristine_capture_context:
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
    layer_index_offset = 1 if quant_input_embeddings else 0

    extensions = getattr(looper, "extensions", None)
    for layer_index in pb:
        if layer_index < getattr(looper, "start_step", 0):
            continue
        boundary_futures = []
        # Iterate over every transformer layer (plus lm_head when enabled) as
        # progress-bar controlled units of work.
        layer_start = time.perf_counter()
        progress_index = layer_index
        log.info(
            "StageLayer: layer lifecycle begin layer=%s progress_index=%s total=%s",
            progress_index,
            progress_index,
            len(pb),
        )

        if looper._check_loop_stop():
            break
        is_input_embeddings_module = quant_input_embeddings and progress_index == 0
        model_layer_index = progress_index - layer_index_offset
        is_output_embeddings_module = quant_output_embeddings and model_layer_index >= layer_count
        is_lm_head_module = (
            not is_output_embeddings_module
            and looper.gptq_model.quantize_config.lm_head
            and model_layer_index >= layer_count
        )
        is_embeddings_module = is_input_embeddings_module or is_output_embeddings_module or is_lm_head_module

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

        if not looper.gptq_model.should_quantize_layer(
            module,
            layer_name,
            layer_index,
            looper.gptq_model.quantize_config,
        ):
            # Excluded layers have no bundles and remain skipped on resume.
            continue

        defer_module_names: Set[str] = set()
        if is_embeddings_module or not layer_modules:
            module = looper.gptq_model.pre_quantize(module, layer_name=layer_name or "")
        else:
            skip_module_names = _build_pre_quantize_skip_plan(layer_modules)
            defer_module_names = _build_pre_quantize_defer_plan(looper, layer_modules)
            module = looper.gptq_model.pre_quantize(
                module,
                skip_module_names=skip_module_names,
                defer_module_names=defer_module_names,
                layer_name=layer_name or "",
            )

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

            fused_forward_cfg = getattr(
                looper.gptq_model.quantize_config, "fused_forward", None
            )
            if fused_forward_cfg is not None:
                install_fused_group_forward(
                    layer_module=module,
                    layer_modules_blocks=planning_layer_modules,
                    enabled=True,
                    splice=getattr(fused_forward_cfg, "splice", "view"),
                    logger=log,
                )

            layers[layer_index] = module

            if layer_name:
                layer_descriptor = layer_name
            else:
                layer_descriptor = str(layer_index)

        materialize_start = time.perf_counter()
        materialize_model(module)
        if durable_progress_logs:
            log.info(
                "StageLayer: layer=%s materialize_model took %.3fs",
                layer_index if not is_embeddings_module else "embeddings",
                time.perf_counter() - materialize_start,
            )

        cur_layer_device = get_device(module)
        if getattr(cur_layer_device, "type", None) == "meta":
            # Lazy shell layers can stay meta until a later subset stage materializes them.
            cur_layer_device = normalize_device_like(looper.gptq_model.quantize_config.device) or CPU
        full = find_modules(module, name=embedding_module_name or (looper.gptq_model.lm_head if is_lm_head_module else ""))

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
            subset_plans_start = time.perf_counter()
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
                    "StageLayer: layer=%s processor=%s subset plans built in %.3fs",
                    layer_index if not is_lm_head_module else "lm_head",
                    processor.name(),
                    time.perf_counter() - subset_plans_start,
                )
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
            is_terminal_quantized_layer = bool(
                not is_embeddings_module
                and last_quantized_layer_index is not None
                and model_layer_index == last_quantized_layer_index
                and p_index == len(looper.processors) - 1
            )
            needs_downstream_outputs = not is_last_module and not is_terminal_quantized_layer
            weight_prefetch_state: Dict[int, object] = {}
            for subset_plan_index, subset_plan in enumerate(subset_plans):
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
                    is_embeddings_module=is_embeddings_module,
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
                    next_plan=(
                        subset_plans[subset_plan_index + 1]
                        if subset_plan_index + 1 < len(subset_plans)
                        else None
                    ),
                    weight_prefetch_state=weight_prefetch_state,
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

            if defer_module_names:
                _materialize_unclaimed_deferred_modules(
                    looper,
                    module=module,
                    layer_name=layer_name,
                    defer_module_names=defer_module_names,
                    device=cur_layer_device,
                )

            # When dynamic exclusions remove every tracked module from a layer,
            # no subset stage runs, so nothing materializes that layer's
            # outputs. Processors that enable post-process forward replay
            # (`fwd_replay_after_process`) still need one forward of the untouched
            # layer so the next layer receives the correct activations.
            replay_skipped_layer = (
                needs_downstream_outputs
                and not subset_plans
                and execution_config.require_fwd
                and execution_config.fwd_replay_after_process
            )

            # Some processors consume outputs only after `process()` updates the
            # current layer. In that case, replay the layer once using the
            # metadata already computed by the final subset plan.
            replay_after_process = (
                needs_downstream_outputs
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
                    is_embeddings_module=is_embeddings_module,
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
                sync_start = time.perf_counter() if region_timer is not None else None
                torch_sync()
                if region_timer is not None and sync_start is not None:
                    region_timer.record(
                        "torch_sync",
                        time.perf_counter() - sync_start,
                        source=f"stage_layer post_process layer={layer_index}",
                    )

                # Ensure any temporary packed native-replay module is restored to the
                # dense leaf before post_quantize() finalizes/packs the layer.
                cleanup_native_replay = getattr(processor, "cleanup_native_replay", None)
                if callable(cleanup_native_replay) and full is not None:
                    cleanup_native_replay(full)

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
                sync_start = time.perf_counter() if region_timer is not None else None
                torch_sync()
                if region_timer is not None and sync_start is not None:
                    region_timer.record(
                        "torch_sync",
                        time.perf_counter() - sync_start,
                        source=f"stage_layer pre_finalize layer={layer_index}",
                    )

                # Gather finalize tasks (can offload to disk); run them via the pool
                finalize_tasks = []

                for reverse_p in reversed(looper.processors):
                    # Collect finalize tasks in reverse to mirror the processor
                    # execution order and honor downstream dependencies.
                    for module in processed_subset.values():
                        actual_module = module.module if isinstance(module, NamedModule) else module

                        if isinstance(actual_module, torch.nn.Module):
                            clear_fused_group_forward_caches(actual_module)

                        try:
                            get_device_new(
                                actual_module,
                                recursive=True,
                                assert_mode=True,
                                expected=CPU,
                            )
                        except AssertionError:
                            param_devices = [(n, str(p.device)) for n, p in actual_module.named_parameters(recurse=True)]
                            buffer_devices = [(n, str(b.device)) for n, b in actual_module.named_buffers(recurse=True)]
                            log.error(
                                f"Device assert failed for {getattr(module, 'full_name', getattr(module, 'name', '?'))}: "
                                f"actual_module type={type(actual_module).__name__} device={get_device(actual_module)} "
                                f"params={param_devices} buffers={buffer_devices}"
                            )
                            raise
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
                    """Runs processor finalization and optional disk offload for one module.

                    The processor returns the new quantized module so we can pass it
                    (and the already-known full name) to `offload_to_disk()` without
                    triggering a `model.named_modules()` scan from inside the worker.
                    """

                    resolved_label = module_label or getattr(module, "full_name", getattr(module, "name", ""))
                    module_start = time.perf_counter()
                    start = module_start if region_timer is not None else None
                    submodule_elapsed = 0.0
                    offload_elapsed = 0.0
                    try:
                        submodule_start = time.perf_counter()
                        with log_time_block(
                            "submodule_finalize",
                            logger=log,
                            module_name=resolved_label,
                        ):
                            qmodule = process.submodule_finalize(module, looper.gptq_model)

                        # Drop the original dense module reference from the wrapper
                        # so checkpoint-backed weights can be released before the
                        # next layer is materialized.
                        if isinstance(qmodule, torch.nn.Module) and isinstance(module, NamedModule):
                            module.module = qmodule

                        submodule_elapsed = time.perf_counter() - submodule_start

                        # Disk offload (lifecycle TODO note preserved)
                        if isinstance(qmodule, torch.nn.Module):
                            quant_config = getattr(looper.gptq_model, "quantize_config", None)
                            if quant_config and getattr(quant_config, "offload_to_disk", False):
                                offload_path = getattr(quant_config, "offload_to_disk_path", None)
                                if offload_path and qmodule is not None:
                                    module_full_name = getattr(module, "full_name", None)
                                    offload_start = time.perf_counter() if region_timer is not None else None
                                    with log_time_block(
                                        "disk_offload",
                                        logger=log,
                                        module_name=resolved_label,
                                    ):
                                        # Pass the quantized module and its full
                                        # name to avoid `get_module_fullname()`,
                                        # which scans `model.named_modules()` and
                                        # races with concurrent sibling updates.
                                        offload_to_disk(
                                            model=looper.gptq_model.model,
                                            module=qmodule,
                                            disk_path=offload_path,
                                            module_full_name=module_full_name,
                                            force=True,
                                        )
                                    offload_elapsed = time.perf_counter() - offload_start
                                    if region_timer is not None and offload_start is not None:
                                        region_timer.record(
                                            "submodule_finalize_offload",
                                            offload_elapsed,
                                            source=resolved_label,
                                        )
                                elif not offload_path:
                                    log.warning(
                                        "Skipping disk offload for %s: no offload path configured",
                                        module_label,
                                    )
                    finally:
                        total_elapsed = time.perf_counter() - module_start
                        if idx == 1 or idx == total or idx % 50 == 0 or total_elapsed > 1.0 or submodule_elapsed > 0.5 or offload_elapsed > 0.5:
                            log.info(
                                "StageLayer: finalized %s (%d/%d) in %.3fs (submodule_finalize %.3fs, offload %.3fs)",
                                resolved_label,
                                idx,
                                total,
                                total_elapsed,
                                submodule_elapsed,
                                offload_elapsed,
                            )
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
                    # Schedule finalize work on the device thread pool so CPU- and
                    # I/O-bound pack/offload work can run in parallel. Each worker
                    # only mutates one leaf under its direct parent lock, and the
                    # whole-tree scans have been removed from the finalize path.
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
                    drain_label,
                ):
                    """Consumes finalize futures, updating progress and surfacing errors.

                    Returns True only if every future completed successfully —
                    callers must not treat the layer as a durable resume point
                    otherwise (a failed future means some module was never
                    packed/offloaded even though the loop-stop is deferred).
                    """

                    drain_start = time.perf_counter()
                    completed_local = 0
                    progress_interval = max(1, finalize_count_local // 10)
                    try:
                        log.info(
                            "StageLayer: %s finalize drain started for %d module(s)",
                            drain_label,
                            finalize_count_local,
                        )
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

                            if completed_local % progress_interval == 0:
                                elapsed = time.perf_counter() - drain_start
                                log.info(
                                    "StageLayer: %s finalize progress %d/%d (%.1f%%) elapsed %.3fs",
                                    drain_label,
                                    completed_local,
                                    finalize_count_local,
                                    100.0 * completed_local / finalize_count_local,
                                    elapsed,
                                )
                    finally:
                        elapsed = time.perf_counter() - drain_start
                        log.info(
                            "StageLayer: %s finalize drain completed %d/%d in %.3fs",
                            drain_label,
                            completed_local,
                            finalize_count_local,
                            elapsed,
                        )
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
                    drain_label = f"layer {layer_index if not is_lm_head_module else 'lm_head'}"
                    if drain_sync:
                        # Synchronous: wait for all finalization to complete before proceeding to next layer
                        # This ensures all packing and writing tasks are done
                        _drain_finalize_futures(
                            [future for future, *_ in finalize_futures_snapshot],
                            finalize_pb,
                            finalize_count,
                            layer_index,
                            drain_label,
                        )
                        if region_timer is not None:
                            region_timer.flush()
                        if looper.gptq_model.quantize_config.gc_mode == GcMode.ON_STAGE_END:
                            torch_empty_cache(device=cur_layer_device, sync=True)
                        elif _should_empty_cache_after_sync_finalize(
                            looper,
                            finalize_tasks=finalize_tasks,
                        ):
                            torch_empty_cache(device=cur_layer_device, gc=False, sync=True)
                    else:
                        # Queue the watcher behind already-submitted CPU pack
                        # tasks on the process-wide threadx pool. This lets the
                        # next layer start without creating an ad-hoc lifecycle
                        # thread, while the pool's final wait still accounts for
                        # both pack work and watcher completion.
                        DEVICE_THREAD_POOL.submit(
                            CPU,
                            capture_device_telemetry(_drain_finalize_futures),
                            [future for future, *_ in finalize_futures_snapshot],
                            finalize_pb,
                            finalize_count,
                            layer_index,
                            drain_label,
                        )
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

        disk_telemetry.log_summary(
            log,
            label=f"layer {layer_index if not is_lm_head_module else 'lm_head'} handoff",
        )

        # Keep per-shard safetensors handles open across layers. Re-opening and
        # re-registering the host mmap for every layer is more expensive than the
        # file descriptors, and LazyTurtle cleans them up on destruction.

        layer_elapsed = time.perf_counter() - layer_start
        layer_label = f"layer {layer_index if not is_lm_head_module else 'lm_head'}"
        log.info(
            "StageLayer: layer lifecycle end %s wall_clock=%.3fs",
            layer_label,
            layer_elapsed,
        )
        if region_timer is not None:
            region_timer.flush_period(label=layer_label)
        # `layer_index - 1`'s cached output (e.g. this model's DSA-style
        # indexer top-k indices, threaded via reuse_kv) is only ever read
        # during layer_index's own forward/quantize/replay work above, which
        # has just finished. Evict it now -- otherwise shared_kv_cache_dict
        # grows by one entry per layer for the whole run and is never
        # reclaimed (harmless per-entry, but unbounded over many layers).
        shared_kv_cache_dict.pop(layer_index - 1, None)
