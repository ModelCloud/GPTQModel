# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Subset-level planning and execution for forward replay and quantization.

This module is intentionally split into two responsibilities:
- `build_subset_plan()` / `build_layer_subset_plans()` decide what should happen
- `run_subset_stage()` / `_run_single_subset_pass()` execute that decision

The goal is to keep planning branches out of the hot execution path so replay,
MoE chunking, coverage handling, and device routing are easier to reason about.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple

import pcre
import torch

from .awq_processor import AWQProcessor
from .paroquant_processor import ParoQuantProcessor
from .qqq_processor import QQQProcessor
from .. import DEBUG_ON, DEVICE_THREAD_POOL
from ..looper.gptq_processor import GPTQProcessor
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models._const import META
from ..quantization.config import VramStrategy, GcMode, ExpertsRoutingBypass
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.looper_helpers import normalize_device_like, select_forward_devices
from ..utils.torch import torch_empty_cache, torch_sync

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .module_looper import ModuleLooper


ForwardMode = Literal["parallel", "serial"]


@dataclass
class CalibrationCoveragePolicy:
    """Describe how calibration coverage gaps are handled for one subset.

    Some quantizers need every module to observe routed calibration traffic.
    When a MoE expert never fires, we either keep it alive because fallback is
    enabled, or prune it from quantization and record a dynamic exclusion.
    """

    # Whether the processor should validate that calibration reached each module.
    validate_input_coverage: bool
    # Whether uncovered modules are allowed to remain because fallback can take over.
    fallback_enabled: bool
    # Whether uncovered modules are removed from the quantization worklist.
    prune_uncovered_modules: bool
    # Whether uncovered modules should be recorded in qcfg.dynamic.
    record_dynamic_exclusions: bool


@dataclass
class SubsetPlan:
    """Freeze all subset execution decisions before forward or quant work begins.

    The plan answers:
    - which modules belong to this subset
    - whether forward runs at all
    - whether the layer needs a post-process replay
    - whether forward is serial or parallel
    - how MoE groups are arranged for scheduling
    - which modules are pinned to specific forward devices
    - how uncovered calibration modules are handled
    """

    modules: Dict[str, NamedModule]
    subset_index: int
    subset_total: int
    execute_forward: bool
    replay_after_process: bool
    forward_mode: ForwardMode
    batch_count: int
    forward_row_counts: List[int]
    forward_total_rows: int
    moe_groups: Dict[str, List[str]]
    forward_device_map: Dict[str, torch.device]
    calibration_coverage_policy: CalibrationCoveragePolicy
    module_chunks: List[Dict[str, NamedModule]]

    @property
    def subset_forward_serial(self) -> bool:
        """Whether the forward executor should stay on one device."""

        return self.forward_mode == "serial"

    @property
    def need_forward_outputs(self) -> bool:
        """Whether this subset consumes forward outputs before process()."""

        return self.execute_forward and not self.replay_after_process

    @property
    def batching_enabled(self) -> bool:
        """Whether the subset will execute as multiple forward/quant chunks."""

        return len(self.module_chunks) > 1

    @property
    def preserve_module_devices(self) -> bool:
        """Whether per-module forward device overrides are active."""

        return bool(self.forward_device_map)

    def for_modules(self, modules: Dict[str, NamedModule]) -> "SubsetPlan":
        """Reuse the same execution policy for one chunk or replay-only subset."""

        return replace(self, modules=modules, module_chunks=[modules])


@dataclass
class SubsetStageResult:
    """Returns processed modules plus the updated layer input cache for a subset."""

    processed_subset: Dict[str, NamedModule]
    layer_inputs: List[List[torch.Tensor]]
    plan: Optional[SubsetPlan]


def _resolve_cache_flush_device(
    cur_layer_device: Optional[torch.device],
    used_devices,
) -> Optional[torch.device]:
    """Keep cache flush local unless the preceding work fanned out across devices."""

    current = normalize_device_like(cur_layer_device)
    if current is None:
        return None

    accelerator_devices = set()
    for device in used_devices:
        normalized = normalize_device_like(device)
        if normalized is None or normalized.type == "cpu":
            continue
        accelerator_devices.add(str(normalized))

    if not accelerator_devices:
        return current
    if accelerator_devices == {str(current)}:
        return current
    return None


def _resolve_forward_flush_device(
    plan: SubsetPlan,
    cur_layer_device: Optional[torch.device],
) -> Optional[torch.device]:
    used_devices = list(plan.forward_device_map.values())

    if not plan.subset_forward_serial:
        selected_devices = select_forward_devices(cur_layer_device)
        active_forward_devices = {
            str(normalize_device_like(device))
            for device in selected_devices
            if normalize_device_like(device) is not None and normalize_device_like(device).type != "cpu"
        }
        if len(active_forward_devices) > 1:
            used_devices.extend(selected_devices)

    return _resolve_cache_flush_device(cur_layer_device, used_devices)


def _resolve_quant_flush_device(
    cur_layer_device: Optional[torch.device],
    quant_target_devices: Dict[str, torch.device],
) -> Optional[torch.device]:
    return _resolve_cache_flush_device(cur_layer_device, quant_target_devices.values())


def _resolve_subset_calibration_coverage_policy(
    processor: LoopProcessor,
    fallback,
) -> CalibrationCoveragePolicy:
    """Resolve how this subset handles modules that never receive calibration traffic."""

    validate_input_coverage = isinstance(processor, (GPTQProcessor, QQQProcessor, AWQProcessor, ParoQuantProcessor))
    fallback_enabled = fallback is not None
    prune_uncovered_modules = validate_input_coverage and not fallback_enabled

    return CalibrationCoveragePolicy(
        validate_input_coverage=validate_input_coverage,
        fallback_enabled=fallback_enabled,
        prune_uncovered_modules=prune_uncovered_modules,
        record_dynamic_exclusions=prune_uncovered_modules,
    )


def _collect_subset_forward_progress(
    looper: "ModuleLooper",
    processor: LoopProcessor,
    layer_inputs: List[List[torch.Tensor]],
    *,
    execute_forward: bool,
) -> Tuple[int, List[int], int]:
    """Normalize batch and row progress for the subset's forward execution."""

    if not execute_forward:
        return 0, [], 1

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


def _resolve_forward_baseline_devices(
    subset: Dict[str, NamedModule],
    full,
) -> Dict[str, torch.device]:
    """Capture the baseline device for each layer module before subset execution mutates it."""

    candidates: Dict[str, object] = {}
    if full:
        candidates.update(full)
    for name, named_module in subset.items():
        candidates.setdefault(name, named_module)

    baseline: Dict[str, torch.device] = {}
    fallback_device: Optional[torch.device] = None
    for name, module_ref in candidates.items():
        actual_module = module_ref.module if isinstance(module_ref, NamedModule) else module_ref
        try:
            device = get_device(actual_module)
        except Exception:
            device = None
        if device is not None and device != META and fallback_device is None:
            fallback_device = device
        baseline[name] = device

    if fallback_device is None:
        return {}

    resolved: Dict[str, torch.device] = {}
    for name, device in baseline.items():
        if device is None or device == META:
            device = fallback_device
        if device is not None and device != META:
            resolved[name] = device

    return resolved


def build_subset_plan(
    looper: "ModuleLooper",
    *,
    processor: LoopProcessor,
    subset: Dict[str, NamedModule],
    subset_index: int,
    subset_total: int,
    full,
    fallback,
    layer_inputs: List[List[torch.Tensor]],
) -> SubsetPlan:
    """Plan subset execution before any hooks, forwards, or quant work begin.

    The returned plan is the single source of truth for:
    - whether this subset runs forward at all
    - whether replay happens later in the layer stage
    - whether forward stays serial or can fan out
    - whether modules are chunked for staged MoE execution
    - how uncovered calibration modules are handled
    """

    execution_config = processor.execution_config
    calibration_coverage_policy = _resolve_subset_calibration_coverage_policy(processor, fallback)

    moe_groups: Dict[str, List[str]] = {}
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
    is_moe_subset = len(moe_module_names) >= looper._moe_subset_threshold
    moe_modules_set = set(moe_module_names)

    if is_moe_subset:
        combined_names: List[str] = list(subset.keys())
        if full is not None:
            for candidate in full.keys():
                if candidate not in subset:
                    combined_names.append(candidate)

        for module_name in combined_names:
            # Group experts across the full MoE family so device placement is
            # consistent even when the current subset only contains one slice.
            group_key = looper._extract_moe_group_key(module_name)
            if group_key is None:
                continue
            moe_groups.setdefault(group_key, []).append(module_name)

        for name, named_module in subset.items():
            setattr(named_module, "moe_enabled", name in moe_modules_set)

        if looper._vram_strategy == VramStrategy.BALANCED:
            devices = [
                dev for dev in looper._quant_devices
                if dev is not None and getattr(dev, "type", None) != "cpu"
            ]
            if len(devices) > 1 and moe_groups:
                assignable_group_keys: List[str] = []
                for group_key, module_names in moe_groups.items():
                    suffixes = {name.rsplit(".", 1)[-1] for name in module_names}
                    # Some MoE families route pairs like gate/up or w1/w3 together.
                    if {"gate_proj", "up_proj"}.issubset(suffixes) or {"w1", "w3"}.issubset(suffixes):
                        assignable_group_keys.append(group_key)

                if assignable_group_keys:
                    groups_per_device = max(
                        math.ceil(len(assignable_group_keys) / len(devices)), 1
                    )
                    for group_index, group_key in enumerate(assignable_group_keys):
                        device_idx = min(group_index // groups_per_device, len(devices) - 1)
                        target_device = devices[device_idx]
                        for module_name in moe_groups[group_key]:
                            forward_device_map[module_name] = target_device

                    # Once balanced MoE routing pins any expert family to explicit
                    # devices, the rest of the layer must also be anchored to the
                    # baseline layer placement. Otherwise untouched modules can
                    # inherit stale round-robin quant devices from earlier subsets.
                    baseline_devices = _resolve_forward_baseline_devices(
                        subset=subset,
                        full=full,
                    )
                    for module_name, baseline_device in baseline_devices.items():
                        forward_device_map.setdefault(module_name, baseline_device)

        # Balanced MoE subsets stay serial so replica fan-out does not fight the
        # explicit per-expert device assignment planned above.
        subset_forward_serial = looper._vram_strategy == VramStrategy.BALANCED
        if subset_forward_serial:
            active_group_count = len(moe_groups)
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

    # Forward progress is normalized here so the executor and any later replay
    # reuse the same batch and row accounting instead of recomputing it.
    execute_forward = execution_config.require_fwd
    batch_count, forward_row_counts, forward_total_rows = _collect_subset_forward_progress(
        looper,
        processor,
        layer_inputs,
        execute_forward=execute_forward,
    )

    # ExpertsRoutingBypass is the only routing mode that exposes a deterministic
    # module chunk size for staged MoE execution.
    moe_routing = looper.gptq_model.quantize_config.moe
    batch_size = None
    if moe_routing is not None and isinstance(moe_routing.routing, ExpertsRoutingBypass):
        batch_size = moe_routing.routing.batch_size

    module_chunks = [subset]
    if is_moe_subset and batch_size is not None and batch_size > 0 and execute_forward:
        sorted_module_names = sorted(subset.keys())
        module_chunks = [
            {name: subset[name] for name in sorted_module_names[start:start + batch_size]}
            for start in range(0, len(sorted_module_names), batch_size)
        ]

    return SubsetPlan(
        modules=subset,
        subset_index=subset_index,
        subset_total=subset_total,
        execute_forward=execute_forward,
        replay_after_process=execute_forward and execution_config.fwd_replay_after_process,
        forward_mode="serial" if subset_forward_serial else "parallel",
        batch_count=batch_count,
        forward_row_counts=forward_row_counts,
        forward_total_rows=forward_total_rows,
        moe_groups=moe_groups,
        forward_device_map=forward_device_map,
        calibration_coverage_policy=calibration_coverage_policy,
        module_chunks=module_chunks,
    )


def build_layer_subset_plans(
    looper: "ModuleLooper",
    *,
    processor: LoopProcessor,
    module: torch.nn.Module,
    layer_modules: List[List[str]],
    layer_inputs: List[List[torch.Tensor]],
    full,
    is_lm_head_module: bool,
    layer_index: int,
    layers_prefix: Optional[str],
    fallback,
) -> List[SubsetPlan]:
    """Build every subset plan for one processor before layer execution starts."""

    execution_config = processor.execution_config
    module_name_groups = [[looper.gptq_model.lm_head]] if is_lm_head_module else layer_modules

    if execution_config.fwd_all_modules_in_single_pass:
        # Native-style processors consume one merged replay over the whole layer.
        # Build one plan up front so the layer stage does not keep re-deriving
        # merged subset state while it is also coordinating execution.
        module_name_groups = [sum(module_name_groups, [])]

    subsets: List[Dict[str, NamedModule]] = []
    for names in module_name_groups:
        subset = looper.create_named_modules(
            module,
            full,
            is_lm_head_module,
            layer_index,
            layers_prefix,
            names,
            processor,
            fallback,
            layer_module=module,
        )
        # Skip empty subsets caused by per-layer structure differences or dynamic
        # exclusions so execution only sees real work.
        if subset:
            subsets.append(subset)

    subset_total = len(subsets)
    return [
        build_subset_plan(
            looper,
            processor=processor,
            subset=subset,
            subset_index=index,
            subset_total=subset_total,
            full=full,
            fallback=fallback,
            layer_inputs=layer_inputs,
        )
        for index, subset in enumerate(subsets)
    ]


def _run_single_subset_pass(
    looper: 'ModuleLooper',
    processor: LoopProcessor,
    module: torch.nn.Module,
    plan: SubsetPlan,
    layer_inputs: List[List[torch.Tensor]],
    layer_input_kwargs: List[Dict[str, torch.Tensor]],
    position_ids: List[torch.Tensor],
    attention_masks: List[torch.Tensor],
    cur_layer_device: torch.device,
    is_lm_head_module: bool,
    layer_descriptor: str,
    layer_title: str,
    layer_index: int,
    full,
    fallback,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    pb,
    logger,
    is_awq_processor: bool,
    region_timer=None,
    previous_processed_subset: Optional[Dict[str, NamedModule]] = None,
    subset_event_cb: Optional[Callable[..., None]] = None,
    return_outputs: bool = False,
    disable_moe_hooks: bool = False,
    execute_forward: Optional[bool] = None,
) -> Tuple[Dict[str, NamedModule], Optional[List[List[torch.Tensor]]], bool]:
    """Execute forward and quantization for a specific subset/chunk.

    This function assumes planning is already done. Apart from the optional
    `execute_forward` override used by replay-only and quant-only paths, it
    should consume the plan rather than re-derive execution mode.
    """

    # Pull frequently used plan fields into locals so the execution flow below
    # reads linearly without re-deriving policy from processor state.
    subset = plan.modules
    subset_index = plan.subset_index
    subset_total = plan.subset_total
    execution_config = processor.execution_config
    calibration_coverage_policy = plan.calibration_coverage_policy
    forward_row_counts = plan.forward_row_counts
    batch_count = plan.batch_count
    forward_device_map = plan.forward_device_map
    execute_forward = plan.execute_forward if execute_forward is None else execute_forward

    handle = []
    subset_size = len(subset)

    if execute_forward:
        for named_module in subset.values():
            if isinstance(named_module, NamedModule):
                looper._prepare_named_module_for_forward(
                    named_module=named_module,
                    fallback_device=cur_layer_device,
                )

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

    if execute_forward:
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
                enable_stop = (
                    execution_config.fwd_replay_after_process
                    or execution_config.subset_forward_early_stop
                )
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

    capture_layer_forward_context = execute_forward and execution_config.capture_layer_forward_context
    if capture_layer_forward_context:
        subset_capture_override = getattr(processor, "capture_layer_forward_context_during_subset", None)
        if callable(subset_capture_override):
            capture_layer_forward_context = bool(subset_capture_override())
    need_outputs = execute_forward and (plan.need_forward_outputs or capture_layer_forward_context)
    fwd_start = None
    forward_source = f"{layer_descriptor}:subset{subset_index + 1}/{subset_total}"
    if execute_forward:
        if subset_event_cb:
            subset_event_cb(stage="forward_start", layer_idx=layer_index, subset_index=subset_index, subset_total=subset_total, module_names=list(subset.keys()), processor=getattr(processor, "name", type(processor).__name__))

        fwd_start = time.perf_counter()
        reuse_kv = bool(getattr(module, "reuse_kv", False))
        forward_msg = (
            "Forward: "
            f"Layer=`{layer_descriptor}`, subset={subset_index + 1}/{subset_total}, "
            f"batches={batch_count}"
        )
        forward_pb = (
            logger.pb(range(plan.forward_total_rows))
               .manual()
               .set(show_left_steps=False)
        )
        forward_pb.title(forward_msg).subtitle(
            f"Row 0/{plan.forward_total_rows}"
        ).draw()

    previous_forward_devices: Dict[str, torch.device] = {}
    preserve_devices = plan.preserve_module_devices
    if forward_device_map:
        previous_forward_devices = looper._apply_forward_device_overrides(
            subset,
            forward_device_map,
            fallback_modules=full,
        )

    forward_outputs = None
    if execute_forward:
        try:
            # MoE lifecycle hooks need to know which subset is currently active.
            # Replay-only passes can disable that when they only need outputs.
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
                progress_total_rows=plan.forward_total_rows,
                force_serial=plan.subset_forward_serial,
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
    if execute_forward and capture_layer_forward_context:
        processor.receive_layer_forward_context(
            layer_index=layer_index,
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            layer_outputs=forward_outputs,
            subset_index=subset_index,
            subset_total=subset_total,
        )

    if execute_forward and plan.need_forward_outputs:
        # For pre-process consumers, the next stage needs the forward outputs
        # immediately rather than after the later layer replay step.
        processor.receive_layer_inputs(forward_outputs)
        if return_outputs:
             returned_outputs = processor.inputs_cache.layer_inputs
        del forward_outputs

    if execute_forward and subset_event_cb:
        subset_event_cb(stage="forward_end", layer_idx=layer_index, subset_index=subset_index, subset_total=subset_total, module_names=list(subset.keys()), processor=getattr(processor, "name", type(processor).__name__))

    fwd_time = (time.perf_counter() - fwd_start) if fwd_start is not None else 0.0
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

    if execute_forward:
        for name in subset:
            # Reset inline hook attributes on NamedModule wrappers so future passes
            # do not reuse state from this subset run.
            if hasattr(subset[name], 'forward_hook'):
                subset[name].forward_hook = None
                subset[name].forward_hook_last = False

    forward_flush_device = _resolve_forward_flush_device(plan, cur_layer_device)
    if looper.gptq_model.quantize_config.gc_mode == GcMode.ON_STAGE_END:
        torch_empty_cache(device=forward_flush_device, sync=True)
    moe_skip_modules = []
    if calibration_coverage_policy.validate_input_coverage:
        # Coverage validation is a policy decision captured by the plan.
        # The executor only applies that policy; it does not decide when the
        # processor should tolerate or prune never-invoked modules.
        for name in subset:
            # Skip MoE experts that never fired; they likely lacked calibration
            # traffic and would produce invalid statistics.
            if not processor.has_captured_input_ids(name):
                # only log for moe if `fallback` is not enabled
                if not calibration_coverage_policy.fallback_enabled:
                    logger.error(
                        f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it. "
                        f"Please enable and use `fallback` config option."
                    )
                moe_skip_modules.append(name)

        if calibration_coverage_policy.prune_uncovered_modules:
            for name in moe_skip_modules:
                skipped_module = subset.pop(name)
                task_map = getattr(processor, "tasks", None)
                if task_map is not None:
                    task_map.pop(name, None)

                # No calibration data was routed to these MoE expert modules.
                # We skip quantization them and record them in `qcfg.dynamic` as dynamically excluded modules.
                if calibration_coverage_policy.record_dynamic_exclusions:
                    if processor.qcfg.dynamic is None:
                        processor.qcfg.dynamic = {}
                    processor.qcfg.dynamic[
                        f"-:{pcre.escape(skipped_module.full_name)}"
                    ] = {}

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
            if target_device == META:
                target_device = cur_layer_device
            setattr(named_module, "target_device", target_device)
            setattr(named_module.module, "target_device", target_device)

        quant_target_devices[name] = target_device

    quant_flush_device = _resolve_quant_flush_device(cur_layer_device, quant_target_devices)

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
        """Runs `processor.process()` for one module on the device worker pool."""

        module_label = getattr(nm, "full_name", getattr(nm, "name", repr(nm)))
        proc_name = proc.name() if hasattr(proc, "name") else type(proc).__name__
        module_ref = nm.module if isinstance(nm, NamedModule) else nm
        module_weight = getattr(module_ref, "weight", None)
        if module_weight is not None and expected_device is not None:
            target_device = expected_device if isinstance(expected_device, torch.device) else torch.device(expected_device)
            actual_device = get_device(module_weight)
            assert actual_device == META or actual_device == target_device, (
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
    if looper.gptq_model.quantize_config.gc_mode == GcMode.ON_STAGE_END:
        torch_empty_cache(device=quant_flush_device, sync=True)
    else:
        torch_sync()

    if subset_event_cb:
        subset_event_cb(stage="quant_complete", layer_idx=layer_index, subset_index=subset_index, subset_total=subset_total, module_names=list(subset.keys()), processor=getattr(processor, "name", type(processor).__name__))

    used_data_parallel = False
    if execute_forward and forward_flush_device is None:
        used_data_parallel = True
    if quant_target_devices and quant_flush_device is None:
        used_data_parallel = True

    return processed_subset, returned_outputs, used_data_parallel


def run_subset_stage(
    looper: 'ModuleLooper',
    *,
    plan: SubsetPlan,
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
    full,
    fallback: bool,
    shared_kv_cache_dict: Dict[int, torch.Tensor],
    pb,
    log=None,
    region_timer=None,
    previous_processed_subset: Optional[Dict[str, NamedModule]] = None,
    subset_event_cb: Optional[Callable[..., None]] = None,
) -> SubsetStageResult:
    """Process one subset using a precomputed plan.

    The stage has three execution shapes:
    - chunked MoE execution driven by `plan.module_chunks`
    - one forward + quant pass for normal subsets
    - quant-only execution for processors that do not need forward replay
    """
    logger = log or setup_logger()

    processor_name = processor.name() if hasattr(processor, "name") else type(processor).__name__
    processor_name_lower = processor_name.lower()
    is_awq_processor = processor_name_lower.startswith("awq")

    def emit_subset_event(stage: str) -> None:
        """Emits a normalized subset lifecycle callback when one is registered."""

        if subset_event_cb is None:
            return
        subset_event_cb(
            stage=stage,
            layer_idx=layer_index,
            subset_index=plan.subset_index,
            subset_total=plan.subset_total,
            module_names=list(plan.modules.keys()),
            processor=processor_name,
        )

    if DEBUG_ON and logger.isEnabledFor(logging.DEBUG):
        if is_awq_processor:
            logger.debug(
                "StageSubset[awq]: layer=%s subset=%s/%s modules=%s sample=%s",
                layer_index,
                plan.subset_index + 1,
                plan.subset_total,
                len(plan.modules),
                list(plan.modules.keys())[:8],
            )
        else:
            logger.debug(
                "StageSubset: layer=%s subset=%s/%s processor=%s created %s modules (sample=%s)",
                layer_index,
                plan.subset_index + 1,
                plan.subset_total,
                processor_name,
                len(plan.modules),
                list(plan.modules.keys())[:8],
            )
    processed_results = {}

    # Keep the helper callsite compact while still passing the fully resolved
    # execution context into every chunk or single-pass invocation.
    common_args = dict(
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
        full=full,
        fallback=fallback,
        shared_kv_cache_dict=shared_kv_cache_dict,
        pb=pb,
        logger=logger,
        is_awq_processor=is_awq_processor,
        region_timer=region_timer,
        previous_processed_subset=previous_processed_subset,
    )

    # Once a plan exists, subset execution is just a dispatch over the plan's
    # shape rather than another round of subset analysis.
    if plan.batching_enabled:
        if DEBUG_ON and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "MoE Expert Batching Enabled: Processing %s modules in %s batches.",
                len(plan.modules),
                len(plan.module_chunks),
            )

        # Create progress bar for MOE chunks
        moe_chunk_pb = logger.pb(range(len(plan.module_chunks))).manual()
        moe_chunk_pb.title(f"MoE Chunk")

        for chunk_idx in moe_chunk_pb:
            chunk_plan = plan.for_modules(plan.module_chunks[chunk_idx])

            moe_chunk_pb.subtitle(f"({len(chunk_plan.modules)} modules)").draw()
            if DEBUG_ON and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Processing MoE Chunk %s/%s (%s modules)...",
                    chunk_idx + 1,
                    len(plan.module_chunks),
                    len(chunk_plan.modules),
                )

            chunk_result, _, chunk_used_data_parallel = _run_single_subset_pass(
                **common_args,
                plan=chunk_plan,
                subset_event_cb=None,
                return_outputs=False,
            )
            processed_results.update(chunk_result)

            # Force cleanup between chunks
            if looper.gptq_model.quantize_config.gc_mode == GcMode.ON_STAGE_END:
                flush_device = None if chunk_used_data_parallel else cur_layer_device
                torch_empty_cache(device=flush_device)

        # Close MOE chunks progress bar
        moe_chunk_pb.close()

        # Chunked execution does not produce a single coherent next-layer input
        # stream while chunks are being processed. When replay is not deferred
        # to the layer stage, the subset stage must do one final replay here to
        # rebuild the real layer outputs.
        if not plan.replay_after_process:
             replay_plan = plan.for_modules({})
             _, new_layer_inputs, _ = _run_single_subset_pass(
                 **common_args,
                 plan=replay_plan,  # Empty modules prevent quant hooks during replay.
                 subset_event_cb=None,
                 return_outputs=True,
                 disable_moe_hooks=True,
             )
             if new_layer_inputs is not None:
                 layer_inputs = new_layer_inputs

    elif plan.execute_forward:
        # Single pass
        processed_results, new_layer_inputs, _ = _run_single_subset_pass(
            **common_args,
            plan=plan,
            subset_event_cb=subset_event_cb,
            return_outputs=True,
        )
        if new_layer_inputs is not None:
             layer_inputs = new_layer_inputs
    else:
        # No forward required; still run process() for each module.
        if DEBUG_ON:
            logger.debug(
                "StageSubset: processor=%s layer=%s subset=%s/%s skipping forward (require_fwd=False)",
                processor_name,
                layer_index,
                plan.subset_index + 1,
                plan.subset_total,
            )
        emit_subset_event("forward_start")
        emit_subset_event("forward_end")
        processed_results, _, _ = _run_single_subset_pass(
            **common_args,
            plan=plan,
            subset_event_cb=subset_event_cb,
            return_outputs=False,
            execute_forward=False,
        )

    return SubsetStageResult(
        processed_subset=processed_results,
        layer_inputs=layer_inputs,
        plan=plan,
    )
