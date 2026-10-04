# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import threading
import time
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import CPU
from ..models.shared_input import SharedInputPlan
from ..models.writer import (
    PROCESS_LOG_FWD_TIME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    PROCESS_USED_MEMORY,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from ..nn_modules.qlinear.torch import TorchQuantEmbeddings
from ..nn_modules.qlinear.w4a_boundary import layer_mlp_policy
from ..quantization import FOEM, GPTAQ, GPTQ
from ..quantization.activation_floatx import (
    NVFP4ActivationHeadroom,
    fp8_token_qdq,
    normalize_nvfp4_recipe,
    nvfp4_block_qdq,
    nvfp4_global_scale,
    nvfp4_uses_headroom,
)
from ..quantization.config import METHOD, FOEMConfig, GPTAQConfig, HessianConfig, QuantizeConfig, resolve_quant_format
from ..quantization.gptq import SharedHessianFactorCache
from ..utils.device import get_device
from ..utils.fallback import normalize_fallback
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import create_quant_module, pack_module
from ..utils.module_locks import parent_module_lock
from ..utils.torch import HAS_NPU


log = setup_logger()
lock = threading.Lock()


def snapshot_eora_reconstructed_weight(weight: torch.Tensor) -> torch.Tensor:
    """Keep EoRA's reconstructed weight independent from module rematerialization.

    Sequential processors can reload the dense checkpoint into the module's
    parameter storage before EoRA runs. A plain tensor alias would then stop
    containing GPTQ's reconstruction and make the base packer derive different
    logical codes. The CPU copy also avoids retaining a second full weight on
    the quantization device.
    """

    return weight.detach().to(device=CPU, copy=True)


def enable_w4afp8_replay(linear: torch.nn.Module) -> None:
    """Apply input FP8 QDQ after the target GPTQ weight has been solved.

    The instance is replaced by a packed qlinear during finalization. Its
    original forward is left intact until then so Hessian capture sees the
    unquantized input of the current target.
    """
    if not isinstance(linear, torch.nn.Linear):
        raise NotImplementedError("W4AFP8 replay currently supports torch.nn.Linear targets only.")
    if getattr(linear, "_w4afp8_replay_enabled", False):
        return
    original_forward = linear.forward

    def quantized_input_forward(self, x):
        # Version 2 replay installs a pre-hook before GPTQ Hessian capture.
        # Applying another QDQ here would round the same input twice.
        return original_forward(x if getattr(self, "_w4a_stream_replay_pre_hook", False) else fp8_token_qdq(x))

    linear.forward = MethodType(quantized_input_forward, linear)
    linear._w4afp8_replay_enabled = True


def enable_w4a_nvfp4_replay(linear: torch.nn.Module, global_scale: float,
                             recipe: str = "least_squares") -> None:
    """Replay a solved target's input with calibrated NVFP4 block rounding."""
    if not isinstance(linear, torch.nn.Linear):
        raise NotImplementedError("W4A NVFP4 replay currently supports torch.nn.Linear targets only.")
    if getattr(linear, "_w4a_nvfp4_replay_enabled", False):
        return
    original_forward = linear.forward

    def quantized_input_forward(self, x):
        return original_forward(
            x if getattr(self, "_w4a_stream_replay_pre_hook", False)
            else nvfp4_block_qdq(x, global_scale, recipe)
        )

    linear.forward = MethodType(quantized_input_forward, linear)
    linear._w4a_nvfp4_replay_enabled = True


def clone_gptq_config_for_module(
    qcfg: QuantizeConfig,
    module_full_name: str,
    *,
    fallback=None,
) -> Optional[QuantizeConfig]:
    """Clones and applies per-module GPTQ dynamic overrides, or skips the module."""

    # entire module is skipped
    if qcfg.dynamic_get(layer_name=module_full_name) is False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)

    # dynamic overrides
    if qcfg.dynamic is not None:
        qcfg_clone.bits = qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits)
        qcfg_clone.sym = qcfg.dynamic_get(module_full_name, "sym", qcfg_clone.sym)
        qcfg_clone.mse = qcfg.dynamic_get(module_full_name, "mse", qcfg_clone.mse)

        qcfg_clone.group_size = qcfg.dynamic_get(module_full_name, "group_size", qcfg_clone.group_size)
        desc_act_override = qcfg.dynamic_get(module_full_name, "desc_act", None)
        if desc_act_override is not None:
            qcfg_clone.desc_act = desc_act_override
        act_group_aware_override = qcfg.dynamic_get(module_full_name, "act_group_aware", None)
        if act_group_aware_override is not None:
            qcfg_clone.act_group_aware = act_group_aware_override
        qcfg_clone.damp_percent = qcfg.dynamic_get(module_full_name, "damp_percent", qcfg_clone.damp_percent)
        qcfg_clone.static_groups = qcfg.dynamic_get(module_full_name, "static_groups", qcfg_clone.static_groups)
        fallback_override = qcfg.dynamic_get(module_full_name, "fallback", None)
        if fallback_override is not None:
            qcfg_clone.fallback = normalize_fallback(fallback_override, qcfg_clone.fallback)
        hessian_override = qcfg.dynamic_get(module_full_name, "hessian", None)
        if hessian_override is not None:
            if isinstance(hessian_override, dict):
                qcfg_clone.hessian = HessianConfig(**hessian_override)
            elif isinstance(hessian_override, HessianConfig):
                qcfg_clone.hessian = hessian_override
            else:
                raise ValueError("QuantizeConfig: dynamic `hessian` must be a HessianConfig or dict.")
        gptaq_override = qcfg.dynamic_get(module_full_name, "gptaq", None)
        foem_override = qcfg.dynamic_get(module_full_name, "foem", None)
        if gptaq_override is not None:
            if isinstance(gptaq_override, dict):
                qcfg_clone.gptaq = GPTAQConfig(**gptaq_override)
            elif isinstance(gptaq_override, GPTAQConfig):
                qcfg_clone.gptaq = gptaq_override
            else:
                raise ValueError("QuantizeConfig: dynamic `gptaq` must be a GPTAQConfig or dict.")
        if foem_override is not None:
            if isinstance(foem_override, dict):
                qcfg_clone.foem = FOEMConfig(**foem_override)
            elif isinstance(foem_override, FOEMConfig):
                qcfg_clone.foem = foem_override
            else:
                raise ValueError("QuantizeConfig: dynamic `foem` must be a FOEMConfig or dict.")

        qcfg_clone._resolve_activation_ordering(desc_act_override, act_group_aware_override)

    qcfg_clone.fallback = normalize_fallback(fallback, qcfg_clone.fallback)
    return qcfg_clone

class GPTQProcessor(LoopProcessor):
    """Captures activations and quantizes modules with GPTQ or GPTAQ/FOEM."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        require_fwd: bool = True,
        calculate_w_wq_diff: bool = False,
        calibration_concat_separator: Optional[str] = None,
    ):
        """Initializes GPTQ processing and optional weight-delta tracking."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            prepare_dataset_func=prepare_dataset_func,
            batch_size=batch_size,
            execution_config=ExecutionConfig(
                require_fwd=require_fwd,
                fwd_replay_after_process=True,
                subset_forward_early_stop=True,
            ),
        )

        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses = []
        # Preserve per-sample keep-mask semantics when batch quantization uses
        # padded calibration rows. GPTQ then consumes the original [B, S, H]
        # activations and applies the current batch mask itself.
        self.preserve_batch_keep_mask = True

        # Shared-input Hessian dedup state. The plan is derived once per
        # effective model tree; `_shared_input_leaders` is `{follower: leader}` for the subset
        # pass currently being captured and is cleared once followers adopt.
        self._shared_input_plan: Optional[SharedInputPlan] = None
        self._shared_input_plan_owner: Optional[Any] = None
        self._shared_input_plan_lock = threading.Lock()
        self._shared_input_leaders: Dict[str, str] = {}
        self._shared_input_capture_generation = 0
        self.shared_input_dedup_count = 0
        self.shared_input_dedup_telemetry: Dict[str, Any] = {}
        self._activation_amax: Dict[str, float] = {}
        self._activation_headroom: Dict[str, NVFP4ActivationHeadroom] = {}
        self._activation_global_scales: Dict[str, float] = {}
        self._activation_headroom_probe = False

    @staticmethod
    def _activation_key(name: str, module: torch.nn.Module | None = None) -> str:
        return (getattr(module, "_w4a_activation_key", None)
                or getattr(module, "module_name", None) or name)

    def _target_recipe(self, name: str, module: torch.nn.Module | None = None) -> Optional[str]:
        """Resolve the effective NVFP4 recipe for one projection target.

        A mixed stream may stage attention and MLP boundaries with different
        recipes, so the top-level recipe is not authoritative. Return ``None``
        for a target that carries an FP8 carrier and owns no NVFP4 scale.
        """
        if self.qcfg.activation_mode != "w4a_nvfp4":
            return None
        recipe = self.qcfg.activation_recipe
        full_name = name
        if "layers." not in full_name:
            full_name = (getattr(module, "full_name", None)
                         or getattr(module, "module_name", None) or name)
        if ".mlp." in full_name:
            mlp_fp8_layers = tuple(getattr(self.qcfg, "activation_mlp_fp8_layers", None) or ())
            try:
                layer_index = int(full_name.split(".")[2])
            except (IndexError, ValueError):
                layer_index = -1
            mlp_mode, mlp_recipe = layer_mlp_policy(
                layer_index, self.qcfg.activation_mode, recipe, mlp_fp8_layers
            )
            return mlp_recipe if mlp_mode == "w4a_nvfp4" else None
        if ".self_attn." in full_name:
            attention_mode = (getattr(self.qcfg, "activation_attention_mode", None)
                              or self.qcfg.activation_mode)
            if attention_mode != "w4a_nvfp4":
                return None
            attention_recipe = getattr(self.qcfg, "activation_attention_recipe", None)
            if attention_recipe is None:
                attention_recipe = recipe
            return normalize_nvfp4_recipe(attention_recipe)
        return recipe

    def _record_activation_amax(self, name: str, inp: torch.Tensor,
                                module: torch.nn.Module | None = None) -> None:
        recipe = self._target_recipe(name, module)
        if recipe is None or not torch.is_tensor(inp):
            return
        keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)
        if (
            torch.is_tensor(keep_mask) and inp.ndim >= 3 and keep_mask.ndim == 2
            and keep_mask.shape == inp.shape[:2]
        ):
            inp = inp[keep_mask.to(device=inp.device, dtype=torch.bool)]
        if inp.numel() == 0:
            return
        key = self._activation_key(name, module)
        if nvfp4_uses_headroom(recipe):
            if self._activation_headroom_probe:
                collector = self._activation_headroom.get(key)
                if collector is None:
                    raise RuntimeError(f"Missing NVFP4 headroom collector for `{key}`.")
                collector.collect(inp)
            return
        current = float(inp.detach().abs().amax().item())
        if not torch.isfinite(torch.tensor(current)):
            raise ValueError(f"NVFP4 calibration input for `{name}` contains NaN or infinity.")
        with self.lock:
            self._activation_amax[key] = max(current, self._activation_amax.get(key, 0.0))

    def begin_activation_scale_probe(self, subset, layer=None) -> bool:
        """Start the calibration-only pass required by NVIDIA headroom scales."""
        del layer
        self._activation_headroom_probe = False
        for name, named in subset.items():
            target = named.module if isinstance(named, NamedModule) else named
            full_name = getattr(named, "full_name", None) or name
            recipe = self._target_recipe(full_name, target)
            if recipe is None or not nvfp4_uses_headroom(recipe):
                continue
            key = getattr(named, "full_name", None) or self._activation_key(name, target)
            target._w4a_activation_key = key
            self._activation_headroom[key] = NVFP4ActivationHeadroom()
            target._w4a_headroom_probe = True
            target._w4a_activation_global_scale = None
            self._activation_headroom_probe = True
        return self._activation_headroom_probe

    def end_activation_scale_probe(self, subset, layer=None) -> None:
        """Freeze the probed scale before the GPTQ Hessian capture pass."""
        if not self._activation_headroom_probe:
            return
        for name, named in subset.items():
            target = named.module if isinstance(named, NamedModule) else named
            full_name = getattr(named, "full_name", None) or name
            recipe = self._target_recipe(full_name, target)
            if recipe is None or not nvfp4_uses_headroom(recipe):
                continue
            key = getattr(named, "full_name", None) or self._activation_key(name, target)
            collector = self._activation_headroom.pop(key, None)
            if collector is None:
                raise RuntimeError(f"Missing NVFP4 headroom statistics for `{key}`.")
            scale = float(nvfp4_global_scale(collector.compute_amax(), recipe=recipe))
            self._activation_global_scales[key] = scale
            target._w4a_activation_global_scale = scale
            target._w4a_headroom_probe = False
        del layer
        self._activation_headroom_probe = False

    def set_calibration_dataset(self, calibration_dataset):
        """Rejects dataset replacement because GPTQ capture is fixed at construction."""

        raise NotImplementedError("GPTQProcessor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Builds the per-module GPTQ/GPTAQ/FOEM task after applying dynamic overrides."""

        qcfg_clone = clone_gptq_config_for_module(
            self.qcfg,
            module.full_name,
            fallback=fallback,
        )
        if qcfg_clone is None:
            return

        # store last used qcfg_dynamic
        self.qcfg_dynamic = qcfg_clone

        region_timer = kwargs.get("region_timer", None)

        if qcfg_clone.gptaq is not None:
            tmp = GPTAQ(module=module, qcfg=qcfg_clone, region_timer=region_timer)
        elif qcfg_clone.foem is not None:
            tmp = FOEM(module=module, qcfg=qcfg_clone, region_timer=region_timer)
        else:
            tmp = GPTQ(module=module, qcfg=qcfg_clone, region_timer=region_timer)
            tmp.fallback = qcfg_clone.fallback
            tmp.expected_nsamples = getattr(self, "total_calibration_tokens", None)

        tmp.quantizer.configure(
            perchannel=True,
        )
        self.tasks[module.name] = tmp

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether preprocessing omitted this module from GPTQ work."""

        # gptq has no dynamic method of full override (removal)
        t = self.tasks.get(module.name, False)
        if t is False:
            return True
        else:
            return False

    def _resolve_shared_input_plan(self, model) -> Optional[SharedInputPlan]:
        """Derive the explicit `:in=<tag>` plan once per model instance."""

        # The effective module tree is instance-specific (method overrides and
        # auto-detection must not share a plan solely by model class).  Keep a
        # strong reference and compare by identity; using ``id`` alone would
        # permit false hits after object collection and id reuse.
        owner = model
        with self._shared_input_plan_lock:
            if self._shared_input_plan_owner is owner:
                return self._shared_input_plan

            plan: Optional[SharedInputPlan] = None
            shared_input_plan = getattr(model, "shared_input_plan", None)
            if callable(shared_input_plan):
                inner = getattr(model, "model", None)
                plan = shared_input_plan(
                    model_config=getattr(inner, "config", None),
                    quantize_config=getattr(model, "quantize_config", None),
                )

            self._shared_input_plan = plan
            self._shared_input_plan_owner = owner
            return plan

    @staticmethod
    def _hessian_accumulation_settings(task: GPTQ) -> Tuple[str, Optional[int], Optional[int]]:
        """Settings that alter the numerical result of Hessian accumulation; members must match to share `H`."""
        hessian = task.qcfg.hessian
        return (str(hessian.staging_dtype), hessian.chunk_size, hessian.chunk_bytes)

    def begin_shared_input_capture(
        self,
        model,
        subset_names: List[str],
        is_lm_head_module: bool = False,
    ) -> Dict[str, str]:
        """Elects one Hessian capture leader per explicit shared-input group present in this subset.

        Only plain `GPTQ` tasks participate (GPTAQ/FOEM also consume outputs). Groups that
        are split across subsets dedup only among the members captured in this pass.
        """

        with self.lock:
            self._shared_input_leaders = {}
            tasks = dict(self.tasks)
        # A cache belongs to one capture cohort.  Never let a later layer or
        # subset accidentally reuse artifacts from a prior cohort.
        for name in subset_names:
            task = tasks.get(name)
            if type(task) is GPTQ:
                task.clear_shared_hessian_cache()

        if is_lm_head_module or not self.qcfg.hessian.dedup_shared_inputs:
            return {}

        plan = self._resolve_shared_input_plan(model)
        if plan is None or not plan.shared_groups:
            return {}

        leaders: Dict[str, str] = {}
        with self.lock:
            self._shared_input_capture_generation += 1
        for group in plan.shared_groups:
            if not group.explicit:
                continue
            members = [
                name for name in subset_names
                if name in group.modules
                and type(tasks.get(name)) is GPTQ
                and tasks[name].qcfg.hessian.dedup_shared_inputs
            ]
            if len(members) < 2:
                continue
            leader = members[0]
            columns = tasks[leader].columns
            leader_settings = self._hessian_accumulation_settings(tasks[leader])
            for follower in members[1:]:
                if tasks[follower].columns != columns:
                    log.warn(
                        f"Quantization: shared-input group `{group.key}` mixes input widths "
                        f"({leader}={columns}, {follower}={tasks[follower].columns}); not sharing Hessian."
                    )
                    continue
                follower_settings = self._hessian_accumulation_settings(tasks[follower])
                if follower_settings != leader_settings:
                    log.warn(
                        f"Quantization: shared-input group `{group.key}` mixes Hessian accumulation settings "
                        f"({leader}={leader_settings}, {follower}={follower_settings}); not sharing Hessian."
                    )
                    continue
                leaders[follower] = leader

        with self.lock:
            self._shared_input_leaders = leaders
        return dict(leaders)

    def end_shared_input_capture(self, subset_names: List[str]) -> Dict[str, Any]:
        """Copies finalized Hessians and returns lifecycle telemetry for this capture pass."""

        with self.lock:
            leaders = self._shared_input_leaders
            self._shared_input_leaders = {}
            tasks = dict(self.tasks)

        adopted = 0
        caches: Dict[str, SharedHessianFactorCache] = {}
        for follower, leader in leaders.items():
            follower_task = tasks.get(follower)
            leader_task = tasks.get(leader)
            if follower_task is None or leader_task is None:
                continue
            follower_task.adopt_hessian_from(leader_task)
            # Publish the factor cache only after the follower owns its Hessian copy.
            cache = caches.get(leader)
            if cache is None:
                cache = SharedHessianFactorCache(
                    self._shared_input_capture_generation,
                    leader,
                    leader=leader,
                )
                caches[leader] = cache
                leader_task.attach_shared_hessian_cache(cache)
            follower_task.attach_shared_hessian_cache(cache)
            adopted += 1

        with self.lock:
            self.shared_input_dedup_count += adopted
            telemetry = {
                "enabled": bool(self.qcfg.hessian.dedup_shared_inputs),
                "expected_followers": len(leaders),
                "adopted_followers": adopted,
                "cumulative_adopted_followers": self.shared_input_dedup_count,
                "leader_count": len(set(leaders.values())),
                "follower_to_leader": dict(leaders),
                "subset_module_count": len(subset_names),
                "status": "verified" if adopted == len(leaders) else "mismatch",
            }
            self.shared_input_dedup_telemetry = telemetry
        return dict(telemetry)

    def shared_input_leader(self, name: str) -> Optional[str]:
        """Returns the module whose Hessian `name` will adopt in the current pass, if any."""

        leaders = getattr(self, "_shared_input_leaders", None)
        if not leaders:
            return None

        lock = getattr(self, "lock", None)
        if lock is None:
            return leaders.get(name)
        with lock:
            return leaders.get(name)

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that feeds captured batches into the GPTQ task."""

        if self.shared_input_leader(name) is not None:
            def skip(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                """Follower of a shared-input group: the leader collects this module's Hessian."""

                self._record_activation_amax(name, inp[0], module)
                del module, inp, out
            return skip

        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            """Records one activation batch for GPTQ Hessian/statistics accumulation."""

            g = self.tasks[name]  # noqa: F821
            batch_idx = self.current_batch_index()
            inp_tensor = inp[0]
            self._record_activation_amax(name, inp_tensor, module)
            if self._activation_headroom_probe:
                del inp, out
                return
            keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)

            if (
                isinstance(getattr(g, "module", None), torch.nn.Embedding)
                and torch.is_tensor(inp_tensor)
                and torch.is_tensor(keep_mask)
                and inp_tensor.dim() == 2
                and keep_mask.ndim == 2
                and keep_mask.shape == inp_tensor.shape
            ):
                keep_on_input = keep_mask.to(device=inp_tensor.device, dtype=torch.bool)
                selected_ids = inp_tensor[keep_on_input].contiguous()
                if torch.is_tensor(out) and out.shape[:2] == inp_tensor.shape:
                    selected_out = out[keep_on_input].contiguous()
                else:
                    selected_out = out
                g.add_batch(selected_ids.data, selected_out.data, batch_index=batch_idx)  # noqa: F821
            elif (
                torch.is_tensor(inp_tensor)
                and torch.is_tensor(keep_mask)
                and inp_tensor.dim() >= 3
                and keep_mask.ndim == 2
                and keep_mask.shape[:2] == inp_tensor.shape[:2]
            ):
                out_tensor = out if torch.is_tensor(out) else None
                # Keep per-sample boundaries here so batched calibration
                # accumulates GPTQ stats with the same semantics as batch_size=1.
                for sample_index, sample_keep in enumerate(keep_mask):
                    if not bool(sample_keep.any().item()):
                        continue

                    input_keep = sample_keep.to(device=inp_tensor.device)
                    sample_inp = inp_tensor[
                        sample_index : sample_index + 1, input_keep, :
                    ].contiguous()
                    if out_tensor is not None and out_tensor.dim() >= 3 and out_tensor.shape[:2] == inp_tensor.shape[:2]:
                        output_keep = sample_keep.to(device=out_tensor.device)
                        sample_out = out_tensor[
                            sample_index : sample_index + 1, output_keep, :
                        ].contiguous()
                    else:
                        sample_out = out
                    g.add_batch(sample_inp.data, sample_out.data, batch_index=batch_idx)  # noqa: F821
            else:
                g.add_batch(inp_tensor.data, out.data, batch_index=batch_idx)  # noqa: F821
            del inp, out
        return tmp

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Runs GPTQ quantization for one module and stores pack-ready tensors."""

        # Reset peak memory stats
        #torch.cuda.reset_peak_memory_stats()
        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        # logger.info(f"Quantizing module START: {name}, {gptq[name].shape()}")
        ## Need to return the quantized_weight for offloading
        with self.lock:
            g = self.tasks[module.name]

        expected_device = getattr(module, "target_device", None)
        if expected_device is None:
            expected_device = getattr(module.module, "target_device", None)
        if expected_device is None:
            expected_device = get_device(module.module)

        if expected_device is not None:
            expected_device = torch.device(expected_device)

            module_weight = getattr(module.module, "weight", None)
            if module_weight is not None:
                assert module_weight.device == expected_device, (
                    f"Module '{module.full_name}' weight device {module_weight.device} does not match "
                    f"assigned target device {expected_device}."
                )
                assert module_weight.data.device == expected_device, (
                    f"Module '{module.full_name}' weight.data device {module_weight.data.device} does not match "
                    f"assigned target device {expected_device}."
                )

            g_module = getattr(g, "module", None)
            g_weight = getattr(g_module, "weight", None) if g_module is not None else None
            if g_weight is not None:
                assert g_weight.device == expected_device, (
                    f"GPTQ task for module '{module.full_name}' expected device {expected_device}, "
                    f"but found weight on {g_weight.device}."
                )
                assert g_weight.data.device == expected_device, (
                    f"GPTQ task for module '{module.full_name}' weight.data on {g_weight.data.device} "
                    f"does not match target device {expected_device}."
                )

            g_h = getattr(g, "H", None)
            if g_h is not None:
                assert torch.device(g_h.device) == expected_device, (
                    f"GPTQ Hessian tensor for '{module.full_name}' lives on {g_h.device}, expected {expected_device}."
                )

            if expected_device.type == "cuda" and torch.cuda.is_available():
                current_cuda_device = torch.device("cuda", torch.cuda.current_device())
                assert current_cuda_device == expected_device, (
                    f"CUDA thread context {current_cuda_device} does not match expected device {expected_device} "
                    f"while processing '{module.full_name}'."
                )
            if expected_device.type == "npu" and HAS_NPU:
                current_npu_device = torch.device("npu", torch.npu.current_device())
                assert current_npu_device == expected_device, (
                    f"NPU thread context {current_npu_device} does not match expected device {expected_device} "
                    f"while processing '{module.full_name}'."
                )

        wq, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp_percent, nsamples = g.quantize()

        workspace_summary = getattr(g, "_borrow_workspace_last_summary", None)
        workspace_totals = getattr(g, "_borrow_workspace_totals", None)

        module.stream_state_payload_to_cpu(
            {
                "q_scales": q_scales,
                "q_zeros": q_zeros,
                "q_g_idx": q_g_idx,
            },
        )
        del q_scales, q_zeros, q_g_idx

        with self.lock:
            self.durations.append(duration)
            if isinstance(avg_loss, (int, float)):
                self.avg_losses.append(avg_loss)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")
        ## Assign the quantized weight to the weight
        #gptq[name].layer.weight.data = q_full_weight.to(device=gptq[name].device)

        ## Offload the quantized weight to CPU for EoRA
        #quantized_weights['model.layers.%d.%s' % (module_index, name)] = q_full_weights.cpu()

        # if task is not None:
        #     task.get_logger().report_scalar(
        #         title='Quantization Loss',
        #         series=f'layer_{module_index}_loss',
        #         value=avg_loss,
        #         iteration=name_index,
        #     )
        #
        #     task.get_logger().report_scalar(
        #         title='Quantization Time',
        #         series=f'layer_{module_index}_time',
        #         value=duration,
        #         iteration=name_index,
        #     )



        if isinstance(avg_loss, str):
            loss_display = avg_loss
        else:
            loss_display = f"{avg_loss:.10f}" if isinstance(avg_loss, (int, float)) else "unknown"

        stat = {
            PROCESS_LOG_NAME:  self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: loss_display,
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
        }

        if workspace_summary:
            requests = int(workspace_summary.get("requests", 0) or 0)
            if requests:
                hit_rate = float(workspace_summary.get("hit_rate", 0.0) or 0.0)
                chunk_rows = workspace_summary.get("chunk_rows")
                stat["workspace_cache_requests"] = str(requests)
                stat["workspace_cache_hit_rate"] = f"{hit_rate:.1%}"
                stat["workspace_stage_dtype"] = workspace_summary.get("staging_dtype", "")
                if chunk_rows is not None:
                    stat["workspace_chunk_rows"] = str(chunk_rows)
        if workspace_totals:
            total_requests = int(workspace_totals.get("requests", 0) or 0)
            if total_requests:
                cumulative_hit_rate = (
                    float(workspace_totals.get("materialized_hits", 0) or 0.0) / total_requests
                )
                stat["workspace_total_requests"] = str(total_requests)
                stat["workspace_total_hit_rate"] = f"{cumulative_hit_rate:.1%}"

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self.lock:
            self.log.append(stat)

        # Log the new row
        self.log_new_row(stat)

        g.log_workspace_stats(context="gptq_process")

        if self.calculate_w_wq_diff:
            # diff in float32
            w_wq_diff = module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)
            # assert module.weight.data.dtype in (torch.float16, torch.bfloat16)

            with self.lock:
                module.state.update({
                    "w_wq_diff": w_wq_diff,
                })

        with self.lock:
            self.tasks[module.name].free()

            # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
            if self.calculate_w_wq_diff:
                module.state.update({
                    # The following EoRA pass rematerializes the dense module.
                    # Preserve GPTQ's reconstructed weight in independent
                    # storage before exposing ``wq`` through the parameter.
                    "wq": snapshot_eora_reconstructed_weight(wq),
                })

        # single largest deallocation of vram happens here
        module.weight.data = wq
        if self.qcfg.activation_mode == "w4afp8":
            enable_w4afp8_replay(module.module)
        elif self.qcfg.activation_mode == "w4a_nvfp4":
            recipe = self._target_recipe(module.full_name, module.module)
            if recipe is not None and nvfp4_uses_headroom(recipe):
                try:
                    scale = self._activation_global_scales.pop(module.full_name)
                except KeyError as exc:
                    raise RuntimeError(
                        f"Missing calibrated NVFP4 headroom scale for `{module.full_name}`."
                    ) from exc
            else:
                observed = self._activation_amax.get(
                    module.full_name, self._activation_amax.get(module.name, 0.0)
                )
                scale = float(nvfp4_global_scale(
                    observed, recipe=recipe or self.qcfg.activation_recipe or "least_squares"
                ))
            module.state["activation_global_scale"] = scale
            enable_w4a_nvfp4_replay(
                module.module, scale, recipe or self.qcfg.activation_recipe or "least_squares"
            )

    # submodule_finalized is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Creates the quantized module and packs the saved GPTQ tensors into it."""

        # generate complete, safe to move to cpu
        # module.weight.data = move_to(module.state.pop("wq"), device=CPU) # large weights is slow to init on cpu

        # cleanup all memory or states vars persistently added by this processor
        module.stream_sync()
        with (self.lock):
            # if calculate_w_wq_diff is enabled (eora), we need to revert our original wq
            if self.calculate_w_wq_diff:
                module.weight.data = module.state.pop("wq").to(CPU)

            module.state.pop("w", None) #
            module.state.pop("w_wq_diff", None)

            # need to clone to due to steamed pinned memory and access on diff thread
            q_zeros = module.state.pop("q_zeros").clone()
            q_scales = module.state.pop("q_scales").clone()
            q_g_idx = module.state.pop("q_g_idx").clone()

        assert q_zeros.device == CPU
        assert q_scales.device == CPU
        assert q_g_idx.device == CPU

        layers = {module.full_name: model.model.get_submodule(module.full_name)}
        module_label = getattr(module, "full_name", getattr(module, "name", ""))
        parent_key = getattr(module, "full_name", getattr(module, "name", None))

        # replace module with quantized module
        timer = getattr(model, "quant_region_timer", None)

        create_start = time.perf_counter() if timer is not None else None
        with log_time_block(
            "create_quant_module",
            logger=log,
            module_name=module_label,
        ):
            with parent_module_lock(parent_key):
                create_quant_module(
                    name=module.full_name,
                    linear_cls=model.qlinear_kernel,
                    bits=self.qcfg.runtime_bits,
                    desc_act=self.qcfg.desc_act,
                    dynamic=self.qcfg.dynamic,
                    group_size=self.qcfg.group_size,
                    module=model.model,
                    submodule=module,
                    sym=self.qcfg.sym,
                    device=self.qcfg.device,
                    lm_head_name=model.lm_head,
                    pack_dtype=self.qcfg.pack_dtype,
                    format=resolve_quant_format(self.qcfg.format, self.qcfg.method),
                    register_buffers=False,
                )
        if timer is not None and create_start is not None:
            timer.record(
                "submodule_finalize_create",
                time.perf_counter() - create_start,
                source=module_label,
            )

        # pack module
        qmodule = model.model.get_submodule(module.full_name)
        if self.qcfg.activation_mode == "w4a_nvfp4":
            qmodule.activation_global_scale.fill_(module.state.pop("activation_global_scale"))
        qModules = (
            {module.full_name: qmodule}
            if isinstance(qmodule, (model.qlinear_kernel, TorchQuantEmbeddings))
            else {}
        )
        pack_start = time.perf_counter() if timer is not None else None
        with log_time_block(
            "pack",
            logger=log,
            module_name=module_label,
        ):
            with parent_module_lock(parent_key):
                packer_label = pack_module(
                    name=module.full_name,
                    qModules=qModules,
                    q_scales=q_scales,
                    q_zeros=q_zeros,
                    q_g_idx=q_g_idx,
                    layers=layers,
                    quant_linear_cls=model.qlinear_kernel,
                    lock=self.lock,
                    quantize_config=self.qcfg,
                )
        if timer is not None and pack_start is not None:
            timer.record(
                "submodule_finalize_pack",
                time.perf_counter() - pack_start,
                source=f"{module_label} [{packer_label or 'module.pack_original'}]",
            )

        # TODO: store module quant results in module, not global processor result
        with self.lock:
            self.result_pop(module.full_name)

        del q_scales, q_zeros, q_g_idx
        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
        """Marks the model as GPTQ-quantized and runs shared finalization logic."""

        # print("finalize")
        # print_module_tree(model.model)

        # set quantized state
        model.quantized = True
        model.quantize_config.method = METHOD.GPTQ

        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Ensures GPTQ received calibration data before the quantization loop starts."""

        if self.calibration_dataset is None:
            raise ValueError("GPTQProcessor's calibration_dataset must be provided.")
        else:
            return True

    def name(self) -> str:
        """Returns `gptaq` when GPTAQ overrides are active, otherwise `gptq`."""

        # TODO fix me..this hacks inherited base class logic, why not override name in gptaq?
        qcfg = self.qcfg_dynamic if self.qcfg_dynamic is not None else self.qcfg
        if qcfg.gptaq is not None:
            return "gptaq"
        if qcfg.foem is not None:
            return "foem"
        else:
            return "gptq"

    def _release_host_buffers(self, *tensors: torch.Tensor) -> None:
        """Retain the old cleanup hook for streaming tests and external callers.

        Host buffers are now owned by the stream ticket lifecycle instead of a
        dedicated GPTQProcessor pool, so release is intentionally a no-op.
        """
        _ = tensors

    def has_captured_input_ids(self, name: str) -> bool:
        """Reports whether the module saw at least one captured forward batch."""

        return self.tasks[name].fwd_counter > 0
