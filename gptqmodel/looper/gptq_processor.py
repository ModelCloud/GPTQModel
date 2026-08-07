# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import CPU, DEVICE
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
from ..nn_modules.fused_group_forward import FusedGroupForward
from ..nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ..quantization import FOEM, GPTAQ, GPTQ
from ..quantization.config import (
    METHOD,
    AdaptiveDampingConfig,
    DampConfig,
    FOEMConfig,
    GPTAQConfig,
    HessianConfig,
    LengthAwareConfig,
    LengthAwareMode,
    QuantizeConfig,
    normalize_scale_search,
    resolve_quant_format,
)
from ..quantization.diagnostics import (
    QuantizationDiagnosticsMode,
    analyze_group_index,
    analyze_output_error,
    analyze_quantization_losses,
    analyze_reconstruction_error,
    analyze_scale_channels,
    compare_quant_code_samples,
    render_quantization_diagnostics_markdown,
    resolve_quantization_diagnostics_mode,
    sample_packed_quant_codes,
    sample_reconstructed_quant_codes,
    summarize_quant_code_fingerprints,
)
from ..utils.backend import BACKEND
from ..utils.device import get_device
from ..utils.fallback import normalize_fallback
from ..utils.importer import select_quant_linear
from ..utils.logger import log_time_block, setup_logger
from ..utils.marlin import replace_parameter
from ..utils.model import create_quant_module, pack_module, recurse_getattr, recurse_setattr
from ..utils.module_locks import parent_module_lock
from ..utils.torch import HAS_NPU


log = setup_logger()
lock = threading.Lock()


def _set_module_weight(module: NamedModule, weight: torch.Tensor) -> None:
    """Assign a reconstructed weight, updating the shared fused buffer if active."""

    target = module.module if isinstance(module, NamedModule) and hasattr(module, "module") else module
    fg = getattr(target, "_fused_group_forward", None)
    if isinstance(fg, FusedGroupForward):
        if fg.update_member_weight(target, weight):
            return
    module.weight.data = weight


def snapshot_eora_reconstructed_weight(weight: torch.Tensor) -> torch.Tensor:
    """Keep EoRA's reconstructed weight independent from module rematerialization.

    Sequential processors can reload the dense checkpoint into the module's
    parameter storage before EoRA runs. A plain tensor alias would then stop
    containing GPTQ's reconstruction and make the base packer derive different
    logical codes. The CPU copy also avoids retaining a second full weight on
    the quantization device.
    """

    return weight.detach().to(device=CPU, copy=True)


def log_scale_search_config(qcfg: QuantizeConfig) -> str:
    """Emit one durable startup line with the resolved ScaleSearch policy."""

    message = f"ScaleSearch config: {qcfg.scale_search_cli_summary()}"
    log.info(message)
    return message


def clone_gptq_config_for_module(
    qcfg: QuantizeConfig,
    module_full_name: str,
    *,
    fallback=None,
) -> Optional[QuantizeConfig]:
    """Clones and applies per-module GPTQ dynamic overrides, or skips the module."""

    # Resolve all dynamic overrides for this module with a single pattern scan.
    # The result is cached on the source QuantizeConfig.
    dynamic_overrides = qcfg.dynamic_get(layer_name=module_full_name)

    # entire module is skipped
    if dynamic_overrides is False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)

    # dynamic overrides
    if dynamic_overrides is not None:
        qcfg_clone.bits = dynamic_overrides.get("bits", qcfg_clone.bits)
        qcfg_clone.sym = dynamic_overrides.get("sym", qcfg_clone.sym)
        dynamic_mse_present = "mse" in dynamic_overrides
        dynamic_scale_search_present = "scale_search" in dynamic_overrides
        if dynamic_mse_present:
            qcfg_clone.mse = dynamic_overrides["mse"]
        if dynamic_scale_search_present:
            qcfg_clone.scale_search = normalize_scale_search(dynamic_overrides["scale_search"])
            if qcfg_clone.scale_search is None:
                qcfg_clone.mse = 0.0
            elif not dynamic_mse_present:
                # Activation- and Hessian-aware objectives are quadratic; do
                # not inherit a custom legacy MSE exponent from the parent.
                qcfg_clone.mse = 2.0
        elif dynamic_mse_present:
            # Preserve the legacy per-module contract where mse=0 disables
            # search and mse>0 selects the uniform weight-error objective.
            qcfg_clone.scale_search = "mse" if float(qcfg_clone.mse or 0.0) > 0 else None
        qcfg_clone._normalize_scale_search()

        qcfg_clone.group_size = dynamic_overrides.get("group_size", qcfg_clone.group_size)
        desc_act_override = dynamic_overrides.get("desc_act", None)
        if desc_act_override is not None:
            qcfg_clone.desc_act = desc_act_override
        act_group_aware_override = dynamic_overrides.get("act_group_aware", None)
        if act_group_aware_override is not None:
            qcfg_clone.act_group_aware = act_group_aware_override
            qcfg_clone._act_group_aware_user_value = act_group_aware_override
        damp_percent_override = dynamic_overrides.get("damp_percent")
        if damp_percent_override is not None:
            cfg = qcfg_clone.adaptive_damping
            if isinstance(cfg, AdaptiveDampingConfig):
                cfg.base_percdamp = damp_percent_override
                cfg.min = max(cfg.min, damp_percent_override)
                cfg.max = max(cfg.max, damp_percent_override)
            elif isinstance(cfg, DampConfig):
                cfg.min = damp_percent_override
                cfg.max = damp_percent_override
            else:
                qcfg_clone.adaptive_damping = DampConfig(
                    min=damp_percent_override,
                    max=damp_percent_override,
                    step=qcfg_clone.damp_auto_increment,
                )
            qcfg_clone.damp_percent = damp_percent_override
            qcfg_clone._damp_percent_user_value = damp_percent_override
        qcfg_clone.static_groups = dynamic_overrides.get("static_groups", qcfg_clone.static_groups)
        fallback_override = dynamic_overrides.get("fallback", None)
        if fallback_override is not None:
            qcfg_clone.fallback = normalize_fallback(fallback_override, qcfg_clone.fallback)
        hessian_override = dynamic_overrides.get("hessian", None)
        if hessian_override is not None:
            if isinstance(hessian_override, dict):
                qcfg_clone.hessian = HessianConfig(**hessian_override)
            elif isinstance(hessian_override, HessianConfig):
                qcfg_clone.hessian = hessian_override
            else:
                raise ValueError("QuantizeConfig: dynamic `hessian` must be a HessianConfig or dict.")
        gptaq_override = dynamic_overrides.get("gptaq", None)
        foem_override = dynamic_overrides.get("foem", None)
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

    # GPTQ Hessian capture consumes projection inputs only. MoE bypass can call
    # HookedLinear's capture hook directly and skip an otherwise-unused GEMM.
    moe_input_capture_without_forward = True

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

        log_scale_search_config(qcfg)

        # Small group sizes interact badly with activation-aware reordering:
        # the calibration Hessian is overfit when each group contains only a
        # few columns, so we disable GAR automatically unless the user explicitly
        # requested it.
        if qcfg._normalize_act_group_aware_for_small_groups():
            log.warn(
                f"QuantizeConfig: group_size={qcfg.group_size} <= 32; auto-disabling "
                f"`act_group_aware` because activation-aware reordering overfits the "
                f"calibration Hessian for small groups. Set `act_group_aware=False` "
                f"explicitly to silence this warning."
            )

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

        # Compute per-sequence token lengths from the prepared calibration dataset
        # (using attention masks when available) and materialize any EQUAL_PER_BUCKET_WEIGHT
        # config that does not yet have bucket_boundaries/bucket_weights.
        self._calibration_sequence_lengths = self._extract_calibration_sequence_lengths(
            self.calibration_dataset
        )
        self._ensure_length_aware_materialized(self.qcfg)

        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses = []
        self.quantization_diagnostics_mode = resolve_quantization_diagnostics_mode(
            qcfg.quantization_diagnostics
        )
        self._scale_channel_diagnostics = []
        self._reconstruction_diagnostics = []
        self._code_fingerprint_diagnostics = []
        self._output_error_diagnostics = []
        log.info(
            "Quantization diagnostics: mode=%s; auto=module-loss summary, "
            "channel=scale-channel scan, bounded output replay, and sampled code-lifecycle fingerprints",
            self.quantization_diagnostics_mode.value,
        )
        # Preserve per-sample keep-mask semantics when batch quantization uses
        # padded calibration rows. GPTQ then consumes the original [B, S, H]
        # activations and applies the current batch mask itself.
        self.preserve_batch_keep_mask = True
        self._enable_shared_hessian_cache = bool(getattr(qcfg, "enable_shared_hessian_cache", True))

        # GPTQ same-input Hessian sharing:
        # Q/K/V and gate/up projections often receive the exact same activation
        # tensor inside one subset. Their Hessian XtX accumulation and
        # inverse/Cholesky setup are identical, so the processor computes them
        # once when `QuantizeConfig.enable_shared_hessian_cache` is enabled and
        # attaches the shared result to every compatible plain GPTQ task.
        # GPTAQ/FOEM and embeddings keep isolated statistics and do not
        # participate in this cache. Disabling the toggle restores the original
        # per-module accumulation and inverse lifecycle.
        self._shared_hessian_lock = threading.Lock()
        # Cache one processed calibration batch per shared group so later
        # modules only record sample ownership instead of recomputing XtX.
        self._shared_hessian_batch_cache = {}
        # Per-group partial Hessian state; materialized lazily by GPTQ tasks.
        self._shared_hessian_states = {}
        self._shared_hessian_inverse_lock = threading.Lock()
        # Per-group inverse/Cholesky cache used during quantization after the
        # shared Hessian has been materialized.
        self._shared_hessian_inverse_cache = {}
        # Refcounts allow Hinv cache entries to be released as soon as the last
        # compatible same-input module consumes them, rather than waiting for
        # subset cleanup.
        self._shared_hessian_inverse_ref_counts = {}
        self._shared_hessian_group_counts: Dict[Tuple[object, ...], int] = {}
        # Subset setup owns processor-wide cache dictionaries. The stage runner
        # cleans one subset before preparing the next; retain an explicit token
        # so an accidental overlapping prepare fails before it can invalidate
        # active quantization workers.
        self._active_shared_hessian_subset = None
        self._shared_hessian_stats = {
            "batch_requests": 0,
            "batch_hits": 0,
            "batch_misses": 0,
            "inverse_hits": 0,
            "inverse_misses": 0,
        }

    def set_calibration_dataset(self, calibration_dataset):
        """Rejects dataset replacement because GPTQ capture is fixed at construction."""

        raise NotImplementedError("GPTQProcessor's calibration_dataset cannot be modified")


    def _hessian_group_signature_for_module(self, task: GPTQ, name: str) -> Tuple[object, ...]:
        """Build the shared-Hessian group signature, isolating per-expert down projections for MoE bypass."""

        signature = (int(task.columns), self._hessian_config_signature(task.qcfg))
        if not self._is_bypass_moe_routing():
            return signature

        expert_key = self._module_expert_isolation_key(name)
        is_down = self._module_is_expert_down_proj(name)
        if expert_key is not None and is_down:
            # Each expert's down projection consumes a distinct intermediate activation,
            # so its Hessian must not be shared with other experts.
            signature = signature + (expert_key,)
        return signature

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

        # Apply the small-group GAR safeguard to per-module dynamic clones too.
        if qcfg_clone._normalize_act_group_aware_for_small_groups():
            if not getattr(self, "_gar_small_group_warned", False):
                log.warn(
                    f"QuantizeConfig: group_size={qcfg_clone.group_size} <= 32; "
                    f"auto-disabling `act_group_aware` because activation-aware "
                    f"reordering overfits the calibration Hessian for small groups. "
                    f"Set `act_group_aware=False` explicitly to silence this warning."
                )
                self._gar_small_group_warned = True

        # Materialize bucketed length-aware configs after dynamic overrides so every
        # module/task path uses the same resolved boundaries/weights.
        self._ensure_length_aware_materialized(qcfg_clone)

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

    def _capture_output_error_inputs(self, task: GPTQ, inp: torch.Tensor, *, maximum_rows: int = 128) -> None:
        """Retain a bounded CPU activation sample for channel-mode post-quant replay."""

        if self.quantization_diagnostics_mode != QuantizationDiagnosticsMode.CHANNEL:
            return
        existing = getattr(task, "_diagnostic_output_inputs", None)
        existing_rows = int(existing.shape[0]) if existing is not None else 0
        if existing_rows >= maximum_rows or isinstance(task.module, nn.Embedding):
            return
        _, reshaped, _ = task._reshape_input(inp)
        take = min(maximum_rows - existing_rows, int(reshaped.shape[0]))
        if take <= 0:
            return
        sample = reshaped[:take].detach().to(device=CPU, dtype=torch.float32, copy=True)
        task._diagnostic_output_inputs = sample if existing is None else torch.cat((existing, sample), dim=0)

    @staticmethod
    def _length_aware_hashable(cfg: LengthAwareConfig) -> Tuple[object, ...]:
        """Convert a LengthAwareConfig into a hashable tuple for shared-Hessian grouping."""

        return tuple(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in cfg.to_dict().items()
        )

    @staticmethod
    def _extract_calibration_sequence_lengths(calibration_dataset) -> List[int]:
        """Return one true token count per sequence, using attention masks when available."""

        lengths: List[int] = []
        for row in calibration_dataset:
            if not isinstance(row, dict):
                continue
            mask = row.get("attention_mask")
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    if mask.ndim == 0:
                        continue
                    if mask.ndim == 1:
                        lengths.append(int(mask.sum().item()))
                    else:
                        for i in range(mask.shape[0]):
                            lengths.append(int(mask[i].sum().item()))
                else:
                    for seq_mask in mask:
                        try:
                            lengths.append(int(sum(seq_mask)))
                        except Exception:
                            pass
                continue
            input_ids = row.get("input_ids")
            if input_ids is None:
                continue
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() == 0:
                    continue
                if input_ids.dim() == 1:
                    lengths.append(int(input_ids.numel()))
                else:
                    for i in range(input_ids.shape[0]):
                        lengths.append(int(input_ids[i].numel()))
            else:
                lengths.append(len(input_ids))
        return lengths

    def _ensure_length_aware_materialized(self, qcfg: QuantizeConfig) -> None:
        """Materialize unresolved EQUAL_PER_BUCKET_WEIGHT length-aware configs from calibration lengths."""

        hessian_cfg = getattr(qcfg, "hessian", None)
        if hessian_cfg is None:
            return
        la = getattr(hessian_cfg, "length_aware", None)
        if not isinstance(la, LengthAwareConfig) or not bool(la):
            return
        if la.mode is not LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT:
            return
        if la.bucket_boundaries is not None:
            return
        if not self._calibration_sequence_lengths:
            log.warn(
                "LengthAwareConfig EQUAL_PER_BUCKET_WEIGHT has no bucket_boundaries and no "
                "calibration sequence lengths are available; disabling length-aware normalization."
            )
            hessian_cfg.length_aware = LengthAwareConfig(mode=LengthAwareMode.DISABLED)
            return
        hessian_cfg.length_aware = LengthAwareConfig.from_lengths(
            self._calibration_sequence_lengths,
            mode=la.mode,
            min_length=la.min_length,
            min_bucket_size=la.min_bucket_size,
            max_bucket_ratio=la.max_bucket_ratio,
            bucket_weight_exponent=la.bucket_weight_exponent,
            target_bucket_count=la.target_bucket_count,
        )

    @staticmethod
    def _hessian_config_signature(qcfg: QuantizeConfig) -> Tuple[object, ...]:
        """Return the Hessian accumulation fields that must match to share XtX."""

        hessian = qcfg.hessian
        return (
            hessian.chunk_size,
            hessian.chunk_bytes,
            str(hessian.staging_dtype),
            GPTQProcessor._length_aware_hashable(hessian.length_aware),
        )

    @staticmethod
    def _hessian_inverse_signature(qcfg: QuantizeConfig) -> Tuple[object, ...]:
        """Return the quantization fields that must match to share Hessian inverse."""

        return (
            float(qcfg.damp.min),
            float(qcfg.damp.step),
            int(qcfg.group_size),
            bool(qcfg.desc_act),
            bool(qcfg.act_group_aware),
        )

    @staticmethod
    def _tensor_cache_fingerprint(tensor: torch.Tensor) -> Tuple[object, ...]:
        """Build a same-batch source activation identity for shared Hessian cache."""

        try:
            storage_ptr = tensor.untyped_storage().data_ptr()
        except Exception:
            storage_ptr = tensor.data_ptr()
        return (
            str(tensor.device),
            str(tensor.dtype),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            int(tensor.storage_offset()),
            int(storage_ptr),
        )

    def prepare_subset(
        self,
        subset: Dict[str, NamedModule],
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Enable GPTQ same-input Hessian sharing for one subset.

        The grouping is conservative: modules must be plain GPTQ, non-Embedding,
        have the same input-column count, and use the same Hessian config. The
        runtime cache key later also includes the actual source activation view.
        """

        subset_token = (subset_index, subset_total, subset)
        with self._shared_hessian_lock:
            if self._active_shared_hessian_subset is not None:
                raise RuntimeError(
                    "GPTQProcessor.prepare_subset() cannot overlap an active subset; "
                    "wait for cleanup_subset() after all quantization workers finish."
                )
            self._active_shared_hessian_subset = subset_token
            self._shared_hessian_batch_cache.clear()
            self._shared_hessian_states.clear()
            self._shared_hessian_inverse_cache.clear()
            self._shared_hessian_inverse_ref_counts.clear()
            self._shared_hessian_group_counts.clear()

        for name in subset:
            task = self.tasks.get(name)
            if not isinstance(task, GPTQ):
                continue
            task._hessian_is_shared = False
            task._shared_hessian_source = None
            task._shared_hessian_accum_key = None
            task._shared_hessian_inverse_key = None
            task._shared_hessian_state = None
            task._shared_hessian_inverse_cache = None
            task._shared_hessian_inverse_lock = None
            task._shared_hessian_inverse_ref_counts = None
            task._shared_hessian_stats = None
            task._shared_hessian_logical_key = False

        if not self._enable_shared_hessian_cache:
            return

        groups: Dict[Tuple[object, ...], list[str]] = {}
        for name, named_module in subset.items():
            if named_module.state.get("capture_only"):
                continue

            task = self.tasks.get(name)
            # GPTAQ and FOEM subclass GPTQ but maintain extra statistics, so
            # only the plain GPTQ task participates in shared Hessian caching.
            if type(task) is not GPTQ:
                continue
            if isinstance(task.module, nn.Embedding):
                continue

            # Same-input sharing is only valid for modules whose Hessian shape
            # and accumulation settings match. Output rows may differ.  For MoE
            # bypass routing, each expert's down projection consumes a distinct
            # intermediate activation, so it must not share a Hessian with other
            # experts.  Gate/up projections see the same hidden state for every
            # expert and continue to share across experts.
            group_signature = self._hessian_group_signature_for_module(task, name)
            groups.setdefault(group_signature, []).append(name)

        for group_index, (group_signature, names) in enumerate(groups.items()):
            if len(names) < 2:
                continue

            # The group key intentionally includes ordered module names and the
            # subset identity. This prevents reuse across unrelated subsets
            # even if Python storage pointers are recycled later.
            accum_key = (
                "gptq-shared-hessian",
                subset_index,
                subset_total,
                group_index,
                tuple(names),
                group_signature,
            )
            self._shared_hessian_group_counts[accum_key] = len(names)

            # In MoE bypass mode the same hidden-state tensor is dispatched to
            # many experts and may be copied across devices.  The per-device
            # copies have different storage pointers and devices, so the default
            # tensor-identity cache key would create one cache entry per copy and
            # inflate each module's sample count by the number of devices.  Mark
            # these groups as logical: the cache key is based on the group/batch
            # identity instead of the physical storage pointer.
            is_logical_group = (
                self._is_bypass_moe_routing()
                and any(self._module_expert_isolation_key(n) is not None for n in names)
            )

            shared_state = {
                "lock": threading.Lock(),
                "partials": {},
                "sample_counts": {},
                "sequence_counts": {},
                "H": None,
                "dirty": False,
                "total_samples": 0,
                "total_sequences": 0,
            }
            self._shared_hessian_states[accum_key] = shared_state
            inverse_ref_counts: Dict[Tuple[object, ...], int] = {}

            for name in names:
                task = self.tasks[name]
                inverse_key = (
                    accum_key,
                    self._hessian_inverse_signature(task.qcfg),
                )
                inverse_ref_counts[inverse_key] = inverse_ref_counts.get(inverse_key, 0) + 1
                task._hessian_is_shared = True
                task._shared_hessian_accum_key = accum_key
                task._shared_hessian_inverse_key = inverse_key
                task._shared_hessian_state = shared_state
                task._shared_hessian_inverse_cache = self._shared_hessian_inverse_cache
                task._shared_hessian_inverse_lock = self._shared_hessian_inverse_lock
                task._shared_hessian_inverse_ref_counts = self._shared_hessian_inverse_ref_counts
                task._shared_hessian_stats = self._shared_hessian_stats
                task._shared_hessian_logical_key = is_logical_group

            self._shared_hessian_inverse_ref_counts.update(inverse_ref_counts)

    def prepare_shared_hessian_subset(
        self,
        subset: Dict[str, NamedModule],
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Compatibility wrapper for tests/tools that call the GPTQ-specific hook."""

        self.prepare_subset(
            subset,
            subset_index=subset_index,
            subset_total=subset_total,
        )

    def cleanup_subset(
        self,
        subset: Optional[Dict[str, NamedModule]] = None,
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Drop per-subset shared Hessian tensors after all workers finish."""

        with self._shared_hessian_lock:
            expected_token = None if subset is None else (subset_index, subset_total, subset)
            active_token = self._active_shared_hessian_subset
            if (
                expected_token is not None
                and active_token is not None
                and (
                    expected_token[0] != active_token[0]
                    or expected_token[1] != active_token[1]
                    or expected_token[2] is not active_token[2]
                )
            ):
                mismatch = True
            else:
                mismatch = False
            active_subset = active_token[2] if active_token is not None else subset
            self._shared_hessian_batch_cache.clear()
            self._shared_hessian_states.clear()
            self._shared_hessian_inverse_cache.clear()
            self._shared_hessian_inverse_ref_counts.clear()
            self._shared_hessian_group_counts.clear()
            self._active_shared_hessian_subset = None

        task_names = tuple(self.tasks) if active_subset is None else tuple(active_subset)
        for name in task_names:
            task = self.tasks.get(name)
            if not isinstance(task, GPTQ):
                continue
            task._hessian_is_shared = False
            task._shared_hessian_source = None
            task._shared_hessian_accum_key = None
            task._shared_hessian_inverse_key = None
            task._shared_hessian_state = None
            task._shared_hessian_inverse_cache = None
            task._shared_hessian_inverse_lock = None
            task._shared_hessian_inverse_ref_counts = None
            task._shared_hessian_stats = None
            task._shared_hessian_logical_key = False

        if mismatch:
            raise RuntimeError("GPTQProcessor.cleanup_subset() does not match the active subset.")

    def clear_shared_hessian_subset(self) -> None:
        """Compatibility wrapper for tests/tools that call the GPTQ-specific hook."""

        self.cleanup_subset()

    def shared_hessian_stats(self, reset: bool = False) -> Dict[str, int]:
        """Return GPTQ same-input Hessian sharing counters for tests/benchmarks."""

        stats = dict(self._shared_hessian_stats)
        if reset:
            for key in self._shared_hessian_stats:
                self._shared_hessian_stats[key] = 0
        return stats

    def _shared_hessian_cache_key(
        self,
        task: GPTQ,
        source_tensor: torch.Tensor,
        batch_index: Optional[int],
        cache_extra: Optional[Tuple[object, ...]],
    ):
        """Return a cache key when this GPTQ task belongs to a shared Hessian group."""

        group_key = getattr(task, "_shared_hessian_accum_key", None)
        if group_key is None:
            return None
        # Logical-key groups share one cache entry per batch because every module
        # in the group consumes the same logical activation (e.g. all MoE gate/up
        # projections under routing=bypass).  Using a fixed placeholder instead of
        # the tensor storage pointer stops device-copy aliasing from inflating the
        # per-module sample count.
        if getattr(task, "_shared_hessian_logical_key", False):
            return (
                group_key,
                batch_index,
                cache_extra,
                0,
            )
        return (
            group_key,
            batch_index,
            cache_extra,
            self._tensor_cache_fingerprint(source_tensor),
        )

    @staticmethod
    def _accumulate_shared_hessian_state(
        shared_state,
        batch_token_size: int,
        xtx: torch.Tensor,
        device: torch.device,
        sequence_count: int = 1,
    ) -> None:
        """Accumulate one XtX result into the processor-owned shared Hessian state."""

        dev = torch.device(device)
        with shared_state["lock"]:
            partials = shared_state["partials"]
            existing = partials.get(dev)
            if existing is None:
                partials[dev] = xtx.to(device=dev, dtype=torch.float32, copy=True).detach()
            else:
                if xtx.device != existing.device or xtx.dtype != torch.float32:
                    existing.add_(xtx.to(device=existing.device, dtype=torch.float32))
                else:
                    existing.add_(xtx)
            shared_state["sample_counts"][dev] = shared_state["sample_counts"].get(dev, 0) + batch_token_size
            if sequence_count:
                shared_state["sequence_counts"][dev] = shared_state["sequence_counts"].get(dev, 0) + sequence_count
            shared_state["dirty"] = True

    def _add_batch_with_shared_hessian(
        self,
        task: GPTQ,
        inp: torch.Tensor,
        out: torch.Tensor,
        *,
        batch_index: Optional[int],
        cache_source: torch.Tensor,
        cache_extra: Optional[Tuple[object, ...]] = None,
    ) -> None:
        """Record a GPTQ batch, reusing same-input Hessian work when possible."""

        self._capture_output_error_inputs(task, inp)
        if not self._enable_shared_hessian_cache:
            task.add_batch(inp, out, batch_index=batch_index)
            return

        cache_key = self._shared_hessian_cache_key(
            task,
            cache_source,
            batch_index,
            cache_extra,
        )
        if cache_key is None:
            task.add_batch(inp, out, batch_index=batch_index)
            return

        shared_state = getattr(task, "_shared_hessian_state", None)
        if shared_state is None:
            task.add_batch(inp, out, batch_index=batch_index)
            return

        cached = None
        batch_token_size = 0
        with self._shared_hessian_lock:
            self._shared_hessian_stats["batch_requests"] += 1
            cached = self._shared_hessian_batch_cache.get(cache_key)
            if cached is not None:
                self._shared_hessian_stats["batch_hits"] += 1
                cached["remaining"] -= 1
                if cached["remaining"] <= 0:
                    self._shared_hessian_batch_cache.pop(cache_key, None)
            else:
                # Keep cache-miss computation inside the lock so free-threaded
                # device workers cannot both accumulate the same logical MoE
                # activation before either publishes its cache entry.
                batch_token_size, xtx, device = task.process_batch(inp)
                if batch_token_size == 0 or xtx is None:
                    return

                sequence_count = getattr(task, "_last_batch_sequence_count", 1)
                self._accumulate_shared_hessian_state(
                    shared_state, batch_token_size, xtx, device, sequence_count=sequence_count
                )
                self._shared_hessian_stats["batch_misses"] += 1
                group_key = getattr(task, "_shared_hessian_accum_key", None)
                remaining = max(0, self._shared_hessian_group_counts.get(group_key, 1) - 1)
                if remaining:
                    self._shared_hessian_batch_cache[cache_key] = {
                        "batch_token_size": batch_token_size,
                        "remaining": remaining,
                        "sequence_count": sequence_count,
                    }

        if cached is not None:
            task.record_shared_hessian_batch(
                cached["batch_token_size"],
                shared_state,
                sequence_count=cached.get("sequence_count", 1),
            )
            return

        task.record_shared_hessian_batch(
            batch_token_size, shared_state, sequence_count=sequence_count
        )

    def moe_shared_input_group_key(self, name: str):
        """Return the logical shared-Hessian key eligible for one-call MoE capture."""

        if self.quantization_diagnostics_mode == QuantizationDiagnosticsMode.CHANNEL:
            return None
        task = self.tasks.get(name)
        if not isinstance(task, GPTQ) or not getattr(task, "_shared_hessian_logical_key", False):
            return None
        return getattr(task, "_shared_hessian_accum_key", None)

    def record_moe_shared_input_followers(
        self,
        *,
        source_name: str,
        follower_names: list[str],
        source_nsamples_before: int,
        source_fwd_counter_before: int,
    ) -> None:
        """Fan one logical MoE input observation out to shared-Hessian tasks.

        The source hook has already performed the sole XtX accumulation. Every
        follower receives the same sample/fwd counters and shared-state handle,
        matching the ordinary cache-hit path without hundreds of Python hooks.
        """

        if not follower_names:
            return
        source_task = self.tasks[source_name]
        batch_token_size = int(source_task.nsamples) - int(source_nsamples_before)
        observation_count = int(source_task.fwd_counter) - int(source_fwd_counter_before)
        shared_state = getattr(source_task, "_shared_hessian_state", None)
        group_key = getattr(source_task, "_shared_hessian_accum_key", None)
        if batch_token_size <= 0 or observation_count <= 0 or shared_state is None or group_key is None:
            raise RuntimeError(f"Cannot fan out an empty shared MoE Hessian observation from `{source_name}`.")

        for name in follower_names:
            task = self.tasks[name]
            if getattr(task, "_shared_hessian_accum_key", None) != group_key:
                raise RuntimeError(f"MoE Hessian follower `{name}` does not share source `{source_name}`.")
            task.record_shared_hessian_batch(
                batch_token_size,
                shared_state,
                observation_count=observation_count,
            )

        batch_index = self.current_batch_index()
        with self._shared_hessian_lock:
            stale_keys = [
                key
                for key in self._shared_hessian_batch_cache
                if key[0] == group_key and key[1] == batch_index
            ]
            for key in stale_keys:
                self._shared_hessian_batch_cache.pop(key, None)
            fanout_observations = len(follower_names) * observation_count
            self._shared_hessian_stats["batch_requests"] += fanout_observations
            self._shared_hessian_stats["batch_hits"] += fanout_observations

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether preprocessing omitted this module from GPTQ work."""

        # gptq has no dynamic method of full override (removal)
        t = self.tasks.get(module.name, False)
        if t is False:
            return True
        else:
            return False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that feeds captured batches into the GPTQ task."""

        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            """Records one activation batch for GPTQ Hessian/statistics accumulation."""

            g = self.tasks[name]  # noqa: F821
            batch_idx = self.current_batch_index()
            inp_tensor = inp[0]
            keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)

            if (
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

                    # Ensure the boolean index is on the same device as the
                    # activation tensor when multi-GPU balancing places inputs
                    # on different devices than the attention mask.
                    sample_keep_device = sample_keep.to(inp_tensor.device)

                    sample_inp = inp_tensor[sample_index : sample_index + 1, sample_keep_device, :].contiguous()
                    if out_tensor is not None and out_tensor.dim() >= 3 and out_tensor.shape[:2] == inp_tensor.shape[:2]:
                        sample_out = out_tensor[sample_index : sample_index + 1, sample_keep_device, :].contiguous()
                    else:
                        sample_out = out
                    self._add_batch_with_shared_hessian(
                        g,
                        sample_inp.data,
                        sample_out.data if torch.is_tensor(sample_out) else None,
                        batch_index=batch_idx,
                        cache_source=inp_tensor,
                        cache_extra=(
                            "sample",
                            sample_index,
                            self._tensor_cache_fingerprint(sample_keep),
                            int(sample_keep.sum().item()),
                        ),
                    )
            else:
                # Flattened (2-D) activations have lost per-sequence membership, so
                # length-aware normalization is unsafe. Disable it before accumulating.
                g._disable_length_aware_for_flat_input(inp_tensor)

                self._add_batch_with_shared_hessian(
                    g,
                    inp_tensor.data,
                    out.data if torch.is_tensor(out) else None,
                    batch_index=batch_idx,
                    cache_source=inp_tensor,
                )
            del inp, out
        return tmp

    @staticmethod
    def _infer_linear_module_shape(module: nn.Module) -> Optional[Tuple[int, int, bool]]:
        """Return (in_features, out_features, has_bias) for a leaf linear module."""
        bias = getattr(module, "bias", None) is not None
        if isinstance(module, nn.Linear):
            return module.in_features, module.out_features, bias
        if type(module).__name__ == "Conv1D" and hasattr(module, "weight") and module.weight.dim() == 2:
            return module.weight.shape[0], module.weight.shape[1], bias
        return None

    def _pack_into_tmp_qmodule(
        self,
        module: NamedModule,
        g,
        q_scales: torch.Tensor,
        q_zeros: torch.Tensor,
        q_g_idx: torch.Tensor,
        register_buffers: bool = False,
    ) -> Optional[nn.Module]:
        """Pack GPTQ q_scales/q_zeros/q_g_idx into a temporary packable kernel module.

        The dense leaf is temporarily moved to the CPU because ``pack_original()``
        performs CPU-side packing arithmetic. On success the leaf is left on the CPU
        so it can be restored by ``cleanup_native_replay``; on failure it is moved
        back to its original device before returning ``None``.
        """
        original = module.module
        shape = self._infer_linear_module_shape(original)
        if shape is None:
            return None
        in_f, out_f, has_bias = shape

        original_device = get_device(original)

        # pack_original() does CPU arithmetic and requires all operands on the same device.
        original = original.to(CPU)
        module.module = original

        def _restore_original():
            nonlocal original
            if original is not None:
                original = original.to(original_device)
                module.module = original

        packable_cls: Type[BaseQuantLinear] = self.gptq_model.qlinear_kernel
        if not (isinstance(packable_cls, type) and issubclass(packable_cls, PackableQuantLinear)):
            try:
                packable_cls = select_quant_linear(
                    bits=self.qcfg.runtime_bits,
                    group_size=self.qcfg.group_size,
                    desc_act=self.qcfg.desc_act,
                    sym=self.qcfg.sym,
                    device=DEVICE.CPU,
                    backend=BACKEND.GPTQ_TORCH,
                    pack=True,
                    dynamic=self.qcfg.dynamic,
                    pack_dtype=self.qcfg.pack_dtype,
                    format=resolve_quant_format(self.qcfg.format, self.qcfg.method),
                    quant_method=self.qcfg.method,
                )
            except Exception as exc:
                log.warn(f"Native replay: could not select a packable fallback kernel: {exc}")
                _restore_original()
                return None

        weight_dtype = getattr(original.weight, "dtype", torch.float16)
        tmp = packable_cls(
            bits=self.qcfg.runtime_bits,
            group_size=self.qcfg.group_size,
            desc_act=self.qcfg.desc_act,
            sym=self.qcfg.sym,
            in_features=in_f,
            out_features=out_f,
            bias=has_bias,
            pack_dtype=self.qcfg.pack_dtype,
            backend=BACKEND.GPTQ_TORCH,
            name=module.full_name,
            lm_head_name=self.gptq_model.lm_head,
            dtype=weight_dtype,
            register_buffers=register_buffers,
        )
        try:
            tmp.pack_original(linear=original, scales=q_scales, zeros=q_zeros, g_idx=q_g_idx)
        except Exception as exc:
            log.warn(f"Native replay: failed to pack into {packable_cls.__name__}: {exc}")
            _restore_original()
            return None

        # pack_original() may reuse the original bias/scales/g_idx tensors when dtype/device match.
        # Clone the small metadata tensors so the temporary replay module can be moved to the
        # replay device without mutating the original dense module or the saved state tensors.
        for attr in ("bias", "scales", "g_idx"):
            tensor = getattr(tmp, attr, None)
            if tensor is not None:
                setattr(tmp, attr, tensor.detach().clone())

        return tmp

    def _try_build_marlin_replay_module(
        self,
        module: NamedModule,
        g,
        device: torch.device,
        tmp: nn.Module,
    ) -> Optional[nn.Module]:
        """Build a Marlin module from a temporary packed module for native-kernel replay."""
        if device.type != "cuda":
            return None

        try:
            marlin_cls = select_quant_linear(
                bits=self.qcfg.runtime_bits,
                group_size=self.qcfg.group_size,
                desc_act=self.qcfg.desc_act,
                sym=self.qcfg.sym,
                device=DEVICE.CUDA,
                backend=BACKEND.GPTQ_MARLIN,
                pack=False,
                dynamic=self.qcfg.dynamic,
                pack_dtype=self.qcfg.pack_dtype,
                format=resolve_quant_format(self.qcfg.format, self.qcfg.method),
                quant_method=self.qcfg.method,
            )
        except Exception as exc:
            log.warn(f"Native replay: Marlin kernel not available for replay: {exc}")
            return None

        if not (isinstance(marlin_cls, type) and hasattr(marlin_cls, "post_init")):
            return None

        original = module.module
        shape = self._infer_linear_module_shape(original)
        if shape is None:
            return None
        in_f, out_f, has_bias = shape
        weight_dtype = getattr(original.weight, "dtype", torch.float16)

        try:
            marlin = marlin_cls(
                bits=self.qcfg.runtime_bits,
                group_size=self.qcfg.group_size,
                desc_act=self.qcfg.desc_act,
                sym=self.qcfg.sym,
                in_features=in_f,
                out_features=out_f,
                bias=has_bias,
                pack_dtype=self.qcfg.pack_dtype,
                backend=BACKEND.GPTQ_MARLIN,
                name=module.full_name,
                lm_head_name=self.gptq_model.lm_head,
                dtype=weight_dtype,
            )
            replace_parameter(marlin, "qweight", tmp.qweight)
            replace_parameter(marlin, "scales", tmp.scales)
            replace_parameter(marlin, "g_idx", tmp.g_idx)
            if has_bias and getattr(tmp, "bias", None) is not None:
                marlin.bias = tmp.bias
            marlin = marlin.to(device)
            marlin.post_init()
        except Exception as exc:
            log.warn(f"Native replay: failed to initialize Marlin replay module: {exc}")
            return None
        return marlin

    def _prepare_native_replay_qmodule(
        self,
        module: NamedModule,
        g,
        device: torch.device,
    ) -> Optional[nn.Module]:
        """Build a packed native-kernel module for post-quantization layer replay.

        The dense leaf in ``module.module`` is temporarily moved to the CPU for packing
        by ``_pack_into_tmp_qmodule``. The model tree leaf is swapped to the packed
        module and must be restored with ``cleanup_native_replay``.
        """
        if not getattr(self.qcfg, "native_kernel_replay", False):
            return None
        if self.gptq_model is None:
            return None
        if self.calculate_w_wq_diff:
            return None
        if getattr(module.module, "_fused_group_forward", None) is not None:
            return None

        module.stream_sync()
        with self.lock:
            q_scales = module.state.get("q_scales")
            q_zeros = module.state.get("q_zeros")
            q_g_idx = module.state.get("q_g_idx")
        if q_scales is None or q_zeros is None or q_g_idx is None:
            return None

        tmp = self._pack_into_tmp_qmodule(module, g, q_scales, q_zeros, q_g_idx, register_buffers=False)
        if tmp is None:
            return None

        native_qmodule = self._try_build_marlin_replay_module(module, g, device, tmp)
        if native_qmodule is not None:
            del tmp
            return native_qmodule

        # Fallback: run replay through the packable TorchLinear kernel.
        tmp = tmp.to(device)
        post_init = getattr(tmp, "post_init", None)
        if callable(post_init):
            try:
                post_init()
            except Exception as exc:
                log.warn(f"Native replay: failed to post_init fallback kernel: {exc}")
                return None
        return tmp

    def cleanup_native_replay(self, full: Dict[str, NamedModule]) -> None:
        """Restore the original dense module after native-kernel replay."""
        if self.gptq_model is None:
            return
        for module in full.values():
            if not isinstance(module, NamedModule):
                continue
            qmodule = module.state.pop("_native_replay_qmodule", None)
            if qmodule is None:
                continue
            original = module.state.pop("_native_replay_restore_module", None)
            if original is None:
                original = module.module
            with parent_module_lock(module.full_name):
                recurse_setattr(self.gptq_model.model, module.full_name, original)
            del qmodule

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

        # Crash early if a module loses calibration samples; this regressed
        # before when MoE down_proj modules were reported with 0 samples.
        self._assert_calibration_sample_count(module.name, nsamples)

        workspace_summary = getattr(g, "_borrow_workspace_last_summary", None)
        workspace_totals = getattr(g, "_borrow_workspace_totals", None)
        scale_channel_summary = None
        reconstruction_summary = None
        if self.quantization_diagnostics_mode == QuantizationDiagnosticsMode.CHANNEL:
            runtime_bits = int(getattr(g.qcfg, "runtime_bits", g.qcfg.bits))
            runtime_group_size = int(g.qcfg.group_size)
            runtime_sym = bool(g.qcfg.sym)
            runtime_desc_act = bool(g.qcfg.desc_act)
            record_identity = {
                PROCESS_LOG_LAYER: module.layer_index,
                PROCESS_LOG_MODULE: module.name,
                "full_name": module.full_name,
                "bits": runtime_bits,
                "group_size": runtime_group_size,
                "sym": runtime_sym,
                "desc_act": runtime_desc_act,
            }
            reconstruction_summary = analyze_reconstruction_error(module.weight.data, wq)
            reconstruction_summary.update(record_identity)
            with self.lock:
                self._reconstruction_diagnostics.append(reconstruction_summary)

            diagnostic_inputs = getattr(g, "_diagnostic_output_inputs", None)
            if diagnostic_inputs is not None:
                output_error_summary = analyze_output_error(
                    diagnostic_inputs,
                    module.weight.data,
                    wq,
                    bias=getattr(module, "bias", None),
                )
                output_error_summary.update(record_identity)
                with self.lock:
                    self._output_error_diagnostics.append(output_error_summary)
                del g._diagnostic_output_inputs

            scale_channel_summary = analyze_scale_channels(q_scales)
            scale_channel_summary.update(record_identity)
            scale_channel_summary["group_index"] = analyze_group_index(
                q_g_idx,
                group_count=scale_channel_summary["group_count"],
            )
            with self.lock:
                self._scale_channel_diagnostics.append(scale_channel_summary)
            code_fingerprint = sample_reconstructed_quant_codes(
                wq,
                q_scales,
                q_zeros,
                q_g_idx,
                bits=runtime_bits,
            )
            if code_fingerprint is not None:
                code_fingerprint.update(record_identity)
                with self.lock:
                    module.state["_quant_code_fingerprint"] = code_fingerprint

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
            "full_name": module.full_name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: loss_display,
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
            "bits": int(getattr(g.qcfg, "runtime_bits", g.qcfg.bits)),
            "group_size": int(g.qcfg.group_size),
            "sym": bool(g.qcfg.sym),
            "desc_act": bool(g.qcfg.desc_act),
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
        if scale_channel_summary is not None:
            stat["scale_max"] = f"{scale_channel_summary['max_scale']:.6g}"
            stat["scale_p99"] = f"{scale_channel_summary['p99_channel_max_scale']:.6g}"
            stat["scale_max_channel"] = str(scale_channel_summary["max_scale_output_channel"])
            stat["scale_max_group"] = str(scale_channel_summary["max_scale_group"])
            ratio = scale_channel_summary["max_to_median_ratio"]
            stat["scale_max/median"] = f"{ratio:.3f}" if ratio is not None else "n/a"
            stat["scale_nonfinite"] = str(scale_channel_summary["nonfinite_scale_count"])
        if reconstruction_summary is not None and reconstruction_summary.get("available"):
            stat["weight_rel_rmse"] = f"{reconstruction_summary['relative_rmse']:.8f}"
            stat["weight_cosine"] = f"{reconstruction_summary['cosine_similarity']:.8f}"
            sqnr_db = reconstruction_summary["sqnr_db"]
            stat["weight_sqnr_db"] = f"{sqnr_db:.4f}" if sqnr_db is not None else "n/a"

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
        _set_module_weight(module, wq)

        replay_device = expected_device if expected_device is not None else device
        replay_qmodule = self._prepare_native_replay_qmodule(module, g, replay_device)
        if replay_qmodule is not None:
            log.debug(
                f"Native replay prepared for {module.full_name}: "
                f"original={type(module.module).__name__} on {get_device(module.module)}, "
                f"qmodule={type(replay_qmodule).__name__} on {get_device(replay_qmodule)}"
            )
            with self.lock:
                module.state["_native_replay_restore_module"] = module.module
                module.state["_native_replay_qmodule"] = replay_qmodule
            # Sibling modules of the same subset are processed concurrently, and
            # the rest of this file guards model-tree leaf replacements with
            # parent_module_lock for exactly that reason.
            with parent_module_lock(module.full_name):
                recurse_setattr(self.gptq_model.model, module.full_name, replay_qmodule)
            log.debug(
                f"Native replay swap: {module.full_name} now in model tree: "
                f"{type(recurse_getattr(self.gptq_model.model, module.full_name)).__name__}"
            )

    # submodule_finalized is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Creates the quantized module and packs the saved GPTQ tensors into it."""

        # generate complete, safe to move to cpu
        # module.weight.data = move_to(module.state.pop("wq"), device=CPU) # large weights is slow to init on cpu

        # cleanup all memory or states vars persistently added by this processor
        module.stream_sync()
        self.cleanup_native_replay({module.full_name: module})
        with (self.lock):
            # if calculate_w_wq_diff is enabled (eora), we need to revert our original wq
            if self.calculate_w_wq_diff:
                _set_module_weight(module, module.state.pop("wq").to(CPU))

            module.state.pop("w", None) #
            module.state.pop("w_wq_diff", None)

            # need to clone to due to steamed pinned memory and access on diff thread
            q_zeros = module.state.pop("q_zeros").clone()
            q_scales = module.state.pop("q_scales").clone()
            q_g_idx = module.state.pop("q_g_idx").clone()
            code_fingerprint = module.state.pop("_quant_code_fingerprint", None)

        assert q_zeros.device == CPU
        assert q_scales.device == CPU
        assert q_g_idx.device == CPU

        module_label = getattr(module, "full_name", getattr(module, "name", ""))
        parent_key = getattr(module, "full_name", getattr(module, "name", None))
        # Snapshot the original leaf before create_quant_module replaces it.
        # We must not call find_modules(model.model) or model.named_modules()
        # here because another thread may be replacing a sibling leaf under a
        # shared ancestor while we hold a stale iterator.
        original_layer = module.module
        prepack_comparison = None
        if code_fingerprint is not None:
            prepack_sample = sample_reconstructed_quant_codes(
                module.weight.data,
                q_scales,
                q_zeros,
                q_g_idx,
                bits=code_fingerprint["bits"],
                input_indexes=code_fingerprint["input_indexes"],
                output_indexes=code_fingerprint["output_indexes"],
            )
            prepack_comparison = compare_quant_code_samples(
                code_fingerprint["codes"],
                prepack_sample["codes"] if prepack_sample is not None else None,
            )

        # replace module with quantized module
        timer = getattr(model, "quant_region_timer", None)

        create_start = time.perf_counter() if timer is not None else None
        with log_time_block(
            "create_quant_module",
            logger=log,
            module_name=module_label,
        ):
            with parent_module_lock(parent_key):
                qmodule = create_quant_module(
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

        if qmodule is None:
            return None

        # Use the quantized module returned by create_quant_module and the
        # original leaf captured above. Build one-entry dicts so pack_module
        # never has to scan the whole model tree to locate the target.
        qModules = {module.full_name: qmodule}
        layers = {module.full_name: original_layer}
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
                    lock=None,
                    quantize_config=self.qcfg,
                )
        if timer is not None and pack_start is not None:
            timer.record(
                "submodule_finalize_pack",
                time.perf_counter() - pack_start,
                source=f"{module_label} [{packer_label or 'module.pack_original'}]",
            )

        if code_fingerprint is not None:
            packed_sample = (
                sample_packed_quant_codes(
                    qmodule.qweight,
                    bits=code_fingerprint["bits"],
                    input_indexes=code_fingerprint["input_indexes"],
                    output_indexes=code_fingerprint["output_indexes"],
                )
                if qmodule is not None
                else None
            )
            packed_comparison = compare_quant_code_samples(code_fingerprint["codes"], packed_sample)
            record = {
                PROCESS_LOG_LAYER: module.layer_index,
                PROCESS_LOG_MODULE: module.name,
                "full_name": module.full_name,
                "bits": code_fingerprint["bits"],
                "group_size": code_fingerprint["group_size"],
                "sym": code_fingerprint["sym"],
                "desc_act": code_fingerprint["desc_act"],
                "packer": packer_label,
                "post_gptq": {
                    key: code_fingerprint[key]
                    for key in (
                        "sample_code_count",
                        "nonfinite_count",
                        "below_range_count",
                        "above_range_count",
                    )
                },
                "prepack": prepack_comparison,
                "packed": packed_comparison,
            }
            with self.lock:
                self._code_fingerprint_diagnostics.append(record)

        # TODO: store module quant results in module, not global processor result
        with self.lock:
            self.result_pop(module.full_name)

        del q_scales, q_zeros, q_g_idx
        module.unregister_parameter("weight")

        return qmodule

    def finalize(self, model: BaseQModel, **kwargs):
        """Marks the model as GPTQ-quantized and runs shared finalization logic."""

        diagnostics = None
        if self.quantization_diagnostics_mode != QuantizationDiagnosticsMode.OFF:
            loss_summary = analyze_quantization_losses(self.log)
            module_groups = model.simple_layer_modules(
                model_config=model.model.config,
                quantize_config=self.qcfg,
                is_awq_quantize=False,
                include_capture_only=False,
            )
            diagnostics = {
                "schema": "gptqmodel-quantization-diagnostics-v3",
                "stage": "during_quantization",
                "evidence_status": "observed",
                "mode": self.quantization_diagnostics_mode.value,
                "module_grouping": {
                    "basis": "gptqmodel_model_definition_module_group",
                    "source": f"{type(model).__module__}.{type(model).__name__}.module_tree",
                    "groups": module_groups,
                },
                "index_semantics": {
                    "layer_indices": "preserved from the model lifecycle; no +1 conversion",
                    "tensor_indices": "zero-based",
                    "weight_axis_0": "output_row",
                    "weight_axis_1": "input_feature",
                    "prepack_scale_layout": "[output_channel, group]",
                },
                "quantization_config": {
                    "method": getattr(self.qcfg.method, "value", str(self.qcfg.method)),
                    "format": getattr(self.qcfg.format, "value", str(self.qcfg.format)),
                    "bits": int(self.qcfg.runtime_bits),
                    "group_size": int(self.qcfg.group_size),
                    "sym": bool(self.qcfg.sym),
                    "desc_act": bool(self.qcfg.desc_act),
                    "pack_impl": str(self.qcfg.pack_impl),
                    "dynamic_rule_count": len(self.qcfg.dynamic or {}),
                },
                "loss": loss_summary,
            }
            top = loss_summary["top"]
            if top:
                worst = top[0]
                log.info(
                    "Quantization loss summary: modules=%d mean=%.10g median=%.10g max=%.10g "
                    "layer=%s module=%s role_median_ratio=%.3fx total_share=%.2f%%",
                    loss_summary["module_count"],
                    loss_summary["mean_loss"],
                    loss_summary["median_loss"],
                    worst["loss"],
                    worst["layer"],
                    worst["module"],
                    worst["role_median_ratio"] or 0.0,
                    100.0 * worst["total_loss_share"],
                )
                if loss_summary["severe_concentration"]:
                    log.warn(
                        "Severe pre-pack quantization-loss concentration: layer=%s module=%s loss=%.10g "
                        "is %.1fx the all-module mean and %.2f%% of total loss. Compare this module/channel against "
                        "a higher-bit or dense reference before debugging packing or inference kernels.",
                        worst["layer"],
                        worst["module"],
                        worst["loss"],
                        loss_summary["max_to_mean_ratio"],
                        100.0 * loss_summary["max_total_loss_share"],
                    )

            if self.quantization_diagnostics_mode == QuantizationDiagnosticsMode.CHANNEL:
                reconstruction_records = sorted(
                    self._reconstruction_diagnostics,
                    key=lambda item: float(item.get("relative_rmse") or -1.0),
                    reverse=True,
                )
                diagnostics["reconstruction"] = {
                    "module_count": len(reconstruction_records),
                    "records": reconstruction_records,
                    "top": reconstruction_records[:5],
                }
                output_error_records = sorted(
                    self._output_error_diagnostics,
                    key=lambda item: float(item.get("softmax_kld_mean") or -1.0),
                    reverse=True,
                )
                output_sample_count = sum(int(item.get("sample_count", 0)) for item in output_error_records)
                diagnostics["output_error"] = {
                    "module_count": len(output_error_records),
                    "sample_count": output_sample_count,
                    "mean_module_absolute_error": (
                        sum(float(item["mean_absolute_error"]) for item in output_error_records)
                        / len(output_error_records)
                        if output_error_records
                        else None
                    ),
                    "mean_module_softmax_kld": (
                        sum(float(item["softmax_kld_mean"]) for item in output_error_records)
                        / len(output_error_records)
                        if output_error_records
                        else None
                    ),
                    "records": output_error_records,
                    "top": output_error_records[:5],
                }
                if output_error_records:
                    worst_output = output_error_records[0]
                    log.info(
                        "Post-quant output error: layer=%s module=%s samples=%d mean_abs=%.6g "
                        "relative_l2=%.6g softmax_kld_mean=%.6g top1_agreement=%.2f%%",
                        worst_output[PROCESS_LOG_LAYER],
                        worst_output[PROCESS_LOG_MODULE],
                        worst_output["sample_count"],
                        worst_output["mean_absolute_error"],
                        worst_output["relative_l2_error"],
                        worst_output["softmax_kld_mean"],
                        100.0 * worst_output["top1_agreement"],
                    )
                channel_records = sorted(
                    self._scale_channel_diagnostics,
                    key=lambda item: float(item.get("max_scale") or -1.0),
                    reverse=True,
                )
                channel_top = channel_records[:5]
                diagnostics["scale_channels"] = {
                    "module_count": len(channel_records),
                    "top": channel_top,
                    "records": channel_records,
                }
                for item in channel_top:
                    ratio = item["max_to_median_ratio"]
                    log.info(
                        "Quantization scale-channel candidate: layer=%s module=%s output_channel=%d "
                        "max_scale=%.6g p99=%.6g max/median=%s nonfinite=%d above_10=%d",
                        item[PROCESS_LOG_LAYER],
                        item[PROCESS_LOG_MODULE],
                        item["max_scale_output_channel"],
                        item["max_scale"],
                        item["p99_channel_max_scale"],
                        f"{ratio:.3f}x" if ratio is not None else "n/a",
                        item["nonfinite_scale_count"],
                        item["scale_count_above_10"],
                    )
                fingerprint_summary = summarize_quant_code_fingerprints(
                    self._code_fingerprint_diagnostics
                )
                diagnostics["code_fingerprints"] = fingerprint_summary
                log.info(
                    "Quantization code fingerprints: modules=%d sampled_codes=%d "
                    "post_gptq_to_prepack_mismatch=%d (%.4f%%) "
                    "post_gptq_to_packed_mismatch=%d (%.4f%%)",
                    fingerprint_summary["module_count"],
                    fingerprint_summary["sample_code_count"],
                    fingerprint_summary["prepack_mismatch_count"],
                    100.0 * fingerprint_summary["prepack_mismatch_rate"],
                    fingerprint_summary["packed_mismatch_count"],
                    100.0 * fingerprint_summary["packed_mismatch_rate"],
                )
                if (
                    fingerprint_summary["prepack_mismatch_count"]
                    or fingerprint_summary["packed_mismatch_count"]
                ):
                    worst_fingerprint = fingerprint_summary["top"][0]
                    log.warn(
                        "Quantization code lifecycle mismatch: layer=%s module=%s "
                        "prepack=%s packed=%s. The reconstructed weight changed after GPTQ or packing.",
                        worst_fingerprint[PROCESS_LOG_LAYER],
                        worst_fingerprint[PROCESS_LOG_MODULE],
                        worst_fingerprint["prepack"]["mismatch_count"],
                        worst_fingerprint["packed"]["mismatch_count"],
                    )

            model.quantization_diagnostics = diagnostics
            model.quantization_diagnostics_markdown = render_quantization_diagnostics_markdown(diagnostics)

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
