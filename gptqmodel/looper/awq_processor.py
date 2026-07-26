# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import inspect
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import nn
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, ExecutionConfig, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import DEVICE, SUPPORTS_MODULE_TYPES
from ..models.writer import (PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
                             PROCESS_LOG_TIME, PROCESS_USED_MEMORY, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES)
from ..nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from ..nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from ..nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from ..nn_modules.qlinear.torch_awq import AwqTorchLinear
from ..quantization.awq.quantize.scale import apply_clip, apply_scale
from ..quantization.awq.utils.module import append_str_prefix, get_op_name, get_op_by_name
from ..quantization.config import FORMAT, METHOD, QuantizeConfig, resolve_quant_format
from ..utils.attn_mask import normalize_seq_mask
from ..utils.ctx import ctx
from ..utils.device import get_device
from ..utils.fallback import normalize_fallback
from ..utils.logger import setup_logger, log_time_block
from ..utils.model import find_modules, get_module_by_name_prefix, move_to, create_quant_module, pack_module
from ..utils.module_locks import parent_module_lock
from ..utils.torch import CPU

log = setup_logger()


def _resolve_module_exec_device(module: nn.Module) -> torch.device:
    """Prefer a module's planned target device, then fall back to its current device."""

    target_device = getattr(module, "target_device", None)
    if target_device is not None:
        try:
            return torch.device(target_device)
        except (TypeError, RuntimeError, ValueError):
            pass
    return get_device(module)


@dataclass
class _AWQLayerState:
    """Tracks subset progress and cached state needed to quantize one AWQ layer."""

    modules: Dict[str, NamedModule] = field(default_factory=dict)
    subset_total: Optional[int] = None
    processed_subsets: Set[int] = field(default_factory=set)
    layer_module: Optional[torch.nn.Module] = None
    previous_weight_scale: Optional[float] = None
    quantized: bool = False
    pending_modules: Set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _accumulate_awq_weight_mean(
    layers: List[nn.Linear],
    group_size: int,
) -> Tuple[torch.Tensor, int]:
    """Accumulates normalized per-channel weight sums across a group of linears."""

    if not layers:
        raise ValueError("Expected at least one linear layer to compute the AWQ weight mean.")

    first_weight = layers[0].weight.detach()
    num_channels = first_weight.shape[1]
    effective_group_size = group_size if group_size > 0 else num_channels
    if num_channels % effective_group_size != 0:
        raise ValueError(
            f"Expected in_features ({num_channels}) to be divisible by group_size ({effective_group_size})."
        )

    if first_weight.device.type == CPU:
        weights = []
        row_count = 0
        for layer in layers:
            weight = layer.weight.detach()
            if weight.shape[1] != num_channels:
                raise ValueError(
                    f"Expected consistent in_features across layers ({num_channels}), "
                    f"got {weight.shape[1]} for layer {layer}."
                )
            weights.append(weight.to(dtype=torch.float32))
            row_count += weight.shape[0]

        weight = torch.cat(weights, dim=0)
        weight = weight.abs().reshape(row_count, -1, effective_group_size)
        group_scale = weight.amax(dim=2, keepdim=True)
        weight.div_(group_scale.add_(1e-6))
        return weight.reshape(row_count, num_channels).sum(dim=0), row_count

    w_sum = torch.zeros(num_channels, dtype=torch.float32, device=first_weight.device)
    row_count = 0

    for layer in layers:
        weight = layer.weight.detach()
        if weight.shape[1] != num_channels:
            raise ValueError(
                f"Expected consistent in_features across layers ({num_channels}), "
                f"got {weight.shape[1]} for layer {layer}."
            )

        rows = weight.shape[0]
        weight_abs = weight.abs()
        weight_view = weight_abs.reshape(rows, -1, effective_group_size)
        group_scale = weight_view.amax(dim=2, keepdim=True)
        weight_view.div_(group_scale.add_(1e-6))
        w_sum += weight_abs.sum(dim=0, dtype=torch.float32)
        row_count += rows

    return w_sum, row_count


def _compute_awq_weight_mean(
    layers: List[nn.Linear],
    group_size: int,
) -> torch.Tensor:
    """Returns the average normalized per-channel weight magnitude for AWQ scaling."""

    w_sum, row_count = _accumulate_awq_weight_mean(layers, group_size)
    first_weight = layers[0].weight.detach()
    if row_count == 0:
        return torch.zeros(first_weight.shape[1], dtype=first_weight.dtype, device=first_weight.device)
    return (w_sum / row_count).to(first_weight.dtype)


class AWQProcessor(LoopProcessor):
    """Captures activations and quantizes layers with the AWQ scaling workflow."""

    @staticmethod
    def resolve_quant_source_module(named_module: NamedModule) -> nn.Module:
        """Return the dense module view AWQ should use for weight quantization."""

        quant_source = named_module.state.get("quant_source_module")
        if isinstance(quant_source, nn.Module):
            return quant_source
        return named_module.module

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        gptq_model,
        model,
        require_fwd: bool = True,
        calculate_w_wq_diff: bool = False,
        calibration_concat_separator: Optional[str] = None,
    ):
        """Initializes AWQ processing, layer tracking, and kernel selection."""

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
                enable_activation_capture=True,
            ),
        )

        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses = []
        self.nsamples = 0
        self._nsamples_total = 0
        self._quant_batch_size = batch_size

        self._layer_states: Dict[int, _AWQLayerState] = {}
        self._layer_states_lock = threading.Lock()
        self._scale_context = threading.local()
        self.gptq_model = gptq_model

        self.adjacent_model = getattr(qcfg, "adjacent_model", None)
        if self.adjacent_model is not None:
            from ..quantization.adjacent_model import AdjacentModelConfig

            if not isinstance(self.adjacent_model, AdjacentModelConfig):
                raise TypeError("adjacent_model must be an AdjacentModelConfig.")
            if int(qcfg.bits) not in (2, 3, 4, 8):
                raise ValueError("AWQ AdjacentExact supports 2, 3, 4, or 8 bits.")
            if not 1 <= int(qcfg.group_size) <= 128:
                raise ValueError("AWQ AdjacentExact requires group_size in [1, 128].")

        model_kernel = getattr(self.gptq_model, "qlinear_kernel", None)
        self.format = resolve_quant_format(qcfg.format, qcfg.method)
        self.qlinear_kernel = model_kernel or self._select_qlinear_kernel_for_format(self.format)

        self.model = model
        # Whether to apply clipping to the model during quantization. Some models may perform better with this set to False.
        self.apply_clip = True
        # "The loss computation and per-channel mean is optimized into chunked computations."
        # " Adjust this parameter to increase or decrease memory usage for these computations."
        # " Default is 1GB (1024 * 1024 * 1024)."
        self.max_chunk_memory = 1024 * 1024 * 1024

        # Whether to scale using both w/x or just x.
        self.duo_scaling = True

        self._module_forward_kwargs: Dict[str, torch.Tensor] = {}
        # Forward signatures are static during quantization. Cache accepted
        # kwarg names so AWQ's multi-ratio scale search does not repeatedly call
        # inspect.signature for the same inspected module.
        self._module_forward_signature_cache: Dict[int, Set[str]] = {}
        self._rotary_lock = threading.Lock()
        self._rotary_cache: Dict[str, nn.Module] = {}
        self._rotary_source_id: Optional[int] = None
        self._enable_activation_x_mean_cache = bool(getattr(qcfg, "enable_activation_x_mean_cache", True))

        # AWQ same-input activation sharing:
        # Q/K/V and gate/up projections often receive identical calibration
        # activations. AWQ does not have a Hessian, but it repeatedly copies
        # these activations to CPU and reduces abs(input).mean(dim=0) during
        # scale search. When `QuantizeConfig.enable_activation_x_mean_cache` is
        # enabled, the caches below deduplicate those two costs while keeping
        # final replay tensors independent because apply_scale mutates
        # input_feat_dict values in place. Disabling the toggle restores the
        # original per-module activation copy and x-mean reduction path.
        self._shared_activation_lock = threading.Lock()
        # Per-subset cache of CPU activation copies keyed by source tensor view.
        self._shared_activation_feature_cache: Dict[Tuple[object, ...], torch.Tensor] = {}
        # Maps folded replay tensors back to their shared captured source.
        self._awq_feature_share_keys: Dict[int, Tuple[object, ...]] = {}
        self._awq_feature_share_keys_by_name: Dict[str, Tuple[object, ...]] = {}
        # Per-layer cache of chunked x_mean reductions for same-source features.
        self._awq_x_mean_cache: Dict[Tuple[object, ...], torch.Tensor] = {}
        self._shared_activation_stats = {
            "feature_requests": 0,
            "feature_hits": 0,
            "feature_misses": 0,
            "x_mean_requests": 0,
            "x_mean_hits": 0,
            "x_mean_misses": 0,
        }
        self._initialize_sample_counts()
        self._module_forward_kwargs.setdefault("attention_mask", None)
        # Preserve fallback preference so AWQ can optionally fall back when no calibration data or activations are available.
        self.fallback = qcfg.fallback

    def _capture_adjacent_reference_weights(
        self,
        named_linears: Dict[str, NamedModule],
    ) -> Dict[str, torch.Tensor]:
        """Snapshot scaled dense weights before AWQ clipping mutates them."""

        if self.adjacent_model is None:
            return {}

        references: Dict[str, torch.Tensor] = {}
        for name, named_module in named_linears.items():
            linear_layer = self.resolve_quant_source_module(named_module)
            weight = getattr(linear_layer, "weight", None)
            if not isinstance(weight, torch.Tensor):
                raise TypeError(f"AWQ AdjacentExact expected `{named_module.full_name}` to expose a weight tensor.")
            references[name] = weight.detach().to(device="cpu", copy=True).contiguous()
        return references

    def _record_adjacent_skip(self, named_module: NamedModule, reason: str) -> None:
        """Expose intentional AWQ fallback skips through the shared Adjacent statistics sink."""

        if self.adjacent_model is None:
            return
        self.adjacent_model.record(
            {
                "module": named_module.full_name,
                "quant_method": "awq",
                "status": "skipped",
                "reason": reason,
                "bits": int(self.qcfg.bits),
                "group_size": int(self.qcfg.group_size),
            }
        )

    @staticmethod
    def _tensor_cache_fingerprint(tensor: torch.Tensor) -> Tuple[object, ...]:
        """Build a stable identity key for a tensor view within one capture lifecycle."""

        try:
            storage_ptr = tensor.untyped_storage().data_ptr()
        except Exception:
            storage_ptr = tensor.data_ptr()
        try:
            version = tensor._version
        except RuntimeError:
            # Inference tensors can omit version counters. The subset-local
            # cache still includes storage pointer, view metadata, and batch id.
            version = None
        return (
            str(tensor.device),
            str(tensor.dtype),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            int(tensor.storage_offset()),
            int(storage_ptr),
            int(version) if version is not None else None,
        )

    def _forward_kwarg_names(self, module: torch.nn.Module) -> Set[str]:
        """Returns cached forward kwarg names accepted by a module."""

        cache_key = id(module)
        cached = self._module_forward_signature_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            cached = set(inspect.signature(module.forward).parameters)
        except (ValueError, TypeError):
            cached = set()

        self._module_forward_signature_cache[cache_key] = cached
        return cached

    def prepare_subset(
        self,
        subset: Dict[str, NamedModule],
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Reset AWQ same-input activation sharing before subset forward capture."""

        del subset, subset_index, subset_total
        with self._shared_activation_lock:
            self._shared_activation_feature_cache.clear()

    def cleanup_subset(
        self,
        subset: Optional[Dict[str, NamedModule]] = None,
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Release AWQ same-input activation sharing after subset processing."""

        del subset, subset_index, subset_total
        self._clear_awq_activation_caches(clear_feature_cache=True)

    def shared_activation_stats(self, reset: bool = False) -> Dict[str, int]:
        """Return AWQ activation sharing counters used by targeted regression tests."""

        with self._shared_activation_lock:
            stats = dict(self._shared_activation_stats)
            if reset:
                for key in self._shared_activation_stats:
                    self._shared_activation_stats[key] = 0
        return stats

    def _clear_awq_activation_caches(self, *, clear_feature_cache: bool = False) -> None:
        """Drop AWQ activation caches whose tensors are scoped to one subset or layer."""

        with self._shared_activation_lock:
            if clear_feature_cache:
                self._shared_activation_feature_cache.clear()
            self._awq_feature_share_keys.clear()
            self._awq_feature_share_keys_by_name.clear()
            self._awq_x_mean_cache.clear()

    def _activation_feature_cache_key(self, feature: torch.Tensor) -> Tuple[object, ...]:
        """Return a cache key for AWQ same-input module captures in the current batch."""

        return (
            "awq-activation-feature",
            self.current_batch_index(),
            # CUDA can reuse a freed storage pointer for another activation in
            # the same subset. Include the live Tensor object's identity so the
            # cache only deduplicates modules that received the same input
            # object, not two unrelated tensors with recycled storage metadata.
            id(feature),
            self._tensor_cache_fingerprint(feature),
        )

    def _cache_input_feature(self, feature: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[object, ...]]]:
        """Copy one AWQ capture to CPU, reusing the copy for identical same-batch inputs."""

        if not self._enable_activation_x_mean_cache:
            if feature.device.type != "cpu":
                return feature.detach().cpu(), None
            return feature.detach(), None

        cache_key = self._activation_feature_cache_key(feature)
        with self._shared_activation_lock:
            self._shared_activation_stats["feature_requests"] += 1
            cached = self._shared_activation_feature_cache.get(cache_key)
            if cached is not None:
                self._shared_activation_stats["feature_hits"] += 1
                return cached, cache_key

        if feature.device.type != "cpu":
            cached_feature = feature.detach().cpu()
        else:
            cached_feature = feature.detach()

        with self._shared_activation_lock:
            existing = self._shared_activation_feature_cache.get(cache_key)
            if existing is None:
                self._shared_activation_feature_cache[cache_key] = cached_feature
                self._shared_activation_stats["feature_misses"] += 1
                return cached_feature, cache_key

            self._shared_activation_stats["feature_hits"] += 1
            return existing, cache_key

    @staticmethod
    def _feature_share_key(
        input_cache_keys: List[Optional[Tuple[object, ...]]],
        feature: torch.Tensor,
    ) -> Optional[Tuple[object, ...]]:
        """Map folded AWQ replay features back to the same captured source activations."""

        if not input_cache_keys or any(key is None for key in input_cache_keys):
            return None
        return (
            "awq-feature",
            tuple(input_cache_keys),
            str(feature.device),
            str(feature.dtype),
            tuple(feature.shape),
        )

    def _register_feature_share_key(
        self,
        module_name: str,
        feature: torch.Tensor,
        input_cache_keys: List[Optional[Tuple[object, ...]]],
    ) -> None:
        """Remember which folded feature tensors can reuse activation mean computation."""

        if not self._enable_activation_x_mean_cache:
            return

        share_key = self._feature_share_key(input_cache_keys, feature)
        if share_key is None:
            return
        with self._shared_activation_lock:
            self._awq_feature_share_keys[id(feature)] = share_key
            self._awq_feature_share_keys_by_name[module_name] = share_key

    def _activation_x_mean_cache_key(self, inp: torch.Tensor) -> Optional[Tuple[object, ...]]:
        """Return the cache key for activation means if this input came from a shared capture."""

        if not self._enable_activation_x_mean_cache:
            return None

        with self._shared_activation_lock:
            feature_share_key = self._awq_feature_share_keys.get(id(inp))
        if feature_share_key is None:
            return None
        return (
            "awq-x-mean",
            feature_share_key,
            str(inp.device),
            str(inp.dtype),
            tuple(inp.shape),
        )

    def _compute_activation_x_mean(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute or reuse the chunked per-channel AWQ activation mean.

        This is the AWQ analog to GPTQ Hessian sharing: only the reusable
        activation reduction is shared. Weight-dependent scale search still runs
        independently for each AWQ scaling group.
        """

        cache_key = self._activation_x_mean_cache_key(inp)
        if cache_key is not None:
            with self._shared_activation_lock:
                self._shared_activation_stats["x_mean_requests"] += 1
                cached = self._awq_x_mean_cache.get(cache_key)
                if cached is not None:
                    self._shared_activation_stats["x_mean_hits"] += 1
                    return cached

        # Keep the original activation view and take abs() inside each chunk.
        # Materializing inp.abs() for the full calibration tensor defeats the
        # memory cap below on large AWQ inputs.
        inp_flat = inp.reshape(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = torch.float32.itemsize  # accumulation happens in FP32

        # Calculate chunk size dynamically based on the available memory budget (default 1 GiB).
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)
        chunk_size = max(chunk_size, 1)

        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk = inp_flat[i:end]
            # Use one private FP32 scratch tensor per chunk and take abs()
            # in-place on that scratch. This avoids materializing both a
            # low-precision abs tensor and a FP32 conversion at the same time.
            chunk_fp32 = chunk.to(dtype=torch.float32, copy=True)
            chunk_fp32.abs_()
            x_sum.add_(chunk_fp32.sum(dim=0))
            del chunk_fp32

        x_mean = (x_sum / num_elements).to(inp.dtype)
        del x_sum

        if cache_key is not None:
            with self._shared_activation_lock:
                existing = self._awq_x_mean_cache.get(cache_key)
                if existing is None:
                    self._awq_x_mean_cache[cache_key] = x_mean
                    self._shared_activation_stats["x_mean_misses"] += 1
                    return x_mean

                self._shared_activation_stats["x_mean_hits"] += 1
                return existing

        return x_mean

    def _get_root_rotary(self) -> Optional[nn.Module]:
        """Returns the model rotary module used to refresh position embeddings."""

        if self.gptq_model.rotary_embedding:
            rotary, _ = get_module_by_name_prefix(self.model, [self.gptq_model.rotary_embedding])
            return rotary
        return getattr(getattr(self.model, "model", self.model), "rotary_emb", None)

    def _get_rotary_device(self, rotary: Optional[nn.Module], fallback: Optional[torch.device] = None) -> Optional[torch.device]:
        """Resolves the effective device for a rotary module or falls back safely."""

        if rotary is None:
            return fallback

        rotary_device = getattr(getattr(rotary, "inv_freq", None), "device", None)
        if rotary_device is not None:
            return rotary_device

        try:
            return get_device(rotary)
        except Exception:
            return fallback

    def _get_rotary_for_device(self, target_device: Optional[torch.device]) -> Optional[nn.Module]:
        """Returns a rotary module copy materialized on the requested device."""

        rotary = self._get_root_rotary()
        if rotary is None or target_device is None:
            return rotary

        target_device = torch.device(target_device)
        if self._get_rotary_device(rotary) == target_device:
            return rotary

        cache_key = str(target_device)
        with self._rotary_lock:
            rotary = self._get_root_rotary()
            if rotary is None:
                return None

            source_id = id(rotary)
            if self._rotary_source_id != source_id:
                self._rotary_cache.clear()
                self._rotary_source_id = source_id

            if self._get_rotary_device(rotary) == target_device:
                return rotary

            cached = self._rotary_cache.get(cache_key)
            if cached is None:
                try:
                    cached = copy.deepcopy(rotary)
                except Exception:
                    cached = rotary

                move_to(cached, device=target_device)
                if cached is not rotary:
                    self._rotary_cache[cache_key] = cached

            return cached

    def set_calibration_dataset(self, calibration_dataset):
        """Rejects dataset replacement because AWQ capture is fixed at construction."""

        raise NotImplementedError("AWQProcessor's calibration_dataset cannot be modified")

    def _quant_device_is_npu(self) -> bool:
        """Return whether this processor is quantizing for an Ascend NPU runtime."""

        device = getattr(self.qcfg, "device", None)
        if isinstance(device, DEVICE):
            return device == DEVICE.NPU
        if isinstance(device, torch.device):
            return device.type == "npu"
        if isinstance(device, str):
            return device.split(":", 1)[0].lower() == "npu"
        return False

    def _select_qlinear_kernel_for_format(self, format_value: FORMAT):
        """Maps the resolved AWQ format to its concrete quantized linear kernel."""

        fmt = FORMAT(format_value) if not isinstance(format_value, FORMAT) else format_value
        if fmt == FORMAT.GEMM:
            if self._quant_device_is_npu():
                return AwqTorchLinear
            return AwqGEMMLinear
        if self._quant_device_is_npu():
            raise ValueError(
                "NPU AWQ quantization requires FORMAT.GEMM so the AwqTorchLinear runtime can run on NPU; "
                f"actual format is `{fmt}`."
            )
        if fmt == FORMAT.GEMV:
            return AwqGEMVLinear
        if fmt == FORMAT.GEMV_FAST:
            return AwqGEMVFastLinear
        if fmt == FORMAT.LLM_AWQ:
            return LLMAwqLinear
        # We do not allow saving to marlin format
        # if fmt == FORMAT.MARLIN:
        #     return AwqMarlinLinear
        raise ValueError(f"METHOD.AWQ does not support this FORMAT: {format_value}")

    def _resolve_qlinear_kernel(self, module_name: Optional[str] = None):
        """Resolves the AWQ kernel after applying any dynamic format override."""

        # Honor per-module dynamic format overrides when present.
        format_override = self.qcfg.dynamic_get(module_name, "format", None) if module_name else None
        target_format = resolve_quant_format(format_override or self.qcfg.format, self.qcfg.method)
        if target_format == self.format:
            model_kernel = getattr(self.gptq_model, "qlinear_kernel", None)
            if model_kernel is not None:
                return model_kernel
        return self._select_qlinear_kernel_for_format(target_format)

    def _get_layer_state(self, layer_index: int) -> _AWQLayerState:
        """Returns the mutable tracking state for a specific transformer layer."""

        with self._layer_states_lock:
            state = self._layer_states.get(layer_index)
            if state is None:
                state = _AWQLayerState()
                self._layer_states[layer_index] = state
        return state

    def _initialize_sample_counts(self) -> None:
        """Computes sample and token totals from the calibration dataset."""

        total = 0
        dataset = getattr(self, "calibration_dataset", None)
        if dataset is None:
            self._nsamples_total = 0
            self.nsamples = 0
            return

        for row in dataset:
            if not isinstance(row, dict):
                continue
            input_ids = row.get("input_ids")
            if input_ids is None:
                continue
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() <= 1:
                    total += 1
                else:
                    total += input_ids.shape[0]
            else:
                try:
                    total += len(input_ids)
                except TypeError:
                    total += 1

        total_tokens = getattr(self, "total_calibration_tokens", 0) or total
        self._nsamples_total = total_tokens
        self.nsamples = total_tokens

    def _record_input_feature(self, module_name: str, feature: torch.Tensor) -> None:
        """Caches one captured input feature tensor for a named module."""

        if not torch.is_tensor(feature):
            return

        # Preserve a leading sample axis for flattened [seq, hidden] captures so later
        # concatenation produces [samples, seq, hidden] instead of collapsing into one giant sequence.
        if feature.dim() <= 2:
            feature = feature.unsqueeze(0)

        feature, feature_cache_key = self._cache_input_feature(feature)

        with self.lock:
            entry = self.tasks.get(module_name)
            if entry is None:
                entry = {"inputs": [], "batch_indices": [], "input_cache_keys": []}
                self.tasks[module_name] = entry
            inputs_list = entry.setdefault("inputs", [])
            inputs_list.append(feature)
            entry.setdefault("batch_indices", []).append(self.current_batch_index())
            entry.setdefault("input_cache_keys", []).append(feature_cache_key)

    @staticmethod
    def _can_concat_batch_tensors(tensors: List[torch.Tensor]) -> bool:
        """Return whether cached tensors share the same non-batch shape."""

        if not tensors:
            return False

        first = tensors[0]
        return all(
            tensor.dim() == first.dim() and tensor.shape[1:] == first.shape[1:]
            for tensor in tensors
        )

    @staticmethod
    def _concat_batch_metadata(values: List[Any]) -> Any:
        """Merge cached per-batch metadata while preserving feature alignment.

        When every tensor has the same trailing shape, we concatenate on dim 0
        so the metadata still lines up with a concatenated feature tensor.

        When sequence lengths differ across batches, the cached metadata no
        longer shares a common shape. Real AWQ repros hit cases like
        `attention_mask` `[1, 1, 423, 423]` plus `[1, 1, 36, 36]` and
        `position_ids` `[1, 423]` plus `[1, 36]`. Those values cannot be
        concatenated into one batch-aligned tensor, so we return the most
        recent non-None value. `_layer_input_features()` makes the same choice
        for the corresponding feature tensor by keeping `tensors[-1]`.
        """

        non_none = [value for value in values if value is not None]
        if not non_none:
            return None

        first = non_none[0]
        if torch.is_tensor(first):
            tensors = [value for value in values if torch.is_tensor(value)]
            if len(tensors) != len(non_none):
                return non_none[-1]
            if AWQProcessor._can_concat_batch_tensors(tensors):
                return torch.cat(tensors, dim=0)
            return non_none[-1]

        return non_none[-1]

    def _feature_kwargs_from_batch_indices(self, batch_indices: List[Optional[int]]) -> Dict[str, Any]:
        """Build kwargs aligned to the captured feature batches.

        This helper mirrors how `_layer_input_features()` collapses cached
        activations. If all selected metadata tensors share the same trailing
        dimensions we concatenate them. If sequence lengths differ across
        batches, we keep the most recent metadata entry so it stays aligned
        with the most recent feature tensor fallback.
        """

        feature_kwargs: Dict[str, Any] = dict(getattr(self, "_module_forward_kwargs", {}))
        cache = getattr(self, "inputs_cache", None)
        if cache is None:
            return feature_kwargs

        valid_indices = [int(idx) for idx in batch_indices if idx is not None and idx >= 0]
        if not valid_indices:
            return feature_kwargs

        attention_masks = getattr(cache, "attention_masks", None) or []
        if attention_masks:
            selected_masks = [
                attention_masks[idx]
                for idx in valid_indices
                if idx < len(attention_masks)
            ]
            merged_mask = self._concat_batch_metadata(selected_masks)
            if merged_mask is not None:
                feature_kwargs["attention_mask"] = merged_mask

        position_ids = getattr(cache, "position_ids", None) or []
        if position_ids:
            selected_pos = [
                position_ids[idx]
                for idx in valid_indices
                if idx < len(position_ids)
            ]
            merged_pos = self._concat_batch_metadata(selected_pos)
            if merged_pos is not None:
                feature_kwargs["position_ids"] = merged_pos

        input_kwargs = getattr(cache, "layer_input_kwargs", None) or []
        if input_kwargs:
            selected_kwargs = [
                input_kwargs[idx]
                for idx in valid_indices
                if idx < len(input_kwargs)
            ]
            if selected_kwargs:
                merged_input_kwargs: Dict[str, Any] = {}
                all_keys = {
                    key
                    for kwargs in selected_kwargs
                    for key in kwargs.keys()
                }
                for key in all_keys:
                    merged_value = self._concat_batch_metadata(
                        [kwargs.get(key) for kwargs in selected_kwargs]
                    )
                    if merged_value is not None:
                        merged_input_kwargs[key] = merged_value
                feature_kwargs.update(merged_input_kwargs)

        return feature_kwargs

    @staticmethod
    def _pack_kept_token_rows(batch_tensor: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """Pack per-batch row tensors into one kept-token row tensor."""

        batch = keep_mask.shape[0]
        source = batch_tensor
        if source.shape[0] == 1 and batch > 1:
            source = source.expand(batch, *source.shape[1:])
        packed_rows = [source[batch_index, keep_mask[batch_index]] for batch_index in range(batch)]
        return torch.cat(packed_rows, dim=0).unsqueeze(0)

    @staticmethod
    def _pack_square_token_blocks(batch_tensor: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """Pack per-batch square token tensors into one block-diagonal tensor."""

        lengths = [int(row.sum().item()) for row in keep_mask]
        total_kept = sum(lengths)
        fill_value = torch.finfo(batch_tensor.dtype).min if batch_tensor.is_floating_point() else 0
        packed = torch.full(
            (1, *batch_tensor.shape[1:-2], total_kept, total_kept),
            fill_value=fill_value,
            dtype=batch_tensor.dtype,
            device=batch_tensor.device,
        )
        offset = 0
        for batch_index, length in enumerate(lengths):
            if length <= 0:
                continue
            keep = keep_mask[batch_index].to(device=batch_tensor.device, dtype=torch.bool)
            block = batch_tensor[batch_index:batch_index + 1][..., keep, :]
            block = block[..., keep]
            packed[..., offset:offset + length, offset:offset + length] = block
            offset += length
        return packed

    @staticmethod
    def _pack_attention_mask_for_feature(
        attention_mask: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pack a per-sample mask to match a flattened kept-token feature tensor."""

        batch = keep_mask.shape[0]

        if attention_mask.ndim in (3, 4) and attention_mask.shape[0] == batch:
            return AWQProcessor._pack_square_token_blocks(attention_mask, keep_mask)

        if attention_mask.ndim == 2 and attention_mask.shape[1] == keep_mask.shape[1]:
            return AWQProcessor._pack_kept_token_rows(attention_mask, keep_mask)

        return attention_mask

    @staticmethod
    def _pack_position_ids_for_feature(
        position_ids: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pack per-sample position ids to match flattened kept-token feature tensors."""

        batch = keep_mask.shape[0]
        if position_ids.ndim != 2 or position_ids.shape[1] != keep_mask.shape[1]:
            return position_ids

        if position_ids.shape[0] not in (1, batch):
            return position_ids

        return AWQProcessor._pack_kept_token_rows(position_ids, keep_mask)

    def _align_module_kwargs_to_input(
        self,
        inp: torch.Tensor,
        module_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Align masks and positions to packed feature tensors used during AWQ replay.

        AWQ capture hooks can flatten kept tokens from multiple batch rows into
        one replay tensor shaped like `[1, kept_tokens, hidden]`. The original
        cached kwargs still describe the pre-packed batch layout, for example an
        attention mask shaped `[B, 1, S, S]` or position ids shaped `[B, S]`.
        Passing those batched kwargs to replay produces shape mismatches in
        attention because the replay activations no longer have the original
        batch and sequence structure.

        When the packed token count matches the keep-mask derived from the
        cached attention mask, we rebuild `attention_mask` and `position_ids`
        to match the packed layout. If the counts do not match, we drop those
        kwargs rather than pass incompatible shapes into attention replay.
        """

        aligned_kwargs = dict(module_kwargs)
        attention_mask = aligned_kwargs.get("attention_mask")
        if not torch.is_tensor(inp) or inp.dim() < 3 or not torch.is_tensor(attention_mask):
            return aligned_kwargs

        if inp.shape[0] != 1 or attention_mask.ndim < 2:
            return aligned_kwargs

        try:
            keep_mask = normalize_seq_mask(attention_mask)
        except Exception:
            return aligned_kwargs

        if keep_mask is None or keep_mask.ndim != 2:
            return aligned_kwargs

        total_kept = int(keep_mask.to(dtype=torch.int64).sum().item())
        if total_kept != int(inp.shape[1]):
            aligned_kwargs.pop("attention_mask", None)
            aligned_kwargs.pop("position_ids", None)
            return aligned_kwargs

        aligned_kwargs["attention_mask"] = self._pack_attention_mask_for_feature(attention_mask, keep_mask)

        position_ids = aligned_kwargs.get("position_ids")
        if torch.is_tensor(position_ids):
            aligned_kwargs["position_ids"] = self._pack_position_ids_for_feature(position_ids, keep_mask)

        return aligned_kwargs

    def _capture_previous_subset_scale(self, previous_subset: Optional[Dict[str, NamedModule]]) -> Optional[float]:
        """Estimates the average weight scale of the previous subset for reuse heuristics."""

        if not previous_subset:
            return None

        values: List[float] = []
        for named_module in previous_subset.values():
            weight = getattr(named_module.module, "weight", None)
            if weight is None:
                continue
            with torch.no_grad():
                values.append(float(weight.detach().abs().mean().item()))

        if not values:
            return None

        return float(sum(values) / len(values))

    def _layer_input_features(self, state: _AWQLayerState) -> Dict[str, torch.Tensor]:
        """Collapse cached per-batch inputs into one replay tensor per module.

        Most batches can be concatenated on dim 0. Variable-length calibration
        batches cannot: for example `[1, 423, H]` and `[1, 36, H]` represent
        different sequence lengths after masking or packing. In that case we
        keep the most recent feature tensor and rebuild kwargs from the same
        batch index so activations, masks, and position ids remain aligned.
        """

        features: Dict[str, torch.Tensor] = {}
        feature_kwargs: Dict[str, Dict[str, Any]] = {}
        root_buckets: Dict[str, List[torch.Tensor]] = {}
        latest_ref_counts: Dict[int, int] = {}
        self._clear_awq_activation_caches(clear_feature_cache=False)

        for name in list(state.modules):
            entry = self.tasks.get(name) or {}
            tensors: List[torch.Tensor] = entry.get("inputs", [])  # type: ignore[arg-type]
            if tensors:
                latest_ref_counts[id(tensors[-1])] = latest_ref_counts.get(id(tensors[-1]), 0) + 1

        # Iterate over a snapshot since quantization may mutate state.modules concurrently
        for name in list(state.modules):
            entry = self.tasks.get(name) or {}
            tensors: List[torch.Tensor] = entry.get("inputs", [])  # type: ignore[arg-type]
            batch_indices: List[Optional[int]] = entry.get("batch_indices", [])  # type: ignore[arg-type]
            input_cache_keys: List[Optional[Tuple[object, ...]]] = entry.get("input_cache_keys", [])  # type: ignore[arg-type]
            if not tensors:
                features[name] = torch.empty(0)
                feature_kwargs[name] = {}
                continue
            if self._can_concat_batch_tensors(tensors):
                features[name] = torch.cat(tensors, dim=0)
                feature_kwargs[name] = self._feature_kwargs_from_batch_indices(batch_indices)
                self._register_feature_share_key(name, features[name], input_cache_keys)
                entry["inputs"] = [features[name]]
                entry["batch_indices"] = [None]
                entry["input_cache_keys"] = [self._awq_feature_share_keys_by_name.get(name)]
            else:
                # Variable-length captures such as `[1, 423, H]` and `[1, 36, H]`
                # cannot be concatenated on dim 0. Keep the latest capture and
                # reuse metadata from the same batch index so replay stays aligned.
                features[name] = tensors[-1]
                if latest_ref_counts.get(id(features[name]), 0) > 1:
                    # AWQ shared activation cache stores one captured tensor for
                    # same-input modules, but `apply_scale` mutates
                    # input_feat_dict values in place. Clone here so q/k/v or
                    # gate/up replay tensors cannot scale each other.
                    features[name] = features[name].clone()
                last_batch_index = batch_indices[-1] if batch_indices else None
                last_cache_key = input_cache_keys[-1] if input_cache_keys else None
                feature_kwargs[name] = self._feature_kwargs_from_batch_indices([last_batch_index])
                self._register_feature_share_key(name, features[name], [last_cache_key])
                entry["inputs"] = [features[name]]
                entry["batch_indices"] = [last_batch_index]
                entry["input_cache_keys"] = [last_cache_key]
            root = name.split(".", 1)[0]
            root_buckets.setdefault(root, []).extend(tensors)
            if features[name] is not None and features[name].numel() > 0:
                pass  # previously logged input feature shapes
                # log.info(
                #     "AWQProcessor: input feature `%s` shape=%s",
                #     name,
                #     tuple(features[name].shape),
                # )

        # for root, tensors in root_buckets.items():
        #     if not tensors or root in features:
        #         continue
        #     try:
        #         features[root] = torch.cat(tensors, dim=0)
        #     except RuntimeError:
        #         features[root] = tensors[0]
        self._awq_feature_kwargs = feature_kwargs
        return features

    def _quantize_layer_fallback(
        self,
        layer_index: int,
        state: _AWQLayerState,
        reason: str,
    ) -> None:
        """Falls back to direct quantization when AWQ scaling cannot proceed safely."""

        def unwrap(mod):
            """Returns the underlying module when wrapped in `NamedModule`."""

            return mod.module if isinstance(mod, NamedModule) else mod

        named_childs = {
            name: module
            for name, module in state.modules.items()
            if isinstance(unwrap(module), tuple(SUPPORTS_MODULE_TYPES))
        }

        log.warning(
            "AWQProcessor: layer %s using fallback quantization (%s).",
            layer_index,
            reason,
        )

        if named_childs:
            self.apply_quant(named_childs, scales_list=[])
        else:
            log.warning("AWQProcessor: layer %s had no supported modules to quantize.", layer_index)

        state.quantized = True
        state.modules.clear()
        state.pending_modules.clear()
        state.layer_module = None
        state.processed_subsets.clear()
        state.subset_total = None
        state.previous_weight_scale = None

        with self.lock:
            for name in list(named_childs.keys()):
                task_entry = self.tasks.pop(name, None)
                if task_entry and "inputs" in task_entry:
                    task_entry["inputs"].clear()

        if hasattr(self._scale_context, "layer_index"):
            delattr(self._scale_context, "layer_index")
        if hasattr(self._scale_context, "prev_scale"):
            delattr(self._scale_context, "prev_scale")
        self._clear_awq_activation_caches(clear_feature_cache=False)

    def _refresh_forward_kwargs_from_cache(self) -> None:
        """Refreshes cached kwargs such as masks and rotary embeddings for AWQ search."""

        cache = getattr(self, "inputs_cache", None)
        if cache is None:
            return

        refreshed: Dict[str, torch.Tensor] = {}

        if getattr(cache, "attention_masks", None):
            mask = cache.attention_masks[-1]
            refreshed["attention_mask"] = mask
        else:
            refreshed["attention_mask"] = None

        rotary = self._get_root_rotary()
        pos_ids_cache = cache.position_ids[-1] if getattr(cache, "position_ids", None) else None
        hidden_cache = None
        if getattr(cache, "layer_inputs", None):
            last_inputs = cache.layer_inputs[-1]
            if last_inputs:
                hidden_cache = last_inputs[0]

        if rotary is not None and hidden_cache is not None:
            x_for_rotary = hidden_cache
            if x_for_rotary.dim() == 2:
                x_for_rotary = x_for_rotary.unsqueeze(0)
            seq_len = x_for_rotary.shape[1] if x_for_rotary.dim() >= 2 else x_for_rotary.shape[0]
            batch = x_for_rotary.shape[0] if x_for_rotary.dim() >= 2 else 1

            rotary = self._get_rotary_for_device(x_for_rotary.device)
            target_device = self._get_rotary_device(rotary, x_for_rotary.device)
            if target_device is not None and x_for_rotary.device != target_device:
                x_for_rotary = x_for_rotary.to(target_device)

            if pos_ids_cache is not None and pos_ids_cache.shape[-1] == seq_len:
                pos_for_rotary = pos_ids_cache.to(x_for_rotary.device)
            else:
                pos_for_rotary = torch.arange(seq_len, device=x_for_rotary.device, dtype=torch.long)
                pos_for_rotary = pos_for_rotary.unsqueeze(0).expand(batch, -1)

            refreshed["position_ids"] = pos_for_rotary
            if rotary is not None:
                try:
                    pe = rotary(x_for_rotary, pos_for_rotary)
                    refreshed["position_embeddings"] = pe
                except Exception as exc:
                    # Cache refresh can proceed without precomputed rotary embeddings; log for debugging.
                    log.debug(
                        "AWQProcessor: failed to refresh rotary position_embeddings "
                        "(seq_len=%s, device=%s): %s",
                        seq_len,
                        x_for_rotary.device,
                        exc,
                    )
        elif pos_ids_cache is not None:
            refreshed["position_ids"] = pos_ids_cache

        if getattr(cache, "layer_input_kwargs", None):
            latest_kwargs = cache.layer_input_kwargs[-1] or {}
            for key, value in latest_kwargs.items():
                refreshed[key] = value

        self._module_forward_kwargs = refreshed

    def _should_fallback_group(
        self,
        layer_names: List[str],
        input_feat: Dict[str, torch.Tensor],
    ) -> bool:
        """Returns whether a scaling group lacks enough captured activations for AWQ."""

        from ..utils.fallback import should_use_fallback

        captured_tokens = 0
        for name in layer_names:
            feat = input_feat.get(name)
            if feat is None or feat.numel() == 0:
                return True

            hidden = feat.shape[-1] if feat.dim() >= 1 else 1
            captured_tokens += feat.numel() // max(hidden, 1)

        expected_tokens = getattr(self, "total_calibration_tokens", None) or self._nsamples_total
        return should_use_fallback(
            self.fallback,
            float(captured_tokens),
            float(expected_tokens) if expected_tokens else None,
        )

    def _quantize_layer(self, layer_index: int, state: _AWQLayerState) -> None:
        """Runs the AWQ scaling, clipping, and quantization flow for one layer."""

        if state.quantized:
            return

        layer_module = state.layer_module
        if layer_module is None and state.modules:
            sample_module = next(iter(state.modules.values()))
            layer_path = sample_module.full_name.rsplit(".", 1)[0]
            layer_module, _ = get_module_by_name_prefix(self.gptq_model.model, layer_path)
            state.layer_module = layer_module

        layer_module_ref = state.layer_module

        if layer_module_ref is None:
            raise RuntimeError(f"AWQProcessor: unable to resolve layer module for layer index {layer_index}")

        log.info(
            "AWQProcessor: layer %s tracking %s modules before quantization (subsets processed=%s/%s); first modules=%s",
            layer_index,
            len(state.modules),
            len(state.processed_subsets),
            state.subset_total,
            list(state.modules.keys())[:8],
        )

        input_feat = self._layer_input_features(state)
        missing = [name for name, tensor in input_feat.items() if tensor.numel() == 0]
        if missing and not self.fallback:
            raise RuntimeError(
                f"AWQProcessor error: missing activation features for modules {missing} "
                f"with fallback disabled."
            )

        # Filtering MLP modules like Qwen3MoeSparseMoeBlock
        def unwrap(m):
            """Returns the underlying module when wrapped in `NamedModule`."""

            return m.module if isinstance(m, NamedModule) else m

        named_childs = {
            name: module
            for name, module in state.modules.items()
            if isinstance(unwrap(module), tuple(SUPPORTS_MODULE_TYPES))
        }

        module_kwargs_global = dict(self._module_forward_kwargs)
        module_kwargs_global["_awq_feature_kwargs"] = dict(
            getattr(self, "_awq_feature_kwargs", {})
        )

        setattr(self._scale_context, "layer_index", layer_index)
        setattr(self._scale_context, "prev_scale", state.previous_weight_scale)

        module_config = self.gptq_model.awq_get_modules_for_scaling(
            layer_module_ref,
            input_feat,
            module_kwargs_global,
        )

        if not module_config:
            log.warning(
                "AWQProcessor: no module configuration generated for layer index %s; skipping quantization.",
                layer_index,
            )
            state.quantized = True
            state.modules.clear()
            state.pending_modules.clear()
            state.layer_module = None
            state.processed_subsets.clear()
            state.subset_total = None
            state.previous_weight_scale = None
            if hasattr(self._scale_context, "layer_index"):
                delattr(self._scale_context, "layer_index")
            if hasattr(self._scale_context, "prev_scale"):
                delattr(self._scale_context, "prev_scale")
            self._clear_awq_activation_caches(clear_feature_cache=False)
            return

        sanitized_module_config: List[Dict] = []
        for entry in module_config:
            entry = dict(entry)
            inspect_module = entry.get("module2inspect") or layer_module_ref
            entry_kwargs = entry.get("kwargs") or module_kwargs_global
            entry["kwargs"] = self._sanitize_kwargs(entry_kwargs, inspect_module)
            sanitized_module_config.append(entry)

        filtered_module_config: List[Dict] = []
        skipped_groups: List[Tuple[List[str], List[str]]] = []
        fallback_names = set()
        for cfg in sanitized_module_config:
            layers_sample = cfg.get("layers") or []
            prev_module = cfg.get("prev_op")

            layer_names = [
                get_op_name(layer_module_ref, layer) if isinstance(layer, torch.nn.Module) else str(layer)
                for layer in layers_sample
            ]
            if self.fallback and self._should_fallback_group(layer_names, input_feat):
                fallback_names.update(layer_names)
                continue

            first_layer_module = layers_sample[0] if layers_sample else None
            # Some configs alias prev_op to the first layer (e.g. gate_proj); treat that as valid
            same_module = prev_module is first_layer_module
            if (
                isinstance(prev_module, nn.Linear)
                and isinstance(first_layer_module, nn.Linear)
                and not same_module
                and prev_module.weight.shape[0] != first_layer_module.weight.shape[1]
            ):
                try:
                    prev_name = get_op_name(layer_module_ref, prev_module)
                except Exception:
                    prev_name = str(prev_module)
                try:
                    first_name = get_op_name(layer_module_ref, first_layer_module)
                except Exception:
                    first_name = str(first_layer_module)
                log.debug(
                    "AWQProcessor: layer %s skipping scaling group due to dimension mismatch prev_op=%s shape=%s first_layer=%s shape=%s",
                    layer_index,
                    prev_name,
                    tuple(prev_module.weight.shape),
                    first_name,
                    tuple(first_layer_module.weight.shape),
                )
                continue

            missing_layers = [name for name in layer_names if name not in input_feat]
            if missing_layers:
                skipped_groups.append((layer_names, missing_layers))
                continue
            filtered_module_config.append(cfg)

        if skipped_groups:
            log.debug(
                "AWQProcessor: layer %s skipping %d scaling groups due to missing features (sample=%s)",
                layer_index,
                len(skipped_groups),
                skipped_groups[:3],
            )

        sanitized_module_config = filtered_module_config
        if not sanitized_module_config and not fallback_names:
            log.warning(
                "AWQProcessor: no valid scaling groups for layer %s after filtering; marking layer as quantized.",
                layer_index,
            )
            state.quantized = True
            state.processed_subsets.clear()
            state.subset_total = None
            state.previous_weight_scale = None
            if hasattr(self._scale_context, "layer_index"):
                delattr(self._scale_context, "layer_index")
            if hasattr(self._scale_context, "prev_scale"):
                delattr(self._scale_context, "prev_scale")
            self._clear_awq_activation_caches(clear_feature_cache=False)
            return

        sample_groups = []
        for cfg in sanitized_module_config[:3]:
            layers_sample = cfg.get("layers") or []
            layers_names = [get_op_name(layer_module_ref, layer) if isinstance(layer, torch.nn.Module) else str(layer) for layer in layers_sample[:4]]
            sample_groups.append(layers_names)

        log.info(
            "AWQProcessor: layer %s sanitized %d scaling groups; sample=%s",
            layer_index,
            len(sanitized_module_config),
            sample_groups,
        )

        scales_list = [
            self._search_best_scale(layer_module_ref, **layer)
            for layer in sanitized_module_config
        ]

        try:
            apply_scale(layer_module_ref, scales_list, input_feat_dict=input_feat)
        except RuntimeError as exc:
            debug_entries = []
            for prev_op_name, layer_names, scales, _loss in scales_list:
                entry = {"prev": prev_op_name, "prev_shape": None, "layer_shapes": [], "scale_elems": None}
                try:
                    prev_module = get_op_by_name(layer_module_ref, prev_op_name)
                except Exception:
                    prev_module = None
                weight = getattr(prev_module, "weight", None) if prev_module is not None else None
                if isinstance(weight, torch.Tensor):
                    entry["prev_shape"] = tuple(weight.shape)
                elif prev_module is None:
                    entry["prev_shape"] = "missing"
                else:
                    entry["prev_shape"] = f"{type(prev_module).__name__} (no weight)"

                layer_shapes = []
                for lname in layer_names:
                    try:
                        layer_module = get_op_by_name(layer_module_ref, lname)
                    except Exception:
                        layer_shapes.append((lname, "missing"))
                        continue
                    weight = getattr(layer_module, "weight", None)
                    if isinstance(weight, torch.Tensor):
                        layer_shapes.append((lname, tuple(weight.shape)))
                    else:
                        layer_shapes.append((lname, f"{type(layer_module).__name__} (no weight)"))
                entry["layer_shapes"] = layer_shapes
                if hasattr(scales, "numel"):
                    try:
                        entry["scale_elems"] = int(scales.numel())
                    except Exception:
                        entry["scale_elems"] = "unknown"
                debug_entries.append(entry)

            log.error(
                "AWQProcessor: apply_scale failed at layer %s with %s. Shape summary (first 5 groups): %s",
                layer_index,
                exc,
                debug_entries[:5],
            )
            raise

        adjacent_targets = {
            name: named
            for name, named in named_childs.items()
            if name in input_feat and name not in fallback_names
        }
        adjacent_references = self._capture_adjacent_reference_weights(adjacent_targets)

        scales_list = append_str_prefix(
            scales_list,
            get_op_name(self.model, layer_module_ref) + ".",
        )

        if self.apply_clip:
            clip_list = self._search_best_clip(
                layer_module_ref,
                {name: named.module for name, named in named_childs.items() if name not in fallback_names},
                input_feat,
            )
            apply_clip(layer_module_ref, clip_list)
            clip_list = append_str_prefix(
                clip_list,
                get_op_name(self.model, layer_module_ref) + ".",
            )

        fallback_named_childs = {
            n: named_childs[n]
            for n in fallback_names
            if n in named_childs
        }

        named_childs = {name: named for name, named in named_childs.items() if name in input_feat and name not in fallback_names}

        self.apply_quant(
            named_childs,
            scales_list,
            input_features=input_feat if self.adjacent_model is not None else None,
            adjacent_references=adjacent_references if self.adjacent_model is not None else None,
        )

        if fallback_named_childs:
            log.warning(
                "AWQProcessor: layer %s fallback quant %d modules: %s",
                layer_index,
                len(fallback_named_childs),
                list(fallback_named_childs)[:6],
            )
            self.apply_quant(fallback_named_childs, scales_list=[])

        state.quantized = True
        state.modules.clear()
        state.pending_modules.clear()
        state.layer_module = None
        state.processed_subsets.clear()
        state.subset_total = None
        state.previous_weight_scale = None

        with self.lock:
            for name in named_childs:
                task_entry = self.tasks.pop(name, None)
                if task_entry and "inputs" in task_entry:
                    task_entry["inputs"].clear()

        if hasattr(self._scale_context, "layer_index"):
            delattr(self._scale_context, "layer_index")
        if hasattr(self._scale_context, "prev_scale"):
            delattr(self._scale_context, "prev_scale")
        self._clear_awq_activation_caches(clear_feature_cache=False)

    @torch.inference_mode()
    def _search_best_scale(
            self,
            module,
            prev_op,
            layers: List[nn.Linear],
            inp: torch.Tensor,
            module2inspect=None,
            kwargs={},
    ):
        """Searches the best per-channel AWQ scale for a module group."""

        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Keep AWQ scale search on the low-memory path by default. Disabling this
        # restores the legacy eager behavior that materializes full activations on
        # the module device before loss evaluation.
        use_chunked_scale_search = getattr(self.qcfg, "scale_search_chunked_activations", True)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # Stream across each Linear instead of concatenating all weights at once. This mirrors the
        # previous cat()+view pipeline while keeping peak memory low: for every group we normalise
        # |w| by its per-group max so the values land on a [0, 1] scale, then accumulate the totals
        # per output channel so the mean can be computed without allocating the combined tensor.
        w_mean = _compute_awq_weight_mean(layers, self.qcfg.group_size)

        # [STEP 2]: Compute or reuse the per-channel activation mean. Same-input
        # AWQ groups such as q/k/v share this expensive chunked reduction.
        x_mean = self._compute_activation_x_mean(inp)

        # [STEP 3]: Compute output of module
        module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
        global_kwargs = getattr(self, "_module_forward_kwargs", {})
        global_allowed_kwargs = self._sanitize_kwargs(global_kwargs, module2inspect)
        for key, value in global_allowed_kwargs.items():
            module_kwargs.setdefault(key, value)
        module_kwargs = self._align_module_kwargs_to_input(inp, module_kwargs)
        refine_steps = self._resolve_scale_search_refine_steps(module, layers)

        if use_chunked_scale_search:
            # Build the FP reference output one micro-batch at a time and move each
            # chunk back to CPU immediately so scale search does not keep the full
            # activation tensor resident on GPU.
            with ctx(torch.inference_mode()):
                fp16_output = [
                    output.clip(torch.finfo(output.dtype).min, torch.finfo(output.dtype).max).detach().cpu()
                    for output in self._iter_module_forward_outputs(inp, module2inspect, module_kwargs)
                ]
        else:
            # Legacy path: run the full forward eagerly on the module device and keep
            # the dense activation tensor for the later reconstruction-loss pass.
            inp = inp.to(next(module2inspect.parameters()).device)
            with ctx(torch.inference_mode()):
                fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)

        # [STEP 4]: Compute loss
        best_scales, loss = self._compute_best_scale(
            inp,
            w_mean,
            x_mean,
            module2inspect,
            layers,
            fp16_output,
            module_kwargs,
            refine_steps=refine_steps,
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
            loss
        )

    def _resolve_scale_search_refine_steps(
        self,
        module: torch.nn.Module,
        layers: List[nn.Linear],
    ) -> int:
        """Resolve per-module dynamic overrides for one inseparable AWQ scale group."""

        global_steps = int(getattr(self.qcfg, "scale_search_refine_steps", 0))
        if not layers or not getattr(self.qcfg, "dynamic", None):
            return global_steps

        try:
            module_prefix = str(get_op_name(self.model, module)) if self.model is not None else ""
        except Exception:
            module_prefix = ""

        resolved_by_name: Dict[str, int] = {}
        for layer in layers:
            relative_name = str(get_op_name(module, layer))
            full_name = f"{module_prefix}.{relative_name}" if module_prefix else relative_name
            resolved_by_name[full_name] = int(
                self.qcfg.dynamic_get(full_name, "scale_search_refine_steps", global_steps)
            )

        resolved_steps = set(resolved_by_name.values())
        if len(resolved_steps) != 1:
            raise ValueError(
                "AWQ ScaleSearch applies one scale to every module in a scaling group, but dynamic "
                f"`scale_search_refine_steps` values conflict: {resolved_by_name}. "
                "Use patterns that assign the same value to the full group."
            )
        return resolved_steps.pop()

    @torch.inference_mode()
    def _search_best_clip(self, layer, named_linears, input_feat):
        """Searches per-layer clipping thresholds for AWQ-eligible linears."""

        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            linear = named_linears[name]
            original_device = get_device(linear)
            target_device = _resolve_module_exec_device(linear)
            move_to(linear, device=target_device)

            max_val = self._compute_best_clip(
                linear.weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            move_to(linear, device=original_device)

        return clip_list

    @staticmethod
    def _initial_awq_clip_errors_like(org_max_val: torch.Tensor) -> torch.Tensor:
        """Create AWQ clip-search error sentinels without a ones-like multiply."""

        min_errs = torch.empty_like(org_max_val)
        # The legacy `ones_like(...)*1e9` expression overflows fp16 to inf.
        # Preserve that sentinel exactly while avoiding the temporary ones tensor.
        sentinel = float("inf") if org_max_val.dtype == torch.float16 else 1e9
        min_errs.fill_(sentinel)
        return min_errs

    @torch.inference_mode()
    def _compute_best_clip(
            self,
            w: torch.Tensor,
            input_feat: torch.Tensor,
            n_grid=20,
            max_shrink=0.5,
            n_sample_token=512,
    ):
        """Finds the clipping bound that minimizes reconstruction error for a weight tensor."""

        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.qcfg.group_size if self.qcfg.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        device = w_all.device
        # Pre-allocate the final clip tensor so chunked output rows do not
        # require a full-size torch.cat temporary after the search loop.
        best_max_val_all = torch.empty(
            (org_w_shape[0], 1, w_all.shape[2], 1),
            dtype=w_all.dtype,
            device=device,
        )
        # Pre-allocate scratch buffers so the inner clamp loop never allocates large temporaries.
        scratch_clamp = torch.empty_like(w_all[:oc_batch_size])
        scratch_quant = torch.empty_like(scratch_clamp)
        input_feat = input_feat.to(device)

        for i_b in range(org_w_shape[0] // oc_batch_size):
            row_start = i_b * oc_batch_size
            row_end = row_start + oc_batch_size
            w = w_all[row_start:row_end]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # [co_batch, 1, n_group, 1]

            best_max_val = org_max_val.clone()
            min_errs = self._initial_awq_clip_errors_like(org_max_val)
            clamp_slice = scratch_clamp[: w.shape[0]]
            quant_slice = scratch_quant[: w.shape[0]]

            org_out = (input_feat * w).sum(dim=-1)  # [co_batch, n_token, n_group]

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                torch.clamp(w, min_val, max_val, out=clamp_slice)
                self._pseudo_quantize_tensor_into(clamp_slice, quant_slice)
                cur_out = (input_feat * quant_slice).sum(dim=-1)

                # cur_out is no longer needed after this point, so reuse its
                # storage for squared error instead of allocating diff/pow
                # temporaries for every shrink step.
                cur_out.sub_(org_out).pow_(2)
                err = cur_out.mean(dim=1).view(min_errs.shape)
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all[row_start:row_end].copy_(best_max_val)

        del input_feat
        del org_out

        return best_max_val_all.squeeze(1)

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        """Simulates AWQ quantization and returns dequantized weights plus scales/zeros."""

        org_w_shape = w.shape
        if self.qcfg.group_size > 0:
            assert org_w_shape[-1] % self.qcfg.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.qcfg.group_size})!"
            w = w.reshape(-1, self.qcfg.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if not self.qcfg.sym:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.qcfg.bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.qcfg.bits - 1) - 1
            min_int = -(2 ** (self.qcfg.bits - 1))
            scales = max_val / max_int
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)

        # Symmetric quantization produces signed int values (e.g. int4 in [-8, 7]),
        # which cannot be packed directly. To make it packable, we shift the signed
        # representation to unsigned by adding 2^(bits-1), i.e. q_u = q_s + 2^(bits-1).
        # This is equivalent to using an affine form with zero_point = 2^(bits-1),
        # without changing the numerical behavior of symmetric quantization.
        if self.qcfg.sym:
            zeros = torch.full_like(scales, 0 + 2 ** (self.qcfg.bits - 1))

        w = w.reshape(org_w_shape)

        return w, scales, zeros

    @torch.inference_mode()
    def _pseudo_quantize_tensor_into(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Writes pseudo-quantized values into a destination tensor without reallocating."""

        # Quantize `src` into `dst` without allocating a new tensor (mirrors pseudo_quantize_tensor)
        org_shape = src.shape
        if self.qcfg.group_size > 0:
            src_view = src.view(-1, self.qcfg.group_size)
            dst_view = dst.view(-1, self.qcfg.group_size)
        else:
            src_view = src.reshape(org_shape[0], -1)
            dst_view = dst.reshape_as(src_view)

        if not self.qcfg.sym:
            max_val = src_view.amax(dim=1, keepdim=True)
            min_val = src_view.amin(dim=1, keepdim=True)
            max_int = 2 ** self.qcfg.bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

            dst_view.copy_(src_view)
            dst_view.div_(scales)
            torch.round(dst_view, out=dst_view)
            dst_view.add_(zeros)
            dst_view.clamp_(min_int, max_int)
            dst_view.sub_(zeros)
            dst_view.mul_(scales)
        else:
            max_val = src_view.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            max_int = 2 ** (self.qcfg.bits - 1) - 1
            min_int = -(2 ** (self.qcfg.bits - 1))
            scales = max_val / max_int

            dst_view.copy_(src_view)
            dst_view.div_(scales)
            torch.round(dst_view, out=dst_view)
            dst_view.clamp_(min_int, max_int)
            dst_view.mul_(scales)


    def _select_awq_weight_restore_device(
        self,
        device: torch.device,
        linears2scale: List[nn.Linear],
    ) -> torch.device:
        """Choose where AWQ scale search keeps pristine restore weights."""

        if not getattr(self.qcfg, "scale_search_gpu_weight_restore", True):
            return torch.device(CPU)

        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return torch.device(CPU)

        required_bytes = sum(
            fc.weight.numel() * fc.weight.element_size()
            for fc in linears2scale
        )
        if required_bytes <= 0:
            return torch.device(CPU)

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except Exception:
            return device

        # Keep enough slack for the active module weights, temporary quantized
        # outputs, and allocator fragmentation. If that headroom is unavailable,
        # prefer the existing CPU restore path over risking an OOM.
        reserve_bytes = max(required_bytes, int(total_bytes * 0.05), 512 << 20)
        if free_bytes >= required_bytes + reserve_bytes:
            return device
        return torch.device(CPU)

    @staticmethod
    def _restore_awq_weight_from_master(fc: nn.Linear, master_weight: torch.Tensor) -> None:
        """Restore one AWQ search weight from its pristine master copy."""

        source = master_weight
        if source.dtype != fc.weight.dtype:
            source = source.to(dtype=fc.weight.dtype)
        fc.weight.copy_(source)

    @staticmethod
    def _accumulate_awq_chunk_loss(
        ref_chunk: torch.Tensor,
        int_w_output: torch.Tensor,
        *,
        clamp: bool = True,
    ) -> Tuple[float, int]:
        """Score one AWQ output chunk while reusing the quantized output tensor as scratch."""

        if clamp:
            int_w_output.clamp_(
                torch.finfo(int_w_output.dtype).min,
                torch.finfo(int_w_output.dtype).max,
            )
        ref_chunk = ref_chunk.to(device=int_w_output.device, dtype=int_w_output.dtype)
        # int_w_output is not reused after scoring this candidate chunk. Store
        # the signed error in-place, then square the FP32 view in-place to avoid
        # per-ratio diff and pow temporaries.
        int_w_output.sub_(ref_chunk)
        diff = int_w_output.float()
        diff.pow_(2)
        return diff.sum().item(), diff.numel()

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
        *,
        refine_steps: Optional[int] = None,
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        # Keep the canonical AWQ grid as the first stage. Group-wise rounding
        # makes the reconstruction objective non-smooth, so the winning alpha
        # is then refined with deterministic local grids instead of assuming a
        # unimodal loss (as golden-section/ternary search would).
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")
        evaluated_errors: Dict[float, float] = {}

        try:
            device = next(module2inspect.parameters()).device
        except StopIteration:
            device = x.device
        except Exception:
            device = x.device

        # Clone original weights once so candidate ratios can mutate in-flight
        # module weights without load_state_dict overhead. By default this uses
        # a GPU master copy when headroom allows, avoiding repeated CPU->GPU restores
        # during AWQ ratio search. Low-memory runs keep the CPU restore path.
        restore_device = self._select_awq_weight_restore_device(device, linears2scale)
        try:
            orig_weights_master: Dict[nn.Linear, torch.Tensor] = {
                fc: fc.weight.detach().to(
                    device=restore_device,
                    dtype=fc.weight.dtype,
                    copy=True,
                ).contiguous()
                for fc in linears2scale
            }
        except RuntimeError as exc:
            if restore_device.type != "cuda" or "out of memory" not in str(exc).lower():
                raise
            restore_device = torch.device(CPU)
            orig_weights_master = {
                fc: fc.weight.detach().to(
                    device=restore_device,
                    dtype=fc.weight.dtype,
                    copy=True,
                ).contiguous()
                for fc in linears2scale
            }

        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        prev_scale_hint = getattr(self._scale_context, "prev_scale", None)
        if prev_scale_hint is not None:
            w_mean = w_mean * float(prev_scale_hint)

        # This flag also controls how each candidate scale is scored: either stream
        # ref/int outputs chunk-by-chunk for lower memory, or reuse the legacy eager
        # tensor-vs-tensor loss computation.
        use_chunked_scale_search = getattr(self.qcfg, "scale_search_chunked_activations", True)
        if refine_steps is None:
            refine_steps = int(getattr(self.qcfg, "scale_search_refine_steps", 0))

        def _ratio_key(ratio: float) -> float:
            # Refinement ratios are produced through different arithmetic paths.
            # A rounded key prevents duplicate forwards for numerically identical
            # candidates while retaining far more precision than the search uses.
            return round(min(max(float(ratio), 0.0), 1.0), 12)

        def _evaluate_ratio(ratio: float) -> float:
            nonlocal best_error, best_ratio, best_scales

            ratio = _ratio_key(ratio)
            previous_error = evaluated_errors.get(ratio)
            if previous_error is not None:
                return previous_error

            # NOTE: s^-1 * x is fused here, according to the AWQ paper.
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()

            # Avoid scale values that overflow before creating the broadcast view.
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1
            scales_view = scales.view(1, -1).to(device)

            try:
                # Q(W * s). Temporarily apply the candidate scale and quantize
                # the in-flight weights without allocating another weight tensor.
                for fc in linears2scale:
                    fc.weight.mul_(scales_view)
                    self._pseudo_quantize_tensor_into(fc.weight, fc.weight)
                    fc.weight.div_(scales_view)

                if use_chunked_scale_search:
                    # Compare reference and quantized outputs chunk-by-chunk so
                    # refinement does not increase peak activation memory.
                    total_loss = 0.0
                    total_elements = 0
                    for ref_chunk, int_w_output in zip(
                        self._iter_reference_output_chunks(x, fp16_output),
                        self._iter_module_forward_outputs(x, module2inspect, kwargs),
                    ):
                        chunk_loss, chunk_elements = self._accumulate_awq_chunk_loss(ref_chunk, int_w_output)
                        total_loss += chunk_loss
                        total_elements += chunk_elements

                    loss = total_loss / max(total_elements, 1)
                else:
                    int_w_output = self._module_forward(x, module2inspect, kwargs)
                    int_w_output = int_w_output.clip(
                        torch.finfo(int_w_output.dtype).min,
                        torch.finfo(int_w_output.dtype).max,
                    )
                    loss = self._compute_loss(fp16_output, int_w_output, device)
            finally:
                # Candidate evaluation is deliberately transactional: even a
                # failed forward must not leave partially quantized weights.
                for fc in linears2scale:
                    self._restore_awq_weight_from_master(fc, orig_weights_master[fc])

            history.append(loss)
            evaluated_errors[ratio] = loss
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                # Each ratio creates a fresh scales tensor and never mutates it
                # after scoring, so retaining the winner avoids a clone.
                best_scales = scales
            return loss

        try:
            # Refined search includes alpha=1.0, making its coarse candidates a
            # strict superset of the historical range(n_grid) search. Disabling
            # refinement retains the exact legacy 20-candidate path for A/Bs.
            coarse_count = n_grid + 1 if refine_steps > 1 else n_grid
            coarse_ratios = [_ratio_key(index / n_grid) for index in range(coarse_count)]
            for ratio in coarse_ratios:
                _evaluate_ratio(ratio)

            if refine_steps > 1:
                def _rank_key(ratio: float) -> Tuple[float, float]:
                    loss = evaluated_errors[_ratio_key(ratio)]
                    return (loss if math.isfinite(loss) else float("inf"), ratio)

                # Preserve multiple promising basins because group-wise integer
                # rounding can create narrow minima that a single local search
                # misses. Three coarse cells remain far cheaper than a dense grid.
                centers = sorted(coarse_ratios, key=_rank_key)[:3]
                radius = 0.5 / n_grid

                # Two refinement stages give a default final spacing of 5e-4
                # (20 coarse intervals, five subdivisions per half-cell) while
                # evaluating at most 81 candidates instead of a 2001-point grid.
                for _ in range(2):
                    spacing = radius / refine_steps
                    next_centers = []
                    for center in centers:
                        local_ratios = sorted({
                            _ratio_key(center + offset * spacing)
                            for offset in range(-refine_steps, refine_steps + 1)
                        })
                        for ratio in local_ratios:
                            _evaluate_ratio(ratio)
                        next_centers.append(min(local_ratios, key=_rank_key))

                    # Adjacent coarse cells can converge to the same refined
                    # minimum. Deduplicate them before the next stage.
                    centers = list(dict.fromkeys(next_centers))
                    radius = 0.5 * spacing
        finally:
            # Reset weights one final time so callers always see pristine FP
            # weights, including when candidate generation or scoring raises.
            for fc in linears2scale:
                self._restore_awq_weight_from_master(fc, orig_weights_master[fc])
            orig_weights_master.clear()

        if best_ratio == -1:
            log.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu(), best_error

    @torch.inference_mode()
    def _compute_loss(
            self,
            fp16_output: torch.Tensor,
            int_w_output: torch.Tensor,
            device: torch.device,
    ):
        """Computes chunked mean-squared reconstruction loss under a memory cap."""

        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            int_w_chunk = int_w_chunk.to(device)
            chunk_loss, _ = self._accumulate_awq_chunk_loss(fp16_chunk, int_w_chunk, clamp=False)
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.inference_mode()
    def _module_forward(
            self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        """Runs a module forward with sanitized kwargs and optional micro-batching."""

        outputs = list(self._iter_module_forward_outputs(x, module, module_kwargs))
        if not outputs:
            return torch.empty_like(x)
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def _iter_module_forward_outputs(
            self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ):
        """Yields forward outputs chunk-by-chunk to reduce peak activation memory."""

        module_kwargs = dict(module_kwargs)

        target_device = None
        try:
            target_device = next(module.parameters()).device
        except StopIteration:
            target_device = None
        except Exception:
            target_device = None

        for key, value in list(module_kwargs.items()):
            if isinstance(value, torch.Tensor):
                if target_device is not None and value.device != target_device:
                    module_kwargs[key] = value.to(target_device)
            elif isinstance(value, (list, tuple)):
                converted = []
                changed = False
                for item in value:
                    if isinstance(item, torch.Tensor) and target_device is not None and item.device != target_device:
                        converted.append(item.to(target_device))
                        changed = True
                    else:
                        converted.append(item)
                if changed:
                    module_kwargs[key] = type(value)(converted)

        forward_kwargs = self._forward_kwarg_names(module)
        supports_position_ids = "position_ids" in forward_kwargs
        supports_position_embeddings = "position_embeddings" in forward_kwargs

        if x.dim() == 2 and (supports_position_ids or supports_position_embeddings):
            x = x.unsqueeze(0)

        seq_len = None
        batch_dim = None
        if x.dim() >= 2:
            batch_dim = x.shape[0]
            seq_len = x.shape[1] if x.dim() >= 3 else x.shape[0]
        else:
            batch_dim = 1
            seq_len = x.shape[0]

        rotary = self._get_root_rotary()
        if seq_len is not None and rotary is not None and supports_position_embeddings:
            rotary = self._get_rotary_for_device(target_device or x.device)
            rotary_device = self._get_rotary_device(rotary, target_device or x.device)
            pos_ids = module_kwargs.get("position_ids") if supports_position_ids else None
            if not supports_position_ids:
                pos_ids = None
            if pos_ids is None or pos_ids.shape[-1] != seq_len:
                pos_values = torch.arange(seq_len, device=rotary_device or x.device, dtype=torch.long)
                if x.dim() >= 2:
                    pos_values = pos_values.unsqueeze(0).expand(batch_dim, -1)
                if supports_position_ids:
                    module_kwargs["position_ids"] = pos_values
                pos_for_rotary = pos_values
            else:
                pos_for_rotary = pos_ids.to(rotary_device or pos_ids.device)
                if supports_position_ids:
                    module_kwargs["position_ids"] = pos_for_rotary

            x_for_rotary = x if rotary_device is None else x.to(rotary_device)
            module_kwargs["position_embeddings"] = rotary(x_for_rotary, pos_for_rotary)
        elif supports_position_ids and seq_len is not None and "position_ids" not in module_kwargs:
            pos_values = torch.arange(seq_len, device=target_device or x.device, dtype=torch.long)
            if x.dim() >= 2:
                pos_values = pos_values.unsqueeze(0).expand(batch_dim, -1)
            module_kwargs["position_ids"] = pos_values

        effective_quant_batch_size = self._quant_batch_size if self._quant_batch_size and self._quant_batch_size > 0 else None

        if (
            effective_quant_batch_size is None
            or x.dim() == 0
            or x.shape[0] <= effective_quant_batch_size
        ):
            x_forward = x.to(target_device) if target_device is not None and x.device != target_device else x
            module_output = module(x_forward, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
            yield module_output
            return

        full_batch_size = int(x.shape[0]) if x.dim() > 0 else 1

        def _slice_value(val, start, length):
            """Slices batch-shaped kwargs to match a micro-batched forward chunk."""

            if isinstance(val, torch.Tensor):
                if val.ndim > 0 and val.shape[0] == full_batch_size:
                    return val[start:start + length]
                return val
            if isinstance(val, (list, tuple)):
                sliced = [_slice_value(item, start, length) for item in val]
                return type(val)(sliced)
            return val

        batch_offset = 0
        for x_partial in torch.split(x, effective_quant_batch_size, dim=0):
            x_forward = x_partial.to(target_device) if target_device is not None and x_partial.device != target_device else x_partial
            partial_kwargs = {
                key: _slice_value(value, batch_offset, x_forward.shape[0])
                for key, value in module_kwargs.items()
            }
            partial_output = module(x_forward, **partial_kwargs)
            if isinstance(partial_output, tuple):
                partial_output = partial_output[0]
            yield partial_output
            batch_offset += x_forward.shape[0]

    def _iter_reference_output_chunks(self, x: torch.Tensor, reference_output):
        """Normalizes reference outputs to the same chunking contract as forward outputs."""

        if isinstance(reference_output, torch.Tensor):
            effective_quant_batch_size = self._quant_batch_size if self._quant_batch_size and self._quant_batch_size > 0 else None
            if (
                effective_quant_batch_size is None
                or x.dim() == 0
                or x.shape[0] <= effective_quant_batch_size
            ):
                yield reference_output
                return

            for ref_chunk in torch.split(reference_output, effective_quant_batch_size, dim=0):
                yield ref_chunk
            return

        for ref_chunk in reference_output:
            yield ref_chunk

    def apply_quant(
        self,
        named_linears: Dict[str, NamedModule],
        scales_list,
        *,
        input_features: Optional[Dict[str, torch.Tensor]] = None,
        adjacent_references: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Pseudo-quantizes selected linears and stages AWQ tensors for packing."""

        if (input_features is None) != (adjacent_references is None):
            raise ValueError("AWQ AdjacentExact requires both input features and reference weights.")
        adjacent_context_supplied = input_features is not None

        start_time = time.time()
        for name, named_module in named_linears.items():
            base_title = f"Quantizing {named_module.name} in layer"
            self.draw_progress(base_title)

            linear_layer = self.resolve_quant_source_module(named_module)
            move_to(linear_layer, device=_resolve_module_exec_device(linear_layer))

            tp_info = named_module.state.get("tp_pad_info")
            pad_cols = 0
            original_cols = linear_layer.weight.data.shape[1]
            if isinstance(tp_info, dict):
                pad_cols = int(tp_info.get("pad_cols", 0) or 0)
                original_cols = int(tp_info.get("original_columns", original_cols))

            weight_for_quant = linear_layer.weight.data
            if pad_cols:
                pad = weight_for_quant.new_zeros(weight_for_quant.shape[0], pad_cols)
                weight_for_quant = torch.cat((weight_for_quant, pad), dim=1)

            adjacent_reference = None
            adjacent_activations = None
            if self.adjacent_model is not None:
                if not adjacent_context_supplied:
                    self._record_adjacent_skip(
                        named_module,
                        "AWQ fallback quantization has no scaled activation/reference pair.",
                    )
                else:
                    assert adjacent_references is not None and input_features is not None
                    if name not in adjacent_references or name not in input_features:
                        raise RuntimeError(
                            f"AWQ AdjacentExact context is missing module `{named_module.full_name}`."
                        )
                    adjacent_reference = adjacent_references[name].to(
                        device=weight_for_quant.device,
                        dtype=weight_for_quant.dtype,
                    )
                    adjacent_activations = input_features[name]
                    if adjacent_reference.shape[0] != weight_for_quant.shape[0]:
                        raise ValueError(
                            f"AWQ AdjacentExact reference rows do not match `{named_module.full_name}`."
                        )
                    if pad_cols:
                        reference_pad = adjacent_reference.new_zeros(adjacent_reference.shape[0], pad_cols)
                        adjacent_reference = torch.cat((adjacent_reference, reference_pad), dim=1)
                    if adjacent_reference.shape != weight_for_quant.shape:
                        raise ValueError(
                            f"AWQ AdjacentExact reference shape does not match `{named_module.full_name}`."
                        )

            wq, scales, zeros = self.pseudo_quantize_tensor(
                weight_for_quant
            )

            if adjacent_reference is not None and adjacent_activations is not None:
                from ..quantization.adjacent_model import apply_adjacent_awq_hybrid

                wq, adjacent_stats = apply_adjacent_awq_hybrid(
                    module_name=named_module.full_name,
                    reference_weight=adjacent_reference,
                    activations=adjacent_activations,
                    classic_quantized=wq,
                    scales=scales,
                    zeros=zeros,
                    bits=int(self.qcfg.bits),
                    group_size=int(self.qcfg.group_size),
                    config=self.adjacent_model,
                )
                self.adjacent_model.record(adjacent_stats)

            if pad_cols:
                wq = wq[:, :original_cols]
                if self.qcfg.group_size > 0:
                    valid_groups = max(1, math.ceil(original_cols / self.qcfg.group_size))
                    scales = scales[:, :valid_groups]
                    if zeros is not None:
                        zeros = zeros[:, :valid_groups]
                else:
                    scales = scales[:, :original_cols]
                    if zeros is not None:
                        zeros = zeros[:, :original_cols]

            named_module.stream_state_payload_to_cpu(
                {
                    "q_scales": scales,
                    "q_zeros": zeros,
                },
            )
            del scales, zeros

            if self.calculate_w_wq_diff:
                if named_module.weight.data.dtype == torch.float16:
                    # diff in float16
                    w_wq_diff = linear_layer.weight.data - wq
                else:
                    # diff in float32
                    w_wq_diff = linear_layer.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)

                named_module.state.update({
                    "w_wq_diff": w_wq_diff,
                })

            named_module.state.update({
                "wq": wq,  # fp16, quantized weight but not int4 (packed qweight)
            })

            if isinstance(tp_info, dict):
                named_module.state.pop("tp_pad_info", None)

            linear_layer.weight.data = wq

            # records
            duration = time.time() - start_time

            avg_loss_value = None
            for _, layer_names, _, loss in scales_list:
                if any(named_module.name in layer_name for layer_name in layer_names):
                    avg_loss_value = loss
                    break

            if avg_loss_value is None:
                # Scaling not applied for this layer in AWQ; no meaningful loss metric.
                loss_summary = "not applicable"
            else:
                loss_summary = f"{avg_loss_value:.10f}"

            # TODO "loss" and "nsamples" may not be consistent with the semantics of gptq quantization.
            stat = {
                PROCESS_LOG_NAME: self.name(),
                PROCESS_LOG_LAYER: named_module.layer_index,
                PROCESS_LOG_MODULE: named_module.name,
                MODULE_FEATURE_COLUMN: self.module_feature_summary(named_module),
                DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(named_module),
                QUANT_LOG_LOSS: loss_summary,
                QUANT_LOG_NSAMPLES: f"{self._nsamples_total}",
                # QUANT_LOG_DAMP: f"{damp_percent:.5f}",
                PROCESS_LOG_TIME: f"{duration:.3f}",
                # PROCESS_LOG_FWD_TIME: f"{self.fwd_time:.3f}",
                PROCESS_USED_MEMORY: self.device_memory_report(),
            }
            with self.lock:
                self.module_names.append(f"layer-{named_module.layer_index}-{named_module.name}")
                self.log.append(stat)

            # Log the new row
            self.log_new_row(stat)

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = self._forward_kwarg_names(module)
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Registers a module with its layer state and initializes input capture buckets."""

        # entire module is skipped
        if self.qcfg.dynamic_get(layer_name=module.full_name) is False:
            return

        # Track the most recent preference so the processor can decide whether
        # to fall back to simple quantization when activations are missing.
        self.fallback = normalize_fallback(fallback, self.qcfg.fallback)
        layer_state = self._get_layer_state(module.layer_index)
        with layer_state.lock:
            layer_state.modules[module.name] = module
            layer_module_ref = module.state.get("layer_module")
            if layer_state.layer_module is None and layer_module_ref is not None:
                layer_state.layer_module = layer_module_ref
            effective_layer = layer_state.layer_module or layer_module_ref
            if not layer_state.pending_modules and effective_layer is not None:
                try:
                    all_linears = find_modules(effective_layer)
                except Exception:
                    all_linears = {}
                layer_state.pending_modules.update(all_linears.keys())
            layer_state.pending_modules.add(module.name)
        with self.lock:
            entry = self.tasks.get(module.name)
            if entry is None:
                self.tasks[module.name] = {"inputs": [], "batch_indices": [], "input_cache_keys": []}
            else:
                entry.setdefault("inputs", [])
                entry.setdefault("batch_indices", [])
                entry.setdefault("input_cache_keys", [])

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether preprocessing excluded this module from AWQ work."""

        return not self.tasks.get(module.name, False)

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that caches module input activations for AWQ."""

        def hook(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            """Records the module input tensor for later AWQ scale and clip search."""

            if not inp:
                return
            feature = inp
            if isinstance(feature, (tuple, list)) and feature:
                feature = feature[0]
            self._record_input_feature(name, feature)
        return hook

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Accumulates subset progress and triggers layer quantization when ready."""

        self._refresh_forward_kwargs_from_cache()
        layer_index = module.layer_index
        state = self._get_layer_state(layer_index)

        with state.lock:
            if subset is not None:
                state.modules.update(subset)
                if state.layer_module is None:
                    for candidate in subset.values():
                        layer_module_ref = candidate.state.get("layer_module")
                        if layer_module_ref is not None:
                            state.layer_module = layer_module_ref
                            break

            if subset_total is not None:
                state.subset_total = subset_total
            if subset_index is not None:
                state.processed_subsets.add(subset_index)

            if module is not None:
                state.pending_modules.discard(module.name)

            if previous_subset:
                state.previous_weight_scale = self._capture_previous_subset_scale(previous_subset)

            # Only gate on modules that were scheduled for subset processing to
            # avoid early quantization while parallel workers are still running.
            pending_in_scope = state.pending_modules
            if state.modules:
                pending_in_scope = state.pending_modules.intersection(state.modules.keys())

            should_quantize = (
                not state.quantized
                and bool(state.modules)
                and not pending_in_scope
                and (
                    state.subset_total is None
                    or len(state.processed_subsets) >= state.subset_total
                )
            )

            if should_quantize:
                self._quantize_layer(layer_index, state)

    # submodule_finalized is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Delegates AWQ module packing to the shared pack helper."""

        return self.pack_module(module)

    def pack_module(self, module):
        """Creates the AWQ quantized module and packs saved scales/zero-points into it."""

        # generate complete, safe to move to cpu
        # cleanup all memory or states vars persistently added by this processor
        module.stream_sync()
        with (self.lock):
            # if calculate_w_wq_diff is enabled (eora), we need to revert our original wq
            if self.calculate_w_wq_diff:
                module.weight.data = module.state.pop("wq").to(CPU)

            module.state.pop("w", None)  #
            module.state.pop("w_wq_diff", None)

            if module.state.get("q_zeros") is None:
                print("modddd", module.full_name)
            # need to clone to due to steamed pinned memory and access on diff thread
            q_zeros = module.state.pop("q_zeros").clone()
            q_scales = module.state.pop("q_scales").clone()
        assert q_zeros.device == CPU
        assert q_scales.device == CPU
        quant_linear_cls = self._resolve_qlinear_kernel(module.full_name)
        module_label = getattr(module, "full_name", getattr(module, "name", ""))
        parent_key = getattr(module, "full_name", getattr(module, "name", None))
        # Capture the original leaf and use the module returned by
        # create_quant_module directly. Avoid model-wide scans that race with
        # concurrent sibling replacements under a shared parent.
        original_layer = module.module
        # replace module with quantized module
        timer = getattr(self.gptq_model, "quant_region_timer", None)
        create_start = time.perf_counter() if timer is not None else None
        with log_time_block(
                "create_quant_module",
                logger=log,
                module_name=module_label,
        ):
            with parent_module_lock(parent_key):
                qmodule = create_quant_module(
                    name=module.full_name,
                    linear_cls=quant_linear_cls,
                    bits=self.qcfg.runtime_bits,
                    desc_act=self.qcfg.desc_act,
                    dynamic=self.qcfg.dynamic,
                    group_size=self.qcfg.group_size,
                    module=self.gptq_model.model,
                    submodule=module,
                    sym=self.qcfg.sym,
                    device=self.qcfg.device,
                    lm_head_name=self.gptq_model.lm_head,
                    pack_dtype=self.qcfg.pack_dtype,
                    format=self.format,
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

        # pack module using the already-installed quantized module and the
        # original leaf captured above. One-entry dicts avoid `find_modules()`.
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
                    q_g_idx=None,
                    layers=layers,
                    quant_linear_cls=quant_linear_cls,
                    lock=None,
                    quantize_config=self.qcfg,
                )
        if timer is not None and pack_start is not None:
            timer.record(
                "submodule_finalize_pack",
                time.perf_counter() - pack_start,
                source=f"{module_label} [{packer_label or 'module.pack_original'}]",
            )

        return qmodule

    def finalize(self, model: BaseQModel, **kwargs):
        """Marks the model as AWQ-quantized and runs shared finalization logic."""

        # set quantized state
        model.quantized = True

        model.quantize_config.method = METHOD.AWQ

        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Ensures AWQ received calibration data before the quantization loop starts."""

        if self.calibration_dataset is None:
            raise ValueError("GPTQProcessor's calibration_dataset must be provided.")
        else:
            return True

    @classmethod
    def name(cls) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "awq"

    def has_captured_input_ids(self, name: str) -> bool:
        """Reports whether a module has any non-empty captured AWQ activations."""

        entry = self.tasks.get(name) or {}
        tensors: List[torch.Tensor] = entry.get("inputs", [])
        return tensors is not None and len(tensors) > 0 and all(t.numel() > 0 for t in tensors)
