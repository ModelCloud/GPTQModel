# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import functools
import inspect
import math
import re
import threading
import time
import weakref
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

import torch
import transformers
from torch import nn
from torch.nn import Module

from .._alloc_conf import clamp_budget_below_allocator_split
from ..looper.loop_processor import (
    DTYPE_SIZE_COLUMN,
    MODULE_FEATURE_COLUMN,
    CalibrationTypes,
    CheckpointSupport,
    ForwardConfig,
    LoopProcessor,
    ProcessorLifecycleConfig,
    active_padding_mask,
    bind_resolved_module_config,
    get_resolved_module_config,
)
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import DEVICE, SUPPORTS_MODULE_TYPES
from ..models.module_plan import _model_module_plan
from ..models.writer import (
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    PROCESS_USED_MEMORY,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from ..nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from ..nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from ..nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from ..nn_modules.qlinear.torch_awq import AwqTorchLinear
from ..quantization.awq.quantize.scale import apply_clip, apply_scale
from ..quantization.awq.utils.module import append_str_prefix, get_op_by_name, get_op_name
from ..quantization.awq.utils.packing_utils import source_weight_out_in
from ..quantization.config import FORMAT, METHOD, CalibrationConfig, MSEConfig, QuantizeConfig, resolve_quant_format
from ..utils.attn_mask import (
    align_single_head_capture,
    exclude_attention_sinks,
    normalize_seq_mask,
)
from ..utils.ctx import ctx
from ..utils.device import get_device
from ..utils.fallback import normalize_fallback
from ..utils.inspect import safe_kwargs_call
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import (
    create_quant_module,
    find_modules,
    get_module_by_name_prefix,
    install_quant_module,
    move_to,
    pack_module,
    resolve_position_limit,
    restore_moe_topk,
    set_moe_topk,
)
from ..utils.module_locks import parent_module_lock
from ..utils.torch import CPU


log = setup_logger()


class _AWQScaleGroupError(ValueError):
    """A declared group cannot be evaluated safely with its current replay."""


#: ``RuntimeError`` messages Torch emits when a replay's tensor geometry does
#: not fit the module it is fed to. A packed token-row replay that exceeds a
#: module's fixed context (for example ``GPTNeoSelfAttention``'s
#: ``[max_position_embeddings, max_position_embeddings]`` causal buffer) raises
#: exactly these instead of a shape check in this processor, so they must reach
#: the configured scale-group fallback rather than abort ``quantize()`` (#5085).
_REPLAY_GEOMETRY_ERROR_MARKERS = (
    "must match the size of tensor",
    "expanded size of the tensor",
    "shapes cannot be multiplied",
    "mat1 and mat2 shapes",
)


def _is_replay_geometry_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a Torch tensor-geometry failure from a replay forward.

    Only ``RuntimeError`` is considered, and only messages that name a broadcast
    or matrix-shape mismatch: memory, kernel and numerical failures must keep
    propagating so they are not silently downgraded to plain quantization.
    """

    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _REPLAY_GEOMETRY_ERROR_MARKERS)


# Carries the retained rows' real positions past `_sanitize_kwargs` for the
# rotary rebuild. Attention modules that take `position_embeddings` do not
# accept `position_ids`, so the ordinary key is dropped before
# `_prepare_module_forward` would otherwise fall back to `arange`.
_AWQ_ROTARY_POSITION_IDS = "_awq_rotary_position_ids"
# Kwargs whose entries line up one-for-one with the replayed token rows. AWQ
# replays a packed sample of captured rows, so these values must be packed with
# the same provenance (and their row count validated) or a router that indexes
# them (`hash_moe`: `tid2eid[input_ids]`) sees a different token count than the
# hidden states and raises deep inside the block (#4880).
_AWQ_TOKEN_ALIGNED_KWARGS = ("input_ids",)


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
    previous_weight_scale_source: Optional[Dict[str, NamedModule]] = None
    quantized: bool = False
    pending_modules: Set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _TokenRowPlan:
    """Precomputed token-row quota and sampling plan for a layer."""

    lengths: List[int]
    budget: int
    max_tokens: int
    quotas: List[int]
    local_indices: Dict[int, torch.Tensor]


def _accumulate_awq_weight_mean(
    layers: List[nn.Linear],
    group_size: int | List[int],
) -> Tuple[torch.Tensor, int]:
    """Accumulates normalized per-channel weight sums across a group of linears.

    A scaling group is anchored on one source tensor (a norm, a linear or a
    MoE block), not on one device. With ``VramStrategy.BALANCED`` the routed
    experts of a norm-anchored MoE group live on every device of the MoE pool,
    so each member's reduction is gathered onto the accumulator's device
    instead of relying on an implicit same-device add.
    """

    if not layers:
        raise ValueError("Expected at least one linear layer to compute the AWQ weight mean.")

    first_weight = layers[0].weight.detach()
    num_channels = first_weight.shape[1]
    group_sizes = (
        [group_size] * len(layers)
        if isinstance(group_size, int)
        else list(group_size)
    )
    if len(group_sizes) != len(layers):
        raise ValueError("Expected one AWQ group size per linear layer.")

    w_sum = torch.zeros(num_channels, dtype=torch.float32, device=first_weight.device)
    row_count = 0

    for layer, configured_group_size in zip(layers, group_sizes):
        weight = layer.weight.detach()
        if weight.shape[1] != num_channels:
            raise ValueError(
                f"Expected consistent in_features across layers ({num_channels}), "
                f"got {weight.shape[1]} for layer {layer}."
            )
        effective_group_size = (
            configured_group_size if configured_group_size > 0 else num_channels
        )
        if num_channels % effective_group_size != 0:
            raise ValueError(
                f"Expected in_features ({num_channels}) to be divisible by "
                f"group_size ({effective_group_size})."
            )

        rows = weight.shape[0]
        weight_abs = weight.abs().to(dtype=torch.float32)
        weight_view = weight_abs.reshape(rows, -1, effective_group_size)
        group_scale = weight_view.amax(dim=2, keepdim=True)
        weight_view.div_(group_scale.add_(1e-6))
        # `to()` is a no-op on the accumulator's own device, so a single-device
        # group keeps the previous allocation-free add.
        w_sum += weight_abs.sum(dim=0, dtype=torch.float32).to(w_sum.device)
        row_count += rows

    return w_sum, row_count


def _compute_awq_weight_mean(
    layers: List[nn.Linear],
    group_size: int | List[int],
) -> torch.Tensor:
    """Returns the average normalized per-channel weight magnitude for AWQ scaling."""

    w_sum, row_count = _accumulate_awq_weight_mean(layers, group_size)
    first_weight = layers[0].weight.detach()
    if row_count == 0:
        return torch.zeros(first_weight.shape[1], dtype=first_weight.dtype, device=first_weight.device)
    return (w_sum / row_count).to(first_weight.dtype)


def _awq_affine_qparams(
    w: torch.Tensor, bits: int, *, scaled_zero_offset: bool = False
):
    """Choose affine metadata whose used codes can be decoded in the input dtype."""

    zero = torch.zeros_like(w[:, :1])
    maximum = torch.maximum(w.amax(dim=1, keepdim=True), zero)
    minimum = torch.minimum(w.amin(dim=1, keepdim=True), zero)
    max_code = (1 << bits) - 1
    limits = torch.finfo(w.dtype)
    scales = (maximum - minimum).clamp(min=1e-5) / max_code

    # A finite input range can overflow its low-precision subtraction. The
    # maximum code can also decode outside the dtype even when the scale itself
    # is finite (for example 15 * 4368 > FP16_MAX).
    scale_limit = limits.max / max_code
    repair_mask = ~torch.isfinite(scales) | (scales.float() > scale_limit)
    repair = bool(repair_mask.any())
    if repair:
        wide_scales = (
            maximum.float() / max_code - minimum.float() / max_code
        ).clamp(min=1e-5 / max_code)
        scales = torch.where(~torch.isfinite(scales), wide_scales.to(w.dtype), scales)

        # One-sided groups can keep their endpoint code by lowering the scale
        # one representable step below the largest safe scale.
        one_sided = (minimum == 0) | (maximum == 0)
        safe_scale = torch.tensor(
            scale_limit * (1 - limits.eps), dtype=w.dtype, device=w.device
        )
        scales = torch.where(
            one_sided & (scales.float() > scale_limit),
            safe_scale,
            scales,
        )

    zeros = (-torch.round(minimum / scales)).clamp_(0, max_code)
    if repair and scaled_zero_offset:
        # GEMV-fast stores -scale * zero in FP16. Both that offset and every
        # possible decoded code must fit, including after zero-point rounding.
        for _ in range(max_code + 1):
            distance = torch.maximum(zeros, max_code - zeros).float().clamp(min=1)
            safe_scale = (
                (limits.max / distance) * (1 - limits.eps)
            ).to(w.dtype)
            scales = torch.where(scales < safe_scale, scales, safe_scale)
            zeros = (-torch.round(minimum / scales)).clamp_(0, max_code)
            distance = torch.maximum(zeros, max_code - zeros).float()
            offset_wide = -scales.float() * zeros.float()
            offset_exact = offset_wide.to(w.dtype).float() == offset_wide
            # A rounded offset biases zero-valued weights in the native half
            # FMA. For boundary groups, move down to the nearest scale whose
            # offset is exactly representable in the checkpoint dtype.
            needs_step = (scales.float() * distance > limits.max) | (repair_mask & ~offset_exact)
            if not bool(needs_step.any()):
                break
            scales = torch.where(
                needs_step,
                (scales.float() * (1 - limits.eps)).to(w.dtype),
                scales,
            )
        else:
            raise ValueError("AWQ scaled zero point cannot be represented")

    return scales, zeros, repair


def _awq_bound_affine_codes(
    codes: torch.Tensor, zeros: torch.Tensor, scales: torch.Tensor, max_code: int
) -> torch.Tensor:
    """Move an overflowing endpoint code one step into the finite range."""

    limit = torch.finfo(codes.dtype).max
    decoded = (codes.float() - zeros.float()) * scales.float()
    codes = torch.where(decoded < -limit, codes + 1, codes)
    codes = torch.where(decoded > limit, codes - 1, codes)
    return codes.clamp_(0, max_code)


#: Sentinel for "the root rotary has not been resolved yet". A resolved `None`
#: must be cached too, otherwise every replayed module rescans the model.
_ROTARY_UNRESOLVED = object()

#: Sentinel for "the model's declared position limit has not been resolved yet".
#: A resolved `None` must be cached too: configs without a limit are common.
_POSITION_LIMIT_UNRESOLVED = object()

#: Wrapper segments that own their own positional embeddings. A vision tower's
#: rotary embedding (for example `model.visual.rotary_pos_emb` on Qwen2-VL)
#: never builds the decoder's `position_embeddings`.
_NON_DECODER_TOWER_SEGMENTS = frozenset({
    "visual",
    "vision",
    "vision_tower",
    "vision_model",
    "image_encoder",
    "image_tower",
    "audio",
    "audio_tower",
    "audio_encoder",
    "multi_modal_projector",
    "projector",
})


def _is_rotary_embedding_module(module: nn.Module, leaf_name: str) -> bool:
    """Whether ``module`` builds ``(cos, sin)`` from a rotary frequency buffer."""

    leaf = leaf_name.lower()
    class_name = type(module).__name__.lower()
    named_rotary = leaf in ("rotary_emb", "rotary", "rotary_embedding")
    if not named_rotary and not ("rotary" in class_name and "embedding" in class_name):
        return False
    if "inv_freq" in getattr(module, "_buffers", {}):
        return True
    # Some rotaries keep no buffer (they recompute from `base`) but still take
    # `(hidden_states, position_ids[, layer_type])`.
    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _forward_accepts_var_kwargs(module: nn.Module) -> bool:
    """Whether ``module.forward`` collects unknown keywords into ``**kwargs``.

    A forward declared as ``(hidden_states, past_key_value=None, **kwargs)``
    reads its real inputs out of the catch-all: Hymba's ``HymbaBlock`` takes
    ``attention_mask``, ``position_ids``, ``kv_last_layer`` and ``use_swa``
    that way (#4891). Every name-based signature filter therefore has to treat
    such a module as accepting every key, not as accepting none of them.

    This inspects the *declared* signature: ``inspect.signature`` follows
    ``__wrapped__``, so an instance-level forward wrapper that carries
    ``functools.wraps(original_forward)`` is reported with the signature of the
    forward it replaces. That distinction keeps this predicate from mistaking
    AWQ's all-experts replay wrapper for a real ``**kwargs`` block (#5095).
    """

    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _is_recurrent_attention_layer_type(layer_type: Optional[str]) -> bool:
    """Whether ``layer_type`` is a recurrent / convolution mixer, not attention.

    ``transformers`` 5 maps each declared layer type to the function that builds
    its mask (``LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING``). The ``linear_attention``
    and ``conv`` entries build the mask with ``create_recurrent_attention_mask``,
    which returns ``None`` for an unpadded batch and a planar ``[batch, seq]``
    keep mask otherwise. Those mixers apply it with
    ``hidden * attention_mask[:, :, None]``, so they can never consume the
    additive ``[1, 1, T, T]`` grid an attention layer takes (issue #4671).
    """

    if not layer_type:
        return False
    try:
        from transformers.masking_utils import (
            LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING,
            create_recurrent_attention_mask,
        )
    except ImportError:  # pragma: no cover - older transformers
        return False
    builder = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING.get(layer_type)
    return builder is create_recurrent_attention_mask


def _kernel_shape_supported(
    kernel, in_features: int, out_features: int, group_size: Optional[int] = None
) -> bool:
    """Return whether ``kernel`` declares it can pack this module shape.

    Uses the kernel's own static shape contract
    (``SUPPORTS_{IN,OUT}_FEATURES_DIVISIBLE_BY`` through
    ``_validate_shape_constraints``) plus, when the kernel has one, its static
    group-size/shape rule (``_validate_group_size_shape``), rather than the full
    ``validate``: the full check also probes runtime availability, and a shape
    decision must not depend on whether a JIT extension happens to be built.
    """

    check = getattr(kernel, "_validate_shape_constraints", None)
    if callable(check):
        try:
            ok, _ = check(in_features=in_features, out_features=out_features)
        except Exception:
            return False
        if not ok:
            return False
    group_check = getattr(kernel, "_validate_group_size_shape", None)
    if group_size is not None and callable(group_check):
        try:
            ok, _ = group_check(
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
            )
        except Exception:
            return False
        if not ok:
            return False
    return True


def _resolve_module_sym(qcfg, module_name: Optional[str], resolved_cfg=None) -> bool:
    """Resolve one module's symmetry mode from its dynamic rule.

    A real ``sym=False`` override is a legitimate value, so the resolution
    cannot go through ``dynamic_get`` (which also returns ``False`` for an
    excluded module). The rule snapshot distinguishes the two cases.
    """

    if not module_name:
        return bool(qcfg.sym)
    if resolved_cfg is None:
        resolved_cfg = qcfg.resolve_module_config(module_name)
    if resolved_cfg.excluded:
        return bool(qcfg.sym)
    override = resolved_cfg.get("sym", qcfg.sym)
    return bool(qcfg.sym if override is None else override)


class AWQProcessor(LoopProcessor):
    """Captures activations and quantizes layers with the AWQ scaling workflow."""

    lifecycle = ProcessorLifecycleConfig(
        forward=ForwardConfig(
            required=True,
            replay_after_process=True,
            subset_early_stop=True,
            activation_capture=True,
        ),
        calibration=CalibrationTypes.EACH_MODULE_NEEDS_INPUT,
        checkpointing=CheckpointSupport.PRIMARY,
    )

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
        calibration: CalibrationConfig,
        gptq_model,
        model,
        forward: Optional[ForwardConfig] = None,
        calculate_w_wq_diff: bool = False,
    ):
        """Initializes AWQ processing, layer tracking, and kernel selection."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            forward=forward,
        )

        self.calculate_w_wq_diff = calculate_w_wq_diff
        # AWQ masks every capture itself in `_apply_active_keep_mask_to_feature`,
        # which also recognizes the `[B*S, H]` and MLA single-head layouts. Keep
        # the raw activation here so a routed-expert subset stays
        # distinguishable from a capture that was already filtered: a routed
        # subset can coincidentally have exactly `keep_mask.sum()` rows while
        # still carrying padding (#5360).
        self.preserve_batch_keep_mask = True
        self.avg_losses = []
        self.nsamples = 0
        self._nsamples_total = 0
        self._quant_batch_size = self.calibration.batch_size

        self._layer_states: Dict[int, _AWQLayerState] = {}
        self._layer_states_lock = threading.Lock()
        self._scale_context = threading.local()
        self.gptq_model = gptq_model

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
        self._feature_stats: Dict[str, Dict[str, Any]] = {}
        self._scale_feature_by_module: Dict[str, str] = {}
        self._feature_task_names: Set[str] = set()
        self._awq_feature_kwargs: Dict[str, Dict[str, Any]] = {}
        # Sibling projections (q/k/v, gate/up) are hooked separately but receive the
        # *same* source activation object in the same batch. Cache one host copy per
        # source tensor so the D2H transfer, the host allocation and the later
        # `torch.cat` are paid once per unique input instead of once per module.
        # Entries are keyed by (batch, id(source), source._version) and validated by
        # a weakref so a recycled id can never alias a different tensor.
        self._capture_host_cache: Dict[Tuple[Any, int, int], Tuple[Any, torch.Tensor]] = {}
        self._capture_sampled_cache: Dict[Any, torch.Tensor] = {}
        self._token_row_plans: Dict[Tuple[int, Optional[int]], _TokenRowPlan] = {}
        self._reservoir_locks: Dict[str, threading.Lock] = {}
        self._capture_host_batch: Any = None
        self._rotary_lock = threading.Lock()
        self._rotary_cache: Dict[str, nn.Module] = {}
        self._rotary_source_id: Optional[int] = None
        self._initialize_sample_counts()
        self._module_forward_kwargs.setdefault("attention_mask", None)
        # Preserve fallback preference so AWQ can optionally fall back when no calibration data or activations are available.
        self.fallback = qcfg.fallback
        self._ensure_group_size_divisibility()

    def _get_root_rotary(self) -> Optional[nn.Module]:
        """Returns the decoder's rotary module used to refresh position embeddings.

        Multimodal wrappers nest the decoder one level deeper than a text model
        (``model.language_model.rotary_emb`` on Qwen2-VL, Qwen2.5-VL, Gemma 3
        and Mistral 3), and most of those definitions do not declare
        ``rotary_embedding``. Looking only at ``model.rotary_emb`` therefore
        found nothing, so AWQ dropped the captured ``position_embeddings`` and
        the attention unpacked ``None``. Resolve the declaration first, then the
        known text-backbone layouts, then scan for a rotary embedding outside a
        vision/audio tower.
        """

        cached = getattr(self, "_root_rotary_resolved", _ROTARY_UNRESOLVED)
        if cached is not _ROTARY_UNRESOLVED:
            return cached
        rotary = self._resolve_root_rotary()
        self._root_rotary_resolved = rotary
        return rotary

    def _resolve_root_rotary(self) -> Optional[nn.Module]:
        declared = getattr(self.gptq_model, "rotary_embedding", None)
        if declared:
            rotary, _ = get_module_by_name_prefix(self.model, [declared])
            if rotary is not None:
                return rotary

        root = getattr(self.model, "model", self.model)
        for candidate in ("rotary_emb", "rotary"):
            rotary = getattr(root, candidate, None)
            if isinstance(rotary, nn.Module):
                return rotary

        return self._scan_root_rotary()

    def _scan_root_rotary(self) -> Optional[nn.Module]:
        """Find the decoder rotary when a wrapper nests it below ``model``."""

        named_modules = getattr(self.model, "named_modules", None)
        if not callable(named_modules):
            # Partial models in focused unit tests are not `nn.Module`s.
            return None
        best = None
        best_depth = None
        for name, module in named_modules():
            if not isinstance(module, nn.Module) or module is self.model:
                continue
            segments = name.split(".")
            if _NON_DECODER_TOWER_SEGMENTS & set(segments):
                continue
            # A rotary inside a numbered container belongs to one decoder layer
            # (Nanbeige/DeciLM). Those are old-style `(x, seq_len)` modules and
            # are handled by the module-local fallback in
            # `_prepare_module_forward`; they are not the outer forward's rotary.
            if any(segment.isdigit() for segment in segments):
                continue
            if not _is_rotary_embedding_module(module, segments[-1]):
                continue
            # Shallowest wins: a model-level rotary is the one the outer forward
            # used, and per-layer rotaries live deeper inside the decoder.
            depth = len(segments)
            if best_depth is None or depth < best_depth:
                best = module
                best_depth = depth
        return best

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

    def _resolve_qlinear_kernel(
        self,
        module_name: Optional[str] = None,
        *,
        resolved_cfg=None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        """Resolves the AWQ kernel after applying any dynamic format override.

        ``in_features``/``out_features`` are optional because a few callers only
        know the module name. When they are given, the candidate is checked
        against the kernel's own packing contract and a kernel that cannot pack
        the shape falls back to the portable torch kernel of the same checkpoint
        format (issue #4567).
        """

        target_format = self._module_format(module_name, resolved_cfg=resolved_cfg)
        group_size = self._module_group_size(module_name, resolved_cfg=resolved_cfg)
        # RCO can allocate a different width per module. The CUDA GEMM/GEMV
        # kernels unpack 4-bit codes only, so a non-4-bit module must use the
        # portable torch kernel, which packs and runs 2-8 bits.
        if target_format == FORMAT.GEMM:
            if (
                self._module_bits(module_name, resolved_cfg=resolved_cfg) != 4
                or group_size not in AwqGEMMLinear.SUPPORTS_GROUP_SIZE
            ):
                return AwqTorchLinear
        if target_format == self.format:
            model_kernel = getattr(self.gptq_model, "qlinear_kernel", None)
            if model_kernel is not None:
                return self._portable_kernel_for_shape(
                    model_kernel,
                    module_name,
                    in_features=in_features,
                    out_features=out_features,
                    group_size=group_size,
                    resolved_cfg=resolved_cfg,
                )
        # The format-only selection above cannot see the module shape, and a
        # per-channel GEMM module only packs when `out_features` is a multiple
        # of `in_features` (#5391). Apply the same shape fallback here so a GQA
        # `k_proj`/`v_proj`/`down_proj` does not reach `create_quant_module`.
        return self._portable_kernel_for_shape(
            self._select_qlinear_kernel_for_format(target_format),
            module_name,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            resolved_cfg=resolved_cfg,
        )

    def _portable_kernel_for_shape(
        self,
        kernel,
        module_name: Optional[str],
        *,
        in_features: Optional[int],
        out_features: Optional[int],
        group_size: Optional[int] = None,
        resolved_cfg=None,
    ):
        """Fall back to ``AwqTorchLinear`` when ``kernel`` rejects the shape.

        A model-wide kernel is selected once, before any module shape is known,
        but the CUDA GEMM/GEMV kernels only pack some output widths
        (``AwqGEMMLinear`` needs ``out_features % 64 == 0``). Without this check
        a shape the format itself supports (FalconMamba/Mamba ``x_proj``
        8192->288) fails inside ``create_quant_module`` after the whole
        calibration has run. ``AwqTorchLinear`` packs the same ``FORMAT.GEMM``
        checkpoint layout for every shape the format admits, and AUTO already
        selects a compatible kernel per layer at load, so the saved checkpoint
        stays loadable.

        Only the portable kernel is used as a fallback: when it also rejects the
        shape the original kernel is returned so the real, module-specific
        error still surfaces.
        """

        if kernel is AwqTorchLinear or in_features is None or out_features is None:
            return kernel
        if _kernel_shape_supported(kernel, in_features, out_features, group_size):
            return kernel
        if not _kernel_shape_supported(AwqTorchLinear, in_features, out_features, group_size):
            # Neither kernel packs the shape: keep the requested one so the
            # module-specific error still names it.
            return kernel
        log.warning(
            "AWQ: `%s` cannot pack `%s` (%d->%d); using the portable `%s` kernel "
            "for this module instead of failing at pack time.",
            getattr(kernel, "__name__", kernel),
            module_name or "<model-wide>",
            in_features,
            out_features,
            AwqTorchLinear.__name__,
        )
        return AwqTorchLinear

    def _module_bits(self, module_name: Optional[str], *, resolved_cfg=None) -> int:
        """Resolve the AWQ width this module is staged and packed at.

        AWQ stages affine codes in :meth:`pseudo_quantize_tensor` and packs them
        later in :meth:`pack_module`. ``create_quant_module`` already applies the
        per-module ``dynamic`` override, so both staging and packing must resolve
        the same width here -- reading the global ``qcfg.bits`` for a 3-bit
        module would pack 4-bit codes into a 3-bit container and silently corrupt
        the layer.
        """
        if not module_name:
            return int(self.qcfg.runtime_bits)
        override = resolved_cfg.get("bits", self.qcfg.runtime_bits) if resolved_cfg is not None else (
            self.qcfg.dynamic_get(module_name, "bits", self.qcfg.runtime_bits)
        )
        # `dynamic_get` returns the raw override, which is `False` for an excluded
        # module and may be `None` when a rule only sets another key. Neither is a
        # width, so fall back to the global default instead of coercing to 0.
        if override is False or override is None:
            return int(self.qcfg.runtime_bits)
        return int(override)

    def _module_group_size(self, module_name: Optional[str], *, resolved_cfg=None) -> int:
        """Resolve the group size this module is staged and packed with."""
        if not module_name:
            return int(self.qcfg.group_size)
        override = resolved_cfg.get("group_size", self.qcfg.group_size) if resolved_cfg is not None else (
            self.qcfg.dynamic_get(module_name, "group_size", self.qcfg.group_size)
        )
        if override is False or override is None:
            return int(self.qcfg.group_size)
        return int(override)

    def _module_format(self, module_name: Optional[str], *, resolved_cfg=None) -> FORMAT:
        """Resolve the packing format this module is staged and packed with."""
        if not module_name:
            return resolve_quant_format(self.qcfg.format, self.qcfg.method)
        override = resolved_cfg.get("format", self.qcfg.format) if resolved_cfg is not None else (
            self.qcfg.dynamic_get(module_name, "format", self.qcfg.format)
        )
        if override is False or override is None:
            override = self.qcfg.format
        return resolve_quant_format(override, self.qcfg.method)

    def _module_sym(self, module_name: Optional[str], *, resolved_cfg=None) -> bool:
        """Resolve the symmetric/affine mode this module is staged and packed with.

        The packing layer already receives the per-module ``sym`` from
        ``create_quant_module``; staging and the clip search must use the same
        mode or an affine module is packed with symmetric zeros (and the other
        way around).
        """
        return _resolve_module_sym(self.qcfg, module_name, resolved_cfg)

    def _full_module_name(self, layer_module: Optional[nn.Module], module: nn.Module) -> Optional[str]:
        """Build a model-level module name from a layer and one of its children."""
        try:
            layer_name = get_op_name(self.model, layer_module) if layer_module is not None else ""
            relative = get_op_name(layer_module, module) if layer_module is not None else ""
        except Exception:
            return None
        if not layer_name:
            return relative or None
        return f"{layer_name}.{relative}" if relative else layer_name

    def _bits_for(self, layer_module: Optional[nn.Module], module: nn.Module) -> int:
        """Resolve the resolved width of one layer child."""
        return self._module_bits(self._full_module_name(layer_module, module))

    def _group_size_for(
        self, layer_module: Optional[nn.Module], module: nn.Module
    ) -> int:
        """Resolve the resolved group size of one layer child."""
        return self._module_group_size(
            self._full_module_name(layer_module, module)
        )

    def _resolve_module_kernel(
        self,
        module_name: Optional[str],
        resolved_cfg=None,
        *,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        """Resolve the packing kernel for one module, falling back to the default."""

        try:
            return self._resolve_qlinear_kernel(
                module_name,
                resolved_cfg=resolved_cfg,
                in_features=in_features,
                out_features=out_features,
            )
        except Exception:
            return self.qlinear_kernel

    def _kernel_group_size_supported(
        self,
        kernel,
        module_name: Optional[str],
        in_features: int,
        out_features: int,
        group_size: int,
        *,
        dtype: Optional[torch.dtype] = None,
        resolved_cfg=None,
    ) -> bool:
        """Ask the resolved kernel whether it can pack this shape at ``group_size``.

        The kernel owns the packing contract. ``AwqGEMMLinear`` requires both
        ``in_features`` and ``out_features`` to be multiples of the group size,
        so checking only the input axis let ``gate_proj``/``up_proj`` with
        ``out_features=448`` reach ``create_quant_module`` and raise there.
        Re-running the kernel's own ``validate`` keeps this retune decision and
        the packing gate in lockstep instead of duplicating a partial predicate.

        ``dtype`` and ``device`` are left to ``create_quant_module`` when they
        are not known yet: this runs from the processor constructor, before the
        final per-module placement is resolved.
        """

        validate = getattr(kernel, "validate", None)
        if not callable(validate):
            return in_features % group_size == 0
        try:
            ok, _ = validate(
                bits=self._module_bits(module_name, resolved_cfg=resolved_cfg),
                group_size=group_size,
                desc_act=self.qcfg.desc_act,
                sym=_resolve_module_sym(self.qcfg, module_name, resolved_cfg),
                in_features=in_features,
                out_features=out_features,
                pack_dtype=self.qcfg.pack_dtype,
                dtype=dtype,
            )
        except Exception:
            return False
        return bool(ok)

    def _largest_supported_group_divisor(
        self,
        module_name: Optional[str],
        in_features: int,
        out_features: int,
        configured_group_size: int,
        *,
        dtype: Optional[torch.dtype] = None,
        resolved_cfg=None,
    ) -> int:
        """Pick the largest group size the kernel can pack this module with.

        Returns ``-1`` (per-channel) when no positive supported group size
        passes the kernel's own packing validation but per-channel does;
        ``BaseQuantLinear`` normalizes ``-1`` to ``in_features``, so every AWQ
        stage sees one full-width group. When even per-channel does not pack
        the shape, the configured size is returned unchanged so the caller does
        not record an override for a module the kernel cannot quantize (for
        example a non-quantized MoE router).
        """

        kernel = self._resolve_module_kernel(
            module_name,
            resolved_cfg=resolved_cfg,
            in_features=in_features,
            out_features=out_features,
        )
        supported = {
            int(size)
            for size in getattr(kernel, "SUPPORTS_GROUP_SIZE", ()) or ()
            if isinstance(size, int) and size > 0
        }
        for size in sorted(supported, reverse=True):
            if size <= configured_group_size and self._kernel_group_size_supported(
                kernel,
                module_name,
                in_features,
                out_features,
                size,
                dtype=dtype,
                resolved_cfg=resolved_cfg,
            ):
                return size
        if self._kernel_group_size_supported(
            kernel,
            module_name,
            in_features,
            out_features,
            -1,
            dtype=dtype,
            resolved_cfg=resolved_cfg,
        ):
            return -1
        return configured_group_size

    def _ensure_group_size_divisibility(self) -> None:
        """Retune AWQ modules whose shape the kernel cannot pack as configured.

        AWQ's weight-mean, clip search, pseudo-quantization and packing all
        reshape the weight matrix into fixed-size groups, so a module whose
        ``in_features`` *or* ``out_features`` is not a multiple of the
        configured ``group_size`` cannot be quantized as-is. The failure
        previously surfaced only after the whole calibration pass (or, for the
        output axis, not until ``create_quant_module`` packing). Resolve a
        supported group size up front and record it as a per-module ``dynamic``
        override so staging, packing, saving and reload all resolve the same
        value.
        """

        if self.model is None:
            return
        try:
            modules = find_modules(self.model)
        except Exception:
            return

        lm_head_name = getattr(self.gptq_model, "lm_head", None)
        lm_head_quantized = bool(getattr(self.qcfg, "lm_head", False))
        adjusted: List[Tuple[str, int, int, Dict[str, Any]]] = []
        for name, module in modules.items():
            if not isinstance(module, nn.Linear):
                continue
            if name == lm_head_name and not lm_head_quantized:
                continue
            resolved_cfg = self.qcfg.resolve_module_config(name)
            if resolved_cfg.excluded:
                continue
            in_features = int(getattr(module, "in_features", 0) or 0)
            if in_features <= 0:
                continue
            out_features = int(getattr(module, "out_features", 0) or 0)
            if out_features <= 0:
                continue
            configured = self._module_group_size(name, resolved_cfg=resolved_cfg)
            if configured <= 0:
                continue
            weight = getattr(module, "weight", None)
            dtype = weight.dtype if isinstance(weight, torch.Tensor) else None
            kernel = self._resolve_module_kernel(
                name,
                resolved_cfg=resolved_cfg,
                in_features=in_features,
                out_features=out_features,
            )
            if self._kernel_group_size_supported(
                kernel,
                name,
                in_features,
                out_features,
                configured,
                dtype=dtype,
                resolved_cfg=resolved_cfg,
            ):
                continue
            chosen = self._largest_supported_group_divisor(
                name,
                in_features,
                out_features,
                configured,
                dtype=dtype,
                resolved_cfg=resolved_cfg,
            )
            if chosen == configured:
                continue
            adjusted.append((name, configured, chosen, resolved_cfg.overrides))

        if not adjusted:
            return

        overrides = {}
        for name, _, chosen, resolved_overrides in adjusted:
            payload = dict(resolved_overrides)
            payload["group_size"] = chosen
            overrides[f"+:^{re.escape(name)}$"] = payload
        dynamic = dict(self.qcfg.dynamic or {})
        merged = dict(overrides)
        for pattern, value in dynamic.items():
            merged.setdefault(pattern, value)
        self.qcfg.dynamic = merged

        examples = ", ".join(
            f"{name}: {configured}->{chosen}" for name, configured, chosen, _ in adjusted[:4]
        )
        suffix = "" if len(adjusted) <= 4 else f", ... ({len(adjusted)} modules)"
        log.warning(
            "AWQ: group_size is not supported by the packing kernel for %s%s; "
            "using a supported per-module group size so calibration can proceed.",
            examples,
            suffix,
        )

    def _format_for(
        self, layer_module: Optional[nn.Module], module: nn.Module
    ) -> FORMAT:
        """Resolve the resolved packing format of one layer child."""
        return self._module_format(
            self._full_module_name(layer_module, module)
        )

    def _sym_for(
        self, layer_module: Optional[nn.Module], module: nn.Module
    ) -> bool:
        """Resolve the resolved symmetry mode of one layer child."""
        return self._module_sym(
            self._full_module_name(layer_module, module)
        )

    def _get_layer_state(self, layer_index: int) -> _AWQLayerState:
        """Returns the mutable tracking state for a specific transformer layer."""

        with self._layer_states_lock:
            state = self._layer_states.get(layer_index)
            if state is None:
                state = _AWQLayerState()
                self._layer_states[layer_index] = state
        return state

    def continuation_state_dict(self):
        state = super().continuation_state_dict()
        state["awq"] = {
            "avg_losses": self.avg_losses,
            "nsamples": self.nsamples,
            "nsamples_total": self._nsamples_total,
            "quant_batch_size": self._quant_batch_size,
        }
        return state

    def load_continuation_state_dict(self, state):
        super().load_continuation_state_dict(state)
        self.avg_losses = state["awq"]["avg_losses"]
        self.nsamples = state["awq"]["nsamples"]
        self._nsamples_total = state["awq"]["nsamples_total"]
        self._quant_batch_size = state["awq"]["quant_batch_size"]

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

    def _feature_aggregation_policy(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Return an optional model-specific activation aggregation policy.

        The selector is a pure function of the model class and the module name,
        but the default implementation rebuilds the model's ``ModulePlan`` on
        every call. ``_record_input_feature`` asks for it once per captured
        module per batch -- 51k times on a 2-layer MoE run, 3.4 s of pure plan
        rebuilding (#5284) -- so the resolved policy is memoized per
        ``(class, module_name)``. The memo keeps a strong reference to the model
        it was resolved for, so a reused processor cannot answer for a
        different model through a recycled ``id``.
        """

        selector = getattr(self.gptq_model, "awq_input_feature_aggregation", None)
        if not callable(selector):
            return None
        memo = getattr(self, "_feature_aggregation_policies", None)
        if memo is None:
            memo = {}
            self._feature_aggregation_policies = memo
        # The selector's own identity is part of the key, so a monkeypatched or
        # otherwise replaced selector is never answered from another one's cache.
        selector_function = getattr(selector, "__func__", selector)
        try:
            key = (type(self.gptq_model), selector_function, module_name)
            hash(key)
        except TypeError:
            key = None
        cached = memo.get(key) if key is not None else None
        if cached is not None and cached[0] is self.gptq_model:
            policy = cached[1]
        else:
            policy = selector(module_name)
            if policy is not None and not isinstance(policy, Mapping):
                raise TypeError(
                    "awq_input_feature_aggregation must return a mapping or None "
                    f"for {module_name!r}, got {type(policy).__name__}."
                )
            policy = None if policy is None else dict(policy)
            if key is not None:
                memo[key] = (self.gptq_model, policy)
        # Callers may keep or mutate the mapping; hand out a fresh copy so the
        # memoized value cannot be changed from outside.
        return None if policy is None else dict(policy)

    def _record_input_feature(
        self,
        module_name: str,
        feature: torch.Tensor,
        *,
        dedupe_batch: bool = False,
    ) -> None:
        """Caches one captured input feature tensor for a named module."""

        batch_index = self.current_batch_index()
        reserved_batch_index = False
        if dedupe_batch and batch_index is not None:
            # Reserve before the potentially large GPU -> CPU transfer. MoE roots
            # can replay the same input once per expert subset; checking after the
            # transfer would discard the duplicate but still copy a full hidden
            # state tensor for every layer.
            with self.lock:
                entry = self.tasks.get(module_name)
                if entry is None:
                    entry = {"inputs": [], "batch_indices": []}
                    self.tasks[module_name] = entry
                seen_batch_indices = entry.setdefault("seen_batch_indices", set())
                if batch_index in seen_batch_indices:
                    return
                seen_batch_indices.add(batch_index)
                reserved_batch_index = True

        policy = self._feature_aggregation_policy(module_name)
        aggregation_mode = (policy or {}).get("mode")
        is_token_rows = aggregation_mode == "token_rows"

        try:
            feature = self._materialize_host_feature(feature)
        except Exception:
            if reserved_batch_index:
                with self.lock:
                    entry = self.tasks.get(module_name)
                    if entry is not None:
                        entry.get("seen_batch_indices", set()).discard(batch_index)
            raise

        if not is_token_rows:
            with self.lock:
                entry = self.tasks.get(module_name)
                if entry is None:
                    entry = {"inputs": [], "batch_indices": []}
                    self.tasks[module_name] = entry
                if dedupe_batch and batch_index is not None and not reserved_batch_index:
                    seen_batch_indices = entry.setdefault("seen_batch_indices", set())
                    if batch_index in seen_batch_indices:
                        return
                    seen_batch_indices.add(batch_index)
                inputs_list = entry.setdefault("inputs", [])
                inputs_list.append(feature)
                entry.setdefault("batch_indices", []).append(batch_index)
            return

        hidden = max(int(feature.shape[-1]), 1)
        length = int(feature.numel() // hidden)
        if length == 0:
            with self.lock:
                entry = self.tasks.get(module_name)
                if entry is None:
                    entry = {"inputs": [], "batch_indices": []}
                    self.tasks[module_name] = entry
                entry.setdefault("batch_indices", []).append(batch_index)
            return

        configured_max_tokens = (policy or {}).get("max_tokens")
        plan = self._token_row_plan_for_module(module_name, hidden, configured_max_tokens)

        # Plan match: Dense modules, MoE roots, and experts under ExpertsRoutingOverride
        if (
            plan is not None
            and batch_index is not None
            and 0 <= batch_index < len(plan.lengths)
            and length == plan.lengths[batch_index]
        ):
            quota = plan.quotas[batch_index]
            local_idx = plan.local_indices.get(batch_index)

            share_key = self._capture_share_key(feature)
            sampled_cache = getattr(self, "_capture_sampled_cache", None)
            if sampled_cache is None:
                sampled_cache = {}
                self._capture_sampled_cache = sampled_cache

            sampled_key = (share_key, quota) if share_key is not None else None
            sampled = None
            if sampled_key is not None:
                with self.lock:
                    sampled = sampled_cache.get(sampled_key)

            if sampled is None:
                if quota <= 0 or local_idx is None or len(local_idx) == 0:
                    sampled = feature.new_empty((1, 0, hidden))
                elif quota == length:
                    sampled = feature.reshape(1, length, hidden)
                else:
                    sampled = feature.reshape(-1, hidden).index_select(0, local_idx).unsqueeze(0)
                if sampled_key is not None and sampled is not None:
                    with self.lock:
                        sampled_cache[sampled_key] = sampled

            with self.lock:
                entry = self.tasks.get(module_name)
                if entry is None:
                    entry = {"inputs": [], "batch_indices": []}
                    self.tasks[module_name] = entry
                if dedupe_batch and batch_index is not None and not reserved_batch_index:
                    seen_batch_indices = entry.setdefault("seen_batch_indices", set())
                    if batch_index in seen_batch_indices:
                        return
                    seen_batch_indices.add(batch_index)

                # Provenance rows name the capture slot, not the calibration
                # batch id: captures can complete out of batch order and
                # ``_layer_input_features`` replays them in batch order, so the
                # slot index is what the replay kwargs index by.
                source_index = len(entry.setdefault("batch_row_counts", []))
                entry["batch_row_counts"].append(length)
                entry["raw_tokens"] = entry.get("raw_tokens", 0) + length
                entry["max_tokens"] = plan.max_tokens
                entry.setdefault("batch_indices", []).append(batch_index)

            if quota > 0 and local_idx is not None and len(local_idx) > 0:
                prov_rows = list(range(length)) if quota == length else local_idx.tolist()
            else:
                prov_rows = []

            # A plan-matched batch can arrive after this module switched to the
            # dynamic reservoir (routed experts see the shortest batches last).
            # Append-or-merge under the module's reservoir lock so the merge
            # cannot overwrite rows this capture just appended, and a capture
            # cannot race the reservoir creation that seeds from ``inputs``
            # (#5264).
            with self._reservoir_lock_for(module_name):
                if entry.get("reservoir_buffer") is not None:
                    budget_cap = self._reservoir_budget_cap(configured_max_tokens, plan, entry)
                    row_indices = torch.tensor(prov_rows, dtype=torch.int64)
                    priorities = self._reservoir_priorities(source_index, length).index_select(
                        0, row_indices
                    )
                    entry["max_tokens"] = budget_cap
                    self._merge_reservoir_rows(
                        entry,
                        sampled.reshape(-1, hidden),
                        priorities,
                        source_index=source_index,
                        budget_cap=budget_cap,
                        row_indices=row_indices,
                    )
                else:
                    entry.setdefault("inputs", []).append(sampled)
                    if prov_rows:
                        entry.setdefault("provenance", []).extend(
                            (source_index, int(row)) for row in prov_rows
                        )
            return

        # Dynamic / streaming reservoir: routed MoE experts or cache-less runs.
        #
        # A routed expert keeps a different number of rows per batch, so the
        # precomputed plan never matches. Bound the capture with a uniform
        # priority reservoir instead: every captured row gets a deterministic
        # priority and only the ``budget_cap`` lowest priorities survive.
        #
        # Resampling the kept buffer evenly together with each new batch (the
        # previous behaviour) is *not* a reservoir: once the buffer is full, the
        # ``budget_cap`` kept rows -- which stand for every earlier batch --
        # weigh the same as the fresh batch, so each new batch replaces about
        # ``L / (cap + L)`` of the sample and the surviving share decays
        # geometrically. On a 128-document calibration set that put 70-87% of
        # every routed expert's rows in the last quarter of the batches (#5204).
        #
        # Keeping the rows with the lowest uniform priority is the deterministic
        # equivalent of Algorithm R: it is a uniform sample of every row seen, so
        # a batch keeps a share proportional to its length regardless of where it
        # falls in the stream, and the memory stays O(cap).
        feat_2d = feature.reshape(-1, hidden)
        module_lock = self._reservoir_lock_for(module_name)
        with self.lock:
            entry = self.tasks.get(module_name)
            if entry is None:
                entry = {"inputs": [], "batch_indices": []}
                self.tasks[module_name] = entry
            if dedupe_batch and batch_index is not None and not reserved_batch_index:
                seen_batch_indices = entry.setdefault("seen_batch_indices", set())
                if batch_index in seen_batch_indices:
                    return
                seen_batch_indices.add(batch_index)

            source_idx = len(entry.setdefault("batch_row_counts", []))
            entry["batch_row_counts"].append(length)
            entry["raw_tokens"] = entry.get("raw_tokens", 0) + length
            # Size the reservoir from this module's own captures, the current
            # one included, so a routed expert keeps the pre-#5175 per-expert
            # budget instead of the layer-wide plan budget (#5284). The budget
            # only grows, and a growing buffer carries the rows it held.
            budget_cap = self._reservoir_budget_cap(configured_max_tokens, plan, entry)
            entry["max_tokens"] = budget_cap
            entry.setdefault("batch_indices", []).append(batch_index)

        # The priority draw and the slot merge are per-module work. Holding the
        # processor-wide lock across them serialized every routed expert's
        # capture on one lock, which dominated AWQ quantize time on MoE models
        # (#5233). ``self.lock`` is not reentrant, so only the O(1) bookkeeping
        # above runs under it.
        new_priorities = self._reservoir_priorities(source_idx, length)
        with module_lock:
            self._merge_reservoir_rows(
                entry,
                feat_2d,
                new_priorities,
                source_index=source_idx,
                budget_cap=budget_cap,
            )

    def _reservoir_lock_for(self, module_name: str) -> threading.Lock:
        """Return the per-module lock that serializes reservoir mutations."""

        with self.lock:
            reservoir_locks = getattr(self, "_reservoir_locks", None)
            if reservoir_locks is None:
                reservoir_locks = {}
                self._reservoir_locks = reservoir_locks
            module_lock = reservoir_locks.get(module_name)
            if module_lock is None:
                module_lock = threading.Lock()
                reservoir_locks[module_name] = module_lock
            return module_lock

    def _reservoir_budget_cap(self, configured_max_tokens, plan, entry=None) -> int:
        """Resolve the routed-expert reservoir size for one capture.

        A configured ``max_tokens`` wins. Otherwise a routed expert sizes its
        reservoir from its own captures -- the largest capture, raised to one
        row per capture, exactly the pre-#5175 per-expert budget -- instead of
        the layer-wide plan budget. A layer-wide cap for every expert made the
        clip search (and the replay forward) run on ~5x the rows on OLMoE
        (#5284), while the reservoir still samples the same stream uniformly.
        The budget only grows (a larger capture, or more captures), and a
        growing buffer keeps every row it already held.
        """

        if configured_max_tokens is not None:
            budget_cap = self._cap_token_rows(int(configured_max_tokens))
        else:
            budget_cap = 0
            if entry is not None:
                budget_cap = self._token_row_budget_for_counts(
                    entry.get("batch_row_counts") or []
                )
                budget_cap = self._cap_token_rows(int(budget_cap))
            if budget_cap <= 0:
                budget_cap = self._cap_token_rows(plan.budget if plan is not None else 906)
        return budget_cap if budget_cap > 0 else 906

    @staticmethod
    def _reservoir_priorities(source_index: int, count: int) -> torch.Tensor:
        """Return one deterministic uniform priority per captured row.

        A per-batch seed keeps the priorities independent and identically
        uniform across batches, which is what makes the lowest-priority
        survivors a uniform sample of the whole stream. The seed is derived from
        the capture slot, so the same calibration run always keeps the same
        rows.
        """

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            (0x9E3779B97F4A7C15 * (int(source_index) + 1) + 0x165667B19E3779F9)
            & 0x7FFFFFFFFFFFFFFF
        )
        return torch.rand(count, generator=generator, dtype=torch.float64)

    def _merge_reservoir_rows(
        self,
        entry: Dict[str, Any],
        rows: torch.Tensor,
        priorities: torch.Tensor,
        *,
        source_index: int,
        budget_cap: int,
        row_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Merge one capture into the module's priority reservoir.

        The buffer holds exactly ``budget_cap`` slots and is deliberately not
        kept in capture order: every surviving row carries its
        ``(source_index, row)`` provenance, which ``_layer_input_features``
        materializes once per layer. A capture therefore copies only the new
        rows that displace kept rows instead of concatenating, re-sorting and
        re-listing the whole buffer under the processor-wide lock (#5233).

        ``row_indices`` names each row's index inside its calibration batch.
        The dynamic path captures every row of the batch, so it defaults to
        ``arange(count)``; the token-row plan path keeps only the sampled
        ``local_indices`` and must pass them so replay kwargs stay aligned.

        The budget only grows, but two captures of one module can compute
        their caps under the processor lock and then merge under the module
        lock in the opposite order. A merge with an older, smaller cap must
        therefore never resize the buffer down: that would keep only the
        ``budget_cap`` lowest priorities of the rows already merged and drop
        the rest for good. Raising the cap to the buffer's high-water mark
        keeps a stale merge from evicting rows a larger capture already
        merged (#5333).
        """

        buffer = entry.get("reservoir_buffer")
        if buffer is not None:
            budget_cap = max(int(budget_cap), int(buffer.shape[0]))
            entry["max_tokens"] = budget_cap
        if (
            buffer is None
            or buffer.shape[0] != budget_cap
            or buffer.shape[1] != rows.shape[1]
        ):
            # The per-expert budget only grows (a larger capture, or more
            # captures), so a resize must keep every row the reservoir already
            # held; rebuilding from ``inputs`` cannot, because the provenance
            # list is materialized only at replay (#5284).
            carried = None
            if buffer is not None and buffer.shape[1] == rows.shape[1]:
                previous_used = int(entry.get("reservoir_used", 0))
                if previous_used > 0:
                    order = torch.arange(previous_used)
                    if previous_used > budget_cap:
                        order = torch.topk(
                            entry["reservoir_priorities"][:previous_used],
                            budget_cap,
                            largest=False,
                        ).indices
                        order, _ = torch.sort(order)
                    carried = (
                        buffer[:previous_used].index_select(0, order).clone(),
                        entry["reservoir_priorities"][:previous_used].index_select(0, order).clone(),
                        entry["reservoir_sources"][:previous_used].index_select(0, order).clone(),
                        entry["reservoir_rows"][:previous_used].index_select(0, order).clone(),
                    )
            buffer = rows.new_empty((budget_cap, rows.shape[1]))
            entry["reservoir_buffer"] = buffer
            entry["reservoir_priorities"] = torch.empty((budget_cap,), dtype=torch.float64)
            entry["reservoir_sources"] = torch.empty((budget_cap,), dtype=torch.int64)
            entry["reservoir_rows"] = torch.empty((budget_cap,), dtype=torch.int64)
            entry["reservoir_used"] = 0
            if carried is not None:
                carried_used = int(carried[0].shape[0])
                buffer[:carried_used] = carried[0]
                entry["reservoir_priorities"][:carried_used] = carried[1]
                entry["reservoir_sources"][:carried_used] = carried[2]
                entry["reservoir_rows"][:carried_used] = carried[3]
                entry["reservoir_used"] = carried_used
            else:
                # A module can capture a plan-matched batch first and fall out
                # of the plan later (for example a row shorter than the planned
                # length). Keep those rows instead of starting from empty.
                self._seed_reservoir_from_captures(entry, budget_cap)

        priority_buffer = entry["reservoir_priorities"]
        source_buffer = entry["reservoir_sources"]
        row_buffer = entry["reservoir_rows"]
        used = int(entry.get("reservoir_used", 0))
        count = int(rows.shape[0])
        if row_indices is None:
            row_indices = torch.arange(count, dtype=torch.int64)
        elif int(row_indices.numel()) != count:
            raise ValueError(
                "reservoir row indices must match the captured rows: "
                f"{int(row_indices.numel())} != {count}"
            )

        if used + count <= budget_cap:
            buffer[used:used + count] = rows
            priority_buffer[used:used + count] = priorities
            source_buffer[used:used + count] = source_index
            row_buffer[used:used + count] = row_indices
            used += count
        else:
            # A full reservoir usually rejects the whole capture: a routed
            # expert sees one batch per expert per layer, so most merges offer
            # rows that all lose to the kept ones. Comparing the capture's best
            # priority with the worst kept priority is O(cap) on host floats and
            # skips the drop-set search below (#5284). The condition is strict
            # so ties keep the slow path and the kept set is unchanged for
            # every capture this short-circuits.
            if used >= budget_cap and count and bool(
                (priorities > priority_buffer[:used].max()).all()
            ):
                entry["reservoir_used"] = used
                entry["inputs"] = [buffer[:used].unsqueeze(0)]
                entry.pop("provenance", None)
                return
            # The kept set is the ``budget_cap`` lowest priorities of old + new,
            # so ``overflow`` rows are dropped and only the ``overflow`` largest
            # old rows can be among them (any other old row already has
            # ``overflow`` old rows at or above it). Selecting the drop set from
            # those plus the capture moves only the rows that survive, instead
            # of writing every surviving row of the capture into a slot
            # (#5284).
            overflow = used + count - budget_cap
            old_probe = min(overflow, used)
            old_largest = (
                torch.topk(priority_buffer[:used], old_probe, largest=True)
                if old_probe
                else None
            )
            union_values = (
                torch.cat([old_largest.values, priorities]) if old_probe else priorities
            )
            drop = torch.topk(union_values, overflow, largest=True).indices
            new_drop = drop[drop >= old_probe] - old_probe
            old_drop = drop[drop < old_probe]
            keep_new = torch.ones((count,), dtype=torch.bool)
            keep_new[new_drop] = False
            kept_new = keep_new.nonzero(as_tuple=False).flatten()
            slots = torch.cat(
                [
                    old_largest.indices[old_drop] if old_probe else kept_new.new_empty(0),
                    torch.arange(used, budget_cap),
                ]
            )
            buffer[slots] = rows.index_select(0, kept_new)
            priority_buffer[slots] = priorities.index_select(0, kept_new)
            source_buffer[slots] = source_index
            row_buffer[slots] = row_indices.index_select(0, kept_new)
            used = budget_cap

        entry["reservoir_used"] = used
        # ``inputs`` stays a cheap view of the live buffer so capture-time
        # coverage checks see the rows; provenance is materialized at replay.
        entry["inputs"] = [buffer[:used].unsqueeze(0)]
        entry.pop("provenance", None)

    def _seed_reservoir_from_captures(self, entry: Dict[str, Any], budget_cap: int) -> None:
        """Fold rows captured before the module switched to the reservoir.

        Dense modules and MoE roots use the precomputed token-row plan; a batch
        that does not match it falls through to the reservoir. Seeding keeps
        the plan rows instead of silently dropping the earlier batches.
        """

        inputs = entry.get("inputs") or []
        provenance = entry.get("provenance") or []
        if not inputs or not provenance:
            return
        rows = torch.cat([t.reshape(-1, t.shape[-1]) for t in inputs], dim=0)
        if rows.shape[0] != len(provenance):
            # Unaligned capture state cannot be relabelled safely; drop it.
            entry["inputs"] = []
            entry.pop("provenance", None)
            return

        counts = entry.get("batch_row_counts") or []
        priorities = torch.empty((rows.shape[0],), dtype=torch.float64)
        per_source: Dict[int, torch.Tensor] = {}
        for index, (source, row) in enumerate(provenance):
            source = int(source)
            row = int(row)
            if source not in per_source:
                count = int(counts[source]) if source < len(counts) else row + 1
                per_source[source] = self._reservoir_priorities(source, max(count, row + 1))
            priorities[index] = per_source[source][row]

        if rows.shape[0] > budget_cap:
            keep = torch.topk(priorities, budget_cap, largest=False).indices
            rows = rows.index_select(0, keep)
            priorities = priorities[keep]
            provenance = [provenance[int(index)] for index in keep.tolist()]

        used = int(rows.shape[0])
        entry["reservoir_buffer"][:used] = rows
        entry["reservoir_priorities"][:used] = priorities
        entry["reservoir_sources"][:used] = torch.tensor(
            [int(source) for source, _row in provenance], dtype=torch.int64
        )
        entry["reservoir_rows"][:used] = torch.tensor(
            [int(row) for _source, row in provenance], dtype=torch.int64
        )
        entry["reservoir_used"] = used

    def _materialize_reservoir_provenance(self, entry: Dict[str, Any]) -> None:
        """Expose the reservoir's rows and provenance in the replay layout."""

        buffer = entry.get("reservoir_buffer")
        if buffer is None:
            return
        used = int(entry.get("reservoir_used", 0))
        entry["inputs"] = [buffer[:used].unsqueeze(0)]
        entry["provenance"] = list(
            zip(
                entry["reservoir_sources"][:used].tolist(),
                entry["reservoir_rows"][:used].tolist(),
            )
        )
        for key in (
            "reservoir_buffer",
            "reservoir_priorities",
            "reservoir_sources",
            "reservoir_rows",
            "reservoir_used",
            "reservoir_worst_priority",
        ):
            entry.pop(key, None)

    def _capture_share_key(self, feature: torch.Tensor) -> Optional[Tuple[Any, ...]]:
        """Identity of one captured source tensor, or ``None`` when it cannot be shared.

        A masked capture is shared too: the cached host copy is the masked
        result, and the key carries the identity of the active keep-mask, which
        the looper binds once per calibration batch. Before
        ``CalibrationConfig.exclude_attention_sinks`` only padded batches had a
        mask; with it every batch does, and refusing to share would copy each
        q/k/v and gate/up activation once per sibling. Calibration runs under
        ``inference_mode``, so the captured tensor
        is usually an inference tensor and ``_version`` raises. Keying on the
        version alone therefore disabled sharing for every real calibration
        capture. Use the storage pointer plus shape/stride/dtype instead; the
        cached weakref still proves the hit is the same object, so a recycled
        pointer cannot alias a different tensor. A real version counter is kept
        when one exists so an in-place mutation still invalidates the host copy.
        """

        keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)
        if not torch.is_tensor(feature):
            return None
        try:
            data_ptr = feature.data_ptr()
        except Exception:
            return None
        try:
            version = int(feature._version)
        except Exception:
            # Inference tensors have no version counter and cannot be modified
            # in place outside inference mode.
            version = None
        return (
            self.current_batch_index(),
            data_ptr,
            tuple(feature.shape),
            tuple(feature.stride()),
            feature.dtype,
            version,
            None if keep_mask is None else id(keep_mask),
        )

    def _materialize_host_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Return the host copy of one captured input, reusing a sibling's copy.

        ``q_proj``/``k_proj``/``v_proj`` and ``gate_proj``/``up_proj`` are hooked
        separately but read the same activation object, so the D2H transfer and the
        host allocation are identical for every sibling. Copy once per unique
        source tensor instead of once per module.
        """

        batch_index = self.current_batch_index()
        if self._capture_host_batch != batch_index:
            # A capture cache only ever describes one calibration batch.
            with self.lock:
                self._capture_host_cache.clear()
                if hasattr(self, "_capture_sampled_cache"):
                    self._capture_sampled_cache.clear()
                self._capture_host_batch = batch_index

        key = self._capture_share_key(feature)
        if key is not None:
            cached = self._capture_host_cache.get(key)
            if cached is not None and cached[0]() is feature:
                return cached[1]

        host = self._apply_active_keep_mask_to_feature(
            self._drop_padding_only_rows(feature)
        )
        # Preserve a leading sample axis for flattened [seq, hidden] captures so later
        # concatenation produces [samples, seq, hidden] instead of collapsing into one giant sequence.
        if host.dim() <= 2:
            host = host.unsqueeze(0)
        if host.device.type != "cpu":
            host = host.detach().cpu()
        else:
            host = host.detach()

        if key is not None:
            with self.lock:
                self._capture_host_cache[key] = (weakref.ref(feature), host)
        return host

    def _drop_padding_only_rows(self, feature: torch.Tensor) -> torch.Tensor:
        """Drop zeroed calibration-padding rows from a routed-expert capture (#5337).

        The MoE-root hook zeroes padding positions before routing (#4035), so a
        natively routed expert receives them as all-zero rows. Their count is
        unrelated to the keep mask, which is why the mask cannot filter them,
        and they used to count toward the expert's sample. Only drop when the
        batch actually has padding: an all-zero row can be a real activation.
        """

        if not torch.is_tensor(feature) or feature.dim() != 2:
            return feature

        tls = getattr(self, "_mask_tls", None)
        keep_mask = getattr(tls, "value", None)
        if not torch.is_tensor(keep_mask) or keep_mask.ndim != 2:
            return feature
        # A `[B * S, H]` capture is filtered by the mask itself in
        # `_apply_active_keep_mask_to_feature`. Anything else is a routed subset
        # (pads routed to this expert, real rows elsewhere), whose row count is
        # unrelated to the mask: it can even equal `keep_mask.sum()` while still
        # containing padding, so the count is only used to recognize the full
        # grid (#5360). A real activation can be all-zero, so the zero check
        # alone cannot decide either.
        if int(feature.shape[0]) == int(keep_mask.numel()):
            return feature

        padding_present = getattr(tls, "mask_has_padding", None)
        if padding_present is None:
            padding_present = not bool(keep_mask.all())
        if not padding_present:
            return feature

        keep_rows = (feature != 0).any(dim=-1)
        if bool(keep_rows.all()):
            return feature
        return feature[keep_rows].contiguous()

    def _apply_active_keep_mask_to_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Remove padded rows from an activation captured by the current worker.

        Regular module hooks normally receive ``[B, S, H]`` inputs, while the
        all-expert calibration path flattens the same input to ``[B * S, H]``.
        Handle both layouts so root and expert statistics exclude the same
        padding rows. Inputs that cannot be matched unambiguously to the active
        mask retain their existing behavior.
        """

        keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)
        if not torch.is_tensor(feature) or not torch.is_tensor(keep_mask) or keep_mask.ndim != 2:
            return feature

        feature = align_single_head_capture(feature, keep_mask)
        keep_mask = keep_mask.to(device=feature.device, dtype=torch.bool)
        if feature.dim() >= 3 and tuple(feature.shape[:2]) == tuple(keep_mask.shape):
            flattened = feature.reshape(-1, *feature.shape[2:])
            return flattened[keep_mask.reshape(-1)].contiguous()

        if feature.dim() == 2 and feature.shape[0] == keep_mask.numel():
            return feature[keep_mask.reshape(-1)].contiguous()

        return feature

    def record_moe_root_input_feature(self, module_name: str, feature: torch.Tensor) -> None:
        """Capture one pointwise MoE-root input per calibration batch when requested.

        MoE layers are normally hooked only at expert projections.  Model-specific
        AWQ scale groups may instead need the shared pre-router input.  The policy
        gate keeps this opt-in so sequence-sensitive models retain their existing
        replay contract.
        """

        if getattr(self, "coverage_only", False):
            return

        policy = self._feature_aggregation_policy(module_name)
        if not policy or not policy.get("capture_root", False):
            return
        # Capture only while at least one routed projection under this root has
        # an active AWQ task. Replay-only passes after task cleanup must not
        # repopulate the root cache.
        task_prefix = f"{module_name}."
        with self.lock:
            if not any(name.startswith(task_prefix) for name in self.tasks):
                return
        self._record_input_feature(module_name, feature, dedupe_batch=True)

    def register_moe_root_capture_hook(
        self,
        moe_block: Module,
        moe_block_name: str,
        handles: List[Any],
    ) -> bool:
        """Register AWQ's shared MoE-root capture before the block executes.

        AWQ is the only lifecycle that consumes this shared pointwise feature.
        Keep the activation-capture and model-policy gates here, alongside the
        recorder they protect, so the generic subset executor remains unaware
        of AWQ-specific state and math.
        """

        if not self.forward.activation_capture:
            return False

        def zero_padded_rows(_module, args, kwargs):
            # Mirror the GPTQ MoE padding contract (#4035): zero the padded
            # positions of the block input so a natively routed expert receives
            # them as all-zero rows, which `_drop_padding_only_rows` then
            # removes from its capture (#5337). Hash routers select by input
            # id, so zeroing does not change their routing. A BOS sink is a
            # real token and is kept: only the padding mask is applied.
            keep_mask = active_padding_mask(self)
            if not torch.is_tensor(keep_mask) or keep_mask.ndim != 2:
                return None

            tls = getattr(self, "_mask_tls", None)
            padding_present = getattr(tls, "mask_has_padding", None)
            if padding_present is None:
                padding_present = not bool(keep_mask.all())
            if not padding_present:
                return None

            hidden_states = kwargs.get("hidden_states") if kwargs else None
            from_kwargs = hidden_states is not None
            if hidden_states is None:
                if not args or not torch.is_tensor(args[0]):
                    return None
                hidden_states = args[0]
            if hidden_states.dim() != 3 or tuple(hidden_states.shape[:2]) != tuple(keep_mask.shape):
                return None

            row_mask = keep_mask.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ).unsqueeze(-1)
            masked = hidden_states * row_mask
            if from_kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = masked
                return args, kwargs
            return (masked,) + tuple(args[1:]), kwargs

        root_policy = self._feature_aggregation_policy(moe_block_name)
        capture_root = bool(root_policy and root_policy.get("capture_root", False))
        if capture_root:

            def capture_moe_root_input(_module, inputs):
                if inputs:
                    self.record_moe_root_input_feature(moe_block_name, inputs[0])

            handles.append(moe_block.register_forward_pre_hook(capture_moe_root_input))

        # Registered after the capture hook so `handles[0]` stays the capture;
        # the capture sees the padded input and drops those rows itself.
        handles.append(
            moe_block.register_forward_pre_hook(zero_padded_rows, with_kwargs=True)
        )
        return capture_root

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

    def _position_limit(self) -> Optional[int]:
        """Return the declared sequence limit of the model being quantized."""

        cached = getattr(self, "_cached_position_limit", _POSITION_LIMIT_UNRESOLVED)
        if cached is not _POSITION_LIMIT_UNRESOLVED:
            return cached
        limit = None
        for source in (
            getattr(self.gptq_model, "model", None),
            getattr(self, "model", None),
        ):
            limit = resolve_position_limit(getattr(source, "config", None))
            if limit is not None:
                break
        self._cached_position_limit = limit
        return limit

    def _cap_token_rows(self, budget: int) -> int:
        """Clamp a token-row budget to the model's declared sequence limit."""

        limit = self._position_limit()
        if limit is None or budget <= 0:
            return budget
        return min(int(budget), int(limit))

    def _extract_cache_batch_lengths(self) -> Optional[List[int]]:
        """Return the unpadded token count for each batch in inputs_cache."""
        cache = getattr(self, "inputs_cache", None)
        if cache is None:
            return None
        layer_inputs = getattr(cache, "layer_inputs", None)
        if not layer_inputs:
            return None
        # ``_record_input_feature`` asks for this once per captured module per
        # batch, so recomputing the full batch scan each time is quadratic in
        # the calibration batch count. The layer's input list is replaced as a
        # whole when the looper advances, so memoize on its identity and keep a
        # strong reference to rule out ``id`` reuse.
        memo = getattr(self, "_token_row_lengths_cache", None)
        if memo is None:
            memo = {}
            self._token_row_lengths_cache = memo
        memo_key = id(layer_inputs)
        cached = memo.get(memo_key)
        if cached is not None and cached[0] is layer_inputs:
            return cached[1]
        masks = getattr(cache, "attention_masks", None) or []
        exclude_sinks = bool(
            getattr(getattr(self, "calibration", None), "exclude_attention_sinks", False)
        )
        lengths = []
        for b, batch_inputs in enumerate(layer_inputs):
            mask = masks[b] if b < len(masks) else None
            sink_positions = self.attention_sink_positions_for_batch(b)
            if torch.is_tensor(mask):
                form = self._captured_mask_form(mask)
                m = mask.detach().to(device="cpu")
                if m.dim() == 4:
                    m = m[:, 0]
                if m.dim() == 3 and m.shape[-1] == m.shape[-2] and m.shape[0] > 0:
                    if m.dtype == torch.bool:
                        allowed_grid = m if form == self._MASK_FORM_BOOL_KEEP else ~m
                    elif form == self._MASK_FORM_POSITIVE_KEEP:
                        allowed_grid = m > 0
                    else:
                        allowed_grid = m >= 0
                    keep = allowed_grid.diagonal(dim1=-2, dim2=-1)
                else:
                    try:
                        keep = normalize_seq_mask(m)
                    except ValueError:
                        keep = None
                if keep is not None:
                    if exclude_sinks:
                        keep = exclude_attention_sinks(keep, sink_positions)
                    lengths.append(int(keep.sum().item()))
                    continue
            if batch_inputs:
                t = batch_inputs[0] if isinstance(batch_inputs, (list, tuple)) else batch_inputs
                h = max(int(t.shape[-1]), 1)
                raw_len = int(t.numel() // h)
                if exclude_sinks and torch.is_tensor(sink_positions):
                    # ``sink_positions`` is the ``[B, S]`` BOS mask, not a list
                    # of positions: subtracting ``numel()`` drops the whole
                    # batch and zeroes every planned length (#5275). The masked
                    # branch above already subtracts the excluded positions.
                    raw_len = max(0, raw_len - int(sink_positions.sum().item()))
                lengths.append(raw_len)
            else:
                lengths.append(0)
        with self.lock:
            memo.clear()
            memo[memo_key] = (layer_inputs, lengths)
        return lengths

    def _token_row_plan_for_module(
        self,
        module_name: str,
        hidden: int,
        configured_max_tokens: Optional[int] = None,
    ) -> Optional[_TokenRowPlan]:
        """Compute or retrieve a cached token-row sampling plan for the current layer."""
        cache = getattr(self, "inputs_cache", None)
        if cache is None:
            return None
        lengths = self._extract_cache_batch_lengths()
        if not lengths:
            return None

        cache_plans = getattr(self, "_token_row_plans", None)
        if cache_plans is None:
            cache_plans = {}
            self._token_row_plans = cache_plans

        cache_key = (id(cache), configured_max_tokens)
        with self.lock:
            plan = cache_plans.get(cache_key)
            if plan is not None:
                return plan

        raw_tokens = sum(lengths)
        largest_batch = max(lengths)
        contributing_batches = len([l for l in lengths if l > 0])
        budget = min(raw_tokens, max(largest_batch, contributing_batches))
        max_tokens = (
            self._cap_token_rows(int(configured_max_tokens))
            if configured_max_tokens is not None
            else self._cap_token_rows(budget)
        )
        quotas, local_indices = self._compute_token_row_quotas(lengths, max_tokens)
        plan = _TokenRowPlan(
            lengths=lengths,
            budget=budget,
            max_tokens=max_tokens,
            quotas=quotas,
            local_indices=local_indices,
        )
        with self.lock:
            cache_plans[cache_key] = plan
        return plan

    @staticmethod
    def _token_row_budget_for_counts(row_counts: List[int]) -> int:
        """Derive token-row budget directly from a list of batch row counts."""
        valid_counts = [int(c) for c in row_counts if int(c) > 0]
        if not valid_counts:
            return 0
        raw_tokens = sum(valid_counts)
        largest_batch = max(valid_counts)
        contributing_batches = len(valid_counts)
        return min(raw_tokens, max(largest_batch, contributing_batches))

    @staticmethod
    def _token_row_budget(tensors: List[torch.Tensor]) -> int:
        """Derive a pointwise sample budget from observed calibration batches.

        Use one largest-batch equivalent, raised to the number of non-empty
        batches so every contributing batch can retain at least one row. This
        adapts to actual batch size, sequence lengths, and expert occupancy
        while limiting any increase beyond the largest observed activation
        batch to the minimum needed for one-row-per-batch coverage.

        Callers that own the model must still bound the result to its declared
        sequence limit (``AWQProcessor._cap_token_rows``): packed rows are
        replayed as one sequence, so a ``batch_size > 1`` batch, or a corpus
        with more batches than the limit, can otherwise ask a module to attend
        over more positions than the model declares (#5085).
        """

        row_counts = []
        for tensor in tensors:
            if tensor.dim() < 2:
                continue
            hidden = int(tensor.shape[-1])
            if hidden <= 0:
                continue
            rows = int(tensor.numel() // hidden)
            if rows > 0:
                row_counts.append(rows)

        return AWQProcessor._token_row_budget_for_counts(row_counts)

    #: One in this many single-row quotas keeps the batch's first position.
    #:
    #: Both extremes are bad. Keeping every single-row quota at row 0 collapsed
    #: the sample onto the BOS/attention-sink activation, identical for every
    #: document (#5265: Llama-3.2-1B PPL 21.8 -> 91.7). Keeping none removes
    #: the sink rows entirely, and the scale search then never sees the
    #: outlier channels those positions carry (#5310: Qwen2.5-0.5B PPL 28.0,
    #: worse than RTN's 24.8). A 1/8 share is close to the sink share of the
    #: ``batch_size=4`` sample that #4176 accepted.
    SINGLE_ROW_SINK_PERIOD = 8

    @staticmethod
    def _single_token_row(length: int, ordinal: int) -> torch.Tensor:
        """Row a batch contributes when its quota is exactly one.

        ``ordinal`` counts the single-row quotas in batch order, so the sink
        rows are spread deterministically over the calibration set instead of
        clustering in its first eighth.
        """

        if ordinal % AWQProcessor.SINGLE_ROW_SINK_PERIOD == 1:
            return torch.zeros(1, dtype=torch.int64)
        return AWQProcessor._stratified_token_row(1, int(length))

    @staticmethod
    def _stratified_token_row(quota: int, length: int) -> torch.Tensor:
        """Rows at the centre of ``quota`` equal-width bins over one batch.

        Used for every quota above one, and for the non-sink single-row quotas
        (see ``_single_token_row``): a batch that contributes one row must not
        pin position 0 every time, because the hidden state there is the
        BOS/attention-sink activation, identical for every document (#5265).
        """
        index = torch.arange(int(quota), dtype=torch.int64)
        return ((2 * index + 1) * int(length)) // (2 * int(quota))

    @staticmethod
    def _compute_token_row_quotas(
        lengths: List[int],
        max_tokens: int,
    ) -> Tuple[List[int], Dict[int, torch.Tensor]]:
        """Compute per-batch token row quotas and deterministic local indices.

        Returns (quotas, local_indices_by_batch) where local_indices_by_batch[b]
        is a 1D int64 tensor of row indices within batch b to retain.

        Empty batches are excluded from the quota allocation, matching
        ``_pack_token_rows``: they contribute no rows, so counting them would
        steal budget (or select a zero-row batch) and break equivalence.

        A batch that contributes a single row keeps the centre of the batch,
        not row 0, which is the BOS/attention-sink position (#5265).
        """
        quotas = [0] * len(lengths)
        local_indices: Dict[int, torch.Tensor] = {}
        active = [index for index, length in enumerate(lengths) if int(length) > 0]
        if not active:
            return quotas, local_indices
        active_lengths = [int(lengths[index]) for index in active]
        batch_count = len(active_lengths)
        total_tokens = sum(active_lengths)
        retained_tokens = min(total_tokens, max_tokens)
        if retained_tokens <= 0:
            return quotas, local_indices

        if retained_tokens < batch_count:
            if retained_tokens == 1:
                batch_ids = [0]
            else:
                batch_ids = [
                    int(i * (batch_count - 1) // (retained_tokens - 1))
                    for i in range(retained_tokens)
                ]
            batch_ids = list(dict.fromkeys(batch_ids))
            for ordinal, active_index in enumerate(batch_ids, start=1):
                batch_index = active[active_index]
                quotas[batch_index] = 1
                local_indices[batch_index] = AWQProcessor._single_token_row(
                    active_lengths[active_index], ordinal
                )
            return quotas, local_indices

        local_quotas = [1] * batch_count
        remaining = retained_tokens - batch_count
        if remaining > 0:
            exact = [length * remaining / total_tokens for length in active_lengths]
            floors = [int(share) for share in exact]
            for index, floor_share in enumerate(floors):
                local_quotas[index] += floor_share
            leftover = retained_tokens - sum(local_quotas)
            order = sorted(
                range(batch_count),
                key=lambda index: (-(exact[index] - floors[index]), index),
            )
            while leftover > 0:
                distributed = False
                for index in order:
                    if leftover <= 0:
                        break
                    if local_quotas[index] < active_lengths[index]:
                        local_quotas[index] += 1
                        leftover -= 1
                        distributed = True
                if not distributed:
                    break

        single_row_ordinal = 0
        for active_index, batch_index in enumerate(active):
            length = active_lengths[active_index]
            quota = min(local_quotas[active_index], length)
            quotas[batch_index] = quota
            if quota <= 0:
                continue
            if quota == length:
                local_indices[batch_index] = torch.arange(length, dtype=torch.int64)
            elif quota == 1:
                # Row 0 for one quota in eight, the bin centre otherwise: both
                # an all-sink and a sink-free sample measure badly (#5265,
                # #5310).
                single_row_ordinal += 1
                local_indices[batch_index] = AWQProcessor._single_token_row(
                    length, single_row_ordinal
                )
            else:
                local_indices[batch_index] = (
                    torch.arange(quota, dtype=torch.int64)
                    * (length - 1)
                    // (quota - 1)
                )

        return quotas, local_indices

    @staticmethod
    def _reorder_token_row_capture(entry: Dict[str, Any], order: List[int]) -> None:
        """Follow the capture permutation for token-row provenance metadata.

        ``entry["inputs"]`` is replayed in calibration-batch order, but captures
        can complete out of order. ``batch_row_counts`` and ``provenance`` are
        recorded in arrival order, so both must be permuted with the tensors;
        otherwise the rebuilt mask and position ids describe a different batch
        than the rows they accompany.
        """

        counts = entry.get("batch_row_counts")
        provenance = entry.get("provenance")
        if not counts and not provenance:
            return
        if counts and len(counts) == len(order):
            entry["batch_row_counts"] = [counts[index] for index in order]
        if provenance:
            # Provenance rows are stored in capture order, which is the row
            # order of the unreordered ``inputs`` list. Re-emit whole source
            # groups in the new tensor order and renumber each group's source
            # slot so packed rows and rebuilt kwargs stay aligned.
            grouped: Dict[int, List[int]] = {}
            for source, row in provenance:
                grouped.setdefault(int(source), []).append(int(row))
            entry["provenance"] = [
                (new_source, row)
                for new_source, old_source in enumerate(order)
                for row in grouped.get(int(old_source), [])
            ]

    @staticmethod
    def _pack_token_rows(
        tensors: List[torch.Tensor],
        *,
        max_tokens: int,
        with_provenance: bool = False,
    ) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, List[Tuple[int, int]]]:
        """Pack ragged pointwise activations into deterministic bounded token rows.

        The returned tensor always has shape ``[1, retained_tokens, hidden]``.
        Whenever ``max_tokens`` is at least the number of non-empty cached
        batches, every non-empty calibration batch contributes at least one
        row; the remaining budget is distributed proportionally to batch length
        (largest remainder, ties broken by batch order) and rows are sampled
        evenly inside each batch.
        This avoids the previous latest-batch-only bias without allocating a
        full concatenated activation tensor first. When ``max_tokens`` is
        smaller than the batch count, one row is taken from each of
        ``max_tokens`` evenly spaced batches instead.

        ``with_provenance`` additionally returns ``(source_batch, row)`` for
        every retained row, so a replay can rebuild a per-document attention
        mask and the original position ids for the packed sequence.
        """

        if not tensors:
            empty = torch.empty(0)
            return (empty, 0, []) if with_provenance else (empty, 0)
        if max_tokens <= 0:
            raise ValueError(f"AWQ token-row max_tokens must be positive, got {max_tokens}.")

        hidden = int(tensors[0].shape[-1])
        rows = []
        row_sources: List[int] = []
        total_tokens = 0
        for source_index, tensor in enumerate(tensors):
            if tensor.dim() < 2 or int(tensor.shape[-1]) != hidden:
                raise ValueError(
                    "AWQ token-row aggregation requires tensors with one common "
                    f"hidden dimension ({hidden}); got shape={tuple(tensor.shape)}."
                )
            flattened = tensor.reshape(-1, hidden)
            total_tokens += int(flattened.shape[0])
            if flattened.shape[0] > 0:
                rows.append(flattened)
                row_sources.append(source_index)

        if total_tokens == 0:
            empty = tensors[0].new_empty((1, 0, hidden))
            return (empty, 0, []) if with_provenance else (empty, 0)

        retained_tokens = min(total_tokens, max_tokens)
        if retained_tokens == total_tokens:
            packed = torch.cat(rows, dim=0).unsqueeze(0)
            if not with_provenance:
                return packed, total_tokens
            provenance = [
                (row_sources[batch_index], row_index)
                for batch_index, tensor_rows in enumerate(rows)
                for row_index in range(int(tensor_rows.shape[0]))
            ]
            return packed, total_tokens, provenance

        lengths = [int(tensor_rows.shape[0]) for tensor_rows in rows]
        quotas, local_indices_by_batch = AWQProcessor._compute_token_row_quotas(lengths, retained_tokens)

        selected = []
        provenance: List[Tuple[int, int]] = []
        for batch_index, (tensor_rows, length) in enumerate(zip(rows, lengths)):
            quota = quotas[batch_index]
            if quota <= 0:
                continue
            local_indices = local_indices_by_batch.get(batch_index)
            if local_indices is None:
                continue
            if quota == length:
                selected.append(tensor_rows)
            else:
                selected.append(tensor_rows.index_select(0, local_indices))
            if with_provenance:
                source_index = row_sources[batch_index]
                provenance.extend(
                    (source_index, int(row_index)) for row_index in local_indices.tolist()
                )
        packed = torch.cat(selected, dim=0).unsqueeze(0)
        if not with_provenance:
            return packed, total_tokens
        return packed, total_tokens, provenance

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
                feature_kwargs[_AWQ_ROTARY_POSITION_IDS] = merged_pos

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
        rows, columns = torch.nonzero(keep_mask, as_tuple=True)
        return source[rows, columns].unsqueeze(0)

    @staticmethod
    def _pack_square_token_blocks(batch_tensor: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """Pack per-batch square token tensors into one block-diagonal tensor."""

        batch = keep_mask.shape[0]
        if batch == 1:
            lengths = [int(keep_mask[0].sum().item())]
        else:
            # One reduction and one host transfer instead of one per batch row.
            lengths = keep_mask.sum(dim=1).tolist()
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

    # Forms a captured attention mask can take. The packed replay must
    # reproduce the form the model's own attention consumes: an additive float
    # mask is summed into the logits, a `bool_keep` mask marks attended keys
    # (Hugging Face SDPA), and a `bool_masked` mask marks masked keys
    # (TeleChat2's `torch.ones(S, S).triu(1)` fed to `masked_fill`).
    _MASK_FORM_ADDITIVE = "additive"
    _MASK_FORM_BOOL_KEEP = "bool_keep"
    _MASK_FORM_BOOL_MASKED = "bool_masked"
    _MASK_FORM_POSITIVE_KEEP = "positive_keep"

    @staticmethod
    def _captured_mask_form(mask: Optional[torch.Tensor]) -> Optional[str]:
        """Classify a captured mask's form without materializing a new grid.

        Bool masks appear in both polarities. A real token always attends to
        itself, so the diagonal is True for Hugging Face's keep masks and False
        for a `masked_fill` mask such as TeleChat2's `triu(1)`.
        """

        if not torch.is_tensor(mask) or mask.dim() < 2:
            return None
        if mask.dtype == torch.bool:
            m = mask
            if m.dim() == 4:
                m = m[:, 0]
            if m.dim() == 3 and m.shape[-1] == m.shape[-2] and m.shape[0] > 0:
                if bool(m.diagonal(dim1=-2, dim2=-1).any()):
                    return AWQProcessor._MASK_FORM_BOOL_KEEP
                return AWQProcessor._MASK_FORM_BOOL_MASKED
            # A planar [batch, seq] mask is a keep mask by construction.
            return AWQProcessor._MASK_FORM_BOOL_KEEP
        if mask.dtype.is_floating_point:
            # Additive masks are summed into the floating-point logits.
            return AWQProcessor._MASK_FORM_ADDITIVE
        # Integer square grids are multiplicative 0/1 keep masks, not additive
        # masks (LLaDA2 expands the tokenizer mask with `tril * padding`).
        return AWQProcessor._MASK_FORM_POSITIVE_KEEP

    @staticmethod
    def _token_row_source_structure(
        mask: Optional[torch.Tensor],
        row_count: int,
        sink_positions: Optional[torch.Tensor] = None,
        exclude_sinks: bool = False,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[str]]]:
        """Map one capture batch's kept rows back onto its document grid.

        ``row_count`` is the number of kept rows the capture hook stored for
        this batch (padding already removed). Returns ``(documents, positions,
        allowed, form)`` where ``documents``/``positions`` index the original
        ``[batch, seq]`` grid for every kept row, ``allowed`` is the
        ``[row_count, row_count]`` attention relation among those rows (True =
        attended), and ``form`` is the captured mask's form. ``None`` means the
        cached mask does not describe exactly these rows.

        ``sink_positions`` turns on ``CalibrationConfig.exclude_attention_sinks``
        handling: it is the batch's ``[B, S]`` BOS positions, and the kept set
        loses the same sink positions the capture hook dropped. Without a
        captured mask the batch is an unpadded ``[B, S]`` grid; it is rebuilt
        from that shape so retained rows keep their true positions, and
        ``form`` is ``None`` because there was no captured mask form to keep.
        """

        if row_count <= 0:
            return None
        if mask is None and exclude_sinks:
            if not torch.is_tensor(sink_positions) or sink_positions.dim() != 2:
                return None
            batch, seq_len = int(sink_positions.shape[0]), int(sink_positions.shape[1])
            keep = exclude_attention_sinks(None, sink_positions, batch=batch, seq_len=seq_len, device="cpu")
            if keep is None or int(keep.sum().item()) != row_count:
                return None
            flat_index = torch.nonzero(keep.reshape(-1), as_tuple=False).reshape(-1)
            documents = torch.div(flat_index, seq_len, rounding_mode="floor")
            positions = flat_index % seq_len
            allowed = positions.unsqueeze(1) >= positions.unsqueeze(0)
            return documents, positions, allowed, None
        if mask is None or not torch.is_tensor(mask):
            return None

        form = AWQProcessor._captured_mask_form(mask)
        m = mask.detach().to(device="cpu")
        if m.dim() == 4:
            # [B, heads, S, S]; every head sees the same keep pattern.
            m = m[:, 0]

        allowed_grid = None
        if m.dim() == 3 and m.shape[-1] == m.shape[-2] and m.shape[0] > 0:
            # [B, S, S] attention grid. Normalize it to "True = attended" using
            # the captured form, then take the diagonal as the keep mask: a
            # real token always attends to itself, while a padding query row
            # of a real HF mask still sees the real keys (only padding *keys*
            # are masked), so `any(dim=-1)` would count pad rows as real
            # tokens and the structure would never be recovered.
            if m.dtype == torch.bool:
                allowed_grid = m if form == AWQProcessor._MASK_FORM_BOOL_KEEP else ~m
            elif form == AWQProcessor._MASK_FORM_POSITIVE_KEEP:
                # Multiplicative 0/1 keep grid: positive marks an attended key.
                allowed_grid = m > 0
            else:
                # An additive mask is summed into the logits: 0 is attended and
                # only negatives suppress, so an all-zero (unpadded) grid means
                # "everything attended". There is no positive-only additive
                # convention; bool, planar and 0/1 keep masks are classified
                # above.
                allowed_grid = m >= 0
            keep = allowed_grid.diagonal(dim1=-2, dim2=-1)
        else:
            try:
                keep = normalize_seq_mask(m)
            except ValueError:
                return None

        if keep is None or keep.dim() != 2:
            return None
        if exclude_sinks:
            keep = exclude_attention_sinks(keep, sink_positions)

        seq_len = int(keep.shape[-1])
        if seq_len <= 0:
            return None
        flat_keep = keep.reshape(-1)
        if int(flat_keep.sum().item()) != row_count:
            return None

        flat_index = torch.nonzero(flat_keep, as_tuple=False).reshape(-1)
        documents = torch.div(flat_index, seq_len, rounding_mode="floor")
        positions = flat_index % seq_len
        if allowed_grid is not None:
            allowed = allowed_grid[documents, positions][:, positions]
        else:
            allowed = positions.unsqueeze(1) >= positions.unsqueeze(0)
        return documents, positions, allowed, form

    def _token_row_layer_type_masks(
        self,
        cache,
        batch_indices: List[Optional[int]],
        layer: Optional[nn.Module],
    ) -> Tuple[Dict[int, Optional[torch.Tensor]], Optional[str]]:
        """Return each capture batch's mask for ``layer``'s attention type.

        ``transformers`` 5 builds one mask per declared layer type and the
        capture hook stores every type under
        ``_gptqmodel_layer_type_replay_kwargs``. Replay must select the entry
        that matches the layer being quantized; reusing layer 0's sliding mask
        for a full-attention layer hides every long-range pair.

        Also returns the resolved layer type so the replay can tell a recurrent
        mixer from an attention block: the former never accepts an attention
        grid (issue #4671).
        """

        if layer is None:
            return {}, None
        from ..models.base import _LAYER_TYPE_REPLAY_KWARGS_KEY

        resolver = getattr(self.gptq_model, "_resolve_layer_replay_type", None)
        if not callable(resolver):
            return {}, None
        layer_input_kwargs = getattr(cache, "layer_input_kwargs", None) or []
        masks: Dict[int, Optional[torch.Tensor]] = {}
        resolved_type: Optional[str] = None
        for batch_index in batch_indices:
            if batch_index is None:
                continue
            batch_index = int(batch_index)
            if not (0 <= batch_index < len(layer_input_kwargs)):
                continue
            entry = layer_input_kwargs[batch_index] or {}
            layer_type_kwargs = entry.get(_LAYER_TYPE_REPLAY_KWARGS_KEY)
            if not layer_type_kwargs:
                continue
            layer_types = layer_type_kwargs.get("layer_types") or ()
            by_type = layer_type_kwargs.get("by_type") or {}
            layer_type = resolver(layer, layer_types)
            if layer_type is None:
                continue
            if resolved_type is None:
                resolved_type = layer_type
            type_entry = by_type.get(layer_type) or {}
            if "attention_mask" in type_entry:
                masks[batch_index] = type_entry["attention_mask"]
        return masks, resolved_type

    def _kept_row_grid_positions(
        self,
        batch_index: int,
        row_count: int,
        mask_overrides: Optional[Dict[int, Optional[torch.Tensor]]] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Map one capture batch's kept rows onto its ``[batch, seq]`` grid.

        The capture hook removes padding (and excluded attention-sink) rows
        before storing a feature, so a provenance ``row_index`` counts *kept*
        rows. Returns ``(documents, positions)`` per kept row, or ``None`` when
        the cached mask does not describe exactly these rows.
        """

        cache = getattr(self, "inputs_cache", None)
        if cache is None:
            return None
        masks = getattr(cache, "attention_masks", None) or []
        mask_overrides = mask_overrides or {}
        mask = None
        if int(batch_index) in mask_overrides:
            mask = mask_overrides[int(batch_index)]
        elif 0 <= int(batch_index) < len(masks):
            mask = masks[int(batch_index)]
        structure = self._token_row_source_structure(
            mask,
            row_count,
            self.attention_sink_positions_for_batch(batch_index),
            exclude_sinks=bool(
                getattr(getattr(self, "calibration", None), "exclude_attention_sinks", False)
            ),
        )
        if structure is None:
            return None
        documents, positions, _allowed, _form = structure
        return documents, positions

    def _pack_token_aligned_kwargs(
        self,
        provenance: List[Tuple[int, int]],
        batch_indices: List[Optional[int]],
        tensors: Optional[List[torch.Tensor]] = None,
        mask_overrides: Optional[Dict[int, Optional[torch.Tensor]]] = None,
        row_counts: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Pack per-token kwargs (for example ``input_ids``) with the replayed rows.

        ``token_rows`` keeps a bounded, evenly sampled subset of the captured
        rows. A token-aligned kwarg must be selected with the same provenance,
        or the module receives metadata for a different token count: DeepSeek-V4
        ``hash_moe`` routes with ``tid2eid[input_ids.reshape(-1)]`` and indexed
        the hidden states with a stale batch's ids (#4880).

        A provenance ``row_index`` counts *kept* rows, not grid positions: the
        capture hook strips padding (and excluded sinks) before storing the
        feature. Each index is therefore mapped back through the same
        ``_token_row_source_structure`` the mask and position replay uses, so a
        padded ``batch_size > 1`` calibration batch does not hand the router
        another document's id (#5015). The identity is used only when the
        capture kept every grid row.

        Values that cannot be aligned are skipped here; the replay guard in
        :meth:`_prepare_module_forward` turns the remaining mismatch into an
        ``_AWQScaleGroupError`` so the configured fallback applies instead of an
        unhandled ``IndexError``.
        """

        cache = getattr(self, "inputs_cache", None)
        layer_kwargs = getattr(cache, "layer_input_kwargs", None) or []
        if not provenance or not layer_kwargs:
            return {}

        packed: Dict[str, Any] = {}
        for key in _AWQ_TOKEN_ALIGNED_KWARGS:
            rows: List[torch.Tensor] = []
            usable = True
            structures: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}
            for source_index, row_index in provenance:
                batch_index = (
                    batch_indices[source_index]
                    if source_index < len(batch_indices)
                    else None
                )
                if batch_index is None or not 0 <= int(batch_index) < len(layer_kwargs):
                    usable = False
                    break
                entry = layer_kwargs[int(batch_index)]
                value = entry.get(key) if isinstance(entry, Mapping) else None
                if not torch.is_tensor(value) or value.numel() == 0:
                    usable = False
                    break
                grid = value.detach().reshape(-1)
                if not 0 <= int(row_index) < int(grid.numel()):
                    usable = False
                    break
                if source_index not in structures:
                    row_count = None
                    if row_counts is not None and source_index < len(row_counts):
                        row_count = row_counts[source_index]
                    elif tensors is not None and source_index < len(tensors):
                        source = tensors[source_index]
                        hidden = max(int(source.shape[-1]), 1)
                        row_count = int(source.numel() // hidden)
                    mapping = None
                    if row_count is not None:
                        mapping = self._kept_row_grid_positions(
                            int(batch_index), row_count, mask_overrides,
                        )
                    structures[source_index] = mapping
                mapping = structures[source_index]
                if mapping is None:
                    # No structure: safe only when the capture kept every grid
                    # row, so the kept index is the grid index.
                    if int(row_index) >= int(grid.numel()):
                        usable = False
                        break
                    rows.append(grid[int(row_index)].reshape(1))
                    continue
                documents, positions = mapping
                if not 0 <= int(row_index) < int(documents.numel()):
                    usable = False
                    break
                document = int(documents[int(row_index)])
                position = int(positions[int(row_index)])
                seq_len = int(value.shape[-1]) if value.dim() >= 2 else int(grid.numel())
                if seq_len <= 0:
                    usable = False
                    break
                flat_index = document * seq_len + position
                if not 0 <= flat_index < int(grid.numel()):
                    usable = False
                    break
                rows.append(grid[flat_index].reshape(1))
            if not usable or len(rows) != len(provenance):
                continue
            packed[key] = torch.cat(rows, dim=0).unsqueeze(0)
        return packed

    def _token_row_replay_kwargs(
        self,
        tensors: List[torch.Tensor],
        batch_indices: List[Optional[int]],
        provenance: List[Tuple[int, int]],
        dtype: torch.dtype,
        mask_overrides: Optional[Dict[int, Optional[torch.Tensor]]] = None,
        recurrent: bool = False,
        row_counts: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Rebuild the attention mask and positions for packed ``token_rows``.

        ``token_rows`` concatenates rows sampled from several calibration
        documents into one ``[1, tokens, hidden]`` replay tensor. Replaying it
        with ``attention_mask=None`` gives the attention module a single causal
        mask across that synthetic sequence, so document A attends to document
        B, and models whose attention requires the mask argument raise instead
        (issue #4091). Keeping every retained row on its original document and
        position makes the replay attention block-diagonal per document, and
        rotary sees the original offsets instead of synthetic ones.

        The rebuilt mask keeps the captured mask's form and polarity
        (`_captured_mask_form`): an additive float grid stays additive, a bool
        keep grid stays bool with True = attended, and a `masked_fill` bool grid
        (TeleChat2) stays bool with True = masked. Converting a bool grid to an
        additive float one breaks consumers that call `masked_fill`.
        """

        cache = getattr(self, "inputs_cache", None)
        masks = getattr(cache, "attention_masks", None) or []
        mask_overrides = mask_overrides or {}
        token_aligned = self._pack_token_aligned_kwargs(
            provenance, batch_indices, tensors, mask_overrides, row_counts=row_counts,
        )
        if cache is None or not provenance or (not masks and not mask_overrides):
            # Without a captured mask there is no document structure to replay;
            # keep the previous kwargs rather than invent one. Token-aligned
            # kwargs are still packed: `hash_moe` routing needs them even when
            # the module's attention takes no mask (#4880).
            return token_aligned

        cached_position_ids = getattr(cache, "position_ids", None) or []

        packed_count = len(provenance)
        allowed = torch.zeros((packed_count, packed_count), dtype=torch.bool)
        positions = torch.zeros((packed_count,), dtype=torch.long)
        # A captured mask that described an [S, S] attention grid, as opposed
        # to a planar [batch, seq] keep mask.
        grid_mask = False
        form = None
        mask_dtype = None

        grouped: Dict[int, List[Tuple[int, int]]] = {}
        for packed_index, (source_index, row_index) in enumerate(provenance):
            grouped.setdefault(int(source_index), []).append((packed_index, int(row_index)))

        for source_index, entries in grouped.items():
            if row_counts is not None and source_index < len(row_counts):
                row_count = row_counts[source_index]
            elif source_index < len(tensors):
                source = tensors[source_index]
                hidden = max(int(source.shape[-1]), 1)
                row_count = int(source.numel() // hidden)
            else:
                row_count = 0
            batch_index = (
                batch_indices[source_index] if source_index < len(batch_indices) else None
            )
            mask = None
            if batch_index is not None:
                if int(batch_index) in mask_overrides:
                    mask = mask_overrides[int(batch_index)]
                elif 0 <= int(batch_index) < len(masks):
                    mask = masks[int(batch_index)]
            if torch.is_tensor(mask):
                if mask.dim() >= 3:
                    grid_mask = True
                if form is None:
                    form = self._captured_mask_form(mask)
                if mask_dtype is None:
                    mask_dtype = mask.dtype

            packed_indices = torch.tensor([item[0] for item in entries], dtype=torch.long)
            local_rows = torch.tensor([item[1] for item in entries], dtype=torch.long)
            structure = self._token_row_source_structure(
                mask,
                row_count,
                self.attention_sink_positions_for_batch(batch_index),
                exclude_sinks=bool(
                    getattr(getattr(self, "calibration", None), "exclude_attention_sinks", False)
                ),
            )
            if structure is None:
                documents = torch.zeros_like(local_rows)
                source_positions = local_rows.clone()
                source_allowed = (
                    source_positions.unsqueeze(1) >= source_positions.unsqueeze(0)
                )
            else:
                source_documents, kept_positions, kept_allowed, structure_form = structure
                documents = source_documents[local_rows]
                source_positions = kept_positions[local_rows]
                source_allowed = kept_allowed[local_rows][:, local_rows]
                if form is None:
                    form = structure_form

            same_document = documents.unsqueeze(1) == documents.unsqueeze(0)
            allowed[
                packed_indices.unsqueeze(1), packed_indices.unsqueeze(0)
            ] = source_allowed & same_document

            row_positions = source_positions
            if batch_index is not None and 0 <= int(batch_index) < len(cached_position_ids):
                cached = cached_position_ids[int(batch_index)]
                if torch.is_tensor(cached):
                    cached = cached.detach().to(device="cpu")
                    usable = (
                        cached.dim() == 2
                        and int(cached.shape[-1]) > int(source_positions.max())
                    )
                    if usable and int(cached.shape[0]) == 1:
                        row_positions = cached[0, source_positions]
                    elif usable and int(cached.shape[0]) > int(documents.max()):
                        row_positions = cached[documents, source_positions]

            positions[packed_indices] = row_positions.to(dtype=torch.long)

        if recurrent:
            # A recurrent / convolution mixer consumes a planar [batch, seq]
            # keep mask (`apply_mask_to_padding_states` multiplies
            # `hidden * attention_mask[:, :, None]`). The additive
            # [1, 1, T, T] grid is an attention-only contract and raises on the
            # first replay: `tensor a (768) must match tensor b (366)`. The
            # mixer applies its own causal convolution/scan, so the packed rows
            # need no explicit causal structure. `_pack_token_rows` emits rows
            # in source order, so each document stays contiguous and the
            # recurrent state crosses only at a document boundary -- the same
            # limitation the padded multi-row batch path already has.
            attention_mask = torch.ones((1, packed_count), dtype=torch.bool)
        elif not grid_mask and form is not None:
            # A captured planar [batch, seq] mask means the attention
            # implementation (for example FlashAttention) expects a keep mask
            # of that rank. Keep that form; every packed row is a kept token,
            # so cross-document separation is not expressible here and causal
            # masking stays inside the implementation.
            attention_mask = torch.ones((1, packed_count), dtype=torch.bool)
        elif form == self._MASK_FORM_BOOL_MASKED:
            # Preserve the captured polarity: True marks a masked key.
            attention_mask = (~allowed).unsqueeze(0).unsqueeze(0)
        elif form == self._MASK_FORM_BOOL_KEEP:
            # Preserve the captured polarity: True marks an attended key.
            attention_mask = allowed.unsqueeze(0).unsqueeze(0)
        elif form == self._MASK_FORM_POSITIVE_KEEP:
            # Preserve the captured multiplicative 0/1 keep grid (LLaDA2).
            attention_mask = allowed.to(
                dtype=mask_dtype if mask_dtype is not None else torch.int64
            ).unsqueeze(0).unsqueeze(0)
        else:
            # Additive block-diagonal form. This is also the fallback when no
            # mask was captured: the module then relied on its internal causal
            # path, and an explicit causal mask is the equivalent replay.
            if not dtype.is_floating_point:
                dtype = torch.float32
            additive = torch.zeros((packed_count, packed_count), dtype=dtype)
            additive.masked_fill_(~allowed, torch.finfo(dtype).min)
            attention_mask = additive.unsqueeze(0).unsqueeze(0)
        return {
            **token_aligned,
            "attention_mask": attention_mask,
            "position_ids": positions.unsqueeze(0),
            _AWQ_ROTARY_POSITION_IDS: positions.unsqueeze(0),
        }

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
        if not torch.is_tensor(inp) or inp.dim() < 2 or not torch.is_tensor(attention_mask):
            return aligned_kwargs

        # `_iter_module_forward_outputs` promotes [tokens, hidden] captures to
        # [1, tokens, hidden] for attention modules after this helper runs.
        # Treat that 2-D form as a single batch here so expanded HF masks do
        # not bypass alignment and fail during replay.
        input_batch = int(inp.shape[0]) if inp.dim() >= 3 else 1
        input_seq = int(inp.shape[1]) if inp.dim() >= 3 else int(inp.shape[0])
        if input_batch != 1 or attention_mask.ndim < 2:
            return aligned_kwargs

        # A mask that already matches the packed replay length needs no
        # repacking. Repacking re-derives a keep mask with Hugging Face's bool
        # convention, which misreads a `masked_fill` mask (True = masked) and
        # would drop the mask entirely.
        if (
            attention_mask.shape[0] == 1
            and attention_mask.shape[-1] == input_seq
            and (attention_mask.ndim < 3 or attention_mask.shape[-2] in (1, input_seq))
        ):
            return aligned_kwargs

        try:
            keep_mask = normalize_seq_mask(attention_mask)
        except Exception:
            return aligned_kwargs

        if keep_mask is None or keep_mask.ndim != 2:
            return aligned_kwargs

        total_kept = int(keep_mask.to(dtype=torch.int64).sum().item())
        # Multi-row masks are intentionally packed below. Only a single-row
        # mask can be an expanded max-cache mask that needs sequence cropping.
        mask_seq_matches = attention_mask.shape[0] != 1 or (
            attention_mask.shape[-1] == input_seq
            and (attention_mask.ndim < 3 or attention_mask.shape[-2] in (1, input_seq))
        )
        if total_kept != input_seq or not mask_seq_matches:
            # Transformers may hand a layer a max-cache causal mask (for
            # example [1, 1, 2048, 2048]) while the captured replay input is
            # only 232 tokens. Crop the metadata to the actual sequence so
            # attention remains causal without padding activations to 2048.
            # For unsupported/batched layouts retain the defensive drop used
            # by the original path rather than passing an incompatible mask.
            cropped_mask = attention_mask
            if attention_mask.shape[0] == 1:
                if attention_mask.ndim == 2 and attention_mask.shape[-1] >= input_seq:
                    cropped_mask = attention_mask[..., :input_seq]
                elif attention_mask.ndim == 3 and attention_mask.shape[-1] >= input_seq:
                    if attention_mask.shape[-2] == 1 or attention_mask.shape[-2] >= input_seq:
                        cropped_mask = attention_mask[..., :input_seq, :input_seq]
                elif attention_mask.ndim == 4 and attention_mask.shape[-1] >= input_seq:
                    if attention_mask.shape[-2] == 1:
                        cropped_mask = attention_mask[..., :1, :input_seq]
                    elif attention_mask.shape[-2] >= input_seq:
                        cropped_mask = attention_mask[..., :input_seq, :input_seq]

            if cropped_mask is attention_mask:
                aligned_kwargs.pop("attention_mask", None)
                aligned_kwargs.pop("position_ids", None)
                return aligned_kwargs

            aligned_kwargs["attention_mask"] = cropped_mask
            position_ids = aligned_kwargs.get("position_ids")
            if torch.is_tensor(position_ids) and position_ids.ndim >= 2 and position_ids.shape[-1] >= input_seq:
                aligned_kwargs["position_ids"] = position_ids[..., :input_seq]
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

        values: List[torch.Tensor] = []
        for named_module in previous_subset.values():
            weight = getattr(named_module.module, "weight", None)
            if weight is None:
                continue
            with torch.no_grad():
                values.append(weight.detach().abs().mean())

        if not values:
            return None

        if len({value.device for value in values}) != 1:
            return float(sum(float(value.item()) for value in values) / len(values))
        return float(sum(torch.stack(values).tolist()) / len(values))

    def _previous_subset_scale(
        self,
        state: _AWQLayerState,
        previous_subset: Optional[Dict[str, NamedModule]],
    ) -> Optional[float]:
        """Returns the cached previous-subset weight scale for one stage."""

        if not previous_subset:
            return None
        if state.previous_weight_scale_source is not previous_subset:
            state.previous_weight_scale = self._capture_previous_subset_scale(previous_subset)
            # The mapping is alive for the whole stage and is replaced by the
            # next subset, so this reference does not extend module lifetimes
            # beyond the stage. ``_finalize_quantized_layer`` drops it early.
            state.previous_weight_scale_source = previous_subset
        return state.previous_weight_scale

    @staticmethod
    def _sibling_capture_get(
        cache: Dict[Tuple[int, ...], Tuple[Tuple[Any, ...], Any]],
        tensors: List[torch.Tensor],
    ) -> Tuple[Tuple[int, ...], Any]:
        """Return the payload cached for exactly these captured tensors.

        Sibling projections are hooked separately but read one activation
        object, so the identity of their cached capture tensors names the
        shared row set (and the replay kwargs derived from it). ``id()`` is
        only stable while the object lives, and a routed-expert reservoir
        materializes a fresh ``buffer[:used].unsqueeze(0)`` view per module
        (#5236): the loop drops the previous module's view when it rebinds
        ``tensors``, so CPython can hand the same id to the next module's
        view. A dead weak reference proves that id was free to be reused, so
        it must not match; a live one must be the same object. Keeping only
        weak references also avoids retaining every reservoir buffer for the
        rest of the layer (#5267).
        """

        cache_key = tuple(id(tensor) for tensor in tensors)
        cached = cache.get(cache_key)
        if cached is None:
            return cache_key, None
        refs, payload = cached
        if len(refs) != len(tensors) or any(
            ref() is not tensor for ref, tensor in zip(refs, tensors)
        ):
            # Either an entry whose tensors were freed (its id may now belong
            # to this capture) or a corrupt entry: recompute the payload.
            return cache_key, None
        return cache_key, payload

    @staticmethod
    def _sibling_capture_put(
        cache: Dict[Tuple[int, ...], Tuple[Tuple[Any, ...], Any]],
        cache_key: Tuple[int, ...],
        tensors: List[torch.Tensor],
        payload: Any,
    ) -> None:
        """Cache one shared capture payload without owning the capture tensors."""

        cache[cache_key] = (tuple(weakref.ref(tensor) for tensor in tensors), payload)

    def _layer_input_features(self, state: _AWQLayerState) -> Dict[str, torch.Tensor]:
        """Collapse cached per-batch inputs into one replay tensor per module.

        Model-specific pointwise policies pack variable-length batches into
        bounded token rows, so every calibration batch contributes to AWQ scale
        statistics. Generic sequence-sensitive modules retain the legacy
        behavior: concatenate equal shapes, otherwise keep the latest feature
        and aligned kwargs.
        """

        features: Dict[str, torch.Tensor] = {}
        feature_kwargs: Dict[str, Dict[str, Any]] = {}
        feature_stats: Dict[str, Dict[str, Any]] = {}
        feature_names = list(state.modules)
        roots = {name.split(".", 1)[0] for name in feature_names}
        for root in sorted(roots):
            if root in self.tasks and root not in feature_names:
                policy = self._feature_aggregation_policy(root)
                if policy and policy.get("capture_root", False):
                    feature_names.append(root)

        self._feature_task_names = set(feature_names)
        # Sibling modules can share one host tensor per batch, so their cached
        # input lists are the same objects in the same order. Concatenate that
        # identical list once and hand every sibling the same result.
        cat_cache: Dict[Tuple[int, ...], Tuple[Tuple[Any, ...], torch.Tensor]] = {}
        # Sibling projections share one captured input list, so their packed
        # replay mask and position ids are identical too. Build the (bounded,
        # but quadratic) block-diagonal mask once per unique capture list.
        token_row_kwargs_cache: Dict[Tuple[int, ...], Tuple[Tuple[Any, ...], Dict[str, Any]]] = {}
        # Iterate over a snapshot since quantization may mutate state.modules concurrently
        for name in feature_names:
            entry = self.tasks.get(name) or {}
            if entry.get("reservoir_buffer") is not None:
                self._materialize_reservoir_provenance(entry)
            tensors: List[torch.Tensor] = entry.get("inputs", [])  # type: ignore[arg-type]
            batch_indices: List[Optional[int]] = entry.get("batch_indices", [])  # type: ignore[arg-type]
            if not tensors:
                features[name] = torch.empty(0)
                feature_kwargs[name] = {}
                feature_stats[name] = {
                    "mode": "empty",
                    "raw_tokens": 0,
                    "retained_tokens": 0,
                    "batches": 0,
                }
                continue

            # Capture hooks can complete in device/runtime order, not
            # necessarily calibration-batch order.  Restore the serial replay
            # order before concatenation (or before choosing the last
            # variable-length capture) so activation statistics and their
            # cached kwargs stay aligned with the calibration corpus.
            if (
                len(batch_indices) == len(tensors)
                and all(index is not None and index >= 0 for index in batch_indices)
            ):
                order = sorted(range(len(tensors)), key=lambda index: int(batch_indices[index]))
                batch_indices = [batch_indices[index] for index in order]
                tensors = [tensors[index] for index in order]
                self._reorder_token_row_capture(entry, order)

            policy = self._feature_aggregation_policy(name)
            aggregation_mode = (policy or {}).get("mode")
            if aggregation_mode == "token_rows":
                if "provenance" in entry and entry["provenance"]:
                    cache_key, cat_tensor = self._sibling_capture_get(cat_cache, tensors)
                    if cat_tensor is None:
                        cat_tensor = (
                            torch.cat(tensors, dim=-2)
                            if tensors[0].dim() == 3
                            else torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=0).unsqueeze(0)
                        )
                        self._sibling_capture_put(cat_cache, cache_key, tensors, cat_tensor)
                    features[name] = cat_tensor
                    raw_tokens = entry.get("raw_tokens", int(features[name].shape[-2]))
                    retained_tokens = int(features[name].shape[-2])
                    max_tokens = entry.get("max_tokens", retained_tokens)
                    provenance = entry["provenance"]

                    kwargs_key, replay_kwargs = self._sibling_capture_get(
                        token_row_kwargs_cache, tensors
                    )
                    if replay_kwargs is None:
                        type_masks, replay_layer_type = self._token_row_layer_type_masks(
                            getattr(self, "inputs_cache", None),
                            batch_indices,
                            state.layer_module,
                        )
                        replay_kwargs = self._token_row_replay_kwargs(
                            tensors,
                            batch_indices,
                            provenance,
                            features[name].dtype,
                            mask_overrides=type_masks,
                            recurrent=_is_recurrent_attention_layer_type(replay_layer_type),
                            row_counts=entry.get("batch_row_counts"),
                        )
                        self._sibling_capture_put(
                            token_row_kwargs_cache, kwargs_key, tensors, replay_kwargs
                        )
                    feature_kwargs[name] = replay_kwargs
                    entry["inputs"] = [features[name]]
                    entry["batch_indices"] = [None]
                    feature_stats[name] = {
                        "mode": "token_rows",
                        "raw_tokens": raw_tokens,
                        "retained_tokens": retained_tokens,
                        "batches": len(entry.get("batch_row_counts") or tensors),
                        "max_tokens": max_tokens,
                    }
                else:
                    configured_max_tokens = (policy or {}).get("max_tokens")
                    max_tokens = (
                        self._cap_token_rows(int(configured_max_tokens))
                        if configured_max_tokens is not None
                        else self._cap_token_rows(self._token_row_budget(tensors))
                    )
                    if max_tokens == 0 and configured_max_tokens is None:
                        features[name] = tensors[0].new_empty((1, 0, tensors[0].shape[-1]))
                        raw_tokens = 0
                        replay_kwargs = {}
                    else:
                        features[name], raw_tokens, provenance = self._pack_token_rows(
                            tensors,
                            max_tokens=max_tokens,
                            with_provenance=True,
                        )
                        cache_key, replay_kwargs = self._sibling_capture_get(
                            token_row_kwargs_cache, tensors
                        )
                        if replay_kwargs is None:
                            type_masks, replay_layer_type = self._token_row_layer_type_masks(
                                getattr(self, "inputs_cache", None),
                                batch_indices,
                                state.layer_module,
                            )
                            replay_kwargs = self._token_row_replay_kwargs(
                                tensors,
                                batch_indices,
                                provenance,
                                features[name].dtype,
                                mask_overrides=type_masks,
                                recurrent=_is_recurrent_attention_layer_type(replay_layer_type),
                            )
                            self._sibling_capture_put(
                                token_row_kwargs_cache, cache_key, tensors, replay_kwargs
                            )
                    feature_kwargs[name] = replay_kwargs
                    retained_tokens = int(features[name].shape[-2])
                    entry["inputs"] = [features[name]]
                    entry["batch_indices"] = [None]
                    feature_stats[name] = {
                        "mode": "token_rows",
                        "raw_tokens": raw_tokens,
                        "retained_tokens": retained_tokens,
                        "batches": len(tensors),
                        "max_tokens": max_tokens,
                    }
            elif aggregation_mode not in (None, "batch"):
                raise ValueError(
                    f"Unsupported AWQ feature aggregation mode {aggregation_mode!r} "
                    f"for {name!r}."
                )
            elif self._can_concat_batch_tensors(tensors):
                cat_key, cat_tensor = self._sibling_capture_get(cat_cache, tensors)
                if cat_tensor is None:
                    cat_tensor = torch.cat(tensors, dim=0)
                    self._sibling_capture_put(cat_cache, cat_key, tensors, cat_tensor)
                features[name] = cat_tensor
                feature_kwargs[name] = self._feature_kwargs_from_batch_indices(batch_indices)
                entry["inputs"] = [features[name]]
                entry["batch_indices"] = [None]
                hidden = max(int(features[name].shape[-1]), 1)
                tokens = int(features[name].numel() // hidden)
                feature_stats[name] = {
                    "mode": "batch",
                    "raw_tokens": tokens,
                    "retained_tokens": tokens,
                    "batches": len(tensors),
                }
            else:
                # Variable-length captures such as `[1, 423, H]` and `[1, 36, H]`
                # cannot be concatenated on dim 0. Keep the latest capture and
                # reuse metadata from the same batch index so replay stays aligned.
                features[name] = tensors[-1]
                last_batch_index = batch_indices[-1] if batch_indices else None
                feature_kwargs[name] = self._feature_kwargs_from_batch_indices([last_batch_index])
                entry["inputs"] = [features[name]]
                entry["batch_indices"] = [last_batch_index]
                hidden = max(int(features[name].shape[-1]), 1)
                raw_tokens = sum(int(tensor.numel() // hidden) for tensor in tensors)
                retained_tokens = int(features[name].numel() // hidden)
                feature_stats[name] = {
                    "mode": "latest_batch",
                    "raw_tokens": raw_tokens,
                    "retained_tokens": retained_tokens,
                    "batches": len(tensors),
                }
            if features[name] is not None and features[name].numel() > 0:
                pass  # previously logged input feature shapes
                # log.info(
                #     "AWQProcessor: input feature `%s` shape=%s",
                #     name,
                #     tuple(features[name].shape),
                # )

        self._awq_feature_kwargs = feature_kwargs
        self._feature_stats = feature_stats
        return features

    def _feature_nsamples_for_log(self, feature_stat: Mapping[str, Any]) -> int:
        """Return the sample count exposed through the legacy quantization log."""

        if feature_stat.get("mode") == "token_rows":
            retained_tokens = int(feature_stat.get("retained_tokens", 0))
            if retained_tokens > 0:
                return retained_tokens
        return self._nsamples_total

    def _finalize_quantized_layer(
        self,
        state: _AWQLayerState,
        fallback_task_names: Optional[Set[str]] = None,
    ) -> None:
        """Release all per-layer AWQ state after a successful completion."""

        state.quantized = True
        state.modules.clear()
        state.pending_modules.clear()
        state.layer_module = None
        state.processed_subsets.clear()
        state.subset_total = None
        state.previous_weight_scale = None
        state.previous_weight_scale_source = None

        task_names = self._feature_task_names or (fallback_task_names or set())
        with self.lock:
            for name in task_names:
                task_entry = self.tasks.pop(name, None)
                if task_entry and "inputs" in task_entry:
                    task_entry["inputs"].clear()
                getattr(self, "_reservoir_locks", {}).pop(name, None)

        self._feature_task_names = set()
        self._feature_stats = {}
        self._scale_feature_by_module = {}
        self._awq_feature_kwargs = {}
        with self.lock:
            self._capture_host_cache.clear()
            if hasattr(self, "_capture_sampled_cache"):
                self._capture_sampled_cache.clear()
            if hasattr(self, "_token_row_plans"):
                self._token_row_plans.clear()
            if hasattr(self, "_token_row_lengths_cache"):
                self._token_row_lengths_cache.clear()
            self._capture_host_batch = None

        if hasattr(self._scale_context, "layer_index"):
            delattr(self._scale_context, "layer_index")
        if hasattr(self._scale_context, "prev_scale"):
            delattr(self._scale_context, "prev_scale")

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

        self._finalize_quantized_layer(state, fallback_task_names=set(named_childs))

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

    def _starved_group_members(
        self,
        layer_names: List[str],
        input_feat: Dict[str, torch.Tensor],
    ) -> List[str]:
        """Returns the scaling-group members that lack enough activations.

        Coverage is evaluated per layer. Summing retained tokens over the
        group's siblings let q/k/v (3x) and gate/up (2x) pass a percentage
        threshold while a lone down_proj fell below it and silently became
        RTN (#4063).

        The members are returned instead of a single verdict because a group
        does not always have to be all-or-nothing. ``_search_best_scale`` reads
        one captured input per group, and a member's own captured rows only
        reach the objective through the replayed block. A member with no (or
        too few) rows therefore contributes nothing to the scale search, yet
        the norm it reads is still divided by ``s``, so its weights must still
        be multiplied by ``s``. Callers fold the scale into these members and
        quantize them with the fallback strategy while the well-covered
        members keep AWQ (#4675).
        """

        from ..utils.fallback import should_use_fallback

        expected_tokens = getattr(self, "total_calibration_tokens", None) or self._nsamples_total
        expected = float(expected_tokens) if expected_tokens else None
        starved = []
        for name in layer_names:
            feat = input_feat.get(name)
            if feat is None or feat.numel() == 0:
                starved.append(name)
                continue
            if should_use_fallback(self.fallback, float(self._captured_tokens(name, feat)), expected):
                starved.append(name)

        return starved

    def _captured_tokens(self, name: str, feat: torch.Tensor) -> int:
        """Rows of calibration data the scale search would see for one module."""

        feature_stat = self._feature_stats.get(name, {})
        if feature_stat.get("mode") == "token_rows" and "raw_tokens" in feature_stat:
            # Fallback asks "did enough of the calibration corpus route to
            # this module", so compare the raw routed token count against
            # the expected corpus total. Token-row retention is a deliberate
            # sampling bound and must not trigger low-coverage fallback.
            return int(feature_stat["raw_tokens"])
        if "retained_tokens" in feature_stat:
            # Other modes must measure the rows that scale search actually
            # receives. In particular, latest-batch aggregation discards
            # earlier variable-length captures.
            return int(feature_stat["retained_tokens"])
        hidden = feat.shape[-1] if feat.dim() >= 1 else 1
        return feat.numel() // max(hidden, 1)

    def _group_coverage_is_sufficient(
        self,
        layer_names: List[str],
        input_feat: Dict[str, torch.Tensor],
    ) -> bool:
        """Whether the group as a whole has enough routed rows to search with.

        Each member of a routed-expert gate/up group can fall below the
        per-module threshold while the layer still has thousands of routed
        tokens. Only a group with no usable data at all has to fall back
        wholesale (#4675).
        """

        from ..utils.fallback import should_use_fallback

        expected_tokens = getattr(self, "total_calibration_tokens", None) or self._nsamples_total
        expected = float(expected_tokens) if expected_tokens else None
        total = 0.0
        for name in layer_names:
            feat = input_feat.get(name)
            if feat is None or feat.numel() == 0:
                continue
            total += float(self._captured_tokens(name, feat))
        if total <= 0:
            return False
        return not should_use_fallback(self.fallback, total, expected)

    def _should_fallback_group(
        self,
        layer_names: List[str],
        input_feat: Dict[str, torch.Tensor],
    ) -> bool:
        """Returns whether any layer in a scaling group lacks enough activations.

        A group shares one scale vector, so a starved member always has to fall
        back. Callers that can fold the scale into the starved member and keep
        AWQ for its siblings use ``_starved_group_members`` instead (#4675).
        """

        return bool(self._starved_group_members(layer_names, input_feat))

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
        self._scale_feature_by_module = {}
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
                "AWQProcessor: %s produced no scaling groups for layer %s. "
                "AWQ scale search is disabled for this layer; using plain quantization.",
                type(self.gptq_model).__name__,
                layer_index,
            )
            self._quantize_layer_fallback(
                layer_index,
                state,
                "no AWQ scaling groups were generated",
            )
            return

        sanitized_module_config: List[Dict] = []
        for entry in module_config:
            entry = dict(entry)
            inspect_module = entry.get("module2inspect") or layer_module_ref
            entry_kwargs = entry.get("kwargs") or module_kwargs_global
            entry["kwargs"] = self._sanitize_kwargs(entry_kwargs, inspect_module)
            sanitized_module_config.append(entry)

        self._validate_scale_group_conflicts(layer_module_ref, sanitized_module_config)
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
            search_layer_names = layer_names
            # The split below is only sound when the search reads a captured
            # input that is not one of the group's members: routed-expert
            # gate/up groups replay the MoE root capture, and native routing may
            # leave a member unexecuted without the group being broken.
            shared_search_input = cfg.get("_input_feature_name") not in layer_names
            if shared_search_input:
                cfg["_awq_allow_partial_replay"] = True
            if self.fallback and self._should_fallback_group(layer_names, input_feat):
                starved_names = self._starved_group_members(layer_names, input_feat)
                covered_names = [name for name in layer_names if name not in starved_names]
                if not shared_search_input or not starved_names:
                    # A caller-level veto or a dense group: the whole group
                    # still falls back.
                    fallback_names.update(layer_names)
                    continue
                if covered_names:
                    # One starved expert must not send its whole MoE gate/up
                    # group to RTN (#4675). The starved members contribute no
                    # rows to the scale search, but the norm they read is still
                    # divided by the folded scale, so keep them in the fold and
                    # quantize them with the fallback strategy after it.
                    search_layer_names = covered_names
                    fallback_names.update(starved_names)
                    log.warning(
                        "AWQProcessor: layer %s keeps AWQ for %d of %d members of a scaling group; "
                        "%d starved member(s) are scale-folded then plain-quantized: %s",
                        layer_index,
                        len(covered_names),
                        len(layer_names),
                        len(starved_names),
                        starved_names[:6],
                    )
                elif self._group_coverage_is_sufficient(layer_names, input_feat):
                    # Every member is individually below the threshold, but the
                    # group as a whole has routed data: search the members that
                    # the replay executes and fold the rest.
                    log.warning(
                        "AWQProcessor: layer %s has no individually-covered member in a "
                        "scaling group, but the group's routed tokens pass the threshold; "
                        "searching the executed members instead of falling back the group",
                        layer_index,
                    )
                else:
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

            missing_layers = [name for name in search_layer_names if name not in input_feat]
            if missing_layers:
                skipped_groups.append((layer_names, missing_layers))
                continue
            if search_layer_names is not layer_names:
                covered = set(search_layer_names)
                cfg["_awq_search_layers"] = [
                    layer
                    for layer in layers_sample
                    if (
                        get_op_name(layer_module_ref, layer)
                        if isinstance(layer, nn.Module)
                        else str(layer)
                    ) in covered
                ]
            filtered_module_config.append(cfg)

        if skipped_groups:
            log.debug(
                "AWQProcessor: layer %s skipping %d scaling groups due to missing features (sample=%s)",
                layer_index,
                len(skipped_groups),
                skipped_groups[:3],
            )

        sanitized_module_config = filtered_module_config
        # Siblings such as q/k/v can share one host tensor object, so several names
        # may map to the same id. Keep the first (declaration order) for the
        # telemetry column instead of letting dict order pick one arbitrarily.
        feature_name_by_id: Dict[int, str] = {}
        for name, tensor in input_feat.items():
            feature_name_by_id.setdefault(id(tensor), name)
        for cfg in sanitized_module_config:
            feature_name = cfg.pop("_input_feature_name", None)
            if feature_name not in input_feat:
                feature_name = feature_name_by_id.get(id(cfg.get("inp")))
            if feature_name is None:
                continue
            for layer in cfg.get("layers") or []:
                layer_name = (
                    get_op_name(layer_module_ref, layer)
                    if isinstance(layer, torch.nn.Module)
                    else str(layer)
                )
                self._scale_feature_by_module[layer_name] = feature_name
        if not sanitized_module_config and not fallback_names:
            log.warning(
                "AWQProcessor: no valid scaling groups for layer %s after filtering; falling back to plain quantization.",
                layer_index,
            )
            self._quantize_layer_fallback(
                layer_index,
                state,
                "no scaling group survived filtering",
            )
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

        # Non-quantized consumers of a folded norm (MoE router, qwen2
        # `shared_expert_gate`) must receive the same scale, or their output -
        # and therefore the routing - changes. Collect them per fold source.
        consumer_map: Dict[str, List[str]] = {}
        scale_consumers_by_group: List[Optional[List[str]]] = []
        for cfg in sanitized_module_config:
            consumer_names = cfg.pop("_scale_consumers", None)
            scale_consumers_by_group.append(consumer_names)
            if not consumer_names:
                continue
            try:
                prev_op_name = get_op_name(layer_module_ref, cfg.get("prev_op"))
            except ValueError:
                continue
            consumer_map.setdefault(prev_op_name, [])
            for consumer_name in consumer_names:
                if consumer_name not in consumer_map[prev_op_name]:
                    consumer_map[prev_op_name].append(consumer_name)

        scales_list = []
        for layer, scale_consumers in zip(sanitized_module_config, scale_consumers_by_group):
            search_layers = layer.pop("_awq_search_layers", None)
            allow_partial_replay = bool(layer.pop("_awq_allow_partial_replay", False))
            allow_outside_fold = bool(layer.pop("_awq_allow_outside_fold", False))
            call_kwargs = dict(layer)
            if search_layers is not None:
                # Search with the covered members only; fold into every member.
                call_kwargs["fold_layers"] = layer["layers"]
                call_kwargs["layers"] = search_layers
            try:
                scales_list.append(self._search_best_scale(
                    layer_module_ref,
                    scale_consumers=scale_consumers,
                    allow_partial_replay=allow_partial_replay,
                    allow_outside_fold=allow_outside_fold,
                    **call_kwargs,
                ))
            except _AWQScaleGroupError as exc:
                if not self.fallback:
                    raise
                layer_names = [get_op_name(layer_module_ref, target) for target in layer["layers"]]
                fallback_names.update(layer_names)
                log.warning(
                    "AWQProcessor: layer %s using configured plain-quantization fallback for %s: %s",
                    layer_index, layer_names, exc,
                )

        try:
            apply_scale(
                layer_module_ref,
                scales_list,
                input_feat_dict=input_feat,
                consumer_map=consumer_map,
            )
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
        scales_list = append_str_prefix(
            scales_list,
            get_op_name(self.model, layer_module_ref) + ".",
        )

        staged_teacher = None
        staged_gsq = getattr(self.qcfg, "gsq_training", None)
        if staged_gsq is not None and staged_gsq.enabled:
            from .gsq_training_awq import capture_awq_staged_teacher

            plan = _model_module_plan(self.gptq_model)
            projections = tuple(
                name for name, named in named_childs.items()
                if plan is None or plan.roles(name)
            )
            staged_teacher = capture_awq_staged_teacher(
                self, layer_module_ref, named_childs, fallback_names, projections=projections,
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

        self.apply_quant(named_childs, scales_list)

        if staged_teacher is not None:
            from .gsq_training_awq import refine_awq_staged_layer

            refine_awq_staged_layer(self, staged_teacher, named_childs, layer_index=layer_index)

        if fallback_named_childs:
            log.warning(
                "AWQProcessor: layer %s fallback quant %d modules: %s",
                layer_index,
                len(fallback_named_childs),
                list(fallback_named_childs)[:6],
            )
            self.apply_quant(fallback_named_childs, scales_list=[])

        self._finalize_quantized_layer(state, fallback_task_names=set(named_childs))

    @staticmethod
    def _validate_scale_group_conflicts(module, groups):
        """Reject overlapping folds before any group can modify the layer."""

        sources = set()
        targets = set()
        for group in groups:
            source = group.get("prev_op")
            source_weight = getattr(source, "weight", None)
            source_key = id(source_weight) if isinstance(source_weight, torch.Tensor) else id(source)
            if source_key in sources:
                source_name = get_op_name(module, source)
                raise _AWQScaleGroupError(f"AWQ: source `{source_name}` has multiple scaling groups; merge them first.")
            sources.add(source_key)
            group_targets = list(group.get("layers") or [])
            group_targets.extend(get_op_by_name(module, name) for name in group.get("_scale_consumers") or [])
            for target in group_targets:
                target_weight = getattr(target, "weight", None)
                target_key = id(target_weight) if isinstance(target_weight, torch.Tensor) else id(target)
                if target_key in targets:
                    target_name = get_op_name(module, target)
                    raise _AWQScaleGroupError(f"AWQ: target `{target_name}` has overlapping scaling groups or tied weights.")
                targets.add(target_key)

    @staticmethod
    def _validate_scale_group(
        module, prev_op, layers, inp, module2inspect, scale_consumers, extra_layers=None,
        allow_outside_fold=False,
    ):
        """Check exact bindings, widths and the existing replay boundary.

        ``extra_layers`` are fold-only members: they must still bind and match
        the source width, but the search does not score them, so they are not
        replayed targets. They normally also have to live inside the replay
        boundary; ``allow_outside_fold`` lifts that for a group the tree
        declares across sibling branches of one norm (Falcon-H1's attention and
        Mamba mixer). Those members read the same tensor, so the same folded
        scale is correct for them even though this replay cannot score them.

        ``allow_outside_fold`` lifts the boundary for the group's scale
        consumers too. A Cohere2-MoE group carries the MoE router as a consumer
        of the shared `input_layernorm`; the router lives in the sibling MoE
        branch, so it can never run inside the `self_attn` replay. The fold is
        still exact for it - it reads the same normed tensor - and
        ``apply_scale`` compensates every declared consumer through its
        consumer map. Only members the search scores are replayed (#4898).
        """

        try:
            source_name = get_op_name(module, prev_op)
            inspect_name = get_op_name(module, module2inspect)
            targets = [(get_op_name(module, layer), layer) for layer in layers]
            consumers = [(name, get_op_by_name(module, name)) for name in scale_consumers or ()]
        except ValueError as exc:
            raise _AWQScaleGroupError(f"AWQ: scaling group does not resolve on the live layer: {exc}") from exc
        if not targets or not isinstance(inp, torch.Tensor) or inp.ndim < 2 or inp.numel() == 0:
            raise _AWQScaleGroupError(f"AWQ: source `{source_name}` requires targets and nonempty replay features.")
        width = inp.shape[-1]
        weights = set()
        seen = {id(layer) for layer in layers}
        extra = []
        for layer in extra_layers or ():
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            extra.append((get_op_name(module, layer), layer))
        for name, target in targets + extra + consumers:
            weight = getattr(target, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[1] != width:
                raise _AWQScaleGroupError(f"AWQ: target `{name}` does not accept the {width}-channel source input.")
            if id(weight) in weights:
                raise _AWQScaleGroupError(f"AWQ: duplicate or tied target `{name}` in one scaling group.")
            weights.add(id(weight))
        source_weight = getattr(prev_op, "weight", None)
        if isinstance(source_weight, torch.Tensor) and prev_op is not layers[0]:
            if source_weight.ndim in (1, 2) and source_weight.shape[0] != width:
                raise _AWQScaleGroupError(f"AWQ: source `{source_name}` cannot produce {width} channels.")
        inspection_modules = {id(child) for child in module2inspect.modules()}
        checked = targets if allow_outside_fold else targets + extra + consumers
        outside = [name for name, target in checked if id(target) not in inspection_modules]
        if outside:
            raise _AWQScaleGroupError(
                f"AWQ: replay boundary `{inspect_name}` does not contain targets {outside} for source "
                f"`{source_name}`; a matching replay input and boundary are required."
            )
        if allow_outside_fold:
            # Consumers outside the boundary are folded, not replayed: they are
            # compensated through ``apply_scale``'s consumer map.
            consumers = [
                (name, consumer)
                for name, consumer in consumers
                if id(consumer) in inspection_modules
            ]
        return targets + consumers

    def _replay_moe_routing_override(self) -> Optional[int]:
        """Resolve the MoE top-k override that produced the captured features.

        Calibration forwards run through ``ForwardExecutor``, which applies
        ``ModuleLooper.MoERoutingOverrideContext`` when the user configured
        ``MoEConfig(routing=ExpertsRoutingOverride(...))``. AWQ then replays
        ``module2inspect`` directly instead of going through the forward
        executor. Without re-entering the same override the replay routes with
        the model's native top-k, so experts that only ran because of the
        override are missing from the reference output and the coverage guard
        rejects the complete group (#4151).

        The resolution mirrors ``ModuleLooper``: an override only applies to
        models that declare ``dynamic_expert_index`` and a scalar expert count.
        """

        gptq_model = getattr(self, "gptq_model", None)
        if gptq_model is None or not getattr(gptq_model, "dynamic_expert_index", None):
            return None
        get_num_experts = getattr(gptq_model, "get_num_experts", None)
        model_config = getattr(getattr(gptq_model, "model", None), "config", None)
        resolver = getattr(self.qcfg, "moe_routing_override", None)
        if not callable(get_num_experts) or model_config is None or not callable(resolver):
            return None
        num_experts = get_num_experts(model_config)
        if num_experts is None:
            return None
        return resolver(num_experts)

    @contextmanager
    def _moe_replay_routing(
            self,
            module: nn.Module,
            module2inspect: Optional[nn.Module] = None,
            replay_targets=(),
    ):
        """Run a replay with the same MoE routing that captured its features."""

        with ExitStack() as stack:
            override = self._replay_moe_routing_override()
            if override:
                state = set_moe_topk(module, override)
                stack.callback(restore_moe_topk, state)

            bypass = self._moe_bypass_replay(module, module2inspect)
            if bypass is not None:
                hooks, moe_block, experts_module = bypass
                stack.enter_context(
                    self._all_experts_replay_forward(
                        hooks, moe_block, experts_module, replay_targets,
                    )
                )
            yield

    def _moe_bypass_replay(self, module: nn.Module, module2inspect: Optional[nn.Module]):
        """Resolve the all-experts replay scaffolding for a bypass-mode MoE layer.

        ``MoEConfig(routing=ExpertsRoutingBypass())`` makes ``ForwardExecutor``
        enter ``ModuleLooper.MoELifecycleContext``, which feeds the block input
        to every expert before the routed forward. AWQ replays ``module2inspect``
        directly, so without re-entering that lifecycle only the natively routed
        experts execute and the coverage guard rejects the whole group (#4223).
        """

        if module2inspect is None or module2inspect is module:
            return None
        routing_bypass = getattr(self.qcfg, "moe_routing_bypass", None)
        if not callable(routing_bypass) or not routing_bypass():
            return None
        gptq_model = getattr(self, "gptq_model", None)
        if gptq_model is None:
            return None
        hooks = getattr(gptq_model, "moe_lifecycle_hooks", None)
        if hooks is None:
            return None
        try:
            moe_block = hooks.get_moe_block_for_subset(module, gptq_model)
        except Exception:
            return None
        if moe_block is None:
            return None
        # The replay boundary must contain the MoE block that owns the experts.
        if not any(child is moe_block for child in module2inspect.modules()):
            return None
        experts_module = hooks.get_experts_module(moe_block, gptq_model)
        if experts_module is None or not hasattr(experts_module, "__iter__"):
            return None
        return hooks, moe_block, experts_module

    @contextmanager
    def _all_experts_replay_forward(
            self,
            hooks,
            moe_block: nn.Module,
            experts_module,
            replay_targets,
    ):
        """Feed every replayed expert the block input, like the bypass lifecycle.

        The wrapper mirrors ``ExpertProjectionMoELifecycleHooks.forward_to_all_experts``:
        each expert in the scale group runs on the block input, then the original
        (natively routed) forward produces the output the loss compares against.
        """

        targets = {id(target) for _name, target in replay_targets}
        gate_name = getattr(hooks, "gate_proj_name", None)
        up_name = getattr(hooks, "up_proj_name", None)
        down_name = getattr(hooks, "down_proj_name", None)
        if not targets or not gate_name or not up_name:
            yield
            return

        experts = []
        for expert in experts_module:
            gate = getattr(expert, gate_name, None)
            up = getattr(expert, up_name, None)
            down = getattr(expert, down_name, None) if down_name else None
            if (
                id(gate) in targets
                or id(up) in targets
                or (down is not None and id(down) in targets)
            ):
                experts.append((expert, gate, up, down))
        if not experts:
            yield
            return

        original_forward = moe_block.forward

        # Keep the block's real signature visible. `inspect.signature` follows
        # `__wrapped__`, so the replay machinery still sees e.g.
        # `OlmoeSparseMoeBlock.forward(hidden_states)` instead of this
        # catch-all. Without it, `_forward_accepts_var_kwargs` classifies the
        # MoE block like Hymba's real `**kwargs` block and `_sanitize_kwargs` /
        # `_prepare_module_forward` hand it layer kwargs it never accepted
        # (#5095).
        @functools.wraps(original_forward)
        def all_experts_forward(hidden_states, **kwargs):
            if hidden_states.dim() == 3:
                hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            else:
                hidden_states_2d = hidden_states
            for expert, gate, up, down in experts:
                gate_ref = gate if gate is not None and id(gate) in targets else None
                up_ref = up if up is not None and id(up) in targets else None
                down_ref = down if down is not None and id(down) in targets else None
                if gate_ref is None and up_ref is None and down_ref is None:
                    continue
                anchor = gate_ref if gate_ref is not None else up_ref
                if anchor is None:
                    anchor = down_ref if down_ref is not None else expert
                expert_device = get_device(anchor)
                expert_input = move_to(hidden_states_2d, expert_device)
                gate_out = gate_ref(expert_input) if gate_ref is not None else None
                up_out = up_ref(expert_input) if up_ref is not None else None
                if down_ref is not None:
                    if gate_out is None and gate is not None:
                        gate_out = gate(expert_input)
                    if up_out is None and up is not None:
                        up_out = up(expert_input)
                    if gate_out is not None and up_out is not None:
                        intermediate = hooks.apply_expert_activation(
                            experts_module=experts_module,
                            expert=expert,
                            gate_out=gate_out,
                            up_out=up_out,
                        )
                        down_ref(intermediate)
                        del intermediate
                del gate_out, up_out, expert_input
            # The wrapper's `**kwargs` exists only to stay call-compatible with
            # the replay caller; it must not widen the original block's
            # contract. Forward just the keywords that block declares.
            return safe_kwargs_call(original_forward, hidden_states, kwargs=kwargs)

        moe_block.forward = all_experts_forward
        try:
            yield
        finally:
            moe_block.forward = original_forward

    @torch.inference_mode()
    def _search_best_scale(
            self,
            module,
            prev_op,
            layers: List[nn.Linear],
            inp: torch.Tensor,
            module2inspect=None,
            kwargs={},
            bits: Optional[int] = None,
            scale_consumers: Optional[List[str]] = None,
            fold_layers: Optional[List[nn.Linear]] = None,
            allow_partial_replay: bool = False,
            allow_outside_fold: bool = False,
    ):
        """Searches the best per-channel AWQ scale for a module group.

        Unless explicitly overridden by ``bits``, evaluate each member at its
        own configured width, even though the group shares one channel scale.
        ``scale_consumers`` names the non-quantized modules that read the same
        normed input (MoE router, ``shared_expert_gate``). Search replays the
        original input with ``Q(W * s) / s`` weights, so these consumers stay
        unchanged until the final fold compensates the scaled source.

        ``fold_layers`` names every member that must receive the folded scale
        even though only ``layers`` is scored. Starved members of a MoE gate/up
        group are folded but excluded from the search (#4675): they have no
        rows in the objective, yet the norm they read is still divided by
        ``s``, so their weights must still be multiplied by it.

        ``allow_partial_replay`` permits a member to be missing from the
        reference replay. Routed-expert groups set it because native routing
        only executes the experts it selects; a dense group keeps failing
        closed, since a member that does not run there is a broken replay.
        """

        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        # Every member that must receive the folded scale, including members the
        # search drops below. Defaults to the searched members themselves.
        fold_layers = list(fold_layers) if fold_layers is not None else list(layers)
        replay_targets = self._validate_scale_group(
            module, prev_op, layers, inp, module2inspect, scale_consumers,
            extra_layers=fold_layers, allow_outside_fold=allow_outside_fold,
        )
        kwargs = dict(kwargs)
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
        group_sizes = [
            self._group_size_for(module, layer) for layer in layers
        ]
        default_format = self._module_format(None)
        formats = [self._format_for(module, layer) for layer in layers]
        w_mean = _compute_awq_weight_mean(layers, group_sizes)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # Stream directly on the source device to avoid creating full CPU copies while still enforcing
        # a predictable memory bound derived from max_chunk_memory.
        inp_flat = inp.abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        float32_size = torch.tensor([], dtype=torch.float32).element_size()
        element_size_bytes = float32_size  # accumulation happens in FP32

        # Calculate chunk size dynamically based on the available memory budget (default 1 GiB).
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)
        chunk_size = max(chunk_size, 1)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)

        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk = inp_flat[i:end]
            # Accumulate each chunk in FP32 to balance precision and memory usage.
            chunk_sum = chunk.to(torch.float32).sum(dim=0)
            x_sum += chunk_sum

        x_mean = (x_sum / num_elements).to(inp.dtype)
        del x_sum

        # [STEP 3]: Compute output of module
        module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
        global_kwargs = getattr(self, "_module_forward_kwargs", {})
        global_allowed_kwargs = self._sanitize_kwargs(global_kwargs, module2inspect)
        for key, value in global_allowed_kwargs.items():
            module_kwargs.setdefault(key, value)
        module_kwargs = self._align_module_kwargs_to_input(inp, module_kwargs)
        replay_batch_size = self._scale_search_replay_batch_size(inp)
        prepared_forward = None

        observed = set()
        handles = []

        def record_target(target, _args):
            observed.add(id(target))

        # Both the reference and candidate replays must route like the capture
        # did. Otherwise an all-expert override captures every expert but the
        # replay executes native top-k only, and the coverage guard below
        # rejects the otherwise complete group (#4151).
        with self._moe_replay_routing(module, module2inspect, replay_targets):
            try:
                for _name, target in replay_targets:
                    handles.append(target.register_forward_pre_hook(record_target))
                if use_chunked_scale_search:
                    # Resolve the device placement and rotary scaffolding once. Both the FP
                    # reference and every candidate-scale replay reuse this prepared forward.
                    prepared_x, prepared_kwargs = self._prepare_module_forward(inp, module2inspect, module_kwargs)
                    prepared_forward = (prepared_x, prepared_kwargs)

                    fp16_output = self._build_chunked_reference_outputs(
                        prepared_x,
                        module2inspect,
                        prepared_kwargs,
                        replay_batch_size,
                    )
                else:
                    # Legacy path: run the full forward eagerly on the module device and keep
                    # the dense activation tensor for the later reconstruction-loss pass.
                    inp = inp.to(next(module2inspect.parameters()).device)
                    with ctx(torch.inference_mode()):
                        fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
                    fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)
            except RuntimeError as exc:
                # A replay whose packed geometry does not fit the module (GPT-Neo's
                # fixed causal buffer, a stale mask, a squeezed batch) raises from
                # inside the block. Report it as a scale-group failure so the
                # configured fallback applies instead of aborting `quantize()`.
                if not _is_replay_geometry_error(exc):
                    raise
                raise _AWQScaleGroupError(
                    f"AWQ: replaying {type(module2inspect).__name__} with the packed "
                    f"token rows failed: {exc}"
                ) from exc
            finally:
                for handle in handles:
                    handle.remove()
            unobserved = [name for name, target in replay_targets if id(target) not in observed]
            if unobserved:
                # Native routing only executes the experts it routes to. The
                # replay input is one captured tensor, so an expert that the
                # corpus routed to can still be unrouted in this sample and
                # never run (#4675). Score the members that did run and keep the
                # rest in the fold: their weights are still multiplied by ``s``
                # because the norm they read is divided by it.
                if not allow_partial_replay:
                    raise _AWQScaleGroupError(
                        f"AWQ: reference replay did not execute targets {unobserved}; "
                        "the scale objective would not score the complete group."
                    )
                executed = {id(target) for _name, target in replay_targets if id(target) in observed}
                executed_layers = [layer for layer in layers if id(layer) in executed]
                if not executed_layers:
                    raise _AWQScaleGroupError(
                        f"AWQ: reference replay did not execute targets {unobserved}; "
                        "the scale objective would not score the complete group."
                    )
                log.warning(
                    "AWQProcessor: reference replay did not execute %d of %d scaling-group targets "
                    "(%s); scoring only the executed members. Every member still receives the folded "
                    "scale (#4675).",
                    len(unobserved),
                    len(layers),
                    unobserved[:6],
                )
                layers = executed_layers
                group_sizes = [self._group_size_for(module, layer) for layer in layers]
                formats = [self._format_for(module, layer) for layer in layers]
                w_mean = _compute_awq_weight_mean(layers, group_sizes)

            # [STEP 4]: Compute loss
            best_scale_kwargs = {
                "replay_batch_size": replay_batch_size,
                "prepared_forward": prepared_forward,
                "bits": [self._bits_for(module, layer) for layer in layers] if bits is None else bits,
                "group_sizes": group_sizes,
                "syms": [self._sym_for(module, layer) for layer in layers],
            }
            if any(module_format != default_format for module_format in formats):
                best_scale_kwargs["formats"] = formats

            consumer_modules = None
            if scale_consumers:
                consumer_modules = [get_op_by_name(module, name) for name in scale_consumers]
            if consumer_modules:
                best_scale_kwargs["scale_consumers"] = consumer_modules

            best_scales, loss = self._compute_best_scale(
                inp,
                w_mean,
                x_mean,
                module2inspect,
                layers,
                fp16_output,
                module_kwargs,
                **best_scale_kwargs,
            )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in (fold_layers or layers)]),
            best_scales,
            loss
        )

    @torch.inference_mode()
    def _search_best_clip(self, layer, named_linears, input_feat):
        """Searches per-layer clipping thresholds for AWQ-eligible linears."""

        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any(_ in name for _ in avoid_clipping):
                continue

            features = input_feat[name]
            if features.numel() == 0:
                # No calibration sample reached this projection (a subset
                # early-stop used to starve a sibling, #4679). Clip search needs
                # at least one token row; leave the module unclipped instead of
                # reshaping an empty tensor.
                log.warn(
                    "AWQ: module `%s` received no calibration samples; skipping clip search.",
                    name,
                )
                continue

            linear = named_linears[name]
            original_device = get_device(linear)
            target_device = _resolve_module_exec_device(linear)
            move_to(linear, device=target_device)

            max_val = self._compute_best_clip(
                linear.weight,
                features,
                bits=self._bits_for(layer, linear),
                group_size=self._group_size_for(layer, linear),
                format=self._format_for(layer, linear),
                sym=self._sym_for(layer, linear),
            )
            clip_list.append((name, max_val))
            move_to(linear, device=original_device)

        return clip_list

    @torch.inference_mode()
    def _compute_best_clip(
            self,
            w: torch.Tensor,
            input_feat: torch.Tensor,
            n_grid=20,
            max_shrink=0.5,
            n_sample_token=512,
            bits: Optional[int] = None,
            group_size: Optional[int] = None,
            format: Optional[FORMAT] = None,
            sym: Optional[bool] = None,
    ):
        """Finds the clipping bound that minimizes reconstruction error for a weight tensor.

        ``bits`` is the width this module will be packed at; the clip grid must
        minimise the error of *that* quantizer, not of the global default.
        """

        if w.dim() != 2:
            raise ValueError(
                "AWQ clipping expects a 2-D weight tensor, "
                f"got shape {tuple(w.shape)}"
            )
        if input_feat.numel() == 0:
            raise ValueError(
                "AWQ clipping needs at least one calibration token row; "
                "this module received no calibration samples."
            )
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        sym = self.qcfg.sym if sym is None else bool(sym)
        configured_group_size = (
            self.qcfg.group_size if group_size is None else group_size
        )
        group_size = (
            configured_group_size
            if configured_group_size > 0
            else org_w_shape[1]
        )
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        # Clip search owns a tile size, not a second module-shape predicate. The
        # packer admits any width divisible by its own packing factor, so a
        # partial final tile must be searched instead of rejected (#4474).
        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        oc_batch_size = min(oc_batch_size, org_w_shape[0])
        w_all = w
        best_max_val_all = []
        device = w_all.device
        n_group, group_w = w_all.shape[2], w_all.shape[3]
        n_tokens = input_feat.shape[1]
        steps = int(max_shrink * n_grid)
        input_feat = input_feat.to(device)

        # Batch the shrink candidates so each grid step no longer launches its
        # own clamp/quantize/error kernel chain. Candidate order and the
        # first-minimum tie break below are identical to stepping through the
        # shrink factors one at a time.
        factors = torch.tensor(
            [1 - i_s / n_grid for i_s in range(steps)],
            dtype=torch.float32 if w_all.dtype in (torch.float16, torch.bfloat16) else w_all.dtype,
            device=device,
        )
        elem_size = torch.empty((), dtype=torch.result_type(input_feat, w_all)).element_size()
        # Three workspaces are capped independently at `budget`: the per-
        # candidate clamped+quantized pair, the full-token error input
        # `cur_out`, and each (cand, co, token, group, group_w) product slice.
        # On CPU a large transient only evicts the L3-resident working set the
        # sequential loop enjoyed, so the CPU default stays small; on GPU the
        # larger chunk amortizes launches and feeds the wide memory bus.
        mse_cfg = getattr(self.qcfg, "mse", None)
        budget = MSEConfig.resolve_budget_bytes(
            mse_cfg.clip_grid_budget if isinstance(mse_cfg, MSEConfig) else None,
            device,
        )
        # The caching allocator refuses to serve a sub-cap request from a free
        # block at or above `max_split_size`, so a workspace sized just under
        # the cap (the 256 MiB GPU default) maps one new segment per module:
        # MoE layers grew reserved VRAM by ~256 MiB per expert projection while
        # allocated memory stayed flat. Staying below the cap keeps every
        # segment splittable, so the next module reuses it. This only changes
        # chunk sizes; each candidate is still reduced over the same axes, so
        # the clip search results are bitwise unchanged.
        budget = clamp_budget_below_allocator_split(budget)
        row_elems = oc_batch_size * n_group * group_w
        out_elems = oc_batch_size * n_tokens * n_group
        cand_chunk = max(
            1,
            min(
                steps,
                budget // max(2 * row_elems * elem_size, 1),
                budget // max(out_elems * elem_size, 1),
            ),
        )
        token_chunk = max(
            1, min(n_tokens, budget // max(cand_chunk * row_elems * elem_size, 1))
        )

        out_dtype = torch.result_type(input_feat, w_all)

        def alloc_scratch(rows):
            clamp = torch.empty(
                (cand_chunk, rows, 1, n_group, group_w), dtype=w_all.dtype, device=device
            )
            return (
                clamp,
                torch.empty_like(clamp),
                torch.empty(
                    (cand_chunk, rows, n_tokens, n_group),
                    dtype=out_dtype,
                    device=device,
                ),
            )

        # One workspace set serves every full tile. Allocating inside the loop
        # left the previous tile's buffers alive through the `prod`/`clamp`/
        # `quant`/`cur_out` views while the next set was created, so the peak
        # carried two sets and every AWQ run paid +52-62% (#4682).
        full_scratch_clamp, full_scratch_quant, full_scratch_out = alloc_scratch(oc_batch_size)
        full_org_out = torch.empty_like(full_scratch_out[0])
        full_scratch_prod = torch.empty(
            (cand_chunk, oc_batch_size, token_chunk, n_group, group_w),
            dtype=full_scratch_out.dtype,
            device=device,
        )

        for i_b in range(0, org_w_shape[0], oc_batch_size):
            # The final tile may be shorter than `oc_batch_size`. It gets its
            # own exact-size set: cropping a full-size buffer to `co` rows
            # leaves a non-contiguous slice whose `reshape` is a copy, so
            # `_pseudo_quantize_tensor_into` wrote the quantized candidates
            # into that copy and the error reduction read stale rows (#4540).
            # A full tile takes the shared set, where `[:co]` is the whole
            # extent and therefore still contiguous.
            w = w_all[i_b: i_b + oc_batch_size]
            co = w.shape[0]
            if co == oc_batch_size:
                scratch_clamp = full_scratch_clamp
                scratch_quant = full_scratch_quant
                scratch_out = full_scratch_out
                org_out = full_org_out
                scratch_prod = full_scratch_prod
            else:
                scratch_clamp, scratch_quant, scratch_out = alloc_scratch(co)
                org_out = torch.empty_like(scratch_out[0])
                scratch_prod = torch.empty(
                    (cand_chunk, co, token_chunk, n_group, group_w),
                    dtype=scratch_out.dtype,
                    device=device,
                )

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # [co_batch, 1, n_group, 1]

            # Half/bfloat16 scalar multiplication computes in float32 and only
            # rounds the result. Do not round the factors to the weight dtype.
            cand_max_val = (
                org_max_val.unsqueeze(0) * factors.view(-1, 1, 1, 1, 1)
            ).to(w_all.dtype)

            # Keep the pristine bound as the initial candidate. The previous loop
            # compared against an FP16 `1e9` sentinel, which overflows to `+inf`, so
            # an all-NaN/all-infinite grid retained `org_max_val`. Stacking the
            # errors and selecting once preserves that behavior while replacing two
            # `where` launches per grid step with one `argmin` per output batch.
            err_chunks = [torch.full_like(org_max_val, float("inf")).unsqueeze(0)]
            max_val_chunks = [org_max_val.unsqueeze(0)]

            for t0 in range(0, n_tokens, token_chunk):
                t1 = min(t0 + token_chunk, n_tokens)
                prod = scratch_prod[0, :co, : t1 - t0]
                torch.mul(input_feat[:, t0:t1], w, out=prod)
                torch.sum(prod, dim=-1, out=org_out[:co, t0:t1])

            for s0 in range(0, steps, cand_chunk):
                tc = min(cand_chunk, steps - s0)
                max_val = cand_max_val[s0:s0 + tc]  # [tc, co_batch, 1, n_group, 1]
                clamp_slice = scratch_clamp[:tc, :co]
                quant_slice = scratch_quant[:tc, :co]
                torch.clamp(w.unsqueeze(0), -max_val, max_val, out=clamp_slice)
                # `clamp(w, -max_val, max_val)` bounds every group by `max_val`, so the
                # symmetric magnitude the quantizer needs is `min(|w|.amax(), max_val)`.
                # `org_max_val` is `|w|.amax()` and `max_val` is it scaled by a factor in
                # `[0.5, 1]`, so the minimum is `max_val` itself: passing it straight
                # through drops a kernel and a buffer per grid step without changing a
                # single value.
                # Flatten the candidate axis into the batch axis: the per-row
                # stats inside `_pseudo_quantize_tensor_into` then reduce over
                # the same `group_size` (or whole-row) axis as the per-step
                # calls did, and `abs_max` flattens in the same order.
                quantize_kwargs = {
                    "abs_max": max_val,
                    "bits": bits,
                    "group_size": configured_group_size,
                    "sym": sym,
                }
                if format is not None and format != self._module_format(None):
                    quantize_kwargs["format"] = format
                self._pseudo_quantize_tensor_into(
                    clamp_slice.reshape(-1, 1, n_group, group_w),
                    quant_slice.reshape(-1, 1, n_group, group_w),
                    **quantize_kwargs,
                )

                # (input_feat * quant).sum(-1) per candidate, sliced over tokens
                # so the product stays under the workspace cap. Slicing the
                # token axis does not change any element's sum over `group_w`,
                # and the mean below still reduces the same full token axis.
                cur_out = scratch_out[:tc, :co]
                for t0 in range(0, n_tokens, token_chunk):
                    t1 = min(t0 + token_chunk, n_tokens)
                    torch.mul(
                        input_feat[:, t0:t1],
                        quant_slice,
                        out=scratch_prod[:tc, :co, : t1 - t0],
                    )
                    torch.sum(
                        scratch_prod[:tc, :co, : t1 - t0],
                        dim=-1,
                        out=cur_out[:, :, t0:t1],
                    )

                # `cur_out` is scratch, so the difference and square happen in place.
                cur_out.sub_(org_out[:co].unsqueeze(0)).pow_(2)
                err_chunks.append(cur_out.mean(dim=2).view(tc, *org_max_val.shape))
                max_val_chunks.append(max_val)

            errs = torch.cat(err_chunks, dim=0)
            max_vals = torch.cat(max_val_chunks, dim=0)

            # `argmin` returns the first occurrence on ties, matching the old
            # strict-`<` update. NaN candidates were never selected before, so map
            # them to `+inf` and let the pristine-bound sentinel absorb an all-NaN
            # grid.
            best_index = errs.nan_to_num(nan=torch.inf).argmin(dim=0)
            best_max_val = torch.gather(
                max_vals,
                dim=0,
                index=best_index.unsqueeze(0),
            ).squeeze(0)
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)
        # Every output row must get a bound before the caller mutates the weight.
        if best_max_val.shape[0] != org_w_shape[0]:
            raise RuntimeError(
                "AWQ clip search covered "
                f"{best_max_val.shape[0]} of {org_w_shape[0]} output rows."
            )
        del input_feat
        del org_out

        return best_max_val.squeeze(1)

    def pseudo_quantize_tensor(
        self,
        w: torch.Tensor,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
        format: Optional[FORMAT] = None,
        sym: Optional[bool] = None,
    ):
        """Simulates AWQ quantization and returns dequantized weights plus scales/zeros.

        ``bits`` defaults to the global width. Pass the per-module width when the
        config carries a ``dynamic`` override, so the staged codes match the
        container they are packed into. ``sym`` follows the same rule: the
        per-module mode must be the one the packed layer is labelled with.
        """

        org_w_shape = w.shape
        input_dtype = w.dtype
        sym = self.qcfg.sym if sym is None else bool(sym)
        bits = int(self.qcfg.runtime_bits if bits is None else bits)
        configured_group_size = (
            self.qcfg.group_size if group_size is None else group_size
        )
        if configured_group_size > 0:
            assert org_w_shape[-1] % configured_group_size == 0, (
                f"org_w_shape ({org_w_shape[-1]}) must be a multiple of "
                f"group_size ({configured_group_size})!"
            )
            w = w.reshape(-1, configured_group_size)
        if w.dim() != 2:
            raise ValueError(
                "AWQ clipping expects a 2-D weight tensor, "
                f"got shape {tuple(w.shape)}"
            )

        if not sym:
            max_int = 2**bits - 1
            effective_format = self.qcfg.format if format is None else format
            scales, zeros, repair = _awq_affine_qparams(
                w,
                bits,
                scaled_zero_offset=effective_format in (FORMAT.GEMV_FAST, FORMAT.LLM_AWQ),
            )
            codes = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
            if repair:
                codes = _awq_bound_affine_codes(codes, zeros, scales, max_int)
            w = (codes - zeros) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (bits - 1) - 1
            min_int = -(2 ** (bits - 1))
            scales = max_val / max_int
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        # A single reduction and host read covers both guards. NaN scales and
        # NaN dequantized weights are independent failures, but each full-tensor
        # scan previously synchronized the stream separately.
        assert not bool(torch.isnan(scales).any() | torch.isnan(w).any())
        if input_dtype.is_floating_point:
            limits = torch.finfo(input_dtype)
            w.clamp_(limits.min, limits.max)

        scales = scales.view(org_w_shape[0], -1)

        # Symmetric quantization produces signed int values (e.g. int4 in [-8, 7]),
        # which cannot be packed directly. To make it packable, we shift the signed
        # representation to unsigned by adding 2^(bits-1), i.e. q_u = q_s + 2^(bits-1).
        # This is equivalent to using an affine form with zero_point = 2^(bits-1),
        # without changing the numerical behavior of symmetric quantization.
        if sym:
            zeros = torch.full_like(scales, 0 + 2 ** (bits - 1))

        w = w.reshape(org_w_shape).to(dtype=input_dtype)

        return w, scales, zeros

    @torch.inference_mode()
    def _pseudo_quantize_tensor_into(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        abs_max: torch.Tensor = None,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
        format: Optional[FORMAT] = None,
        sym: Optional[bool] = None,
    ) -> None:
        """Writes pseudo-quantized values into a destination tensor without reallocating.

        ``bits`` follows :meth:`pseudo_quantize_tensor`: the clip search must use
        the same width the module will be packed at, otherwise a 3-bit module is
        clipped against a 4-bit reconstruction error. ``sym`` follows the same
        rule for the module's symmetry mode.
        """

        # Quantize `src` into `dst` without allocating a new tensor (mirrors pseudo_quantize_tensor)
        org_shape = src.shape
        sym = self.qcfg.sym if sym is None else bool(sym)
        bits = int(self.qcfg.runtime_bits if bits is None else bits)
        configured_group_size = (
            self.qcfg.group_size if group_size is None else group_size
        )
        if configured_group_size > 0:
            src_view = src.view(-1, configured_group_size)
            dst_view = dst.view(-1, configured_group_size)
        else:
            src_view = src.reshape(org_shape[0], -1)
            dst_view = dst.reshape_as(src_view)

        if not sym:
            max_int = 2 ** bits - 1
            effective_format = self.qcfg.format if format is None else format
            scales, zeros, repair = _awq_affine_qparams(
                src_view,
                bits,
                scaled_zero_offset=effective_format in (FORMAT.GEMV_FAST, FORMAT.LLM_AWQ),
            )

            dst_view.copy_(src_view)
            dst_view.div_(scales)
            torch.round(dst_view, out=dst_view)
            dst_view.add_(zeros)
            dst_view.clamp_(0, max_int)
            if repair:
                dst_view.copy_(
                    _awq_bound_affine_codes(dst_view, zeros, scales, max_int)
                )
            dst_view.sub_(zeros)
            dst_view.mul_(scales)
        else:
            if abs_max is None:
                max_val = src_view.abs().amax(dim=1, keepdim=True)
            else:
                # The clip search clamps the block by a bound it already knows, so
                # the symmetric magnitude it needs is `min(|w|.amax(), bound)`. That
                # skips the `abs()` plus reduction pair on every grid step.
                max_val = abs_max.reshape(src_view.shape[0], 1)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (bits - 1) - 1
            scales = max_val / max_int

            torch.div(src_view, scales, out=dst_view)
            torch.round(dst_view, out=dst_view)
            dst_view.mul_(scales)
        if dst_view.dtype.is_floating_point:
            limits = torch.finfo(dst_view.dtype)
            dst_view.clamp_(limits.min, limits.max)


    def _build_chunked_reference_outputs(
        self,
        prepared_x: torch.Tensor,
        module2inspect: torch.nn.Module,
        prepared_kwargs: Dict,
        replay_batch_size: Optional[int],
    ) -> List[torch.Tensor]:
        """Build the FP reference outputs for chunked scale search.

        The chunks stay on the replay device. ``_compute_best_scale`` moves every
        one of them back to the module device for the whole 20-ratio grid, so a
        host round-trip here would only add a blocking D2H + H2D plus a duplicate
        host copy per scaling group without lowering peak device memory.
        ``_iter_prepared_forward_outputs`` still streams the replay, so the
        reference is materialized once rather than per grid step.
        """

        with ctx(torch.inference_mode()):
            return [
                output.clip(torch.finfo(output.dtype).min, torch.finfo(output.dtype).max).detach()
                for output in self._iter_prepared_forward_outputs(
                    prepared_x,
                    module2inspect,
                    prepared_kwargs,
                    batch_size=replay_batch_size,
                )
            ]


    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
        replay_batch_size: Optional[int] = None,
        prepared_forward: Optional[Tuple[torch.Tensor, Dict]] = None,
        bits: Optional[int | List[int]] = None,
        group_sizes: Optional[int | List[int]] = None,
        formats: Optional[FORMAT | List[FORMAT]] = None,
        syms: Optional[bool | List[bool]] = None,
        scale_consumers: Optional[List[nn.Module]] = None,
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X

        ``scale_consumers`` are non-quantized modules that read the same normed
        input as ``linears2scale``. They stay unchanged: replay uses the original
        input and folds the inverse scale into the quantized weights instead.
        """
        n_grid = 20
        history = []
        best_scales = None
        best_error = None
        if bits is None or isinstance(bits, int):
            layer_bits = [self.qcfg.runtime_bits if bits is None else bits] * len(linears2scale)
        else:
            layer_bits = list(bits)
        if len(layer_bits) != len(linears2scale):
            raise ValueError("AWQ scale search requires one bit width per linear")
        if group_sizes is None:
            layer_group_sizes = [self.qcfg.group_size] * len(linears2scale)
        elif isinstance(group_sizes, int):
            layer_group_sizes = [group_sizes] * len(linears2scale)
        else:
            layer_group_sizes = list(group_sizes)
        if len(layer_group_sizes) != len(linears2scale):
            raise ValueError("AWQ scale search requires one group size per linear")
        default_format = resolve_quant_format(self.qcfg.format, self.qcfg.method)
        if formats is None:
            layer_formats = [default_format] * len(linears2scale)
        elif isinstance(formats, (FORMAT, str)):
            layer_formats = [
                resolve_quant_format(formats, self.qcfg.method)
            ] * len(linears2scale)
        else:
            layer_formats = [
                resolve_quant_format(module_format, self.qcfg.method)
                for module_format in formats
            ]
        if len(layer_formats) != len(linears2scale):
            raise ValueError("AWQ scale search requires one format per linear")
        if syms is None:
            layer_syms = [bool(self.qcfg.sym)] * len(linears2scale)
        elif isinstance(syms, bool):
            layer_syms = [syms] * len(linears2scale)
        else:
            layer_syms = [bool(sym) for sym in syms]
        if len(layer_syms) != len(linears2scale):
            raise ValueError("AWQ scale search requires one symmetry mode per linear")

        # Stash a pristine master copy of every scaled linear once so the per-ratio
        # quantize/restore cycle never reaches back to host memory. The master stays on
        # the module device while the group fits the activation budget; larger groups
        # fall back to a host copy to bound peak device memory.
        master_bytes = sum(fc.weight.numel() * fc.weight.element_size() for fc in linears2scale)
        master_to_host = master_bytes > self.max_chunk_memory
        orig_weights: Dict[nn.Linear, torch.Tensor] = {}
        for fc in linears2scale:
            master = fc.weight.detach().to(dtype=fc.weight.dtype, copy=True)
            if master_to_host:
                master = master.to("cpu").contiguous()
            orig_weights[fc] = master

        def restore_weights():
            for fc in linears2scale:
                master = orig_weights[fc]
                if master.device != fc.weight.device or master.dtype != fc.weight.dtype:
                    master = master.to(device=fc.weight.device, dtype=fc.weight.dtype)
                fc.weight.copy_(master)

        def apply_scaled_weights(scales_view: torch.Tensor) -> None:
            """Restore the pristine master and apply the candidate scale in one pass.

            `restore_weights` exists because `mul_` followed by `div_` is not an
            exact identity in floating point, so every grid step has to start from
            the untouched master. Forming the product from that master directly
            keeps the guarantee and replaces the copy plus the multiply with a
            single launch per linear.
            """
            for fc in linears2scale:
                master = orig_weights[fc]
                if master.device != fc.weight.device or master.dtype != fc.weight.dtype:
                    master = master.to(device=fc.weight.device, dtype=fc.weight.dtype)
                # Members of one group can live on different devices when a
                # balanced MoE pool splits a norm-anchored group (#5104). The
                # channel scale is one tensor; each member reads it from its own
                # device.
                torch.mul(master, scales_view.to(fc.weight.device), out=fc.weight)

        try:
            device = next(module2inspect.parameters()).device
        except StopIteration:
            device = x.device
        except Exception:
            device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        prev_scale_hint = getattr(self._scale_context, "prev_scale", None)
        if prev_scale_hint is not None:
            w_mean = w_mean * float(prev_scale_hint)

        # This flag also controls how each candidate scale is scored: either stream
        # ref/int outputs chunk-by-chunk for lower memory, or reuse the legacy eager
        # tensor-vs-tensor loss computation.
        use_chunked_scale_search = getattr(self.qcfg, "scale_search_chunked_activations", True)

        # Everything except the candidate scale is invariant across the grid: the
        # module scaffolding (kwargs device placement, position ids, rotary
        # embeddings) and the FP reference activations. Resolve them once instead of
        # rebuilding them on each of the `n_grid` iterations.
        prepared_x = x
        prepared_kwargs = kwargs
        reference_chunks = None
        if use_chunked_scale_search:
            if prepared_forward is None:
                prepared_x, prepared_kwargs = self._prepare_module_forward(x, module2inspect, kwargs)
            else:
                prepared_x, prepared_kwargs = prepared_forward
            reference_chunks = [
                chunk.to(device=device)
                for chunk in self._iter_reference_output_chunks(x, fp16_output)
            ]

        try:
            for ratio in range(n_grid):
                # create new scales
                ratio = ratio / n_grid

                # NOTE: s^-1 * x is fused here, according to paper
                if self.duo_scaling:
                    scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
                else:
                    scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                scales_view = scales.view(1, -1).to(device)

                # Avoid scaling values that overflow. The in-place `nan_to_num_` replaces the
                # two masked assignments (four kernels: two predicates plus two index writes)
                # with one elementwise pass, and keeps them in place so `scales_view`, which
                # aliases `scales` whenever the scales are already on the module device, still
                # observes the fixups.
                scales.nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)

                # Q(W * s)
                # Temporarily apply the candidate scale, quantize the in-flight weights without allocating,
                # and rely on the master copy to restore the original FP values after evaluation.
                # `apply_scaled_weights` rebuilds the product from that master, so the
                # restore the previous iteration needed is folded into this step.
                apply_scaled_weights(scales_view)
                for fc, module_bits, module_group_size, module_format, module_sym in zip(
                    linears2scale, layer_bits, layer_group_sizes, layer_formats, layer_syms
                ):
                    quantize_kwargs = {
                        "bits": module_bits,
                        "group_size": module_group_size,
                        "sym": module_sym,
                    }
                    if module_format != default_format:
                        quantize_kwargs["format"] = module_format
                    self._pseudo_quantize_tensor_into(
                        fc.weight,
                        fc.weight,
                        **quantize_kwargs,
                    )
                    fc.weight.div_(scales_view.to(fc.weight.device))

                # W * X
                if use_chunked_scale_search:
                    # Compare chunked reference outputs against chunked quantized outputs
                    # so reconstruction loss is accumulated without assembling the full
                    # output activation on GPU.
                    #
                    # The running sum stays on the device in float64 and is only
                    # materialized once per ratio. Reducing each micro-batch chunk
                    # with `.item()` stalls the stream after every chunk forward and
                    # leaves the GPU idle between them. Accumulating in float64 keeps
                    # the reduction bit-identical to the previous host-side sum.
                    sq_sum = None
                    total_elements = 0
                    for ref_chunk, int_w_output in zip(
                        reference_chunks,
                        self._iter_prepared_forward_outputs(
                            prepared_x,
                            module2inspect,
                            prepared_kwargs,
                            batch_size=replay_batch_size,
                        ),
                        strict=True,
                    ):
                        int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)
                        ref_chunk = ref_chunk.to(device=int_w_output.device, dtype=int_w_output.dtype)
                        diff = (ref_chunk - int_w_output).float()
                        chunk_sq = diff.pow(2).sum().double()
                        sq_sum = chunk_sq if sq_sum is None else sq_sum + chunk_sq
                        total_elements += diff.numel()

                    if sq_sum is None:
                        loss = torch.zeros((), dtype=torch.float64, device=device)
                    else:
                        # A Python scalar denominator selects a CUDA division path
                        # that rounds one ULP away from host IEEE division. A
                        # float64 device denominator keeps the legacy loss exact.
                        loss = sq_sum / torch.tensor(
                            max(total_elements, 1),
                            dtype=torch.float64,
                            device=sq_sum.device,
                        )
                else:
                    # Legacy eager scoring path: compute one dense quantized output tensor
                    # and evaluate loss against the dense FP reference tensor.
                    int_w_output = self._module_forward(x, module2inspect, kwargs)
                    int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)
                    loss = self._compute_loss(fp16_output, int_w_output, device)

                # Keep the whole grid on the device. Comparing a 0-D loss in Python
                # forces a host synchronization after every ratio; instead, update
                # the running minimum and its scale tensor with the same strict `<`
                # predicate the host loop used, which also preserves the first
                # minimum on ties. The scalar is read once after the grid.
                history.append(loss)
                if best_error is None:
                    best_error = torch.full(
                        (),
                        float("inf"),
                        dtype=loss.dtype,
                        device=loss.device,
                    )
                    # The first candidate can alias the local scales tensor; every
                    # later replacement writes through `where` into that tensor.
                    best_scales = scales
                    improved = loss < best_error
                    torch.where(improved, loss, best_error, out=best_error)
                else:
                    improved = loss < best_error
                    torch.where(improved, loss, best_error, out=best_error)
                    torch.where(improved, scales, best_scales, out=best_scales)

        finally:
            restore_weights()
            orig_weights.clear()

        best_error_value = best_error.item()
        if best_error_value == float("inf"):
            log.debug([loss.item() for loss in history])
            module_name = type(module2inspect).__name__
            width = int(linears2scale[0].weight.shape[1]) if linears2scale else 0
            raise _AWQScaleGroupError(
                f"AWQ: scale search for {module_name} (input width {width}) produced "
                f"no finite reconstruction loss across {len(history)} candidates; "
                "every replay output was NaN or infinite."
            )

        return best_scales.detach().cpu(), best_error_value

    @torch.inference_mode()
    def _compute_loss(
            self,
            fp16_output: torch.Tensor,
            int_w_output: torch.Tensor,
            device: torch.device,
    ):
        """Computes chunked mean-squared reconstruction loss under a memory cap."""

        loss = None
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
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().double()
            loss = chunk_loss if loss is None else loss + chunk_loss

        # Normalize the loss by the total number of elements
        if loss is None:
            loss = torch.zeros((), dtype=torch.float64, device=device)
        # Match the chunked path and the old host division bit-for-bit.
        loss = loss / torch.tensor(
            num_elements,
            dtype=torch.float64,
            device=loss.device,
        )

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
    def _prepare_module_forward(
            self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ):
        """Resolves the forward scaffolding that is invariant across candidate scales.

        Device placement of ``module_kwargs``, batch promotion and the rotary
        ``position_embeddings`` depend only on the module and the captured input,
        never on the AWQ scale being evaluated. The chunked scale search resolves
        them once and replays the result for every ratio in the grid.
        """

        module_kwargs = dict(module_kwargs)
        replay_inputs = getattr(self.gptq_model, "awq_module_replay_inputs", None)
        if callable(replay_inputs):
            try:
                x, module_kwargs = replay_inputs(module, x, module_kwargs)
            except NotImplementedError as exc:
                raise _AWQScaleGroupError(f"AWQ: unsupported replay for {type(module).__name__}: {exc}") from exc
        module_kwargs = dict(module_kwargs)
        rotary_position_ids = module_kwargs.pop(_AWQ_ROTARY_POSITION_IDS, None)

        # A token-aligned kwarg must describe the rows that are actually
        # replayed. DeepSeek-V4 `hash_moe` routes with
        # `tid2eid[input_ids.reshape(-1)]`, so a stale batch's ids either index
        # the wrong experts or raise an `IndexError` deep inside the block, which
        # the scale-group fallback cannot catch (#4880).
        if x.dim() >= 3:
            token_rows = int(x.shape[0]) * int(x.shape[1])
        elif x.dim() == 2:
            token_rows = int(x.shape[0])
        elif x.dim() == 1:
            token_rows = 1
        else:
            token_rows = None
        if token_rows is not None:
            for key in _AWQ_TOKEN_ALIGNED_KWARGS:
                value = module_kwargs.get(key)
                if torch.is_tensor(value) and int(value.numel()) != token_rows:
                    raise _AWQScaleGroupError(
                        f"AWQ: replayed `{key}` describes {int(value.numel())} token rows but "
                        f"`{type(module).__name__}` received {token_rows}; the replay cannot "
                        "align a token-aligned kwarg with the captured activation."
                    )

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

        supports_position_ids = False
        supports_position_embeddings = False
        try:
            signature = inspect.signature(module.forward).parameters
            supports_position_ids = "position_ids" in signature
            supports_position_embeddings = "position_embeddings" in signature
        except (ValueError, TypeError):
            pass

        if not supports_position_ids and _forward_accepts_var_kwargs(module):
            # A `**kwargs` forward (Hymba's `HymbaBlock`) reads `position_ids`
            # out of the catch-all, so the 2-D activation still has to be
            # promoted to `[batch, seq, hidden]` for the replay (#4891).
            supports_position_ids = True

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
        preserve_position_embeddings = (
            getattr(self.gptq_model, "awq_preserve_explicit_position_embeddings", False)
            and "position_embeddings" in module_kwargs
        )
        captured_position_ids = module_kwargs.get("position_ids")
        stale_position_ids = (
            supports_position_ids
            and seq_len is not None
            and (
                not torch.is_tensor(captured_position_ids)
                or captured_position_ids.shape[-1] != seq_len
            )
        )
        stale_position_embeddings = (
            supports_position_embeddings
            and seq_len is not None
            and "position_embeddings" in module_kwargs
            and not self._position_embeddings_match(
                module_kwargs["position_embeddings"], seq_len
            )
        )
        if (
            rotary is None
            and supports_position_embeddings
            and not preserve_position_embeddings
            and (stale_position_embeddings or stale_position_ids)
        ):
            # Some families keep the rotary on the attention module
            # (Nanbeige/DeciLM expose `self_attn.rotary_emb`) instead of on the
            # model root. Use it to rebuild the position scaffolding rather
            # than replaying another sample's sequence length.
            rotary = getattr(module, "rotary_emb", None) or getattr(module, "rotary", None)
        if (
            seq_len is not None
            and rotary is not None
            and supports_position_embeddings
            and not preserve_position_embeddings
        ):
            # `_get_rotary_for_device` resolves the model-root rotary again, so
            # only use it when that is where `rotary` came from. A module-local
            # rotary has no root counterpart and must be used as-is.
            if self._get_root_rotary() is not None:
                rotary = self._get_rotary_for_device(target_device or x.device)
            rotary_device = self._get_rotary_device(rotary, target_device or x.device)
            # Attention modules that take `position_embeddings` (HF Llama,
            # Qwen3, Gemma) do not accept `position_ids`; the retained rows'
            # real positions arrive under the private key instead.
            pos_ids = module_kwargs.get("position_ids")
            if not torch.is_tensor(pos_ids) or pos_ids.shape[-1] != seq_len:
                pos_ids = rotary_position_ids
            if (
                pos_ids is None
                or not torch.is_tensor(pos_ids)
                or pos_ids.shape[-1] != seq_len
            ):
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
            rotary_kwargs = self._rotary_layer_type_kwargs(rotary, module)
            built_embeddings = self._build_position_embeddings(
                rotary,
                x_for_rotary,
                pos_for_rotary,
                rotary_kwargs,
                module_kwargs.get("position_embeddings"),
            )
            if built_embeddings is None:
                # This rotary keys its buffers by a label the attention does not
                # expose (DeepSeek V4 decouples architecture layer types from its
                # `main`/`compress` rope types). Keep captured embeddings only
                # when they describe this sample; never call the rotary with a
                # label it cannot resolve.
                if not self._position_embeddings_match(
                    module_kwargs.get("position_embeddings"), seq_len
                ):
                    module_kwargs.pop("position_embeddings", None)
            else:
                module_kwargs["position_embeddings"] = built_embeddings
        elif supports_position_ids and seq_len is not None and stale_position_ids:
            # Rebuild whenever the captured ids describe a different sample.
            # The old code only filled a *missing* key, so kwargs captured from
            # a different-length sample were replayed with this sample's
            # activations and crashed in `apply_rotary_pos_emb` (EXAONE-3.0
            # 522 vs 430; DeciLM/Nanbeige 48 vs 20).
            pos_values = torch.arange(seq_len, device=target_device or x.device, dtype=torch.long)
            if x.dim() >= 2:
                pos_values = pos_values.unsqueeze(0).expand(batch_dim, -1)
            module_kwargs["position_ids"] = pos_values

        if (
            stale_position_embeddings
            and not preserve_position_embeddings
            and not self._position_embeddings_match(
                module_kwargs.get("position_embeddings"), seq_len
            )
        ):
            # No rotary could rebuild them; drop the other sample's cos/sin so
            # the module falls back to its own rotary instead of broadcasting
            # a foreign sequence length into this replay.
            module_kwargs.pop("position_embeddings", None)

        if target_device is not None and x.device != target_device:
            x = x.to(target_device)

        return x, module_kwargs

    @staticmethod
    def _position_embeddings_match(position_embeddings, seq_len: int) -> bool:
        """Whether ``position_embeddings`` already has ``seq_len`` positions."""

        if position_embeddings is None or seq_len is None:
            return True
        if isinstance(position_embeddings, Mapping):
            values = tuple(position_embeddings.values())
        elif isinstance(position_embeddings, (tuple, list)):
            values = tuple(position_embeddings)
        else:
            values = (position_embeddings,)
        # Some families (DeepSeek V4) keep one cos/sin pair per rope type in a
        # mapping of tuples; flatten one level so the sample length is checked.
        flattened = []
        for value in values:
            if isinstance(value, Mapping):
                flattened.extend(value.values())
            elif isinstance(value, (tuple, list)):
                flattened.extend(value)
            else:
                flattened.append(value)
        lengths = {
            int(value.shape[-2])
            for value in flattened
            if torch.is_tensor(value) and value.dim() >= 2
        }
        if len(lengths) != 1:
            # Unknown layout: never drop a value we cannot inspect.
            return True
        return lengths.pop() == seq_len

    @staticmethod
    def _rotary_layer_type_parameter(rotary: nn.Module) -> Optional[str]:
        """Name of the per-layer-type selector ``rotary.forward`` accepts."""

        try:
            parameters = inspect.signature(rotary.forward).parameters
        except (TypeError, ValueError):
            return None
        for name in ("layer_type", "attention_type"):
            if name in parameters:
                return name
        return None

    @staticmethod
    def _rotary_layer_types(rotary: nn.Module) -> Tuple[str, ...]:
        """Layer-type labels a typed rotary owns, when it declares them."""

        layer_types = getattr(rotary, "layer_types", None)
        if isinstance(layer_types, (list, tuple, set, frozenset)):
            labels = tuple(value for value in layer_types if isinstance(value, str))
            if labels:
                return labels
        rope_type = getattr(rotary, "rope_type", None)
        if isinstance(rope_type, Mapping):
            return tuple(key for key in rope_type if isinstance(key, str))
        return ()

    @classmethod
    def _build_position_embeddings(
            cls,
            rotary: nn.Module,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            selector_kwargs: Optional[Dict[str, Any]],
            captured,
    ):
        """Rebuild the cos/sin scaffolding without changing its container.

        Typed rotaries take a layer-type label; a mapping-shaped
        ``position_embeddings`` (DeepSeek V4's ``{"main": ..., "compress": ...}``)
        must stay a mapping so the attention can index its own rope type. When
        the selector cannot be resolved and the container is not a mapping,
        return ``None`` and let the caller keep or drop the captured value.
        """

        parameter_name = cls._rotary_layer_type_parameter(rotary)
        if isinstance(captured, Mapping):
            keys = [key for key in captured if isinstance(key, str)]
            if keys:
                if parameter_name is not None:
                    valid_types = set(cls._rotary_layer_types(rotary))
                    if valid_types and not all(key in valid_types for key in keys):
                        # Captured keys are not labels this rotary can resolve;
                        # keep the captured mapping instead of guessing one.
                        return None
                built = {}
                for key in keys:
                    kwargs = dict(selector_kwargs or {})
                    if parameter_name is not None:
                        kwargs[parameter_name] = key
                    built[key] = rotary(x, position_ids, **kwargs)
                return built
        if selector_kwargs is None:
            return None
        return rotary(x, position_ids, **selector_kwargs)

    def _rotary_layer_type_kwargs(
            self, rotary: nn.Module, module: torch.nn.Module
    ) -> Optional[Dict[str, Any]]:
        """Return the per-layer-type selector a typed rotary needs.

        Gemma 3/4 and OLMo 3 rotary embeddings take a required ``layer_type``
        (OLMo 3 spells it ``attention_type``) so each attention type gets its
        own cos/sin pair. AWQ scales one decoder layer at a time, so the
        selector comes from the attention module being inspected. Rotaries
        without that parameter get an empty mapping and are called as before.

        DeepSeek V4 decouples its architecture layer types
        (``sliding_attention``/``compressed_sparse_attention``) from the rope
        labels it indexes (``main``/``compress``). Prefer the attention's own
        ``rope_layer_type`` selector and only pass a value the rotary declares.
        Returns ``None`` when no valid selector can be resolved.
        """

        parameter_name = self._rotary_layer_type_parameter(rotary)
        if parameter_name is None:
            return {}
        valid_types = set(self._rotary_layer_types(rotary))
        # `module` is normally the attention block, but a later AWQ group whose
        # root was already consumed (for example `o_proj`) falls back to the
        # decoder layer. Look through both shapes.
        candidates = (
            module,
            getattr(module, "self_attn", None),
            getattr(module, "attention", None),
        )
        # `rope_layer_type` is the decoupled rope label (DeepSeek V4); the
        # architecture selectors remain the fallback for Gemma 3/4 and OLMo 3.
        for attribute in ("rope_layer_type", "layer_type", "attention_type"):
            for candidate in candidates:
                if candidate is None:
                    continue
                value = getattr(candidate, attribute, None)
                if not isinstance(value, str):
                    continue
                if valid_types and value not in valid_types:
                    continue
                return {parameter_name: value}
        # Some attentions expose the selector only through the model config:
        # `LagunaAttention` keeps `is_local_attention`/`sliding_window` and the
        # `full_attention`/`sliding_attention` label lives in
        # `config.layer_types[layer_idx]` (#4874). Resolve it for the layer that
        # is being scaled, and still only pass a label the rotary declares.
        config_layer_type = self._config_layer_type()
        if config_layer_type is not None and (
                not valid_types or config_layer_type in valid_types
        ):
            return {parameter_name: config_layer_type}
        return None

    def _config_layer_type(self) -> Optional[str]:
        """``config.layer_types[layer_idx]`` for the layer being scaled."""

        scale_context = getattr(self, "_scale_context", None)
        layer_index = getattr(scale_context, "layer_index", None)
        if not isinstance(layer_index, int):
            return None
        config = getattr(getattr(self, "model", None), "config", None)
        layer_types = getattr(config, "layer_types", None)
        if not isinstance(layer_types, (list, tuple)) or not layer_types:
            return None
        if not 0 <= layer_index < len(layer_types):
            return None
        value = layer_types[layer_index]
        return value if isinstance(value, str) else None

    def _scale_search_replay_batch_size(self, x: torch.Tensor) -> Optional[int]:
        """Rows per scale-search replay forward.

        The capture batch size is tuned for activation capture and is often 1.
        Replaying one capture batch at a time turns every candidate scale into a
        series of tiny GEMMs.  Coalesce the captured batches up to the activation
        budget already used by the chunked loss so peak memory stays bounded.
        Returns ``0`` when the whole replay fits in one forward.

        A "row" is one slice of the leading dimension, so the per-row element
        count is ``x.numel() // x.shape[0]``, not ``x.shape[-1]``.  Equal-length
        captures are concatenated to ``[batches, seq, hidden]``; sizing such a
        capture by ``hidden`` alone undercounts it by ``seq`` and lets the whole
        calibration set run as one forward.  A 1-D input therefore has one
        element per row, and a 0-element input returns ``None`` (one forward over
        nothing) instead of dividing by an empty row.
        """

        if x.dim() == 0:
            return None
        rows = int(x.shape[0])
        if rows <= 1:
            return None
        numel = int(x.numel())
        if numel == 0:
            return None
        element_size = max(int(x.element_size()), 2)
        elements_per_row = numel // rows
        # Input, forward output, and the CPU reference copy are resident together.
        bytes_per_row = max(1, elements_per_row) * element_size * 3
        budget_rows = max(1, int(self.max_chunk_memory) // bytes_per_row)
        if budget_rows >= rows:
            return 0
        return max(int(self._quant_batch_size or 1), budget_rows)

    def _invoke_replay_module(
            self,
            module: torch.nn.Module,
            x: torch.Tensor,
            module_kwargs: Dict,
    ):
        """Runs one module replay and reports its failures as scale-group errors.

        A replayed block can reject the captured context with a bare
        ``AssertionError`` instead of a tensor-geometry ``RuntimeError``. Hymba's
        ``reuse_kv`` mixer asserts ``kv_last_layer is not None``: the producer's
        K/V is threaded between layers by the forward executor and is not part of
        the per-linear capture, so no replay boundary that needs it can be scored
        (#5098). Convert only a failure raised by the module itself; the
        processor's own assertions keep propagating, and the configured
        plain-quantization fallback still handles a group that cannot be scored.
        """

        try:
            return module(x, **module_kwargs)
        except AssertionError as exc:
            raise _AWQScaleGroupError(
                f"AWQ: replaying {type(module).__name__} with the packed token "
                f"rows failed: {exc!r}"
            ) from exc

    @torch.inference_mode()
    def _iter_prepared_forward_outputs(
            self,
            x: torch.Tensor,
            module: torch.nn.Module,
            module_kwargs: Dict,
            batch_size: Optional[int] = None,
    ):
        """Yields forward outputs chunk-by-chunk for an already-prepared forward.

        ``x`` and ``module_kwargs`` must originate from
        :meth:`_prepare_module_forward`, which guarantees they already live on the
        module device.
        """

        effective_quant_batch_size = self._quant_batch_size if batch_size is None else batch_size
        effective_quant_batch_size = (
            effective_quant_batch_size
            if effective_quant_batch_size and effective_quant_batch_size > 0
            else None
        )

        if (
            effective_quant_batch_size is None
            or x.dim() == 0
            or x.shape[0] <= effective_quant_batch_size
        ):
            module_output = self._invoke_replay_module(module, x, module_kwargs)
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
            if isinstance(val, Mapping):
                return {
                    key: _slice_value(item, start, length)
                    for key, item in val.items()
                }
            if isinstance(val, (list, tuple)):
                sliced = [_slice_value(item, start, length) for item in val]
                return type(val)(sliced)
            return val

        chunk_kwargs = getattr(self.gptq_model, "awq_module_replay_chunk_kwargs", None)
        batch_offset = 0
        for x_partial in torch.split(x, effective_quant_batch_size, dim=0):
            partial_kwargs = {
                key: _slice_value(value, batch_offset, x_partial.shape[0])
                for key, value in module_kwargs.items()
            }
            if callable(chunk_kwargs):
                overrides = cast(Dict[str, Any], chunk_kwargs(
                    module,
                    module_kwargs,
                    start=batch_offset,
                    length=x_partial.shape[0],
                    full_batch_size=full_batch_size,
                ))
                partial_kwargs.update(overrides)
            partial_output = self._invoke_replay_module(module, x_partial, partial_kwargs)
            if isinstance(partial_output, tuple):
                partial_output = partial_output[0]
            yield partial_output
            batch_offset += x_partial.shape[0]

    @torch.inference_mode()
    def _iter_module_forward_outputs(
            self,
            x: torch.Tensor,
            module: torch.nn.Module,
            module_kwargs: Dict,
            batch_size: Optional[int] = None,
    ):
        """Yields forward outputs chunk-by-chunk to reduce peak activation memory."""

        x, prepared_kwargs = self._prepare_module_forward(x, module, module_kwargs)
        yield from self._iter_prepared_forward_outputs(
            x, module, prepared_kwargs, batch_size=batch_size
        )

    def _iter_reference_output_chunks(
        self,
        x: torch.Tensor,
        reference_output,
    ):
        """Normalizes reference outputs to the same chunking contract as forward outputs."""

        if isinstance(reference_output, torch.Tensor):
            effective_quant_batch_size = self._quant_batch_size
            effective_quant_batch_size = (
                effective_quant_batch_size
                if effective_quant_batch_size and effective_quant_batch_size > 0
                else None
            )
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

    def apply_quant(self, named_linears: Dict[str, NamedModule], scales_list):
        """Pseudo-quantizes selected linears and stages AWQ tensors for packing."""

        start_time = time.time()
        for name, named_module in named_linears.items():
            base_title = f"Quantizing {named_module.name} in layer"
            self.draw_progress(base_title)

            linear_layer = self.resolve_quant_source_module(named_module)
            move_to(linear_layer, device=_resolve_module_exec_device(linear_layer))
            # Stage the codes at the width this module will be packed at. RCO
            # assigns widths per module through `dynamic`, so reading the global
            # `qcfg.bits` here would stage the wrong code range.
            resolved_cfg = get_resolved_module_config(named_module)
            module_bits = self._module_bits(named_module.full_name, resolved_cfg=resolved_cfg)
            module_group_size = self._module_group_size(
                named_module.full_name, resolved_cfg=resolved_cfg,
            )
            module_format = self._module_format(
                named_module.full_name, resolved_cfg=resolved_cfg,
            )

            # transformers.Conv1D (GPT-2 and friends) stores [in, out]. AWQ
            # pads and groups along the input-channel axis, so pseudo-quantize
            # in the Linear [out, in] layout, exactly like the GPTQ packer
            # (#4887).
            is_conv1d = isinstance(linear_layer, transformers.pytorch_utils.Conv1D)
            weight_for_quant = source_weight_out_in(linear_layer)

            tp_info = named_module.state.get("tp_pad_info")
            pad_cols = 0
            original_cols = weight_for_quant.shape[1]
            if isinstance(tp_info, dict):
                pad_cols = int(tp_info.get("pad_cols", 0) or 0)
                original_cols = int(tp_info.get("original_columns", original_cols))

            if pad_cols:
                pad = weight_for_quant.new_zeros(weight_for_quant.shape[0], pad_cols)
                weight_for_quant = torch.cat((weight_for_quant, pad), dim=1)

            module_sym = self._module_sym(named_module.full_name, resolved_cfg=resolved_cfg)
            wq, scales, zeros = self.pseudo_quantize_tensor(
                weight_for_quant,
                bits=module_bits,
                group_size=module_group_size,
                format=module_format,
                sym=module_sym,
            )

            from ..quantization.gsq_scalar import gsq_enabled_for, refine_affine_scalar

            gsq = getattr(self.qcfg, "gsq", None)
            if gsq_enabled_for(gsq, named_module.full_name):
                group_size = (
                    module_group_size
                    if module_group_size > 0
                    else wq.shape[1]
                )
                groups = torch.arange(wq.shape[1], device=wq.device, dtype=torch.int32) // group_size
                fitted = refine_affine_scalar(
                    wq, scales, zeros, groups, target=weight_for_quant.detach().float(),
                    bits=module_bits, config=gsq,
                    packing={FORMAT.GEMM: "awq_gemm", FORMAT.GEMV: "awq_gemv",
                             FORMAT.GEMV_FAST: "awq_gemv_fast", FORMAT.LLM_AWQ: "awq_gemv_fast"}[module_format],
                    scale_dtype=(scales.dtype if module_format == FORMAT.GEMM and
                                 scales.dtype in (torch.float16, torch.bfloat16) else torch.float16),
                )
                wq, scales, zeros = fitted.weight, fitted.scales, fitted.zeros
                named_module.state["gsq_diagnostics"] = {
                    "objective": "weight_mse", "before": fitted.before,
                    "after": fitted.after, "learn_scales": gsq.learn_scales,
                    "skipped": fitted.skipped,
                }

            if pad_cols:
                wq = wq[:, :original_cols]
                if module_group_size > 0:
                    valid_groups = max(
                        1, math.ceil(original_cols / module_group_size)
                    )
                    scales = scales[:, :valid_groups]
                    if zeros is not None:
                        zeros = zeros[:, :valid_groups]
                else:
                    scales = scales[:, :original_cols]
                    if zeros is not None:
                        zeros = zeros[:, :original_cols]

            if is_conv1d:
                # Restore the Conv1D [in, out] storage layout for the in-place
                # weight swap below and for the AWQ packers, which transpose
                # it back into the Linear layout.
                wq = wq.t().contiguous()

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
            activation_stat = self._feature_stats.get(
                named_module.name,
                {},
            )
            scale_feature = self._scale_feature_by_module.get(
                named_module.name,
                named_module.name,
            )
            scale_stat = self._feature_stats.get(
                scale_feature,
                activation_stat,
            )
            retained_samples = self._feature_nsamples_for_log(scale_stat)
            stat = {
                PROCESS_LOG_NAME: self.name(),
                PROCESS_LOG_LAYER: named_module.layer_index,
                PROCESS_LOG_MODULE: named_module.name,
                MODULE_FEATURE_COLUMN: self.module_feature_summary(named_module),
                DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(named_module),
                QUANT_LOG_LOSS: loss_summary,
                QUANT_LOG_NSAMPLES: f"{retained_samples}",
                "activation_raw_tokens": int(activation_stat.get("raw_tokens", 0)),
                "activation_retained_tokens": int(activation_stat.get("retained_tokens", 0)),
                "activation_batches": int(activation_stat.get("batches", 0)),
                "activation_aggregation": activation_stat.get("mode", "unknown"),
                "scale_feature": scale_feature,
                "scale_raw_tokens": int(scale_stat.get("raw_tokens", 0)),
                "scale_retained_tokens": int(scale_stat.get("retained_tokens", 0)),
                "scale_batches": int(scale_stat.get("batches", 0)),
                "scale_aggregation": scale_stat.get("mode", "unknown"),
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
        if _forward_accepts_var_kwargs(module):
            # The catch-all *is* the module's input list. Filtering by name
            # handed Hymba's `HymbaBlock` an empty dict, so its replay raised
            # `KeyError: 'attention_mask'` (#4891). Private keys are the
            # processor's own bookkeeping (`_awq_feature_kwargs`) and must not
            # reach the module; the retained-rows position key is the one
            # private key `_prepare_module_forward` consumes.
            return {
                k: v
                for k, v in inputs_kwargs.items()
                if not k.startswith("_") or k == _AWQ_ROTARY_POSITION_IDS
            }
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature or k == _AWQ_ROTARY_POSITION_IDS:
                sanitized_kwargs[k] = v
        return sanitized_kwargs

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Registers a module with its layer state and initializes input capture buckets."""

        # entire module is skipped
        resolved_cfg = self.qcfg.resolve_module_config(module.full_name)
        if resolved_cfg.excluded:
            return
        bind_resolved_module_config(module, resolved_cfg)

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
            # Seed a policy-requested MoE root before lifecycle forwarding.
            # This removes dependence on hook/preprocess ordering for the
            # shared pointwise input used by gate/up scale search.
            root_name = module.name.split(".", 1)[0]
            root_policy = self._feature_aggregation_policy(root_name)
            if root_policy and root_policy.get("capture_root", False):
                self.tasks.setdefault(root_name, {"inputs": [], "batch_indices": []})
            entry = self.tasks.get(module.name)
            if entry is None:
                self.tasks[module.name] = {"inputs": []}
            else:
                entry.setdefault("inputs", [])

    def prune_uncovered_module(self, module: NamedModule) -> None:
        """Remove a coverage-pruned module from its AWQ layer state.

        ``preprocess`` registered the module in ``_AWQLayerState.modules`` and
        ``pending_modules``. Left there, it kept ``pending_in_scope`` non-empty,
        so ``_quantize_layer`` never ran and every module of the layer reached
        ``pack_module`` without zero points (#4456). Dropping it from the layer
        state also keeps it out of ``_layer_input_features``: a scaling group
        that contains it loses its missing member and is skipped like any other
        group with missing features, so the dense module is never folded or
        quantized while the rest of the layer is.
        """

        with self.lock:
            super().prune_uncovered_module(module)
        layer_index = getattr(module, "layer_index", None)
        if layer_index is None:
            return
        state = self._get_layer_state(layer_index)
        with state.lock:
            state.modules.pop(module.name, None)
            state.pending_modules.discard(module.name)

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
                # ``process`` runs once per module, so every worker receives the
                # same previous-subset mapping. Recomputing its weight scale for
                # each module turned one reduction per module into O(modules^2)
                # reductions and device reads per layer. Cache by identity; the
                # processed subset is immutable once handed to the next stage.
                state.previous_weight_scale = self._previous_subset_scale(state, previous_subset)

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

        self.pack_module(module)

    def pack_module(self, module):
        """Creates the AWQ quantized module and packs saved scales/zero-points into it."""

        # generate complete, safe to move to cpu
        # cleanup all memory or states vars persistently added by this processor
        module.stream_sync()
        # NamedModule attribute access takes the parent module lock. Do not hold
        # the processor lock here, because create_quant_module/pack_module take
        # these locks in the opposite order during EoRA finalization.
        if self.calculate_w_wq_diff:
            module.weight.data = module.state.pop("wq").to(CPU)
        else:
            # The dense fake-quantized weight only feeds the diff above. Dropping
            # it here keeps the last layer's projection out of the
            # ``ModuleLooper -> processors`` cycle, where it waited for a cyclic
            # GC after ``quantize()`` returned (#4747).
            module.state.pop("wq", None)

        with self.lock:
            module.state.pop("w", None)  #
            module.state.pop("w_wq_diff", None)

            if module.state.get("q_zeros") is None:
                raise RuntimeError(
                    f"AWQProcessor: module `{module.full_name}` reached packing without "
                    "quantized zero points; its layer was finalized without quantization."
                )
            # need to clone to due to steamed pinned memory and access on diff thread
            q_zeros = module.state.pop("q_zeros").clone()
            q_scales = module.state.pop("q_scales").clone()
        assert q_zeros.device == CPU
        assert q_scales.device == CPU
        resolved_cfg = get_resolved_module_config(module)
        layers = {module.full_name: self.gptq_model.model.get_submodule(module.full_name)}
        packed_source = layers[module.full_name]
        # Resolve against the real module geometry so a kernel that cannot pack
        # this width falls back here instead of raising inside
        # `create_quant_module` after every layer was calibrated (#4567).
        quant_linear_cls = self._resolve_qlinear_kernel(
            module.full_name,
            resolved_cfg=resolved_cfg,
            in_features=int(getattr(packed_source, "in_features", 0) or 0) or None,
            out_features=int(getattr(packed_source, "out_features", 0) or 0) or None,
        )
        module_label = getattr(module, "full_name", getattr(module, "name", ""))
        parent_key = getattr(module, "full_name", getattr(module, "name", None))
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
                    bits=self._module_bits(module.full_name, resolved_cfg=resolved_cfg),
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
                    dtype=module.weight.dtype,
                    resolved_module_config=resolved_cfg,
                    install=False,
                )
        if timer is not None and create_start is not None:
            timer.record(
                "submodule_finalize_create",
                time.perf_counter() - create_start,
                source=module_label,
            )
        if qmodule is None:
            raise RuntimeError(
                f"AWQProcessor: module `{module.full_name}` resolved to an excluded "
                "quant config but still reached packing."
            )
        # pack module
        qModules = (
            {module.full_name: qmodule}
            if isinstance(qmodule, quant_linear_cls)
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
                    q_g_idx=None,
                    layers=layers,
                    quant_linear_cls=quant_linear_cls,
                    lock=self.lock,
                    quantize_config=self.qcfg,
                    device=module.state.get("quant_device"),
                )
        if timer is not None and pack_start is not None:
            timer.record(
                "submodule_finalize_pack",
                time.perf_counter() - pack_start,
                source=f"{module_label} [{packer_label or 'module.pack_original'}]",
            )

        # Publish the packed kernel with one assignment; installing it before
        # packing left an unusable projection behind when packing was
        # interrupted (issue #5089).
        with parent_module_lock(parent_key):
            install_quant_module(self.gptq_model.model, module.full_name, qmodule)

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
