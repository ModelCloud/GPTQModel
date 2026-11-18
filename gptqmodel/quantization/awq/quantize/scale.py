# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import logging
from importlib import import_module
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
from transformers.activations import GELUActivation, GELUTanh, NewGELUActivation

from gptqmodel._rocm_env import running_on_rocm
from gptqmodel.quantization.awq.modules.act import ScaledActivation
from gptqmodel.quantization.awq.utils.module import get_op_by_name, set_op_by_name
from gptqmodel.quantization.norm_tuning import resolve_gain_offset
from gptqmodel.utils.device import get_device
from gptqmodel.utils.model import move_to


log = logging.getLogger(__name__)


class UnresolvableGainOffset(RuntimeError):
    """The norm's gain parameterization could not be measured on the live module."""


# Fold sources: framework modules plus the vendored Transformers 5.23 norm and
# activation classes that definitions load (never ``transformers.models``).
_VENDORED_FOLD_NORMS = (
    ("llama", "LlamaRMSNorm"),
    ("gemma", "GemmaRMSNorm"),
    ("gemma2", "Gemma2RMSNorm"),
    ("gemma4", "Gemma4RMSNorm"),
    ("cohere", "CohereLayerNorm"),
    ("cohere2", "Cohere2LayerNorm"),
    ("cohere_compass", "CohereCompassLayerNorm"),
    ("cohere2_moe", "Cohere2MoeLayerNorm"),
)
_VENDORED_FOLD_ACTIVATIONS = (("bloom", "BloomGelu"),)


def _vendored(family, name):
    return getattr(import_module(f"gptqmodel.models.ports.native.{family}.modeling_{family}"), name)


allowed_norms = [nn.LayerNorm, *(_vendored(family, name) for family, name in _VENDORED_FOLD_NORMS)]
allowed_act_fns = [
    nn.GELU,
    NewGELUActivation,
    GELUTanh,
    GELUActivation,
    *(_vendored(family, name) for family, name in _VENDORED_FOLD_ACTIVATIONS),
]

# The identical Transformers classes (an unsupported model built by
# Transformers itself) are recognized by class path, without importing them.
_TRANSFORMERS_FOLD_NORM_PATHS = frozenset(
    f"transformers.models.{family}.modeling_{family}.{name}" for family, name in _VENDORED_FOLD_NORMS
)
_TRANSFORMERS_FOLD_ACTIVATION_PATHS = frozenset(
    f"transformers.models.{family}.modeling_{family}.{name}" for family, name in _VENDORED_FOLD_ACTIVATIONS
)


def _class_paths(module):
    return {f"{cls.__module__}.{cls.__qualname__}" for cls in type(module).__mro__}


def _is_fold_norm(module) -> bool:
    return any(isinstance(module, t) for t in allowed_norms) or bool(
        _class_paths(module) & _TRANSFORMERS_FOLD_NORM_PATHS
    )


def _is_fold_activation(module) -> bool:
    return any(isinstance(module, t) for t in allowed_act_fns) or bool(
        _class_paths(module) & _TRANSFORMERS_FOLD_ACTIVATION_PATHS
    )


def is_foldable_source(prev_op: nn.Module) -> bool:
    """Whether ``prev_op`` can absorb a consumer input scale without changing its output.

    A definition can still name a module the fold cannot use: a weightless norm
    (OLMo, HRM), a norm with an input-dependent gate (A.X-K2), a ``Conv1D``
    (GPT-2), or a HyperConnection. Those groups are skipped instead of raising
    in ``apply_scale``; the consumer then falls back to plain quantization.
    """

    if isinstance(prev_op, nn.Linear):
        return True
    if _is_fold_activation(prev_op):
        return True
    is_norm = _is_fold_norm(prev_op) or "rmsnorm" in str(
        prev_op.__class__
    ).lower()
    if not is_norm:
        return False
    weight = getattr(prev_op, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.ndim == 1


def _norm_gain_offset(ln: nn.Module) -> float:
    """Resolve whether ``ln`` applies ``weight`` or ``1 + weight``.

    The semantics are measured from the live module, not selected from a class
    list, so Gemma 3, Qwen3.5, Qwen3-Next and future offset-gain norms are
    folded correctly. When the probe cannot decide, the scale is not folded:
    source text cannot distinguish a foldable post-gain from a norm that scales
    before normalizing, and guessing produced silently corrupted layers.
    """
    try:
        return resolve_gain_offset(ln)
    except Exception as exc:  # noqa: BLE001 - report the module, not the raw error
        raise UnresolvableGainOffset(
            f"AWQ: could not resolve the gain parameterization of "
            f"{type(ln).__name__} ({id(ln)}): {exc}"
        ) from exc


@torch.inference_mode()
def _resolve_module_exec_device(module: nn.Module) -> torch.device:
    """Prefer a module's planned target device, then fall back to its current device."""

    target_device = getattr(module, "target_device", None)
    if target_device is not None:
        try:
            return torch.device(target_device)
        except (TypeError, RuntimeError, ValueError):
            pass
    return get_device(module)


def _release_rocm_layer_cache(module: nn.Module) -> None:
    """Return AWQ's freed per-layer scratch to the driver on ROCm.

    AWQ's clip search pre-allocates a ``scratch_prod`` buffer per clip-eligible
    module sized from the GPU MSE budget (256 MiB by default). PyTorch rounds
    those 255 MiB requests up to 256 MiB segments. Under a configured
    ``max_split_size_mb:256`` - the ROCm default until #5072 - the allocator
    refuses to serve any request below 256 MiB from a free block of at least
    that size (see ``get_free_block``'s "Do not return an oversized block for a
    large request"). Consecutive modules then cannot reuse each other's
    segments: Qwen3-0.6B adds four 256 MiB plus one 250 MiB segment per decoder
    layer, and the reserved cache grows by ~1 GiB per layer while active memory
    stays flat. The default no longer pins the cap, but a caller can, and this
    release bounds that case too. ``garbage_collection_threshold`` does not
    help because it is inert without a memory fraction.

    ``apply_clip`` runs once per AWQ layer, after the clip search has released
    all of those scratch tensors and before the next layer allocates its own.
    Releasing the unused cache here keeps reserved memory flat. CUDA is not
    touched, so its allocator keeps its cache and pays no extra ``cudaMalloc``.
    """

    if not running_on_rocm():
        return
    try:
        if not torch.cuda.is_available():
            return
        device = _resolve_module_exec_device(module)
        if device.type != "cuda":
            # Offloaded layers keep their weights on the host while the search
            # scratch still lives on the quantization device.
            device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - cleanup must never fail a run
        log.debug("AWQ: ROCm per-layer cache release failed", exc_info=True)


@torch.inference_mode()
def apply_clip(module, clip_list: Tuple[str, torch.Tensor]):
    # One layer boundary: the clip-search scratch is already freed here and the
    # next layer has not allocated yet. See _release_rocm_layer_cache.
    _release_rocm_layer_cache(module)
    for name, max_val in clip_list:
        layer: nn.Linear = get_op_by_name(module, name)
        original_device = get_device(layer)
        target_device = _resolve_module_exec_device(layer)
        move_to(layer, device=target_device)
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        move_to(layer, device=original_device)


@torch.inference_mode()
def scale_input_columns(consumer: nn.Module, scales: torch.Tensor) -> None:
    """Multiply the input columns of one non-quantized consumer of a scaled norm.

    ``scale_ln_fcs`` divides a norm by ``s`` and multiplies the quantized
    consumers' input columns by ``s``. Every other module that reads the same
    normed hidden state - a MoE router, ``shared_expert_gate``, or any other
    declared non-quantized linear consumer - has to receive the same ``s`` or
    its output changes. Routers are usually a small ``nn.Module`` that owns a
    plain ``weight`` of shape ``[experts, hidden]`` and applies it as a linear
    map (``Qwen3MoeTopKRouter``, ``DeepseekV2TopkRouter``, ...), so the scale is
    applied to the parameter rather than to a module class.
    """

    weight = getattr(consumer, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(
            f"AWQ: cannot absorb a norm scale into {type(consumer).__name__}: "
            "it has no 2-D `weight`"
        )
    if weight.shape[1] != scales.numel():
        raise ValueError(
            f"AWQ: cannot absorb a {scales.numel()}-wide norm scale into "
            f"{type(consumer).__name__} with weight {tuple(weight.shape)}"
        )
    weight.mul_(scales.to(device=weight.device, dtype=weight.dtype).view(1, -1))


def apply_scale(module, scales_list, input_feat_dict=None, consumer_map=None):
    """Apply every searched scale group of one layer.

    ``consumer_map`` maps a ``prev_op`` name to the layer-relative names of
    non-quantized modules that read that norm's output (MoE router,
    ``shared_expert_gate``). Those consumers must receive the product of every
    scale folded into the same norm, because each group divides the norm again.
    """

    # Sibling captures (q/k/v, gate/up) can be backed by one host tensor, so a
    # name-keyed in-place division would scale that tensor once per module.
    # Track the group that already scaled each object and divide it once per
    # group; a tensor reached by two different groups restarts from the value
    # captured before the first division.
    divided: Dict[int, int] = {}
    pristine: Dict[int, torch.Tensor] = {}
    groups_per_feature: Dict[int, Set[int]] = {}
    if input_feat_dict is not None:
        for group_index, (_prev_op_name, group_layer_names, _scales, _loss) in enumerate(scales_list):
            for layer_name in group_layer_names:
                feature = input_feat_dict.get(layer_name)
                if feature is not None:
                    groups_per_feature.setdefault(id(feature), set()).add(group_index)

    # {prev_op_name: product of the scales folded into it during this call}
    applied_norm_scales = {}
    for group_index, (prev_op_name, layer_names, scales, _) in enumerate(scales_list):
        prev_op = get_op_by_name(module, prev_op_name)
        if not is_foldable_source(prev_op):
            # A weightless norm, a gated norm, Conv1D, or a HyperConnection has
            # no plain gain to divide. Skipping keeps the layer quantized plainly
            # instead of mutating a tensor that does not represent the source.
            log.warning(
                "AWQ: skipping scale group %s -> %s: %s is not a foldable scale source",
                prev_op_name,
                layer_names,
                type(prev_op).__name__,
            )
            continue
        layers = [get_op_by_name(module, name) for name in layer_names]

        original_prev_device = get_device(prev_op)
        original_layer_devices = [get_device(layer) for layer in layers]
        target_device = _resolve_module_exec_device(layers[0] if layers else prev_op)

        move_to(prev_op, device=target_device)
        for layer in layers:
            move_to(layer, device=target_device)
        scales = scales.to(target_device)

        try:
            if (
                isinstance(prev_op, nn.Linear)
                and isinstance(layers, list)
                and isinstance(layers[0], nn.Linear)
            ):
                scale_fc_fcs(prev_op, layers, scales)

            elif isinstance(prev_op, nn.Linear):
                assert len(layers) == 1
                scale_fc_fc(prev_op, layers[0], scales)

            elif (
                _is_fold_norm(prev_op)
                or "rmsnorm" in str(prev_op.__class__).lower()
            ):
                # `scale_ln_fcs` renormalises the per-channel scale for
                # offset-gain norms (see its docstring). The captured input
                # features and the non-quantized consumers must use the same
                # effective scale as the folded weights, so take it back.
                scales = scale_ln_fcs(prev_op, layers, scales)

            elif _is_fold_activation(prev_op):
                new_module = ScaledActivation(prev_op, scales)
                set_op_by_name(module, prev_op_name, new_module)
                scale_gelu_fc(prev_op, layers[0], scales)

            else:
                raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")
        except UnresolvableGainOffset as exc:
            log.warning(
                "AWQ: skipping scale group %s -> %s: %s",
                prev_op_name,
                layer_names,
                exc,
            )
            move_to(prev_op, device=original_prev_device)
            for layer, original_device in zip(layers, original_layer_devices):
                move_to(layer, device=original_device)
            continue

        if consumer_map and prev_op_name in consumer_map:
            folded = scales.detach().to(device=target_device, dtype=torch.float32)
            previous = applied_norm_scales.get(prev_op_name)
            applied_norm_scales[prev_op_name] = folded if previous is None else previous * folded

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                # Skip the modules that are not quantized
                if layer_name in input_feat_dict:
                    inp = input_feat_dict[layer_name]
                    if inp is None or inp.numel() == 0:
                        # A starved member captured no rows: there is nothing to
                        # divide. Its weights were still multiplied by the
                        # folded scale, which is what keeps the layer equivalent
                        # to the unscaled one (#4675).
                        continue
                    previous_group = divided.get(id(inp))
                    if previous_group == group_index:
                        # A sibling in this group already applied the same scale.
                        continue
                    if previous_group is not None:
                        backup = pristine.get(id(inp))
                        if backup is not None:
                            inp.copy_(backup)
                    elif len(groups_per_feature.get(id(inp), ())) > 1:
                        # Keep the pre-division value for the second group.
                        pristine[id(inp)] = inp.clone()
                    inp.div_(scales.view(1, -1).to(inp.device))
                    divided[id(inp)] = group_index

        move_to(prev_op, device=original_prev_device)
        for layer, original_device in zip(layers, original_layer_devices):
            move_to(layer, device=original_device)

    # A norm that was divided by `s` no longer emits the tensor its non-quantized
    # consumers were trained on. Fold the accumulated scale into those consumers
    # so routing and the shared-expert gate keep their original inputs.
    for prev_op_name, total_scales in applied_norm_scales.items():
        for consumer_name in consumer_map[prev_op_name]:
            consumer = get_op_by_name(module, consumer_name)
            original_device = get_device(consumer)
            move_to(consumer, device=_resolve_module_exec_device(consumer))
            try:
                scale_input_columns(consumer, total_scales)
            finally:
                move_to(consumer, device=original_device)


@torch.inference_mode()
def scale_ln_fcs(ln: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor) -> torch.Tensor:
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    # Offset-gain norms (Gemma, Gemma 2/3, Qwen3.5, Qwen3-Next, ...) multiply
    # the output by (1 + weight) instead of weight. Resolve that semantics from
    # the live module and fold in float32 so the stored fp16/bf16 gain does not
    # absorb the rounding of the intermediate `1 + weight` value.
    gain_offset = _norm_gain_offset(ln)
    if gain_offset != 0.0:
        # A uniform factor on `s` is an exact gauge of per-group symmetric
        # quantisation: dividing every channel by the same constant and
        # multiplying the consumers by it leaves the block output unchanged.
        # Offset-gain folds must use that gauge because `(1 + w) / s - 1`
        # approaches `-1` as `s` grows, and `1 + w'` is no longer representable
        # in bf16 (Gemma-3 MLP scales reach ~1e3, where the round trip loses
        # ~26% and the "neutral" fold changes the model). Renormalising by the
        # peak keeps `max(s) <= 1`; plain-gain norms store `w / s`, which stays
        # representable and is left untouched.
        peak = scales.max()
        if peak > 1:
            scales = scales / peak
        folded = ln.weight.detach().to(torch.float32, copy=True)
        folded.add_(gain_offset).div_(scales.to(torch.float32)).sub_(gain_offset)
        ln.weight.copy_(folded.to(ln.weight.dtype))
    else:
        ln.weight.div_(scales)

    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

    return scales


@torch.inference_mode()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.inference_mode()
def scale_fc_fcs(fc1: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.inference_mode()
def scale_gelu_fc(gelu: allowed_act_fns, fc: nn.Linear, scales: torch.Tensor):
    assert _is_fold_activation(gelu)
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0
