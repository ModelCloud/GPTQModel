# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import inspect
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from packaging.version import Version
from transformers import __version__ as transformers_version
from transformers.activations import GELUActivation, NewGELUActivation


if Version(transformers_version) >= Version("4.57.0"):
    from transformers.activations import GELUTanh
else:
    from transformers.activations import PytorchGELUTanh as GELUTanh
from transformers.models.bloom.modeling_bloom import BloomGelu
from transformers.models.cohere.modeling_cohere import CohereLayerNorm
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from gptqmodel.quantization.awq.modules.act import ScaledActivation
from gptqmodel.quantization.awq.utils.module import get_op_by_name, set_op_by_name
from gptqmodel.quantization.norm_tuning import resolve_gain_offset
from gptqmodel.utils.device import get_device
from gptqmodel.utils.model import move_to


log = logging.getLogger(__name__)


try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
except Exception:  # pragma: no cover - older transformers builds do not expose Gemma 4 yet
    Gemma4RMSNorm = None

try:
    from transformers.models.cohere_compass.modeling_cohere_compass import CohereCompassLayerNorm
except Exception:  # pragma: no cover - older transformers builds do not expose Cohere Compass yet
    CohereCompassLayerNorm = None


allowed_norms = [nn.LayerNorm, LlamaRMSNorm, GemmaRMSNorm, Gemma2RMSNorm, CohereLayerNorm]
if Gemma4RMSNorm is not None:
    allowed_norms.append(Gemma4RMSNorm)
if CohereCompassLayerNorm is not None:
    allowed_norms.append(CohereCompassLayerNorm)
allowed_act_fns = [
    nn.GELU,
    BloomGelu,
    NewGELUActivation,
    GELUTanh,
    GELUActivation,
]


def _source_gain_offset(ln: nn.Module) -> Optional[float]:
    """Infer a ``1 + weight`` gain from the module's forward source.

    Only used when the live gain probe cannot run (for example a norm whose
    forward needs extra inputs). Returns ``None`` when the source is
    unavailable or does not show an additive offset.
    """
    try:
        source = inspect.getsource(type(ln).forward)
    except (OSError, TypeError):
        return None
    compact = "".join(source.split())
    if any(
        pattern in compact
        for pattern in (
            "1.0+self.weight",
            "1+self.weight",
            "self.weight+1.0",
            "self.weight+1",
        )
    ):
        return 1.0
    return None


def _norm_gain_offset(ln: nn.Module) -> float:
    """Resolve whether ``ln`` applies ``weight`` or ``1 + weight``.

    The semantics are measured from the live module, not selected from a class
    list, so Gemma 3, Qwen3.5, Qwen3-Next and future offset-gain norms are
    folded correctly. A forward-source check keeps unusual modules working when
    the probe cannot execute.
    """
    try:
        return resolve_gain_offset(ln)
    except Exception as exc:  # noqa: BLE001 - fall back to source semantics
        offset = _source_gain_offset(ln)
        if offset is not None:
            return offset
        log.warning(
            "AWQ: could not resolve the gain parameterization of %s (%s): %s; "
            "assuming a plain `weight` gain",
            type(ln).__name__,
            id(ln),
            exc,
        )
        return 0.0


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


@torch.inference_mode()
def apply_clip(module, clip_list: Tuple[str, torch.Tensor]):
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


def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales, _ in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        original_prev_device = get_device(prev_op)
        original_layer_devices = [get_device(layer) for layer in layers]
        target_device = _resolve_module_exec_device(layers[0] if layers else prev_op)

        move_to(prev_op, device=target_device)
        for layer in layers:
            move_to(layer, device=target_device)
        scales = scales.to(target_device)

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
            any(isinstance(prev_op, t) for t in allowed_norms)
            or "rmsnorm" in str(prev_op.__class__).lower()
        ):
            scale_ln_fcs(prev_op, layers, scales)

        elif any(isinstance(prev_op, t) for t in allowed_act_fns):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)

        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                # Skip the modules that are not quantized
                if layer_name in input_feat_dict:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))

        move_to(prev_op, device=original_prev_device)
        for layer, original_device in zip(layers, original_layer_devices):
            move_to(layer, device=original_device)


@torch.inference_mode()
def scale_ln_fcs(ln: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    # Offset-gain norms (Gemma, Gemma 2/3, Qwen3.5, Qwen3-Next, ...) multiply
    # the output by (1 + weight) instead of weight. Resolve that semantics from
    # the live module and fold in float32 so the stored fp16/bf16 gain does not
    # absorb the rounding of the intermediate `1 + weight` value.
    gain_offset = _norm_gain_offset(ln)
    if gain_offset != 0.0:
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
    assert any(isinstance(gelu, t) for t in allowed_act_fns)
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0
