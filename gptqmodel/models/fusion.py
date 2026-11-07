# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Post-quantization projection fusion, selected exclusively from ModulePlan.

The original registered leaves remain the checkpoint owners. A parent invocation
scopes the split outputs; only concatenated *packed* operands may be cached.
"""

from __future__ import annotations

import copy
import math
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

from ..nn_modules.qlinear._runtime_cache import tensor_cache_key
from ..nn_modules.qlinear.torch import TorchLinear
from ..nn_modules.qlinear.torch_awq import AwqTorchLinear


_FAMILIES = {"qkv": ("q", "k", "v"), "gate_up": ("gate", "up")}
_FIELDS = (
    "bits", "requested_group_size", "group_size", "sym", "desc_act", "pack_dtype",
    "backend", "format", "_qzeros_format", "compute_dtype", "in_features",
)


@dataclass(frozen=True)
class FusionDecision:
    family: str
    parent: str
    paths: tuple[str, ...]
    reason: str = ""

    @property
    def enabled(self):
        return not self.reason


def _effective_config(config, path):
    values = copy.deepcopy(config.to_dict())
    values.pop("dynamic", None)
    override = config.dynamic_get(path)
    if override is False:
        return None

    def merge(base, patch):
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    merge(values, override or {})
    return values


def _incompatible(modules):
    first = modules[0]
    if type(first) not in (TorchLinear, AwqTorchLinear) or any(type(m) is not type(first) for m in modules):
        return "executor requires identical Torch GPTQ or Torch AWQ kernels"
    if first.bits not in (2, 4, 8):
        return "executor requires 2, 4 or 8-bit word-aligned storage"
    if first.pack_dtype != torch.int32:
        return "executor requires int32 packed words"
    if type(first) is AwqTorchLinear and len(modules) == 3 and len({m.out_features for m in modules}) > 1:
        # The packed weights concatenate exactly, but changing this GEMM's N
        # changes CUDA reduction rounding. Real Llama 3.2 1B exceeded the logits
        # drift gate; keep its separate QKV calls until a validated executor exists.
        return "AWQ GQA QKV fusion requires a numerically validated executor"
    signature = tuple(getattr(first, field, None) for field in _FIELDS)
    for module in modules:
        if signature != tuple(getattr(module, field, None) for field in _FIELDS):
            return "quantized kernel settings differ"
        if module.adapter is not None or module.online_full_had or module.online_partial_had:
            return "adapters and online rotation are not supported by this executor"
        if module._forward_hooks or module._forward_pre_hooks or hasattr(module, "_hf_hook"):
            return "projection has runtime hooks"
        if module.out_features * module.bits % module.pack_dtype_bits:
            return "projection output is not packed-word aligned"
        for name in ("qweight", "qzeros", "scales"):
            tensor = getattr(module, name, None)
            reference = getattr(first, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
                return "quantized operands are not materialized"
            if tensor.device != reference.device or tensor.dtype != reference.dtype or tensor.device != module.qweight.device:
                return "operand devices or dtypes differ"
            if tensor.requires_grad:
                return "trainable quantized operands"
        if module.scales.dtype not in module.SUPPORTS_DTYPES:
            return "operand dtype is unsupported by the executor"
        columns = module.out_features * module.bits // module.pack_dtype_bits
        rows = math.ceil(module.in_features / module.group_size)
        weight_shape = ((math.ceil(module.in_features * module.bits / module.pack_dtype_bits), module.out_features)
                        if isinstance(first, TorchLinear) else (module.in_features, columns))
        if (tuple(module.qweight.shape) != weight_shape or tuple(module.qzeros.shape) != (rows, columns)
                or tuple(module.scales.shape) != (rows, module.out_features)):
            return "packed operand shapes require padding or a different layout"
        if (module.bias is None) != (first.bias is None):
            return "bias layouts differ"
        if module.bias is not None and (
            module.bias.device != first.scales.device or module.bias.dtype != first.scales.dtype
            or module.bias.requires_grad
        ):
            return "bias device, dtype or gradient policy differs"
        if isinstance(first, TorchLinear) and (
            not isinstance(module.g_idx, torch.Tensor)
            or module.g_idx.device.type == "meta"
            or module.g_idx.shape != first.g_idx.shape
            or module.g_idx.numel() != module.in_features
            or not torch.equal(module.g_idx, first.g_idx)
        ):
            return "activation-order group indices differ"
    if len({id(m) for m in modules}) != len(modules):
        return "projections alias the same module"
    return ""


class _FusionGroup:
    def __init__(self, decision, parent, modules):
        self.decision = decision
        self.parent = parent
        self.modules = modules
        self.originals = [m.forward for m in modules]
        self.original_clears = [m.clear_weight_cache for m in modules]
        self.parent_forward = parent.forward
        self.context = ContextVar(f"fusion_{decision.parent}", default=None)
        self.packed = None
        self.packed_key = None
        self.calls = 0

    def _key(self):
        return (id(getattr(self.parent, "_hf_hook", None)), tuple(
            (tuple(getattr(m, field, None) for field in _FIELDS), m.training,
             id(m.adapter), m.online_full_had, m.online_partial_had,
             tuple(m._forward_hooks), tuple(m._forward_pre_hooks), id(getattr(m, "_hf_hook", None)),
             tuple(tensor_cache_key(getattr(m, name, None)) for name in ("qweight", "qzeros", "scales", "bias", "g_idx")))
            for m in self.modules
        ))

    def _operand(self, key):
        # Inference tensors use the kernels' explicit clear_weight_cache contract
        # for writes without version counters (installed on every source below).
        if self.packed is not None and key == self.packed_key:
            return self.packed
        first = self.modules[0]
        kwargs = {
            "bits": first.bits, "group_size": first.requested_group_size, "sym": first.sym,
            "desc_act": first.desc_act, "in_features": first.in_features,
            "out_features": sum(m.out_features for m in self.modules), "bias": first.bias is not None,
            "pack_dtype": first.pack_dtype, "register_buffers": False, "dtype": first.scales.dtype,
            "backend": first.backend,
        }
        if isinstance(first, TorchLinear):
            kwargs["format"] = first.format
        fused = type(first)(**kwargs)
        for name in ("qweight", "qzeros", "scales"):
            fused.register_buffer(name, torch.cat([getattr(m, name) for m in self.modules], dim=1))
        fused.register_buffer("bias", None if first.bias is None else torch.cat([m.bias for m in self.modules]))
        if isinstance(first, TorchLinear):
            fused.register_buffer("g_idx", first.g_idx)
            fused._qzeros_format = first._qzeros_format
            fused._init_wf_unsqueeze_buffers()
        fused.training = False
        self.packed, self.packed_key = fused, key
        return fused

    def install(self):
        @wraps(self.parent_forward)
        def parent_forward(*args, **kwargs):
            token = self.context.set({})
            try:
                return self.parent_forward(*args, **kwargs)
            finally:
                self.context.reset(token)

        self.parent.forward = parent_forward
        for index, (module, original, clear) in enumerate(zip(self.modules, self.originals, self.original_clears)):
            def clear_weight_cache(_clear=clear):
                self.packed = self.packed_key = None
                state = self.context.get()
                if state is not None:
                    state.clear()
                _clear()

            module.clear_weight_cache = clear_weight_cache
            def forward(x, _index=index, _original=original):
                state = self.context.get()
                if state is None or self.parent.training or x.requires_grad or torch.compiler.is_compiling():
                    return _original(x)
                if (torch.nn.modules.module._global_forward_pre_hooks
                        or torch.nn.modules.module._global_forward_hooks):
                    return _original(x)
                key = self._key()
                if state.get("key") != key:
                    state.clear()
                if "outputs" not in state:
                    if (any(m.training for m in self.modules) or hasattr(self.parent, "_hf_hook")
                            or ((self.packed is None or key != self.packed_key) and _incompatible(self.modules))):
                        self.packed = self.packed_key = None
                        return _original(x)
                    operand = self._operand(key)
                    output = operand(x)
                    state.update(input=x, version=None if x.is_inference() else x._version, key=key,
                                 outputs=output.split([m.out_features for m in self.modules], dim=-1))
                    self.calls += 1
                version = None if x.is_inference() else x._version
                if state["input"] is not x or state["version"] != version:
                    return _original(x)
                return state["outputs"][_index]
            module.forward = forward

    def remove(self):
        self.parent.forward = self.parent_forward
        for module, original, clear in zip(self.modules, self.originals, self.original_clears):
            module.forward = original
            module.clear_weight_cache = clear
        self.packed = self.packed_key = None


class FusionPlanner:
    """Inspectable decisions and optional execution for a quantized wrapper."""

    def __init__(self, model, overrides=None):
        self.decisions = []
        self.groups = []
        if not model.quantized:
            raise RuntimeError("Fusion requires a post-quantized model")
        if not isinstance(model.model, torch.nn.Module):
            return
        config = model.model.config
        if getattr(config, "model_type", None) not in model.fusion_model_types:
            return
        policy = {**model.fusion_overrides, **(overrides or {})}
        if any(type(value) is not bool for value in policy.values()):
            raise TypeError("Fusion overrides must map family or parent paths to bool")
        plan = model.module_plan()
        for prefix in model.extract_layers_node():
            layers = model.model.get_submodule(prefix)
            for index, layer in enumerate(layers):
                for parent in dict.fromkeys(spec.parent for spec in plan.specs if spec.roles):
                    if not parent or plan.is_moe(parent) or "{expert_index}" in parent:
                        continue
                    for family, roles in _FAMILIES.items():
                        candidates = [plan.role_paths_in_parent(parent, role) for role in roles]
                        if not any(candidates):
                            continue
                        full_parent = f"{prefix}.{index}.{parent}"
                        paths = tuple(paths[0] for paths in candidates if len(paths) == 1)
                        reason = ""
                        if not policy.get(full_parent, policy.get(parent, policy.get(family, True))):
                            reason = "disabled by fusion override"
                        elif any(len(paths) != 1 for paths in candidates) or len(set(paths)) != len(roles):
                            reason = "requires exactly one distinct declared projection per role"
                        elif any(not plan.find(path).quantize for path in paths):
                            reason = "projection is excluded or capture-only"
                        try:
                            owner = layer.get_submodule(parent)
                            modules = [layer.get_submodule(path) for path in paths]
                        except AttributeError:
                            reason = reason or "declared projections are absent on this layer"
                            modules = []
                        if not reason:
                            configs = [_effective_config(model.quantize_config, f"{prefix}.{index}.{p}") for p in paths]
                            if configs[0] is None or any(c != configs[0] for c in configs[1:]):
                                reason = "effective quantization configurations differ or exclude a projection"
                            elif type(owner) not in (LlamaAttention, LlamaMLP):
                                reason = "parent forward is outside the validated Llama executor"
                            elif hasattr(owner, "_hf_hook"):
                                reason = "parent has offload hooks"
                            else:
                                reason = _incompatible(modules)
                        decision = FusionDecision(family, full_parent, paths, reason)
                        self.decisions.append(decision)
                        if decision.enabled:
                            self.groups.append(_FusionGroup(decision, owner, modules))

    def apply(self):
        for group in self.groups:
            group.install()
        return self

    def remove(self):
        for group in reversed(self.groups):
            group.remove()

    def to_dict(self):
        return {"groups": [{"family": d.family, "parent": d.parent, "paths": list(d.paths),
                                "enabled": d.enabled, "reason": d.reason} for d in self.decisions],
                "fused_calls": sum(g.calls for g in self.groups)}

    def render(self):
        lines = [f"FusionPlanner: {len(self.groups)} eligible / {len(self.decisions)} declared groups"]
        for decision in self.decisions:
            status = "eligible" if decision.enabled else f"skip: {decision.reason}"
            lines.append(f"  {decision.parent} [{decision.family}] [{status}]")
        return "\n".join(lines)

    def print(self, *, file=None):
        print(self.render(), file=file)
