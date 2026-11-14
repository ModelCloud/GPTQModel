# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Lossless NF12 storage for dense source modules during quantization forwards."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..nn_modules.hooked_linear import (
    STOP_FORWARD_EXCEPTION,
    HookedLinear,
    _restore_output_device,
    copy_hooked_linear_runtime,
    prepare_hooked_linear_input,
)
from ..nn_modules.qlinear.nf12 import NF12Linear
from ..quantization.nf12 import quantize_nf12_weight
from . import nf12_codec


@dataclass(frozen=True)
class SourceModuleSpec:
    kind: str
    dtype: torch.dtype
    shape: tuple[int, int]
    bias: bool = False
    padding_idx: int | None = None
    norm_type: float = 2.0
    scale_grad_by_freq: bool = False
    sparse: bool = False


def nf12_supported_device(device: torch.device | str) -> bool:
    """Check the actual accelerator, keeping NVIDIA and HIP paths distinct."""

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return False
    if torch.version.hip is not None:
        from .nf12_hip import nf12_hip_supported

        return nf12_hip_supported(target)
    return torch.cuda.get_device_capability(target)[0] >= 8


def fp16_is_bf16_exact(weight: torch.Tensor) -> bool:
    """Require an exact FP16 bit-pattern round trip, including signed zeros."""

    if weight.dtype != torch.float16:
        return False
    source = weight.detach().contiguous()
    restored = source.to(torch.bfloat16).to(torch.float16)
    return torch.equal(source.view(torch.int16), restored.view(torch.int16))


def eligible_source_module(module: nn.Module, device: torch.device | str) -> SourceModuleSpec | None:
    """Return the source contract only for matrices NF12 can faithfully store."""

    if not nf12_supported_device(device):
        return None
    if type(module) is nn.Linear:
        weight = module.weight
        kind = "linear"
    elif type(module) is HookedLinear:
        weight = module.weight
        kind = "hooked_linear"
    elif type(module) is nn.Embedding and module.max_norm is None:
        weight = module.weight
        kind = "embedding"
    else:
        return None
    if weight.is_meta or weight.ndim != 2:
        return None
    n, k = weight.shape
    if n % 16 or k % 64:
        return None
    if weight.dtype != torch.bfloat16 and not fp16_is_bf16_exact(weight):
        return None
    if kind != "embedding":
        return SourceModuleSpec(kind, weight.dtype, (n, k), module.bias is not None)
    return SourceModuleSpec(
        kind,
        weight.dtype,
        (n, k),
        padding_idx=module.padding_idx,
        norm_type=module.norm_type,
        scale_grad_by_freq=module.scale_grad_by_freq,
        sparse=module.sparse,
    )


def _rowmajor_ops(device: torch.device):
    """Resolve the row-major pack callable and native lookup namespace."""

    if device.type != "cuda":
        return None
    if torch.version.hip is not None:
        from . import nf12_hip

        if not hasattr(nf12_hip, "rowmajor_pack"):
            return None
        return nf12_hip.rowmajor_pack, "nf12_hip"
    if torch.cuda.get_device_capability(device)[0] < 8:
        return None
    import gptqmodel_nf12 as nf12

    return (
        lambda packed, escapes, base: nf12.rowmajor_pack(nf12.NF12Weight(packed, escapes, base)),
        "nf12",
    )


class AutoNF12Embedding(nn.Module):
    """Packed BF16 embedding table with a lookup-oriented row layout."""

    def __init__(self, source: nn.Embedding, *, output_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        result = quantize_nf12_weight(source.weight.detach())
        device = source.weight.device
        self.num_embeddings, self.embedding_dim = result.n, result.k
        self.output_dtype = output_dtype
        self._cast_embedding_output = output_dtype is not torch.bfloat16
        self._exp_base = int(result.exp_base)
        packed = result.packed.to(device)
        escapes = result.escapes.to(device)
        ops = _rowmajor_ops(device)
        self.row_major = ops is not None
        if self.row_major:
            row_packed, row_escapes, original, collisions = ops[0](packed, escapes, result.exp_base)
            self.original_escapes = int(original)
            self.selected_half_collisions = int(collisions)
            self.register_buffer("nf12_row_packed", row_packed)
            self.register_buffer("nf12_row_escapes", row_escapes)
            # Packing has loaded the extension, so bind the native dispatcher
            # directly and keep the Python wrapper off the forward hot path.
            self._rowmajor_embedding_op = getattr(getattr(torch.ops, ops[1]), "rowmajor_embedding")
        else:
            self._rowmajor_embedding_op = None
            self.register_buffer("nf12_packed", packed)
            self.register_buffer("nf12_escapes", escapes)
        self.register_buffer(
            "nf12_exp_base",
            torch.tensor([result.exp_base], dtype=torch.uint8, device=device),
        )
        self.eval()

    def dequantize_weight(self, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        target = torch.device(device)
        base = self._exp_base
        if self.row_major:
            ids = torch.arange(self.num_embeddings, dtype=torch.int64, device=target)
            weight = self._rowmajor_embedding_op(
                ids,
                self.nf12_row_packed.to(target),
                self.nf12_row_escapes.to(target),
                base,
            )
            return weight.to(dtype)
        return nf12_codec.decode(
            self.nf12_packed.to(target),
            self.nf12_escapes.to(target),
            base,
            self.num_embeddings,
            self.embedding_dim,
        ).to(dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self.row_major:
            output = self._rowmajor_embedding_op(
                ids,
                self.nf12_row_packed,
                self.nf12_row_escapes,
                self._exp_base,
            )
            return output.to(self.output_dtype) if self._cast_embedding_output else output
        if ids.device != self.nf12_packed.device:
            raise ValueError("AUTO_NF12 embedding indices and packed weight must share a device")
        base = self._exp_base
        if ids.is_cuda and torch.version.hip is not None:
            from . import nf12_hip

            if nf12_hip.available(ids.device):
                output = nf12_hip.embedding(ids, self.nf12_packed, self.nf12_escapes, base)
                return output if output.dtype == self.output_dtype else output.to(self.output_dtype)
        elif ids.is_cuda:
            import gptqmodel_nf12 as nf12

            try:
                weight = nf12.NF12Weight(self.nf12_packed, self.nf12_escapes, base)
                output = nf12.embedding(ids, weight)
                return output if output.dtype == self.output_dtype else output.to(self.output_dtype)
            except (RuntimeError, OSError):
                # The vendored extension is optional; the packed representation
                # stays resident even when a device cannot build its kernel.
                pass
        dense = self.dequantize_weight(device=ids.device, dtype=torch.bfloat16)
        return torch.nn.functional.embedding(ids, dense).to(self.output_dtype)


class AutoNF12HookedLinear(NF12Linear):
    """Packed linear replacement that keeps calibration-hook behavior."""

    SUPPORTS_BACKEND_SELECTION = False

    @classmethod
    def from_linear(cls, source: HookedLinear) -> "AutoNF12HookedLinear":
        n, k = source.weight.shape
        layer = cls(
            bits=12,
            in_features=k,
            out_features=n,
            bias=source.bias is not None,
            register_buffers=False,
        )
        layer.pack_original(source)
        copy_hooked_linear_runtime(
            source,
            layer,
            copy_registered_hooks=True,
        )
        return layer.eval()

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_device = input.device
        if input.device != self.nf12_packed.device:
            from .p2p import safe_move_to

            input = safe_move_to(input, self.nf12_packed.device)
        prepared = prepare_hooked_linear_input(self, input)
        output = self._forward_impl(prepared.value, apply_rotation=False)
        if prepared.replay_mode is not None:
            output = output.to(prepared.replay_output_dtype)
        hook = getattr(self, "forward_hook", None)
        if hook is not None:
            hook(self, (prepared.value,), output)
            if getattr(self, "forward_hook_last", False):
                raise STOP_FORWARD_EXCEPTION.with_traceback(None)
        return _restore_output_device(output, original_device)


# QuantLinear's validator inspects the concrete class dict, so mirror the
# inherited capability declarations and keep backend selection disabled.
for _name in dir(NF12Linear):
    if _name.startswith("SUPPORTS_") and _name != "SUPPORTS_BACKEND_SELECTION":
        setattr(AutoNF12HookedLinear, _name, getattr(NF12Linear, _name))
AutoNF12HookedLinear.SUPPORTS_BACKEND_SELECTION = False


def cast_source_module(module: nn.Module, spec: SourceModuleSpec) -> nn.Module:
    """Replace a source module with packed inference storage on its device."""

    if spec.kind == "embedding":
        return AutoNF12Embedding(module, output_dtype=spec.dtype)
    if spec.kind == "hooked_linear":
        return AutoNF12HookedLinear.from_linear(module)
    n, k = spec.shape
    replacement = NF12Linear(
        bits=12,
        in_features=k,
        out_features=n,
        bias=spec.bias,
        register_buffers=False,
    )
    replacement.pack_original(module)
    return replacement.eval()


def release_packed_source_storage(packed: nn.Module) -> None:
    """Drop packed bytes after the quantizer source has been materialized."""

    for name, tensor in tuple(packed.named_buffers(recurse=False)):
        if tensor.numel():
            setattr(packed, name, tensor.new_empty(0))
    clear_weight_cache = getattr(packed, "clear_weight_cache", None)
    if clear_weight_cache is not None:
        clear_weight_cache()


def fused_w_minus_wq(packed: nn.Module, wq: torch.Tensor) -> torch.Tensor | None:
    """Return ``decode_fp32(packed) - wq`` without materializing the source.

    EoRA needs ``w - wq`` in FP32. Decoding the packed source and subtracting in
    the same widening store means the module never holds an FP32 copy of the
    source: only ``wq`` plus the result. Returns ``None`` when the fused native
    path is unavailable so callers can fall back to the dense expression.
    """

    if not isinstance(packed, NF12Linear):
        return None
    if wq.dtype != torch.float32 or wq.ndim != 2 or not wq.is_cuda:
        return None
    packed_weight = getattr(packed, "nf12_packed", None)
    escapes = getattr(packed, "nf12_escapes", None)
    if not torch.is_tensor(packed_weight) or not torch.is_tensor(escapes):
        return None
    if packed_weight.numel() == 0:
        return None
    if packed_weight.device != wq.device:
        return None
    if tuple(wq.shape) != (packed.out_features, packed.in_features):
        return None
    try:
        exp_base = packed._exp_base()
    except Exception:  # noqa: BLE001 - unreadable metadata falls back to dense
        return None
    try:
        if torch.version.hip is not None:
            from . import nf12_hip

            if not nf12_hip.available(wq.device):
                return None
            return nf12_hip.sub_fp32(packed_weight, escapes, exp_base, wq)
        from . import nf12

        if not nf12.available():
            return None
        return nf12.sub_fp32(packed_weight, escapes, exp_base, wq)
    except Exception:  # noqa: BLE001 - optional accelerator; caller falls back
        return None


def restore_source_module(
    packed: nn.Module,
    spec: SourceModuleSpec,
    *,
    device: torch.device | str,
) -> nn.Module:
    """Materialize a dense module only when its quantizer needs the weight."""

    target = torch.device(device)
    n, k = spec.shape
    weight = packed.dequantize_weight(device=target, dtype=spec.dtype)
    if spec.kind == "embedding":
        module = nn.Embedding(
            n,
            k,
            padding_idx=spec.padding_idx,
            norm_type=spec.norm_type,
            scale_grad_by_freq=spec.scale_grad_by_freq,
            sparse=spec.sparse,
            device="meta",
            dtype=spec.dtype,
        )
    elif spec.kind == "hooked_linear":
        module = HookedLinear(k, n)
        copy_hooked_linear_runtime(
            packed,
            module,
            copy_registered_hooks=True,
        )
        module.bias = None
        if spec.bias:
            module.bias = nn.Parameter(packed.bias.detach().to(target).clone(), requires_grad=False)
    else:
        module = nn.Linear(k, n, bias=spec.bias, device="meta", dtype=spec.dtype)
        if spec.bias:
            module.bias = nn.Parameter(packed.bias.detach().to(target).clone(), requires_grad=False)
    module.weight = nn.Parameter(weight, requires_grad=False)
    return module.eval()


__all__ = [
    "AutoNF12Embedding",
    "SourceModuleSpec",
    "cast_source_module",
    "eligible_source_module",
    "fp16_is_bf16_exact",
    "fused_w_minus_wq",
    "nf12_supported_device",
    "release_packed_source_storage",
    "restore_source_module",
]
