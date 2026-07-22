# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import dis
import functools
import inspect
from contextvars import ContextVar
from weakref import ref

import torch


try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-minimal environments
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_PRISM_Q2_HIDDEN_SIZE = 2048
_PRISM_Q2_INTERMEDIATE_SIZE = 6144
_PRISM_Q2_DECODER_PARAMETERS = (
    "self",
    "hidden_states",
    "attention_mask",
    "position_ids",
    "past_key_values",
    "use_cache",
    "position_embeddings",
    "kwargs",
)
_PRISM_Q2_DECODER_NAMES = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
_PRISM_Q2_RESIDUAL_CONTEXT = ContextVar(
    "gptqmodel_prism_q2_residual_context",
    default=None,
)


if _TRITON_AVAILABLE:

    @triton.jit
    def _prism_q2_residual_gemv_kernel(
        input_ptr,
        qweight_ptr,
        scale_ptr,
        residual_ptr,
        output_ptr,
        OUTPUT_FEATURES,
        NUM_BLOCKS: tl.constexpr,
        stride_qn,
        stride_qb: tl.constexpr,
        stride_sb,
        stride_sn,
    ):
        output_index = tl.program_id(0)
        byte_offsets = tl.arange(0, 32)
        code_shifts = tl.arange(0, 4) * 2
        accumulator = tl.zeros((128,), dtype=tl.float32)

        for block_index in range(0, NUM_BLOCKS):
            packed = tl.load(
                qweight_ptr
                + output_index * stride_qn
                + block_index * stride_qb
                + 2
                + byte_offsets
            )
            codes = (packed[:, None] >> code_shifts[None, :]) & 0x03
            values = tl.reshape(codes, (128,))
            activation = tl.load(input_ptr + block_index * 128 + tl.arange(0, 128))
            scale = tl.load(scale_ptr + block_index * stride_sb + output_index * stride_sn)
            weight = (tl.cast(values, tl.float16) - 1.0) * scale
            accumulator += tl.cast(activation * weight, tl.float32)

        projection = tl.cast(tl.sum(accumulator, axis=0), tl.float16)
        residual = tl.load(
            residual_ptr + output_index,
            mask=output_index < OUTPUT_FEATURES,
            other=0.0,
        )
        result = tl.cast(projection, tl.float32) + tl.cast(residual, tl.float32)
        tl.store(
            output_ptr + output_index,
            tl.cast(result, tl.float16),
            mask=output_index < OUTPUT_FEATURES,
        )


def _launch_prism_q2_residual_projection(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    values = hidden_states.reshape(1, hidden_states.shape[-1])
    residual_values = residual.reshape(1, residual.shape[-1])
    scale = module._get_q2_native_scale(values.device)
    output = torch.empty((1, module.out_features), device=values.device, dtype=values.dtype)
    _prism_q2_residual_gemv_kernel[(module.out_features,)](
        values,
        module.qweight,
        scale,
        residual_values,
        output,
        module.out_features,
        scale.shape[0],
        module.qweight.stride(0),
        module.gguf_type_size,
        scale.stride(0),
        scale.stride(1),
        num_warps=2,
        num_stages=2,
    )
    return output.reshape(*hidden_states.shape[:-1], module.out_features)


def _supports_prism_q2_residual_projection(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> bool:
    model = module._gptqmodel_prism_q2_residual_model_ref()
    installed_device = module._gptqmodel_prism_q2_residual_device
    return (
        _TRITON_AVAILABLE
        and model is not None
        and not model.training
        and not module.training
        and hidden_states.shape == (1, 1, module.in_features)
        and residual.shape == (1, 1, module.out_features)
        and hidden_states.dtype == torch.float16
        and residual.dtype == torch.float16
        and hidden_states.device == installed_device
        and residual.device == installed_device
        and module.qweight.device == installed_device
        and hidden_states.is_contiguous()
        and residual.is_contiguous()
    )


def _prism_q2_residual_projection_forward(
    module_ref,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    module = module_ref()
    if module is None:  # pragma: no cover - the callable is owned by the module
        raise RuntimeError("Prism Q2 residual projection was released before its forward callable.")
    original = module._gptqmodel_prism_q2_residual_original_forward
    context = _PRISM_Q2_RESIDUAL_CONTEXT.get()
    if context is None or context[0] is not module:
        return original(module, hidden_states)

    _, residual, consumed = context
    consumed[0] = True
    if not _supports_prism_q2_residual_projection(module, hidden_states, residual):
        return original(module, hidden_states) + residual
    return _launch_prism_q2_residual_projection(module, hidden_states, residual)


def _call_with_residual(projection: torch.nn.Module, residual: torch.Tensor, operation):
    consumed = [False]
    token = _PRISM_Q2_RESIDUAL_CONTEXT.set((projection, residual, consumed))
    try:
        output = operation()
    finally:
        _PRISM_Q2_RESIDUAL_CONTEXT.reset(token)
    if not consumed[0]:
        if isinstance(output, tuple):
            output = (residual + output[0], *output[1:])
        else:
            output = residual + output
    return output


def _supports_prism_q2_residual_layer(layer: torch.nn.Module, hidden_states: torch.Tensor) -> bool:
    model = layer._gptqmodel_prism_q2_residual_model_ref()
    return (
        _TRITON_AVAILABLE
        and model is not None
        and not model.training
        and not layer.training
        and getattr(layer, "_gptqmodel_prism_q2_residual", False)
        and hidden_states.shape == (1, 1, _PRISM_Q2_HIDDEN_SIZE)
        and hidden_states.dtype == torch.float16
        and hidden_states.device == layer._gptqmodel_prism_q2_residual_device
        and hidden_states.is_contiguous()
    )


def _prism_q2_residual_decoder_forward(
    layer_ref,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    use_cache: bool | None = False,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    layer = layer_ref()
    if layer is None:  # pragma: no cover - the callable is owned by the layer
        raise RuntimeError("Prism Q2 residual decoder layer was released before its forward callable.")
    original = layer._gptqmodel_prism_q2_residual_original_forward
    if not _supports_prism_q2_residual_layer(layer, hidden_states):
        return original(
            layer,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states, _ = _call_with_residual(
        layer.self_attn.o_proj,
        residual,
        lambda: layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        ),
    )

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    return _call_with_residual(
        layer.mlp.down_proj,
        residual,
        lambda: layer.mlp(hidden_states),
    )


def _is_supported_decoder_forward(forward) -> bool:
    function = getattr(forward, "__func__", None)
    if function is None:
        return False
    if tuple(inspect.signature(function).parameters) != _PRISM_Q2_DECODER_PARAMETERS:
        return False
    if function.__code__.co_names != _PRISM_Q2_DECODER_NAMES:
        return False
    additions = sum(
        instruction.opname == "BINARY_ADD"
        or (instruction.opname == "BINARY_OP" and instruction.argrepr == "+")
        for instruction in dis.get_instructions(function)
    )
    return additions == 2


def _projection_is_supported(module, *, in_features: int, device: torch.device) -> bool:
    return (
        module.in_features == in_features
        and module.padded_in_features == in_features
        and module.out_features == _PRISM_Q2_HIDDEN_SIZE
        and module.gguf_tensor_qtype == "Q2_0"
        and module.bias is None
        and not module.adapter
        and module.qweight.device == device
        and callable(getattr(module, "_get_q2_native_scale", None))
        and callable(getattr(module.forward, "__func__", None))
        and not getattr(module, "_gptqmodel_prism_q2_residual", False)
    )


def install_prism_q2_residuals(model: torch.nn.Module) -> int:
    """Fuse Prism Q2 attention/MLP output projections with their decoder residual adds on sm80."""

    config = getattr(model, "config", None)
    layers = [module for module in model.modules() if module.__class__.__name__ == "Qwen3DecoderLayer"]
    if (
        not _TRITON_AVAILABLE
        or model.training
        or getattr(config, "model_type", None) != "qwen3"
        or getattr(config, "hidden_size", None) != _PRISM_Q2_HIDDEN_SIZE
        or getattr(config, "intermediate_size", None) != _PRISM_Q2_INTERMEDIATE_SIZE
        or getattr(config, "hidden_act", None) != "silu"
        or getattr(config, "num_attention_heads", None) != 16
        or getattr(config, "num_key_value_heads", None) != 8
        or getattr(config, "head_dim", 128) != 128
        or getattr(config, "num_hidden_layers", None) != len(layers)
        or not layers
    ):
        return 0

    from ..qlinear.gguf_triton import GGUFTritonKernel

    records = []
    devices = set()
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        output_projection = getattr(attention, "o_proj", None)
        down_projection = getattr(mlp, "down_proj", None)
        if (
            getattr(layer, "_gptqmodel_prism_q2_residual", False)
            or not _is_supported_decoder_forward(layer.forward)
            or attention.__class__.__name__ != "Qwen3Attention"
            or mlp.__class__.__name__ != "Qwen3MLP"
            or not getattr(attention, "_gptqmodel_prism_q2_qkv", False)
            or not getattr(mlp, "_gptqmodel_prism_q2_swiglu", False)
            or not isinstance(output_projection, GGUFTritonKernel)
            or not isinstance(down_projection, GGUFTritonKernel)
        ):
            return 0
        device = output_projection.qweight.device
        if (
            device.type != "cuda"
            or torch.cuda.get_device_capability(device) != (8, 0)
            or not _projection_is_supported(
                output_projection,
                in_features=_PRISM_Q2_HIDDEN_SIZE,
                device=device,
            )
            or not _projection_is_supported(
                down_projection,
                in_features=_PRISM_Q2_INTERMEDIATE_SIZE,
                device=device,
            )
        ):
            return 0
        records.append((layer, output_projection, down_projection, device))
        devices.add(device)

    projections = [projection for _, output, down, _ in records for projection in (output, down)]
    if len(devices) != 1 or len({id(projection) for projection in projections}) != len(projections):
        return 0

    model_reference = ref(model)
    for layer, output_projection, down_projection, device in records:
        for projection in (output_projection, down_projection):
            projection._gptqmodel_prism_q2_residual_original_forward = projection.forward.__func__
            projection._gptqmodel_prism_q2_residual_model_ref = model_reference
            projection._gptqmodel_prism_q2_residual_device = device
            projection.forward = functools.partial(_prism_q2_residual_projection_forward, ref(projection))
            projection._gptqmodel_prism_q2_residual = True

        layer._gptqmodel_prism_q2_residual_original_forward = layer.forward.__func__
        layer._gptqmodel_prism_q2_residual_model_ref = model_reference
        layer._gptqmodel_prism_q2_residual_device = device
        layer.forward = functools.partial(_prism_q2_residual_decoder_forward, ref(layer))
        layer._gptqmodel_prism_q2_residual = True
    return len(records)


__all__ = ["install_prism_q2_residuals"]
