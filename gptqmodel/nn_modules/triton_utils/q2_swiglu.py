# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from types import MethodType

import torch


try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-minimal environments
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_PRISM_Q2_SWIGLU_IN_FEATURES = 2048
_PRISM_Q2_SWIGLU_OUT_FEATURES = 6144


if _TRITON_AVAILABLE:

    @triton.jit
    def _prism_q2_swiglu_gemv_kernel(
        input_ptr,
        gate_qweight_ptr,
        gate_scale_ptr,
        up_qweight_ptr,
        up_scale_ptr,
        output_ptr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
        gate_stride_qn,
        gate_stride_sb,
        gate_stride_sn,
        up_stride_qn,
        up_stride_sb,
        up_stride_sn,
    ):
        output_index = tl.program_id(0)
        block_offsets = tl.arange(0, 128)
        byte_offsets = tl.arange(0, 32)
        code_shifts = tl.arange(0, 4) * 2
        gate_accumulator = tl.zeros((128,), dtype=tl.float32)
        up_accumulator = tl.zeros((128,), dtype=tl.float32)

        for block_index in range(0, NUM_BLOCKS):
            activation = tl.load(input_ptr + block_index * 128 + block_offsets)

            gate_packed = tl.load(
                gate_qweight_ptr
                + output_index * gate_stride_qn
                + block_index * BLOCK_BYTES
                + 2
                + byte_offsets
            )
            gate_codes = (gate_packed[:, None] >> code_shifts[None, :]) & 0x03
            gate_values = tl.reshape(gate_codes, (128,))
            gate_scale = tl.load(
                gate_scale_ptr + block_index * gate_stride_sb + output_index * gate_stride_sn
            )
            gate_weight = (tl.cast(gate_values, tl.float16) - 1.0) * gate_scale
            gate_accumulator += tl.cast(activation * gate_weight, tl.float32)

            up_packed = tl.load(
                up_qweight_ptr
                + output_index * up_stride_qn
                + block_index * BLOCK_BYTES
                + 2
                + byte_offsets
            )
            up_codes = (up_packed[:, None] >> code_shifts[None, :]) & 0x03
            up_values = tl.reshape(up_codes, (128,))
            up_scale = tl.load(up_scale_ptr + block_index * up_stride_sb + output_index * up_stride_sn)
            up_weight = (tl.cast(up_values, tl.float16) - 1.0) * up_scale
            up_accumulator += tl.cast(activation * up_weight, tl.float32)

        gate = tl.cast(tl.sum(gate_accumulator, axis=0), tl.float16)
        up = tl.cast(tl.sum(up_accumulator, axis=0), tl.float16)
        activated_gate = tl.cast(
            tl.cast(gate, tl.float32) * tl.sigmoid(tl.cast(gate, tl.float32)),
            tl.float16,
        )
        tl.store(output_ptr + output_index, activated_gate * up)


def _launch_prism_q2_swiglu(
    hidden_states: torch.Tensor,
    gate_proj: torch.nn.Module,
    up_proj: torch.nn.Module,
) -> torch.Tensor:
    values = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    gate_scale = gate_proj._get_q2_native_scale(values.device)
    up_scale = up_proj._get_q2_native_scale(values.device)
    output = torch.empty((1, gate_proj.out_features), device=values.device, dtype=values.dtype)
    _prism_q2_swiglu_gemv_kernel[(gate_proj.out_features,)](
        values,
        gate_proj.qweight,
        gate_scale,
        up_proj.qweight,
        up_scale,
        output,
        NUM_BLOCKS=gate_scale.shape[0],
        BLOCK_BYTES=gate_proj.gguf_type_size,
        gate_stride_qn=gate_proj.qweight.stride(0),
        gate_stride_sb=gate_scale.stride(0),
        gate_stride_sn=gate_scale.stride(1),
        up_stride_qn=up_proj.qweight.stride(0),
        up_stride_sb=up_scale.stride(0),
        up_stride_sn=up_scale.stride(1),
        num_warps=1,
        num_stages=1,
    )
    return output.reshape(*hidden_states.shape[:-1], gate_proj.out_features)


def _prism_q2_swiglu_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    gate_proj = module.gate_proj
    up_proj = module.up_proj
    installed_device = module._gptqmodel_prism_q2_swiglu_device
    rows = hidden_states.numel() // hidden_states.shape[-1]
    if (
        not _TRITON_AVAILABLE
        or module.training
        or hidden_states.dtype != torch.float16
        or hidden_states.device != installed_device
        or hidden_states.shape[-1] != _PRISM_Q2_SWIGLU_IN_FEATURES
        or rows != 1
        or gate_proj.qweight.device != installed_device
        or up_proj.qweight.device != installed_device
    ):
        return module._gptqmodel_prism_q2_swiglu_original_forward(hidden_states)

    intermediate = _launch_prism_q2_swiglu(hidden_states, gate_proj, up_proj)
    return module.down_proj(intermediate)


def install_prism_q2_swiglu(model: torch.nn.Module) -> int:
    """Fuse the profiled Qwen3 Q2 gate/up GEMVs and SwiGLU for batch-one sm80 decode."""

    config = getattr(model, "config", None)
    if (
        not _TRITON_AVAILABLE
        or getattr(config, "model_type", None) != "qwen3"
        or getattr(config, "hidden_act", None) != "silu"
    ):
        return 0

    from ..qlinear.gguf_triton import GGUFTritonKernel

    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3MLP" or getattr(
            module, "_gptqmodel_prism_q2_swiglu", False
        ):
            continue

        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)
        down_proj = getattr(module, "down_proj", None)
        if not isinstance(gate_proj, GGUFTritonKernel) or not isinstance(up_proj, GGUFTritonKernel):
            continue
        if down_proj is None or gate_proj.gguf_tensor_qtype != "Q2_0" or up_proj.gguf_tensor_qtype != "Q2_0":
            continue
        if gate_proj.bias is not None or up_proj.bias is not None or gate_proj.adapter or up_proj.adapter:
            continue
        if (
            gate_proj.in_features,
            gate_proj.out_features,
            up_proj.in_features,
            up_proj.out_features,
        ) != (
            _PRISM_Q2_SWIGLU_IN_FEATURES,
            _PRISM_Q2_SWIGLU_OUT_FEATURES,
            _PRISM_Q2_SWIGLU_IN_FEATURES,
            _PRISM_Q2_SWIGLU_OUT_FEATURES,
        ):
            continue

        device = gate_proj.qweight.device
        if device.type != "cuda" or up_proj.qweight.device != device:
            continue
        if torch.cuda.get_device_capability(device) != (8, 0):
            continue

        module._gptqmodel_prism_q2_swiglu = True
        module._gptqmodel_prism_q2_swiglu_device = device
        module._gptqmodel_prism_q2_swiglu_original_forward = module.forward
        module.forward = MethodType(_prism_q2_swiglu_forward, module)
        installed += 1

    return installed


__all__ = ["install_prism_q2_swiglu"]
