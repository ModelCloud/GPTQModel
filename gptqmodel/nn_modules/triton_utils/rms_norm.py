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


_PRISM_Q2_RMS_NORM_WIDTHS = {128, 2048}


def _select_prism_q2_rms_norm_num_warps(*, dtype: torch.dtype, n_cols: int, rows: int) -> int:
    if dtype != torch.float16:
        return 4
    if (n_cols, rows) == (2048, 1):
        return 8
    if n_cols == 128 and rows in {8, 16}:
        return 1
    return 4


if _TRITON_AVAILABLE:

    @triton.jit
    def _prism_q2_rms_norm_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        values = tl.load(input_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / n_cols
        normalized = values * tl.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_ptr + row * n_cols + offsets, normalized * weight, mask=mask)


def _torch_rms_norm(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    values = hidden_states.to(torch.float32)
    variance = values.pow(2).mean(-1, keepdim=True)
    values = values * torch.rsqrt(variance + module.variance_epsilon)
    return module.weight * values.to(input_dtype)


def _prism_q2_rms_norm_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    installed_device = getattr(module, "_gptqmodel_prism_q2_rms_norm_device", None)
    if (
        not _TRITON_AVAILABLE
        or hidden_states.device != installed_device
        or hidden_states.dtype not in {torch.float16, torch.bfloat16}
        or module.weight.device != hidden_states.device
        or hidden_states.shape[-1] not in _PRISM_Q2_RMS_NORM_WIDTHS
    ):
        return _torch_rms_norm(module, hidden_states)

    values = hidden_states.contiguous()
    output = torch.empty_like(values)
    n_cols = values.shape[-1]
    rows = values.numel() // n_cols
    num_warps = _select_prism_q2_rms_norm_num_warps(dtype=values.dtype, n_cols=n_cols, rows=rows)
    _prism_q2_rms_norm_kernel[(rows,)](
        values,
        module.weight,
        output,
        n_cols=n_cols,
        eps=module.variance_epsilon,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def install_prism_q2_rms_norms(model: torch.nn.Module) -> int:
    """Install the profiled sm80 inference path on Qwen3 RMSNorm modules already resident on CUDA."""

    if not _TRITON_AVAILABLE or getattr(getattr(model, "config", None), "model_type", None) != "qwen3":
        return 0

    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3RMSNorm" or getattr(
            module, "_gptqmodel_prism_q2_rms_norm", False
        ):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.device.type != "cuda":
            continue
        if torch.cuda.get_device_capability(weight.device) != (8, 0):
            continue

        module._gptqmodel_prism_q2_rms_norm = True
        module._gptqmodel_prism_q2_rms_norm_device = weight.device
        module.forward = MethodType(_prism_q2_rms_norm_forward, module)
        installed += 1

    return installed


__all__ = ["install_prism_q2_rms_norms"]
