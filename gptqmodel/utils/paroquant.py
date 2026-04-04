# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant rotation helpers adapted from the ParoQuant paper and public
# project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""Utility helpers for ParoQuant rotations and extension loading."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


_SUPPORTED_ROTATION_KERNEL_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}

# Process-local cache for resolved fused launch configs. The steady-state
# rotation path uses this to avoid sending autotune sentinels back into the
# native op on every invocation once a shape has already been measured.
_ROTATION_LAUNCH_CONFIG_CACHE: dict[tuple[object, ...], tuple[int, int]] = {}
# The cache lock protects the Python-side resolved launch map for free-threaded
# runtimes, while the serialize lock prevents duplicate native autotune passes
# from multiple threads racing the same cold shape.
_ROTATION_LAUNCH_CONFIG_CACHE_LOCK = threading.Lock()
_ROTATION_AUTOTUNE_SERIALIZE_LOCK = threading.Lock()


def _normalize_group_size(group_size: int, in_features: int) -> int:
    """Validate and normalize a ParoQuant group size."""
    normalized = in_features if group_size == -1 else int(group_size)
    if normalized <= 0:
        raise ValueError(f"ParoQuant: invalid group_size `{group_size}` for in_features={in_features}.")
    if in_features % normalized != 0:
        raise ValueError(
            f"ParoQuant: in_features ({in_features}) must be divisible by group_size ({normalized})."
        )
    if normalized % 2 != 0:
        raise ValueError(f"ParoQuant: group_size ({normalized}) must be even.")
    return normalized


def build_identity_rotation_buffers(
    *,
    in_features: int,
    group_size: int,
    krot: int,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the identity rotation buffers used as the default runtime state."""
    normalized_group_size = _normalize_group_size(group_size, in_features)
    if krot <= 0:
        raise ValueError(f"ParoQuant: `krot` must be positive, got {krot}.")

    pairs_row = []
    local_pairs = torch.arange(normalized_group_size, dtype=torch.int16)
    num_groups = in_features // normalized_group_size
    for _ in range(num_groups):
        pairs_row.append(local_pairs)
    pairs_single = torch.cat(pairs_row, dim=0)
    pairs = pairs_single.unsqueeze(0).repeat(krot, 1)

    theta = torch.zeros((krot, in_features // 2), dtype=dtype)
    channel_scales = torch.ones((1, in_features), dtype=dtype)

    if device is not None:
        pairs = pairs.to(device=device)
        theta = theta.to(device=device)
        channel_scales = channel_scales.to(device=device)

    return pairs.contiguous(), theta.contiguous(), channel_scales.contiguous()


def is_identity_rotation(theta: torch.Tensor, channel_scales: Optional[torch.Tensor]) -> bool:
    """Check whether a ParoQuant rotation reduces to a no-op."""
    if theta is None:
        return True
    if torch.count_nonzero(theta).item() != 0:
        return False
    if channel_scales is None:
        return True
    return bool(torch.all(channel_scales == 1))


def apply_paroquant_rotation_reference(
    x: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of the ParoQuant input rotation."""
    orig_shape = x.shape
    if x.dim() < 2:
        raise ValueError(f"ParoQuant rotation expects rank >= 2, got shape {tuple(orig_shape)}.")

    x2d = x.reshape(-1, orig_shape[-1])
    hidden = x2d.shape[-1]
    group_size = _normalize_group_size(group_size, hidden)

    if scales is not None:
        scale_tensor = scales.reshape(1, -1).to(device=x2d.device, dtype=x2d.dtype)
        out = x2d * scale_tensor
    else:
        out = x2d.clone()

    num_groups = hidden // group_size
    half_group = group_size // 2
    pairs = pairs.to(device=x2d.device, dtype=torch.int64)
    theta = theta.to(device=x2d.device, dtype=torch.float32)

    for rot_idx in range(pairs.shape[0]):
        next_out = out.clone()
        pair_row = pairs[rot_idx].reshape(num_groups, half_group, 2)
        theta_row = theta[rot_idx].reshape(num_groups, half_group)
        for group_idx in range(num_groups):
            group_offset = group_idx * group_size
            idx_i = group_offset + pair_row[group_idx, :, 0]
            idx_j = group_offset + pair_row[group_idx, :, 1]
            cos_t = torch.cos(theta_row[group_idx]).to(dtype=out.dtype)
            sin_t = torch.sin(theta_row[group_idx]).to(dtype=out.dtype)
            xi = out[:, idx_i]
            xj = out[:, idx_j]
            next_out[:, idx_i] = xi * cos_t + xj * sin_t
            next_out[:, idx_j] = -xi * sin_t + xj * cos_t
        out = next_out

    return out.reshape(orig_shape)


def _rotation_sources() -> list[str]:
    """Return the native extension sources for the fused CUDA rotation op.

    Build this as a plain custom-op library instead of a Python extension.
    The Python-module path pulls in pybind11 initialization that segfaults on
    this host during ``PyInit_*`` even though the CUDA op itself is fine.
    """
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "paroquant"
    return [
        str(root / "rotation.cu"),
    ]


def _rotation_kernel_ready(
    x: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    scales: Optional[torch.Tensor],
    group_size: int,
) -> bool:
    """Check whether the fused CUDA rotation kernel can service this call."""
    return (
        x.device.type == "cuda"
        and x.dtype in _SUPPORTED_ROTATION_KERNEL_DTYPES
        and pairs.device.type == "cuda"
        and theta.device.type == "cuda"
        and (scales is None or scales.device.type == "cuda")
        and int(group_size) in {128}
        and int(theta.shape[0]) in {1, 8}
    )


def _rotation_visible_cuda_capabilities() -> set[tuple[int, int]]:
    """Return the visible CUDA SM versions for this process."""

    if not torch.cuda.is_available():
        return set()
    return {
        tuple(map(int, torch.cuda.get_device_capability(device_index)))
        for device_index in range(torch.cuda.device_count())
    }


def _rotation_extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags()
    # On measured SM80/A100 builds, relaxed constexpr consistently improves the
    # fused rotation kernel. The same flag regresses SM89/4090 here, so only
    # enable it when the visible target set is pure SM80.
    if _rotation_visible_cuda_capabilities() == {(8, 0)}:
        flags.append("--expt-relaxed-constexpr")
    return flags


# Shared singleton so ParoQuant uses the same torch.ops JIT lifecycle helpers as
# AWQ and other custom-op extensions.
_PAROQUANT_ROTATION_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_paroquant_rotation",
    namespace="gptqmodel_paroquant",
    required_ops=("rotate", "launch_config", "clear_autotune_cache", "autotune_cache_size"),
    sources=_rotation_sources,
    build_root_env="GPTQMODEL_PAROQUANT_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("paroquant"),
    display_name="ParoQuant rotation",
    extra_cuda_cflags=_rotation_extra_cuda_cflags,
    extra_cflags=lambda: default_jit_cflags(opt_level="O2"),
    force_rebuild_env="GPTQMODEL_PAROQUANT_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def clear_paroquant_rotation_extension_cache() -> None:
    """Delete cached ParoQuant rotation JIT artifacts before the next load attempt."""

    clear_paroquant_rotation_autotune_cache()
    _PAROQUANT_ROTATION_EXTENSION.clear_cache()


def clear_paroquant_rotation_autotune_cache() -> None:
    """Drop the native launch-plan cache used by fused rotation autotune."""

    with _ROTATION_LAUNCH_CONFIG_CACHE_LOCK:
        _ROTATION_LAUNCH_CONFIG_CACHE.clear()
    clear_cache = _rotation_native_op_if_loaded("clear_autotune_cache")
    if clear_cache is not None:
        clear_cache()


def _load_rotation_extension() -> bool:
    """JIT-build and load the optional fused CUDA rotation extension once."""

    return _extension_api().is_available("paroquant")


def _rotation_env_int(name: str, allowed: set[int]) -> Optional[int]:
    """Parse a bounded integer environment override."""
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value in allowed else None


def _rotation_env_flag(name: str, default: bool) -> bool:
    """Parse a boolean environment flag used by rotation autotune."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _rotation_autotune_enabled() -> bool:
    """Report whether fused rotation launch autotune is enabled for this process."""
    return _rotation_env_flag("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE", True)


def _rotation_native_op_if_loaded(op_name: str):
    """Return a loaded torch.ops handle without triggering JIT compilation."""
    namespace = _PAROQUANT_ROTATION_EXTENSION._namespace_cache
    if namespace is None or not hasattr(namespace, op_name):
        return None
    cached = _PAROQUANT_ROTATION_EXTENSION._op_cache.get(op_name)
    if cached is not None:
        return cached
    op = getattr(namespace, op_name)
    _PAROQUANT_ROTATION_EXTENSION._op_cache[op_name] = op
    return op


def _rotation_requested_launch() -> tuple[int, int]:
    """Resolve the requested fused launch policy from env overrides or autotune."""
    cta_m_override = _rotation_env_int("GPTQMODEL_PAROQUANT_ROTATE_CTA_M", {4, 8, 16})
    row_pad_override = _rotation_env_int("GPTQMODEL_PAROQUANT_ROTATE_ROW_PAD", {0, 2})
    if cta_m_override is not None or row_pad_override is not None:
        return (
            -1 if cta_m_override is None else int(cta_m_override),
            -1 if row_pad_override is None else int(row_pad_override),
        )
    if _rotation_autotune_enabled():
        return (-2, -2)
    return (-1, -1)


def _rotation_autotune_cache_size() -> int:
    """Return the native fused-rotation autotune cache size for tests and benchmarks."""
    cache_size = _rotation_native_op_if_loaded("autotune_cache_size")
    if cache_size is not None:
        return int(cache_size())
    if not _load_rotation_extension():
        return 0
    return int(_extension_api().op("paroquant", "autotune_cache_size")())


def _run_rotation_op(
    x: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    scales: Optional[torch.Tensor],
    group_size: int,
    *,
    cta_m: int,
    row_pad: int,
) -> torch.Tensor:
    """Execute the fused rotation op with one requested launch policy."""
    return _extension_api().op("paroquant", "rotate")(
        x,
        pairs,
        theta,
        scales,
        int(group_size),
        int(cta_m),
        int(row_pad),
    )


def _rotation_launch_cache_key(
    x: torch.Tensor,
    *,
    krot: int,
    has_scale: bool,
    group_size: int,
    requested_cta_m: int,
    requested_row_pad: int,
) -> tuple[object, ...]:
    """Build the process-local cache key for one resolved launch shape."""
    hidden = int(x.shape[-1])
    batch_rows = int(x.numel() // hidden)
    return (
        int(x.device.index if x.device.index is not None else -1),
        x.dtype,
        batch_rows,
        hidden,
        int(group_size),
        int(krot),
        bool(has_scale),
        int(requested_cta_m),
        int(requested_row_pad),
    )


def _resolve_rotation_launch(
    x: torch.Tensor,
    *,
    scales: Optional[torch.Tensor],
    group_size: int,
    krot: int,
    requested_cta_m: int,
    requested_row_pad: int,
) -> tuple[int, int]:
    """Resolve one fused launch config and memoize autotuned shape choices."""
    if requested_cta_m != -2 and requested_row_pad != -2:
        return int(requested_cta_m), int(requested_row_pad)

    cache_key = _rotation_launch_cache_key(
        x,
        krot=krot,
        has_scale=scales is not None,
        group_size=group_size,
        requested_cta_m=requested_cta_m,
        requested_row_pad=requested_row_pad,
    )
    with _ROTATION_LAUNCH_CONFIG_CACHE_LOCK:
        cached = _ROTATION_LAUNCH_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _ROTATION_AUTOTUNE_SERIALIZE_LOCK:
        with _ROTATION_LAUNCH_CONFIG_CACHE_LOCK:
            cached = _ROTATION_LAUNCH_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

        cta_m, row_pad = _extension_api().op("paroquant", "launch_config")(
            x,
            int(krot),
            scales is not None,
            int(group_size),
            int(requested_cta_m),
            int(requested_row_pad),
        )
        resolved = (int(cta_m), int(row_pad))
        with _ROTATION_LAUNCH_CONFIG_CACHE_LOCK:
            cached = _ROTATION_LAUNCH_CONFIG_CACHE.setdefault(cache_key, resolved)
        return cached


def _rotation_launch_config(
    x: torch.Tensor,
    pairs: Optional[torch.Tensor] = None,
    theta: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    group_size: int = 128,
    *,
    extension_loaded: bool = False,
    kernel_ready: bool = False,
) -> tuple[int, int]:
    """Query the native fused-kernel launch shape for one CUDA tensor."""
    del pairs, kernel_ready
    if x.device.type != "cuda":
        raise ValueError("ParoQuant launch config requires a CUDA tensor.")
    if not extension_loaded and not _load_rotation_extension():
        raise RuntimeError("ParoQuant launch config requires the fused rotation extension.")
    requested_cta_m, requested_row_pad = _rotation_requested_launch()
    krot = 8 if theta is None else int(theta.shape[0])
    if requested_cta_m == -2 or requested_row_pad == -2:
        cta_m, row_pad = _resolve_rotation_launch(
            x,
            scales=scales,
            group_size=int(group_size),
            krot=krot,
            requested_cta_m=requested_cta_m,
            requested_row_pad=requested_row_pad,
        )
    else:
        cta_m, row_pad = _extension_api().op("paroquant", "launch_config")(
            x,
            int(krot),
            scales is not None,
            int(group_size),
            int(requested_cta_m),
            int(requested_row_pad),
        )
    return int(cta_m), int(row_pad)


def _apply_fused_rotation(
    x: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    scales: Optional[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    """Run the fused rotation op with explicit overrides or native autotune sentinels."""
    requested_cta_m, requested_row_pad = _rotation_requested_launch()
    cta_m, row_pad = _resolve_rotation_launch(
        x,
        scales=scales,
        group_size=int(group_size),
        krot=int(theta.shape[0]),
        requested_cta_m=requested_cta_m,
        requested_row_pad=requested_row_pad,
    )
    return _run_rotation_op(
        x,
        pairs,
        theta,
        scales,
        group_size,
        cta_m=cta_m,
        row_pad=row_pad,
    )


def prewarm_paroquant_rotation_extension(
    *,
    fused_rotation: bool,
    group_size: int,
    krot: int,
    device: Optional[torch.device | str] = None,
) -> bool:
    """Eagerly build the fused rotation extension before timed quantization starts."""
    if not fused_rotation:
        return False
    if int(group_size) not in {128}:
        return False
    if int(krot) not in {1, 8}:
        return False

    if device is not None and torch.device(device).type != "cuda":
        return False
    if device is None and not torch.cuda.is_available():
        return False

    return _load_rotation_extension()


def apply_paroquant_rotation(
    x: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Apply the fused rotation when available, else fall back to the reference path."""
    if _rotation_kernel_ready(x, pairs, theta, scales, group_size) and _load_rotation_extension():
        return _apply_fused_rotation(x, pairs, theta, scales, int(group_size))
    return apply_paroquant_rotation_reference(x, pairs, theta, scales=scales, group_size=group_size)


class _ParoQuantRotateTensorFunc(torch.autograd.Function):
    """Autograd wrapper around the fused ParoQuant CUDA rotation kernel."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        pairs: torch.Tensor,
        theta: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ) -> torch.Tensor:
        scale_tensor = None if scales is None else scales.contiguous()
        ctx.orig_shape = x.shape
        ctx.orig_dtype = x.dtype
        ctx.has_scale = scale_tensor is not None
        ctx.group_size = int(group_size)
        ctx.needs_x_grad = bool(x.requires_grad)
        ctx.needs_theta_grad = bool(theta.requires_grad)
        ctx.needs_scale_grad = bool(scale_tensor is not None and scale_tensor.requires_grad)

        y = _apply_fused_rotation(x, pairs, theta, scale_tensor, int(group_size))
        saved = (x, pairs, theta, y, scale_tensor) if ctx.has_scale else (x, pairs, theta, y)
        ctx.save_for_backward(*saved)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        saved_tensors = ctx.saved_tensors
        x, pairs, theta, y = saved_tensors[:4]
        scale_tensor = saved_tensors[4] if ctx.has_scale else None
        group_size = ctx.group_size

        _, hidden = pairs.shape
        num_groups = hidden // group_size
        half_group = group_size // 2
        batch_rows = y.numel() // hidden
        grad = grad_out.reshape(batch_rows, hidden)
        grad_theta = None

        if ctx.needs_theta_grad:
            rotated = y.reshape(batch_rows, hidden)
            grad_theta = torch.zeros_like(theta)
            offsets = (
                torch.arange(num_groups, device=pairs.device, dtype=torch.long).unsqueeze(1) * group_size
            )

            for rot_idx in range(pairs.shape[0] - 1, -1, -1):
                pair_row = pairs.narrow(0, rot_idx, 1)
                neg_theta = theta.narrow(0, rot_idx, 1).neg()
                rotated = _apply_fused_rotation(rotated, pair_row, neg_theta, None, group_size)
                grad = _apply_fused_rotation(grad, pair_row, neg_theta, None, group_size)

                pair_view = pair_row.reshape(num_groups, group_size)
                idx_i = (pair_view[:, 0::2] + offsets).reshape(-1)
                idx_j = (pair_view[:, 1::2] + offsets).reshape(-1)

                xi = rotated[:, idx_i].reshape(batch_rows, num_groups, half_group)
                xj = rotated[:, idx_j].reshape(batch_rows, num_groups, half_group)
                grad_i = grad[:, idx_i].reshape(batch_rows, num_groups, half_group)
                grad_j = grad[:, idx_j].reshape(batch_rows, num_groups, half_group)

                theta_view = theta.narrow(0, rot_idx, 1).reshape(num_groups, half_group)
                sin_t = theta_view.sin()
                cos_t = theta_view.cos()
                grad_theta[rot_idx] = (
                    (
                        (grad_i * xj - grad_j * xi).sum(0) * cos_t
                        - (grad_i * xi + grad_j * xj).sum(0) * sin_t
                    )
                    .reshape(-1)
                    .to(theta.dtype)
                )
        else:
            for rot_idx in range(pairs.shape[0] - 1, -1, -1):
                pair_row = pairs.narrow(0, rot_idx, 1)
                neg_theta = theta.narrow(0, rot_idx, 1).neg()
                grad = _apply_fused_rotation(grad, pair_row, neg_theta, None, group_size)

        if ctx.has_scale:
            scale_flat = scale_tensor.reshape(-1)
            grad_x = None
            if ctx.needs_x_grad:
                grad_x = (grad * scale_flat.unsqueeze(0)).reshape(ctx.orig_shape).to(ctx.orig_dtype)
            grad_scale = None
            if ctx.needs_scale_grad:
                grad_scale = (x.reshape(batch_rows, hidden) * grad).sum(0).to(
                    dtype=scale_tensor.dtype,
                    device=scale_tensor.device,
                )
                grad_scale = grad_scale.reshape_as(scale_tensor)
        else:
            grad_x = grad.reshape(ctx.orig_shape).to(ctx.orig_dtype) if ctx.needs_x_grad else None
            grad_scale = None

        return grad_x, None, grad_theta, grad_scale, None


def apply_paroquant_rotation_autograd(
    x: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Apply the fused rotation with custom backward support when available."""
    if _rotation_kernel_ready(x, pairs, theta, scales, group_size) and _load_rotation_extension():
        return _ParoQuantRotateTensorFunc.apply(x, pairs, theta, scales, int(group_size))
    return apply_paroquant_rotation_reference(x, pairs, theta, scales=scales, group_size=group_size)


__all__ = [
    "apply_paroquant_rotation",
    "apply_paroquant_rotation_autograd",
    "apply_paroquant_rotation_reference",
    "_rotation_launch_config",
    "build_identity_rotation_buffers",
    "clear_paroquant_rotation_autotune_cache",
    "clear_paroquant_rotation_extension_cache",
    "is_identity_rotation",
    "prewarm_paroquant_rotation_extension",
]
