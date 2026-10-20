# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant rotation helpers adapted from the ParoQuant paper and public
# project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""Utility helpers for ParoQuant rotations and extension loading."""

from __future__ import annotations

import contextlib
import hashlib
import json
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
from .env import env_flag


_SUPPORTED_ROTATION_KERNEL_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}

# Reverse lookup used to rebuild persisted launch-cache keys from their names.
_ROTATION_AUTOTUNE_DTYPE_BY_NAME = {str(dtype): dtype for dtype in _SUPPORTED_ROTATION_KERNEL_DTYPES}

# Process-local cache for resolved fused launch configs. The steady-state
# rotation path uses this to avoid sending autotune sentinels back into the
# native op on every invocation once a shape has already been measured.
_ROTATION_LAUNCH_CONFIG_CACHE: dict[tuple[object, ...], tuple[int, int]] = {}
# The cache lock protects the Python-side resolved launch map for free-threaded
# runtimes, while the serialize lock prevents duplicate native autotune passes
# from multiple threads racing the same cold shape.
_ROTATION_LAUNCH_CONFIG_CACHE_LOCK = threading.Lock()
_ROTATION_AUTOTUNE_SERIALIZE_LOCK = threading.Lock()
# The requested launch policy is derived from process-static environment
# overrides, so resolving it once per lifecycle keeps the steady-state rotation
# path free of repeated ``os.getenv`` calls. ``clear_*`` and the prewarm entry
# point reset it so a benchmark can still switch policies between runs.
_ROTATION_LAUNCH_POLICY_CACHE: Optional[tuple[int, int]] = None
# Latched once the fused rotation extension has loaded successfully in this
# process. A failed attempt is never latched so a later rebuild can retry.
_ROTATION_EXTENSION_READY = False

# Autotuning a cold rotation shape costs ~130ms of native benchmarking, and a
# full lifecycle measures ~18 shapes, so the resolved launch configs are also
# persisted per device/torch/HIP fingerprint. Bump the format whenever the
# native candidate set or its scoring changes so stale picks are discarded.
_ROTATION_AUTOTUNE_CACHE_FORMAT = 1
_ROTATION_AUTOTUNE_CACHE_ENV = "GPTQMODEL_PAROQUANT_AUTOTUNE_CACHE"
_ROTATION_AUTOTUNE_DISK_LOCK = threading.Lock()
# device index -> loaded flag, so the on-disk cache is read at most once.
_ROTATION_AUTOTUNE_DISK_LOADED: set[int] = set()
# Mirrors what the current process has written so the file can be rewritten
# wholesale without re-reading it.
_ROTATION_AUTOTUNE_DISK_ENTRIES: dict[tuple[object, ...], tuple[int, int]] = {}


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
        return True if channel_scales is None else bool(torch.all(channel_scales == 1))

    if not theta.is_cuda:
        if torch.count_nonzero(theta).item() != 0:
            return False
        if channel_scales is None:
            return True
        return bool(torch.all(channel_scales == 1))

    # Keep both predicates on the device and read the combined scalar once.
    theta_is_identity = torch.count_nonzero(theta) == 0
    if channel_scales is None:
        return bool(theta_is_identity)
    scales_are_identity = torch.all(channel_scales == 1)
    return bool(theta_is_identity & scales_are_identity)


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
        x.is_cuda
        and x.dtype in _SUPPORTED_ROTATION_KERNEL_DTYPES
        and pairs.is_cuda
        and theta.is_cuda
        and (scales is None or scales.is_cuda)
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
    required_ops=("rotate", "launch_config", "grad_theta", "clear_autotune_cache", "autotune_cache_size"),
    sources=_rotation_sources,
    build_root_env="GPTQMODEL_PAROQUANT_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("paroquant"),
    display_name="ParoQuant rotation",
    extra_cuda_cflags=_rotation_extra_cuda_cflags,
    extra_cflags=lambda: default_jit_cflags(opt_level="O2"),
    force_rebuild_env="GPTQMODEL_PAROQUANT_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def clear_paroquant_rotation_extension_cache() -> None:
    """Delete cached ParoQuant rotation JIT artifacts before the next load attempt."""

    global _ROTATION_EXTENSION_READY

    _ROTATION_EXTENSION_READY = False
    clear_paroquant_rotation_autotune_cache()
    _PAROQUANT_ROTATION_EXTENSION.clear_cache()


def clear_paroquant_rotation_autotune_cache() -> None:
    """Drop the native launch-plan cache used by fused rotation autotune."""

    global _ROTATION_LAUNCH_POLICY_CACHE

    _ROTATION_LAUNCH_POLICY_CACHE = None
    with _ROTATION_LAUNCH_CONFIG_CACHE_LOCK:
        _ROTATION_LAUNCH_CONFIG_CACHE.clear()
    with _ROTATION_AUTOTUNE_DISK_LOCK:
        known_devices = sorted(_ROTATION_AUTOTUNE_DISK_LOADED)
        _ROTATION_AUTOTUNE_DISK_LOADED.clear()
        _ROTATION_AUTOTUNE_DISK_ENTRIES.clear()
        for device_index in known_devices:
            with contextlib.suppress(OSError):
                _rotation_autotune_cache_path(device_index).unlink()
    clear_cache = _rotation_native_op_if_loaded("clear_autotune_cache")
    if clear_cache is not None:
        clear_cache()


def _load_rotation_extension() -> bool:
    """JIT-build and load the optional fused CUDA rotation extension once."""

    global _ROTATION_EXTENSION_READY

    if _ROTATION_EXTENSION_READY:
        return True
    loaded = _extension_api().is_available("paroquant")
    if loaded:
        _ROTATION_EXTENSION_READY = True
    return loaded


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
    global _ROTATION_LAUNCH_POLICY_CACHE

    cached = _ROTATION_LAUNCH_POLICY_CACHE
    if cached is not None:
        return cached

    cta_m_override = _rotation_env_int("GPTQMODEL_PAROQUANT_ROTATE_CTA_M", {4, 8, 16})
    row_pad_override = _rotation_env_int("GPTQMODEL_PAROQUANT_ROTATE_ROW_PAD", {0, 2})
    if cta_m_override is not None or row_pad_override is not None:
        resolved = (
            -1 if cta_m_override is None else int(cta_m_override),
            -1 if row_pad_override is None else int(row_pad_override),
        )
    elif _rotation_autotune_enabled():
        resolved = (-2, -2)
    else:
        resolved = (-1, -1)
    _ROTATION_LAUNCH_POLICY_CACHE = resolved
    return resolved


def _paroquant_rotation_op(op_name: str):
    """Return a fused-rotation op handle without re-entering the shared loader.

    ``TorchOpsJitExtension.op`` already memoizes handles, so an already loaded
    namespace only needs that cheap lookup. Going through the shared loader is
    reserved for the first call, which keeps the load contract (and the ability
    for tests to swap the handle source) intact while dropping the per-call
    ``load()`` re-validation from the steady-state rotation path.
    """
    if _PAROQUANT_ROTATION_EXTENSION._namespace_cache is not None:
        return _PAROQUANT_ROTATION_EXTENSION.op(op_name)
    return _extension_api().op("paroquant", op_name)


def _rotation_autotune_cache_size() -> int:
    """Return the native fused-rotation autotune cache size for tests and benchmarks."""
    cache_size = _rotation_native_op_if_loaded("autotune_cache_size")
    if cache_size is not None:
        return int(cache_size())
    if not _load_rotation_extension():
        return 0
    return int(_paroquant_rotation_op("autotune_cache_size")())


def _run_rotation_op(
    x: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    scales: Optional[torch.Tensor],
    group_size: int,
    *,
    cta_m: int,
    row_pad: int,
    reverse: bool = False,
) -> torch.Tensor:
    """Execute the fused rotation op with one requested launch policy.

    ``reverse`` walks the rotation rows backward with negated thetas inside the
    same kernel, which is the inverse transform the autograd backward needs when
    only the input carries a gradient.
    """
    return _paroquant_rotation_op("rotate")(
        x,
        pairs,
        theta,
        scales,
        int(group_size),
        int(cta_m),
        int(row_pad),
        int(reverse),
    )


def _fused_grad_theta_op():
    """Return the fused rotation-gradient op, or None when it cannot be used.

    The autograd backward runs this once per rotation per layer, so the torch
    formulation costs ~21 tiny kernel launches each. The native op folds the
    four gathers, the elementwise math and the two batch reductions into a
    single launch. Only float32 CUDA activations are supported; every other
    case keeps the reference torch path.
    """
    try:
        return _paroquant_rotation_op("grad_theta")
    except Exception:
        return None


def _fused_grad_theta_ready(rotated: torch.Tensor, grad: torch.Tensor) -> bool:
    """Report whether the fused rotation-gradient op can service this pair."""
    return (
        rotated.dtype == torch.float32
        and grad.dtype == torch.float32
        and rotated.is_cuda
        and grad.is_cuda
        and rotated.is_contiguous()
        and grad.is_contiguous()
        and rotated.shape == grad.shape
    )


def _fused_rotate_pair_op():
    """Return the fused two-tensor rotation op, or None when it cannot be used.

    The autograd backward rotates the saved activation and the incoming gradient
    with the same pair/theta set, which costs two launches per rotation. Both
    operands share a shape and differ only in storage, so the native op folds
    them into a single launch. Unsupported dtypes/devices keep the two-call path.
    """
    try:
        return _paroquant_rotation_op("rotate_pair")
    except Exception:
        return None


def _fused_rotate_pair_ready(x: torch.Tensor, grad: torch.Tensor) -> bool:
    """Report whether the fused two-tensor rotation can service this pair."""
    return (
        x.dtype == torch.float32
        and grad.dtype == torch.float32
        and x.is_cuda
        and grad.is_cuda
        and x.is_contiguous()
        and grad.is_contiguous()
        and x.shape == grad.shape
    )


def _apply_fused_rotation_pair(
    x: torch.Tensor,
    grad: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    group_size: int,
    *,
    op=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate two equal-shaped tensors with one fused launch and a shared launch plan."""
    requested_cta_m, requested_row_pad = _rotation_requested_launch()
    cta_m, row_pad = _resolve_rotation_launch(
        x,
        scales=None,
        group_size=int(group_size),
        krot=int(theta.shape[0]),
        requested_cta_m=requested_cta_m,
        requested_row_pad=requested_row_pad,
    )
    if op is None:
        op = _fused_rotate_pair_op()
    if op is None:
        raise RuntimeError("ParoQuant rotate_pair op is unavailable.")
    return tuple(op(x, grad, pairs, theta, int(group_size), int(cta_m), int(row_pad)))


def _fused_rotate_pair_grad_op():
    """Return the fused backward-sweep op, or None when it cannot be used.

    The autograd backward rotates the saved activation and the incoming gradient
    in reverse order and measures the rotation gradient after every step, which
    costs one two-tensor rotation plus one rotation-gradient launch per rotation
    plus the host-side sin/cos and index staging. The native op keeps the running
    rotated state in shared memory and returns the final pair together with the
    whole rotation gradient, so the steady-state backward becomes two launches.
    Unsupported dtypes/devices keep the reference torch path.
    """
    try:
        return _paroquant_rotation_op("rotate_pair_grad")
    except Exception:
        return None


def _fused_rotate_pair_grad_ready(x: torch.Tensor, grad: torch.Tensor) -> bool:
    """Report whether the fused backward-sweep op can service this pair."""
    return (
        x.dtype == torch.float32
        and grad.dtype == torch.float32
        and x.is_cuda
        and grad.is_cuda
        and x.is_contiguous()
        and grad.is_contiguous()
        and x.shape == grad.shape
    )


def _apply_fused_rotation_pair_grad(
    x: torch.Tensor,
    grad: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    group_size: int,
    *,
    op=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the whole backward rotation sweep plus its gradient in two launches."""
    requested_cta_m, requested_row_pad = _rotation_requested_launch()
    cta_m, row_pad = _resolve_rotation_launch(
        x,
        scales=None,
        group_size=int(group_size),
        krot=int(theta.shape[0]),
        requested_cta_m=requested_cta_m,
        requested_row_pad=requested_row_pad,
    )
    if op is None:
        op = _fused_rotate_pair_grad_op()
    if op is None:
        raise RuntimeError("ParoQuant rotate_pair_grad op is unavailable.")
    return tuple(op(x, grad, pairs, theta, int(group_size), int(cta_m), int(row_pad)))


def _rotation_autotune_cache_enabled() -> bool:
    return env_flag(_ROTATION_AUTOTUNE_CACHE_ENV, default=True)


def _rotation_autotune_cache_dir() -> Path:
    """Return the user cache root, matching the other GPTQModel caches."""
    configured = os.getenv("GPTQMODEL_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "gptqmodel"
    return Path.home() / ".cache" / "gptqmodel"


def _rotation_autotune_cache_identity(device_index: int) -> str:
    """
    Fingerprint the tuning environment.

    Launch choices are measured on real hardware, so a cache written for one
    device/driver/torch combination must never be reused for another.
    """
    parts = [f"format={_ROTATION_AUTOTUNE_CACHE_FORMAT}", f"torch={torch.__version__}"]
    parts.append(f"hip={getattr(torch.version, 'hip', None) or 'none'}")
    device_name = "unknown"
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(device_index if device_index >= 0 else 0)
        except Exception:
            device_name = "unknown"
    parts.append(f"device={device_name}")
    return "|".join(parts)


def _rotation_autotune_cache_path(device_index: int) -> Path:
    identity = _rotation_autotune_cache_identity(device_index)
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return _rotation_autotune_cache_dir() / "paroquant" / "rotation_launch" / f"{digest}.json"


def _ensure_rotation_autotune_cache_loaded(device_index: int) -> None:
    """Seed the in-process launch cache from disk once per device."""
    if not _rotation_autotune_cache_enabled():
        return
    with _ROTATION_AUTOTUNE_DISK_LOCK:
        if device_index in _ROTATION_AUTOTUNE_DISK_LOADED:
            return
        _ROTATION_AUTOTUNE_DISK_LOADED.add(device_index)
        try:
            raw = json.loads(_rotation_autotune_cache_path(device_index).read_text())
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict) or raw.get("format") != _ROTATION_AUTOTUNE_CACHE_FORMAT:
            return
        entries = raw.get("entries")
        if not isinstance(entries, list):
            return
        with _ROTATION_LAUNCH_CONFIG_CACHE_LOCK:
            for entry in entries:
                parsed = _rotation_autotune_entry_to_cache(entry)
                if parsed is None:
                    continue
                cache_key, resolved = parsed
                _ROTATION_LAUNCH_CONFIG_CACHE.setdefault(cache_key, resolved)
                _ROTATION_AUTOTUNE_DISK_ENTRIES[cache_key] = resolved


def _rotation_autotune_entry_to_cache(
    entry: object,
) -> Optional[tuple[tuple[object, ...], tuple[int, int]]]:
    """Rebuild one ``(cache_key, resolved)`` pair from its serialized form."""
    if not isinstance(entry, list) or len(entry) != 11:
        return None
    device_index, dtype_name, rows, hidden, group_size, krot, has_scale, req_cta_m, req_row_pad, cta_m, row_pad = entry
    dtype = _ROTATION_AUTOTUNE_DTYPE_BY_NAME.get(str(dtype_name))
    if dtype is None:
        return None
    cache_key = (
        int(device_index),
        dtype,
        int(rows),
        int(hidden),
        int(group_size),
        int(krot),
        bool(has_scale),
        int(req_cta_m),
        int(req_row_pad),
    )
    return cache_key, (int(cta_m), int(row_pad))


def _persist_rotation_autotune_entry(
    cache_key: tuple[object, ...], resolved: tuple[int, int], device_index: int
) -> None:
    """Record one resolved launch config on disk for the next process."""
    if not _rotation_autotune_cache_enabled():
        return
    with _ROTATION_AUTOTUNE_DISK_LOCK:
        _ROTATION_AUTOTUNE_DISK_ENTRIES[cache_key] = resolved
        path = _rotation_autotune_cache_path(device_index)
        payload = {
            "format": _ROTATION_AUTOTUNE_CACHE_FORMAT,
            "identity": _rotation_autotune_cache_identity(device_index),
            "entries": [
                _rotation_autotune_entry_from_cache(key, value)
                for key, value in _ROTATION_AUTOTUNE_DISK_ENTRIES.items()
                if int(key[0]) == int(device_index)
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
            tmp_path.write_text(json.dumps(payload))
            os.replace(tmp_path, path)
        except OSError:
            return


def _rotation_autotune_entry_from_cache(
    cache_key: tuple[object, ...], resolved: tuple[int, int]
) -> list[object]:
    """Serialize one ``(cache_key, resolved)`` pair; dtypes become names."""
    device, dtype, rows, hidden, group_size, krot, has_scale, req_cta_m, req_row_pad = cache_key
    return [
        int(device),
        str(dtype),
        int(rows),
        int(hidden),
        int(group_size),
        int(krot),
        int(bool(has_scale)),
        int(req_cta_m),
        int(req_row_pad),
        int(resolved[0]),
        int(resolved[1]),
    ]


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
    device_index = int(cache_key[0])
    # Seed from the previous process before measuring anything: a full cold
    # lifecycle autotunes ~18 shapes at ~130ms each.
    _ensure_rotation_autotune_cache_loaded(device_index)
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
        _persist_rotation_autotune_entry(cache_key, cached, device_index)
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
    *,
    reverse: bool = False,
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
        reverse=reverse,
    )


def prewarm_paroquant_rotation_extension(
    *,
    fused_rotation: bool,
    group_size: int,
    krot: int,
    device: Optional[torch.device | str] = None,
) -> bool:
    """Eagerly build the fused rotation extension before timed quantization starts."""
    global _ROTATION_LAUNCH_POLICY_CACHE

    # A prewarm marks the start of a new lifecycle, so re-read the launch
    # policy overrides instead of reusing the previous run's resolution.
    _ROTATION_LAUNCH_POLICY_CACHE = None
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
            # The fused native op keeps the running rotated pair in shared memory
            # and accumulates every rotation's gradient in one pass, so the whole
            # reverse sweep plus the rotation gradient costs two launches instead
            # of a per-rotation rotation/gradient pair plus the host-side staging
            # of the negated thetas, the sin/cos tables and the gather indices.
            fused_pair_grad = (
                _fused_rotate_pair_grad_op()
                if _fused_rotate_pair_grad_ready(rotated, grad)
                else None
            )
            if fused_pair_grad is not None:
                rotated, grad, grad_theta = _apply_fused_rotation_pair_grad(
                    rotated,
                    grad,
                    pairs,
                    theta,
                    group_size,
                    op=fused_pair_grad,
                )
                grad_theta = grad_theta.reshape(theta.shape).to(theta.dtype)
            else:
                grad_theta = torch.zeros_like(theta)
                offsets = (
                    torch.arange(num_groups, device=pairs.device, dtype=torch.long).unsqueeze(1)
                    * group_size
                )

                # The rotation set is fixed for the whole backward pass, so derive every
                # per-rotation constant once instead of rebuilding it inside the loop:
                # the negated thetas, their sin/cos, and the gather index pairs. This
                # removes krot sin/cos/neg launches and krot strided-slice+add passes.
                krot = pairs.shape[0]
                neg_theta_all = theta.neg()
                sin_all = theta.reshape(krot, num_groups, half_group).sin()
                cos_all = theta.reshape(krot, num_groups, half_group).cos()
                pair_view_all = pairs.reshape(krot, num_groups, group_size)
                idx_i_all = (pair_view_all[:, :, 0::2] + offsets).reshape(krot, -1)
                idx_j_all = (pair_view_all[:, :, 1::2] + offsets).reshape(krot, -1)

                # The fused native op replaces the four gathers plus the elementwise
                # and reduce ops of one rotation with a single launch. Fall back to
                # the reference formulation for unsupported dtypes/devices.
                fused_grad_theta = (
                    _fused_grad_theta_op() if _fused_grad_theta_ready(rotated, grad) else None
                )
                fused_parts = [None] * krot if fused_grad_theta is not None else None
                # The two backward rotations share a pair/theta set and shape, so the
                # native two-tensor op can service both operands in one launch.
                fused_rotate_pair = (
                    _fused_rotate_pair_op() if _fused_rotate_pair_ready(rotated, grad) else None
                )

                for rot_idx in range(krot - 1, -1, -1):
                    pair_row = pairs.narrow(0, rot_idx, 1)
                    neg_theta = neg_theta_all.narrow(0, rot_idx, 1)
                    if fused_rotate_pair is not None:
                        rotated, grad = _apply_fused_rotation_pair(
                            rotated,
                            grad,
                            pair_row,
                            neg_theta,
                            group_size,
                            op=fused_rotate_pair,
                        )
                    else:
                        rotated = _apply_fused_rotation(
                            rotated, pair_row, neg_theta, None, group_size
                        )
                        grad = _apply_fused_rotation(grad, pair_row, neg_theta, None, group_size)

                    idx_i = idx_i_all[rot_idx]
                    idx_j = idx_j_all[rot_idx]

                    if fused_grad_theta is not None:
                        fused_parts[rot_idx] = fused_grad_theta(
                            rotated,
                            grad,
                            idx_i,
                            idx_j,
                            sin_all[rot_idx].reshape(-1),
                            cos_all[rot_idx].reshape(-1),
                        )
                        continue

                    xi = rotated[:, idx_i].reshape(batch_rows, num_groups, half_group)
                    xj = rotated[:, idx_j].reshape(batch_rows, num_groups, half_group)
                    grad_i = grad[:, idx_i].reshape(batch_rows, num_groups, half_group)
                    grad_j = grad[:, idx_j].reshape(batch_rows, num_groups, half_group)

                    sin_t = sin_all[rot_idx]
                    cos_t = cos_all[rot_idx]
                    grad_theta[rot_idx] = (
                        (
                            (grad_i * xj - grad_j * xi).sum(0) * cos_t
                            - (grad_i * xi + grad_j * xj).sum(0) * sin_t
                        )
                        .reshape(-1)
                        .to(theta.dtype)
                    )

                if fused_parts is not None:
                    grad_theta = torch.stack(fused_parts, dim=0).reshape(theta.shape).to(theta.dtype)
        else:
            # Only the input carries a gradient, so the backward is the plain
            # inverse transform. The inverse sweep is the forward sweep with the
            # rotation rows walked backward and their thetas negated, which the
            # native kernel performs in place: one launch instead of a host-side
            # negate plus one launch per rotation.
            grad = _apply_fused_rotation(
                grad,
                pairs,
                theta,
                None,
                group_size,
                reverse=True,
            )

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
