# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch

from ...utils.backend import BACKEND
from .komodo import (
    AwqKomodoLinear,
    KomodoLinear,
    _assert_fp16_inference_input,
    _fuse_bias_enabled,
    _weight_quant_matmul,
)


_CANNOE_PREFETCH_ENV = "GPTQMODEL_CANNOE_PREFETCH"
_CANNOE_PREFETCH_MAX_BYTES_ENV = "GPTQMODEL_CANNOE_PREFETCH_MAX_BYTES"
_CANNOE_PREFETCH_MIN_BYTES_ENV = "GPTQMODEL_CANNOE_PREFETCH_MIN_BYTES"
_CANNOE_ACTIVE_CORES_ENV = "GPTQMODEL_CANNOE_ACTIVE_CORES"
_CANNOE_SPLIT_K_ENV = "GPTQMODEL_CANNOE_SPLIT_K"
_CANNOE_MIN_SPLIT_K_RATIO_ENV = "GPTQMODEL_CANNOE_MIN_SPLIT_K_RATIO"
_CANNOE_BASE_N_ENV = "GPTQMODEL_CANNOE_BASE_N"
_CANNOE_BASE_K_ENV = "GPTQMODEL_CANNOE_BASE_K"
_CANNOE_FUSED_ENV = "GPTQMODEL_CANNOE_FUSED"
_CANNOE_FUSED_REQUIRE_ENV = "GPTQMODEL_CANNOE_FUSED_REQUIRE"
_CANNOE_FUSED_OP_ENV = "GPTQMODEL_CANNOE_FUSED_OP"
_CANNOE_ASCENDC_ENV = "GPTQMODEL_CANNOE_ASCENDC"
_CANNOE_V3_ENV = "GPTQMODEL_CANNOE_V3"
_CANNOE_INNER_PRECISE_ENV = "GPTQMODEL_CANNOE_INNER_PRECISE"
_CANNOE_STAGED_DEQUANT_ENV = "GPTQMODEL_CANNOE_STAGED_DEQUANT"
_CANNOE_CUBE_CONSUMER_ENV = "GPTQMODEL_CANNOE_CUBE_CONSUMER"
# 910B CANN reserves this system workspace before the user workspace returned by GetUserWorkspace().
_CANNOE_CUBE_WORKSPACE_BYTES = 16 * 1024 * 1024
_NPU_PREFETCH_OP_UNSET = object()
_NPU_PREFETCH_OP = _NPU_PREFETCH_OP_UNSET
_NPU_WEIGHT_QUANT_OP_UNSET = object()
_NPU_WEIGHT_QUANT_OP = _NPU_WEIGHT_QUANT_OP_UNSET
_FUSED_OP_UNSET = object()
_FUSED_OP_CACHE = _FUSED_OP_UNSET
_FUSED_OP_CACHE_KEY = None
_FUSED_OP_LAST_ERROR = ""
_DEFAULT_FUSED_OP_NAMES = (
    "gptqmodel_cannoe.cannoe_w4_a16_matmul",
    "gptqmodel_cannoe.w4a16_matmul",
    "npu.cannoe_w4_a16_matmul",
    "npu.gptqmodel_cannoe_w4_a16_matmul",
)


def _cannoe_env(env_name: str) -> str | None:
    return os.getenv(env_name)


def _cannoe_env_flag(env_name: str, default: bool = False) -> bool:
    value = _cannoe_env(env_name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _cannoe_env_values(*env_names: str) -> tuple[str | None, ...]:
    return tuple(_cannoe_env(name) for name in env_names)


@dataclass(frozen=True)
class CannoeTilingPlan:
    rows: int
    in_features: int
    out_features: int
    group_size: int
    cube_cores: int
    vector_cores: int
    active_cores: int
    split_k: int
    split_k_shard_k: int
    base_m: int
    base_n: int
    base_k: int
    k_tiles_per_split: int
    vector_dequant_tasks: int
    int4_values_per_int32: int
    packed_int4_tile_bytes: int
    dequant_fp16_tile_bytes: int
    l0a_tile_bytes: int
    l0b_tile_bytes: int
    l0c_tile_bytes: int
    staged_dequant: bool
    staging_slots: int
    staging_blocks: int
    staging_tile_bytes: int
    staging_workspace_bytes: int
    staging_workspace_offset: int
    cube_consumer: bool
    cube_workspace_bytes: int
    custom_workspace_bytes: int
    l2_cache_size: int
    prefetch_enabled: bool
    prefetch_min_bytes: int
    prefetch_max_bytes: int
    inner_precise: int
    zero_offsets: bool
    fused_enabled: bool
    fused_supported: bool
    fused_available: bool
    fused_op: str | None
    fused_reason: str
    strategy: str


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = _cannoe_env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as err:
        raise RuntimeError(f"{name} must be an integer; got `{raw}`.") from err
    return value if value > 0 else default


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _cannoe_prefetch_enabled() -> bool:
    return _cannoe_env_flag(_CANNOE_PREFETCH_ENV, default=False)


def _cannoe_fused_enabled() -> bool:
    return _cannoe_env_flag(_CANNOE_FUSED_ENV, default=True)


def _cannoe_fused_required() -> bool:
    return _cannoe_env_flag(_CANNOE_FUSED_REQUIRE_ENV, default=False)


def _cannoe_v3_enabled() -> bool:
    return _cannoe_env_flag(_CANNOE_V3_ENV, default=False)


def _cannoe_staged_dequant_enabled() -> bool:
    return _cannoe_env_flag(_CANNOE_STAGED_DEQUANT_ENV, default=False)


def _cannoe_cube_consumer_enabled() -> bool:
    return _cannoe_env_flag(_CANNOE_CUBE_CONSUMER_ENV, default=False)


def _cannoe_ascendc_enabled() -> bool:
    return _cannoe_env_flag(_CANNOE_ASCENDC_ENV, default=False)


def _cannoe_inner_precise(rows: int, in_features: int, out_features: int, group_size: int) -> int:
    raw = _cannoe_env(_CANNOE_INNER_PRECISE_ENV)
    if raw is None or raw.strip().lower() == "auto":
        if (
            rows <= 16
            and group_size == 32
            and in_features >= 4096
            and out_features >= in_features
            and out_features <= in_features * 2
        ):
            return 1
        return 0
    try:
        value = int(raw)
    except ValueError as err:
        raise RuntimeError(f"{_CANNOE_INNER_PRECISE_ENV} must be 0 or 1; got `{raw}`.") from err
    if value not in (0, 1):
        raise RuntimeError(f"{_CANNOE_INNER_PRECISE_ENV} must be 0 or 1; got `{raw}`.")
    return value


def _cannoe_fused_op_names() -> tuple[str, ...]:
    raw = _cannoe_env(_CANNOE_FUSED_OP_ENV)
    if raw is None:
        return _DEFAULT_FUSED_OP_NAMES
    names = tuple(name.strip() for name in raw.split(",") if name.strip())
    return names or _DEFAULT_FUSED_OP_NAMES


def _cannoe_prefetch_max_bytes(device: torch.device) -> int:
    raw = _cannoe_env(_CANNOE_PREFETCH_MAX_BYTES_ENV)
    if raw is not None:
        try:
            return int(raw)
        except ValueError as err:
            raise RuntimeError(f"{_CANNOE_PREFETCH_MAX_BYTES_ENV} must be an integer; got `{raw}`.") from err

    try:
        props = torch.npu.get_device_properties(device)
        l2_cache_size = int(getattr(props, "L2_cache_size", 0) or 0)
        cube_core_num = int(getattr(props, "cube_core_num", 0) or 0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        l2_cache_size = 0
        cube_core_num = 0

    if l2_cache_size > 0 and cube_core_num > 0:
        return max(4 * 1024 * 1024, min(l2_cache_size // cube_core_num * 2, 32 * 1024 * 1024))
    return 8 * 1024 * 1024


def _cannoe_prefetch_min_bytes() -> int:
    raw = _cannoe_env(_CANNOE_PREFETCH_MIN_BYTES_ENV)
    if raw is None:
        return 4 * 1024 * 1024
    try:
        return int(raw)
    except ValueError as err:
        raise RuntimeError(f"{_CANNOE_PREFETCH_MIN_BYTES_ENV} must be an integer; got `{raw}`.") from err


def _npu_prefetch_op():
    global _NPU_PREFETCH_OP
    if _NPU_PREFETCH_OP is not _NPU_PREFETCH_OP_UNSET:
        return _NPU_PREFETCH_OP
    try:
        _NPU_PREFETCH_OP = getattr(torch.ops.npu, "npu_prefetch", None)
    except (AttributeError, RuntimeError):
        _NPU_PREFETCH_OP = None
    return _NPU_PREFETCH_OP


def _npu_weight_quant_op():
    global _NPU_WEIGHT_QUANT_OP
    if _NPU_WEIGHT_QUANT_OP is _NPU_WEIGHT_QUANT_OP_UNSET:
        _NPU_WEIGHT_QUANT_OP = torch.ops.npu.npu_weight_quant_batchmatmul
    return _NPU_WEIGHT_QUANT_OP


def _resolve_torch_op(qualified_name: str):
    namespace, separator, op_name = qualified_name.rpartition(".")
    if not separator or not namespace or not op_name:
        return None

    target = torch.ops
    for part in namespace.split("."):
        try:
            target = getattr(target, part)
        except (AttributeError, RuntimeError):
            return None
    try:
        return getattr(target, op_name)
    except (AttributeError, RuntimeError):
        return None


def _try_load_cannoe_v3() -> bool:
    global _FUSED_OP_LAST_ERROR

    if not _cannoe_v3_enabled():
        return False
    try:
        from ...utils.cannoe import cannoe_v3_runtime_error, load_cannoe_v3

        loaded = load_cannoe_v3()
        _FUSED_OP_LAST_ERROR = "" if loaded else cannoe_v3_runtime_error()
        return loaded
    except Exception as exc:  # pragma: no cover - depends on local CANN toolchain/runtime
        _FUSED_OP_LAST_ERROR = str(exc)
        return False


def _try_load_cannoe_ascendc() -> bool:
    global _FUSED_OP_LAST_ERROR

    if not _cannoe_ascendc_enabled():
        return False
    try:
        from ...utils.cannoe import cannoe_ascendc_runtime_error, load_cannoe_ascendc

        loaded = load_cannoe_ascendc()
        _FUSED_OP_LAST_ERROR = "" if loaded else cannoe_ascendc_runtime_error()
        return loaded
    except Exception as exc:  # pragma: no cover - depends on local CANN toolchain/runtime
        _FUSED_OP_LAST_ERROR = str(exc)
        return False


def _cannoe_fused_op():
    global _FUSED_OP_CACHE, _FUSED_OP_CACHE_KEY

    names = _cannoe_fused_op_names()
    cache_key = (
        names,
        *_cannoe_env_values(
            _CANNOE_ASCENDC_ENV,
            "GPTQMODEL_CANNOE_ASCENDC_BUILD_ROOT",
            "GPTQMODEL_CANNOE_ASCENDC_FORCE_REBUILD",
            "GPTQMODEL_CANNOE_ASCENDC_OPAPI_LIB",
            _CANNOE_V3_ENV,
            "GPTQMODEL_CANNOE_V3_BUILD_ROOT",
            "GPTQMODEL_CANNOE_V3_FORCE_REBUILD",
        ),
    )
    if _FUSED_OP_CACHE is not _FUSED_OP_UNSET and _FUSED_OP_CACHE_KEY == cache_key:
        return _FUSED_OP_CACHE

    for name in names:
        op = _resolve_torch_op(name)
        if op is not None:
            _FUSED_OP_CACHE = (name, op)
            _FUSED_OP_CACHE_KEY = cache_key
            return _FUSED_OP_CACHE

    if _try_load_cannoe_ascendc():
        for name in names:
            op = _resolve_torch_op(name)
            if op is not None:
                _FUSED_OP_CACHE = (name, op)
                _FUSED_OP_CACHE_KEY = cache_key
                return _FUSED_OP_CACHE

    if _try_load_cannoe_v3():
        for name in names:
            op = _resolve_torch_op(name)
            if op is not None:
                _FUSED_OP_CACHE = (name, op)
                _FUSED_OP_CACHE_KEY = cache_key
                return _FUSED_OP_CACHE

    _FUSED_OP_CACHE = None
    _FUSED_OP_CACHE_KEY = cache_key
    return None


def _cannoe_fused_status(group_size: int) -> tuple[bool, bool, bool, str | None, str]:
    enabled = _cannoe_fused_enabled()
    supported = group_size == 0 or group_size >= 32
    if not enabled:
        return enabled, supported, False, None, "disabled"
    if not supported:
        return enabled, supported, False, None, "unsupported_group_size"

    resolved = _cannoe_fused_op()
    if resolved is None:
        if _cannoe_fused_required():
            names = ", ".join(_cannoe_fused_op_names())
            suffix = f" Last fused loader error: {_FUSED_OP_LAST_ERROR}" if _FUSED_OP_LAST_ERROR else ""
            raise RuntimeError(
                "Cannoe fused W4A16 op was required but no registered torch op was found. "
                f"Checked: {names}.{suffix}"
            )
        if _cannoe_ascendc_enabled():
            reason = "ascendc_extension_unavailable"
        elif _cannoe_v3_enabled():
            reason = "v3_extension_unavailable"
        else:
            reason = "op_not_registered"
        return enabled, supported, False, None, reason

    op_name, _ = resolved
    return enabled, supported, True, op_name, "available"


def _cannoe_device_caps(device: torch.device) -> tuple[int, int, int]:
    override_cores = _cannoe_env(_CANNOE_ACTIVE_CORES_ENV)
    try:
        props = torch.npu.get_device_properties(device)
        cube_cores = int(getattr(props, "cube_core_num", 0) or 0)
        vector_cores = int(getattr(props, "vector_core_num", 0) or 0)
        l2_cache_size = int(getattr(props, "L2_cache_size", 0) or 0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        cube_cores = 0
        vector_cores = 0
        l2_cache_size = 0

    if override_cores is not None:
        try:
            cube_cores = int(override_cores)
            vector_cores = cube_cores
        except ValueError as err:
            raise RuntimeError(f"{_CANNOE_ACTIVE_CORES_ENV} must be an integer; got `{override_cores}`.") from err

    return max(1, cube_cores), max(1, vector_cores), max(0, l2_cache_size)


def _cannoe_split_k(rows: int, in_features: int, out_features: int, cube_cores: int) -> int:
    raw = _cannoe_env(_CANNOE_SPLIT_K_ENV)
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError as err:
            raise RuntimeError(f"{_CANNOE_SPLIT_K_ENV} must be an integer; got `{raw}`.") from err

    min_ratio = _parse_positive_int_env(_CANNOE_MIN_SPLIT_K_RATIO_ENV, 2)
    if rows > 16 or in_features < 4096 or in_features < out_features * min_ratio:
        return 1

    # INT4 Cube K tiles need C0=64 alignment. Keep each Split-K shard large
    # enough to amortize the AIV dequant workspace round-trip described in the
    # W4A16 Ascend paper.
    max_by_k = max(1, in_features // 2048)
    split_k = min(cube_cores, max_by_k)
    while split_k > 1 and in_features % (split_k * 64) != 0:
        split_k -= 1
    return max(1, split_k)


def _cannoe_base_k(rows: int, in_features: int, cube_consumer_requested: bool) -> int:
    raw = _cannoe_env(_CANNOE_BASE_K_ENV)
    if raw is not None:
        try:
            value = int(raw)
        except ValueError as err:
            raise RuntimeError(f"{_CANNOE_BASE_K_ENV} must be an integer; got `{raw}`.") from err
        if value <= 0 or value % 64 != 0:
            raise RuntimeError(f"{_CANNOE_BASE_K_ENV} must be a positive multiple of 64; got `{raw}`.")
        if in_features % value != 0:
            raise RuntimeError(f"{_CANNOE_BASE_K_ENV} must divide K={in_features}; got `{raw}`.")
        return value

    if cube_consumer_requested and in_features % 128 == 0:
        return 128
    return 64


def _cannoe_base_n(rows: int, in_features: int, out_features: int, cube_consumer_requested: bool) -> int:
    raw = _cannoe_env(_CANNOE_BASE_N_ENV)
    if raw is not None:
        try:
            value = int(raw)
        except ValueError as err:
            raise RuntimeError(f"{_CANNOE_BASE_N_ENV} must be an integer; got `{raw}`.") from err
        max_base_n = _align_up(out_features, 16)
        if value <= 0 or value % 16 != 0:
            raise RuntimeError(f"{_CANNOE_BASE_N_ENV} must be a positive multiple of 16; got `{raw}`.")
        if value > max_base_n:
            raise RuntimeError(
                f"{_CANNOE_BASE_N_ENV} must be <= aligned N={max_base_n}; got `{raw}`."
            )
        return value

    base_n = min(256, _align_up(out_features, 16))
    if cube_consumer_requested and (
        out_features == 256
        or (out_features == 640 and in_features in {512, 768})
        or (out_features in {512, 768} and in_features in {512, 768, 896})
        or (rows <= 8 and in_features == 512 and out_features == 1024)
    ):
        return 128
    return base_n


def _cannoe_tiling_plan(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    group_size: int,
    device: torch.device,
    zero_offsets: bool = False,
) -> CannoeTilingPlan:
    cube_cores, vector_cores, l2_cache_size = _cannoe_device_caps(device)
    split_k = _cannoe_split_k(rows, in_features, out_features, cube_cores)
    cube_consumer_requested = _cannoe_staged_dequant_enabled() and _cannoe_cube_consumer_enabled()
    base_m = 16 if rows <= 16 else min(128, _align_up(rows, 16))
    base_n = _cannoe_base_n(rows, in_features, out_features, cube_consumer_requested)
    base_k = _cannoe_base_k(rows, in_features, cube_consumer_requested)
    n_tiles = _ceil_div(out_features, base_n)
    split_k_shard_k = _ceil_div(in_features, split_k)
    k_tiles_per_split = _ceil_div(split_k_shard_k, base_k)
    vector_dequant_tasks = n_tiles * split_k * k_tiles_per_split
    int4_values_per_int32 = 8
    packed_int4_tile_bytes = base_k * base_n // 2
    dequant_fp16_tile_bytes = base_k * base_n * 2
    l0a_tile_bytes = base_m * base_k * 2
    l0b_tile_bytes = dequant_fp16_tile_bytes
    l0c_tile_bytes = base_m * base_n * 4
    staging_slots = 2
    staging_tile_bytes = _align_up(dequant_fp16_tile_bytes, 512)
    scalar_owner_cap = min(vector_cores, max(1, out_features // 8))
    staging_blocks = min(scalar_owner_cap, max(1, n_tiles * split_k))
    staging_workspace_bytes = staging_tile_bytes * staging_slots * staging_blocks
    cube_workspace_bytes = _CANNOE_CUBE_WORKSPACE_BYTES if cube_consumer_requested else 0
    custom_workspace_bytes = staging_workspace_bytes
    dense_dequant_bytes = in_features * out_features * 2
    staged_dequant = (
        _cannoe_staged_dequant_enabled()
        and staging_workspace_bytes > 0
        and custom_workspace_bytes < dense_dequant_bytes
    )
    if not staged_dequant:
        staging_slots = 0
        staging_blocks = 0
        staging_tile_bytes = 0
        staging_workspace_bytes = 0
        cube_workspace_bytes = 0
        custom_workspace_bytes = 0
    staging_workspace_offset = 0
    cube_consumer = bool(cube_consumer_requested and staged_dequant)
    active_cores = min(cube_cores, max(1, n_tiles * split_k))
    prefetch_enabled = _cannoe_prefetch_enabled()
    prefetch_min_bytes = _cannoe_prefetch_min_bytes()
    prefetch_max_bytes = _cannoe_prefetch_max_bytes(device)
    inner_precise = _cannoe_inner_precise(rows, in_features, out_features, group_size)
    fused_enabled, fused_supported, fused_available, fused_op, fused_reason = _cannoe_fused_status(group_size)
    if fused_available:
        strategy = "fused_w4a16_staged_dequant_aic_matmul" if staged_dequant else "fused_w4a16_aiv_dequant_aic_matmul"
    elif staged_dequant:
        strategy = "planned_staged_dequant_aic_matmul"
    elif split_k > 1:
        strategy = "planned_split_k_aiv_dequant_aic_matmul"
    elif prefetch_enabled:
        strategy = "native_quant_matmul_prefetch"
    else:
        strategy = "native_quant_matmul"
    return CannoeTilingPlan(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        cube_cores=cube_cores,
        vector_cores=vector_cores,
        active_cores=active_cores,
        split_k=split_k,
        split_k_shard_k=split_k_shard_k,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        k_tiles_per_split=k_tiles_per_split,
        vector_dequant_tasks=vector_dequant_tasks,
        int4_values_per_int32=int4_values_per_int32,
        packed_int4_tile_bytes=packed_int4_tile_bytes,
        dequant_fp16_tile_bytes=dequant_fp16_tile_bytes,
        l0a_tile_bytes=l0a_tile_bytes,
        l0b_tile_bytes=l0b_tile_bytes,
        l0c_tile_bytes=l0c_tile_bytes,
        staged_dequant=staged_dequant,
        staging_slots=staging_slots,
        staging_blocks=staging_blocks,
        staging_tile_bytes=staging_tile_bytes,
        staging_workspace_bytes=staging_workspace_bytes,
        staging_workspace_offset=staging_workspace_offset,
        cube_consumer=cube_consumer,
        cube_workspace_bytes=cube_workspace_bytes,
        custom_workspace_bytes=custom_workspace_bytes,
        l2_cache_size=l2_cache_size,
        prefetch_enabled=prefetch_enabled,
        prefetch_min_bytes=prefetch_min_bytes,
        prefetch_max_bytes=prefetch_max_bytes,
        inner_precise=inner_precise,
        zero_offsets=bool(zero_offsets),
        fused_enabled=fused_enabled,
        fused_supported=fused_supported,
        fused_available=fused_available,
        fused_op=fused_op,
        fused_reason=fused_reason,
        strategy=strategy,
    )


def cannoe_plan_asdict(plan: CannoeTilingPlan | None) -> dict | None:
    if plan is None:
        return None
    return asdict(plan)


def _cannoe_prefetch(plan: CannoeTilingPlan, *tensors: torch.Tensor | None) -> None:
    prefetch_op = _npu_prefetch_op()
    if prefetch_op is None:
        return

    min_bytes = plan.prefetch_min_bytes
    max_bytes = plan.prefetch_max_bytes
    if max_bytes <= 0:
        return

    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "npu" and tensor.numel() > 0:
            size_bytes = tensor.numel() * tensor.element_size()
            if size_bytes >= min_bytes:
                prefetch_op(tensor, None, min(size_bytes, max_bytes), 0)


def _cannoe_fused_matmul(
    *,
    plan: CannoeTilingPlan,
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if not plan.fused_enabled or not plan.fused_supported or not plan.fused_available:
        return None
    resolved = _cannoe_fused_op()
    if resolved is None:
        if _cannoe_fused_required():
            raise RuntimeError("Cannoe fused W4A16 op disappeared after planning.")
        return None

    _, op = resolved
    return op(
        x,
        packed_weight,
        scales,
        offsets,
        bias,
        int(group_size),
        -int(plan.split_k) if plan.cube_consumer else int(plan.split_k),
        int(plan.base_m),
        -int(plan.base_n) if plan.staged_dequant else int(plan.base_n),
        -int(plan.base_k) if plan.zero_offsets else int(plan.base_k),
    )


def _cannoe_weight_quant_matmul(
    *,
    plan: CannoeTilingPlan,
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    op = _npu_weight_quant_op()
    if plan.inner_precise == 0:
        return op(
            x,
            weight,
            scales,
            offsets,
            None,
            None,
            bias,
            group_size,
        )
    return op(
        x,
        weight,
        scales,
        offsets,
        None,
        None,
        bias,
        group_size,
        int(plan.inner_precise),
    )


class _CannoePlanMixin:
    _cann_plan_cache: dict

    def clear_native_cache(self):
        result = super().clear_native_cache()
        if hasattr(self, "_cann_plan_cache"):
            self._cann_plan_cache.clear()
            self._cann_hot_plan_fast_key = None
            self._cann_hot_plan_key = None
            self._cann_hot_plan = None
            self._cann_native_hot_device = None
            self._cann_native_hot_plan = None
        return result

    def _cann_plan(
        self,
        x_flat: torch.Tensor,
        group_size: int,
        zero_offsets: bool = False,
    ) -> CannoeTilingPlan:
        fast_key = (x_flat.device, x_flat.shape[0], group_size, bool(zero_offsets))
        if getattr(self, "_cann_hot_plan_fast_key", None) == fast_key:
            return self._cann_hot_plan

        env_key = (
            *_cannoe_env_values(
                _CANNOE_PREFETCH_ENV,
                _CANNOE_PREFETCH_MAX_BYTES_ENV,
                _CANNOE_PREFETCH_MIN_BYTES_ENV,
                _CANNOE_ACTIVE_CORES_ENV,
                _CANNOE_SPLIT_K_ENV,
                _CANNOE_MIN_SPLIT_K_RATIO_ENV,
                _CANNOE_BASE_N_ENV,
                _CANNOE_BASE_K_ENV,
                _CANNOE_FUSED_ENV,
                _CANNOE_FUSED_REQUIRE_ENV,
                _CANNOE_FUSED_OP_ENV,
                _CANNOE_ASCENDC_ENV,
                _CANNOE_V3_ENV,
                _CANNOE_INNER_PRECISE_ENV,
                _CANNOE_STAGED_DEQUANT_ENV,
                _CANNOE_CUBE_CONSUMER_ENV,
            ),
        )
        hot_key = (x_flat.device, x_flat.shape[0], group_size, bool(zero_offsets), env_key)
        if getattr(self, "_cann_hot_plan_key", None) == hot_key:
            self._last_cann_plan = self._cann_hot_plan
            return self._cann_hot_plan

        key = (
            hot_key[0],
            hot_key[1],
            self.in_features,
            self.out_features,
            group_size,
            bool(zero_offsets),
            env_key,
        )
        cached = self._cann_plan_cache.get(key)
        if cached is None:
            cached = _cannoe_tiling_plan(
                rows=x_flat.shape[0],
                in_features=self.in_features,
                out_features=self.out_features,
                group_size=group_size,
                device=x_flat.device,
                zero_offsets=zero_offsets,
            )
            self._cann_plan_cache[key] = cached
        self._cann_hot_plan_fast_key = fast_key
        self._cann_hot_plan_key = hot_key
        self._cann_hot_plan = cached
        self._last_cann_plan = cached
        return cached


class CannoeLinear(_CannoePlanMixin, KomodoLinear):
    """Cannoe Ascend CANN experiment based on Komodo's packed int4 plan."""

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_CANNOE]
    SUPPORTS_METHODS = KomodoLinear.SUPPORTS_METHODS
    SUPPORTS_FORMATS = {fmt: 0 for fmt in KomodoLinear.SUPPORTS_FORMATS}
    SUPPORTS_BITS = KomodoLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = KomodoLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = KomodoLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = KomodoLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = KomodoLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = KomodoLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = KomodoLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = KomodoLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = KomodoLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = KomodoLinear.SUPPORTS_DEVICES
    SUPPORTS_PLATFORM = KomodoLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = KomodoLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = KomodoLinear.SUPPORTS_ADAPTERS
    SUPPORTS_DTYPES = KomodoLinear.SUPPORTS_DTYPES
    REQUIRES_FORMAT_V2 = KomodoLinear.REQUIRES_FORMAT_V2
    QUANT_TYPE = "cannoe"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.GPTQ_CANNOE)
        super().__init__(*args, **kwargs)
        self._cann_plan_cache: dict = {}
        self._cann_hot_plan_fast_key = None
        self._cann_hot_plan_key = None
        self._cann_hot_plan = None
        self._cann_native_hot_device = None
        self._cann_native_hot_plan = None
        self._last_cann_plan: CannoeTilingPlan | None = None
        self._last_cann_path: str | None = None

    def _native_forward(self, x: torch.Tensor):
        _assert_fp16_inference_input(x, self.__class__.__name__)
        input_dtype = x.dtype
        compute_dtype = torch.float16
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.dtype != compute_dtype or not x_flat.is_contiguous():
            x_flat = x_flat.to(dtype=compute_dtype).contiguous()

        packed_weight, scales, offsets, native_group_size, input_perm = self._native_plan(
            device=x_flat.device, dtype=compute_dtype
        )
        zero_offsets = bool(self.sym)
        plan = self._cann_native_hot_plan
        if (
            plan is None
            or self._cann_native_hot_device != x_flat.device
            or plan.rows != x_flat.shape[0]
            or plan.group_size != native_group_size
            or plan.zero_offsets != zero_offsets
        ):
            plan = self._cann_plan(
                x_flat,
                native_group_size,
                zero_offsets=zero_offsets,
            )
            if not plan.fused_available and not plan.prefetch_enabled:
                self._cann_native_hot_device = x_flat.device
                self._cann_native_hot_plan = plan
            else:
                self._cann_native_hot_device = None
                self._cann_native_hot_plan = None
        if input_perm is not None:
            x_flat = x_flat.index_select(1, input_perm)
        bias = self.bias
        fuse_bias = bias is not None and _fuse_bias_enabled()
        if fuse_bias and (bias.device != x_flat.device or bias.dtype != x_flat.dtype):
            bias = bias.to(device=x_flat.device, dtype=x_flat.dtype)
        self._maybe_schedule_lookahead(compute_dtype)
        out = None
        if plan.fused_available:
            out = _cannoe_fused_matmul(
                plan=plan,
                x=x_flat,
                packed_weight=packed_weight,
                scales=scales,
                offsets=offsets,
                group_size=native_group_size,
                bias=bias if fuse_bias else None,
            )
        if out is None:
            self._last_cann_path = "native_weight_quant_batchmatmul"
            if plan.prefetch_enabled:
                _cannoe_prefetch(plan, x_flat, packed_weight, scales, offsets, bias if fuse_bias else None)
            matmul_op = _npu_weight_quant_op()
            if plan.inner_precise == 0:
                out = matmul_op(
                    x_flat,
                    packed_weight,
                    scales,
                    offsets,
                    None,
                    None,
                    bias if fuse_bias else None,
                    native_group_size,
                )
            else:
                out = matmul_op(
                    x_flat,
                    packed_weight,
                    scales,
                    offsets,
                    None,
                    None,
                    bias if fuse_bias else None,
                    native_group_size,
                    int(plan.inner_precise),
                )
        else:
            self._last_cann_path = "fused_w4a16_matmul"
        out = out.reshape(out_shape)

        if self.bias is not None and not fuse_bias:
            bias = self.bias
            if bias.device != out.device or bias.dtype != out.dtype:
                bias = bias.to(device=out.device, dtype=out.dtype)
            out.add_(bias)
        if self.adapter:
            out = self.adapter.apply(x=x_flat, out=out)
        if input_dtype == torch.float32:
            out = out.to(torch.float32)
        return out

    def _native_group16_forward(self, x: torch.Tensor):
        _assert_fp16_inference_input(x, self.__class__.__name__)
        input_dtype = x.dtype
        compute_dtype = torch.float16
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.dtype != compute_dtype or not x_flat.is_contiguous():
            x_flat = x_flat.to(dtype=compute_dtype).contiguous()

        (
            packed_groups,
            scale_groups,
            offset_groups,
            packed_stack,
            scale_stack,
            offset_stack,
            bias_stack,
            input_perm,
        ) = self._native_group16_plan(
            device=x_flat.device, dtype=compute_dtype
        )
        plan = self._cann_plan(x_flat, 16)
        if input_perm is not None:
            x_flat = x_flat.index_select(1, input_perm)

        group_size = 16
        group_count = len(packed_groups)
        rows = x_flat.shape[0]
        fused_bias = False
        self._last_cann_path = "native_group16_weight_quant_batchmatmul"
        if plan.prefetch_enabled:
            _cannoe_prefetch(plan, x_flat, packed_stack, scale_stack, offset_stack, bias_stack)
        self._maybe_schedule_lookahead(compute_dtype)
        if self._can_use_native_group16_grouped(rows=rows, group_count=group_count):
            self._native_group16_last_path = "grouped"
            x_groups = x_flat.reshape(rows, group_count, group_size).transpose(0, 1).contiguous()
            group_list = self._native_group16_group_list(
                device=x_flat.device,
                rows=rows,
                group_count=group_count,
            )
            fused_bias = bias_stack is not None and _fuse_bias_enabled()
            out_groups = torch.ops.npu.npu_grouped_matmul(
                [x_groups.reshape(rows * group_count, group_size)],
                [packed_stack],
                bias=[bias_stack] if fused_bias else None,
                antiquant_scale=[scale_stack],
                antiquant_offset=[offset_stack],
                group_list=group_list,
                split_item=2,
                group_type=0,
                group_list_type=1,
            )[0]
            out = out_groups.reshape(group_count, rows, self.out_features).sum(0).reshape(out_shape)
        elif rows == 1:
            self._native_group16_last_path = "loop_narrow"
            out_acc = None
            for group_idx in range(group_count):
                x_group = x_flat.narrow(1, group_idx * group_size, group_size)
                if not x_group.is_contiguous():
                    x_group = x_group.contiguous()
                partial = _weight_quant_matmul(
                    x_group,
                    packed_groups[group_idx],
                    scale_groups[group_idx],
                    offset_groups[group_idx],
                    0,
                ).to(torch.float32)
                out_acc = partial if out_acc is None else out_acc.add_(partial)
            out = out_acc.to(dtype=compute_dtype).reshape(out_shape)
        else:
            self._native_group16_last_path = "loop"
            out_acc = None
            x_groups = x_flat.reshape(x_flat.shape[0], group_count, group_size).transpose(0, 1).contiguous()
            for group_idx in range(group_count):
                partial = _weight_quant_matmul(
                    x_groups[group_idx],
                    packed_groups[group_idx],
                    scale_groups[group_idx],
                    offset_groups[group_idx],
                    0,
                ).to(torch.float32)
                out_acc = partial if out_acc is None else out_acc.add_(partial)
            out = out_acc.to(dtype=compute_dtype).reshape(out_shape)

        if self.bias is not None and not fused_bias:
            bias = self.bias
            if bias.device != out.device or bias.dtype != out.dtype:
                bias = bias.to(device=out.device, dtype=out.dtype)
            out.add_(bias)
        if self.adapter:
            out = self.adapter.apply(x=x_flat, out=out)
        if input_dtype == torch.float32:
            out = out.to(torch.float32)
        return out


class AwqCannoeLinear(_CannoePlanMixin, AwqKomodoLinear):
    """AWQ Cannoe Ascend CANN experiment based on Komodo's packed int4 plan."""

    SUPPORTS_BACKENDS = [BACKEND.AWQ_CANNOE]
    SUPPORTS_METHODS = AwqKomodoLinear.SUPPORTS_METHODS
    SUPPORTS_FORMATS = {fmt: 0 for fmt in AwqKomodoLinear.SUPPORTS_FORMATS}
    SUPPORTS_BITS = AwqKomodoLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = AwqKomodoLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = AwqKomodoLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = AwqKomodoLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = AwqKomodoLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = AwqKomodoLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = AwqKomodoLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = AwqKomodoLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = AwqKomodoLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = AwqKomodoLinear.SUPPORTS_DEVICES
    SUPPORTS_PLATFORM = AwqKomodoLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = AwqKomodoLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = AwqKomodoLinear.SUPPORTS_ADAPTERS
    SUPPORTS_DTYPES = AwqKomodoLinear.SUPPORTS_DTYPES
    REQUIRES_FORMAT_V2 = AwqKomodoLinear.REQUIRES_FORMAT_V2
    QUANT_TYPE = "awq_cannoe"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.AWQ_CANNOE)
        super().__init__(*args, **kwargs)
        self._cann_plan_cache: dict = {}
        self._cann_hot_plan_fast_key = None
        self._cann_hot_plan_key = None
        self._cann_hot_plan = None
        self._cann_native_hot_device = None
        self._cann_native_hot_plan = None
        self._last_cann_plan: CannoeTilingPlan | None = None
        self._last_cann_path: str | None = None

    def _native_forward(self, x: torch.Tensor):
        _assert_fp16_inference_input(x, self.__class__.__name__)
        input_dtype = x.dtype
        compute_dtype = torch.float16
        original_shape = x.shape[:-1] + (self.out_features,)
        device = x.device
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.dtype != compute_dtype or x_flat.device != device:
            x_flat = x_flat.to(device=device, dtype=compute_dtype)
        elif not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        packed_weight, scales, offsets, native_group_size, _ = self._native_plan(device=device, dtype=compute_dtype)
        plan = self._cann_native_hot_plan
        if (
            plan is None
            or self._cann_native_hot_device != x_flat.device
            or plan.rows != x_flat.shape[0]
            or plan.group_size != native_group_size
            or plan.zero_offsets
        ):
            plan = self._cann_plan(x_flat, native_group_size)
            if not plan.fused_available and not plan.prefetch_enabled:
                self._cann_native_hot_device = x_flat.device
                self._cann_native_hot_plan = plan
            else:
                self._cann_native_hot_device = None
                self._cann_native_hot_plan = None
        output = None
        if plan.fused_available:
            output = _cannoe_fused_matmul(
                plan=plan,
                x=x_flat,
                packed_weight=packed_weight,
                scales=scales,
                offsets=offsets,
                group_size=native_group_size,
                bias=None,
            )
        if output is None:
            self._last_cann_path = "native_weight_quant_batchmatmul"
            if plan.prefetch_enabled:
                _cannoe_prefetch(plan, x_flat, packed_weight, scales, offsets)
            matmul_op = _npu_weight_quant_op()
            if plan.inner_precise == 0:
                output = matmul_op(
                    x_flat,
                    packed_weight,
                    scales,
                    offsets,
                    None,
                    None,
                    None,
                    native_group_size,
                )
            else:
                output = matmul_op(
                    x_flat,
                    packed_weight,
                    scales,
                    offsets,
                    None,
                    None,
                    None,
                    native_group_size,
                    int(plan.inner_precise),
                )
        else:
            self._last_cann_path = "fused_w4a16_matmul"

        if self.bias is not None:
            bias = self.bias
            if bias.device != output.device or bias.dtype != output.dtype:
                bias = bias.to(device=output.device, dtype=output.dtype)
            output = output + bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        if output.dtype != input_dtype:
            output = output.to(dtype=input_dtype)

        self._maybe_schedule_lookahead(compute_dtype)
        return output.reshape(original_shape)

__all__ = [
    "AwqCannoeLinear",
    "CannoeLinear",
    "CannoeTilingPlan",
    "cannoe_plan_asdict",
]
