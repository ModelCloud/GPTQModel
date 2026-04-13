# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)
from .logger import setup_logger
from .marlin import (
    _marlin_cuda_version_at_least,
    _marlin_header_install_hint,
    _transform_param,
    awq_to_marlin_zero_points,
    marlin_make_empty_g_idx,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks,
    marlin_sort_g_idx,
    marlin_zero_points,
    pack_cols,
    replace_parameter,
    should_use_atomic_add_reduce,
    unpack_cols,
)
from .marlin_scalar_type import ScalarType
from .rocm import IS_ROCM


log = setup_logger()

_MENTARAY_FP16_OPS_NAME = "gptqmodel_mentaray_fp16_ops"
_MENTARAY_FP16_NAMESPACE = "gptqmodel_mentaray_fp16"
_MENTARAY_BF16_OPS_NAME = "gptqmodel_mentaray_bf16_ops"
_MENTARAY_BF16_NAMESPACE = "gptqmodel_mentaray_bf16"
_MENTARAY_REQUIRED_CUDA_HEADERS = (
    "cuda_runtime_api.h",
    "cusparse.h",
    "cublas_v2.h",
    "cublasLt.h",
    "cusolverDn.h",
)
_MENTARAY_FULL_A100_MIN_SMS = 120


def _mentaray_capability_supported(major: int, minor: int) -> bool:
    return major == 8 and minor == 0


def _mentaray_environment_error() -> str:
    if IS_ROCM:
        return "MentaRay kernel is not supported on ROCm."
    if not torch.cuda.is_available():
        return "MentaRay kernel requires CUDA."
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:  # pragma: no cover - depends on host CUDA runtime
        return f"MentaRay kernel failed to query CUDA device capability: {exc}"
    if not _mentaray_capability_supported(major, minor):
        return (
            "MentaRay kernel is specialized for Ampere sm80 GPUs. "
            f"Detected capability: {major}.{minor}."
        )
    return ""


mentaray_import_exception = _mentaray_environment_error() or None


def _mentaray_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "mentaray"


def _ensure_generated_mentaray_kernels() -> Path:
    root = _mentaray_root()
    generator = root / "generate_kernels.py"
    check_result = subprocess.run(
        [sys.executable, str(generator), "--check"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if check_result.returncode == 0:
        return root

    result = subprocess.run(
        [sys.executable, str(generator)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or check_result.stderr or check_result.stdout or "").strip()
        raise RuntimeError(
            "MentaRay kernel generation failed"
            + (f": {details}" if details else ".")
        )
    return root


def _mentaray_sources(dtype_tag: str) -> list[str]:
    root = _ensure_generated_mentaray_kernels()
    sources = [
        str(root / f"marlin_torch_{dtype_tag}.cpp"),
        str(root / f"gptq_marlin_{dtype_tag}.cu"),
        str(root / "gptq_marlin_repack.cu"),
        str(root / "awq_marlin_repack.cu"),
    ]
    sources.extend(str(path) for path in sorted(root.glob(f"kernel_{dtype_tag}_*.cu")))
    if len(sources) <= 4:
        raise RuntimeError(f"MentaRay {dtype_tag} sources are incomplete under `{root}`.")
    return sources


def _mentaray_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_mentaray_root())],
        required_header_names=_MENTARAY_REQUIRED_CUDA_HEADERS,
    )


def _mentaray_extra_cflags() -> list[str]:
    return default_jit_cflags(enable_bf16=True)


def _mentaray_extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )
    flags.insert(0, "-DMARLIN_NAMESPACE_NAME=mentaray")
    if _marlin_cuda_version_at_least(12, 8):
        flags.insert(0, "-static-global-template-stub=false")
    return flags


_MENTARAY_FP16_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_MENTARAY_FP16_OPS_NAME,
    namespace=_MENTARAY_FP16_NAMESPACE,
    required_ops=("gptq_mentaray_gemm_fp16", "gptq_mentaray_repack", "awq_mentaray_repack"),
    sources=lambda: _mentaray_sources("fp16"),
    build_root_env="GPTQMODEL_MENTARAY_FP16_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("mentaray_fp16"),
    display_name="MentaRay fp16",
    extra_cflags=_mentaray_extra_cflags,
    extra_cuda_cflags=_mentaray_extra_cuda_cflags,
    extra_include_paths=_mentaray_include_paths,
    force_rebuild_env="GPTQMODEL_MENTARAY_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


_MENTARAY_BF16_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_MENTARAY_BF16_OPS_NAME,
    namespace=_MENTARAY_BF16_NAMESPACE,
    required_ops=("gptq_mentaray_gemm_bf16", "gptq_mentaray_repack", "awq_mentaray_repack"),
    sources=lambda: _mentaray_sources("bf16"),
    build_root_env="GPTQMODEL_MENTARAY_BF16_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("mentaray_bf16"),
    display_name="MentaRay bf16",
    extra_cflags=_mentaray_extra_cflags,
    extra_cuda_cflags=_mentaray_extra_cuda_cflags,
    extra_include_paths=_mentaray_include_paths,
    force_rebuild_env="GPTQMODEL_MENTARAY_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def _mentaray_runtime_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    return torch.bfloat16 if dtype == torch.bfloat16 else torch.float16


def _mentaray_kernel_name_for_dtype(dtype: Optional[torch.dtype]) -> str:
    return "mentaray_bf16" if _mentaray_runtime_dtype(dtype) == torch.bfloat16 else "mentaray_fp16"


def clear_mentaray_extension_cache() -> None:
    _MENTARAY_FP16_TORCH_OPS_EXTENSION.clear_cache()
    _MENTARAY_BF16_TORCH_OPS_EXTENSION.clear_cache()


def mentaray_runtime_available(dtype: Optional[torch.dtype] = None) -> bool:
    if mentaray_import_exception is not None:
        return False
    return _extension_api().is_available(_mentaray_kernel_name_for_dtype(dtype))


def mentaray_runtime_error(dtype: Optional[torch.dtype] = None) -> str:
    if mentaray_import_exception is not None:
        return mentaray_import_exception

    extension_name = _mentaray_kernel_name_for_dtype(dtype)
    extension_api = _extension_api()
    if extension_api.is_available(extension_name):
        return ""
    error_text = extension_api.error(extension_name) or "MentaRay runtime unavailable."
    install_hint = _marlin_header_install_hint(error_text).replace("Marlin", "MentaRay")
    if install_hint:
        return f"{error_text} {install_hint}"
    return error_text


def prewarm_mentaray_extension(dtype: Optional[torch.dtype]) -> bool:
    extension_name = _mentaray_kernel_name_for_dtype(dtype)
    return _extension_api().load(name=extension_name)[extension_name]


def _mentaray_resolve_op(*, dtype: Optional[torch.dtype], op_name: str):
    return _extension_api().op(_mentaray_kernel_name_for_dtype(dtype), op_name)


def _validate_mentaray_device_support() -> bool:
    if IS_ROCM or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return _mentaray_capability_supported(major, minor)


def _mentaray_default_blocks_per_sm(device: torch.device) -> int:
    env_raw = os.getenv("GPTQMODEL_MENTARAY_MAX_BLOCKS_PER_SM")
    if env_raw:
        try:
            parsed = int(env_raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    return 1


def mentaray_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def mentaray_repeat_scales_on_all_ranks(act_order: bool, group_size: int, is_row_parallel: bool) -> bool:
    return marlin_repeat_scales_on_all_ranks(act_order, group_size, is_row_parallel)


def mentaray_make_workspace_new(device: torch.device, max_blocks_per_sm: int | None = None) -> torch.Tensor:
    blocks_per_sm = max_blocks_per_sm or _mentaray_default_blocks_per_sm(device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace_blocks = max(sms * blocks_per_sm, 128)
    return torch.zeros(workspace_blocks, dtype=torch.int, device=device, requires_grad=False)


def mentaray_sort_g_idx(g_idx: torch.Tensor):
    return marlin_sort_g_idx(g_idx)


def mentaray_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return marlin_make_empty_g_idx(device)


def mentaray_permute_scales(s: torch.Tensor, size_k: int, size_n: int, group_size: int) -> torch.Tensor:
    return marlin_permute_scales(s, size_k=size_k, size_n=size_n, group_size=group_size)


def mentaray_zero_points(zp: torch.Tensor, size_k: int, size_n: int, num_bits: int) -> torch.Tensor:
    return marlin_zero_points(zp, size_k=size_k, size_n=size_n, num_bits=num_bits)


def awq_to_mentaray_zero_points(q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int) -> torch.Tensor:
    return awq_to_marlin_zero_points(q_zp_packed, size_k=size_k, size_n=size_n, num_bits=num_bits)


def mentaray_permute_bias(s: torch.Tensor) -> torch.Tensor:
    return marlin_permute_bias(s)


def apply_gptq_mentaray_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = True,
    use_atomics: bool = False,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)
    use_atomics = use_atomics and should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )
    output = gptq_mentaray_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        wtype,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=is_k_full,
        use_atomic_add=use_atomics,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )
    return output.reshape(out_shape)


def apply_awq_mentaray_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    quant_type: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = True,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)
    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )
    output = gptq_mentaray_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        quant_type,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )
    return output.reshape(out_shape)


def gptq_mentaray_gemm(
    a: torch.Tensor,
    c: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    op_name = "gptq_mentaray_gemm_bf16" if _mentaray_runtime_dtype(a.dtype) == torch.bfloat16 else "gptq_mentaray_gemm_fp16"
    op = _mentaray_resolve_op(dtype=a.dtype, op_name=op_name)
    return op(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


def gptq_mentaray_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    op = _mentaray_resolve_op(dtype=dtype, op_name="gptq_mentaray_repack")
    return op(b_q_weight, perm, size_k, size_n, num_bits)


def awq_mentaray_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    op = _mentaray_resolve_op(dtype=dtype, op_name="awq_mentaray_repack")
    return op(b_q_weight, size_k, size_n, num_bits)


__all__ = [
    "_mentaray_capability_supported",
    "_validate_mentaray_device_support",
    "_transform_param",
    "ScalarType",
    "apply_awq_mentaray_linear",
    "apply_gptq_mentaray_linear",
    "awq_mentaray_repack",
    "awq_to_mentaray_zero_points",
    "clear_mentaray_extension_cache",
    "gptq_mentaray_gemm",
    "gptq_mentaray_repack",
    "mentaray_import_exception",
    "mentaray_is_k_full",
    "mentaray_make_empty_g_idx",
    "mentaray_make_workspace_new",
    "mentaray_permute_bias",
    "mentaray_permute_scales",
    "mentaray_repeat_scales_on_all_ranks",
    "mentaray_runtime_available",
    "mentaray_runtime_error",
    "mentaray_sort_g_idx",
    "mentaray_zero_points",
    "pack_cols",
    "prewarm_mentaray_extension",
    "replace_parameter",
    "unpack_cols",
]
