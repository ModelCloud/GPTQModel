# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Callable, List, Optional, Tuple, Union

import numpy
import torch

from ..utils.logger import setup_logger
from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    detected_cuda_wheel_include_paths,
    local_nvcc_version_at_least,
)
from .marlin_scalar_type import ScalarType
from .rocm import IS_ROCM


log = setup_logger()

_MARLIN_FP16_OPS_NAME = "gptqmodel_marlin_fp16_ops"
_MARLIN_FP16_NAMESPACE = "gptqmodel_marlin_fp16"
_MARLIN_BF16_OPS_NAME = "gptqmodel_marlin_bf16_ops"
_MARLIN_BF16_NAMESPACE = "gptqmodel_marlin_bf16"
_MARLIN_REQUIRED_CUDA_HEADERS = (
    "cuda_runtime_api.h",
    "cusparse.h",
    "cublas_v2.h",
    "cublasLt.h",
    "cusolverDn.h",
)


def _marlin_capability_supported(major: int, minor: int) -> bool:
    return major > 7 or (major == 7 and minor >= 5)


def _marlin_environment_error() -> str:
    if IS_ROCM:
        return "Marlin kernel is not supported on ROCm."
    if not torch.cuda.is_available():
        return "Marlin kernel requires CUDA."
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:  # pragma: no cover - depends on host CUDA runtime
        return f"Marlin kernel failed to query CUDA device capability: {exc}"
    if not _marlin_capability_supported(major, minor):
        return f"Marlin kernel requires compute capability >= 7.5, got {major}.{minor}."
    return ""


marlin_import_exception = _marlin_environment_error() or None


def _marlin_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "marlin"


def _marlin_cuda_extra_name() -> str | None:
    raw = getattr(torch.version, "cuda", None)
    if not raw:
        return None
    try:
        major = int(str(raw).split(".", maxsplit=1)[0])
    except (TypeError, ValueError):  # pragma: no cover - depends on torch build metadata
        return None
    if major >= 13:
        return "marlin-cuda"
    if major == 12:
        return "marlin-cuda12"
    return None


def _marlin_missing_header_names(error_text: str) -> list[str]:
    return [
        header_name for header_name in _MARLIN_REQUIRED_CUDA_HEADERS
        if f"{header_name}: No such file or directory" in error_text
    ]


def _marlin_header_install_hint(error_text: str) -> str:
    missing_headers = _marlin_missing_header_names(error_text)
    if not missing_headers:
        return ""

    if detected_cuda_wheel_include_paths():
        return ""

    extra_name = _marlin_cuda_extra_name()
    missing_headers_text = ", ".join(missing_headers)
    if extra_name is not None:
        install_text = (
            f"Install the wheel-provided CUDA headers with "
            f"`pip install \"gptqmodel[{extra_name}]\"`."
        )
    else:
        install_text = (
            "Install the CUDA runtime/developer headers that match your Torch CUDA build."
        )

    nvcc_text = (
        "A local `nvcc` on PATH is still required for Marlin JIT."
        if which("nvcc")
        else "Marlin JIT also requires a local `nvcc` on PATH."
    )
    return (
        f"Missing CUDA developer headers for Marlin JIT ({missing_headers_text}). "
        f"{install_text} {nvcc_text}"
    )


def _ensure_generated_marlin_kernels() -> Path:
    root = _marlin_root()
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
            "Marlin kernel generation failed"
            + (f": {details}" if details else ".")
        )
    return root


def _marlin_sources(dtype_tag: str) -> list[str]:
    root = _ensure_generated_marlin_kernels()
    sources = [
        str(root / f"marlin_torch_{dtype_tag}.cpp"),
        str(root / f"gptq_marlin_{dtype_tag}.cu"),
        str(root / "gptq_marlin_repack.cu"),
        str(root / "awq_marlin_repack.cu"),
    ]
    sources.extend(str(path) for path in sorted(root.glob(f"kernel_{dtype_tag}_*.cu")))
    if len(sources) <= 4:
        raise RuntimeError(f"Marlin {dtype_tag} sources are incomplete under `{root}`.")
    return sources


def _marlin_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_marlin_root())],
        required_header_names=_MARLIN_REQUIRED_CUDA_HEADERS,
    )


def _marlin_extra_cflags() -> list[str]:
    return default_jit_cflags(enable_bf16=True)


def _marlin_extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )
    if local_nvcc_version_at_least(12, 8):
        flags.insert(0, "-static-global-template-stub=false")
    return flags


_MARLIN_FP16_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_MARLIN_FP16_OPS_NAME,
    namespace=_MARLIN_FP16_NAMESPACE,
    required_ops=("gptq_marlin_gemm_fp16", "gptq_marlin_repack", "awq_marlin_repack"),
    sources=lambda: _marlin_sources("fp16"),
    build_root_env="GPTQMODEL_MARLIN_FP16_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("marlin_fp16"),
    display_name="Marlin fp16",
    extra_cflags=_marlin_extra_cflags,
    extra_cuda_cflags=_marlin_extra_cuda_cflags,
    extra_include_paths=_marlin_include_paths,
    force_rebuild_env="GPTQMODEL_MARLIN_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


_MARLIN_BF16_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_MARLIN_BF16_OPS_NAME,
    namespace=_MARLIN_BF16_NAMESPACE,
    required_ops=("gptq_marlin_gemm_bf16", "gptq_marlin_repack", "awq_marlin_repack"),
    sources=lambda: _marlin_sources("bf16"),
    build_root_env="GPTQMODEL_MARLIN_BF16_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("marlin_bf16"),
    display_name="Marlin bf16",
    extra_cflags=_marlin_extra_cflags,
    extra_cuda_cflags=_marlin_extra_cuda_cflags,
    extra_include_paths=_marlin_include_paths,
    force_rebuild_env="GPTQMODEL_MARLIN_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def _marlin_runtime_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    return torch.bfloat16 if dtype == torch.bfloat16 else torch.float16


def _marlin_kernel_name_for_dtype(dtype: Optional[torch.dtype]) -> str:
    return "marlin_bf16" if _marlin_runtime_dtype(dtype) == torch.bfloat16 else "marlin_fp16"


def clear_marlin_extension_cache() -> None:
    _MARLIN_FP16_TORCH_OPS_EXTENSION.clear_cache()
    _MARLIN_BF16_TORCH_OPS_EXTENSION.clear_cache()


def marlin_runtime_available(dtype: Optional[torch.dtype] = None) -> bool:
    if marlin_import_exception is not None:
        return False
    return _extension_api().is_available(_marlin_kernel_name_for_dtype(dtype))


def marlin_runtime_error(dtype: Optional[torch.dtype] = None) -> str:
    if marlin_import_exception is not None:
        return marlin_import_exception

    extension_name = _marlin_kernel_name_for_dtype(dtype)
    extension_api = _extension_api()
    if extension_api.is_available(extension_name):
        return ""
    error_text = extension_api.error(extension_name) or "Marlin runtime unavailable."
    install_hint = _marlin_header_install_hint(error_text)
    if install_hint:
        return f"{error_text} {install_hint}"
    return error_text


def prewarm_marlin_extension(dtype: Optional[torch.dtype]) -> bool:
    extension_name = _marlin_kernel_name_for_dtype(dtype)
    return _extension_api().load(name=extension_name)[extension_name]


def _marlin_resolve_op(
    *,
    dtype: Optional[torch.dtype],
    op_name: str,
):
    return _extension_api().op(_marlin_kernel_name_for_dtype(dtype), op_name)


# Validate marlin support
def _validate_marlin_device_support() -> bool:
    """
    Validates if the current device is compatible for Marlin.
    ref: https://github.com/IST-DASLab/marlin?tab=readme-ov-file#requirements

    Returns:
        bool: indicates if CUDA device is compatible for Marlin
    """
    if IS_ROCM or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return _marlin_capability_supported(major, minor)


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(act_order: bool, group_size: int,
                                      is_row_parallel: bool) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_workspace_new(device: torch.device,
                              max_blocks_per_sm: int = 1) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    # Some kernels require a larger fixed minimum than the SM count on
    # lower-SM but still-supported GPUs, so clamp to that floor.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace_blocks = max(sms * max_blocks_per_sm, 128)
    return torch.zeros(workspace_blocks,
                       dtype=torch.int,
                       device=device,
                       requires_grad=False)


def update_tensor_inplace(dst: torch.Tensor, src: torch.Tensor):
    assert dst.dtype == src.dtype, "Tensors must have the same dtype"

    with torch.no_grad():
        # Mutating a registered Parameter must bypass autograd bookkeeping.
        dst.as_strided_(src.shape, src.stride())

        # If not the same underlying storage move tensor data
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src)
            del src


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_parameter(mod: torch.nn.Module, name: str,
                      new: Union[torch.Tensor, torch.nn.Parameter]) -> None:
    old = getattr(mod, name)
    if type(old) is type(new) and old.dtype == new.dtype and \
            old.untyped_storage().nbytes() == new.untyped_storage().nbytes():
        # If we can just update in-place to avoid re-registering
        #   can be faster if the underlying storage is the same
        update_tensor_inplace(old, new)
    else:
        # Fallback re-register parameter, convert to Parameter if necessary
        # this not only ensures we don't register a tensor as a parameter, but
        # also ensures that all parameter subclasses get re-registered as
        # parameters for `torch.compile` compatibility
        if not isinstance(new, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=False)
        mod.register_parameter(name,
                               torch.nn.Parameter(new, requires_grad=False))


def _transform_param(layer: torch.nn.Module, name: Optional[str],
                     fn: Callable) -> None:
    if name is not None and getattr(layer, name, None) is not None:
        old_param = getattr(layer, name)
        new_param = fn(old_param)
        # replace the parameter with torch.nn.Parameter for TorchDynamo
        # compatibility
        replace_parameter(
            layer, name,
            torch.nn.Parameter(new_param.data, requires_grad=False))


def marlin_sort_g_idx(
        g_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_tensor(layer: torch.nn.Module, name: str,
                   new_t: torch.Tensor) -> None:
    # It is important to use resize_() here since it ensures
    # the same buffer is reused
    getattr(layer, name).resize_(new_t.shape)
    getattr(layer, name).copy_(new_t)
    del new_t


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single

def maybe_warn_marlin_atomic_add_env():
    if torch.compiler.is_dynamo_compiling():
        return

    # log.info_once(
    #     "Marlin kernel can achieve better performance for small size_n "
    #     "with experimental use_atomic_add feature.")


def maybe_warn_marlin_atomic_add(device, dtype):
    if torch.compiler.is_dynamo_compiling():
        return
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        log.info_once(
            "You are running Marlin kernel with bf16 on GPUs before SM90. "
            "You can consider change to fp16 to achieve better performance "
            "if possible.")


def should_use_atomic_add_reduce(m: int, n: int, k: int, device: torch.device,
                                 dtype: torch.dtype) -> bool:
    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        maybe_warn_marlin_atomic_add(device, dtype)
        return False

    return True


def apply_gptq_marlin_linear(
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

    use_atomics = use_atomics and should_use_atomic_add_reduce(m=reshaped_x.size(0),
                                                  n=output_size_per_partition,
                                                  k=reshaped_x.size(1),
                                                  device=input.device,
                                                  dtype=input.dtype)

    output = gptq_marlin_gemm(reshaped_x,
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
                              is_zp_float=False)

    return output.reshape(out_shape)


def apply_awq_marlin_linear(
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
        use_fp32_reduce: bool = True) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(m=reshaped_x.size(0),
                                                  n=output_size_per_partition,
                                                  k=reshaped_x.size(1),
                                                  device=input.device,
                                                  dtype=input.dtype)

    output = gptq_marlin_gemm(reshaped_x,
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
                              is_zp_float=False)

    return output.reshape(out_shape)


def gptq_marlin_gemm(a: torch.Tensor,
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
                     is_zp_float: bool = False) -> torch.Tensor:
    if _marlin_runtime_dtype(a.dtype) == torch.bfloat16:
        op_name = "gptq_marlin_gemm_bf16"
    else:
        op_name = "gptq_marlin_gemm_fp16"

    op = _marlin_resolve_op(
        dtype=a.dtype,
        op_name=op_name,
    )
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


# gptq_marlin
def gptq_marlin_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                       size_k: int, size_n: int,
                       num_bits: int,
                       dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    op = _marlin_resolve_op(
        dtype=dtype,
        op_name="gptq_marlin_repack",
    )
    return op(b_q_weight, perm, size_k, size_n, num_bits)


def awq_marlin_repack(b_q_weight: torch.Tensor,
                      size_k: int,
                      size_n: int,
                      num_bits: int,
                      dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    op = _marlin_resolve_op(
        dtype=dtype,
        op_name="awq_marlin_repack",
    )
    return op(b_q_weight, size_k, size_n, num_bits)


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def pack_cols(
        q_w: torch.Tensor,
        num_bits: int,
        size_k: int,
        size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def unpack_cols(
        packed_q_w: torch.Tensor,
        num_bits: int,
        size_k: int,
        size_n: int,
):
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (
        size_k, size_n // pack_factor
    ), "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
        packed_q_w.shape, size_k, size_n, pack_factor)

    orig_device = packed_q_w.device

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def marlin_zero_points(zp: torch.Tensor, size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp


def awq_to_marlin_zero_points(q_zp_packed: torch.Tensor, size_k: int,
                              size_n: int, num_bits: int) -> torch.Tensor:
    # AWQ zero-points are quantized and packed on the column dim.
    # In addition, the values are permuted based on dequantizer.
    # Here we undo both of these, and then apply marlin permutation
    # and pack it back.
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo interleaving (use argsort(..) to get inverse perm)
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits)
    return marlin_zp


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()
