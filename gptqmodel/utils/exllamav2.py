# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)
from .rocm import IS_ROCM


EXLLAMAV2_GPTQ_ROCM_UNSUPPORTED = (
    "ExLlamaV2 GPTQ is CUDA-only: ROCm support is disabled because of known "
    "accuracy regressions, and the native extension cannot be built with hipcc. "
    "Use the portable GPTQ Torch kernel instead "
    "(`backend=AUTO` or `BACKEND.TORCH`)."
)

EXLLAMAV2_AWQ_ROCM_UNSUPPORTED = (
    "ExLlamaV2 AWQ is CUDA-only: ROCm support is disabled because of known "
    "accuracy regressions, and the native extension cannot be built with hipcc. "
    "Use the portable AWQ Torch kernel instead "
    "(`backend=AUTO` or `BACKEND.AWQ_TORCH`)."
)


class ScratchSpace:
    def __init__(self, scratch_bytes, dev):
        self.scratch_bytes = scratch_bytes
        self.device = dev

    def get_slice(self, size_bytes):
        # Callers own the temporary allocation; never keep expanded weights
        # reachable from model/device scratch objects between operations.
        return torch.empty(next_multiple(size_bytes, 128) // 2,
                           dtype=torch.float16, device=self.device)


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def _exllamav2_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "exllamav2"


def _exllamav2_gptq_sources() -> list[str]:
    root = _exllamav2_root()
    return [
        str(root / "ext_gptq.cpp"),
        str(root / "cuda" / "q_matrix.cu"),
        str(root / "cuda" / "q_gemm.cu"),
    ]


def _exllamav2_required_cuda_headers() -> tuple[str, ...]:
    return ("cusparse.h",)


def _exllamav2_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_exllamav2_root())],
        required_header_names=_exllamav2_required_cuda_headers(),
    )


def _exllamav2_gptq_extra_cflags() -> list[str]:
    return default_jit_cflags(opt_level="O2", enable_bf16=True)


def _exllamav2_gptq_extra_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        opt_level="O2",
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )


def _exllamav2_extra_cflags() -> list[str]:
    return default_jit_cflags(enable_bf16=True)


def _exllamav2_extra_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )


def _exllamav2_extra_ldflags() -> list[str]:
    # q_gemm.cu / q_gemm_awq.cu call cublasHgemm. On Linux the symbol resolves
    # at load time, but the MSVC linker needs the import library explicitly or
    # the build fails with `LNK2019: unresolved external symbol cublasHgemm`.
    # Mirrors _extra_ldflags in gptqmodel/exllamav3/ext.py.
    if sys.platform != "win32":
        return []
    flags = ["cublas.lib"]
    if sys.base_prefix != sys.prefix:
        flags.append(f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}")
    return flags


def _exllamav2_awq_extra_cflags() -> list[str]:
    return default_jit_cflags(opt_level=None, enable_bf16=True)


def _exllamav2_awq_extra_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        opt_level=None,
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )


_EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_exllamav2_ops",
    namespace="gptqmodel_exllamav2",
    required_ops=("make_q_matrix", "gemm_half_q_half", "clone_q_matrix", "free_q_matrix"),
    sources=_exllamav2_gptq_sources,
    build_root_env="GPTQMODEL_EXLLAMAV2_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("exllamav2"),
    display_name="ExLlamaV2 GPTQ",
    extra_cflags=_exllamav2_gptq_extra_cflags,
    extra_cuda_cflags=_exllamav2_gptq_extra_cuda_cflags,
    extra_include_paths=_exllamav2_include_paths,
    extra_ldflags=_exllamav2_extra_ldflags,
    force_rebuild_env="GPTQMODEL_EXLLAMAV2_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)

# Shared AWQ singleton so every caller reuses the same torch.ops cache and
# first-use build policy instead of depending on setup-time wheels.
_EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_exllamav2_awq_ops",
    namespace="gptqmodel_exllamav2_awq",
    required_ops=("make_q_matrix_awq", "gemm_half_q_half_awq", "clone_q_matrix_awq", "free_q_matrix_awq"),
    sources=lambda: [
        str(_exllamav2_root() / "ext_awq.cpp"),
        str(_exllamav2_root() / "cuda" / "q_matrix_awq.cu"),
        str(_exllamav2_root() / "cuda" / "q_gemm_awq.cu"),
    ],
    build_root_env="GPTQMODEL_EXLLAMAV2_AWQ_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("exllamav2_awq"),
    display_name="ExLlamaV2 AWQ",
    extra_cflags=_exllamav2_awq_extra_cflags,
    extra_cuda_cflags=_exllamav2_awq_extra_cuda_cflags,
    extra_include_paths=_exllamav2_include_paths,
    extra_ldflags=_exllamav2_extra_ldflags,
    force_rebuild_env="GPTQMODEL_EXLLAMAV2_AWQ_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def clear_exllamav2_gptq_extension_cache() -> None:
    _EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION.clear_cache()


def exllamav2_gptq_supported() -> bool:
    """Whether the native ExLlamaV2 GPTQ extension can run on this host.

    The extension is CUDA-only: hipcc cannot compile the CUDA sources, and the
    kernel had accuracy regressions on ROCm.
    """

    return not IS_ROCM


def exllamav2_gptq_runtime_available() -> bool:
    # ROCm cannot build the native extension (hipcc rejects the CUDA sources),
    # so report it as unavailable instead of triggering a doomed JIT build.
    if not exllamav2_gptq_supported():
        return False
    return _extension_api().is_available("exllamav2")


def exllamav2_gptq_runtime_error() -> str:
    if IS_ROCM:
        return EXLLAMAV2_GPTQ_ROCM_UNSUPPORTED
    extension_api = _extension_api()
    if extension_api.is_available("exllamav2"):
        return ""
    return (
        extension_api.error("exllamav2")
        or "ExLlamaV2 GPTQ CUDA runtime unavailable."
    )


def prewarm_exllamav2_gptq_extension() -> bool:
    return _extension_api().load(name="exllamav2")["exllamav2"]


def exllamav2_make_q_matrix(
    q_weight,
    q_perm,
    q_invperm,
    q_scale,
    q_scale_max,
    q_groups,
    gptq_qzeros,
    gptq_scales,
    gptq_g_idx,
) -> int:
    return int(
        _extension_api().op("exllamav2", "make_q_matrix")(
            q_weight,
            q_perm,
            q_invperm,
            q_scale,
            q_scale_max,
            q_groups,
            gptq_qzeros,
            gptq_scales,
            gptq_g_idx,
        )
    )


def exllamav2_gemm_half_q_half(a, q_handle: int, c, force_cuda: bool = False) -> None:
    _extension_api().op("exllamav2", "gemm_half_q_half")(a, int(q_handle), c, bool(force_cuda))


def clear_exllamav2_awq_extension_cache() -> None:
    _EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION.clear_cache()


def exllamav2_awq_supported() -> bool:
    """Whether the native ExLlamaV2 AWQ extension can run on this host.

    See ``exllamav2_gptq_supported``: the extension is CUDA-only.
    """

    return not IS_ROCM


def exllamav2_awq_runtime_available() -> bool:
    # See exllamav2_gptq_runtime_available: never attempt the ROCm JIT build.
    if not exllamav2_awq_supported():
        return False
    return _extension_api().is_available("exllamav2_awq")


def exllamav2_awq_runtime_error() -> str:
    if IS_ROCM:
        return EXLLAMAV2_AWQ_ROCM_UNSUPPORTED
    extension_api = _extension_api()
    if extension_api.is_available("exllamav2_awq"):
        return ""
    return (
        extension_api.error("exllamav2_awq")
        or "ExLlamaV2 AWQ CUDA runtime unavailable."
    )


def prewarm_exllamav2_awq_extension() -> bool:
    return _extension_api().load(name="exllamav2_awq")["exllamav2_awq"]


def exllamav2_awq_make_q_matrix(
    q_weight,
    q_perm,
    q_invperm,
    q_scale,
    q_scale_max,
    q_groups,
    gptq_qzeros,
    gptq_scales,
    gptq_g_idx,
) -> int:
    return int(
        _extension_api().op("exllamav2_awq", "make_q_matrix_awq")(
            q_weight,
            q_perm,
            q_invperm,
            q_scale,
            q_scale_max,
            q_groups,
            gptq_qzeros,
            gptq_scales,
            gptq_g_idx,
        )
    )


def exllamav2_awq_gemm_half_q_half(a, q_handle: int, c, force_cuda: bool = False) -> None:
    _extension_api().op("exllamav2_awq", "gemm_half_q_half_awq")(a, int(q_handle), c, bool(force_cuda))


def exllamav2_clone_q_matrix(module, *, awq=False, compact=False):
    """Bind already-shuffled native words, retaining permutation ownership in Python."""
    name, suffix = ("exllamav2_awq", "_awq") if awq else ("exllamav2", "")
    tensors = [getattr(module, field) for field in ("qweight", "qzeros", "scales")]
    if compact:
        tensors = [tensor.contiguous() for tensor in tensors]
    metadata = getattr(module, "q_tensors", None) or {}
    handle = _extension_api().op(name, "clone_q_matrix" + suffix)(
        int(module.q_handle), *tensors, metadata.get("q_perm"), metadata.get("q_invperm"))
    return int(handle), tensors  # Keep compact temporaries alive until the consuming launch returns.


def exllamav2_free_q_matrix(handle, *, awq=False):
    name, suffix = ("exllamav2_awq", "_awq") if awq else ("exllamav2", "")
    if handle is not None:
        _extension_api().op(name, "free_q_matrix" + suffix)(int(handle))
