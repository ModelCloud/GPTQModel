# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from pathlib import Path

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


class ScratchSpace:
    def __init__(self, scratch_bytes, dev):
        self.scratch_bytes = scratch_bytes
        self.scratch = torch.empty(
            self.scratch_bytes // 2,
            dtype=torch.float16,
            device=dev,
        )

    def get_slice(self, size_bytes):
        size_halfs = next_multiple(size_bytes, 128) // 2
        scratch_slice = self.scratch.narrow(0, 0, size_halfs)

        return scratch_slice


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
    required_ops=("make_q_matrix", "gemm_half_q_half"),
    sources=_exllamav2_gptq_sources,
    build_root_env="GPTQMODEL_EXLLAMAV2_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("exllamav2"),
    display_name="ExLlamaV2 GPTQ",
    extra_cflags=_exllamav2_gptq_extra_cflags,
    extra_cuda_cflags=_exllamav2_gptq_extra_cuda_cflags,
    extra_include_paths=_exllamav2_include_paths,
    force_rebuild_env="GPTQMODEL_EXLLAMAV2_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)

# Shared AWQ singleton so every caller reuses the same torch.ops cache and
# first-use build policy instead of depending on setup-time wheels.
_EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_exllamav2_awq_ops",
    namespace="gptqmodel_exllamav2_awq",
    required_ops=("make_q_matrix_awq", "gemm_half_q_half_awq"),
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
    force_rebuild_env="GPTQMODEL_EXLLAMAV2_AWQ_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def clear_exllamav2_gptq_extension_cache() -> None:
    _EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION.clear_cache()


def exllamav2_gptq_runtime_available() -> bool:
    return _extension_api().is_available("exllamav2")


def exllamav2_gptq_runtime_error() -> str:
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
    temp_dq,
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
            temp_dq,
        )
    )


def exllamav2_gemm_half_q_half(a, q_handle: int, c, force_cuda: bool = False) -> None:
    _extension_api().op("exllamav2", "gemm_half_q_half")(a, int(q_handle), c, bool(force_cuda))


def clear_exllamav2_awq_extension_cache() -> None:
    _EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION.clear_cache()


def exllamav2_awq_runtime_available() -> bool:
    return _extension_api().is_available("exllamav2_awq")


def exllamav2_awq_runtime_error() -> str:
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
    temp_dq,
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
            temp_dq,
        )
    )


def exllamav2_awq_gemm_half_q_half(a, q_handle: int, c, force_cuda: bool = False) -> None:
    _extension_api().op("exllamav2_awq", "gemm_half_q_half_awq")(a, int(q_handle), c, bool(force_cuda))
