# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
from pathlib import Path

import torch

from .cpp import TorchOpsJitExtension, default_torch_ops_build_root


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


def _exllamav2_cxx11_abi_flag() -> int:
    return int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1))


def _exllamav2_gptq_sources() -> list[str]:
    root = _exllamav2_root()
    return [
        str(root / "ext_gptq.cpp"),
        str(root / "cuda" / "q_matrix.cu"),
        str(root / "cuda" / "q_gemm.cu"),
    ]


def _exllamav2_include_paths() -> list[str]:
    return [str(_exllamav2_root())]


def _exllamav2_extra_cflags() -> list[str]:
    abi = _exllamav2_cxx11_abi_flag()
    return [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ]


def _exllamav2_extra_cuda_cflags() -> list[str]:
    abi = _exllamav2_cxx11_abi_flag()
    return [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--threads",
        os.getenv("NVCC_THREADS", "2"),
        "--optimize=3",
        "-Xptxas",
        "-v,-O3,-dlcm=ca",
        "-lineinfo",
        "-Xfatbin",
        "-compress-all",
        "-diag-suppress=179,39,177",
    ]


_EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_exllamav2_ops",
    namespace="gptqmodel_exllamav2",
    required_ops=("make_q_matrix", "gemm_half_q_half"),
    sources=_exllamav2_gptq_sources,
    build_root_env="GPTQMODEL_EXLLAMAV2_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("exllamav2"),
    display_name="ExLlamaV2 GPTQ",
    extra_cflags=_exllamav2_extra_cflags,
    extra_cuda_cflags=_exllamav2_extra_cuda_cflags,
    extra_include_paths=_exllamav2_include_paths,
    force_rebuild_env="GPTQMODEL_EXLLAMAV2_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def clear_exllamav2_gptq_extension_cache() -> None:
    _EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION.clear_cache()


def exllamav2_gptq_runtime_available() -> bool:
    return _EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION.load()


def exllamav2_gptq_runtime_error() -> str:
    if _EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION.load():
        return ""
    return (
        _EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION.last_error_message()
        or "ExLlamaV2 GPTQ CUDA runtime unavailable."
    )


def prewarm_exllamav2_gptq_extension() -> bool:
    return exllamav2_gptq_runtime_available()


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
        _EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION.op("make_q_matrix")(
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
    _EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION.op("gemm_half_q_half")(a, int(q_handle), c, bool(force_cuda))
