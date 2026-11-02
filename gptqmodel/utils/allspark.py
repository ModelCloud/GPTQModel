# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the AllSpark Ampere W8A16 fused GEMM (bits=8, group_size=-1).

Kernel sources are vendored from vLLM (vllm-project/vllm#12931, Apache-2.0) and
live in ``gptqmodel_ext/allspark``.  Upstream gates the kernel to Ampere
(compute capability 80..89); the CUDA code itself only requires
``__CUDA_ARCH__ >= 800`` and is numerically correct on sm_120, so the runtime
check below is deliberately wider than vLLM's while the default selection keeps
the Ampere window.

Not wired into ``BACKEND`` yet - this module exists so the kernel can be built
and validated before the linear module lands.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .cpp import (
    TorchOpsJitExtension,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


_ALLSPARK_OPS_NAME = "gptqmodel_allspark_ops"
_ALLSPARK_NAMESPACE = "gptqmodel_allspark"

# Kernel requires K % 16 == 0 and N % 16 == 0 (see allspark_repack.cu) and the
# GEMM path pads N up to a multiple of 32.
ALLSPARK_N_ALIGN = 16
ALLSPARK_K_ALIGN = 16
ALLSPARK_M_CUBLAS_THRESHOLD = 1024
ALLSPARK_MIN_CAPABILITY = 80
ALLSPARK_MAX_CAPABILITY = 120


def _allspark_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "allspark"


def _allspark_sources() -> list[str]:
    root = _allspark_root()
    return [
        str(root / "allspark_torch.cpp"),
        str(root / "allspark_qgemm_w8a16.cu"),
        str(root / "allspark_repack.cu"),
    ]


def _allspark_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
    )
    # Torch's cpp_extension defines these to keep its own headers strict; the
    # AllSpark sources rely on implicit float -> half/bfloat16 conversions.
    flags += [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    return flags


_ALLSPARK_EXTENSION = TorchOpsJitExtension(
    name=_ALLSPARK_OPS_NAME,
    namespace=_ALLSPARK_NAMESPACE,
    required_ops=("allspark_w8a16_gemm", "rearrange_kn_weight_as_n32k16_order"),
    sources=_allspark_sources,
    build_root_env="GPTQMODEL_ALLSPARK_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("allspark"),
    display_name="AllSpark W8A16",
    extra_cuda_cflags=_allspark_cuda_cflags,
    extra_ldflags=lambda: ["-lcublas"],
    force_rebuild_env="GPTQMODEL_ALLSPARK_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def prewarm_allspark_extension() -> bool:
    return _ALLSPARK_EXTENSION.load()


def allspark_supported(device: torch.device | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability(device)
    return ALLSPARK_MIN_CAPABILITY <= capability[0] * 10 + capability[1] <= ALLSPARK_MAX_CAPABILITY


def allspark_repack(qweight: torch.Tensor, scales: torch.Tensor, n: int, k: int):
    """Rearrange a (K, N) uint8 weight into the kernel's N32K16 order."""
    n_32align = (n + 32 - 1) // 32 * 32
    qweight_reordered = torch.empty(
        (n_32align // 4, k * 4), dtype=torch.uint8, device=qweight.device
    )
    scales_reordered = torch.empty(
        (n_32align,), dtype=scales.dtype, device=scales.device
    )
    torch.ops.gptqmodel_allspark.rearrange_kn_weight_as_n32k16_order(
        qweight,
        scales,
        None,
        False,
        qweight_reordered,
        scales_reordered,
        None,
        k,
        n,
        n_32align,
    )
    return qweight_reordered.view(n_32align, k), scales_reordered


def allspark_gemm(a: torch.Tensor, b_qweight: torch.Tensor, b_scales: torch.Tensor, n: int):
    """W8A16 GEMM against weights already in N32K16 order."""
    device = a.device
    props = torch.cuda.get_device_properties(device)
    sm_version = props.major * 10 + props.minor
    return torch.ops.gptqmodel_allspark.allspark_w8a16_gemm(
        a,
        b_qweight,
        b_scales,
        None,
        n,
        -1,
        props.multi_processor_count,
        sm_version,
        ALLSPARK_M_CUBLAS_THRESHOLD,
        False,
        True,
    )
