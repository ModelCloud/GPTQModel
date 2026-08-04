# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Swordfish Blackwell (sm100/sm110) weight-quantized GEMM extension loader.

Swordfish is vendored into gptqmodel_ext/swordfish under the GNU AGPL v3.0+
(see that directory's LICENSE and the top-level licenses/SWORDFISH).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)
from .logger import setup_logger
from .marlin_scalar_type import ScalarType, scalar_types
from .machete import _ensure_cutlass_source
from .rocm import IS_ROCM

log = setup_logger()

_SWORDFISH_OPS_NAME = "gptqmodel_swordfish_ops"
_SWORDFISH_OPS_NAMESPACE = "gptqmodel_swordfish"

_SWORDFISH_GENCODE_RE = re.compile(r"code=(?:sm_|compute_)(\d+)(?:[a-z])?")

_SWORDFISH_ARCH_FLAGS = (
    # Blackwell TMA/tcgen05 features require the "a" (all) architecture suffix.
    # The B300 in this environment reports compute capability 10.3 (sm_103a).
    # Emit native cubins for the common Blackwell datacenter variants plus a
    # forward-compatible PTX fallback.
    "-gencode=arch=compute_100a,code=sm_100a",
    "-gencode=arch=compute_103a,code=sm_103a",
    "-gencode=arch=compute_110a,code=sm_110a",
    "-gencode=arch=compute_103a,code=compute_103a",
)

_SWORDFISH_JIT_NVCC_THREADS = "16"

_SWORDFISH_REQUIRED_TORCH_NVCC_UNDEFINES = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
)

_SWORDFISH_REQUIRED_CUDA_HEADERS = (
    "cuda_runtime_api.h",
    "cublas_v2.h",
    "cublasLt.h",
    "cusolverDn.h",
)

SWORDFISH_PREPACKED_BLOCK_SHAPE = (64, 64)


def _swordfish_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _swordfish_source_root() -> Path:
    return _swordfish_project_root() / "gptqmodel_ext" / "swordfish"


def _swordfish_sources() -> List[str]:
    root = _swordfish_source_root()
    return [
        str(root / "swordfish_bindings.cpp"),
        str(root / "libtorch_stable" / "quantization" / "marlin" / "gptq_marlin_repack.cu"),
        str(root / "libtorch_stable" / "quantization" / "swordfish" / "swordfish_prepack.cu"),
        str(root / "libtorch_stable" / "quantization" / "swordfish" / "swordfish_mm.cu"),
        str(root / "libtorch_stable" / "quantization" / "swordfish" / "swordfish_dense_tier.cu"),
        str(root / "libtorch_stable" / "quantization" / "swordfish" / "swordfish_prefill.cu"),
        str(root / "libtorch_stable" / "quantization" / "swordfish" / "swordfish_prefill_f16.cu"),
        str(root / "libtorch_stable" / "quantization" / "swordfish" / "swordfish_moe.cu"),
    ]


def _swordfish_include_paths() -> List[str]:
    cutlass_root = _ensure_cutlass_source()
    include_paths = [
        str(_swordfish_source_root().resolve()),
        str((cutlass_root / "include").resolve()),
        str((cutlass_root / "tools" / "library" / "include").resolve()),
        str((cutlass_root / "examples" / "common" / "include").resolve()),
        str((cutlass_root / "tools" / "util" / "include").resolve()),
    ]
    # Only keep existing directories.
    include_paths = [p for p in include_paths if Path(p).is_dir()]
    return cuda_include_paths_with_fallback(
        include_paths,
        required_header_names=_SWORDFISH_REQUIRED_CUDA_HEADERS,
    )


def _swordfish_extra_cflags() -> List[str]:
    return default_jit_cflags(enable_bf16=True)


def _swordfish_extra_cuda_cflags() -> List[str]:
    if _swordfish_static_runtime_error():
        return []

    # Only emit the sm100f/sm110f family gencodes when the local host could
    # plausibly build them (CUDA 13+). The kernel targets these fixed Blackwell
    # variants, so we do not merge a user-provided TORCH_CUDA_ARCH_LIST here.
    arch_flags: Tuple[str, ...] = _SWORDFISH_ARCH_FLAGS

    return [
        "-DUSE_CUDA",
        # CUTLASS 4.4 does not set CUDA_ARCH_FAMILY(1000) for the sm_100f
        # target on all nvcc builds, which leaves TMA/tcgen05 feature macros
        # disabled. Force-include an arch-macro shim that enables the family
        # macros only in the device-code pass (__CUDA_ARCH__ is not defined
        # during host compilation, so host-side paths keep their stubs).
        "-include",
        str(_swordfish_source_root() / "swordfish_arch_macros.cuh"),
        *_SWORDFISH_REQUIRED_TORCH_NVCC_UNDEFINES,
        *default_jit_cuda_cflags(
            enable_bf16=True,
            include_lineinfo=True,
            include_nvcc_threads=True,
            include_ptxas_optimizations=True,
            include_ptxas_verbosity=False,
            include_fatbin_compression=True,
            include_diag_suppress=True,
            nvcc_threads=_SWORDFISH_JIT_NVCC_THREADS,
        ),
        *arch_flags,
    ]


def _swordfish_extra_ldflags() -> List[str]:
    return ["-lcuda", "-lcublas", "-lcublasLt"]


_SWORDFISH_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_SWORDFISH_OPS_NAME,
    namespace=_SWORDFISH_OPS_NAMESPACE,
    required_ops=(
        "swordfish_prepack_B",
        "swordfish_mm",
        "swordfish_dequant_dense",
        "swordfish_moe_mm",
        "swordfish_prefill_mm",
    ),
    sources=_swordfish_sources,
    build_root_env="GPTQMODEL_SWORDFISH_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("swordfish"),
    display_name="Swordfish",
    extra_cflags=_swordfish_extra_cflags,
    extra_cuda_cflags=_swordfish_extra_cuda_cflags,
    extra_include_paths=_swordfish_include_paths,
    extra_ldflags=_swordfish_extra_ldflags,
    force_rebuild_env="GPTQMODEL_SWORDFISH_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    # The kernel targets sm100f/sm110f family; do not let PyTorch merge the
    # visible sm103 capability into the arch list and drop the family flag.
    merge_visible_cuda_arch_override=False,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def _swordfish_min_supported_compute_capability() -> Tuple[int, int]:
    """Parse _SWORDFISH_ARCH_FLAGS and return the lowest compute capability we
    emit code for, including the PTX fallback.  Any device with this or a
    higher compute capability can load Swordfish (subject to driver PTX JIT)."""
    caps: set[Tuple[int, int]] = set()
    for flag in _SWORDFISH_ARCH_FLAGS:
        for match in _SWORDFISH_GENCODE_RE.finditer(flag):
            cap = int(match.group(1))
            caps.add((cap // 10, cap % 10))
    if not caps:
        # Fallback in case the gencode format ever changes unexpectedly.
        caps = {(10, 0)}
    return min(caps)


_SWORDFISH_MIN_SUPPORTED_COMPUTE_CAPABILITY: Tuple[int, int] = (
    _swordfish_min_supported_compute_capability()
)


def _swordfish_static_runtime_error() -> str:
    if IS_ROCM:
        return "Swordfish kernel is not supported on ROCm."
    if not torch.cuda.is_available():
        return "Swordfish kernel requires CUDA."
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < _SWORDFISH_MIN_SUPPORTED_COMPUTE_CAPABILITY:
        mj, mn = _SWORDFISH_MIN_SUPPORTED_COMPUTE_CAPABILITY
        return (
            f"Swordfish kernel requires Blackwell (sm{mj}{mn}+) or newer; "
            f"found compute capability {major}.{minor}."
        )
    return ""


def _validate_swordfish_device_support() -> bool:
    return _swordfish_static_runtime_error() == ""


def swordfish_runtime_available() -> bool:
    static_error = _swordfish_static_runtime_error()
    if static_error:
        return False
    return _extension_api().is_available("swordfish")


def swordfish_runtime_error() -> str:
    static_error = _swordfish_static_runtime_error()
    if static_error:
        return static_error

    extension_api = _extension_api()
    if extension_api.is_available("swordfish"):
        return ""
    return extension_api.error("swordfish") or "Swordfish runtime unavailable."


def clear_swordfish_extension_cache() -> None:
    _SWORDFISH_TORCH_OPS_EXTENSION.clear_cache()


def prewarm_swordfish_extension() -> bool:
    return _extension_api().load(name="swordfish")["swordfish"]


def swordfish_prepack_B(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int = 4,
    perm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _extension_api().op("swordfish", "swordfish_prepack_B")(
        b_q_weight, perm, size_k, size_n, num_bits
    )


def swordfish_mm(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    group_scales: torch.Tensor,
    group_size: int,
    size_k: int,
    size_n: int,
    group_zps: Optional[torch.Tensor] = None,
    num_bits: int = 4,
) -> torch.Tensor:
    return _extension_api().op("swordfish", "swordfish_mm")(
        a,
        b_packed,
        group_scales,
        group_zps,
        num_bits,
        group_size,
        size_k,
        size_n,
    )


def swordfish_dequant_dense(
    b_packed: torch.Tensor,
    group_scales: torch.Tensor,
    group_zps: Optional[torch.Tensor],
    num_bits: int,
    group_size: int,
    size_k: int,
    size_n: int,
    transpose: bool = False,
) -> torch.Tensor:
    return _extension_api().op("swordfish", "swordfish_dequant_dense")(
        b_packed,
        group_scales,
        group_zps,
        num_bits,
        group_size,
        size_k,
        size_n,
        transpose,
    )


def swordfish_moe_mm(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    group_scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    num_bits: int,
    group_size: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    return _extension_api().op("swordfish", "swordfish_moe_mm")(
        a,
        b_packed,
        group_scales,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        num_bits,
        group_size,
        size_k,
        size_n,
    )


def swordfish_prefill_mm(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    group_scales: torch.Tensor,
    group_size: int,
    size_k: int,
    size_n: int,
    group_zps: Optional[torch.Tensor] = None,
    num_bits: int = 4,
) -> torch.Tensor:
    return _extension_api().op("swordfish", "swordfish_prefill_mm")(
        a,
        b_packed,
        group_scales,
        group_zps,
        num_bits,
        group_size,
        size_k,
        size_n,
    )


def query_swordfish_supported_quant_types(zero_points: bool) -> List[ScalarType]:
    # AWQ-style zero points are only supported for 4-bit weights; the kernel
    # rejects zero points with 8-bit weights.
    if zero_points:
        return [scalar_types.uint4]
    return [scalar_types.uint4b8, scalar_types.uint8b128]


def query_swordfish_supported_act_types(_zero_points: bool) -> List[torch.dtype]:
    return [torch.float16, torch.bfloat16]


def query_swordfish_supported_group_sizes(_act_type: torch.dtype) -> List[int]:
    return [-1, 32, 64, 128]


def check_swordfish_supports_shape(
    in_features: int,
    out_features: int,
) -> tuple[bool, Optional[str]]:
    if in_features % SWORDFISH_PREPACKED_BLOCK_SHAPE[0] != 0:
        return (
            False,
            f"Input features size must be divisible by {SWORDFISH_PREPACKED_BLOCK_SHAPE[0]}",
        )
    if out_features % SWORDFISH_PREPACKED_BLOCK_SHAPE[1] != 0:
        return (
            False,
            f"Output features size must be divisible by {SWORDFISH_PREPACKED_BLOCK_SHAPE[1]}",
        )
    return (True, None)


__all__ = [
    "check_swordfish_supports_shape",
    "clear_swordfish_extension_cache",
    "prewarm_swordfish_extension",
    "query_swordfish_supported_act_types",
    "query_swordfish_supported_group_sizes",
    "query_swordfish_supported_quant_types",
    "swordfish_dequant_dense",
    "swordfish_mm",
    "swordfish_moe_mm",
    "swordfish_prepack_B",
    "swordfish_prefill_mm",
    "swordfish_runtime_available",
    "swordfish_runtime_error",
]
