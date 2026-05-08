# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from enum import Enum
from typing import Any, Optional, Union


class BACKEND(str, Enum):
    AUTO = "auto"  # choose the optimal local kernel based on quant_config compatibility
    AUTO_TRAINABLE = "auto_trainable"  # choose the optimal trainable local kernel for post-quant training

    # GPTQ kernels
    GPTQ_TORCH_FUSED = "gptq_torch_fused"  # optimized for Intel XPU
    GPTQ_TORCH_INT8 = "gptq_torch_int8"  # optimized CPU int8 fused kernel
    GPTQ_TORCH = "gptq_torch"  # GOOD: about 80% of triton
    GPTQ_TRITON = "gptq_triton"  # VERY GOOD: all-around kernel
    BITSANDBYTES = "bitsandbytes"  # bitsandbytes 4-bit/8-bit kernel with optional CPU/CUDA support
    GPTQ_EXLLAMA_V2 = "gptq_exllama_v2"  # FASTER: optimized for batching > 1
    GPTQ_MACHETE = "gptq_machete"  # CUTLASS-based kernel optimized for Hopper (SM90+)
    GPTQ_MARLIN = "gptq_marlin"  # marlin reduce ops, fp32 by default; controlled by GPTQMODEL_MARLIN_USE_FP32
    GPTQ_BITBLAS = "gptq_bitblas"  # BitBLAS AOT-compiled GPTQ kernel
    GPTQ_TORCH_ATEN = "gptq_torch_aten"  # CPU int4pack ATen kernel folded into GPT-QModel

    # QQQ kernels
    QQQ = "qqq"  # marlin-based qqq kernel
    QQQ_TORCH = "qqq_torch"

    # AWQ kernels
    AWQ_GEMM = "awq_gemm"
    AWQ_GEMM_TRITON = "awq_gemm_triton"
    AWQ_GEMV = "awq_gemv"
    AWQ_GEMV_FAST = "awq_gemv_fast"
    AWQ_TORCH_INT8 = "awq_torch_int8"
    AWQ_TORCH_FUSED = "awq_torch_fused"
    AWQ_TORCH_ATEN = "awq_torch_aten"  # CPU int4pack ATen kernel folded into GPT-QModel
    AWQ_TORCH = "awq_torch"
    AWQ_BITBLAS = "awq_bitblas"
    AWQ_MACHETE = "awq_machete"
    AWQ_MARLIN = "awq_marlin"
    AWQ_EXLLAMA_V2 = "awq_exllama_v2"

    # ParoQuant kernels
    PAROQUANT_CUDA = "paroquant_cuda"
    PAROQUANT_TRITON = "paroquant_triton"

    # FP8 kernels
    FP8_TORCH = "fp8_torch"

    # GGUF kernels / engines
    GGUF_TORCH = "gguf_torch"
    GGUF_TRITON = "gguf_triton"
    GGUF_CPP_CPU = "gguf_cpp_cpu"
    GGUF_CPP_CUDA = "gguf_cpp_cuda"

    # EXL3 engines
    EXL3_EXLLAMA_V3 = "exl3_exllama_v3"
    EXL3_TORCH = "exl3_torch"

    # external engines
    VLLM = "vllm"
    SGLANG = "sglang"
    MLX = "mlx"

    # Legacy generic names kept for compatibility with older call sites and saved args.
    TORCH_FUSED = "torch_fused"
    TORCH_INT8 = "torch_int8"
    TORCH = "torch"
    TRITON = "triton"
    EXLLAMA_V2 = "exllama_v2"
    EXLLAMA_V3 = "exllama_v3"
    MACHETE = "machete"
    MARLIN = "marlin"
    BITBLAS = "bitblas"
    GEMM = "gemm"
    GEMM_TRITON = "gemm_triton"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"
    TORCH_INT8_AWQ = "torch_int8_awq"
    TORCH_FUSED_AWQ = "torch_fused_awq"
    TORCH_AWQ = "torch_awq"
    BITBLAS_AWQ = "bitblas_awq"
    PARO = "paroquant"


class PROFILE(str, Enum):
    # Inference profile selects between alternative runtime/load strategies.
    AUTO = "auto"
    FAST = "fast"
    LOW_MEMORY = "low_memory"


_LEGACY_BACKEND_BY_METHOD = {
    "gptq": {
        BACKEND.TORCH_FUSED: BACKEND.GPTQ_TORCH_FUSED,
        BACKEND.TORCH_INT8: BACKEND.GPTQ_TORCH_INT8,
        BACKEND.TORCH: BACKEND.GPTQ_TORCH,
        BACKEND.TRITON: BACKEND.GPTQ_TRITON,
        BACKEND.EXLLAMA_V2: BACKEND.GPTQ_EXLLAMA_V2,
        BACKEND.MACHETE: BACKEND.GPTQ_MACHETE,
        BACKEND.MARLIN: BACKEND.GPTQ_MARLIN,
        BACKEND.BITBLAS: BACKEND.GPTQ_BITBLAS,
    },
    "awq": {
        BACKEND.GEMM: BACKEND.AWQ_GEMM,
        BACKEND.GEMM_TRITON: BACKEND.AWQ_GEMM_TRITON,
        BACKEND.GEMV: BACKEND.AWQ_GEMV,
        BACKEND.GEMV_FAST: BACKEND.AWQ_GEMV_FAST,
        BACKEND.TORCH: BACKEND.AWQ_TORCH,
        BACKEND.TORCH_AWQ: BACKEND.AWQ_TORCH,
        BACKEND.TORCH_INT8: BACKEND.AWQ_TORCH_INT8,
        BACKEND.TORCH_INT8_AWQ: BACKEND.AWQ_TORCH_INT8,
        BACKEND.TORCH_FUSED: BACKEND.AWQ_TORCH_FUSED,
        BACKEND.TORCH_FUSED_AWQ: BACKEND.AWQ_TORCH_FUSED,
        BACKEND.BITBLAS: BACKEND.AWQ_BITBLAS,
        BACKEND.BITBLAS_AWQ: BACKEND.AWQ_BITBLAS,
        BACKEND.MACHETE: BACKEND.AWQ_MACHETE,
        BACKEND.MARLIN: BACKEND.AWQ_MARLIN,
        BACKEND.EXLLAMA_V2: BACKEND.AWQ_EXLLAMA_V2,
    },
    "paroquant": {
        BACKEND.PARO: BACKEND.PAROQUANT_CUDA,
    },
    "fp8": {
        BACKEND.TORCH: BACKEND.FP8_TORCH,
    },
    "exl3": {
        BACKEND.EXLLAMA_V3: BACKEND.EXL3_EXLLAMA_V3,
        BACKEND.TORCH: BACKEND.EXL3_TORCH,
    },
}

_PROFILE_BY_INDEX = {
    0: PROFILE.AUTO,
    1: PROFILE.FAST,
    2: PROFILE.LOW_MEMORY,
}


def _normalize_method(method: Optional[Union[str, Any]]) -> Optional[str]:
    if method is None:
        return None
    value = getattr(method, "value", method)
    return str(value).lower()


def normalize_backend(
    backend: Optional[Union[str, BACKEND]],
    *,
    quant_method: Optional[Union[str, Any]] = None,
) -> Optional[BACKEND]:
    if backend is None:
        return None

    if isinstance(backend, BACKEND):
        resolved = backend
    elif isinstance(backend, str):
        normalized = backend.strip()
        if not normalized:
            return None
        resolved = BACKEND.__members__.get(normalized.upper())
        if resolved is None:
            resolved = BACKEND(normalized.lower())
    else:
        raise TypeError(f"backend must be a string or BACKEND, got `{type(backend)}`")

    method = _normalize_method(quant_method)
    if method is None:
        return resolved
    return _LEGACY_BACKEND_BY_METHOD.get(method, {}).get(resolved, resolved)


def normalize_profile(profile: Optional[Union[str, int, PROFILE]]) -> PROFILE:
    if profile is None:
        return PROFILE.AUTO

    if isinstance(profile, PROFILE):
        return profile

    if isinstance(profile, int):
        if profile in _PROFILE_BY_INDEX:
            return _PROFILE_BY_INDEX[profile]
        raise ValueError(f"Unknown profile index `{profile}`. Expected one of {sorted(_PROFILE_BY_INDEX)}.")

    if not isinstance(profile, str):
        raise TypeError(f"profile must be a string, int, or PROFILE, got `{type(profile)}`")

    normalized = profile.strip()
    if not normalized:
        return PROFILE.AUTO

    alias = normalized.replace("-", "_").replace(" ", "_")
    resolved = PROFILE.__members__.get(alias.upper())
    if resolved is not None:
        return resolved
    return PROFILE(alias.lower())
