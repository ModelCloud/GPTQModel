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
    GPTQ_W4AFP8 = "gptq_w4afp8"  # INT4 GPTQ weights, token-scaled FP8 input on SM12x
    GPTQ_W4A_NVFP4 = "gptq_w4a_nvfp4"  # INT4 GPTQ weights, NVFP4 input on SM12x
    GPTQ_BITBLAS = "gptq_bitblas"  # BitBLAS AOT-compiled GPTQ kernel
    GPTQ_TORCH_ATEN = "gptq_torch_aten"  # CPU int4pack ATen kernel folded into GPT-QModel
    GPTQ_TORCH_CPU_AVX512 = "gptq_torch_cpu_avx512"  # CPU AVX512-BF16 GPTQ kernel ported from HF kernels
    GPTQ_TT_XLA = "gptq_tt_xla"  # Packed GPTQ through Torch/XLA on Tenstorrent PJRT
    GPTQ_TTNN = "gptq_ttnn"  # Packed GPTQ through native TTNN ops on Tenstorrent
    GPTQ_TT_BF8 = "gptq_tt_bf8"  # GPTQ weights re-stored as TT block-float bf8_b (hw unpack)
    # Weights already stored natively as TT block-float (`bf4_b` / `bf8_b`).
    # Unlike GPTQ_TT_BF8 there is no GPTQ layout to convert from:
    # the checkpoint carries packed mantissa codes and one exponent per block.
    TT_BLOCKFLOAT = "tt_blockfloat"

    # AllSpark kernels
    # A single backend covers both AllSpark sub-kernels. The quantization method
    # and the device select the concrete GPTQ or AWQ module (CUDA/HIP).
    ALLSPARK = "allspark"  # W8A16 fused GEMM, decode-oriented

    # Marlin kernels
    # A single backend covers both Marlin sub-kernels. The quantization method
    # and checkpoint format select the concrete GPTQ or AWQ implementation.
    MARLIN = "marlin"  # marlin reduce ops, fp32 by default; controlled by KernelConfig.marlin_fp32

    # Machete kernels
    # A single backend covers both Machete sub-kernels. The quantization method
    # and checkpoint format select the concrete GPTQ or AWQ implementation.
    MACHETE = "machete"  # CUTLASS-based kernel optimized for Hopper (SM90+)

    # Swordfish kernels
    # A single backend covers both Swordfish sub-kernels. The quantization
    # method and checkpoint format select the concrete GPTQ or AWQ implementation.
    SWORDFISH = "swordfish"  # Blackwell (sm100/sm110) w4a16/w8a16 GEMM kernel

    # Humming kernels
    # Humming is an external JIT-compiled CUDA GEMM library for W<8-bit weights
    # (github.com/vllm-project/humming, PyPI `humming-kernels`). It is the same
    # optional dependency vLLM and SGLang consume, so nothing is vendored here.
    # A single backend covers both the GPTQ and AWQ sub-kernels; the checkpoint
    # method/format selects the concrete implementation.
    HUMMING = "humming"  # SM75+ w2/w4/w8 a16 GEMM (dense; 3/5/6/7-bit layouts unsupported)

    # QQQ kernels
    QQQ = "qqq"  # marlin-based qqq kernel
    QQQ_TORCH = "qqq_torch"

    # AWQ kernels
    AWQ_GEMM = "awq_gemm"
    AWQ_GEMM_TRITON = "awq_gemm_triton"
    AWQ_GEMV = "awq_gemv"
    AWQ_GEMV_TRITON = "awq_gemv_triton"  # portable Triton GEMV (same packed layout)
    AWQ_GEMV_FAST = "awq_gemv_fast"
    AWQ_TORCH_INT8 = "awq_torch_int8"
    AWQ_TORCH_FUSED = "awq_torch_fused"
    AWQ_TORCH_ATEN = "awq_torch_aten"  # CPU int4pack ATen kernel folded into GPT-QModel
    AWQ_TORCH_CPU_AVX512 = "awq_torch_cpu_avx512"  # CPU AVX512-BF16 kernel shared with GPTQ
    AWQ_TORCH = "awq_torch"
    AWQ_BITBLAS = "awq_bitblas"
    AWQ_EXLLAMA_V2 = "awq_exllama_v2"

    # Paro kernels
    PARO_CUDA = "paro_cuda"
    PARO_TRITON = "paro_triton"

    # FP8 kernels
    FP8_TORCH = "fp8_torch"

    # VPTQ kernels
    VPTQ_TORCH = "vptq_torch"
    VPTQ_CUDA = "vptq_cuda"

    # NF12 (NyoomFloat12) lossless 12-bit BF16 weights
    NF12 = "nf12"
    CPTC = "cptc"

    # MACaQ MXFP4 kernels
    MACAQ_TORCH = "macaq_torch"
    MACAQ_TRITON = "macaq_triton"

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
    BITBLAS = "bitblas"
    GEMM = "gemm"
    GEMM_TRITON = "gemm_triton"
    GEMV = "gemv"
    GEMV_TRITON = "gemv_triton"
    GEMV_FAST = "gemv_fast"
    TORCH_INT8_AWQ = "torch_int8_awq"
    TORCH_FUSED_AWQ = "torch_fused_awq"
    TORCH_AWQ = "torch_awq"
    BITBLAS_AWQ = "bitblas_awq"
    PARO = "paro"


#: Backends that own Tenstorrent chips through TT-Metal instead of the PJRT
#: runtime. They stage checkpoint tensors on CPU and upload packed weights on
#: demand, so device selection and host staging must treat them alike.
NATIVE_TTNN_BACKENDS = frozenset({BACKEND.GPTQ_TTNN, BACKEND.GPTQ_TT_BF8})


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
        BACKEND.BITBLAS: BACKEND.GPTQ_BITBLAS,
    },
    "awq": {
        BACKEND.GEMM: BACKEND.AWQ_GEMM,
        BACKEND.GEMM_TRITON: BACKEND.AWQ_GEMM_TRITON,
        BACKEND.GEMV: BACKEND.AWQ_GEMV,
        BACKEND.GEMV_TRITON: BACKEND.AWQ_GEMV_TRITON,
        BACKEND.GEMV_FAST: BACKEND.AWQ_GEMV_FAST,
        BACKEND.TORCH: BACKEND.AWQ_TORCH,
        BACKEND.TORCH_AWQ: BACKEND.AWQ_TORCH,
        BACKEND.TORCH_INT8: BACKEND.AWQ_TORCH_INT8,
        BACKEND.TORCH_INT8_AWQ: BACKEND.AWQ_TORCH_INT8,
        BACKEND.TORCH_FUSED: BACKEND.AWQ_TORCH_FUSED,
        BACKEND.TORCH_FUSED_AWQ: BACKEND.AWQ_TORCH_FUSED,
        BACKEND.BITBLAS: BACKEND.AWQ_BITBLAS,
        BACKEND.BITBLAS_AWQ: BACKEND.AWQ_BITBLAS,
        BACKEND.EXLLAMA_V2: BACKEND.AWQ_EXLLAMA_V2,
    },
    "paro": {
        BACKEND.PARO: BACKEND.PARO_CUDA,
    },
    "fp8": {
        BACKEND.TORCH: BACKEND.FP8_TORCH,
    },
    "exl3": {
        BACKEND.EXLLAMA_V3: BACKEND.EXL3_EXLLAMA_V3,
        BACKEND.TORCH: BACKEND.EXL3_TORCH,
    },
    "vptq": {
        BACKEND.TORCH: BACKEND.VPTQ_TORCH,
    },
    "nf12": {
        BACKEND.TORCH: BACKEND.NF12,
    },
    "cptc": {
        BACKEND.TORCH: BACKEND.CPTC,
    },
    "macaq": {
        BACKEND.TORCH: BACKEND.MACAQ_TORCH,
        BACKEND.TRITON: BACKEND.MACAQ_TRITON,
    },
    "qqq": {
        BACKEND.TORCH: BACKEND.QQQ_TORCH,
    },
    # GGUF owns four engines, so the generic aliases must name the matching
    # engine explicitly. They cannot reach the GPTQ-only ``TorchLinear``:
    # ``get_kernel_for_backend`` filters candidates by ``SUPPORTS_METHODS``.
    "gguf": {
        BACKEND.TORCH: BACKEND.GGUF_TORCH,
        BACKEND.TRITON: BACKEND.GGUF_TRITON,
    },
    "bitsandbytes": {
        BACKEND.TORCH: BACKEND.BITSANDBYTES,
    },
}

# Method-qualified Marlin/Machete/Swordfish/AllSpark names from older releases.
# These are accepted by the stable HF-facing selection/repacking APIs, but they
# resolve to the single shared backend; method/format choose the sub-kernel later.
_LEGACY_BACKEND_ALIASES = {
    "gptq_allspark": BACKEND.ALLSPARK,
    "awq_allspark": BACKEND.ALLSPARK,
    "gptq_marlin": BACKEND.MARLIN,
    "awq_marlin": BACKEND.MARLIN,
    "gptq_machete": BACKEND.MACHETE,
    "awq_machete": BACKEND.MACHETE,
    "gptq_swordfish": BACKEND.SWORDFISH,
    "awq_swordfish": BACKEND.SWORDFISH,
    "gptq_humming": BACKEND.HUMMING,
    "awq_humming": BACKEND.HUMMING,
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
    # Imported lazily: configs depend on the enum and normalization rules.
    from .backend_config import BackendConfig

    if isinstance(backend, BackendConfig):
        backend = backend.backend
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
            resolved = _LEGACY_BACKEND_ALIASES.get(normalized.lower())
        if resolved is None:
            resolved = BACKEND(normalized.lower())
    else:
        raise TypeError(f"backend must be a string or BACKEND, got `{type(backend)}`")

    method = _normalize_method(quant_method)
    if method is None:
        return resolved
    return _LEGACY_BACKEND_BY_METHOD.get(method, {}).get(resolved, resolved)


def awq_gemm_pack_backend(
    *,
    device_type: str,
    is_rocm: bool,
    weight_only: bool,
    bits: int,
) -> BACKEND:
    """Return the AWQ GEMM backend used to pack quantized weights.

    The native AWQ GEMM extension is CUDA-only: its sources include
    ``cusparse.h`` and CUDA-specific inline asm, so it cannot be built on
    ROCm/XPU/NPU/CPU hosts.  It also packs only the 4-bit layout; 2/3/8-bit
    GEMM checkpoints use the portable torch kernel, which emits the identical
    AWQ GEMM layout for in-range codes.  Selecting ``AWQ_GEMM`` for another
    global width would fail the model-wide ``select_quant_linear(..., pack=True)``
    gate before calibration (#6135).

    Every ROCm host -- CDNA (gfx950/gfx942) as well as RDNA4 (gfx1200/gfx1201/
    gfx1250) -- takes ``AWQ_TORCH``.  RDNA4's hipBLASLt Tensile libraries exist
    but the AWQ GEMM extension itself is still CUDA-only, so this is a kernel
    availability decision, not an RDNA4 library gap.

    Args:
        device_type: Runtime type of the quantization device (``"cuda"``,
            ``"xpu"``, ``"npu"``, ``"cpu"``, ...).
        is_rocm: Whether the CUDA namespace is backed by the HIP runtime.
        weight_only: Whether the quantization lifecycle is weight-only
            (RTN->AWQ export), which always stays on the portable kernel.
        bits: Global quantization width. Only 4-bit runs use the native
            extension; every other supported width packs with ``AWQ_TORCH``.
    """

    if bits != 4 or weight_only or is_rocm or device_type != "cuda":
        return BACKEND.AWQ_TORCH
    return BACKEND.AWQ_GEMM


def awq_gemv_pack_backend(*, device_type: str = "cuda", is_rocm: bool) -> BACKEND:
    """Return the AWQ GEMV backend used to pack and run quantized weights.

    The native AWQ GEMV extension (``gemv_forward``/``gemmv2_forward``) is
    CUDA-only: its sources include ``cuda_fp16.h`` and CUDA-only inline asm, and
    its hipified warp-sync reduction rejects wave32.  ``AWQ_GEMV_TRITON`` reads
    the identical packed GEMV layout with a portable Triton kernel, so ROCm
    packs with that kernel instead of selecting a CUDA-only class.  A CPU-only
    host has neither accelerator kernel; it packs and runs with the portable
    Torch GEMV reader (``AwqGEMVTorchLinear``) so ``quantize()`` -> ``save()``
    still produces the requested GEMV checkpoint (issue #4554).

    Args:
        device_type: Runtime type of the quantization device (``"cuda"``,
            ``"rocm"``, ``"cpu"``, ...).
        is_rocm: Whether the CUDA namespace is backed by the HIP runtime.
    """

    if device_type == "cpu":
        return BACKEND.AWQ_TORCH
    return BACKEND.AWQ_GEMV_TRITON if is_rocm else BACKEND.AWQ_GEMV


def awq_gemv_fast_pack_backend(*, device_type: str = "cuda", is_rocm: bool) -> BACKEND:
    """Return the AWQ GEMV_FAST / LLM_AWQ backend used to pack the weights.

    The native GEMV_FAST extension (``gemv_fast_forward_decode`` /
    ``gemm_fast_forward_prefill``) is CUDA-only: hipcc rejects its sources, so
    ROCm has never had a backend for these two layouts (issue #5109). The
    portable Torch readers (``AwqGEMVFastTorchLinear``/``LLMAwqTorchLinear``)
    read and write the identical interleaved INT16 layout, so a host without
    the extension packs with ``BACKEND.AWQ_TORCH`` instead of refusing before
    calibration. That mirrors what #4753 did for a CPU-only host.

    Args:
        device_type: Runtime type of the quantization device (``"cuda"``,
            ``"rocm"``, ``"cpu"``, ...).
        is_rocm: Whether the CUDA namespace is backed by the HIP runtime.
    """

    if device_type == "cpu" or is_rocm:
        return BACKEND.AWQ_TORCH
    return BACKEND.AWQ_GEMV_FAST


def backend_for_activation(activation_mode: Optional[str], backend: Optional[BACKEND]) -> Optional[BACKEND]:
    """Bind a saved W4A policy to its matching kernel, including AUTO loads."""
    if activation_mode is None:
        return backend
    expected = {
        "w4afp8": BACKEND.GPTQ_W4AFP8,
        "w4a_nvfp4": BACKEND.GPTQ_W4A_NVFP4,
    }[activation_mode]
    if backend in (None, BACKEND.AUTO, expected):
        return expected
    raise ValueError(f"Activation policy `{activation_mode}` requires backend `{expected.value}`, got `{backend}`.")


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
