# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from enum import Enum


class BACKEND(str, Enum):
    AUTO = "auto"  # choose the optimal local kernel based on quant_config compatibility
    AUTO_TRAINABLE = "auto_trainable" # choose the optimal trainable local kernel for post-quant training

    # gptq
    TORCH_FUSED = "torch_fused" # optimized for Intel XPU
    TORCH = "torch" # GOOD: about 80% of triton
    TRITON = "triton" # VERY GOOD: all-around kernel
    EXLLAMA_V1 = "exllama_v1" # FAST: optimized for batching == 1
    EXLLAMA_V2 = "exllama_v2" # FASTER: optimized for batching > 1
    EXLLAMA_EORA = "exllama_eora"
    MACHETE = "machete" # CUTLASS-based kernel optimized for Hopper (SM90+)
    MARLIN = "marlin" # FASTEST: marlin reduce ops in fp32 (higher precision -> more accurate, slightly slower)
    MARLIN_FP16 = "marlin_fp16" # FASTEST and then some: marlin reduce ops in fp16 (lower precision -> less accurate, slightly faster)
    BITBLAS = "bitblas" # EXTREMELY FAST: speed at the cost of 10+ minutes of AOT (ahead of time compilation with disk cache)

    # qqq
    QQQ = "qqq" # marlin based qqq kernel

    # awq
    GEMM = "gemm"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"

    # external
    VLLM = "vllm" # External inference engine: CUDA + ROCm + IPEX
    SGLANG = "sglang" # External inference engine: CUDA + ROCm
    MLX = "mlx" # External inference engine: Apple MLX on M1+ (Apple Silicon)
