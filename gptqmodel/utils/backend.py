# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class BACKEND(str, Enum):
    AUTO = "auto"  # choose the optimal local kernel based on quant_config compatibility
    AUTO_TRAINABLE = "auto_trainable" # choose the optimal trainable local kernel for post-quant training

    # gptq
    TORCH = "torch" # GOOD: about 80% of triton
    TRITON = "triton" # VERY GOOD: all-around kernel
    EXLLAMA_V1 = "exllama_v1" # FAST: optimized for batching == 1
    EXLLAMA_V2 = "exllama_v2" # FASTER: optimized for batching > 1
    EXLLAMA_EORA = "exllama_eora"
    MARLIN = "marlin" # FASTEST: marlin reduce ops in fp32 (higher precision -> more accurate, slightly slower)
    MARLIN_FP16 = "marlin_fp16" # FASTEST and then some: marlin reduce ops in fp16 (lower precision -> less accurate, slightly faster)
    BITBLAS = "bitblas" # EXTREMELY FAST: speed at the cost of 10+ minutes of AOT (ahead of time compilation with disk cache)
    IPEX = "ipex" # Best kernel for Intel XPU and Intel/AMD CPU with AVX512, AMX, # XMX

    # qqq
    QQQ = "qqq" # marlin based qqq kernel

    # external
    VLLM = "vllm" # External inference engine: CUDA + ROCm + IPEX
    SGLANG = "sglang" # External inference engine: CUDA + ROCm
    MLX = "mlx" # External inference engine: Apple MLX on M1+ (Apple Silicon)
