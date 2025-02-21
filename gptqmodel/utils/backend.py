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
    CUDA = "cuda"
    TORCH = "torch"
    TRITON = "triton"
    EXLLAMA_V1 = "exllama_v1"
    EXLLAMA_V2 = "exllama_v2"
    # EXLLAMA_EORA = "exllama_eora"
    MARLIN = "marlin"
    BITBLAS = "bitblas"
    IPEX = "ipex"
    VLLM = "vllm" # external inference engine (CUDA + ROCM + IPEX)
    SGLANG = "sglang" # external inference engine (CUDA + ROCm)
    MLX = "mlx" # external inference engine (Apple MLX on M1+)
