from enum import Enum


class BACKEND(str, Enum):
    AUTO = "auto"  # choose the optimal local kernel based on quant_config compatibility
    AUTO_TRAINABLE = "auto_trainable" # choose the optimal trainable local kernel for post-quant training
    CUDA = "cuda"
    TORCH = "torch"
    TRITON = "triton"
    EXLLAMA_V1 = "exllama_v1"
    EXLLAMA_V2 = "exllama_v2"
    MARLIN = "marlin"
    BITBLAS = "bitblas"
    IPEX = "ipex"
    VLLM = "vllm" # external inference engine
    SGLANG = "sglang" # external inference engine
