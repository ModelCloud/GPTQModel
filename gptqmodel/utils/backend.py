from enum import Enum


class BACKEND(Enum):
    AUTO = 0  # choose the optimal local kernel based on quant_config compatibility
    AUTO_CPU = 1 # choose the optimal cpu-only local kernel; for transformer/optimum compat
    AUTO_TRAINABLE = 2 # choose the optimal trainable local kernel for post-quant training
    CUDA = 3
    TORCH = 4
    TRITON = 5
    EXLLAMA_V1 = 6
    EXLLAMA_V2 = 7
    MARLIN = 8
    BITBLAS = 9
    IPEX = 10
    VLLM = 11 # external inference engine
    SGLANG = 12 # external inference engine

def get_backend(backend: str):
    try:
        return BACKEND[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
