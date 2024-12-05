from enum import Enum


class BACKEND(Enum):
    AUTO = 0  # choose the optimal local kernel based on quant_config compatibility
    AUTO_TRAINABLE = 1 # choose the optimal trainable local kernel for post-quant training
    CUDA = 2
    TORCH = 3
    TRITON = 4
    EXLLAMA_V1 = 5
    EXLLAMA_V2 = 6
    MARLIN = 7
    BITBLAS = 8
    IPEX = 9
    VLLM = 10 # external inference engine
    SGLANG = 11 # external inference engine

def get_backend(backend: str):
    try:
        backend = backend.upper()
        return BACKEND[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
