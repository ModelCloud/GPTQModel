from enum import Enum


class BACKEND(Enum):
    AUTO = 0  # choose the fastest one based on quant model compatibility
    AUTO_CPU = 1
    CUDA = 2
    TORCH = 3
    TRITON = 4
    EXLLAMA_V1 = 5
    EXLLAMA_V2 = 6
    MARLIN = 7
    BITBLAS = 8
    IPEX = 9
    VLLM = 10
    SGLANG = 11

def get_backend(backend: str):
    try:
        return BACKEND[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
