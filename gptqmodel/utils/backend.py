from enum import Enum


class BACKEND(Enum):
    AUTO = 0  # choose the fastest one based on quant model compatibility
    AUTO_CPU = 1
    TRITON = 2
    EXLLAMA_V1 = 3
    EXLLAMA_V2 = 4
    MARLIN = 5
    BITBLAS = 6
    IPEX = 7
    VLLM = 8
    SGLANG = 9
    CUDA = 10
    TORCH = 11

def get_backend(backend: str):
    try:
        return BACKEND[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
