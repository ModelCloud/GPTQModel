from enum import Enum


class BACKEND(Enum):
    AUTO = 0  # choose the fastest one based on quant model compatibility
    TRITON = 1
    EXLLAMA_V1 = 2
    EXLLAMA_V2 = 3
    MARLIN = 4
    BITBLAS = 5
    IPEX = 6
    VLLM = 7
    SGLANG = 8
    CUDA = 9
    TORCH = 10

def get_backend(backend: str):
    try:
        return BACKEND[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
