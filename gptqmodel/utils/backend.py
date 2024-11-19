from enum import Enum


class BACKEND(Enum):
    AUTO = 0  # choose the fastest one based on quant model compatibility
    TRITON = 1
    EXLLAMA_V2 = 2
    MARLIN = 3
    BITBLAS = 4
    IPEX = 5
    VLLM = 6
    SGLANG = 7
    CUDA = 8

def get_backend(backend: str):
    try:
        return BACKEND[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
