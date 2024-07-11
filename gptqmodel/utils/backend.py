from enum import Enum


class BACKEND(Enum):
    AUTO = 0  # choose the fastest one based on quant model compatibility
    # CUDA_OLD = 1
    # CUDA = 2
    TRITON = 3
    EXLLAMA = 4
    EXLLAMA_V2 = 5
    MARLIN = 6
    BITBLAS = 7
    QBITS = 8
    VLLM = 9
    SGLANG = 10

def get_backend(backend: str):
    try:
        return BACKEND[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
