from enum import Enum


class Backend(Enum):
    AUTO = 0  # choose the fastest one based on quant model compatibility
    CUDA_OLD = 1
    CUDA = 2
    TRITON = 3
    EXLLAMA = 4
    EXLLAMA_V2 = 5
    MARLIN = 6
    BITBLAS = 7


def get_backend(backend: str):
    try:
        return Backend[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
