from enum import Enum
import torch

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

    def validate_for_inference(self):
        if self == BACKEND.TRITON:
            from packaging import version
            from triton import __version__ as triton_version
            if version.parse(triton_version) < version.parse("2.0.0"):
                raise ImportError(f"triton version must be >= 2.0.0: actual = {triton_version}")

            return True
        elif self == BACKEND.IPEX:
            import intel_pytorch_extension as ipex
            if not ipex.is_available():
                raise ImportError("Intel PyTorch Extension is not available.")

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return True

            # XPU is not available, check CPU
            from device_smi import Device
            smi = Device("cpu")
            info = smi.info()
            if "avx512_vnni" not in info["features"]:
                raise ValueError("CPU does not support AVX512_VNNI.")

            return True
        elif self == BACKEND.VLLM:
            from .vllm import VLLM_AVAILABLE

            return VLLM_AVAILABLE
        elif self == BACKEND.SGLANG:
            from .sglang import SGLANG_AVAILABLE

            return SGLANG_AVAILABLE
        else:
            return torch.cuda.is_available()

def get_backend(backend: str):
    try:
        return BACKEND[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
