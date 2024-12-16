import sys
from enum import Enum

import torch
from torch import device

from ..utils import BACKEND

CPU = device("cpu")
CUDA = device("cuda")
CUDA_0 = device("cuda:0")
XPU = device("xpu")
XPU_0 = device("xpu:0")
MPS = device("mps")

class DEVICE(str, Enum):
    CPU = "cpu" # All CPU
    CUDA = "cuda" # Nvidia GPU
    XPU = "xpu" # Intel GPU
    MPS = "mps" # MacOS GPU


def torch_supports_cuda(raise_exception: bool = False):
    has_cuda = sys.platform != "darwin" and hasattr(torch, "cuda") and torch.cuda.is_available()

    if has_cuda:
        at_least_one_cuda_v6 = any(
            torch.cuda.get_device_capability(i)[0] >= 6 for i in range(torch.cuda.device_count()))

        if not at_least_one_cuda_v6:
            if raise_exception:
                raise EnvironmentError(
                    "GPTQModel cuda requires Pascal or later gpu with compute capability >= `6.0`.")
            else:
                has_cuda = False

    return has_cuda


def torch_supports_xpu():
    return sys.platform != "darwin" and hasattr(torch, "xpu") and torch.xpu.is_available()

def torch_supports_mps():
    return sys.platform == "darwin" and hasattr(torch, "mps") and torch.mps.is_available()

def normalize_device(type_value: str|DEVICE|torch.device) -> DEVICE:
    if isinstance(type_value, torch.device):
        type_value = type_value.type

    if isinstance(type_value, DEVICE):
        return type_value

    if not isinstance(type_value, str):
        raise ValueError(f"Invalid device type_value type: {type(type_value)}")

    type_value = type_value.lower()

    for enum_constant in DEVICE: # type: DEVICE
        if enum_constant.startswith(type_value):
            return enum_constant
    raise ValueError(f"Invalid type_value str: {type_value}")


def get_best_device(backend: BACKEND=BACKEND.AUTO) -> torch.device:
    if backend == BACKEND.IPEX:
        return XPU_0 if torch_supports_xpu() else CPU
    elif torch_supports_cuda():
        return CUDA_0
    elif torch_supports_xpu():
        return XPU_0
    elif torch_supports_mps():
        return MPS
    else:
        return CPU

SUPPORTED_MODELS = [
    "bloom",
    "gptj",
    "gpt2",
    "gpt_neox",
    "opt",
    "moss",
    "gpt_bigcode",
    "codegen",
    "chatglm",
    "glm",
    "RefinedWebModel",
    "RefinedWeb",
    "baichuan",
    "internlm",
    "internlm2",
    "qwen",
    "xverse",
    "deci",
    "stablelm_epoch",
    "mpt",
    "llama",
    "longllama",
    "falcon",
    "mistral",
    "Yi",
    "mixtral",
    "qwen2",
    "phi",
    "phi3",
    "gemma",
    "gemma2",
    "starcoder2",
    "cohere",
    "cohere2",
    "minicpm",
    "minicpm3"
    "qwen2_moe",
    "qwen2_vl",
    "dbrx_converted",
    "deepseek_v2",
    "exaone",
    "grinmoe",
    "mllama",
    "granite",
    "mobilellm",
    "hymba",
    "olmo2",
]

EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048

EXPERT_INDEX_PLACEHOLDER = "{expert_index}"


