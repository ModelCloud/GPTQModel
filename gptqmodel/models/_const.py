from enum import Enum

import torch
from torch import device

from ..utils import BACKEND
from ..utils.torch import HAS_CUDA, HAS_MPS, HAS_XPU


CPU = device("cpu")
CUDA = device("cuda")
CUDA_0 = device("cuda:0")
XPU = device("xpu")
XPU_0 = device("xpu:0")
MPS = device("mps")

class DEVICE(str, Enum):
    ALL = "all" # All device
    CPU = "cpu" # All CPU
    CUDA = "cuda" # Nvidia GPU
    XPU = "xpu" # Intel GPU
    MPS = "mps" # MacOS GPU

class PLATFORM(str, Enum):
    ALL = "all" # All platform
    LINUX = "linux" # linux
    WIN32 = "win32" # windows
    DARWIN = "darwin" # macos


def validate_cuda_support(raise_exception: bool = False):
    got_cuda = HAS_CUDA
    if got_cuda:
        at_least_one_cuda_v6 = any(
            torch.cuda.get_device_capability(i)[0] >= 6 for i in range(torch.cuda.device_count()))

        if not at_least_one_cuda_v6:
            if raise_exception:
                raise EnvironmentError(
                    "GPTQModel cuda requires Pascal or later gpu with compute capability >= `6.0`.")
            else:
                got_cuda = False

    return got_cuda

def normalize_device(type_value: str|DEVICE|int|torch.device) -> DEVICE:
    if isinstance(type_value, int):
        if HAS_CUDA:
            return DEVICE.CUDA
        elif HAS_XPU:
            return DEVICE.XPU
        elif HAS_MPS:
            return DEVICE.MPS
        else:
            return DEVICE.CPU

    if isinstance(type_value, torch.device):
        type_value = type_value.type

    # remove device index
    split_results = [s.strip() for s in type_value.split(":") if s]
    if len(split_results) > 1:
        type_value = split_results[0]

    if isinstance(type_value, DEVICE):
        return type_value

    if not isinstance(type_value, str):
        raise ValueError(f"Invalid device type_value type: {type(type_value)}")

    return DEVICE(type_value.lower())


def get_best_device(backend: BACKEND=BACKEND.AUTO) -> torch.device:
    if backend == BACKEND.IPEX:
        return XPU_0 if HAS_XPU else CPU
    elif HAS_CUDA:
        return CUDA_0
    elif HAS_XPU:
        return XPU_0
    elif HAS_MPS:
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


