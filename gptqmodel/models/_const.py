import torch
from enum import Enum
from torch import device

CPU = device("cpu")
CUDA = device("cuda")
CUDA_0 = device("cuda:0")
XPU = device("xpu")
XPU_0 = device("xpu:0")

class DEVICE(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    XPU = "xpu"


def is_torch_support_xpu():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_device_by_type(type_value: str):
    for enum_constant in DEVICE:
        if enum_constant.value == type_value:
            return enum_constant
    raise ValueError(f"Invalid type_value str: {type_value}")


def get_best_device():
    if torch.cuda.is_available():
        return CUDA_0
    elif is_torch_support_xpu():
        return XPU_0
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
    "minicpm",
    "minicpm3"
    "qwen2_moe",
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


