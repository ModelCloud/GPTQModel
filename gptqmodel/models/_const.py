from torch import device

CPU = device("cpu")
CUDA_0 = device("cuda:0")

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
    "RefinedWebModel",
    "RefinedWeb",
    "baichuan",
    "internlm",
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
    "gamma2",
    "starcoder2",
    "cohere",
    "minicpm",
    "qwen2_moe",
    "dbrx_converted",
    "deepseek_v2"
]

EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048

EXPERT_INDEX_PLACEHOLDER = "{expert_index}"


