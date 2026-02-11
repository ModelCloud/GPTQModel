# isort: off
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Many model architectures inherit from LlamaGPTQ, so it’s necessary to import llama first to avoid circular imports.
from __future__ import annotations

import importlib
from typing import Dict, Tuple

from .llama import LlamaQModel

_LAZY: Dict[str, Tuple[str, str]] = {
    "BaiChuanQModel": ("baichuan", "BaiChuanQModel"),
    "BloomQModel": ("bloom", "BloomQModel"),
    "BrumbyQModel": ("brumby", "BrumbyQModel"),
    "ChatGLMQModel": ("chatglm", "ChatGLMQModel"),
    "CodeGenQModel": ("codegen", "CodeGenQModel"),
    "DbrxQModel": ("dbrx", "DbrxQModel"),
    "DbrxConvertedQModel": ("dbrx_converted", "DbrxConvertedQModel"),
    "DeciLMQModel": ("decilm", "DeciLMQModel"),
    "DeepSeekV2QModel": ("deepseek_v2", "DeepSeekV2QModel"),
    "DeepSeekV3QModel": ("deepseek_v3", "DeepSeekV3QModel"),
    "Dots1QModel": ("dots1", "Dots1QModel"),
    "DreamQModel": ("dream", "DreamQModel"),
    "ExaOneQModel": ("exaone", "ExaOneQModel"),
    "Exaone4QModel": ("exaone4", "Exaone4QModel"),
    "Ernie4_5QModel": ("ernie4_5", "Ernie4_5QModel"),
    "Ernie4_5_MoeQModel": ("ernie4_5_moe", "Ernie4_5_MoeQModel"),
    "Gemma2QModel": ("gemma2", "Gemma2QModel"),
    "Gemma3QModel": ("gemma3", "Gemma3QModel"),
    "GlmQModel": ("glm", "GlmQModel"),
    "GPT2QModel": ("gpt2", "GPT2QModel"),
    "GptBigCodeQModel": ("gpt_bigcode", "GptBigCodeQModel"),
    "GptNeoQModel": ("gpt_neo", "GptNeoQModel"),
    "GPTNeoXQModel": ("gpt_neox", "GPTNeoXQModel"),
    "GptJQModel": ("gptj", "GptJQModel"),
    "GrinMoeQModel": ("grinmoe", "GrinMoeQModel"),
    "HymbaQModel": ("hymba", "HymbaQModel"),
    "InstellaQModel": ("instella", "InstellaQModel"),
    "InternLMQModel": ("internlm", "InternLMQModel"),
    "InternLM2QModel": ("internlm2", "InternLM2QModel"),
    "Llama4QModel": ("llama4", "Llama4QModel"),
    "MimoQModel": ("mimo", "MimoQModel"),
    "MiniCpm3QModel": ("minicpm3", "MiniCpm3QModel"),
    "MiniMaxM2GPTQ": ("minimax_m2", "MiniMaxM2GPTQ"),
    "MixtralQModel": ("mixtral", "MixtralQModel"),
    "MLlamaQModel": ("mllama", "MLlamaQModel"),
    "MobileLLMQModel": ("mobilellm", "MobileLLMQModel"),
    "MossQModel": ("moss", "MossQModel"),
    "MptQModel": ("mpt", "MptQModel"),
    "OptQModel": ("opt", "OptQModel"),
    "OvisQModel": ("ovis", "OvisQModel"),
    "PhiQModel": ("phi", "PhiQModel"),
    "Phi3QModel": ("phi3", "Phi3QModel"),
    "QwenQModel": ("qwen", "QwenQModel"),
    "Qwen2QModel": ("qwen2", "Qwen2QModel"),
    "Qwen2_5_VLQModel": ("qwen2_5_vl", "Qwen2_5_VLQModel"),
    "Qwen2MoeQModel": ("qwen2_moe", "Qwen2MoeQModel"),
    "Qwen2VLQModel": ("qwen2_vl", "Qwen2VLQModel"),
    "Qwen3QModel": ("qwen3", "Qwen3QModel"),
    "Qwen3MoeQModel": ("qwen3_moe", "Qwen3MoeQModel"),
    "Qwen3_VLQModel": ("qwen3_vl", "Qwen3_VLQModel"),
    "RwgQModel": ("rw", "RwgQModel"),
    "Starcoder2QModel": ("starcoder2", "Starcoder2QModel"),
    "TeleChat2QModel": ("telechat2", "TeleChat2QModel"),
    "XverseQModel": ("xverse", "XverseQModel"),
    "FalconH1QModel": ("falcon_h1", "FalconH1QModel"),
    "PanguAlphaQModel": ("pangu_alpha", "PanguAlphaQModel"),
    "LongCatFlashQModel": ("longcat_flash", "LongCatFlashQModel"),
    "ApertusQModel": ("apertus", "ApertusQModel"),
    "KlearQModel": ("klear", "KlearQModel"),
    "LlavaQwen2QModel": ("llava_qwen2", "LlavaQwen2QModel"),
    "NemotronHQModel": ("nemotron_h", "NemotronHQModel"),
    "Qwen3OmniMoeGPTQ": ("qwen3_omni_moe", "Qwen3OmniMoeGPTQ"),
    "Mistral3GPTQ": ("mistral3", "Mistral3GPTQ"),
    "AfMoeQModel": ("afmoe", "AfMoeQModel"),
    "Glm4vGPTQ": ("glm4v", "Glm4vGPTQ"),
    "VoxtralGPTQ": ("voxtral", "VoxtralGPTQ"),
}


def __getattr__(name: str):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    mod_name, attr_name = spec
    module = importlib.import_module(f".{mod_name}", __name__)
    obj = getattr(module, attr_name)
    globals()[name] = obj
    return obj


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY.keys()))


__all__ = [
    "BaiChuanQModel", "BloomQModel", "BrumbyQModel", "ChatGLMQModel",
    "CodeGenQModel", "DbrxQModel", "DbrxConvertedQModel", "DeciLMQModel",
    "DeepSeekV2QModel", "DeepSeekV3QModel", "Dots1QModel", "DreamQModel",
    "ExaOneQModel", "Exaone4QModel", "Ernie4_5QModel", "Ernie4_5_MoeQModel",
    "Gemma2QModel", "Gemma3QModel", "GlmQModel", "GPT2QModel",
    "GptBigCodeQModel", "GptNeoQModel", "GPTNeoXQModel", "GptJQModel",
    "GrinMoeQModel", "HymbaQModel", "InstellaQModel", "InternLMQModel",
    "InternLM2QModel", "Llama4QModel", "MimoQModel", "MiniCpm3QModel",
    "MiniMaxM2GPTQ", "MixtralQModel", "MLlamaQModel", "MobileLLMQModel",
    "MossQModel", "MptQModel", "OptQModel", "OvisQModel", "PhiQModel",
    "Phi3QModel", "QwenQModel", "Qwen2QModel", "Qwen2_5_VLQModel",
    "Qwen2MoeQModel", "Qwen2VLQModel", "Qwen3QModel", "Qwen3MoeQModel",
    "Qwen3_VLQModel", "RwgQModel", "Starcoder2QModel", "TeleChat2QModel",
    "XverseQModel", "FalconH1QModel", "PanguAlphaQModel", "LongCatFlashQModel",
    "ApertusQModel", "KlearQModel", "LlavaQwen2QModel", "NemotronHQModel",
    "Qwen3OmniMoeGPTQ", "Mistral3GPTQ", "AfMoeQModel", "Glm4vGPTQ", "VoxtralGPTQ",
]
