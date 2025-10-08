# isort: off
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Many model architectures inherit from LlamaGPTQ, so it’s necessary to import llama first to avoid circular imports.
from .llama import LlamaQModel

# other model
from .baichuan import BaiChuanQModel
from .bloom import BloomQModel
from .chatglm import ChatGLMQModel
from .codegen import CodeGenQModel
from .dbrx import DbrxQModel
from .dbrx_converted import DbrxConvertedQModel
from .decilm import DeciLMQModel
from .deepseek_v2 import DeepSeekV2QModel
from .deepseek_v3 import DeepSeekV3QModel
from .dream import DreamQModel
from .exaone import ExaOneQModel
from .ernie4_5 import Ernie4_5QModel
from .ernie4_5_moe import Ernie4_5_MoeQModel
from .gemma2 import Gemma2QModel
from .gemma3 import Gemma3QModel
from .glm import GlmQModel
from .gpt2 import GPT2QModel
from .gpt_bigcode import GptBigCodeQModel
from .gpt_neo import GptNeoQModel
from .gpt_neox import GPTNeoXQModel
from .gptj import GptJQModel
from .grinmoe import GrinMoeQModel
from .hymba import HymbaQModel
from .instella import InstellaQModel
from .internlm import InternLMQModel
from .internlm2 import InternLM2QModel
from .llama4 import Llama4QModel
from .mimo import MimoQModel
from .minicpm3 import MiniCpm3QModel
from .mixtral import MixtralQModel
from .mllama import MLlamaQModel
from .mobilellm import MobileLLMQModel
from .moss import MossQModel
from .mpt import MptQModel
from .opt import OptQModel
from .ovis import OvisQModel
from .phi import PhiQModel
from .phi3 import Phi3QModel
from .qwen import QwenQModel
from .qwen2 import Qwen2QModel
from .qwen2_5_vl import Qwen2_5_VLQModel
from .qwen2_moe import Qwen2MoeQModel
from .qwen2_vl import Qwen2VLQModel
from .qwen3 import Qwen3QModel
from .qwen3_moe import Qwen3MoeQModel
from .rw import RwgQModel
from .starcoder2 import Starcoder2QModel
from .telechat2 import TeleChat2QModel
from .xverse import XverseQModel
from .falcon_h1 import FalconH1QModel
from .pangu_alpha import PanguAlphaQModel
from .longcat_flash import LongCatFlashQModel
from .apertus import ApertusQModel
from .klear import KlearQModel
from .llava_qwen2 import LlavaQwen2QModel
from .nemotron_h import NemotronHQModel
from .qwen3_omni_moe import Qwen3OmniMoeGPTQ
