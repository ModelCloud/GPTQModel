# isort: off
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

# Many model architectures inherit from LlamaGPTQ, so it’s necessary to import llama first to avoid circular imports.
from .llama import LlamaQModel

# other model
from .afmoe import AfMoeQModel
from .apertus import ApertusQModel
from .baichuan import BaiChuanQModel
from .bailing_moe import BailingMoeQModel
from .bloom import BloomQModel
from .brumby import BrumbyQModel
from .chatglm import ChatGLMQModel
from .codegen import CodeGenQModel
from .dbrx import DbrxQModel
from .dbrx_converted import DbrxConvertedQModel
from .decilm import DeciLMQModel
from .deepseek_v2 import DeepSeekV2QModel
from .deepseek_v3 import DeepSeekV3QModel
from .dots1 import Dots1QModel
from .dream import DreamQModel
from .exaone import ExaOneQModel
from .exaone4 import Exaone4QModel
from .ernie4_5 import Ernie4_5QModel
from .ernie4_5_moe import Ernie4_5_MoeQModel
from .falcon_h1 import FalconH1QModel
from .gemma2 import Gemma2QModel
from .gemma3 import Gemma3ForConditionalGenerationGPTQ, Gemma3QModel
from .gemma4 import Gemma4ForConditionalGenerationGPTQ, Gemma4TextQModel
from .glm import GlmQModel
from .glm4_moe import GLM4MoEGPTQ
from .glm4_moe_lite import Glm4MoeLiteQModel
from .glm4v import Glm4vGPTQ
from .glm_moe_dsa import GlmMoeDsaQModel
from .gpt2 import GPT2QModel
from .gpt_bigcode import GptBigCodeQModel
from .gpt_neo import GptNeoQModel
from .gpt_neox import GPTNeoXQModel
from .gpt_oss import GPTOSSGPTQ
from .gptj import GptJQModel
from .granitemoehybrid import GraniteMoeHybridQModel
from .grinmoe import GrinMoeQModel
from .hymba import HymbaQModel
from .instella import InstellaQModel
from .internlm import InternLMQModel
from .internlm2 import InternLM2QModel
from .klear import KlearQModel
from .lfm2_moe import LFM2MoeQModel
from .llada2 import LLaDA2MoeQModel
from .llama4 import Llama4QModel
from .llava_qwen2 import LlavaQwen2QModel
from .longcat_flash import LongCatFlashQModel
from .mimo import MimoQModel
from .minicpm import MiniCPMGPTQ
from .minicpm3 import MiniCpm3QModel
from .minicpm_o import MiniCPMOQModel
from .minicpm_v import MiniCPMVQModel
from .minimax_m2 import MiniMaxM2GPTQ
from .mistral3 import Mistral3GPTQ
from .mixtral import MixtralQModel
from .mllama import MLlamaQModel
from .mobilellm import MobileLLMQModel
from .moss import MossQModel
from .mpt import MptQModel
from .nemotron_h import NemotronHQModel
from .olmoe import OlmoeGPTQ
from .opt import OptQModel
from .ovis import OvisQModel
from .ovis2 import Ovis2QModel
from .pangu_alpha import PanguAlphaQModel
from .phi import PhiQModel
from .phi3 import Phi3QModel, PhiMoEGPTQForCausalLM
from .phi4 import Phi4MMGPTQ
from .qwen import QwenQModel
from .qwen2 import Qwen2QModel
from .qwen2_5_omni import Qwen2_5_OmniGPTQ
from .qwen2_5_vl import Qwen2_5_VLQModel
from .qwen2_moe import Qwen2MoeQModel
from .qwen2_vl import Qwen2VLQModel
from .qwen3 import Qwen3QModel
from .qwen3_moe import Qwen3MoeQModel
from .qwen3_next import Qwen3NextGPTQ
from .qwen3_omni_moe import Qwen3OmniMoeGPTQ
from .qwen3_vl import Qwen3_VLQModel
from .rw import RwgQModel
from .starcoder2 import Starcoder2QModel
from .telechat2 import TeleChat2QModel
from .voxtral import VoxtralGPTQ
from .xverse import XverseQModel

TRANSFORMERS_SUPPORTS_QWEN3_5 = Version(TRANSFORMERS_VERSION) >= Version("5.2.0")
if TRANSFORMERS_SUPPORTS_QWEN3_5:
    from .qwen3_5 import Qwen3_5QModel
    from .qwen3_5_moe import Qwen3_5_MoeQModel
else:
    Qwen3_5QModel = None
    Qwen3_5_MoeQModel = None
