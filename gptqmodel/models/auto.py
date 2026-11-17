# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# ruff: noqa: I001

from __future__ import annotations

import os
from contextlib import contextmanager

from .._alloc_conf import apply_default_alloc_conf
from ..utils.logger import setup_logger
from ..utils.quant_metadata import describe_unsupported_metadata, find_unsupported_metadata

log = setup_logger()

ASCII_LOGO = r"""
┌─────────────┐    ┌────────────────────────┐    ┌────────────┐    ┌─────────┐
│ GPT-QModel  │ -> │ ▓▓▓▓▓▓▓▓▓▓▓▓ 16bit     │ -> │ ▒▒▒▒ 8bit  │ -> │ ░░ 4bit │
└─────────────┘    └────────────────────────┘    └────────────┘    └─────────┘
"""

# if not os.environ.get("PYTHON_GIL", None):
#     os.environ["PYTHON_GIL"] = '0'
#     log.info("ENV: Auto disable GIL and use free-threading mode when applicable: Python 3.13t+. You must install the -t edition of Python.")

# Normally already installed by `gptqmodel/__init__.py` before Torch touched
# CUDA; repeated here so importing this module directly still gets the default.
_installed_alloc_conf = apply_default_alloc_conf()
if _installed_alloc_conf is not None:
    log.info("ENV: Auto setting PYTORCH_ALLOC_CONF='%s' for memory saving.", _installed_alloc_conf)

if not os.environ.get("CUDA_DEVICE_ORDER", None):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    log.info("ENV: Auto setting CUDA_DEVICE_ORDER=PCI_BUS_ID for correctness.")

# FIX ROCm env conflict with CUDA_VISIBLE_DEVICES if both exits
if 'CUDA_VISIBLE_DEVICES' in os.environ and 'ROCR_VISIBLE_DEVICES' in os.environ:
    del os.environ['ROCR_VISIBLE_DEVICES']

# if not os.environ.get("NCCL_SHM_DISABLE", None):
#     os.environ["NCCL_SHM_DISABLE"] = '1'
#     log.info("ENV: Auto setting NCCL_SHM_DISABLE=1 for multi-gpu memory safety.")

import os.path  # noqa: E402
from os.path import isdir, join  # noqa: E402
from typing import Any, Dict, List, Optional, Union  # noqa: E402

import torch  # noqa: E402
from packaging.version import Version  # noqa: E402
from transformers import AutoConfig, PreTrainedTokenizerBase  # noqa: E402
from transformers import __version__ as TRANSFORMERS_VERSION

from ..adapter.adapter import Adapter, Lora, normalize_adapter  # noqa: E402
from ..nn_modules.qlinear.torch import TorchLinear  # noqa: E402
from ..quantization import (  # noqa: E402
    METHOD,
    QUANT_CONFIG_FILENAME,
    AutoModuleDecoderConfig,
    CalibrationConfig,
    QuantizeConfig,
)
from ..utils import BACKEND, PROFILE  # noqa: E402
from ..utils.backend import normalize_backend, normalize_profile  # noqa: E402
from ..utils.env import env_flag  # noqa: E402
from ..utils.hf import (  # noqa: E402
    get_config_selection_kwargs,
    get_hf_gguf_load_kwargs,
    normalize_model_id_or_path_for_hf_gguf,
    normalize_torch_dtype_kwarg,
    patch_remote_code_before_config_load,
    resolve_trust_remote_code,
)
from ..utils.hub import list_repo_files  # noqa: E402
from ..utils.importer import auto_select_device  # noqa: E402
from ..utils.linalg_warmup import (  # noqa: E402
    kick_torch_linear_prewarm,
    kick_torch_quantize_prewarm,
)
from ..utils.model import auto_dtype, find_modules  # noqa: E402
from ..utils.kernel import KernelConfig, configure_kernel  # noqa: E402
from ..utils.backend_config import Backend, RuntimeConfig, resolve_backend_config, resolve_quantize_backend  # noqa: E402
from ..utils.p2p import P2PConfig  # noqa: E402
from ..utils.torch import torch_empty_cache  # noqa: E402
from ._const import normalize_device  # noqa: E402
from ..utils.looper_helpers import normalize_device_like, select_forward_devices  # noqa: E402
from .base import BaseQModel  # noqa: E402
from .loader import _explicit_requested_devices, native_floatx_source_format  # noqa: E402
from .definitions.afmoe import AfMoeQModel  # noqa: E402
from .definitions.agnes import AgnesQModel  # noqa: E402
from .definitions.apertus import Apertus1p5QModel, Apertus1p5TextQModel, ApertusQModel  # noqa: E402
from .definitions.axk2 import AXK2QModel  # noqa: E402
from .definitions.baichuan import BaiChuanQModel  # noqa: E402
from .definitions.bailing_moe import BailingMoeQModel  # noqa: E402
from .definitions.bloom import BloomQModel  # noqa: E402
from .definitions.brumby import BrumbyQModel  # noqa: E402
from .definitions.chatglm import ChatGLMQModel  # noqa: E402
from .definitions.codegen import CodeGenQModel  # noqa: E402
from .definitions.cohere2_moe import Cohere2MoeQModel  # noqa: E402
from .definitions.cohere_compass import CohereCompassQModel  # noqa: E402
from .definitions.dbrx import DbrxQModel  # noqa: E402
from .definitions.dbrx_converted import DbrxConvertedQModel  # noqa: E402
from .definitions.dots3_note import Dots3NoteQModel  # noqa: E402
from .definitions.decilm import DeciLMQModel  # noqa: E402
from .definitions.deepseek_v2 import DeepSeekV2QModel  # noqa: E402
from .definitions.deepseek_v3 import DeepSeekV3QModel  # noqa: E402
from .definitions.deepseek_v32 import DeepSeekV32QModel  # noqa: E402
from .definitions.raon1_mini import Raon1MiniQModel  # noqa: E402
from .definitions.flint import FlintQModel  # noqa: E402
from .definitions.ricursive import RicursiveQModel  # noqa: E402
from .definitions.deepseek_v4 import DeepSeekV4QModel  # noqa: E402
from .definitions.deepseek_ocr2 import DeepSeekOCR2QModel  # noqa: E402
from .definitions.deepseek_vl import DeepSeekVLQModel  # noqa: E402
from .definitions.deepseek_vl_v2 import DeepSeekVLV2QModel  # noqa: E402
from .definitions.diffusion_gemma import DiffusionGemmaQModel  # noqa: E402
from .definitions.dots1 import Dots1QModel  # noqa: E402
from .definitions.dream import DreamQModel  # noqa: E402
from .definitions.ernie4_5 import Ernie4_5QModel  # noqa: E402
from .definitions.ernie4_5_moe import Ernie4_5_MoeQModel  # noqa: E402
from .definitions.ernie4_5_vl_moe import Ernie4_5_VLMoeQModel  # noqa: E402
from .definitions.exaone import ExaOneQModel  # noqa: E402
from .definitions.exaone4 import Exaone4QModel, Exaone4_5QModel  # noqa: E402
from .definitions.falcon_h1 import FalconH1QModel  # noqa: E402
from .definitions.falcon_mamba import FalconMambaQModel  # noqa: E402
from .definitions.gemma2 import Gemma2QModel  # noqa: E402
from .definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ, Gemma3QModel  # noqa: E402
from .definitions.gemma3n import Gemma3nForConditionalGenerationGPTQ, Gemma3nTextQModel  # noqa: E402
from .definitions.gemma4 import Gemma4ForConditionalGenerationGPTQ, Gemma4TextQModel  # noqa: E402
from .definitions.gemma4_unified import Gemma4UnifiedForConditionalGenerationGPTQ, Gemma4UnifiedTextQModel  # noqa: E402
from .definitions.glm import GlmQModel  # noqa: E402
from .definitions.glm4_moe import GLM4MoEGPTQ  # noqa: E402
from .definitions.glm4_moe_lite import Glm4MoeLiteQModel  # noqa: E402
from .definitions.glm4v import Glm4vGPTQ  # noqa: E402
from .definitions.glm4v_moe import Glm4vMoeQModel, Glm4vMoeTextQModel  # noqa: E402
from .definitions.glm_moe_dsa import GlmMoeDsaQModel  # noqa: E402
from .definitions.glm5_next import Glm5NextQModel  # noqa: E402
from .definitions.glm_ocr import GlmOCRGPTQ  # noqa: E402
from .definitions.glmasr import GlmASRGPTQ  # noqa: E402
from .definitions.gpt2 import GPT2QModel  # noqa: E402
from .definitions.gpt_bigcode import GptBigCodeQModel  # noqa: E402
from .definitions.gpt_neo import GptNeoQModel  # noqa: E402
from .definitions.gpt_neox import GPTNeoXQModel  # noqa: E402
from .definitions.gpt_oss import GPTOSSGPTQ  # noqa: E402
from .definitions.gptj import GptJQModel  # noqa: E402
from .definitions.granitemoe import GraniteMoeQModel
from .definitions.granitemoehybrid import GraniteMoeHybridQModel
from .definitions.granitemoe_swa import GraniteMoeSWAQModel
from .definitions.granite_swa import GraniteSWAQModel
from .definitions.grinmoe import GrinMoeQModel  # noqa: E402
from .definitions.hrm_text import HrmTextQModel  # noqa: E402
from .definitions.hunyuan_v1_dense import HunYuanDenseV1QModel  # noqa: E402
from .definitions.hunyuan_v1_moe import HunYuanMoEV1QModel  # noqa: E402
from .definitions.hunyuan_vl import HunYuanVLQModel  # noqa: E402
from .definitions.hy_v3 import HYV3QModel  # noqa: E402
from .definitions.hymba import HymbaQModel  # noqa: E402
from .definitions.instella import InstellaQModel  # noqa: E402
from .definitions.inkling import InklingMMQModel  # noqa: E402
from .definitions.internlm import InternLMQModel  # noqa: E402
from .definitions.internlm2 import InternLM2QModel  # noqa: E402
from .definitions.intern_s2_preview import InternS2PreviewQModel  # noqa: E402
from .definitions.interns1 import InternS1QModel  # noqa: E402
from .definitions.internvl import InternVLQModel  # noqa: E402
from .definitions.internvl_chat import InternVLChatQModel  # noqa: E402
from .definitions.jamba import JambaQModel  # noqa: E402
from .definitions.k2_horizon import K2HorizonQModel  # noqa: E402
from .definitions.klear import KlearQModel  # noqa: E402
from .definitions.kimi_k25 import KimiK25QModel  # noqa: E402
from .definitions.kimi_k3 import KimiK3QModel, KimiLinearQModel  # noqa: E402
from .definitions.kimi_linear import KimiLinearNativeQModel, is_kimi_k3_config  # noqa: E402
from .definitions.laguna import LagunaQModel  # noqa: E402
from .definitions.lfm2 import LFM2QModel  # noqa: E402
from .definitions.lfm2_moe import LFM2MoeQModel  # noqa: E402
from .definitions.lfm2_vl import LFM2VLQModel  # noqa: E402
from .definitions.lfm3 import LFM3QModel  # noqa: E402
from .definitions.lfm3_omni import LFM3OmniQModel  # noqa: E402
from .definitions.lfm3_vl import LFM3VLQModel  # noqa: E402
from .definitions.llada2 import LLaDA2MoeQModel
from .definitions.llama import LlamaQModel  # noqa: E402
from .definitions.llava import LlavaQModel  # noqa: E402
from .definitions.llama4 import Llama4QModel, Llama4TextQModel  # noqa: E402
from .definitions.llava_qwen2 import LlavaQwen2QModel  # noqa: E402
from .definitions.longcat_flash import LongCatFlashQModel  # noqa: E402
from .definitions.locateanything import LocateAnythingQModel  # noqa: E402
from .definitions.mage_vl import MageVLQModel  # noqa: E402
from .definitions.mimo import MimoQModel  # noqa: E402
from .definitions.mimo_v2 import MimoV2QModel  # noqa: E402
from .definitions.mimo_v2_flash import MimoV2FlashQModel  # noqa: E402
from .definitions.minicpm import MiniCPMGPTQ  # noqa: E402
from .definitions.minicpm3 import MiniCpm3QModel  # noqa: E402
from .definitions.minicpm_o import MiniCPMOQModel  # noqa: E402
from .definitions.minicpmv import MiniCPMVQModel  # noqa: E402
from .definitions.minicpmv_4_6 import MiniCPMV4_6QModel  # noqa: E402
from .definitions.minimax_m2 import MiniMaxM2GPTQ, MiniMaxQModel  # noqa: E402
from .definitions.minimax_m3_vl import MiniMaxM3VLGPTQ  # noqa: E402
from .definitions.ministral3 import Ministral3GPTQ  # noqa: E402
from .definitions.mistral3 import Mistral3GPTQ
from .definitions.mixtral import MixtralQModel  # noqa: E402
from .definitions.mllama import MLlamaQModel, MLlamaTextQModel  # noqa: E402
from .definitions.mobilellm import MobileLLMQModel  # noqa: E402
from .definitions.moss import MossQModel  # noqa: E402
from .definitions.mpt import MptQModel  # noqa: E402
from .definitions.muse_glimmer import MuseGlimmerQModel  # noqa: E402
from .definitions.nanbeige import NanbeigeQModel  # noqa: E402
from .definitions.nemotron_labs_diffusion import NemotronLabsDiffusionQModel  # noqa: E402
from .definitions.nemotron_h import NemotronHQModel  # noqa: E402
from .definitions.nemotron_h_puzzle import NemotronHPuzzleQModel  # noqa: E402
from .definitions.nemotron_omni import NemotronOmniQModel  # noqa: E402
from .definitions.olmo2 import Olmo2QModel  # noqa: E402
from .definitions.olmo3 import Olmo3QModel  # noqa: E402
from .definitions.olmo_hybrid import OlmoHybridQModel  # noqa: E402
from .definitions.olmoe import OlmoeQModel  # noqa: E402
from .definitions.openvla import OpenVLAQModel  # noqa: E402
from .definitions.opt import OptQModel  # noqa: E402
from .definitions.ouro import OuroQModel  # noqa: E402
from .definitions.ovis import OvisQModel  # noqa: E402
from .definitions.ovis2 import Ovis2QModel  # noqa: E402
from .definitions.ovis2_5 import Ovis2_5QModel  # noqa: E402
from .definitions.ovis2_6_moe import Ovis2_6_MoeQModel  # noqa: E402
from .definitions.ovis2_6_moe import Ovis2_6_NextQModel  # noqa: E402
from .definitions.pangu_alpha import PanguAlphaQModel  # noqa: E402
from .definitions.phi import PhiQModel  # noqa: E402
from .definitions.phi3 import Phi3QModel, PhiMoEGPTQForCausalLM  # noqa: E402
from .definitions.phi4 import Phi4MMGPTQ  # noqa: E402
from .definitions.qwen import QwenQModel  # noqa: E402
from .definitions.qwen2 import Qwen2QModel  # noqa: E402
from .definitions.qwen2_5_omni import Qwen2_5_OmniGPTQ
from .definitions.qwen2_5_vl import Qwen2_5_VLQModel  # noqa: E402
from .definitions.qwen2_audio import Qwen2AudioGPTQ  # noqa: E402
from .definitions.qwen2_moe import Qwen2MoeQModel  # noqa: E402
from .definitions.qwen2_vl import Qwen2VLQModel  # noqa: E402
from .definitions.qwen3 import Qwen3QModel  # noqa: E402
from .definitions.qwen3_moe import Qwen3MoeQModel  # noqa: E402
from .definitions.qwen3_next import Qwen3NextGPTQ  # noqa: E402
from .definitions.qwen3_omni_moe import Qwen3OmniMoeGPTQ
from .definitions.qwen3_vl import Qwen3_VLQModel, Qwen3_VL_MoeQModel
from .definitions.qwen_drive import QwenDriveQModel  # noqa: E402
from .definitions.qwen4_exp import Qwen4ExpQModel, Qwen4ExpTextQModel  # noqa: E402
from .definitions.rw import FalconQModel, RwgQModel  # noqa: E402
from .definitions.solar_open import SolarOpenQModel  # noqa: E402
from .definitions.solar_open2 import SolarOpen2QModel  # noqa: E402
from .definitions.smolvlm import SmolVLMQModel  # noqa: E402
from .definitions.spark2_5 import Spark2_5QModel  # noqa: E402
from .definitions.starcoder2 import Starcoder2QModel  # noqa: E402
from .definitions.telechat2 import TeleChat2QModel
from .definitions.unlimited_ocr import UnlimitedOCRQModel  # noqa: E402
from .definitions.voxtral import VoxtralGPTQ  # noqa: E402
from .definitions.xing4_0 import Xing4_0QModel  # noqa: E402
from .definitions.xverse import XverseQModel  # noqa: E402
from .definitions.zamba import ZambaQModel  # noqa: E402
from .definitions.zamba2 import Zamba2QModel  # noqa: E402


TRANSFORMERS_SUPPORTS_QWEN3_5 = Version(TRANSFORMERS_VERSION) >= Version("5.2.0")
if TRANSFORMERS_SUPPORTS_QWEN3_5:
    from .definitions.qwen3_5 import Qwen3_5QModel  # noqa: E402
    from .definitions.qwen3_5_moe import Qwen3_5_MoeQModel  # noqa: E402
    from .definitions.qwen3_5_moe_text import Qwen3_5_MoeTextQModel  # noqa: E402
    from .definitions.qwen3_5_text import Qwen3_5TextQModel  # noqa: E402
    from .definitions.zdtaichu5 import ZDTaichu5QModel  # noqa: E402
else:
    Qwen3_5QModel = None
    Qwen3_5_MoeQModel = None
    Qwen3_5_MoeTextQModel = None
    Qwen3_5TextQModel = None
    ZDTaichu5QModel = None


MODEL_MAP = {
    "dots3_note": Dots3NoteQModel,
    "agnes": AgnesQModel,
    "apertus": ApertusQModel,
    "apertus1p5": Apertus1p5QModel,
    "apertus1p5_text": Apertus1p5TextQModel,
    "axk2": AXK2QModel,
    "diffusion_gemma": DiffusionGemmaQModel,
    "dream": DreamQModel,
    "bloom": BloomQModel,
    "brumby": BrumbyQModel,
    "gpt_neo": GptNeoQModel,
    "kimi_k2": DeepSeekV3QModel, # 100% DeepSeekV3QModel clone
    "kimi_k25": KimiK25QModel,
    "kimi_k3": KimiK3QModel,
    "kimi_linear": KimiLinearQModel,
    "klear": KlearQModel,
    "laguna": LagunaQModel,
    "gpt_neox": GPTNeoXQModel,
    "gptj": GptJQModel,
    "gpt2": GPT2QModel,
    "llama": LlamaQModel,
    "smollm3": LlamaQModel,  # llama-compatible quantization module tree
    "smolvlm": SmolVLMQModel,
    "llama4": Llama4QModel,
    "llama4_text": Llama4TextQModel,
    "opt": OptQModel,
    "moss": MossQModel,
    "chatglm": ChatGLMQModel,
    "glm": GlmQModel,
    "glm4": GlmQModel,
    "glm4v": Glm4vGPTQ,
    "glm4v_moe": Glm4vMoeQModel,
    "glm4v_moe_text": Glm4vMoeTextQModel,
    "glmasr": GlmASRGPTQ,
    "glm_ocr": GlmOCRGPTQ,
    "glm4_moe": GLM4MoEGPTQ,
    "glm4_moe_lite": Glm4MoeLiteQModel,
    "glm_moe_dsa": GlmMoeDsaQModel,
    "glm5_next": Glm5NextQModel,
    "gpt_bigcode": GptBigCodeQModel,
    "codegen": CodeGenQModel,
    "cohere": LlamaQModel, # 100% llama clone
    "cohere2": LlamaQModel, # 100% llama clone
    "cohere2_moe": Cohere2MoeQModel,
    "cohere_compass": CohereCompassQModel,
    "refinedWebModel": RwgQModel,
    "refinedWeb": RwgQModel,
    "falcon": FalconQModel,
    "falcon_mamba": FalconMambaQModel,
    "baichuan": BaiChuanQModel,
    "internlm": InternLMQModel,
    "internlm2": InternLM2QModel,
    "intern_s2_preview": InternS2PreviewQModel,
    "interns1": InternS1QModel,
    "internvl": InternVLQModel,
    "internvl_chat": InternVLChatQModel,
    "hrm_text": HrmTextQModel,
    "hunyuan_v1_dense": HunYuanDenseV1QModel,
    "hunyuan_v1_moe": HunYuanMoEV1QModel,
    "hunyuan_vl": HunYuanVLQModel,
    "hy_v3": HYV3QModel,
    "jamba": JambaQModel,
    "k2_horizon": K2HorizonQModel,
    "qwen": QwenQModel,
    "mistral": LlamaQModel, # 100% llama clone
    "yi": LlamaQModel, # 100% llama clone
    "xverse": XverseQModel,
    "xing4_0": Xing4_0QModel,
    "deci": DeciLMQModel,
    "nemotron-nas": DeciLMQModel,
    "stablelm_epoch": LlamaQModel, # 100% llama clone
    "stablelm": LlamaQModel, # 100% llama clone
    "starcoder2": Starcoder2QModel,
    "mixtral": MixtralQModel,
    "qwen2": Qwen2QModel,
    "qwen3": Qwen3QModel,
    "qwen_drive": QwenDriveQModel,
    "longllama": LlamaQModel,  # 100% llama clone
    "gemma": LlamaQModel, # 100% llama clone
    "gemma2": Gemma2QModel,
    "gemma3_text": Gemma3QModel,
    "gemma3": Gemma3ForConditionalGenerationGPTQ,
    "gemma3n_text": Gemma3nTextQModel,
    "gemma3n": Gemma3nForConditionalGenerationGPTQ,
    "gemma4_text": Gemma4TextQModel,
    "gemma4": Gemma4ForConditionalGenerationGPTQ,
    "gemma4_unified_text": Gemma4UnifiedTextQModel,
    "gemma4_unified": Gemma4UnifiedForConditionalGenerationGPTQ,
    "phi": PhiQModel,
    "phi3": Phi3QModel,
    "phi4mm": Phi4MMGPTQ,
    "phimoe": PhiMoEGPTQForCausalLM,
    "mpt": MptQModel,
    "minicpm": MiniCPMGPTQ,
    "minicpm3": MiniCpm3QModel,
    "minicpmo": MiniCPMOQModel,
    "minicpmv": MiniCPMVQModel,
    "minicpmv4_6": MiniCPMV4_6QModel,
    "minimax": MiniMaxQModel,
    "minimax_m2": MiniMaxM2GPTQ,
    "minimax_m3_vl": MiniMaxM3VLGPTQ,
    "muse_glimmer": MuseGlimmerQModel,
    "nanbeige": NanbeigeQModel,
    "ministral3": Ministral3GPTQ,
    "qwen2_moe": Qwen2MoeQModel,
    "qwen3_moe": Qwen3MoeQModel,
    "qwen3_next": Qwen3NextGPTQ,
    "qwen2_vl": Qwen2VLQModel,
    "qwen2_vl_text": Qwen2VLQModel,
    "qwen2_5_vl": Qwen2_5_VLQModel,
    "qwen2_5_vl_text": Qwen2_5_VLQModel,
    "qwen2_5_omni": Qwen2_5_OmniGPTQ,
    "qwen2_audio": Qwen2AudioGPTQ,
    "qwen3_omni_moe": Qwen3OmniMoeGPTQ,
    "qwen3_vl": Qwen3_VLQModel,
    "qwen3_vl_moe": Qwen3_VL_MoeQModel,
    "qwen4_exp": Qwen4ExpQModel,
    "qwen4_exp_text": Qwen4ExpTextQModel,
    "dbrx": DbrxQModel,
    "dbrx_converted": DbrxConvertedQModel,
    "deepseek_v2": DeepSeekV2QModel,
    "deepseek_v3": DeepSeekV3QModel,
    "deepseek_v32": DeepSeekV32QModel,
    "raon1_mini": Raon1MiniQModel,
    "flint": FlintQModel,
    "ricursive": RicursiveQModel,
    "deepseek_v4": DeepSeekV4QModel,
    "deepseek_ocr2": DeepSeekOCR2QModel,
    "deepseek_vl": DeepSeekVLQModel,
    "deepseek_vl_v2": DeepSeekVLV2QModel,
    "dots1": Dots1QModel,
    "exaone": ExaOneQModel,
    "exaone4": Exaone4QModel,
    "exaone4_5": Exaone4_5QModel,
    "exaone4_5_text": Exaone4QModel,
    "grinmoe": GrinMoeQModel,
    "mllama": MLlamaQModel,
    "mllama_text_model": MLlamaTextQModel,
    "marin": Qwen3QModel,
    "mage_vl": MageVLQModel,
    "granite": LlamaQModel, # 100% llama clone
    "granite_swa": GraniteSWAQModel,
    "granitemoe": GraniteMoeQModel,
    "granitemoehybrid": GraniteMoeHybridQModel,
    "granitemoe_swa": GraniteMoeSWAQModel,
    "mobilellm": MobileLLMQModel,
    "hymba": HymbaQModel,
    "olmo": LlamaQModel, # 100% llama clone (OlmoModel == LlamaModel module layout)
    "olmo2": Olmo2QModel,
    "olmo3": Olmo3QModel,
    "olmo_hybrid": OlmoHybridQModel,
    "helium": LlamaQModel, # 100% llama clone (HeliumModel == LlamaModel module layout)
    "olmoe": OlmoeQModel,
    "openvla": OpenVLAQModel,
    "ouro": OuroQModel,
    "ovis": OvisQModel,
    "ovis2": Ovis2QModel,
    "ovis2_5": Ovis2_5QModel,
    "ovis2_6_moe": Ovis2_6_MoeQModel,
    "ovis2_6_next": Ovis2_6_NextQModel,
    "telechat": TeleChat2QModel,
    "instella": InstellaQModel,
    "inkling_mm_model": InklingMMQModel,
    "mimo": MimoQModel,
    "mimo_v2": MimoV2QModel,
    "mimo_v2_flash": MimoV2FlashQModel,
    "falcon_h1": FalconH1QModel,
    "zamba": ZambaQModel,
    "zamba2": Zamba2QModel,
    "gpt_pangu": PanguAlphaQModel,
    "ernie4_5": Ernie4_5QModel,
    "ernie4_5_moe": Ernie4_5_MoeQModel,
    "ernie4_5_moe_vl": Ernie4_5_VLMoeQModel, # Backward Compatibility alias
    "ernie4_5_vl_moe": Ernie4_5_VLMoeQModel,
    "seed_oss": LlamaQModel, # 100% llama clone
    "solar_open": SolarOpenQModel,
    "solar_open2": SolarOpen2QModel,
    "spark2_5": Spark2_5QModel,
    "gpt_oss": GPTOSSGPTQ,
    "longcat_flash": LongCatFlashQModel,
    "locateanything": LocateAnythingQModel,
    "llava": LlavaQModel,
    "llava_qwen2": LlavaQwen2QModel,
    "nemotron_h": NemotronHQModel,
    "nemotron_h_puzzle": NemotronHPuzzleQModel,
    "nemotron_labs_diffusion": NemotronLabsDiffusionQModel,
    "nemotronh_nano_omni_reasoning_v3": NemotronOmniQModel,
    "bailing_moe": BailingMoeQModel,
    "bailing_hybrid": BailingMoeQModel,
    "lfm2": LFM2QModel,
    "lfm2_moe": LFM2MoeQModel,
    "lfm2_vl": LFM2VLQModel,
    "lfm3": LFM3QModel,
    "lfm3_omni": LFM3OmniQModel,
    "lfm3_vl": LFM3VLQModel,
    "llada2_moe": LLaDA2MoeQModel,
    "mistral3": Mistral3GPTQ,
    "afmoe": AfMoeQModel,
    # The published baidu/Unlimited-OCR config uses `unlimited_ocr`; keep the
    # hyphenated key so older local configs keep resolving.
    "unlimited_ocr": UnlimitedOCRQModel,
    "unlimited-ocr": UnlimitedOCRQModel,
    "voxtral": VoxtralGPTQ,
}

if Qwen3_5QModel is not None:
    MODEL_MAP["qwen3_5"] = Qwen3_5QModel
    MODEL_MAP["qwen3_5_text"] = Qwen3_5TextQModel

if Qwen3_5_MoeQModel is not None:
    MODEL_MAP["qwen3_5_moe"] = Qwen3_5_MoeQModel
    MODEL_MAP["qwen3_5_moe_text"] = Qwen3_5_MoeTextQModel

if ZDTaichu5QModel is not None:
    MODEL_MAP["zdtaichu5_0"] = ZDTaichu5QModel

SUPPORTED_MODELS = list(MODEL_MAP.keys())

# A few legacy registrations retain camelCase model types (for example
# RefinedWeb). The casing of `config.model_type` is not part of the lookup
# contract, so resolve definitions case-insensitively instead of letting
# `.lower()` miss a registered key.
MODEL_MAP_CASEFOLD = {
    model_type.lower(): definition for model_type, definition in MODEL_MAP.items()
}


def _unsupported_quantization_metadata(quantization_config: dict):
    """Return the first unsupported quantization feature in this payload."""

    return find_unsupported_metadata(quantization_config)


def _is_supported_quantization_config(config: AutoConfig) -> bool:
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        return False

    # Fail fast before model selection so checkpoints that expect activation or
    # KV-cache quantization do not proceed down a weight-only loader path.
    unsupported = _unsupported_quantization_metadata(quantization_config)
    if unsupported is not None:
        message = describe_unsupported_metadata(unsupported)
        log.error(message)
        raise ValueError(message)

    quant_format = quantization_config.get("quant_format")
    if isinstance(quant_format, str) and quant_format.lower() in (
        METHOD.GPTQ,
        METHOD.GGUF,
        METHOD.FP8,
        METHOD.BITSANDBYTES,
        METHOD.AWQ,
        METHOD.PARO,
        METHOD.QQQ,
        METHOD.VPTQ,
        METHOD.MACAQ,
        METHOD.NF12,
        METHOD.EXL3,
        METHOD.TT_BLOCKFLOAT,
    ):
        return True

    method = quantization_config.get("method", quantization_config.get("quant_method"))
    if isinstance(method, str) and method.lower() in (
        METHOD.GPTQ,
        METHOD.GGUF,
        METHOD.FP8,
        METHOD.BITSANDBYTES,
        METHOD.AWQ,
        METHOD.PARO,
        METHOD.QQQ,
        METHOD.VPTQ,
        METHOD.MACAQ,
        METHOD.NF12,
        METHOD.EXL3,
        METHOD.TT_BLOCKFLOAT,
    ):
        return True

    return False


def _uses_auto_module_decoder(quantize_config: Optional[QuantizeConfig]) -> bool:
    """Return whether an explicit target config decodes a floatx source checkpoint."""

    if quantize_config is None:
        return False
    return any(
        isinstance(preprocessor, AutoModuleDecoderConfig)
        for preprocessor in (getattr(quantize_config, "preprocessors", None) or [])
    )


@contextmanager
def _hide_unsupported_quantization_config_for_eval(model):
    config = getattr(model, "config", None)
    if config is None:
        yield
        return

    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        yield
        return

    try:
        from transformers.quantizers import AutoQuantizationConfig

        AutoQuantizationConfig.from_dict(dict(quantization_config))
    except Exception:
        pass
    else:
        yield
        return

    setattr(config, "quantization_config", None)
    try:
        yield
    finally:
        setattr(config, "quantization_config", quantization_config)


@contextmanager
def _hide_unsupported_quantization_config_for_lm_eval(model):
    with _hide_unsupported_quantization_config_for_eval(model):
        yield


_LLAVA_HF_ARCHITECTURES = frozenset({"LlavaForConditionalGeneration"})


def _is_original_llava_config(config) -> bool:
    """Return whether ``config`` belongs to an original (non-HF) LLaVA checkpoint.

    Both the Hugging Face checkpoint and the original liuhaotian checkpoint use
    ``model_type: llava``. The Hugging Face one declares
    ``LlavaForConditionalGeneration``; the original one declares a plain
    causal-LM architecture (``LlavaLlamaForCausalLM``) and points at its vision
    tower through ``mm_vision_tower``, so building the Hugging Face shell from
    it fails deep inside the loader. ``text_config`` cannot be used here because
    ``AutoConfig`` fills it with defaults for both layouts.
    """

    architectures = getattr(config, "architectures", None) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    architectures = [str(arch) for arch in architectures]

    if any(arch in _LLAVA_HF_ARCHITECTURES for arch in architectures):
        return False

    if any(arch.endswith("ForCausalLM") for arch in architectures):
        return True

    return bool(getattr(config, "mm_vision_tower", None))


def _get_config_load_kwargs(kwargs: dict) -> dict:
    """Config-selection kwargs for model-type detection.

    Detection must read the same config file the loader will read. Forwarding
    only ``gguf_file`` made a nested checkpoint (``subfolder=...``) or a pinned
    ``revision`` resolve against the repository root/default revision here, so
    dispatch raised "Unrecognized model" even though the loader could load it.
    """

    config_kwargs = get_config_selection_kwargs(kwargs)
    config_kwargs.update(get_hf_gguf_load_kwargs(kwargs))
    return config_kwargs


def check_and_get_model_definition(model_dir, trust_remote_code=False, **config_load_kwargs):
    if "gguf_file" not in config_load_kwargs:
        model_dir = normalize_model_id_or_path_for_hf_gguf(
            model_dir,
            config_load_kwargs,
            api_name="check_and_get_model_definition",
        )
    trust_remote_code = resolve_trust_remote_code(model_dir, trust_remote_code=trust_remote_code)
    patch_remote_code_before_config_load(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code, **config_load_kwargs)
    model_type = config.model_type.lower()

    # Unsupported model types use BaseQModel and fail closed unless the caller
    # supplies an explicit model definition/module tree.
    definition = MODEL_MAP_CASEFOLD.get(model_type, BaseQModel)

    # `model_type: llava` covers both the Hugging Face composite checkpoint and
    # the original liuhaotian checkpoint, which declares a plain causal-LM
    # architecture and cannot be built as a `LlavaForConditionalGeneration`
    # shell (issue #4229).
    if model_type == "llava" and _is_original_llava_config(config):
        raise ValueError(
            f"Original (non-HF) LLaVA checkpoints are not supported: `{model_dir}` "
            "resolves to `model_type=llava` with architecture(s) "
            f"{list(getattr(config, 'architectures', None) or [])}, which is not a "
            "Hugging Face `LlavaForConditionalGeneration` checkpoint. Use the "
            "Hugging Face format instead, e.g. `llava-hf/llava-1.5-7b-hf`."
        )

    # `kimi_linear` covers two incompatible layouts: the legacy AttnRes/K3
    # checkpoint that needs the bounded K3 runner, and native Transformers
    # `KimiLinearForCausalLM` (no AttnRes), which quantizes with the ordinary
    # looper. Dispatch on config fields rather than on the registry key.
    if model_type == "kimi_linear" and not is_kimi_k3_config(config):
        return KimiLinearNativeQModel

    return definition


def _explicit_prewarm_dtype(kwargs) -> Optional[torch.dtype]:
    """
    Return the dtype the caller pinned explicitly, if any.

    Only real `torch.dtype` values count: the loader's own dtype resolution ignores
    string dtypes, so treating one as a decision here would warm the wrong dtype.
    """
    for key in ("dtype", "torch_dtype"):
        value = kwargs.get(key)
        if isinstance(value, torch.dtype):
            return value
    return None


def _config_prewarm_dtype(model_cfg, target: torch.device) -> Optional[torch.dtype]:
    """Mirror the loader's own dtype resolution so the prewarm warms the dtype the model will use."""
    try:
        family = normalize_device(target)
        return auto_dtype(config=model_cfg, device=family, quant_inference=False)
    except Exception as exc:
        log.debug(f"linear prewarm dtype unresolved for {model_cfg}: {exc}")
        return None


def _quantize_warmup_devices(quantize_config, device) -> Optional[List[torch.device]]:
    """Concrete accelerators this load may warm, or ``None`` for the old pool.

    The loopers narrow the quantization/forward pool with
    ``quantize_config.compute_device_filter``, but the eager linalg warmup and
    the quantize/linear prewarms were resolved from the ``device=``/``device_map=``
    load kwargs alone. A caller who excluded a GPU through the documented filter
    therefore still warmed that GPU, which created a full context there and
    reserved hundreds of MiB (issue #4653).

    Return the same filtered pool the loopers use. ``None`` keeps the historical
    family-wide warmup whenever this load has no restriction to honor (no filter
    and no explicit device sequence), or when the filter rejects everything --
    the loopers keep the full pool in that case, so the warmup must match.

    ``device_map`` is not consulted on purpose: a quantization load runs the
    dense model on CPU and places layers through ``quantize_config``, so a map
    describes inference placement rather than the quantization pool.
    """

    explicit_devices = _explicit_requested_devices(device)
    compute_device_filter = getattr(quantize_config, "compute_device_filter", None)
    if explicit_devices is None and compute_device_filter is None:
        return None

    base_device = None
    configured = getattr(quantize_config, "device", None)
    if configured is not None:
        base_device = normalize_device_like(configured)
    elif explicit_devices:
        base_device = explicit_devices[0]
    elif device is not None:
        base_device = normalize_device_like(device)
    else:
        base_device = normalize_device_like(
            auto_select_device(None, None)
        )

    candidates = select_forward_devices(base_device)
    if compute_device_filter is not None:
        filtered = compute_device_filter(candidates)
        # The loopers keep the full pool when the filter rejects everything.
        if not filtered:
            return None
        candidates = list(filtered)
    elif explicit_devices is not None:
        wanted = {str(target) for target in explicit_devices}
        subsets = [target for target in candidates if str(target) in wanted]
        if not subsets:
            return None
        candidates = subsets

    devices: List[torch.device] = []
    for candidate in candidates:
        target = normalize_device_like(candidate)
        if target is not None and target not in devices:
            devices.append(target)
    return devices or None


def _resolve_prewarm_device(device, device_map, quantize_config=None) -> Optional[torch.device]:
    warmup_pool = _quantize_warmup_devices(quantize_config, device)
    if warmup_pool:
        return warmup_pool[0]

    candidates = []
    if device is not None:
        candidates.append(device)
    if isinstance(device_map, str):
        candidates.append(device_map)
    elif isinstance(device_map, dict):
        candidates.extend(device_map.values())
    if not candidates:
        # No explicit placement: the loader auto-selects the first accelerator.
        return torch.device("cuda")
    for candidate in candidates:
        if isinstance(candidate, int):
            candidate = f"cuda:{candidate}"
        try:
            resolved = torch.device(candidate)
        except (RuntimeError, TypeError, ValueError):
            continue
        if resolved.type in ("cuda", "xpu", "npu"):
            return resolved
    return None


def _kick_quantize_prewarms(device, device_map, kwargs, model_cfg=None, quantize_config=None) -> None:
    """
    Hide the one-time accelerator library and kernel inits behind the model load.

    The first `linear` forward of the quantization pass otherwise pays the full
    one-time library init inline (measured ~1.2 s on ROCm), and the first launch
    of each distinct quantization kernel otherwise pays an accelerator module
    load inline (measured 1.5 s of a 2.25 s cold pass on MI350X). Starting both
    on background threads while the CPU-bound model load is still running moves
    that cost off the critical path of the quantization pass.

    Without `model_cfg` only an explicitly pinned dtype is warmed: the config that
    decides the default dtype has not been read yet, and guessing a dtype would
    spend the init on a warmup the first forward cannot use.
    """
    target = _resolve_prewarm_device(device, device_map, quantize_config)
    if target is None:
        return
    dtype = _explicit_prewarm_dtype(kwargs)
    if dtype is None and model_cfg is not None:
        dtype = _config_prewarm_dtype(model_cfg, target)
    if dtype is None:
        return
    kick_torch_linear_prewarm(target, dtype)
    kick_torch_quantize_prewarm(target, dtype)


def _prebuild_device_thread_pool(device=None, quantize_config=None) -> None:
    """
    Build the device thread pool before the quantization loop starts.

    Pool construction kicks the one-time accelerator warmups on its workers (see
    `DeviceThreadPool._kick_worker_warmups`). Building the pool at load time lets
    those warmups overlap with the (CPU-bound) model load instead of serializing
    in front of the first quantization task.

    When the caller named concrete accelerators (`device=("cuda:1",)`), only
    those devices may be warmed. The pool still knows about every visible
    device, but the excluded ones keep no CUDA context and no linalg warmup
    allocations until work is actually submitted to them.
    """
    from .. import resize_device_thread_pool

    warmup_devices = _quantize_warmup_devices(quantize_config, device)
    if not env_flag("GPTQMODEL_POOL_PREWARM", default=True):
        # Do not build eagerly, but still retune a pool an earlier load created.
        resize_device_thread_pool(warmup_devices=warmup_devices, create=False)
        return

    # Resize the singleton to the policy installed for *this* run. A later load
    # that asks for a different ``cpu_workers`` must not inherit the first
    # load's worker count (issue #4396).
    pool = resize_device_thread_pool(warmup_devices=warmup_devices)
    # A retained pool keeps the worker threads (and their already-installed
    # policy) but must still honor this load's warmup allow-list: the eager
    # warmup is submitted once, when the pool is built.
    setter = getattr(pool, "set_warmup_devices", None)
    if callable(setter):
        setter(warmup_devices)


def _install_kernel_policy(kernel) -> None:
    """Install a load's process policy and report retired env vars set since import."""
    from .. import warn_retired_kernel_env_vars

    configure_kernel(kernel)
    warn_retired_kernel_env_vars()


class GPTQModel:
    def __init__(self):
        raise EnvironmentError(
            "GPT-QModel is not designed to be instantiated\n"
            "use `from_pretrained()` to load a pretrained model and prepare for quantization via `.quantize()`.\n"
            "use `from_quantized()` for inference with a post-quantized model."
        )

    @staticmethod
    def quantize_mlx(model_id_or_path: str, output_path: str, method: str = "gptq", **kwargs):
        """Quantize a dense model directly in MLX and save an MLX-LM checkpoint.

        This Apple silicon path uses a fused Metal GPTQ update or MLX-LM's
        native AWQ implementation. It does not load a Torch model.
        """
        from ..quantization.mlx_native import quantize_mlx

        return quantize_mlx(model_id_or_path, output_path, method=method, **kwargs)

    @classmethod
    def load(
            cls,
            model_id_or_path: Optional[str],
            quantize_config: Optional[QuantizeConfig | Dict] = None,
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, torch.device]] = None,
            backend: Backend = BACKEND.AUTO,
            profile: Union[str, int, PROFILE] = PROFILE.AUTO,
            trust_remote_code: bool = False,
            allow_remote_adapter: bool = False,
            adapter_revision: Optional[str] = None,
            adapter_sha256: Optional[str] = None,
            fuse: bool = False,
            p2p: Optional[Union[P2PConfig, Dict]] = None,
            kernel: Optional[Union[KernelConfig, Dict]] = None,
            module_tree: Optional[Union[str, List[Any], tuple]] = None,
            runtime: Optional[RuntimeConfig] = None,
            **kwargs,
    ):
        if type(fuse) is not bool:
            raise TypeError("Fusion enabled must be bool")
        kwargs["fuse"] = fuse
        if isinstance(quantize_config, Dict):
            quantize_config = QuantizeConfig(**quantize_config)
        # from_pretrained resolves the caller's own arguments against the same
        # config again; only the already-quantized redirect forwards the
        # resolved enum/policy pair, because it does not receive the config.
        requested_backend, requested_kernel = backend, kernel
        backend, kernel = resolve_quantize_backend(backend, quantize_config, kernel, runtime)
        _install_kernel_policy(kernel)
        if quantize_config is not None:
            _kick_quantize_prewarms(device, device_map, kwargs, quantize_config=quantize_config)
            _prebuild_device_thread_pool(device, quantize_config=quantize_config)

        model_id_or_path = normalize_model_id_or_path_for_hf_gguf(
            model_id_or_path,
            kwargs,
            api_name="GPTQModel.load",
        )
        if isinstance(model_id_or_path, str):
            model_id_or_path = model_id_or_path.strip()
        requested_trust_remote_code = trust_remote_code
        trust_remote_code = resolve_trust_remote_code(model_id_or_path, trust_remote_code=trust_remote_code)

        # normalize config to cfg instance
        if isinstance(quantize_config, Dict):
            quantize_config = QuantizeConfig(**quantize_config)
        # The P2P policy lives on the quantize config for quantization; allow the
        # same object here so a post-quantized load can control it without env vars.
        if p2p is not None:
            if quantize_config is not None:
                quantize_config.p2p = P2PConfig(**p2p) if isinstance(p2p, dict) else p2p
        elif quantize_config is not None:
            p2p = quantize_config.p2p

        backend = normalize_backend(backend)
        profile = normalize_profile(profile)

        is_gptqmodel_quantized = False
        treat_as_local_path = isinstance(model_id_or_path, str) and (
            isdir(model_id_or_path) or os.path.isabs(model_id_or_path)
        )

        model_cfg = None
        if not (treat_as_local_path and not isdir(model_id_or_path)):
            patch_remote_code_before_config_load(model_id_or_path)
            model_cfg = AutoConfig.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
                **_get_config_load_kwargs(kwargs),
            )

        native_floatx_source = (
            native_floatx_source_format(model_cfg, model_local_path=model_id_or_path)
            if model_cfg is not None
            else None
        )
        if (
            quantize_config is not None
            and model_cfg is not None
            and native_floatx_source is not None
        ):
            log.info(
                "Loader: treating native FP8/NVFP4 checkpoint as a source model for new quantization."
            )
        elif (
            model_cfg is not None
            and _is_supported_quantization_config(model_cfg)
            and (
                not _uses_auto_module_decoder(quantize_config)
                or native_floatx_source is None
            )
        ):
            # only if the model is quantized or compatible with gptqmodel should we set is_quantized to true
            is_gptqmodel_quantized = True
        else:
            # TODO FIX ME...not decoded to check if quant method is compatible or quantized by gptqmodel
            for name in [QUANT_CONFIG_FILENAME, "quant_config.json"]:
                if treat_as_local_path:  # Local paths should never trigger remote Hub lookups
                    if isdir(model_id_or_path) and os.path.exists(join(model_id_or_path, name)):
                        is_gptqmodel_quantized = True
                        break
                else:  # Remote
                    files = list_repo_files(repo_id=model_id_or_path)
                    for f in files:
                        if f == name:
                            is_gptqmodel_quantized = True
                            break

        if is_gptqmodel_quantized:
            m = cls.from_quantized(
                model_id_or_path=model_id_or_path,
                device_map=device_map,
                device=device,
                p2p=p2p,
                kernel=kernel,
                backend=backend,
                trust_remote_code=trust_remote_code,
                tokenizer_trust_remote_code=requested_trust_remote_code,
                allow_remote_adapter=allow_remote_adapter,
                adapter_revision=adapter_revision,
                adapter_sha256=adapter_sha256,
                module_tree=module_tree,
                **kwargs,
            )
        else:
            if allow_remote_adapter or adapter_revision is not None or adapter_sha256 is not None:
                raise ValueError("Remote adapter options are only valid when loading a quantized checkpoint.")
            m = cls.from_pretrained(
                model_id_or_path=model_id_or_path,
                quantize_config=quantize_config,
                device_map=device_map,
                device=device,
                backend=requested_backend,
                profile=profile,
                trust_remote_code=trust_remote_code,
                tokenizer_trust_remote_code=requested_trust_remote_code,
                kernel=requested_kernel,
                runtime=runtime,
                p2p=p2p,
                module_tree=module_tree,
                **kwargs,
            )

        # debug model structure
        # if debug:
        #     print_module_tree(m.model)

        # Weight quantization does not change the cache, so state what context
        # costs as soon as a model is loaded for inference.
        m.log_kv_cache_estimate()

        return m


    @classmethod
    def from_pretrained(
            cls,
            model_id_or_path: str,
            quantize_config: QuantizeConfig,
            backend: Backend = BACKEND.AUTO,
            profile: Union[str, int, PROFILE] = PROFILE.AUTO,
            trust_remote_code: bool = False,
            p2p: Optional[Union[P2PConfig, Dict]] = None,
            kernel: Optional[Union[KernelConfig, Dict]] = None,
            module_tree: Optional[Union[str, List[Any], tuple]] = None,
            runtime: Optional[RuntimeConfig] = None,
            **model_init_kwargs,
    ) -> BaseQModel:
        if isinstance(quantize_config, dict):
            quantize_config = QuantizeConfig(**quantize_config)
        # The model definition re-resolves the caller's arguments against the
        # same config; the already-quantized redirect gets the resolved pair.
        requested_backend, requested_kernel = backend, kernel
        backend, kernel = resolve_quantize_backend(backend, quantize_config, kernel, runtime)
        if p2p is not None and quantize_config is not None:
            # The looper installs this policy before it enables peer access.
            quantize_config.p2p = P2PConfig(**p2p) if isinstance(p2p, dict) else p2p

        # Install the policy before the thread-pool prebuild reads it. The pool
        # is a per-process singleton, so a policy installed after the prebuild
        # would only affect the next process (issue #4396).
        _install_kernel_policy(kernel)
        if quantize_config is None:
            # Dense inference loads do not prebuild the pool, but an earlier
            # load may have left one at a different worker count.
            from .. import resize_device_thread_pool

            resize_device_thread_pool(create=False)

        if quantize_config is not None:
            _kick_quantize_prewarms(
                model_init_kwargs.get("device"),
                model_init_kwargs.get("device_map"),
                model_init_kwargs,
                quantize_config=quantize_config,
            )
            _prebuild_device_thread_pool(
                model_init_kwargs.get("device"), quantize_config=quantize_config
            )

        model_id_or_path = normalize_model_id_or_path_for_hf_gguf(
            model_id_or_path,
            model_init_kwargs,
            api_name="GPTQModel.from_pretrained",
        )
        normalize_torch_dtype_kwarg(
            model_init_kwargs,
            api_name="GPTQModel.from_pretrained",
        )
        backend = normalize_backend(backend)
        profile = normalize_profile(profile)
        requested_trust_remote_code = trust_remote_code
        tokenizer_trust_remote_code = model_init_kwargs.pop(
            "tokenizer_trust_remote_code",
            requested_trust_remote_code,
        )
        trust_remote_code = resolve_trust_remote_code(model_id_or_path, trust_remote_code=trust_remote_code)
        config = AutoConfig.from_pretrained(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            **_get_config_load_kwargs(model_init_kwargs),
        )
        native_floatx_source = native_floatx_source_format(
            config,
            model_local_path=model_id_or_path,
        )
        if (
            quantize_config is not None
            and native_floatx_source is not None
        ):
            log.info(
                "Loader: treating native FP8/NVFP4 checkpoint as a source model for new quantization."
            )
        elif _is_supported_quantization_config(config) and (
            not _uses_auto_module_decoder(quantize_config)
            or native_floatx_source is None
        ):
            log.warn("Model is already quantized, will use `from_quantized` to load quantized model.\n"
                           "If you want to quantize the model, please pass un_quantized model path or id, and use "
                           "`from_pretrained` with `quantize_config`.")
            if profile != PROFILE.AUTO:
                raise ValueError(
                    f"profile={profile.value} cannot be applied when redirecting an already-quantized "
                    "checkpoint to from_quantized(); omit profile or call from_quantized() directly."
                )
            # Preserve every explicit load option on the redirect. The caller
            # asked for this backend, placement, dtype, and fusion behavior;
            # silently selecting different defaults can load the checkpoint on
            # the wrong device or backend. `quantize_config` is a target
            # quantization request and cannot apply to an already-quantized
            # source, so it is intentionally not forwarded.
            return cls.from_quantized(
                model_id_or_path,
                device_map=model_init_kwargs.pop("device_map", None),
                device=model_init_kwargs.pop("device", None),
                backend=backend,
                adapter=model_init_kwargs.pop("adapter", None),
                trust_remote_code=trust_remote_code,
                tokenizer_trust_remote_code=tokenizer_trust_remote_code,
                p2p=p2p,
                kernel=kernel,
                module_tree=module_tree,
                **model_init_kwargs,
            )

        if quantize_config is not None:
            _kick_quantize_prewarms(
                model_init_kwargs.get("device"),
                model_init_kwargs.get("device_map"),
                model_init_kwargs,
                config,
                quantize_config,
            )

        if quantize_config and quantize_config.dynamic:
            log.warn(
                "GPT-QModel's per-module `dynamic` quantization feature is fully supported in latest vLLM and SGLang but not yet available in hf transformers.")

        model_definition = check_and_get_model_definition(
            model_id_or_path,
            trust_remote_code,
            **_get_config_load_kwargs(model_init_kwargs),
        )

        return model_definition.from_pretrained(
            pretrained_model_id_or_path=model_id_or_path,
            quantize_config=quantize_config,
            backend=requested_backend,
            profile=profile,
            trust_remote_code=trust_remote_code,
            tokenizer_trust_remote_code=tokenizer_trust_remote_code,
            module_tree=module_tree,
            p2p=p2p,
            kernel=requested_kernel,
            runtime=runtime,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
            cls,
            model_id_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, int]] = None,
            backend: Backend = BACKEND.AUTO,
            adapter: Optional[Adapter | Dict] = None,
            trust_remote_code: bool = False,
            allow_remote_adapter: bool = False,
            adapter_revision: Optional[str] = None,
            adapter_sha256: Optional[str] = None,
            p2p: Optional[Union[P2PConfig, Dict]] = None,
            kernel: Optional[Union[KernelConfig, Dict]] = None,
            module_tree: Optional[Union[str, List[Any], tuple]] = None,
            runtime: Optional[RuntimeConfig] = None,
            **kwargs,
    ) -> BaseQModel:
        backend, kernel = resolve_backend_config(backend, kernel, runtime)
        _install_kernel_policy(kernel)
        # Retune an existing singleton pool before any module is built. A
        # post-quantized load may be the second load in the process and must
        # honour its own ``cpu_workers`` (issue #4396).
        from .. import resize_device_thread_pool

        resize_device_thread_pool(create=False)
        model_id_or_path = normalize_model_id_or_path_for_hf_gguf(
            model_id_or_path,
            kwargs,
            api_name="GPTQModel.from_quantized",
        )
        normalize_torch_dtype_kwarg(
            kwargs,
            api_name="GPTQModel.from_quantized",
        )
        requested_trust_remote_code = trust_remote_code
        tokenizer_trust_remote_code = kwargs.pop("tokenizer_trust_remote_code", requested_trust_remote_code)
        trust_remote_code = resolve_trust_remote_code(model_id_or_path, trust_remote_code=trust_remote_code)
        # normalize adapter to instance
        adapter = normalize_adapter(adapter)

        print(f"from_quantized: adapter: {adapter}")
        model_definition = check_and_get_model_definition(
            model_id_or_path,
            trust_remote_code,
            **_get_config_load_kwargs(kwargs),
        )

        backend = normalize_backend(backend)

        return model_definition.from_quantized(
            model_id_or_path=model_id_or_path,
            device_map=device_map,
            device=device,
            p2p=p2p,
            kernel=kernel,
            backend=backend,
            trust_remote_code=trust_remote_code,
            tokenizer_trust_remote_code=tokenizer_trust_remote_code,
            adapter=adapter,
            allow_remote_adapter=allow_remote_adapter,
            adapter_revision=adapter_revision,
            adapter_sha256=adapter_sha256,
            module_tree=module_tree,
            **kwargs,
        )

    @staticmethod
    def export(model_id_or_path: str, target_path: str, format: str, trust_remote_code: bool = False):
        normalized_format = str(format).strip().lower()
        if normalized_format not in ("hf", "mlx"):
            raise ValueError(
                f"Unsupported export format: {format!r}. Expected 'hf' or 'mlx'."
            )

        trust_remote_code = resolve_trust_remote_code(model_id_or_path, trust_remote_code=trust_remote_code)

        if normalized_format == "hf":
            # Export through the checkpoint-level dequantizer: it understands
            # every stored layout (AWQ, GPTQ, quantized embeddings, legacy
            # quantize_config.json) instead of assuming a TorchLinear runtime.
            from ..utils.hub import snapshot_download
            from ..utils.model_dequant import dequantize_model

            source_path = model_id_or_path
            if not isdir(source_path):
                source_path = snapshot_download(repo_id=model_id_or_path)
            from pathlib import Path

            target = Path(target_path)
            empty_existing = target.is_dir() and not any(target.iterdir())
            dequantize_model(
                source_path,
                target_path,
                trust_remote_code=trust_remote_code,
                resume=empty_existing,
            )
            return

        # load config
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)

        if not getattr(config, "quantization_config", None):
            raise ValueError("Model is not quantized")

        gptq_config = config.quantization_config

        method = gptq_config.get("method", gptq_config.get("quant_method", ""))
        normalized_method = str(method).lower()
        if normalized_method == METHOD.GGUF.value:
            backend = BACKEND.GGUF_TORCH
        elif normalized_method == METHOD.BITSANDBYTES.value:
            backend = BACKEND.BITSANDBYTES
        elif normalized_method == METHOD.AWQ.value:
            backend = BACKEND.AWQ_TORCH
        elif normalized_method == METHOD.PARO.value:
            backend = BACKEND.PARO_CUDA
        elif normalized_method == METHOD.FP8.value:
            backend = BACKEND.FP8_TORCH
        elif normalized_method == METHOD.NF12.value:
            backend = BACKEND.NF12
        elif normalized_method == METHOD.MACAQ.value:
            backend = BACKEND.MACAQ_TORCH
        elif normalized_method == METHOD.EXL3.value:
            backend = BACKEND.EXL3_TORCH
        else:
            backend = BACKEND.GPTQ_TORCH

        # Load on the host. The MLX converter reads the packed tensors from the
        # checkpoint and never runs a quantized forward, so the accelerator is
        # not needed. Leaving the device implicit picked the accelerator
        # whenever one was visible, and the AWQ GEMV/GEMV_FAST Torch readers are
        # CPU-only, so `export(..., "mlx")` of those formats always failed with
        # `does not support device: cuda:0` (#4809).
        gptq_model = GPTQModel.load(
            model_id_or_path,
            backend=backend,
            device="cpu",
            trust_remote_code=trust_remote_code,
        )

        if normalized_format == "mlx":
            try:
                from mlx_lm.utils import save_config, save_model

                from ..utils.mlx import convert_gptq_to_mlx_weights
            except ImportError:
                raise ValueError(
                    "MLX not installed. Please install via `pip install gptqmodel[mlx] --no-build-isolation`.")

            mlx_weights, mlx_config = convert_gptq_to_mlx_weights(gptq_model.model_local_path or model_id_or_path, gptq_model, gptq_config,
                                                                  gptq_model.lm_head, stock_compatible=True)

            if mlx_config.pop("_gptqmodel_group16_runtime", False) or mlx_config.pop(
                "_gptqmodel_custom_mlx_runtime", False
            ):
                raise ValueError(
                    "MLX export produced a GPT-QModel custom-runtime checkpoint "
                    "that stock `mlx_lm.load` cannot read; export a supported "
                    "format or load it through `GPTQModel.load(backend='mlx')`."
                )

            save_model(target_path, mlx_weights, donate_model=True)

            save_config(mlx_config, config_path=target_path + "/config.json")
        # save tokenizer to target path
        gptq_model.tokenizer.save_pretrained(target_path)

    class adapter:
        @classmethod
        def generate(
            cls,
            # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
            adapter: Adapter,
            model_id_or_path: str, # native model
            quantized_model_id_or_path: str, # gptqmodel quantized model
            calibration: CalibrationConfig,
            *,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            # pass-through vars for load()
            trust_remote_code: bool = False,
            dtype: Optional[Union[str, torch.dtype]] = None,
            device: Optional[Union[str, torch.device]] = None,
        ):
            if not adapter or not isinstance(adapter, Lora):
                raise ValueError(f"Adapter: expected `adapter` type to be `Lora`: actual = `{adapter}`.")
            if not isinstance(calibration, CalibrationConfig):
                raise TypeError(
                    "GPTQModel.adapter.generate(calibration=...) must be a CalibrationConfig, "
                    f"got {type(calibration).__name__}."
                )

            adapter.validate_path(local=True)

            log.info("Model: Quant Model Loading...")
            quantized_model = GPTQModel.load(
                model_id_or_path=quantized_model_id_or_path,
                backend=BACKEND.GPTQ_TORCH,
                device=device,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
            )

            qcfg = quantized_model.quantize_config
            qModules: Dict[str, TorchLinear] = find_modules(module=quantized_model.model, layers=[TorchLinear])
            # for name, module in qModules.items():
            #     quantized_weights[name] = module.dequantize_weight()
            del quantized_model
            torch_empty_cache()

            log.info("Model: Native Model Loading...")
            model = GPTQModel.load(
                model_id_or_path=model_id_or_path,
                quantize_config=qcfg,
                backend=BACKEND.GPTQ_TORCH,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                device=device,
            )

            log.info("Model: Adapter generation started")
            model._eora_generate(
                adapter=adapter,
                quantized_modules=qModules,
                calibration=calibration,
                tokenizer=tokenizer,
            )
            return
