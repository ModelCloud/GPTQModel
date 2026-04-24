# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
from contextlib import contextmanager

from ..utils.logger import setup_logger


log = setup_logger()

ASCII_LOGO = r"""
┌─────────────┐    ┌────────────────────────┐    ┌────────────┐    ┌─────────┐
│ GPT-QModel  │ -> │ ▓▓▓▓▓▓▓▓▓▓▓▓ 16bit     │ -> │ ▒▒▒▒ 8bit  │ -> │ ░░ 4bit │
└─────────────┘    └────────────────────────┘    └────────────┘    └─────────┘
"""

# if not os.environ.get("PYTHON_GIL", None):
#     os.environ["PYTHON_GIL"] = '0'
#     log.info("ENV: Auto disable GIL and use free-threading mode when applicable: Python 3.13t+. You must install the -t edition of Python.")

if not os.environ.get("PYTORCH_ALLOC_CONF", None):
    os.environ["PYTORCH_ALLOC_CONF"] = 'expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7'
    log.info("ENV: Auto setting PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7' for memory saving.")

if not os.environ.get("CUDA_DEVICE_ORDER", None):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    log.info("ENV: Auto setting CUDA_DEVICE_ORDER=PCI_BUS_ID for correctness.")

# FIX ROCm env conflict with CUDA_VISIBLE_DEVICES if both exits
if 'CUDA_VISIBLE_DEVICES' in os.environ and 'ROCR_VISIBLE_DEVICES' in os.environ:
    del os.environ['ROCR_VISIBLE_DEVICES']

# if not os.environ.get("NCCL_SHM_DISABLE", None):
#     os.environ["NCCL_SHM_DISABLE"] = '1'
#     log.info("ENV: Auto setting NCCL_SHM_DISABLE=1 for multi-gpu memory safety.")

import sys  # noqa: E402


# TODO: waiting for pytorch implementgation of aten ops for MPS
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import os.path  # noqa: E402
from os.path import isdir, join  # noqa: E402
from typing import Dict, List, Optional, Union  # noqa: E402

import torch  # noqa: E402
from packaging.version import Version  # noqa: E402
from transformers import AutoConfig, PreTrainedTokenizerBase  # noqa: E402
from transformers import __version__ as TRANSFORMERS_VERSION

from ..adapter.adapter import Adapter, Lora, normalize_adapter  # noqa: E402
from ..nn_modules.qlinear.torch import TorchLinear  # noqa: E402
from ..quantization import METHOD, QUANT_CONFIG_FILENAME, QuantizeConfig  # noqa: E402
from ..utils import BACKEND, PROFILE  # noqa: E402
from ..utils.backend import normalize_backend, normalize_profile  # noqa: E402
from ..utils.hf import (  # noqa: E402
    get_hf_gguf_load_kwargs,
    normalize_model_id_or_path_for_hf_gguf,
    normalize_torch_dtype_kwarg,
    resolve_trust_remote_code,
)
from ..utils.hub import list_repo_files  # noqa: E402
from ..utils.model import find_modules  # noqa: E402
from ..utils.torch import torch_empty_cache  # noqa: E402
from .base import BaseQModel  # noqa: E402
from .definitions.afmoe import AfMoeQModel  # noqa: E402
from .definitions.apertus import ApertusQModel  # noqa: E402
from .definitions.baichuan import BaiChuanQModel  # noqa: E402
from .definitions.bailing_moe import BailingMoeQModel  # noqa: E402
from .definitions.bloom import BloomQModel  # noqa: E402
from .definitions.brumby import BrumbyQModel  # noqa: E402
from .definitions.chatglm import ChatGLMQModel  # noqa: E402
from .definitions.codegen import CodeGenQModel  # noqa: E402
from .definitions.dbrx import DbrxQModel  # noqa: E402
from .definitions.dbrx_converted import DbrxConvertedQModel  # noqa: E402
from .definitions.decilm import DeciLMQModel  # noqa: E402
from .definitions.deepseek_v2 import DeepSeekV2QModel  # noqa: E402
from .definitions.deepseek_v3 import DeepSeekV3QModel  # noqa: E402
from .definitions.dots1 import Dots1QModel  # noqa: E402
from .definitions.dream import DreamQModel  # noqa: E402
from .definitions.ernie4_5 import Ernie4_5QModel  # noqa: E402
from .definitions.ernie4_5_moe import Ernie4_5_MoeQModel  # noqa: E402
from .definitions.exaone import ExaOneQModel  # noqa: E402
from .definitions.exaone4 import Exaone4QModel  # noqa: E402
from .definitions.falcon_h1 import FalconH1QModel  # noqa: E402
from .definitions.falcon_mamba import FalconMambaQModel  # noqa: E402
from .definitions.gemma2 import Gemma2QModel  # noqa: E402
from .definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ, Gemma3QModel  # noqa: E402
from .definitions.gemma3n import Gemma3nForConditionalGenerationGPTQ, Gemma3nTextQModel  # noqa: E402
from .definitions.gemma4 import Gemma4ForConditionalGenerationGPTQ, Gemma4TextQModel  # noqa: E402
from .definitions.glm import GlmQModel  # noqa: E402
from .definitions.glmasr import GlmASRGPTQ  # noqa: E402
from .definitions.glm_ocr import GlmOCRGPTQ  # noqa: E402
from .definitions.glm4_moe import GLM4MoEGPTQ  # noqa: E402
from .definitions.glm4_moe_lite import Glm4MoeLiteQModel  # noqa: E402
from .definitions.glm4v import Glm4vGPTQ  # noqa: E402
from .definitions.glm_moe_dsa import GlmMoeDsaQModel  # noqa: E402
from .definitions.gpt2 import GPT2QModel  # noqa: E402
from .definitions.gpt_bigcode import GptBigCodeQModel  # noqa: E402
from .definitions.gpt_neo import GptNeoQModel  # noqa: E402
from .definitions.gpt_neox import GPTNeoXQModel  # noqa: E402
from .definitions.gpt_oss import GPTOSSGPTQ  # noqa: E402
from .definitions.gptj import GptJQModel  # noqa: E402
from .definitions.granitemoehybrid import GraniteMoeHybridQModel
from .definitions.grinmoe import GrinMoeQModel  # noqa: E402
from .definitions.hymba import HymbaQModel  # noqa: E402
from .definitions.instella import InstellaQModel  # noqa: E402
from .definitions.internlm import InternLMQModel  # noqa: E402
from .definitions.internlm2 import InternLM2QModel  # noqa: E402
from .definitions.klear import KlearQModel  # noqa: E402
from .definitions.lfm2_moe import LFM2MoeQModel  # noqa: E402
from .definitions.llada2 import LLaDA2MoeQModel
from .definitions.llama import LlamaQModel  # noqa: E402
from .definitions.llama4 import Llama4QModel  # noqa: E402
from .definitions.llava_qwen2 import LlavaQwen2QModel  # noqa: E402
from .definitions.longcat_flash import LongCatFlashQModel  # noqa: E402
from .definitions.mimo import MimoQModel  # noqa: E402
from .definitions.minicpm import MiniCPMGPTQ  # noqa: E402
from .definitions.minicpm3 import MiniCpm3QModel  # noqa: E402
from .definitions.minicpm_o import MiniCPMOQModel  # noqa: E402
from .definitions.minicpm_v import MiniCPMVQModel  # noqa: E402
from .definitions.minimax_m2 import MiniMaxM2GPTQ  # noqa: E402
from .definitions.mistral3 import Mistral3GPTQ
from .definitions.mixtral import MixtralQModel  # noqa: E402
from .definitions.mllama import MLlamaQModel  # noqa: E402
from .definitions.mobilellm import MobileLLMQModel  # noqa: E402
from .definitions.moss import MossQModel  # noqa: E402
from .definitions.mpt import MptQModel  # noqa: E402
from .definitions.nemotron_h import NemotronHQModel  # noqa: E402
from .definitions.opt import OptQModel  # noqa: E402
from .definitions.ovis import OvisQModel  # noqa: E402
from .definitions.ovis2 import Ovis2QModel  # noqa: E402
from .definitions.pangu_alpha import PanguAlphaQModel  # noqa: E402
from .definitions.phi import PhiQModel  # noqa: E402
from .definitions.phi3 import Phi3QModel, PhiMoEGPTQForCausalLM  # noqa: E402
from .definitions.phi4 import Phi4MMGPTQ  # noqa: E402
from .definitions.qwen import QwenQModel  # noqa: E402
from .definitions.qwen2 import Qwen2QModel  # noqa: E402
from .definitions.qwen2_5_omni import Qwen2_5_OmniGPTQ
from .definitions.qwen2_5_vl import Qwen2_5_VLQModel  # noqa: E402
from .definitions.qwen2_moe import Qwen2MoeQModel  # noqa: E402
from .definitions.qwen2_vl import Qwen2VLQModel  # noqa: E402
from .definitions.qwen3 import Qwen3QModel  # noqa: E402
from .definitions.qwen3_moe import Qwen3MoeQModel  # noqa: E402
from .definitions.qwen3_next import Qwen3NextGPTQ  # noqa: E402
from .definitions.qwen3_omni_moe import Qwen3OmniMoeGPTQ
from .definitions.qwen3_vl import Qwen3_VLQModel
from .definitions.rw import RwgQModel  # noqa: E402
from .definitions.starcoder2 import Starcoder2QModel  # noqa: E402
from .definitions.telechat2 import TeleChat2QModel
from .definitions.voxtral import VoxtralGPTQ  # noqa: E402
from .definitions.xverse import XverseQModel  # noqa: E402


TRANSFORMERS_SUPPORTS_QWEN3_5 = Version(TRANSFORMERS_VERSION) >= Version("5.2.0")
if TRANSFORMERS_SUPPORTS_QWEN3_5:
    from .definitions.qwen3_5 import Qwen3_5QModel  # noqa: E402
    from .definitions.qwen3_5_moe import Qwen3_5_MoeQModel  # noqa: E402
else:
    Qwen3_5QModel = None
    Qwen3_5_MoeQModel = None


MODEL_MAP = {
    "apertus": ApertusQModel,
    "dream": DreamQModel,
    "bloom": BloomQModel,
    "brumby": BrumbyQModel,
    "gpt_neo": GptNeoQModel,
    "kimi_k2": DeepSeekV3QModel, # 100% DeepSeekV3QModel clone
    "klear": KlearQModel,
    "gpt_neox": GPTNeoXQModel,
    "gptj": GptJQModel,
    "gpt2": GPT2QModel,
    "llama": LlamaQModel,
    "llama4": Llama4QModel,
    "opt": OptQModel,
    "moss": MossQModel,
    "chatglm": ChatGLMQModel,
    "glm": GlmQModel,
    "glm4": GlmQModel,
    "glm4v": Glm4vGPTQ,
    "glmasr": GlmASRGPTQ,
    "glm_ocr": GlmOCRGPTQ,
    "glm4_moe": GLM4MoEGPTQ,
    "glm4_moe_lite": Glm4MoeLiteQModel,
    "glm_moe_dsa": GlmMoeDsaQModel,
    "gpt_bigcode": GptBigCodeQModel,
    "codegen": CodeGenQModel,
    "cohere": LlamaQModel, # 100% llama clone
    "cohere2": LlamaQModel, # 100% llama clone
    "refinedWebModel": RwgQModel,
    "refinedWeb": RwgQModel,
    "falcon": RwgQModel,
    "falcon_mamba": FalconMambaQModel,
    "baichuan": BaiChuanQModel,
    "internlm": InternLMQModel,
    "internlm2": InternLM2QModel,
    "qwen": QwenQModel,
    "mistral": LlamaQModel, # 100% llama clone
    "yi": LlamaQModel, # 100% llama clone
    "xverse": XverseQModel,
    "deci": DeciLMQModel,
    "nemotron-nas": DeciLMQModel,
    "stablelm_epoch": LlamaQModel, # 100% llama clone
    "stablelm": LlamaQModel, # 100% llama clone
    "starcoder2": Starcoder2QModel,
    "mixtral": MixtralQModel,
    "qwen2": Qwen2QModel,
    "qwen3": Qwen3QModel,
    "longllama": LlamaQModel,  # 100% llama clone
    "gemma": LlamaQModel, # 100% llama clone
    "gemma2": Gemma2QModel,
    "gemma3_text": Gemma3QModel,
    "gemma3": Gemma3ForConditionalGenerationGPTQ,
    "gemma3n_text": Gemma3nTextQModel,
    "gemma3n": Gemma3nForConditionalGenerationGPTQ,
    "gemma4_text": Gemma4TextQModel,
    "gemma4": Gemma4ForConditionalGenerationGPTQ,
    "phi": PhiQModel,
    "phi3": Phi3QModel,
    "phi4mm": Phi4MMGPTQ,
    "phimoe": PhiMoEGPTQForCausalLM,
    "mpt": MptQModel,
    "minicpm": MiniCPMGPTQ,
    "minicpm3": MiniCpm3QModel,
    "minicpmo": MiniCPMOQModel,
    "minicpmv": MiniCPMVQModel,
    "minimax": MiniMaxM2GPTQ,
    "minimax_m2": MiniMaxM2GPTQ,
    "qwen2_moe": Qwen2MoeQModel,
    "qwen3_moe": Qwen3MoeQModel,
    "qwen3_next": Qwen3NextGPTQ,
    "qwen2_vl": Qwen2VLQModel,
    "qwen2_vl_text": Qwen2VLQModel,
    "qwen2_5_vl": Qwen2_5_VLQModel,
    "qwen2_5_vl_text": Qwen2_5_VLQModel,
    "qwen2_5_omni": Qwen2_5_OmniGPTQ,
    "qwen3_omni_moe": Qwen3OmniMoeGPTQ,
    "qwen3_vl": Qwen3_VLQModel,
    "dbrx": DbrxQModel,
    "dbrx_converted": DbrxConvertedQModel,
    "deepseek_v2": DeepSeekV2QModel,
    "deepseek_v3": DeepSeekV3QModel,
    "dots1": Dots1QModel,
    "exaone": ExaOneQModel,
    "exaone4": Exaone4QModel,
    "grinmoe": GrinMoeQModel,
    "mllama": MLlamaQModel,
    "marin": Qwen3QModel,
    "granite": LlamaQModel, # 100% llama clone
    "granitemoehybrid": GraniteMoeHybridQModel,
    "mobilellm": MobileLLMQModel,
    "hymba": HymbaQModel,
    "olmo2": LlamaQModel, # 100% llama clone
    "ovis": OvisQModel,
    "ovis2": Ovis2QModel,
    "telechat": TeleChat2QModel,
    "instella": InstellaQModel,
    "mimo": MimoQModel,
    "falcon_h1": FalconH1QModel,
    "gpt_pangu": PanguAlphaQModel,
    "ernie4_5": Ernie4_5QModel,
    "ernie4_5_moe": Ernie4_5_MoeQModel,
    "seed_oss": LlamaQModel, # 100% llama clone
    "gpt_oss": GPTOSSGPTQ,
    "longcat_flash": LongCatFlashQModel,
    "llava_qwen2": LlavaQwen2QModel,
    "nemotron_h": NemotronHQModel,
    "bailing_moe": BailingMoeQModel,
    "lfm2_moe": LFM2MoeQModel,
    "llada2_moe": LLaDA2MoeQModel,
    "mistral3": Mistral3GPTQ,
    "afmoe": AfMoeQModel,
    "voxtral": VoxtralGPTQ,
}

if Qwen3_5QModel is not None:
    MODEL_MAP["qwen3_5"] = Qwen3_5QModel
    MODEL_MAP["qwen3_5_text"] = Qwen3_5QModel

if Qwen3_5_MoeQModel is not None:
    MODEL_MAP["qwen3_5_moe"] = Qwen3_5_MoeQModel

SUPPORTED_MODELS = list(MODEL_MAP.keys())


def _activation_quantization_mode(quantization_config: dict) -> Optional[str]:
    """Return the first activation-quantization field that makes this config unsupported.

    GPT-QModel can load weight-only quantized checkpoints through the Transformers
    surface, but it does not currently implement activation-quantized runtime
    semantics. This helper keeps the rejection logic in one place for both
    ModelOpt-style grouped configs and flatter HF quantization payloads.
    """

    config_groups = quantization_config.get("config_groups")
    if isinstance(config_groups, dict):
        for group_cfg in config_groups.values():
            if not isinstance(group_cfg, dict):
                continue
            input_activations = group_cfg.get("input_activations")
            if isinstance(input_activations, dict) and input_activations:
                return "input_activations"

    kv_cache_scheme = quantization_config.get("kv_cache_scheme")
    if isinstance(kv_cache_scheme, dict) and kv_cache_scheme:
        return "kv_cache_scheme"

    for key in ("input_activations", "activation_quantization", "activations"):
        value = quantization_config.get(key)
        if isinstance(value, dict) and value:
            return key
    return None


def _is_supported_quantization_config(config: AutoConfig) -> bool:
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        return False

    # Fail fast before model selection so activation-quantized checkpoints do
    # not accidentally proceed down a weight-only loader path.
    unsupported_mode = _activation_quantization_mode(quantization_config)
    if unsupported_mode is not None:
        log.error("GPT-QModel currently does not support loading of activation quantized models")
        raise ValueError(
            "GPT-QModel currently does not support loading of activation quantized models. "
            f"Detected unsupported metadata: {unsupported_mode}."
        )

    quant_format = quantization_config.get("quant_format")
    if isinstance(quant_format, str) and quant_format.lower() in (
        METHOD.GPTQ,
        METHOD.GGUF,
        METHOD.FP8,
        METHOD.BITSANDBYTES,
        METHOD.AWQ,
        METHOD.PARO,
        METHOD.QQQ,
        METHOD.EXL3,
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
        METHOD.EXL3,
    ):
        return True

    return False


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


def _get_config_load_kwargs(kwargs: dict) -> dict:
    return get_hf_gguf_load_kwargs(kwargs)


def check_and_get_model_definition(model_dir, trust_remote_code=False, **config_load_kwargs):
    if "gguf_file" not in config_load_kwargs:
        model_dir = normalize_model_id_or_path_for_hf_gguf(
            model_dir,
            config_load_kwargs,
            api_name="check_and_get_model_definition",
        )
    trust_remote_code = resolve_trust_remote_code(model_dir, trust_remote_code=trust_remote_code)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code, **config_load_kwargs)
    model_type = config.model_type.lower()

    # if model_type is not supported, use BaseQModel, will use auto_detect_module_tree to generate module tree
    if model_type not in SUPPORTED_MODELS:
        return BaseQModel

    return MODEL_MAP[model_type]


class GPTQModel:
    def __init__(self):
        raise EnvironmentError(
            "GPT-QModel is not designed to be instantiated\n"
            "use `from_pretrained()` to load a pretrained model and prepare for quantization via `.quantize()`.\n"
            "use `from_quantized()` for inference with a post-quantized model."
        )

    @classmethod
    def load(
            cls,
            model_id_or_path: Optional[str],
            quantize_config: Optional[QuantizeConfig | Dict] = None,
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, torch.device]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            profile: Union[str, int, PROFILE] = PROFILE.AUTO,
            trust_remote_code: bool = False,
            **kwargs,
    ):
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

        backend = normalize_backend(backend)
        profile = normalize_profile(profile)

        is_gptqmodel_quantized = False
        treat_as_local_path = isinstance(model_id_or_path, str) and (
            isdir(model_id_or_path) or os.path.isabs(model_id_or_path)
        )

        model_cfg = None
        if not (treat_as_local_path and not isdir(model_id_or_path)):
            model_cfg = AutoConfig.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
                **_get_config_load_kwargs(kwargs),
            )

        if model_cfg is not None and _is_supported_quantization_config(model_cfg):
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
                backend=backend,
                trust_remote_code=trust_remote_code,
                tokenizer_trust_remote_code=requested_trust_remote_code,
                **kwargs,
            )
        else:
            m = cls.from_pretrained(
                model_id_or_path=model_id_or_path,
                quantize_config=quantize_config,
                device_map=device_map,
                device=device,
                backend=backend,
                profile=profile,
                trust_remote_code=trust_remote_code,
                tokenizer_trust_remote_code=requested_trust_remote_code,
                **kwargs,
            )

        # debug model structure
        # if debug:
        #     print_module_tree(m.model)

        return m


    @classmethod
    def from_pretrained(
            cls,
            model_id_or_path: str,
            quantize_config: QuantizeConfig,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            profile: Union[str, int, PROFILE] = PROFILE.AUTO,
            trust_remote_code: bool = False,
            **model_init_kwargs,
    ) -> BaseQModel:
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
        if _is_supported_quantization_config(config):
            log.warn("Model is already quantized, will use `from_quantized` to load quantized model.\n"
                           "If you want to quantize the model, please pass un_quantized model path or id, and use "
                           "`from_pretrained` with `quantize_config`.")
            return cls.from_quantized(model_id_or_path, trust_remote_code=trust_remote_code)

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
            backend=backend,
            profile=profile,
            trust_remote_code=trust_remote_code,
            tokenizer_trust_remote_code=tokenizer_trust_remote_code,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
            cls,
            model_id_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, int]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            adapter: Optional[Adapter | Dict] = None,
            trust_remote_code: bool = False,
            **kwargs,
    ) -> BaseQModel:
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
            backend=backend,
            trust_remote_code=trust_remote_code,
            tokenizer_trust_remote_code=tokenizer_trust_remote_code,
            adapter=adapter,
            **kwargs,
        )

    @staticmethod
    def export(model_id_or_path: str, target_path: str, format: str, trust_remote_code: bool = False):
        trust_remote_code = resolve_trust_remote_code(model_id_or_path, trust_remote_code=trust_remote_code)
        # load config
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)

        if not config.quantization_config:
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
            backend = BACKEND.PAROQUANT_CUDA
        elif normalized_method == METHOD.FP8.value:
            backend = BACKEND.FP8_TORCH
        elif normalized_method == METHOD.EXL3.value:
            backend = BACKEND.EXL3_TORCH
        else:
            backend = BACKEND.GPTQ_TORCH

        # load gptq model
        gptq_model = GPTQModel.load(model_id_or_path, backend=backend)

        if format == "mlx":
            try:
                from mlx_lm.utils import save_config, save_model

                from ..utils.mlx import convert_gptq_to_mlx_weights
            except ImportError:
                raise ValueError(
                    "MLX not installed. Please install via `pip install gptqmodel[mlx] --no-build-isolation`.")

            mlx_weights, mlx_config = convert_gptq_to_mlx_weights(model_id_or_path, gptq_model, gptq_config,
                                                                  gptq_model.lm_head)

            save_model(target_path, mlx_weights, donate_model=True)

            save_config(mlx_config, config_path=target_path + "/config.json")
        elif format == "hf":
            from ..nn_modules.qlinear.torch import dequantize_model

            dequantized_model = dequantize_model(gptq_model.model)
            dequantized_model.save_pretrained(target_path)

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
            calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
            calibration_dataset_concat_size: Optional[int] = None,
            calibration_dataset_sort: Optional[str] = None,
            batch_size: Optional[int] = 1,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            calibration_concat_separator: Optional[str] = None,
            # pass-through vars for load()
            trust_remote_code: bool = False,
            dtype: Optional[Union[str, torch.dtype]] = None,
            device: Optional[Union[str, torch.device]] = None,
        ):
            if not adapter or not isinstance(adapter, Lora):
                raise ValueError(f"Adapter: expected `adapter` type to be `Lora`: actual = `{adapter}`.")

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
                calibration_dataset=calibration_dataset,
                calibration_dataset_concat_size=calibration_dataset_concat_size,
                calibration_dataset_sort=calibration_dataset_sort,
                batch_size=batch_size,
                tokenizer=tokenizer,
                calibration_concat_separator=calibration_concat_separator,
            )
            return
