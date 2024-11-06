from __future__ import annotations

from typing import Dict, List, Optional, Union

from ..utils import BACKEND
from ..utils.model import check_and_get_model_type
from .base import BaseGPTQModel, QuantizeConfig
from .definitions.baichuan import BaiChuanGPTQ
from .definitions.bloom import BloomGPTQ
from .definitions.chatglm import ChatGLM
from .definitions.codegen import CodeGenGPTQ
from .definitions.cohere import CohereGPTQ
from .definitions.dbrx import DbrxGPTQ
from .definitions.dbrx_converted import DbrxConvertedGPTQ
from .definitions.decilm import DeciLMGPTQ
from .definitions.deepseek_v2 import DeepSeekV2GPTQ
from .definitions.exaone import ExaoneGPTQ
from .definitions.gemma import GemmaGPTQ
from .definitions.gemma2 import Gemma2GPTQ
from .definitions.gpt2 import GPT2GPTQ
from .definitions.gpt_bigcode import GPTBigCodeGPTQ
from .definitions.gpt_neox import GPTNeoXGPTQ
from .definitions.gptj import GPTJGPTQ
from .definitions.granite import GraniteGPTQ
from .definitions.grinmoe import GrinMOEGPTQ
from .definitions.internlm import InternLMGPTQ
from .definitions.internlm2 import InternLM2GPTQ
from .definitions.llama import LlamaGPTQ
from .definitions.longllama import LongLlamaGPTQ
from .definitions.minicpm import MiniCPMGPTQ
from .definitions.minicpm3 import MiniCPM3GPTQ
from .definitions.mistral import MistralGPTQ
from .definitions.mixtral import MixtralGPTQ
from .definitions.mllama import MLlamaGPTQ
from .definitions.mobilellm import MobileLLMGPTQ
from .definitions.moss import MOSSGPTQ
from .definitions.mpt import MPTGPTQ
from .definitions.opt import OPTGPTQ
from .definitions.phi import PhiGPTQ
from .definitions.phi3 import Phi3GPTQ
from .definitions.qwen import QwenGPTQ
from .definitions.qwen2 import Qwen2GPTQ
from .definitions.qwen2_moe import Qwen2MoeGPTQ
from .definitions.rw import RWGPTQ
from .definitions.stablelmepoch import StableLMEpochGPTQ
from .definitions.starcoder2 import Starcoder2GPTQ
from .definitions.xverse import XverseGPTQ
from .definitions.yi import YiGPTQ
from transformers import AutoConfig
from ..quantization.config import FORMAT

MODEL_MAP = {
    "bloom": BloomGPTQ,
    "gpt_neox": GPTNeoXGPTQ,
    "gptj": GPTJGPTQ,
    "gpt2": GPT2GPTQ,
    "llama": LlamaGPTQ,
    "opt": OPTGPTQ,
    "moss": MOSSGPTQ,
    "chatglm": ChatGLM,
    "gpt_bigcode": GPTBigCodeGPTQ,
    "codegen": CodeGenGPTQ,
    "cohere": CohereGPTQ,
    "RefinedWebModel": RWGPTQ,
    "RefinedWeb": RWGPTQ,
    "falcon": RWGPTQ,
    "baichuan": BaiChuanGPTQ,
    "internlm": InternLMGPTQ,
    "internlm2": InternLM2GPTQ,
    "qwen": QwenGPTQ,
    "mistral": MistralGPTQ,
    "Yi": YiGPTQ,
    "xverse": XverseGPTQ,
    "deci": DeciLMGPTQ,
    "stablelm_epoch": StableLMEpochGPTQ,
    "starcoder2": Starcoder2GPTQ,
    "mixtral": MixtralGPTQ,
    "qwen2": Qwen2GPTQ,
    "longllama": LongLlamaGPTQ,
    "gemma": GemmaGPTQ,
    "gemma2": Gemma2GPTQ,
    "phi": PhiGPTQ,
    "phi3": Phi3GPTQ,
    "mpt": MPTGPTQ,
    "minicpm": MiniCPMGPTQ,
    "minicpm3":MiniCPM3GPTQ,
    "qwen2_moe": Qwen2MoeGPTQ,
    "dbrx": DbrxGPTQ,
    "dbrx_converted": DbrxConvertedGPTQ,
    "deepseek_v2": DeepSeekV2GPTQ,
    "exaone": ExaoneGPTQ,
    "grinmoe": GrinMOEGPTQ,
    "mllama": MLlamaGPTQ,
    "granite": GraniteGPTQ,
    "mobilellm": MobileLLMGPTQ,
}


class GPTQModel:
    def __init__(self):
        raise EnvironmentError(
            "GPTQModel is not designed to be instantiated\n"
            "use `GPTQModel.from_pretrained` to load pretrained model and prepare for quantization via `.quantize()`.\n"
            "use `GPTQModel.from_quantized` to inference with post-quantized model."
        )

    @classmethod
    def _from(
            cls,
            model_name_or_path: Optional[str],
            quantize_config: Optional[QuantizeConfig | Dict] = None,
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            max_memory: Optional[dict] = None,
            device: Optional[Union[str, int]] = None,
            backend: BACKEND = BACKEND.AUTO,
            use_safetensors: bool = True,
            trust_remote_code: bool = False,
            verify_hash: Optional[Union[str, List[str]]] = None,
            format: Optional[FORMAT] = None,
            **kwargs,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path)

        if hasattr(config, "quantization_config"):
            return cls.from_quantized(
                model_name_or_path=model_name_or_path,
                device_map=device_map,
                max_memory=max_memory,
                device=device,
                backend=backend,
                format=format,
                quantize_config=quantize_config,
                use_safetensors=use_safetensors,
                trust_remote_code=trust_remote_code,
                verify_hash=verify_hash,
                **kwargs,
            )
        else:
            return cls.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                quantize_config=quantize_config,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: QuantizeConfig,
        trust_remote_code: bool = False,
        **model_init_kwargs,
    ) -> BaseGPTQModel:
        model_type = check_and_get_model_type(pretrained_model_name_or_path, trust_remote_code)
        return MODEL_MAP[model_type].from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        backend: BACKEND = BACKEND.AUTO,
        quantize_config: Optional[QuantizeConfig | Dict] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        format: Optional[FORMAT] = None,
        # verify weight files matches predefined hash during loading
        # usage: hash_format:hash_value, example: md5:ugkdh232
        # supports all hashlib hash methods
        verify_hash: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> BaseGPTQModel:
        model_type = check_and_get_model_type(model_name_or_path, trust_remote_code)
        quant_func = MODEL_MAP[model_type].from_quantized

        return quant_func(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            max_memory=max_memory,
            device=device,
            backend=backend,
            format=format,
            quantize_config=quantize_config,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            verify_hash=verify_hash,
            **kwargs,
        )
