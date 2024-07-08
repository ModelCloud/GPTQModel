from typing import Dict, List, Optional, Union

from ..utils import BACKEND
from ..utils.model import check_and_get_model_type
from .baichuan import BaiChuanGPTQ
from .base import BaseGPTQModel, QuantizeConfig
from .bloom import BloomGPTQ
from .chatglm import ChatGLM
from .codegen import CodeGenGPTQ
from .cohere import CohereGPTQ
from .dbrx import DbrxGPTQ
from .dbrx_converted import DbrxConvertedGPTQ
from .decilm import DeciLMGPTQ
from .deepseek_v2 import DeepSeekV2GPTQ
from .gemma import GemmaGPTQ
from .gemma2 import Gemma2GPTQ
from .gpt2 import GPT2GPTQ
from .gpt_bigcode import GPTBigCodeGPTQ
from .gpt_neox import GPTNeoXGPTQ
from .gptj import GPTJGPTQ
from .internlm import InternLMGPTQ
from .llama import LlamaGPTQ
from .longllama import LongLlamaGPTQ
from .minicpm import MiniCPMGPTQ
from .mistral import MistralGPTQ
from .mixtral import MixtralGPTQ
from .moss import MOSSGPTQ
from .mpt import MPTGPTQ
from .opt import OPTGPTQ
from .phi import PhiGPTQ
from .phi3 import Phi3GPTQ
from .qwen import QwenGPTQ
from .qwen2 import Qwen2GPTQ
from .qwen2_moe import Qwen2MoeGPTQ
from .rw import RWGPTQ
from .stablelmepoch import StableLMEpochGPTQ
from .starcoder2 import Starcoder2GPTQ
from .xverse import XverseGPTQ
from .yi import YiGPTQ
from .internlm2 import InternLM2GPTQ

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
    "qwen2_moe": Qwen2MoeGPTQ,
    "dbrx": DbrxGPTQ,
    "dbrx_converted": DbrxConvertedGPTQ,
    "deepseek_v2": DeepSeekV2GPTQ,
}


class GPTQModel:
    def __init__(self):
        raise EnvironmentError(
            "ModelGPTQ is not designed to be instantiated\n"
            "use `ModelGPTQ.from_pretrained` to load pretrained model and prepare for quantization via `.quantize()`.\n"
            "use `ModelGPTQ.from_quantized` to inference with post-quantized model."
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
        model_basename: Optional[str] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
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
            quantize_config=quantize_config,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            warmup_triton=warmup_triton,
            verify_hash=verify_hash,
            **kwargs,
        )

