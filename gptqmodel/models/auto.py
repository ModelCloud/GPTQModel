from __future__ import annotations

import os.path
from os.path import isdir, join
from typing import Dict, List, Optional, Union

import torch
from ..quantization import QUANT_CONFIG_FILENAME
from huggingface_hub import list_repo_files
from transformers import AutoConfig

from ..integration.integration_vllm import patch_vllm
from ..utils import BACKEND, EVAL
from ..utils.logger import setup_logger
from ..utils.model import check_and_get_model_type
from ._const import get_best_device
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
from .definitions.glm import GLM
from .definitions.gpt2 import GPT2GPTQ
from .definitions.gpt_bigcode import GPTBigCodeGPTQ
from .definitions.gpt_neox import GPTNeoXGPTQ
from .definitions.gptj import GPTJGPTQ
from .definitions.granite import GraniteGPTQ
from .definitions.grinmoe import GrinMOEGPTQ
from .definitions.hymba import HymbaGPTQ
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
from .definitions.olmo2 import Olmo2GPTQ
from .definitions.opt import OPTGPTQ
from .definitions.phi import PhiGPTQ
from .definitions.phi3 import Phi3GPTQ
from .definitions.qwen import QwenGPTQ
from .definitions.qwen2 import Qwen2GPTQ
from .definitions.qwen2_moe import Qwen2MoeGPTQ
from .definitions.qwen2_vl import Qwen2VLGPTQ
from .definitions.rw import RWGPTQ
from .definitions.stablelmepoch import StableLMEpochGPTQ
from .definitions.starcoder2 import Starcoder2GPTQ
from .definitions.xverse import XverseGPTQ
from .definitions.yi import YiGPTQ

logger = setup_logger()

MODEL_MAP = {
    "bloom": BloomGPTQ,
    "gpt_neox": GPTNeoXGPTQ,
    "gptj": GPTJGPTQ,
    "gpt2": GPT2GPTQ,
    "llama": LlamaGPTQ,
    "opt": OPTGPTQ,
    "moss": MOSSGPTQ,
    "chatglm": ChatGLM,
    "glm": GLM,
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
    "qwen2_vl": Qwen2VLGPTQ,
    "dbrx": DbrxGPTQ,
    "dbrx_converted": DbrxConvertedGPTQ,
    "deepseek_v2": DeepSeekV2GPTQ,
    "exaone": ExaoneGPTQ,
    "grinmoe": GrinMOEGPTQ,
    "mllama": MLlamaGPTQ,
    "granite": GraniteGPTQ,
    "mobilellm": MobileLLMGPTQ,
    "hymba": HymbaGPTQ,
    "olmo2": Olmo2GPTQ,
}


class GPTQModel:
    def __init__(self):
        raise EnvironmentError(
            "GPTQModel is not designed to be instantiated\n"
            "use `GPTQModel.from_pretrained` to load pretrained model and prepare for quantization via `.quantize()`.\n"
            "use `GPTQModel.from_quantized` to inference with post-quantized model."
        )

    @classmethod
    def load(
            cls,
            model_id_or_path: Optional[str],
            quantize_config: Optional[QuantizeConfig | Dict] = None,
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, int]] = None,
            backend: BACKEND = BACKEND.AUTO,
            trust_remote_code: bool = False,
            verify_hash: Optional[Union[str, List[str]]] = None,
            **kwargs,
    ):
        if backend == BACKEND.VLLM:
            patch_vllm()
        is_quantized = False
        if hasattr(AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code), "quantization_config"):
            is_quantized = True
        else:
            for name in [QUANT_CONFIG_FILENAME, "quant_config.json"]:
                if isdir(model_id_or_path):  # Local
                    if os.path.exists(join(model_id_or_path, name)):
                        is_quantized = True
                        break

                else:  # Remote
                    files = list_repo_files(repo_id=model_id_or_path)
                    for f in files:
                        if f == name:
                            is_quantized = True
                            break

        if not device and not device_map:
            device = get_best_device()

        if is_quantized:
            return cls.from_quantized(
                model_id_or_path=model_id_or_path,
                device_map=device_map,
                device=device,
                backend=backend,
                trust_remote_code=trust_remote_code,
                verify_hash=verify_hash,
                **kwargs,
            )
        else:
            return cls.from_pretrained(
                model_id_or_path=model_id_or_path,
                quantize_config=quantize_config,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        quantize_config: QuantizeConfig,
        trust_remote_code: bool = False,
        **model_init_kwargs,
    ) -> BaseGPTQModel:
        if hasattr(AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code), "quantization_config"):
            logger.warning("Model is already quantized, will use `from_quantized` to load quantized model.\n"
                           "If you want to quantize the model, please pass un_quantized model path or id, and use "
                           "`from_pretrained` with `quantize_config`.")
            return cls.from_quantized(model_id_or_path, trust_remote_code=trust_remote_code)

        if quantize_config.dynamic:
            logger.warning("GPTQModel's per-module `dynamic` quantization feature is currently not upstreamed to hf/vllm/sglang. If you're using vllm, you need to install this PR: https://github.com/vllm-project/vllm/pull/7086")

        model_type = check_and_get_model_type(model_id_or_path, trust_remote_code)
        return MODEL_MAP[model_type].from_pretrained(
            pretrained_model_id_or_path=model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        cls,
        model_id_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
        device: Optional[Union[str, int]] = None,
        backend: BACKEND = BACKEND.AUTO,
        trust_remote_code: bool = False,
        # verify weight files matches predefined hash during loading
        # usage: hash_format:hash_value, example: md5:ugkdh232
        # supports all hashlib hash methods
        verify_hash: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> BaseGPTQModel:
        model_type = check_and_get_model_type(model_id_or_path, trust_remote_code)
        quant_func = MODEL_MAP[model_type].from_quantized

        if backend == BACKEND.AUTO and not torch.cuda.is_available():
            logger.warning("No cuda found, use IPEX backend")
            backend = BACKEND.IPEX

        return quant_func(
            model_id_or_path=model_id_or_path,
            device_map=device_map,
            device=device,
            backend=backend,
            trust_remote_code=trust_remote_code,
            verify_hash=verify_hash,
            **kwargs,
        )

    @classmethod
    def eval(
            cls,
            model_id_or_path: str,
            framework: EVAL,
            tasks: Union[List[EVAL.LM_EVAL], List[EVAL.EVALPLUS]],
            batch: int = 1,
            trust_remote_code: bool = False,
            output_file: Optional[str] = None,
            backend: str = 'gptqmodel',
            random_seed: int = 1234,  # only for framework=EVAL.LM_EVAL backend=vllm
            extra_model_args: str = "",  # only for framework=EVAL.LM_EVAL backend=vllm
    ):
        if framework is None:
            raise ValueError("eval parameter: `framework` cannot be set to None")

        if not isinstance(tasks, list):
            raise ValueError("eval parameter: `tasks` must be of List type")

        if backend not in ['gptqmodel', 'vllm']:
            raise ValueError('Eval framework support backend: [gptqmodel, vllm]')

        if framework == EVAL.LM_EVAL:
            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(f"lm_eval support tasks: {EVAL.get_all_tasks_string()}")

            from gptqmodel.utils.eval import lm_eval
            from lm_eval.utils import make_table
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)

            model_name = 'hf' if backend == 'gptqmodel' else backend
            def_args = f"pretrained={model_id_or_path}"
            if backend == "gptqmodel":
                def_args += ",gptqmodel=True"
            model_args = f"{def_args},{extra_model_args}" if extra_model_args else def_args

            results = lm_eval(
                model_name=model_name,
                model_args=model_args,
                tasks=[task.value for task in tasks],
                trust_remote_code=trust_remote_code,
                batch_size=batch,
                apply_chat_template=True if tokenizer.chat_template is not None else False,
                output_path=output_file,
                numpy_random_seed=random_seed,
                torch_random_seed=random_seed,
                fewshot_random_seed=random_seed,
            )
            print('--------lm_eval Eval Result---------')
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))
            print('--------lm_eval Result End---------')
            return results
        elif framework == EVAL.EVALPLUS:
            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(f"evalplus support tasks: {EVAL.get_all_tasks_string()}")
            from gptqmodel.utils.eval import evalplus, evalplus_make_table

            results = {}
            for task in tasks:
                base_formatted, plus_formatted, result_path = evalplus(
                    model=model_id_or_path,
                    dataset=task.value,
                    batch=batch,
                    trust_remote_code=trust_remote_code,
                    output_file=output_file,
                    backend=backend
                )
                results[task.value] = {"base tests": base_formatted, "base + extra tests": plus_formatted, "results_path": result_path}
            print('--------evalplus Eval Result---------')
            evalplus_make_table(results)
            print('--------evalplus Result End---------')
            return results
        else:
            raise ValueError("Eval framework support: EVAL.LM_EVAL, EVAL.EVALPLUS")
