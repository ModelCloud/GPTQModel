# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os

from ..utils.logger import setup_logger


log = setup_logger()

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
import random  # noqa: E402
from os.path import isdir, join  # noqa: E402
from typing import Any, Dict, List, Optional, Type, Union  # noqa: E402

import numpy  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import list_repo_files  # noqa: E402
from tokenicer import Tokenicer  # noqa: E402
from transformers import AutoConfig, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase  # noqa: E402

from ..adapter.adapter import Adapter, Lora, normalize_adapter  # noqa: E402
from ..nn_modules.qlinear.torch import TorchQuantLinear  # noqa: E402
from ..quantization import METHOD, QUANT_CONFIG_FILENAME  # noqa: E402
from ..utils import BACKEND  # noqa: E402
from ..utils.eval import EVAL  # noqa: E402
from ..utils.model import find_modules  # noqa: E402
from ..utils.torch import CPU, torch_empty_cache  # noqa: E402
from .base import BaseQModel, QuantizeConfig  # noqa: E402
from .definitions.apertus import ApertusQModel  # noqa: E402
from .definitions.baichuan import BaiChuanQModel  # noqa: E402
from .definitions.bailing_moe import BailingMoeQModel  # noqa: E402
from .definitions.bloom import BloomQModel  # noqa: E402
from .definitions.chatglm import ChatGLMQModel  # noqa: E402
from .definitions.codegen import CodeGenQModel  # noqa: E402
from .definitions.dbrx import DbrxQModel  # noqa: E402
from .definitions.dbrx_converted import DbrxConvertedQModel  # noqa: E402
from .definitions.decilm import DeciLMQModel  # noqa: E402
from .definitions.deepseek_v2 import DeepSeekV2QModel  # noqa: E402
from .definitions.deepseek_v3 import DeepSeekV3QModel  # noqa: E402
from .definitions.dream import DreamQModel  # noqa: E402
from .definitions.ernie4_5 import Ernie4_5QModel  # noqa: E402
from .definitions.ernie4_5_moe import Ernie4_5_MoeQModel  # noqa: E402
from .definitions.exaone import ExaOneQModel  # noqa: E402
from .definitions.falcon_h1 import FalconH1QModel  # noqa: E402
from .definitions.gemma2 import Gemma2QModel  # noqa: E402
from .definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ, Gemma3QModel  # noqa: E402
from .definitions.glm import GlmQModel  # noqa: E402
from .definitions.glm4_moe import GLM4MoEGPTQ  # noqa: E402
from .definitions.gpt2 import GPT2QModel  # noqa: E402
from .definitions.gpt_bigcode import GptBigCodeQModel  # noqa: E402
from .definitions.gpt_neo import GptNeoQModel  # noqa: E402
from .definitions.gpt_neox import GPTNeoXQModel  # noqa: E402
from .definitions.gpt_oss import GPTOSSGPTQ  # noqa: E402
from .definitions.gptj import GptJQModel  # noqa: E402
from .definitions.grinmoe import GrinMoeQModel  # noqa: E402
from .definitions.hymba import HymbaQModel  # noqa: E402
from .definitions.instella import InstellaQModel  # noqa: E402
from .definitions.internlm import InternLMQModel  # noqa: E402
from .definitions.internlm2 import InternLM2QModel  # noqa: E402
from .definitions.klear import KlearQModel  # noqa: E402
from .definitions.lfm2_moe import LFM2MoeQModel  # noqa: E402
from .definitions.llama import LlamaQModel  # noqa: E402
from .definitions.llama4 import Llama4QModel  # noqa: E402
from .definitions.llava_qwen2 import LlavaQwen2QModel  # noqa: E402
from .definitions.longcat_flash import LongCatFlashQModel  # noqa: E402
from .definitions.mimo import MimoQModel  # noqa: E402
from .definitions.minicpm import MiniCPMGPTQ  # noqa: E402
from .definitions.minicpm3 import MiniCpm3QModel  # noqa: E402
from .definitions.mixtral import MixtralQModel  # noqa: E402
from .definitions.mllama import MLlamaQModel  # noqa: E402
from .definitions.mobilellm import MobileLLMQModel  # noqa: E402
from .definitions.moss import MossQModel  # noqa: E402
from .definitions.mpt import MptQModel  # noqa: E402
from .definitions.nemotron_h import NemotronHQModel  # noqa: E402
from .definitions.opt import OptQModel  # noqa: E402
from .definitions.ovis import OvisQModel  # noqa: E402
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
from .definitions.rw import RwgQModel  # noqa: E402
from .definitions.starcoder2 import Starcoder2QModel  # noqa: E402
from .definitions.telechat2 import TeleChat2QModel
from .definitions.xverse import XverseQModel  # noqa: E402


# make quants and inference more determinisitc
torch.manual_seed(787)
random.seed(787)
numpy.random.seed(787)

MODEL_MAP = {
    "apertus": ApertusQModel,
    "dream": DreamQModel,
    "bloom": BloomQModel,
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
    "glm4_moe": GLM4MoEGPTQ,
    "gpt_bigcode": GptBigCodeQModel,
    "codegen": CodeGenQModel,
    "cohere": LlamaQModel, # 100% llama clone
    "cohere2": LlamaQModel, # 100% llama clone
    "refinedWebModel": RwgQModel,
    "refinedWeb": RwgQModel,
    "falcon": RwgQModel,
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
    "phi": PhiQModel,
    "phi3": Phi3QModel,
    "phi4mm": Phi4MMGPTQ,
    "phimoe": PhiMoEGPTQForCausalLM,
    "mpt": MptQModel,
    "minicpm": MiniCPMGPTQ,
    "minicpm3": MiniCpm3QModel,
    "qwen2_moe": Qwen2MoeQModel,
    "qwen3_moe": Qwen3MoeQModel,
    "qwen3_next": Qwen3NextGPTQ,
    "qwen2_vl": Qwen2VLQModel,
    "qwen2_vl_text": Qwen2VLQModel,
    "qwen2_5_vl": Qwen2_5_VLQModel,
    "qwen2_5_vl_text": Qwen2_5_VLQModel,
    "qwen2_5_omni": Qwen2_5_OmniGPTQ,
    "qwen3_omni_moe": Qwen3OmniMoeGPTQ,
    "dbrx": DbrxQModel,
    "dbrx_converted": DbrxConvertedQModel,
    "deepseek_v2": DeepSeekV2QModel,
    "deepseek_v3": DeepSeekV3QModel,
    "exaone": ExaOneQModel,
    "grinmoe": GrinMoeQModel,
    "mllama": MLlamaQModel,
    "granite": LlamaQModel, # 100% llama clone
    "mobilellm": MobileLLMQModel,
    "hymba": HymbaQModel,
    "olmo2": LlamaQModel, # 100% llama clone
    "ovis": OvisQModel,
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
}

SUPPORTED_MODELS = list(MODEL_MAP.keys())


def check_and_get_model_type(model_dir, trust_remote_code=False):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type.lower() not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type.lower()

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
            device: Optional[Union[str, torch.device]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            trust_remote_code: bool = False,
            **kwargs,
    ):
        if isinstance(model_id_or_path, str):
            model_id_or_path = model_id_or_path.strip()

        # normalize config to cfg instance
        if isinstance(quantize_config, Dict):
            quantize_config = QuantizeConfig(**quantize_config)

        if isinstance(backend, str):
            backend = BACKEND(backend)

        is_gptqmodel_quantized = False
        model_cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        if hasattr(model_cfg, "quantization_config") and "quant_format" in model_cfg.quantization_config:
            # only if the model is quantized or compatible with gptqmodel should we set is_quantized to true
            if model_cfg.quantization_config["quant_format"].lower() in (METHOD.GPTQ, METHOD.AWQ, METHOD.QQQ):
                is_gptqmodel_quantized = True
        else:
            # TODO FIX ME...not decoded to check if quant method is compatible or quantized by gptqmodel
            for name in [QUANT_CONFIG_FILENAME, "quant_config.json"]:
                if isdir(model_id_or_path):  # Local
                    if os.path.exists(join(model_id_or_path, name)):
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
                **kwargs,
            )
        else:
            m = cls.from_pretrained(
                model_id_or_path=model_id_or_path,
                quantize_config=quantize_config,
                device_map=device_map,
                device=device,
                trust_remote_code=trust_remote_code,
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
            trust_remote_code: bool = False,
            **model_init_kwargs,
    ) -> BaseQModel:
        if hasattr(AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code),
                   "quantization_config"):
            log.warn("Model is already quantized, will use `from_quantized` to load quantized model.\n"
                           "If you want to quantize the model, please pass un_quantized model path or id, and use "
                           "`from_pretrained` with `quantize_config`.")
            return cls.from_quantized(model_id_or_path, trust_remote_code=trust_remote_code)

        if quantize_config and quantize_config.dynamic:
            log.warn(
                "GPTQModel's per-module `dynamic` quantization feature is fully supported in latest vLLM and SGLang but not yet available in hf transformers.")

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
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            adapter: Optional[Adapter | Dict] = None,
            trust_remote_code: bool = False,
            **kwargs,
    ) -> BaseQModel:
        # normalize adapter to instance
        adapter = normalize_adapter(adapter)

        print(f"from_quantized: adapter: {adapter}")
        model_type = check_and_get_model_type(model_id_or_path, trust_remote_code)

        if isinstance(backend, str):
            backend = BACKEND(backend)

        return MODEL_MAP[model_type].from_quantized(
            model_id_or_path=model_id_or_path,
            device_map=device_map,
            device=device,
            backend=backend,
            trust_remote_code=trust_remote_code,
            adapter=adapter,
            **kwargs,
        )

    @classmethod
    def eval(
            cls,
            model_or_id_or_path: str=None,
            tokenizer: Union[PreTrainedTokenizerBase, Tokenicer]=None,
            tasks: Union[EVAL.LM_EVAL, EVAL.EVALPLUS, List[EVAL.LM_EVAL], List[EVAL.EVALPLUS], EVAL.MMLU_PRO, List[EVAL.MMLU_PRO]] = None, # set to None to fix mutable warning
            framework: Union[Type[EVAL.LM_EVAL],Type[EVAL.EVALPLUS],Type[EVAL.MMLU_PRO]] = EVAL.LM_EVAL,
            batch_size: Union[int, str] = 1,
            trust_remote_code: bool = False,
            output_path: Optional[str] = None,
            llm_backend: str = 'gptqmodel',
            backend: BACKEND = BACKEND.AUTO, # gptqmodel arg only
            random_seed: int = 1234,  # only for framework=EVAL.LM_EVAL backend=vllm
            model_args: Dict[str, Any] = None,  # only for framework=EVAL.LM_EVAL backend=vllm
            ntrain: int = 1,  # only for framework=EVAL.MMLUPRO
            **args
    ):
        from peft import PeftModel
        if model_args is None:
            model_args = {}
        if tasks is None:
            if framework == EVAL.LM_EVAL:
                tasks = [EVAL.LM_EVAL.ARC_CHALLENGE]
            elif framework == EVAL.MMLU_PRO:
                tasks = [EVAL.MMLU_PRO.MATH]
            else:
                tasks = [EVAL.EVALPLUS.HUMAN]

        elif not isinstance(tasks, List):
            tasks = [tasks]

        if framework is None:
            raise ValueError("Eval parameter: `framework` cannot be set to None")

        if not isinstance(tasks, list):
            raise ValueError("Eval parameter: `tasks` must be of List type")

        if llm_backend not in ['gptqmodel', 'vllm']:
            raise ValueError('Eval framework support llm_backend: [gptqmodel, vllm]')

        if llm_backend == "vllm":
            if "tensor_parallel_size" not in model_args:
                try:
                    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
                except Exception:
                    cuda_devices = 0
                if cuda_devices:
                    model_args["tensor_parallel_size"] = cuda_devices
            if "gpu_memory_utilization" not in model_args:
                model_args["gpu_memory_utilization"] = 0.90

        if isinstance(model_or_id_or_path, str):
            load_backend = backend
            load_kwargs = {}

            if llm_backend == "vllm":
                disallowed_keys = {"pretrained", "tokenizer", "gptqmodel", "trust_remote_code", "backend", "model_id_or_path"}
                load_kwargs = {k: v for k, v in model_args.items() if k not in disallowed_keys}

            backend_name = load_backend.value if isinstance(load_backend, BACKEND) else str(load_backend)
            log.info(f"Eval: loading using backend = `{backend_name}`")
            model = GPTQModel.load(
                model_id_or_path=model_or_id_or_path,
                backend=load_backend,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )
            model_id_or_path = model_or_id_or_path
        elif isinstance(model_or_id_or_path, BaseQModel) or isinstance(model_or_id_or_path, (PreTrainedModel, PeftModel)):
            model = model_or_id_or_path
            model_id_or_path = model.config.name_or_path  #
        else:
            raise ValueError(f"`model_or_id_or_path` is invalid. expected: `model instance or str` actual: `{model_or_id_or_path}`")

        if tokenizer is None:
            if isinstance(model, BaseQModel):
                tokenizer = model.tokenizer
            elif isinstance(model, PreTrainedModel) or model_id_or_path.strip():
                tokenizer = Tokenicer.load(model_id_or_path.strip())

        if tokenizer is None:
            raise ValueError("Tokenizer: Auto-loading of tokenizer failed with `model_or_id_or_path`. Please pass in `tokenizer` as argument.")

        if llm_backend == "gptqmodel": # vllm loads tokenizer
            model_args["tokenizer"] = tokenizer

        if framework == EVAL.LM_EVAL:
            from lm_eval.utils import make_table  # hack: circular import

            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(f"Eval.lm_eval supported `tasks`: `{EVAL.get_all_tasks_string()}`, actual = `{task}`")

            model_name = "hf" if llm_backend == "gptqmodel" else llm_backend

            if llm_backend == "gptqmodel":
                model_args["gptqmodel"] = True
            model_args["pretrained"] = model_id_or_path

            try:
                from lm_eval import simple_evaluate
                from lm_eval.models.huggingface import HFLM
            except BaseException:
                raise ValueError("lm_eval is not installed. Please install via `pip install gptqmodel[eval]`.")

            if llm_backend == "gptqmodel" and model is not None:
                model_name = HFLM(
                    pretrained=model,
                    batch_size=batch_size,
                    trust_remote_code=trust_remote_code,
                )

            gen_kwargs = args.pop("gen_kwargs", None)

            # use model.generation_config whenever possible
            if gen_kwargs is None:
                # TODO: move to utils
                if hasattr(model, "generation_config") and isinstance(model.generation_config, GenerationConfig):
                    gen_dict = {
                        "do_sample": model.generation_config.do_sample,
                        "temperature": model.generation_config.temperature,
                        "top_k": model.generation_config.top_k,
                        "top_p": model.generation_config.top_p,
                        "min_p": model.generation_config.min_p,

                    }
                    gen_kwargs = ','.join(f"{key}={value}" for key, value in gen_dict.items() if value not in ["", {}, None, []])
                else:
                    gen_kwargs = "temperature=0.0,top_k=50" # default

            log.info(f"LM-EVAL: `gen_kwargs` = `{gen_kwargs}`")

            # lm-eval has very low scores if apply_chat_template is enabled
            apply_chat_template = args.pop("apply_chat_template", False) # args.pop("apply_chat_template", True if tokenizer.chat_template is not None else False)
            log.info(f"LM-EVAL: `apply_chat_template` = `{apply_chat_template}`")

            results = simple_evaluate(
                model=model_name,
                model_args=model_args,
                tasks=[task.value for task in tasks],
                batch_size=batch_size,
                apply_chat_template=apply_chat_template,
                gen_kwargs=gen_kwargs,
                random_seed=random_seed,
                numpy_random_seed=random_seed,
                torch_random_seed=random_seed,
                fewshot_random_seed=random_seed,
                **args,
            )

            if results is None:
                raise ValueError('lm_eval run fail, check your code!!!')

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
            from ..utils.eval import evalplus, evalplus_make_table

            results = {}
            for task in tasks:
                base_formatted, plus_formatted, result_path = evalplus(
                    model=model_id_or_path,
                    dataset=task.value,
                    batch=batch_size,
                    trust_remote_code=trust_remote_code,
                    output_file=output_path,
                    backend=llm_backend
                )
                results[task.value] = {"base tests": base_formatted, "base + extra tests": plus_formatted,
                                       "results_path": result_path}
            print('--------evalplus Eval Result---------')
            evalplus_make_table(results)
            print('--------evalplus Result End---------')
            return results
        elif framework == EVAL.MMLU_PRO:
            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(f"eval support tasks: {EVAL.get_all_tasks_string()}")
            from ..utils.mmlupro import mmlupro
            selected_subjects = ",".join(tasks)
            results = mmlupro(model,
                              tokenizer,
                              save_dir=output_path,
                              seed=random_seed,
                              selected_subjects=selected_subjects,
                              ntrain=ntrain,
                              batch_size=batch_size)

            print('--------MMLUPro Eval Result---------')
            print(results)
            print('--------MMLUPro Result End---------')
            return results
        else:
            raise ValueError("Eval framework support: EVAL.LM_EVAL, EVAL.EVALPLUS, EVAL.MMLUPRO")

    @staticmethod
    def export(model_id_or_path: str, target_path: str, format: str, trust_remote_code: bool = False):
        # load config
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)

        if not config.quantization_config:
            raise ValueError("Model is not quantized")

        gptq_config = config.quantization_config

        # load gptq model
        gptq_model = GPTQModel.load(model_id_or_path, backend=BACKEND.TORCH)

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

    # Use HfAPI and not Transformers to do upload
    @staticmethod
    def push_to_hub(repo_id: str,
                    quantized_path: str,  # saved local directory path
                    private: bool = False,
                    exists_ok: bool = False,  # set to true if repo already exists
                    token: Optional[str] = None,
                    ):

        if not quantized_path:
            raise RuntimeError("You must pass quantized model path as str to push_to_hub.")

        if not repo_id:
            raise RuntimeError("You must pass repo_id as str to push_to_hub.")

        from huggingface_hub import HfApi
        repo_type = "model"

        api = HfApi()
        # if repo does not exist, create it
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        except Exception:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, token=token, private=private, exist_ok=exists_ok)

        # upload the quantized save folder
        api.upload_large_folder(
            folder_path=quantized_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )

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
            # pass-through vars for load()
            trust_remote_code: bool = False,
            dtype: Optional[Union[str, torch.dtype]] = None,
        ):
            if not adapter or not isinstance(adapter, Lora):
                raise ValueError(f"Adapter: expected `adapter` type to be `Lora`: actual = `{adapter}`.")

            adapter.validate_path(local=True)

            log.info("Model: Quant Model Loading...")
            quantized_model = GPTQModel.load(
                model_id_or_path=quantized_model_id_or_path,
                backend=BACKEND.TORCH,
                device=CPU,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
            )

            qcfg = quantized_model.quantize_config
            qModules: Dict[str, TorchQuantLinear] = find_modules(module=quantized_model.model, layers=[TorchQuantLinear])
            # for name, module in qModules.items():
            #     quantized_weights[name] = module.dequantize_weight()
            del quantized_model
            torch_empty_cache()

            log.info("Model: Native Model Loading...")
            model = GPTQModel.load(
                model_id_or_path=model_id_or_path,
                quantize_config=qcfg,
                backend=BACKEND.TORCH,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                device=CPU,
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
            )
            return
