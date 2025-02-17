# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os

from gptqmodel.adapter.adapter import Adapter, normalize_adapter

from ..eora_test.eora_generate import eora_generate
from ..nn_modules.qlinear.torch import TorchQuantLinear

if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
    print("ENV: Auto setting PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' for memory saving.")

if not os.environ.get("CUDA_DEVICE_ORDER", None):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    print("ENV: Auto setting CUDA_DEVICE_ORDER=PCI_BUS_ID for compatibililty.")

import sys  # noqa: E402

# TODO: waiting for pytorch implementgation of aten ops for MPS
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import os.path  # noqa: E402
import random  # noqa: E402
from os.path import isdir, join  # noqa: E402
from typing import Any, Dict, List, Optional, Union  # noqa: E402

import numpy  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import list_repo_files  # noqa: E402
from transformers import AutoConfig, PreTrainedTokenizerBase  # noqa: E402

from ..quantization import QUANT_CONFIG_FILENAME  # noqa: E402
from ..utils import BACKEND  # noqa: E402
from ..utils.eval import EVAL  # noqa: E402
from ..utils.logger import setup_logger  # noqa: E402
from ..utils.model import check_and_get_model_type, find_modules  # noqa: E402
from .base import BaseGPTQModel, QuantizeConfig  # noqa: E402
from .definitions.baichuan import BaiChuanGPTQ  # noqa: E402
from .definitions.bloom import BloomGPTQ  # noqa: E402
from .definitions.chatglm import ChatGLM  # noqa: E402
from .definitions.codegen import CodeGenGPTQ  # noqa: E402
from .definitions.cohere import CohereGPTQ  # noqa: E402
from .definitions.cohere2 import Cohere2GPTQ  # noqa: E402
from .definitions.dbrx import DbrxGPTQ  # noqa: E402
from .definitions.dbrx_converted import DbrxConvertedGPTQ  # noqa: E402
from .definitions.decilm import DeciLMGPTQ  # noqa: E402
from .definitions.deepseek_v2 import DeepSeekV2GPTQ  # noqa: E402
from .definitions.deepseek_v3 import DeepSeekV3GPTQ  # noqa: E402
from .definitions.exaone import ExaoneGPTQ  # noqa: E402
from .definitions.gemma import GemmaGPTQ  # noqa: E402
from .definitions.gemma2 import Gemma2GPTQ  # noqa: E402
from .definitions.glm import GLM  # noqa: E402
from .definitions.gpt2 import GPT2GPTQ  # noqa: E402
from .definitions.gpt_bigcode import GPTBigCodeGPTQ  # noqa: E402
from .definitions.gpt_neox import GPTNeoXGPTQ  # noqa: E402
from .definitions.gptj import GPTJGPTQ  # noqa: E402
from .definitions.granite import GraniteGPTQ  # noqa: E402
from .definitions.grinmoe import GrinMOEGPTQ  # noqa: E402
from .definitions.hymba import HymbaGPTQ  # noqa: E402
from .definitions.internlm import InternLMGPTQ  # noqa: E402
from .definitions.internlm2 import InternLM2GPTQ  # noqa: E402
from .definitions.llama import LlamaGPTQ  # noqa: E402
from .definitions.longllama import LongLlamaGPTQ  # noqa: E402
from .definitions.minicpm import MiniCPMGPTQ  # noqa: E402
from .definitions.minicpm3 import MiniCPM3GPTQ  # noqa: E402
from .definitions.mistral import MistralGPTQ  # noqa: E402
from .definitions.mixtral import MixtralGPTQ  # noqa: E402
from .definitions.mllama import MLlamaGPTQ  # noqa: E402
from .definitions.mobilellm import MobileLLMGPTQ  # noqa: E402
from .definitions.moss import MOSSGPTQ  # noqa: E402
from .definitions.mpt import MPTGPTQ  # noqa: E402
from .definitions.olmo2 import Olmo2GPTQ  # noqa: E402
from .definitions.opt import OPTGPTQ  # noqa: E402
from .definitions.ovis import OvisGPTQ  # noqa: E402
from .definitions.phi import PhiGPTQ  # noqa: E402
from .definitions.phi3 import Phi3GPTQ, PhiMoEGPTQForCausalLM  # noqa: E402
from .definitions.qwen import QwenGPTQ  # noqa: E402
from .definitions.qwen2 import Qwen2GPTQ  # noqa: E402
from .definitions.qwen2_moe import Qwen2MoeGPTQ  # noqa: E402
from .definitions.qwen2_vl import Qwen2VLGPTQ  # noqa: E402
from .definitions.rw import RWGPTQ  # noqa: E402
from .definitions.stablelmepoch import StableLMEpochGPTQ  # noqa: E402
from .definitions.starcoder2 import Starcoder2GPTQ  # noqa: E402
from .definitions.telechat2 import TeleChat2GPTQ
from .definitions.xverse import XverseGPTQ  # noqa: E402
from .definitions.yi import YiGPTQ  # noqa: E402

# make quants and inference more determinisitc
torch.manual_seed(787)
random.seed(787)
numpy.random.seed(787)

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
    "cohere2": Cohere2GPTQ,
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
    "stablelm": StableLMEpochGPTQ,
    "starcoder2": Starcoder2GPTQ,
    "mixtral": MixtralGPTQ,
    "qwen2": Qwen2GPTQ,
    "longllama": LongLlamaGPTQ,
    "gemma": GemmaGPTQ,
    "gemma2": Gemma2GPTQ,
    "phi": PhiGPTQ,
    "phi3": Phi3GPTQ,
    "phimoe": PhiMoEGPTQForCausalLM,
    "mpt": MPTGPTQ,
    "minicpm": MiniCPMGPTQ,
    "minicpm3": MiniCPM3GPTQ,
    "qwen2_moe": Qwen2MoeGPTQ,
    "qwen2_vl": Qwen2VLGPTQ,
    "dbrx": DbrxGPTQ,
    "dbrx_converted": DbrxConvertedGPTQ,
    "deepseek_v2": DeepSeekV2GPTQ,
    "deepseek_v3": DeepSeekV3GPTQ,
    "exaone": ExaoneGPTQ,
    "grinmoe": GrinMOEGPTQ,
    "mllama": MLlamaGPTQ,
    "granite": GraniteGPTQ,
    "mobilellm": MobileLLMGPTQ,
    "hymba": HymbaGPTQ,
    "olmo2": Olmo2GPTQ,
    "ovis": OvisGPTQ,
    "telechat": TeleChat2GPTQ,
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
            device: Optional[Union[str, torch.device]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            trust_remote_code: bool = False,
            verify_hash: Optional[Union[str, List[str]]] = None,
            **kwargs,
    ):
        # normalize config to cfg instance
        if isinstance(quantize_config, Dict):
            quantize_config = QuantizeConfig(**quantize_config)

        if isinstance(backend, str):
            backend = BACKEND(backend)

        if backend == BACKEND.VLLM:
            from ..integration.integration_vllm import patch_vllm
            patch_vllm()

        is_quantized = False
        if hasattr(AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code),
                   "quantization_config"):
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
                device_map=device_map,
                device=device,
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
        if hasattr(AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code),
                   "quantization_config"):
            logger.warning("Model is already quantized, will use `from_quantized` to load quantized model.\n"
                           "If you want to quantize the model, please pass un_quantized model path or id, and use "
                           "`from_pretrained` with `quantize_config`.")
            return cls.from_quantized(model_id_or_path, trust_remote_code=trust_remote_code)

        if quantize_config and quantize_config.dynamic:
            logger.warning(
                "GPTQModel's per-module `dynamic` quantization feature is currently not upstreamed to hf/vllm/sglang. If you're using vllm, you need to install this PR: https://github.com/vllm-project/vllm/pull/7086")

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
            # verify weight files matches predefined hash during loading
            # usage: hash_format:hash_value, example: md5:ugkdh232
            # supports all hashlib hash methods
            verify_hash: Optional[Union[str, List[str]]] = None,
            **kwargs,
    ) -> BaseGPTQModel:
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
            verify_hash=verify_hash,
            adapter=adapter,
            **kwargs,
        )

    @classmethod
    def eval(
            cls,
            # model: BaseGPTQModel = None,
            model_or_path: Union[str, BaseGPTQModel] = None,
            framework: Type[EVAL] = EVAL.LM_EVAL,
            tasks: Union[List[EVAL.LM_EVAL], List[EVAL.EVALPLUS]] = EVAL.LM_EVAL.ARC_CHALLENGE,
            batch: int = 1,
            trust_remote_code: bool = False,
            output_file: str = None,
            llm_backend: str = 'gptqmodel',
            backend: BACKEND = BACKEND.AUTO, # gptqmodel arg only
            random_seed: int = 1234,  # only for framework=EVAL.LM_EVAL backend=vllm
            model_args: Dict[str, Any] = None,  # only for framework=EVAL.LM_EVAL backend=vllm
            apply_chat_template: Optional[bool] = None,
            gen_kwargs: str="temperature=0.0,top_k=50",
            **kwargs
    ):
        if not model_or_path:
            raise ValueError("Eval parameter: `model_id_or_path` is not passed.")

        if framework is None:
            raise ValueError("Eval parameter: `framework` cannot be set to None")

        if not isinstance(tasks, list):
            raise ValueError("Eval parameter: `tasks` must be of List type")

        if llm_backend not in ['gptqmodel', 'vllm']:
            raise ValueError('Eval framework support `backend`: `[gptqmodel, vllm]`')

        if llm_backend == "gptqmodel":
            if isinstance(model_or_path, str):
                model_or_path = GPTQModel.load(model_id_or_path=model_or_path, backend=backend)
            else:
                os.environ["GPTQMODEL_BACKEND"] = backend # hack so gptqmodel can get var from lm_eval call

        if framework == EVAL.LM_EVAL:
            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(f"Eval.lm_eval supported `tasks`: `{EVAL.get_all_tasks_string()}`, actual = `{task}`")

            from gptqmodel.utils.eval import lm_eval
            from lm_eval.utils import make_table
            from transformers import AutoTokenizer

            model_name = 'hf' if llm_backend == 'gptqmodel' else llm_backend
            if model_args is not None and not isinstance(model_args, Dict):
                 raise TypeError(f"Expected `model_args` to a `Dict`: actual = {model_args.__class__} ")

            if not model_args:
                model_args = {}

            if isinstance(model_or_path, str):
                tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=trust_remote_code)
                # only pass in gptqmodel args if loading via path or id
                model_args.update({"pretrained": model_or_path})
            else:
                tokenizer = model_or_path.tokenizer

            if llm_backend == "gptqmodel":
                model_args.update({"gptqmodel": True})

            if apply_chat_template is None:
                apply_chat_template = True if tokenizer.chat_template is not None else False

            results = lm_eval(
                model=model_or_path if isinstance(model_or_path, BaseGPTQModel) else None,
                model_name=model_name, # model_name is lm-eval model class name/type
                model_args=model_args,
                tasks=[task.value for task in tasks],
                trust_remote_code=trust_remote_code,
                batch_size=batch,
                apply_chat_template=apply_chat_template,
                output_file=output_file,
                random_seed=random_seed,
                gen_kwargs=gen_kwargs,
                **kwargs
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
                    model=model_or_path,
                    dataset=task.value,
                    batch=batch,
                    trust_remote_code=trust_remote_code,
                    output_file=output_file,
                    backend=llm_backend
                )
                results[task.value] = {"base tests": base_formatted, "base + extra tests": plus_formatted,
                                       "results_path": result_path}
            print('--------evalplus Eval Result---------')
            evalplus_make_table(results)
            print('--------evalplus Result End---------')
            return results
        else:
            raise ValueError("Eval framework support: EVAL.LM_EVAL, EVAL.EVALPLUS")

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
                from mlx_lm.utils import save_config, save_weights

                from ..utils.mlx import convert_gptq_to_mlx_weights
            except ImportError:
                raise ValueError(
                    "MLX not installed. Please install via `pip install gptqmodel[mlx] --no-build-isolation`.")

            mlx_weights, mlx_config = convert_gptq_to_mlx_weights(model_id_or_path, gptq_model, gptq_config,
                                                                  gptq_model.lm_head)

            save_weights(target_path, mlx_weights, donate_weights=True)

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
        # if repo does not exists, create it
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

    @classmethod
    def eora_generate(cls,
                      model_id_or_path: str,
                      quantized_model_id_or_path: str,
                      # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
                      adapter: Adapter,
                      calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
                      calibration_dataset_concat_size: Optional[int] = None,
                      batch_size: int = 1,
                      calibration_enable_gpu_cache: bool = True,
                      tokenizer: Optional[PreTrainedTokenizerBase] = None,
                      logger_board: Optional[str] = None,
                      backend: Optional[BACKEND] = BACKEND.AUTO,
                      # Experimental: enables the buffering of fwd inputs to cpu, slower than non-buffered, may reduce vram usage
                      buffered_fwd: bool = False,
                      # torch/cuda GC is auto enabled to reduce vram usage: disable to for small models or you know there is no possibility of oom due to vram to accelerate quantization
                      auto_gc: bool = True,
                      ):
        quantized_model = GPTQModel.load(quantized_model_id_or_path, backend=BACKEND.TORCH)
        quantize_config = quantized_model.quantize_config
        qModules = find_modules(quantized_model.model, [TorchQuantLinear])
        quantized_weights = {}
        for name, module in qModules.items():
            quantized_weights[name] = module.dequantize_weight().T.detach().to("cpu", torch.float16)

        model = GPTQModel.load(model_id_or_path, quantize_config, backend=backend)
        model.eora_generate(model=model,
                            adapter=adapter,
                            quantized_weights=quantized_weights,
                            calibration_dataset=calibration_dataset,
                            calibration_dataset_concat_size=calibration_dataset_concat_size,
                            batch_size=batch_size,
                            calibration_enable_gpu_cache=calibration_enable_gpu_cache,
                            tokenizer=tokenizer,
                            logger_board=logger_board,
                            buffered_fwd=buffered_fwd,
                            auto_gc=auto_gc)
        return
