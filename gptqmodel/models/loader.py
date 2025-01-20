# Copyright 2025 ModelCloud
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
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List, Optional, Union

import torch
import transformers
from huggingface_hub import snapshot_download
from packaging.version import InvalidVersion, Version
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers.modeling_utils import no_init_weights
from transformers.utils import is_flash_attn_2_available
from transformers.utils.generic import ContextManagers

from ..nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.ipex import IPEXQuantLinear
from ..quantization import QuantizeConfig
from ..quantization.config import FORMAT, FORMAT_FIELD_JSON, MIN_VERSION_WITH_V2
from ..utils.backend import BACKEND
from ..utils.importer import auto_select_device, normalize_device_device_map, select_quant_linear
from ..utils.logger import setup_logger
from ..utils.marlin import (_validate_marlin_compatibility,
                            _validate_marlin_device_support, prepare_model_for_marlin_load)
from ..utils.model import (auto_dtype, convert_gptq_v1_to_v2_format, find_layers, get_checkpoints,
                           get_moe_layer_modules, gptqmodel_post_init, make_quant, normalize_tokenizer,
                           simple_dispatch_model, verify_model_hash, verify_sharded_model_hashes,
                           load_checkpoint_in_model_then_tie_weights)
from ._const import DEVICE, SUPPORTED_MODELS, normalize_device

logger = setup_logger()

ATTN_IMPLEMENTATION = "attn_implementation"
USE_FLASH_ATTENTION_2 = "use_flash_attention_2"
def parse_version_string(version_str: str):
    try:
        return Version(version_str)
    except InvalidVersion:
        raise ValueError(f"Invalid version format: {version_str}")


def parse_requirement(req):
    for op in [">=", "<=", ">", "<", "=="]:
        if op in req:
            pkg, version_required = req.split(op, 1)
            return pkg.strip(), op, version_required.strip()
    raise ValueError(f"Unsupported version constraint in: {req}")


def compare_versions(installed_version, required_version, operator):
    installed = parse_version_string(installed_version)
    required = parse_version_string(required_version)
    if operator == ">":
        return installed > required
    elif operator == ">=":
        return installed >= required
    elif operator == "<":
        return installed < required
    elif operator == "<=":
        return installed <= required
    elif operator == "==":
        return installed == required
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def check_versions(model_class, requirements: List[str]):
    if requirements is None:
        return
    for req in requirements:
        pkg, operator, version_required = parse_requirement(req)
        try:
            installed_version = version(pkg)
            if not compare_versions(installed_version, version_required, operator):
                raise ValueError(f"{model_class} requires version {req}, but current {pkg} version is {installed_version} ")
        except PackageNotFoundError:
            raise ValueError(f"{model_class} requires version {req}, but {pkg} not installed.")


def get_model_local_path(pretrained_model_id_or_path, **kwargs):
    is_local = os.path.isdir(pretrained_model_id_or_path)
    if is_local:
        return pretrained_model_id_or_path
    else:
        return snapshot_download(pretrained_model_id_or_path, **kwargs)

def get_tokenizer(model_id_or_path, config, trust_remote_code: bool = False):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        return normalize_tokenizer(config, tokenizer)
    except Exception as e:
        logger.warning(f"Failed to auto-load tokenizer from pretrained_model_id_or_path: {e}. Please pass a tokenizer to `quantize()` or set model.tokenizer after `load()`.")
        return None


def ModelLoader(cls):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_id_or_path: str,
            quantize_config: QuantizeConfig,
            trust_remote_code: bool = False,
            torch_dtype: [str | torch.dtype] = "auto",
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int]] = None,
            **model_init_kwargs,
    ):
        # non-quantized models are always loaded into cpu
        cpu_device_map = {"": "cpu"}

        if quantize_config is None or not isinstance(quantize_config, QuantizeConfig):
            raise AttributeError("`quantize_config` must be passed and be an instance of QuantizeConfig.")

        quantize_config.calculate_bits_per_weight()

        if quantize_config.device is not None:
            if device is not None or device_map is not None:
                raise AttributeError("Passing device and device_map is not allowed when QuantizeConfig.device is set. Non-quantized model is always loaded as cpu. Please set QuantizeConfig.device for accelerator used in quantization or do not set for auto-selection.")

        if quantize_config.desc_act not in cls.supports_desc_act:
            raise ValueError(f"{cls} only supports desc_act={cls.supports_desc_act}, "
                             f"but quantize_config.desc_act is {quantize_config.desc_act}.")

        if cls.require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{pretrained_model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs_version)

        model_local_path = get_model_local_path(pretrained_model_id_or_path, **model_init_kwargs)

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        model_init_kwargs["trust_remote_code"] = trust_remote_code

        config = AutoConfig.from_pretrained(model_local_path, **model_init_kwargs)

        # normalize and auto select quantization device is not passed
        if quantize_config.device is None:
            quantize_config.device = auto_select_device(None, None)
        else:
            quantize_config.device = normalize_device(quantize_config.device)

        if cls.require_dtype:
            torch_dtype = cls.require_dtype

        if torch_dtype is None or torch_dtype == "auto" or not isinstance(torch_dtype, torch.dtype):
            # TODO FIX ME for `dynamic`, non-quantized modules should be in native type
            torch_dtype = auto_dtype(config=config, device=quantize_config.device, quant_inference=False)

        # enforce some values despite user specified
        # non-quantized models are always loaded into cpu
        model_init_kwargs["device_map"] = cpu_device_map
        model_init_kwargs["torch_dtype"] = torch_dtype

        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        model = cls.loader.from_pretrained(model_local_path, **model_init_kwargs)

        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "multimodal_max_length"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096
        model.eval()

        tokenizer = get_tokenizer(pretrained_model_id_or_path, config=config, trust_remote_code=trust_remote_code)

        return cls(
            model,
            quantized=False,
            quantize_config=quantize_config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            model_local_path=model_local_path,
        )

    cls.from_pretrained = from_pretrained

    @classmethod
    def from_quantized(
            cls,
            model_id_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            torch_dtype: [str | torch.dtype] = "auto",
            trust_remote_code: bool = False,
            verify_hash: Optional[Union[str, List[str]]] = None,
            **kwargs,
    ):
        # normalized device + device_map into single device
        device = normalize_device_device_map(device, device_map)

        # TODO need to normalize backend and others in a unified api
        if isinstance(backend, str):
            backend = BACKEND(backend)
        device = auto_select_device(device, backend)
        device_map = device.to_device_map()

        if backend == BACKEND.VLLM:
            import os

            # to optimize vllm inference, set an environment variable 'VLLM_ATTENTION_BACKEND' to 'FLASHINFER'.
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

        if backend == BACKEND.TRITON:
            from ..nn_modules.qlinear.tritonv2 import TRITON_AVAILABLE, TRITON_INSTALL_HINT
            if not TRITON_AVAILABLE:
                raise ValueError(TRITON_INSTALL_HINT)

        """load quantized model from local disk"""
        if cls.require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs_version)

        model_local_path = get_model_local_path(model_id_or_path, **kwargs)

        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }

        # == step1: prepare configs and file names == #
        config: PretrainedConfig = AutoConfig.from_pretrained(
            model_local_path,
            trust_remote_code=trust_remote_code,
            **cached_file_kwargs,
        )

        if cls.require_dtype:
            torch_dtype = cls.require_dtype

        if torch_dtype is None or torch_dtype == "auto" or not isinstance(torch_dtype, torch.dtype) :
            # TODO FIX ME for `dynamic`, non-quantized modules should be in native type
            torch_dtype = auto_dtype(config=config, device=device, quant_inference=True)

        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        quantize_config = QuantizeConfig.from_pretrained(model_local_path, **cached_file_kwargs, **kwargs)

        quantize_config.calculate_bits_per_weight()

        if backend == BACKEND.VLLM or backend == BACKEND.SGLANG:
            if quantize_config.format != FORMAT.GPTQ:
                raise ValueError(f"{backend} backend only supports FORMAT.GPTQ: actual = {quantize_config.format}")
            if backend == BACKEND.VLLM:
                from ..utils.vllm import load_model_by_vllm, vllm_generate

                model = load_model_by_vllm(
                    model=model_local_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )

                model.config = model.llm_engine.model_config

                cls.generate = lambda self, **kwargs: vllm_generate(self.model, **kwargs)

            elif backend == BACKEND.SGLANG:
                from ..utils.sglang import load_model_by_sglang, sglang_generate

                model, hf_config = load_model_by_sglang(
                    model=model_local_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
                model.config = hf_config
                cls.generate = lambda self, **kwargs: sglang_generate(self.model, **kwargs)
            return cls(
                model,
                quantized=True,
                quantize_config=quantize_config,
                qlinear_kernel=None,
                model_local_path=model_local_path,
            )

        if quantize_config.format == FORMAT.MARLIN:
            # format marlin requires marlin kernel
            if backend != BACKEND.MARLIN and backend != BACKEND.AUTO:
                raise TypeError(f"FORMAT.MARLIN requires BACKEND.AUTO or BACKEND.MARLIN: actual = `{backend}`.")
            backend = BACKEND.MARLIN

        marlin_compatible = False if backend == BACKEND.IPEX else _validate_marlin_device_support()

        # check for marlin compat for cuda device onnly
        if backend != BACKEND.MARLIN and device == DEVICE.CUDA:
            unsupported = _validate_marlin_compatibility(quantize_config)
            if unsupported is None and marlin_compatible:
                logger.info(
                    "You passed a model that is compatible with the Marlin kernel. Use `BACKEND.MARLIN` for optimal inference with batching on Nvidia GPU: `model = GPTQModel.load(..., backend=BACKEND.MARLIN)`."
                )

        if quantize_config.format == FORMAT.BITBLAS:
            # format bitblas requires bitblas kernel
            if backend != BACKEND.BITBLAS and backend != BACKEND.AUTO:
                raise TypeError(f"FORMAT.BITBLAS requires BACKEND.AUTO or BACKEND.BITBLAS: actual = `{backend}`.")
            backend = BACKEND.BITBLAS

        if backend == BACKEND.BITBLAS:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        possible_model_basenames = [
            f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g",
            "model",
        ]

        extensions = [".safetensors"]

        model_local_path = str(model_local_path)

        # Retrieve (and if necessary download) the quantized checkpoint(s).
        is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(
            model_id_or_path=model_local_path,
            extensions=extensions,
            possible_model_basenames=possible_model_basenames,
            **cached_file_kwargs,
        )

        # bin files have security issues: disable loading by default
        if ".bin" in resolved_archive_file:
            raise ValueError(
                "Loading of .bin files are not allowed due to safety. Please convert your model to safetensor or pytorch format."
            )

        quantize_config.runtime_format = quantize_config.format

        model_save_name = resolved_archive_file  # In case a model is sharded, this would be `model.safetensors.index.json` which may later break.
        if verify_hash:
            if is_sharded:
                verfieid = verify_sharded_model_hashes(model_save_name, verify_hash)
            else:
                verfieid = verify_model_hash(model_save_name, verify_hash)
            if not verfieid:
                raise ValueError(f"Hash verification failed for {model_save_name}")
            logger.info(f"Hash verification succeeded for {model_save_name}")

        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False

        init_contexts = [no_init_weights()]

        with ContextManagers(init_contexts):
            args = {}
            if device in [DEVICE.CUDA, DEVICE.ROCM]:
                if ATTN_IMPLEMENTATION in kwargs:
                    args[ATTN_IMPLEMENTATION] = kwargs.pop(ATTN_IMPLEMENTATION, None)
                if USE_FLASH_ATTENTION_2 in kwargs:
                    args[USE_FLASH_ATTENTION_2] = kwargs.pop(USE_FLASH_ATTENTION_2, None)
                if not args:
                    has_attn_implementation = Version(transformers.__version__) >= Version("4.46.0")
                    if is_flash_attn_2_available() and has_attn_implementation:
                        args = {ATTN_IMPLEMENTATION: "flash_attention_2"}
                    elif is_flash_attn_2_available() and not has_attn_implementation:
                        args = {USE_FLASH_ATTENTION_2: True}

            model = cls.loader.from_config(
                config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype, **args
            )
            model.checkpoint_file_name = model_save_name

            if cls.dynamic_expert_index is not None:
                num_experts = getattr(config, cls.dynamic_expert_index)
                cls.layer_modules = get_moe_layer_modules(layer_modules=cls.layer_modules,
                                                          num_experts=num_experts)

            layers = find_layers(model)
            ignore_layers = [cls.lm_head] + cls.base_modules

            for name in list(layers.keys()):
                # allow loading of quantized lm_head
                if quantize_config.lm_head and name == cls.lm_head:
                    continue

                if any(name.startswith(ignore_layer) for ignore_layer in ignore_layers) or all(
                        not name.endswith(ignore_layer) for sublist in cls.layer_modules for ignore_layer in sublist
                ):
                    # log non-lm-head quantizerd layers only
                    if name is not cls.lm_head:
                        logger.info(f"The layer {name} is not quantized.")
                    del layers[name]

            preload_qlinear_kernel = make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                backend=backend.AUTO if (backend == BACKEND.MARLIN and quantize_config.format == FORMAT.MARLIN) or backend == BACKEND.BITBLAS else backend,
                format=quantize_config.format,
                lm_head_name=cls.lm_head,
                desc_act=quantize_config.desc_act,
                sym=quantize_config.sym,
                dynamic=quantize_config.dynamic,
                device=device,
            )
            if preload_qlinear_kernel == IPEXQuantLinear:
                quantize_config.runtime_format = FORMAT.IPEX

        load_checkpoint_in_model = False
        # compat: runtime convert checkpoint gptq(v1) to gptq_v2 format
        if quantize_config.format == FORMAT.GPTQ and backend != BACKEND.IPEX:
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=torch_dtype,
                # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=model_save_name,
                device_map=device_map,
                offload_state_dict=True,
                offload_buffers=True,
            )
            # validate sym=False v1 loading needs to be protected for models produced with new v2 format codebase
            if not quantize_config.sym and not quantize_config.is_quantized_by_v2():
                raise ValueError(
                    f"Loading of a sym=False model with format={FORMAT.GPTQ} is only supported if produced by gptqmodel version >= {MIN_VERSION_WITH_V2}"
                )

            t = time.time()
            logger.info(f"Converting `{FORMAT_FIELD_JSON}` from `{FORMAT.GPTQ}` to `{FORMAT.GPTQ_V2}`.")
            model = convert_gptq_v1_to_v2_format(
                model,
                quantize_config=quantize_config,
                qlinear_kernel=preload_qlinear_kernel,
            )
            logger.info(f"Conversion complete: {time.time()-t}s")
            load_checkpoint_in_model = True
            quantize_config.runtime_format = FORMAT.GPTQ_V2

        if backend == BACKEND.MARLIN and (
                preload_qlinear_kernel == ExllamaV2QuantLinear or quantize_config.format == FORMAT.MARLIN):
            if is_sharded:
                raise ValueError(
                    "The loading of sharded checkpoints with Marlin is currently not supported."
                )
            if not _validate_marlin_device_support():
                raise ValueError(
                    f'Marlin kernel does not support this gpu with compute capability of `{torch.cuda.get_device_capability()}`. Please do not use `back=BACKEND.MARLIN`.'
                )

            # Validate the model can run in Marlin.
            if torch_dtype != torch.float16:
                raise ValueError("Marlin kernel requires torch_dtype=torch.float16.")

            _validate_marlin_compatibility(quantize_config, throw_error=True)

            # Prepare model for marlin load.
            # If is marlin serialized load then load directly. Otherwise, convert to marlin.
            model = prepare_model_for_marlin_load(
                model=model,
                quantize_config=quantize_config,
                quant_linear_class=preload_qlinear_kernel,
                torch_dtype=torch_dtype,
                current_model_save_name=model_save_name,
                device_map=device_map,
                desc_act=quantize_config.desc_act,
                sym=quantize_config.sym,
                load_checkpoint_in_model=load_checkpoint_in_model,
            )

        if backend == BACKEND.BITBLAS:
            from ..utils.bitblas import prepare_model_for_bitblas_load

            # Prepare model for bitblas load.
            # If is bitblas serialized load then load directly. Otherwise, convert to bitblas.
            model = prepare_model_for_bitblas_load(
                model=model,
                quantize_config=quantize_config,
                quant_linear_class=preload_qlinear_kernel,
                torch_dtype=torch_dtype,
                model_save_name=model_save_name,
                device_map=device_map,
                desc_act=quantize_config.desc_act,
                sym=quantize_config.sym,
                load_checkpoint_in_model=load_checkpoint_in_model,
            )

        # If we use marlin or bitblas to load the quantized model, the model is already a converted model,
        # and we no longer need to call load_checkpoint_in_model()
        if not load_checkpoint_in_model and backend != BACKEND.MARLIN and backend != BACKEND.BITBLAS:
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=torch_dtype,
                # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=model_save_name,
                device_map=device_map,
                # offload_state_dict=True,
                # offload_buffers=True,
            )

        # TODO: Why are we using this custom function and not dispatch_model?
        model = simple_dispatch_model(model, device_map)

        qlinear_kernel = select_quant_linear(
            bits=quantize_config.bits,
            dynamic=quantize_config.dynamic,
            group_size=quantize_config.group_size,
            desc_act=quantize_config.desc_act,
            sym=quantize_config.sym,
            backend=backend,
            format=quantize_config.format,
            device=device,
        )

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "multimodal_max_length"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        # Any post-initialization that require device information, for example buffers initialization on device.
        model = gptqmodel_post_init(model, use_act_order=quantize_config.desc_act, quantize_config=quantize_config)

        model.eval()

        tokenizer = get_tokenizer(model_id_or_path, config=config, trust_remote_code=trust_remote_code)

        if backend == BACKEND.MLX:
            import tempfile
            try:
                from mlx_lm import load
                from mlx_lm.utils import save_config, save_weights

                from ..utils.mlx import convert_gptq_to_mlx_weights, mlx_generate
            except ModuleNotFoundError as exception:
                raise type(exception)(
                    "GPTQModel load mlx model required dependencies are not installed.",
                    "Please install via `pip install gptqmodel[mlx] --no-build-isolation`.",
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                mlx_weights, mlx_config = convert_gptq_to_mlx_weights(model_id_or_path, model, quantize_config.to_dict())

                save_weights(temp_dir, mlx_weights, donate_weights=True)
                save_config(mlx_config, config_path=temp_dir + "/config.json")
                tokenizer.save_pretrained(temp_dir)

                model, _ = load(temp_dir)

                cls.generate = lambda _, **kwargs: mlx_generate(model=model, tokenizer=tokenizer, **kwargs)


        return cls(
            model,
            quantized=True,
            quantize_config=quantize_config,
            tokenizer=tokenizer,
            qlinear_kernel=qlinear_kernel,
            load_quantized_model=True,
            trust_remote_code=trust_remote_code,
            model_local_path=model_local_path,
        )

    cls.from_quantized = from_quantized

    return cls
