from __future__ import annotations

import copy
import json
import logging
import os
import re
from os.path import isfile, join

from gptqmodel.utils.device import check_cuda
from typing import Dict, List, Optional, Union

import accelerate
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers

from ..nn_modules.qlinear.qlinear_exllamav2 import ExllamaV2QuantLinear
from ..nn_modules.qlinear.qlinear_qbits import QBitsQuantLinear, qbits_dtype
from ..quantization import QuantizeConfig
from ..quantization.config import (FORMAT, FORMAT_FIELD_JSON, MIN_VERSION_WITH_V2)
from ..utils.backend import BACKEND
from ..utils.importer import select_quant_linear
from ..utils.marlin import (_validate_marlin_compatibility,
                            _validate_marlin_device_support, prepare_model_for_marlin_load)
from ..utils.model import (auto_dtype_from_config, convert_gptq_v1_to_v2_format,
                           find_layers, get_checkpoints,
                           get_moe_layer_modules, gptqmodel_post_init, make_quant,
                           simple_dispatch_model, verify_model_hash, verify_sharded_model_hashes)
from ._const import CPU, DEVICE, SUPPORTED_MODELS

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.propagate = False
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModelLoader():
    # some models require a different model loader, such as mllama which uses AutoModelForPreTraining
    model_loader = AutoModelForCausalLM

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            trust_remote_code: bool = False,
            use_liger_kernel: bool = False,
            torch_dtype: [str | torch.dtype] = "auto",
            require_trust_remote_code=None,
            **model_init_kwargs,
    ):
        """load un-quantized pretrained model to cpu"""
        got_cuda = check_cuda(raise_exception=False)

        if not got_cuda:
            try:
                pass
            except Exception as e:
                raise ValueError(
                    f"QBits is not available: {e}. Please install with `pip install -U intel-extension-for-transformers`."
                )

            model_init_kwargs["device"] = "cpu"
            torch_dtype = qbits_dtype()

        if require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{pretrained_model_name_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        model_init_kwargs["trust_remote_code"] = trust_remote_code

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **model_init_kwargs)

        if use_liger_kernel:
            from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

            apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(config.model_type, None)
            if apply_fn is None:
                raise ValueError(f"apply_fn is not defined for model type {config.model_type}")

            apply_fn()

        if torch_dtype == "auto":
            torch_dtype = auto_dtype_from_config(config)
        elif not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"torch_dtype value of `{torch_dtype}` is not a torch.dtype instance.")

        # enforce some values despite user specified
        model_init_kwargs["torch_dtype"] = torch_dtype

        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        if model_init_kwargs.get("cpu") != "cpu":
            torch.cuda.empty_cache()

        model = cls.model_loader.from_pretrained(pretrained_model_name_or_path, **model_init_kwargs)

        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096
        model.eval()

        return model

    @classmethod
    def from_quantized(
            cls,
            model_name_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            max_memory: Optional[dict] = None,
            device: Optional[Union[str, int]] = None,
            backend: BACKEND = BACKEND.AUTO,
            torch_dtype: [str | torch.dtype] = "auto",
            quantize_config: Optional[QuantizeConfig] = None,
            use_safetensors: bool = True,
            trust_remote_code: bool = False,
            format: Optional[FORMAT] = None,
            verify_hash: Optional[Union[str, List[str]]] = None,
            require_trust_remote_code: bool = False,
            dynamic_expert_index: Optional[str] = None,
            base_modules: List[str] = None,
            layer_modules: List[List[str]] = None,
            lm_head: str = "lm_head",
            layer_type: Union[List[str], str] = None,
            **kwargs,
    ):
        if backend == BACKEND.VLLM:
            import os

            # to optimize vllm inference, set an environment variable 'VLLM_ATTENTION_BACKEND' to 'FLASHINFER'.
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

        if backend == BACKEND.QBITS:
            device = CPU
            try:
                pass
            except Exception as e:
                raise ValueError(
                    f"QBits is not available: {e}. Please install with `pip install -U intel-extension-for-transformers`."
                )

            if torch_dtype is None or torch_dtype == "auto":
                torch_dtype = qbits_dtype()

        if backend != BACKEND.QBITS and not torch.cuda.is_available():
            raise EnvironmentError(
                "Load pretrained model to do quantization requires CUDA gpu. Please set backend=BACKEND.QBITS for cpu only quantization and inference.")

        """load quantized model from local disk"""
        if require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{model_name_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

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
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **cached_file_kwargs,
        )

        if torch_dtype == "auto":
            torch_dtype = auto_dtype_from_config(config, quant_inference=True)
        elif not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"torch_dtype value of `{torch_dtype}` is not a torch.dtype instance.")

        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        if quantize_config is None:
            quantize_config = QuantizeConfig.from_pretrained(
                model_name_or_path, format=format, **cached_file_kwargs, **kwargs
            )
        else:
            if not isinstance(quantize_config, QuantizeConfig):
                quantize_config = QuantizeConfig.from_quant_config(quantize_config, format)

        if backend == BACKEND.VLLM or backend == BACKEND.SGLANG:
            if quantize_config.format != FORMAT.GPTQ:
                raise ValueError(f"{backend} backend only supports FORMAT.GPTQ: actual = {quantize_config.format}")
            if backend == BACKEND.VLLM:
                from ..utils.vllm import load_model_by_vllm, vllm_generate

                model = load_model_by_vllm(
                    model=model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )

                model.config = model.llm_engine.model_config

                cls.generate = lambda self, **kwargs: vllm_generate(self.model, **kwargs)

            elif backend == BACKEND.SGLANG:
                from ..utils.sglang import load_model_by_sglang, sglang_generate

                model, hf_config = load_model_by_sglang(
                    model=model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
                model.config = hf_config
                cls.generate = lambda self, **kwargs: sglang_generate(self.model, **kwargs)
            return (
                model,
                quantize_config,
                None,  # qlinear_kernel
                False,  # load_quantized_model
                cls.generate
            )

        if quantize_config.format == FORMAT.MARLIN:
            # format marlin requires marlin kernel
            if backend != BACKEND.MARLIN and backend != BACKEND.AUTO:
                raise TypeError(f"FORMAT.MARLIN requires BACKEND.AUTO or BACKEND.MARLIN: actual = `{backend}`.")
            backend = BACKEND.MARLIN

        marlin_compatible = False if backend == BACKEND.QBITS else _validate_marlin_device_support()

        if backend != BACKEND.MARLIN:
            unsupported = _validate_marlin_compatibility(quantize_config)
            if unsupported is None and marlin_compatible:
                logger.info(
                    "You passed a model that is compatible with the Marlin int4*fp16 GPTQ kernel but backend is not BACKEND.MARLIN. We recommend using `backend=BACKEND.MARLIN` to use the optimized Marlin kernels for inference. Example: `model = GPTQModel.from_quantized(..., backend=BACKEND.MARLIN)`."
                )

        if quantize_config.format == FORMAT.BITBLAS:
            # format bitblas requires bitblas kernel
            if backend != BACKEND.BITBLAS and backend != BACKEND.AUTO:
                raise TypeError(f"FORMAT.BITBLAS requires BACKEND.AUTO or BACKEND.BITBLAS: actual = `{backend}`.")
            backend = BACKEND.BITBLAS

        if backend == BACKEND.BITBLAS:
            from ..nn_modules.qlinear.qlinear_bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        extensions = []
        if use_safetensors:
            extensions.append(".safetensors")
        else:
            extensions += [".pt", ".pth"]

        model_name_or_path = str(model_name_or_path)

        # Retrieve (and if necessary download) the quantized checkpoint(s).
        is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(
            model_name_or_path=model_name_or_path,
            extensions=extensions,
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
            model = cls.model_loader.from_config(
                config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
            )
            model.checkpoint_file_name = model_save_name

            if dynamic_expert_index is not None:
                num_experts = getattr(config, dynamic_expert_index)
                layer_modules = get_moe_layer_modules(layer_modules=layer_modules,
                                                          num_experts=num_experts)

            layers = find_layers(model)
            ignore_layers = [lm_head] + base_modules

            for name in list(layers.keys()):
                # allow loading of quantized lm_head
                if quantize_config.lm_head and name == lm_head:
                    continue

                if any(name.startswith(ignore_layer) for ignore_layer in ignore_layers) or all(
                        not name.endswith(ignore_layer) for sublist in layer_modules for ignore_layer in sublist
                ):
                    # log non-lm-head quantizerd layers only
                    if name is not lm_head:
                        logger.info(f"The layer {name} is not quantized.")
                    del layers[name]

            preload_qlinear_kernel = make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                backend=backend.AUTO if (
                                                backend == BACKEND.MARLIN and quantize_config.format == FORMAT.MARLIN) or backend == BACKEND.BITBLAS else backend,
                format=quantize_config.format,
                desc_act=quantize_config.desc_act,
                dynamic=quantize_config.dynamic,
            )
            if preload_qlinear_kernel == QBitsQuantLinear:
                quantize_config.runtime_format = FORMAT.QBITS
            model.tie_weights()

        # == step3: load checkpoint and dispatch == #
        if isinstance(device_map, str) and device_map not in [
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )
        if isinstance(device_map, dict):
            max_memory = None
        else:
            if device is None and not device_map and not max_memory:
                device_map = "auto"
            if device is not None:
                device = torch.device(device)
                if not max_memory and not device_map:
                    device_map = {"": device.index if device.type == DEVICE.CUDA else device.type}
            if not isinstance(device_map, dict) and device_map != "sequential":
                max_memory = accelerate.utils.get_balanced_memory(
                    model=model,
                    max_memory=max_memory,
                    no_split_module_classes=[layer_type] if isinstance(layer_type, str) else layer_type,
                    low_zero=(device_map == "balanced_low_0"),
                )
        if not isinstance(device_map, dict):
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[layer_type] if isinstance(layer_type, str) else layer_type,
            )

        load_checkpoint_in_model = False
        # compat: runtime convert checkpoint gptq(v1) to gptq_v2 format
        if quantize_config.format == FORMAT.GPTQ:
            accelerate.load_checkpoint_in_model(
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

            logger.info(
                f"Compatibility: converting `{FORMAT_FIELD_JSON}` from `{FORMAT.GPTQ}` to `{FORMAT.GPTQ_V2}`.")
            model = convert_gptq_v1_to_v2_format(
                model,
                quantize_config=quantize_config,
                qlinear_kernel=preload_qlinear_kernel,
            )
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
            accelerate.load_checkpoint_in_model(
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
        )

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
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

        return (
            model,
            quantize_config,
            qlinear_kernel,
            True,  # load_quantized_model
            None, # return None if not SGLANG
        )
