from __future__ import annotations

import copy
import json
import csv
import logging
import os
import re
from os.path import isfile, join
from typing import Dict, List, Optional

import accelerate
import torch
import torch.nn as nn
import transformers
from safetensors.torch import save_file as safe_save
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import no_init_weights, shard_checkpoint
from transformers.utils.generic import ContextManagers

from ..quantization import QuantizeConfig
from ..quantization.config import (FORMAT, META_FIELD_DAMP_AUTO_INCREMENT, META_FIELD_DAMP_PERCENT,
                                   META_FIELD_QUANTIZER, META_FIELD_URI, META_QUANTIZER_GPTQMODEL, META_VALUE_URI,
                                   MIN_VERSION_WITH_V2)
from ..utils.backend import BACKEND
from ..utils.model import (convert_gptq_v2_to_v1_format, copy_py_files, find_layers,get_model_files_size,
                           get_moe_layer_modules, make_quant)
from ..version import __version__
from ._const import CPU

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.propagate = False
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModelWriter():
    # some models require a different model loader, such as mllama which uses AutoModelForPreTraining
    model_loader = AutoModelForCausalLM

    @classmethod
    def save_quantized(
            cls,
            save_dir: str,
            quantized: bool,
            model_name_or_path: str,
            model: PreTrainedModel,
            load_quantized_model: bool,
            qlinear_kernel: nn.Module,
            trust_remote_code: bool,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            use_safetensors: bool = True,
            max_shard_size: Optional[str] = None,
            quantize_config: QuantizeConfig = None,
            dynamic_expert_index: Optional[str] = None,
            base_modules: List[str] = None,
            lm_head: str = None,
            layer_modules: List[List[str]] = None,
            checkpoint_file_name=None,
            quant_log:Optional[List[Dict[str, str]]]=None,
    ):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        if quant_log:
            with open(os.path.join(save_dir, "quant_log.csv"), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["module", "loss", "damp", "layer_forward_time", "w_clone_time", "quant_time"])
                writer.writerows([[f"{entry['layer']}.{entry['module']}", entry['loss'], entry['damp'], entry['layer_forward_time'], entry['w_clone_time'], entry['quant_time']] for entry in quant_log])

        pre_quantized_size_mb = get_model_files_size(model_name_or_path)
        pre_quantized_size_gb = pre_quantized_size_mb / 1024

        # write gptqmodel tooling fingerprint to config
        quantize_config.meta_set_versionable(
            key=META_FIELD_QUANTIZER,
            value=META_QUANTIZER_GPTQMODEL,
            version=__version__,
        )

        quantize_config.meta_set(
            key=META_FIELD_URI,
            value=META_VALUE_URI,
        )

        quantize_config.meta_set(
            key=META_FIELD_DAMP_PERCENT,
            value=quantize_config.damp_percent
        )

        quantize_config.meta_set(
            key=META_FIELD_DAMP_AUTO_INCREMENT,
            value=quantize_config.damp_auto_increment
        )

        # The config, quantize_config and model may be edited in place in save_quantized.
        config = copy.deepcopy(model.config)
        quantize_config = copy.deepcopy(quantize_config)

        if not quantized:
            raise ValueError("Save aborted as model is not quantized. Please call `quantize()` first.")

        if quantize_config.format == FORMAT.GPTQ_V2:
            logger.warning(
                f"Using 'format = {FORMAT.GPTQ_V2}': the serialized model is only supported by GPTQModel version >= {MIN_VERSION_WITH_V2}."
            )

        if not load_quantized_model:
            # # internal is always gptq v2 but allow users to pass gptq (v1) via config
            if quantize_config.format == FORMAT.GPTQ:
                # Model qzeros may be edited in place.
                model = convert_gptq_v2_to_v1_format(
                    model, quantize_config=quantize_config, qlinear_kernel=qlinear_kernel
                )
        else:
            model = cls.get_model_with_quantize(
                quantize_config=quantize_config,
                model_name_or_path=model_name_or_path,
                dynamic_expert_index=dynamic_expert_index,
                lm_head=lm_head,
                base_modules=base_modules,
                layer_modules=layer_modules,
                checkpoint_file_name=checkpoint_file_name,
            )

        model.to(CPU)
        state_dict = model.state_dict()

        model_base_name = "model"

        if use_safetensors:
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            model_save_name = model_base_name + ".safetensors"
        else:
            model_save_name = model_base_name + ".pt"

        if not qlinear_kernel.SUPPORTS_SHARDS and max_shard_size is not None:
            logger.warning("Sharding is not supported for this quant. Disabling sharding.")
            max_shard_size = None

        if max_shard_size is None:
            if use_safetensors:
                if safetensors_metadata is None:
                    safetensors_metadata = {}
                elif not isinstance(safetensors_metadata, dict):
                    raise TypeError("safetensors_metadata must be a dictionary.")
                else:
                    logger.debug(f"Received safetensors_metadata: {safetensors_metadata}")
                    new_safetensors_metadata = {}
                    converted_keys = False
                    for key, value in safetensors_metadata.items():
                        if not isinstance(key, str) or not isinstance(value, str):
                            converted_keys = True
                            try:
                                new_key = str(key)
                                new_value = str(value)
                            except Exception as e:
                                raise TypeError(
                                    f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}"
                                )
                            if new_key in new_safetensors_metadata:
                                logger.warning(
                                    f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting."
                                )
                            new_safetensors_metadata[new_key] = new_value
                    safetensors_metadata = new_safetensors_metadata
                    if converted_keys:
                        logger.debug(
                            f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}"
                        )

                # Format is required to enable Accelerate to load the metadata
                # otherwise it raises an OSError
                safetensors_metadata["format"] = "pt"
                safe_save(state_dict, join(save_dir, model_save_name), safetensors_metadata)
            else:
                logger.warning(
                    "We highly suggest saving quantized model using safetensors format for security reasons. Please set `use_safetensors=True` whenever possible.")
                torch.save(model.state_dict(), join(save_dir, model_save_name))
            total_size_mb = os.path.getsize(join(save_dir, model_save_name)) / (1024 * 1024)
        else:
            # Shard checkpoint
            shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=model_save_name)

            # Clean the folder from a previous save
            for filename in os.listdir(save_dir):
                full_filename = join(save_dir, filename)

                # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
                filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
                reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

                if (
                        filename.startswith(model_base_name)
                        and isfile(full_filename)
                        and filename not in shards.keys()
                        and reg.fullmatch(filename_no_suffix) is not None
                ):
                    os.remove(full_filename)

            total_size_mb = 0
            # Save the model
            for shard_file, shard in shards.items():
                if use_safetensors:
                    if safetensors_metadata is None:
                        safetensors_metadata = {}
                    elif not isinstance(safetensors_metadata, dict):
                        raise TypeError("safetensors_metadata must be a dictionary.")
                    else:
                        logger.debug(f"Received safetensors_metadata: {safetensors_metadata}")
                        new_safetensors_metadata = {}
                        converted_keys = False
                        for key, value in safetensors_metadata.items():
                            if not isinstance(key, str) or not isinstance(value, str):
                                converted_keys = True
                                try:
                                    new_key = str(key)
                                    new_value = str(value)
                                except Exception as e:
                                    raise TypeError(
                                        f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}")
                                if new_key in new_safetensors_metadata:
                                    logger.warning(
                                        f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting.")
                                new_safetensors_metadata[new_key] = new_value
                        safetensors_metadata = new_safetensors_metadata
                        if converted_keys:
                            logger.debug(
                                f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}")

                    # Format is required to enable Accelerate to load the metadata
                    # otherwise it raises an OSError
                    safetensors_metadata["format"] = "pt"

                    safe_save(shard, join(save_dir, shard_file), safetensors_metadata)
                else:
                    torch.save(shard, join(save_dir, shard_file))
                shard_size_mb = os.path.getsize(join(save_dir, shard_file)) / (1024 * 1024)
                total_size_mb += shard_size_mb

            if index is not None:
                index_save_name = model_save_name + ".index.json"
                index_save_path = join(save_dir, index_save_name)
                # Save the index as well
                with open(index_save_path, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)

        total_size_gb = total_size_mb / 1024
        size_diff_mb = pre_quantized_size_mb - total_size_mb
        size_diff_gb = size_diff_mb / 1024
        percent_diff = (size_diff_mb / pre_quantized_size_mb) * 100
        logger.info(f"Pre-Quantized model size: {pre_quantized_size_mb:.2f}MB, {pre_quantized_size_gb:.2f}GB")
        logger.info(f"Quantized model size: {total_size_mb:.2f}MB, {total_size_gb:.2f}GB")
        logger.info(f"Size difference: {size_diff_mb:.2f}MB, {size_diff_gb:.2f}GB - {percent_diff:.2f}%")

        config.quantization_config = quantize_config.to_dict()
        config.save_pretrained(save_dir)

        quantize_config.save_pretrained(save_dir)

        # need to copy .py files for model/tokenizers not yet merged to HF transformers
        if trust_remote_code:
            copy_py_files(save_dir, model_id_or_path=model_name_or_path)

    @classmethod
    def get_model_with_quantize(cls,
                                quantize_config,
                                model_name_or_path,
                                dynamic_expert_index,
                                lm_head,
                                base_modules,
                                layer_modules,
                                checkpoint_file_name,
        ):

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        transformers.modeling_utils._init_weights = False
        init_contexts = [no_init_weights()]
        with ContextManagers(init_contexts):
            model = cls.model_loader.from_config(
                config, torch_dtype=torch.float16
            )

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

            make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                backend=BACKEND.AUTO,
                format=quantize_config.format,
                desc_act=quantize_config.desc_act,
                pack=True,
            )
            model.tie_weights()

        accelerate.load_checkpoint_in_model(
            model,
            dtype=torch.float16,
            # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
            checkpoint=checkpoint_file_name,
            # device_map=device_map,
            # offload_state_dict=True,
            # offload_buffers=True,
        )
        torch.cuda.empty_cache()
        return model
