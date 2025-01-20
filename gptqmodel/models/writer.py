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

import copy
import csv
import json
import os
import re
from os.path import isfile, join
from typing import Dict, Optional

import torch
import transformers
from huggingface_hub import split_torch_state_dict_into_shards
from huggingface_hub.constants import SAFETENSORS_WEIGHTS_FILE_PATTERN
from safetensors.torch import save_file as safe_save
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers

from ..quantization.config import (FORMAT, META_FIELD_DAMP_AUTO_INCREMENT, META_FIELD_DAMP_PERCENT, META_FIELD_MSE,
                                   META_FIELD_QUANTIZER, META_FIELD_STATIC_GROUPS, META_FIELD_TRUE_SEQUENTIAL,
                                   META_FIELD_URI, META_QUANTIZER_GPTQMODEL, META_VALUE_URI, MIN_VERSION_WITH_V2)
from ..utils.backend import BACKEND
from ..utils.logger import setup_logger
from ..utils.model import (convert_gptq_v2_to_v1_format, copy_py_files, find_layers, get_model_files_size,
                           get_moe_layer_modules, get_state_dict_for_save, make_quant,
                           load_checkpoint_in_model_then_tie_weights)
from ..utils.torch import torch_empty_cache
from ..version import __version__
from ._const import CPU

logger = setup_logger()

QUANT_LOG_LAYER = "layer"
QUANT_LOG_MODULE = "module"
QUANT_LOG_LOSS = "loss"
QUANT_LOG_DAMP = "damp"
QUANT_LOG_TIME = "time"
QUANT_LOG_FWD_TIME = "fwd_time"

def ModelWriter(cls):

    def save_pretrained(
            self,
            save_dir: str,
            **kwargs,
    ):
        logger.warning("You are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir=save_dir, **kwargs)

    cls.save_pretrained = save_pretrained

    def save_quantized(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[str] = None,
            meta_quantizer: Optional[str] = None,
    ):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        if self.quant_log:
            with open(os.path.join(save_dir, "quant_log.csv"), mode='w', newline='') as file:
                w = csv.writer(file)
                w.writerow([QUANT_LOG_LAYER, QUANT_LOG_MODULE, QUANT_LOG_LOSS, QUANT_LOG_DAMP, QUANT_LOG_TIME])
                w.writerows([[entry.get(QUANT_LOG_LAYER), entry.get(QUANT_LOG_MODULE), entry.get(QUANT_LOG_LOSS),
                              entry.get(QUANT_LOG_DAMP), entry.get(QUANT_LOG_TIME)] for entry in self.quant_log])

        pre_quantized_size_mb = get_model_files_size(self.model_local_path)
        pre_quantized_size_gb = pre_quantized_size_mb / 1024

        quantizers = [f"{META_QUANTIZER_GPTQMODEL}:{__version__}"]
        if meta_quantizer:
            if len(meta_quantizer.split(":")) == 2:
                quantizers.append(meta_quantizer.replace(" ",""))
            else:
                logger.warning(f"meta_quantizer: '{meta_quantizer}' format is invalid, expected: 'quantizer_name:version'")

        # write gptqmodel tooling fingerprint to config
        self.quantize_config.meta_set_versionable(
            key=META_FIELD_QUANTIZER,
            value=quantizers
        )

        self.quantize_config.meta_set(
            key=META_FIELD_URI,
            value=META_VALUE_URI,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_PERCENT,
            value=self.quantize_config.damp_percent
        )

        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_AUTO_INCREMENT,
            value=self.quantize_config.damp_auto_increment
        )

        self.quantize_config.meta_set(
            key=META_FIELD_STATIC_GROUPS,
            value=self.quantize_config.static_groups
        )

        self.quantize_config.meta_set(
            key=META_FIELD_TRUE_SEQUENTIAL,
            value=self.quantize_config.true_sequential
        )

        self.quantize_config.meta_set(
            key=META_FIELD_MSE,
            value=self.quantize_config.mse
        )


        # The config, quantize_config and model may be edited in place in save_quantized.
        config = copy.deepcopy(self.model.config)
        quantize_config = copy.deepcopy(self.quantize_config)

        if not self.quantized:
            raise ValueError("Save aborted as model is not quantized. Please call `quantize()` first.")

        if quantize_config.format == FORMAT.GPTQ_V2:
            logger.warning(
                f"Using 'format = {FORMAT.GPTQ_V2}': the serialized model is only supported by GPTQModel version >= {MIN_VERSION_WITH_V2}."
            )

        if not self.load_quantized_model:
            model = self.model
            # # internal is always gptq v2 but allow users to pass gptq (v1) via config
            if quantize_config.format == FORMAT.GPTQ:
                # fix ModelCloud/GPTQModel/issues/47
                # fix gptqmodel_cuda cannot be serialized
                # no need to set it back, no calculation below
                if quantize_config.bits != 4:
                    cuda_name_modules = {}
                    from gptqmodel.nn_modules.qlinear.dynamic_cuda import DynamicCudaQuantLinear
                    for name, module in model.named_modules():
                        if isinstance(module, DynamicCudaQuantLinear):
                            cuda_name_modules[name] = module.gptqmodel_cuda
                            module.gptqmodel_cuda = None

                    for name, module in model.named_modules():
                        if isinstance(module, DynamicCudaQuantLinear) and name in cuda_name_modules:
                            module.gptqmodel_cuda = cuda_name_modules[name]

                    del cuda_name_modules

                # Model qzeros may be edited in place.
                model = convert_gptq_v2_to_v1_format(
                    model, quantize_config=quantize_config, qlinear_kernel=self.qlinear_kernel
                )
        else:
            model = self.get_model_with_quantize(
                quantize_config=quantize_config,
                model_id_or_path=self.model_local_path,
            )

        model.to(CPU)
        state_dict = get_state_dict_for_save(model)

        model_base_name = "model"

        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        model_save_name = model_base_name + ".safetensors"

        if not self.qlinear_kernel.SUPPORTS_SHARDS and max_shard_size is not None:
            logger.warning("Sharding is not supported for this quant. Disabling sharding.")
            max_shard_size = None

        if max_shard_size is None:
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
            total_size_mb = os.path.getsize(join(save_dir, model_save_name)) / (1024 * 1024)
        else:
            file_name_pattern = SAFETENSORS_WEIGHTS_FILE_PATTERN

            # Shard checkpoint
            state_dict_split= split_torch_state_dict_into_shards(state_dict, max_shard_size=max_shard_size, filename_pattern=file_name_pattern)

            # Clean the folder from a previous save
            for filename in os.listdir(save_dir):
                full_filename = join(save_dir, filename)

                # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
                filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
                reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

                if (
                        filename.startswith(model_base_name)
                        and isfile(full_filename)
                        and filename not in state_dict_split.filename_to_tensors.keys()
                        and reg.fullmatch(filename_no_suffix) is not None
                ):
                    os.remove(full_filename)

            total_size_mb = 0
            # Save the model
            for filename, tensors in state_dict_split.filename_to_tensors.items():
                shard = {tensor: state_dict[tensor] for tensor in tensors}
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

                safe_save(shard, join(save_dir, filename), safetensors_metadata)
                shard_size_mb = os.path.getsize(join(save_dir, filename)) / (1024 * 1024)
                total_size_mb += shard_size_mb

            if state_dict_split.is_sharded:
                index = {
                    "metadata": state_dict_split.metadata,
                    "weight_map": state_dict_split.tensor_to_filename,
                }

                index_save_name = model_save_name + ".index.json"
                index_save_path = join(save_dir, index_save_name)
                # Save the index as well
                with open(index_save_path, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)

        # If the saved model is a loaded quantized model, do not calculate the size diff.
        if not self.load_quantized_model:
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
        if self.trust_remote_code:
            copy_py_files(save_dir, model_id_or_path=self.model_local_path)

        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

    cls.save_quantized = save_quantized

    def get_model_with_quantize(self, quantize_config, model_id_or_path):

        config = AutoConfig.from_pretrained(
            model_id_or_path,
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
            model = cls.loader.from_config(
                config, torch_dtype=torch.float16
            )

            if self.dynamic_expert_index is not None:
                num_experts = getattr(config, self.dynamic_expert_index)
                _ = get_moe_layer_modules(layer_modules=self.layer_modules,
                                                      num_experts=num_experts)

            layers = find_layers(model)
            ignore_layers = [self.lm_head] + self.base_modules

            for name in list(layers.keys()):
                # allow loading of quantized lm_head
                if quantize_config.lm_head and name == self.lm_head:
                    continue

                if any(name.startswith(ignore_layer) for ignore_layer in ignore_layers) or all(
                        not name.endswith(ignore_layer) for sublist in self.layer_modules for ignore_layer in sublist
                ):
                    # log non-lm-head quantizerd layers only
                    if name is not self.lm_head:
                        logger.info(f"The layer {name} is not quantized.")
                    del layers[name]

            make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                backend=BACKEND.AUTO,
                format=quantize_config.format,
                lm_head_name=cls.lm_head,
                desc_act=quantize_config.desc_act,
                pack=True,
            )

        load_checkpoint_in_model_then_tie_weights(
            model,
            dtype=torch.float16,
            # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
            checkpoint=self.checkpoint_file_name,
            # device_map=device_map,
            # offload_state_dict=True,
            # offload_buffers=True,
        )
        torch_empty_cache()
        return model

    cls.get_model_with_quantize = get_model_with_quantize

    return cls
