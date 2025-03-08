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

import copy
import csv
import json
import os
import re
from os.path import isfile, join
from typing import Dict, Optional, Union

import torch
import transformers
from huggingface_hub import split_torch_state_dict_into_shards
from huggingface_hub.constants import SAFETENSORS_WEIGHTS_FILE_PATTERN
from safetensors.torch import save_file
from safetensors.torch import save_file as safe_save
from transformers import AutoConfig, PreTrainedTokenizerFast, ProcessorMixin
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils.generic import ContextManagers

from ..adapter.adapter import HF_ADAPTER_FILE_NAME, HF_ADAPTER_WEIGHT_KEY_PREFIX, Lora
from ..adapter.peft import LoraConfig
from ..quantization.config import (FORMAT, META_FIELD_DAMP_AUTO_INCREMENT, META_FIELD_DAMP_PERCENT, META_FIELD_MSE,
                                   META_FIELD_QUANTIZER, META_FIELD_STATIC_GROUPS, META_FIELD_TRUE_SEQUENTIAL,
                                   META_FIELD_URI, META_QUANTIZER_GPTQMODEL, META_VALUE_URI, MIN_VERSION_WITH_V2)
from ..utils.backend import BACKEND
from ..utils.logger import setup_logger
from ..utils.model import (convert_gptq_v2_to_v1_format, copy_py_files, find_modules,
                           get_model_files_size, get_moe_layer_modules, get_state_dict_for_save,
                           load_checkpoint_in_model_then_tie_weights, make_quant)
from ..utils.torch import torch_empty_cache
from ..version import __version__
from ._const import CPU, DEFAULT_MAX_SHARD_SIZE

log = setup_logger()

PROCESS_LOG_NAME = "process"
PROCESS_LOG_LAYER = "layer"
PROCESS_LOG_MODULE = "module"
QUANT_LOG_LOSS = "loss"
QUANT_LOG_DAMP = "damp"
PROCESS_LOG_TIME = "time"
PROCESS_LOG_FWD_TIME = "fwd_time"

EORA_DEFAULT_FILE = "eora.safetensors"

def ModelWriter(cls):
    def save_pretrained(
            self,
            save_dir: str,
            **kwargs,
    ):
        log.warn("You are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir=save_dir, **kwargs)

    cls.save_pretrained = save_pretrained

    def _eora_save(self, save_dir: str, model_save_dir: str = None):
        assert isinstance(self.quantize_config.adapter, Lora)

        assert hasattr(self, 'lora_results')

        # save lora tensors
        if self.lora_results:  # TODO REFRACTOR
            weights = {}
            target_modules = set()
            # convert the dict into safetensors compatible dict
            for key, d in self.lora_results.items():
                key = key.lower()
                simple_module_name = key.split(".")[-1] # mlp.gate_proj => gate_proj
                target_modules.add(simple_module_name)

                # while key.startswith('model.'):
                #     key = key.removeprefix('model.') # some HF models use model. or model.model.

                # must normalize key since HF can load weights as `model.` or not based on what AutoModel is used
                key = f"{HF_ADAPTER_WEIGHT_KEY_PREFIX}{key}"
                lora_rank = d.pop("rank")
                for lora_key, lora_weight in d.items():
                    assert isinstance(lora_weight, torch.Tensor)
                    weights[f"{key}.{lora_key}"] = lora_weight
                    log.info(f"Adapter: EoRA weights found -> `{key}.{lora_key}`, rank = `{lora_rank}`")

            weight_file_path = f"{save_dir.removesuffix('/')}/{HF_ADAPTER_FILE_NAME}"

            # dynamic rank
            rank_pattern = {}
            if self.quantize_config.dynamic:
                rank_pattern = self.quantize_config.extract_adapter_rank_patterns()

            lora_cfg = LoraConfig(base_model_name_or_path=model_save_dir,
                                  r=self.quantize_config.adapter.rank,
                                  lora_alpha=self.quantize_config.adapter.rank,
                                  target_modules=list(target_modules),
                                  rank_pattern=rank_pattern)
            lora_cfg.save_pretrained(save_dir=save_dir)

            log.info(f"Adapter: Saving EoRA weights to -> `{save_dir}`")

            save_file(tensors=weights, filename=weight_file_path, metadata={"format": "pt"})

            del self.lora_results  # TODO REFRACTOR

    cls.eora_save = _eora_save

    def save_quantized(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
            eora_path: Optional[str] = None,
    ):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        if self.quant_log:
            with open(os.path.join(save_dir, "quant_log.csv"), mode='w', newline='') as file:
                w = csv.writer(file)
                w.writerow([PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, QUANT_LOG_LOSS, QUANT_LOG_DAMP, PROCESS_LOG_TIME])
                w.writerows([[entry.get(PROCESS_LOG_LAYER), entry.get(PROCESS_LOG_MODULE), entry.get(QUANT_LOG_LOSS),
                              entry.get(QUANT_LOG_DAMP), entry.get(PROCESS_LOG_TIME)] for entry in self.quant_log])

        pre_quantized_size_mb = get_model_files_size(self.model_local_path)
        pre_quantized_size_gb = pre_quantized_size_mb / 1024

        quantizers = [f"{META_QUANTIZER_GPTQMODEL}:{__version__}"]
        if meta_quantizer:
            if len(meta_quantizer.split(":")) == 2:
                quantizers.append(meta_quantizer.replace(" ",""))
            else:
                log.warn(f"meta_quantizer: '{meta_quantizer}' format is invalid, expected: 'quantizer_name:version'")

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
            log.warn(
                f"Using 'format = {FORMAT.GPTQ_V2}': the serialized model is only supported by GPTQModel version >= {MIN_VERSION_WITH_V2}."
            )

        if not self.load_quantized_model:
            model = self.model
            # # internal is always gptq v2 but allow users to pass gptq (v1) via config
            if quantize_config.format == FORMAT.GPTQ:
                # Model qzeros may be edited in place.
                model = convert_gptq_v2_to_v1_format(
                    model, quantize_config=quantize_config, qlinear_kernel=self.qlinear_kernel
                )
        else:
            model = self.get_model_with_quantize(
                qcfg=quantize_config,
                model_id_or_path=self.model_local_path,
            )

        # --- start config save block ---
        # Save quantized config
        config.quantization_config = quantize_config.to_dict()
        self.model.config = config

        # Save model config, including generation_config
        # Use empty state_dict hack to bypass saving weights
        self.model.save_pretrained(save_dir, state_dict={}, is_main_process=True)

        # Save `quantize_config.json`
        quantize_config.save_pretrained(save_dir)

        def debug_saved_config(path):
            # List all files in the directory
            files = os.listdir(path)
            print("Files in directory:")
            for file in files:
                print(file)

            config_file_paths = ["generation_config.json", "config.json"]
            for file_name in config_file_paths:
                full_path = os.path.join(path, file_name)
                if os.path.isfile(full_path):
                    print(f"Content of saved `{file_name}`:")
                    with open(full_path, 'r') as config_file:
                        config_data = json.load(config_file)
                        print(json.dumps(config_data, indent=4))
                else:
                    print(f"`{file_name}` does not exist in the directory.")

        debug_saved_config(save_dir)

        # Save processor related config files. For example: preprocessor_config.json, chat_template.json
        if hasattr(self,"processor") and isinstance(self.processor, ProcessorMixin):
            self.processor.save_pretrained(save_dir)
        # --- end config save block ---

        model.to(CPU)
        state_dict = get_state_dict_for_save(model)

        model_base_name = "model"

        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        model_save_name = model_base_name + ".safetensors"

        if not self.qlinear_kernel.SUPPORTS_SHARDS and max_shard_size is not None:
            log.warn("Sharding is not supported for this quant. Disabling sharding.")
            max_shard_size = None

        if max_shard_size is None:
            if safetensors_metadata is None:
                safetensors_metadata = {}
            elif not isinstance(safetensors_metadata, dict):
                raise TypeError("safetensors_metadata must be a dictionary.")
            else:
                log.debug(f"Received safetensors_metadata: {safetensors_metadata}")
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
                            log.warn(
                                f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting."
                            )
                        new_safetensors_metadata[new_key] = new_value
                safetensors_metadata = new_safetensors_metadata
                if converted_keys:
                    log.debug(
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
                    log.debug(f"Received safetensors_metadata: {safetensors_metadata}")
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
                                log.warn(
                                    f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting.")
                            new_safetensors_metadata[new_key] = new_value
                    safetensors_metadata = new_safetensors_metadata
                    if converted_keys:
                        log.debug(
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

        # save lora
        if self.quantize_config.adapter:
            _eora_save(self, save_dir=eora_path if eora_path else self.quantize_config.adapter.path, model_save_dir=save_dir)

        # If the saved model is a loaded quantized model, do not calculate the size diff.
        if not self.load_quantized_model:
            total_size_gb = total_size_mb / 1024
            size_diff_mb = pre_quantized_size_mb - total_size_mb
            size_diff_gb = size_diff_mb / 1024
            percent_diff = (size_diff_mb / pre_quantized_size_mb) * 100
            log.info(f"Pre-Quantized model size: {pre_quantized_size_mb:.2f}MB, {pre_quantized_size_gb:.2f}GB")
            log.info(f"Quantized model size: {total_size_mb:.2f}MB, {total_size_gb:.2f}GB")
            log.info(f"Size difference: {size_diff_mb:.2f}MB, {size_diff_gb:.2f}GB - {percent_diff:.2f}%")

        # need to copy .py files for model/tokenizers not yet merged to HF transformers
        if self.trust_remote_code:
            copy_py_files(save_dir, model_id_or_path=self.model_local_path)

        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

            # fixed this issue: https://github.com/huggingface/transformers/issues/35832
            saved_tokenizer_config = get_tokenizer_config(save_dir)
            config_tokenizer_class = saved_tokenizer_config.get("tokenizer_class")
            # if the tokenizer is fast, but the tokenizer_config.json does not have Fast suffix, add "Fast" suffix
            if (not config_tokenizer_class.endswith("Fast")) and (
                isinstance(self.tokenizer.tokenizer, PreTrainedTokenizerFast)
                ):
                saved_tokenizer_config["tokenizer_class"] = saved_tokenizer_config["tokenizer_class"] + "Fast"
                with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                    json.dump(saved_tokenizer_config, f, indent=2, ensure_ascii=False)


    cls.save_quantized = save_quantized

    def get_model_with_quantize(self, qcfg, model_id_or_path):

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

            modules = find_modules(model)
            ignore_modules = [self.lm_head] + self.base_modules

            for name in list(modules.keys()):
                # allow loading of quantized lm_head
                if qcfg.lm_head and name == self.lm_head:
                    continue

                if any(name.startswith(ignore_module) for ignore_module in ignore_modules) or all(
                        not name.endswith(ignore_module) for sublist in self.layer_modules for ignore_module in sublist
                ):
                    # log non-lm-head quantizerd modules only
                    if name is not self.lm_head:
                        log.info(f"The layer {name} is not quantized.")
                    del modules[name]

            make_quant(
                model,
                quant_result=modules,
                qcfg=qcfg,
                backend=BACKEND.AUTO,
                lm_head_name=cls.lm_head,
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
