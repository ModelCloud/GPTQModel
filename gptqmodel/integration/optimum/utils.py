#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
from typing import Callable, Optional, Union

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D

from ...utils.logger import setup_logger
from .constants import BLOCK_PATTERNS, SEQLEN_KEYS_TRANFORMERS

ori_save_pretrained = PreTrainedModel.save_pretrained

logger = setup_logger()


"""
Set of utilities to get specific attributes of a model
"""


def get_layers(module: nn.Module, layers=[Conv1D, nn.Conv2d, nn.Linear], prefix: Optional[str] = None, name: str = ""):
    """
    Get all the layers with a specific prefix in the module
    Args:
        module (`nn.Module`):
            The module that contains our layers
        layers (`list`, defaults to `[Conv1D, nn.Conv2d, nn.Linear]`):
            Type of the layers that we want to get
        prefix (`Optional[str]`, defaults to `None`):
            Prefix of layers
        name (`str`, defaults to `""`):
            Used for recursion. Don't modify

    Returns:
        `Dict[str,Union[Conv1D, nn.Conv2d, nn.Linear]]`: Mapping of the name of the layer and the actual layer
    """
    for layer in layers:
        if isinstance(module, layer):
            if prefix is not None:
                if name.startswith(prefix):
                    return {name: module}
            else:
                return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(get_layers(child, layers=layers, prefix=prefix, name=name + "." + name1 if name != "" else name1))
    return res


def get_block_name_with_pattern(model: nn.Module):
    """
    Get the name of the module that contains the transformers blocks by checking if any modules has a specific pattern

    Args:
        model (`nn.Module`):
        The input model
    Returns:
        `str`: The name of the module that contains the Transformer blocks.
    """
    modules_names = [n for n, _ in model.named_modules()]
    for pattern_candidate in BLOCK_PATTERNS:
        pattern_candidate = pattern_candidate
        if any(pattern_candidate in name for name in modules_names):
            return pattern_candidate
    raise ValueError("Block pattern could not be match. Pass `block_name_to_quantize` argument in `quantize_model`")


def get_preceding_modules(model: nn.Module, module_name: str):
    previous_module_name = []
    stop_adding = False

    def _get_preceding_modules(model: nn.Module, module_name: str, name: str = ""):
        nonlocal stop_adding
        for name_bis, child in model.named_children():
            new_name = name + "." + name_bis if name != "" else name_bis
            if new_name == module_name:
                stop_adding = True
                break
            _get_preceding_modules(child, module_name, name=new_name)
        if not stop_adding:
            previous_module_name.append(name)
        return previous_module_name

    return _get_preceding_modules(model, module_name)


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def get_seqlen(model: nn.Module):
    if hasattr(model, "config"):
        model_config = model.config.to_dict()
        if any(k in model_config for k in SEQLEN_KEYS_TRANFORMERS):
            for key in SEQLEN_KEYS_TRANFORMERS:
                if key in model_config:
                    return model_config[key]
    logger.info(
        "We couldn't get the model sequence length. Setting it to 2048. You can overwrite this value by passing `model_seqlen` in` GPTQQuantizer`"
    )
    return 2048


def monkey_patch_gptqmodel_into_transformers():
    # monkey_patch transformers.utils.quantization_config.GPTQConfig.post_init()
    # Because it checks the auto_gptq version
    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        import importlib

        from packaging import version
        print("monkey patch postin")
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")
        if self.dataset is not None:
            if isinstance(self.dataset, str):
                if self.dataset in ["ptb", "ptb-new"]:
                    raise ValueError(
                        f"""{self.dataset} dataset was deprecated. You can only choose between
                        ['wikitext2','c4','c4-new']"""
                    )
                if self.dataset not in ["wikitext2", "c4", "c4-new"]:
                    raise ValueError(
                        f"""You have entered a string value for dataset. You can only choose between
                        ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
                    )
            elif not isinstance(self.dataset, list):
                raise ValueError(
                    f"""dataset needs to be either a list of string or a value in
                    ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
                )

        if self.use_exllama is None:
            # New default behaviour
            self.use_exllama = True

        if self.bits == 4 and self.use_exllama:
            optimum_version = version.parse(importlib.metadata.version("optimum"))
            # autogptq_version = version.parse(importlib.metadata.version("auto_gptq"))
            # if optimum_version <= version.parse("1.13.2") or autogptq_version <= version.parse("0.4.2"):
            if optimum_version <= version.parse("1.13.2"):
                raise ValueError(
                    # f"You need optimum > 1.13.2 and auto-gptq > 0.4.2 . Make sure to have that version installed - detected version : optimum {optimum_version} and autogptq {autogptq_version}"
                    f"You need optimum > 1.13.2 . Make sure to have that version installed - detected version : optimum {optimum_version}"
                )
        if self.modules_in_block_to_quantize is not None:
            optimum_version = version.parse(importlib.metadata.version("optimum"))
            if optimum_version < version.parse("1.15.0"):
                raise ValueError(
                    "You current version of `optimum` does not support `modules_in_block_to_quantize` quantization argument, please upgrade `optimum` package to a version superior than 1.15.0 ."
                )

    from transformers.utils.quantization_config import GPTQConfig
    GPTQConfig.post_init = post_init

    from transformers.quantizers import auto

    from .hf_quantizer_gptq import GptqHfQuantizer

    auto.AUTO_QUANTIZER_MAPPING["gptq"] = GptqHfQuantizer

    # TODO monkey patch GPTQConfig?

    # model.save_pretrained() will not call optimum.quantizer.GPTQModelQuantizer.save(),
    # we need to monkey patch save_pretrained() to convert gptq_v2 to gptq_v1 format.
    def monkey_patch_save_pretrained(self,
                                     save_directory: Union[str, os.PathLike],
                                     is_main_process: bool = True,
                                     state_dict: Optional[dict] = None,
                                     save_function: Callable = torch.save,
                                     push_to_hub: bool = False,
                                     max_shard_size: Union[int, str] = "5GB",
                                     safe_serialization: bool = True,
                                     variant: Optional[str] = None,
                                     token: Optional[Union[str, bool]] = None,
                                     save_peft_format: bool = True,
                                     **kwargs, ):
        hf_quantizer = getattr(self, "hf_quantizer", None)
        if hf_quantizer:
            ori_model = getattr(self, "model", None)
            assert ori_model

            model = hf_quantizer.optimum_quantizer.convert_gptq_v2_to_v1(ori_model)
            setattr(self, "model", model)

        ori_save_pretrained(self, save_directory, is_main_process, state_dict, save_function, push_to_hub,
                            max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)

    PreTrainedModel.save_pretrained = monkey_patch_save_pretrained
