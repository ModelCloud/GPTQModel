# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-only K3 KDA padding conversion; runtime and saved state stay active-sized."""

from dataclasses import replace

import torch
from safetensors import safe_open
from transformers.core_model_loading import ConversionOps, WeightConverter
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_utils import load_state_dict as load_checkpoint


class ActiveKDAHeads(ConversionOps):
    def __init__(self, heads):
        self.heads = heads

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        value = next(iter(input_dict.values()))
        value = value[0] if isinstance(value, list) else value
        if value.shape == (self.heads,):
            return {target_patterns[0]: value}
        if (
            self.heads != 96
            or value.shape != (128,)
            or value.dtype != torch.float32
            or bool(torch.count_nonzero(value[96:].contiguous().view(torch.uint8)))
        ):
            raise ValueError(
                "unsupported or nonzero KDA A_log padding: expected active heads or F32[128] with zero tail for 96 heads"
            )
        return {target_patterns[0]: value[:96].clone()}

    @property
    def reverse_op(self):
        # Zero padding has no runtime meaning and is not reintroduced on save.
        return self


class KimiCheckpointMixin:
    @staticmethod
    def _load_pretrained_model(model, state_dict, checkpoint_files, load_config, expected_keys=None):
        weight_mapping = list(load_config.weight_mapping or [])
        heads = model.config.get_text_config().linear_attn_config
        heads = heads["num_heads"] if heads is not None else None
        if heads is not None:
            validator = ActiveKDAHeads(heads)
            if (
                state_dict is None
                and checkpoint_files
                and not all(file.endswith(".safetensors") for file in checkpoint_files)
            ):
                state_dict = {}
                for file in checkpoint_files:
                    state_dict.update(load_checkpoint(file, weights_only=load_config.weights_only))
            if state_dict is not None:
                for name, tensor in state_dict.items():
                    if name.endswith(".self_attn.A_log"):
                        validator.convert({name: tensor}, [name], [name])
            elif checkpoint_files and all(file.endswith(".safetensors") for file in checkpoint_files):
                for file in checkpoint_files:
                    with safe_open(file, framework="pt", device="cpu") as checkpoint:
                        for name in checkpoint.keys():
                            if name.endswith(".self_attn.A_log"):
                                validator.convert({name: checkpoint.get_tensor(name)}, [name], [name])
        for name, param in model.named_parameters():
            if name.endswith(".self_attn.A_log"):
                weight_mapping.append(
                    WeightConverter(
                        source_patterns=name,
                        target_patterns=name,
                        operations=[ActiveKDAHeads(param.numel())],
                    )
                )
        load_config = replace(load_config, weight_mapping=weight_mapping)
        return PreTrainedModel._load_pretrained_model(model, state_dict, checkpoint_files, load_config, expected_keys)
