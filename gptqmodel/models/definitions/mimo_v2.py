# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
import json
import os
from collections.abc import Callable
from functools import wraps
from importlib import import_module
from typing import Any

from safetensors import safe_open
from torch import nn

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ...utils.mimo import is_mimo_mixed_source
from ...utils.model import nested_move_to
from ...utils.torch import CPU
from ..base import BaseQModel


_REPLAY_INPUTS_KEY = "_mimo_v2_replay_inputs"
_CAPTURE_ATTENTION_MASK_ATTR = "_mimo_v2_capture_attention_mask"


def _compatible_mask(function: Callable) -> Callable:
    """Adapt legacy MiMo keywords within its remote modeling module."""
    if getattr(function, "_gptqmodel_mimo_mask_compat", False):
        return function
    parameters = inspect.signature(function).parameters
    if "inputs_embeds" not in parameters:
        return function

    @wraps(function)
    def mask(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if "cache_position" not in parameters:
            kwargs.pop("cache_position", None)
        return function(*args, **kwargs)

    mask._gptqmodel_mimo_mask_compat = True
    return mask


class MimoV2QModel(BaseQModel):
    # MiMo V2 uses repository-defined configuration/modeling classes.
    require_trust_remote_code = True

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # The remote base model does not instantiate these auxiliary predictors.
    out_of_model_tensors = {"prefixes": ["model.mtp"]}

    # MiMo V2 supports both split q/k/v and fused qkv checkpoints, and individual
    # layers can be dense MLP or routed MoE according to config.moe_layer_freq.
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("qkv_proj:0", "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    @staticmethod
    def _checkpoint_has_tensor(model_local_path: str, tensor_name: str) -> bool:
        if not model_local_path:
            return True

        index_path = os.path.join(model_local_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as fp:
                weight_map = json.load(fp).get("weight_map", {})
            return tensor_name in weight_map

        tensor_file = os.path.join(model_local_path, "model.safetensors")
        if os.path.isfile(tensor_file):
            with safe_open(tensor_file, framework="pt", device="cpu") as handler:
                return tensor_name in handler.keys()

        return True

    @staticmethod
    def _drop_visual_merger_biases_if_checkpoint_omits_them(model, model_local_path: str) -> None:
        visual = getattr(model, "visual", None)
        merger = getattr(visual, "merger", None)
        if not isinstance(merger, nn.Module):
            return

        for module_name, module in merger.named_modules():
            if getattr(module, "bias", None) is None:
                continue

            prefix = "visual.merger"
            if module_name:
                prefix = f"{prefix}.{module_name}"
            weight_name = f"{prefix}.weight"
            bias_name = f"{prefix}.bias"
            if MimoV2QModel._checkpoint_has_tensor(model_local_path, bias_name):
                continue
            if not MimoV2QModel._checkpoint_has_tensor(model_local_path, weight_name):
                continue

            # MiMo V2.5 Base visual merger checkpoints include weights but omit
            # default biases; align the shell so offload-backed save skips them.
            module.register_parameter("bias", None)

    @staticmethod
    def _drop_parameter_if_checkpoint_omits_it(model, model_local_path: str, tensor_name: str) -> None:
        if MimoV2QModel._checkpoint_has_tensor(model_local_path, tensor_name):
            return

        module_path, _, leaf = tensor_name.rpartition(".")
        module = model
        for part in module_path.split("."):
            module = getattr(module, part, None)
            if module is None:
                return

        if not isinstance(module, nn.Module) or leaf not in module._parameters:
            return

        module.register_parameter(leaf, None)

    @staticmethod
    def _drop_checkpoint_omitted_audio_tensors(model, model_local_path: str) -> None:
        # Remote MiMo marks this input embedding as load-missing-ignored and
        # feeds the local transformer via inputs_embeds, so no trained weight exists.
        MimoV2QModel._drop_parameter_if_checkpoint_omits_it(
            model,
            model_local_path,
            "audio_encoder.input_local_transformer.embed_tokens.weight",
        )

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)
        modeling = import_module(type(model).__module__)
        for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
            function = getattr(modeling, name, None)
            if callable(function):
                setattr(modeling, name, _compatible_mask(function))
        if is_mimo_mixed_source(model.config):
            model.config.mimo_mtp_source_quantization_config = copy.deepcopy(
                model.config.quantization_config
            )
        self._drop_visual_merger_biases_if_checkpoint_omits_them(model, self.model_local_path)
        self._drop_checkpoint_omitted_audio_tensors(model, self.model_local_path)
        return model

    def pre_quantize_generate_hook_start(self):
        model = self.model.model
        rotary_emb_cls = type(model.rotary_emb)
        assert "MiMoV2RotaryEmbedding" in rotary_emb_cls.__name__
        config = model.rotary_emb.config
        # MiMoV2RotaryEmbedding cannot be correctly reconstructed via `_build_nonpersistent_buffer_template()`.
        # Since it takes three arguments, `_build_nonpersistent_buffer_template()` is unable to infer the `is_swa` parameter.
        # Therefore, MiMoV2RotaryEmbedding is manually reconstructed here.
        model.rotary_emb = rotary_emb_cls(config=config, is_swa=False, device=CPU)
        model.swa_rotary_emb = rotary_emb_cls(config=config, is_swa=True, device=CPU)

    def run_input_capture(self, example, use_cache: bool, data_device):
        """Keep the raw attention mask so each attention type can rebuild its own replay mask."""

        previous_mask = self.__dict__.get(_CAPTURE_ATTENTION_MASK_ATTR)
        setattr(self, _CAPTURE_ATTENTION_MASK_ATTR, example.get("attention_mask"))
        try:
            return super().run_input_capture(example, use_cache=use_cache, data_device=data_device)
        finally:
            setattr(self, _CAPTURE_ATTENTION_MASK_ATTR, previous_mask)

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Cache the raw mask and both rotary variants before the base modules are offloaded."""

        captured = dict(super().capture_first_layer_input_kwargs(args, kwargs, batch_device, layer_input_kwargs))
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            hidden_states = args[0]
        position_ids = kwargs["position_ids"]
        model = self.model.model
        captured[_REPLAY_INPUTS_KEY] = nested_move_to({
            "attention_mask": self.__dict__.get(_CAPTURE_ATTENTION_MASK_ATTR),
            "position_embeddings": {
                "full_attention": model.rotary_emb(hidden_states, position_ids),
                "sliding_window_attention": model.swa_rotary_emb(hidden_states, position_ids),
            },
        }, device=batch_device)
        return captured

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Select the mask and rotary inputs for each full or sliding attention layer."""

        prepared = dict(super().prepare_layer_replay_kwargs(layer, layer_input, additional_inputs, target_device))
        replay = prepared.pop(_REPLAY_INPUTS_KEY, None)
        attention_type = getattr(layer, "attention_type", None)
        if attention_type is None or not layer_input:
            return prepared
        if replay is None:
            raise ValueError("MiMo layer replay is missing captured attention and rotary inputs")
        if attention_type not in ("full_attention", "sliding_window_attention"):
            raise ValueError(f"Unsupported MiMo replay attention type: {attention_type}")

        raw_mask = replay["attention_mask"]
        if isinstance(raw_mask, dict):
            attention_mask = raw_mask[attention_type]
        else:
            modeling = import_module(type(self.model).__module__)
            mask_builder = (
                modeling.create_sliding_window_causal_mask
                if attention_type == "sliding_window_attention"
                else modeling.create_causal_mask
            )
            attention_mask = mask_builder(
                config=self.model.config,
                input_embeds=layer_input[0],
                attention_mask=nested_move_to(raw_mask, device=target_device),
                cache_position=prepared.get("cache_position"),
                past_key_values=None,
                position_ids=prepared.get("position_ids"),
            )
        prepared["attention_mask"] = nested_move_to(attention_mask, device=target_device)
        prepared["position_embeddings"] = nested_move_to(
            replay["position_embeddings"][attention_type], device=target_device,
        )
        return prepared
