# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os

from safetensors import safe_open
from torch import nn

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ...utils.torch import CPU
from ..base import BaseQModel


class MimoV2QModel(BaseQModel):
    # MiMo V2 uses repository-defined configuration/modeling classes.
    require_trust_remote_code = True

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

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
