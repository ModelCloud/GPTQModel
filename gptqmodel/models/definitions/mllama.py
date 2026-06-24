# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoModelForPreTraining

from ...utils.model import get_module, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


# TODO FIXME: we currently do not support quantizing cross attention layer (pixel_values)
class MLlamaQModel(BaseQModel):
    # AutoModelForPreTraining return a correct MLlamaForConditionalGeneration for mllama.
    loader = AutoModelForPreTraining

    pre_lm_head_norm_module = "model.language_model.norm"
    # Current Transformers shells use `model.language_model.*`, while the
    # released Mllama checkpoints store those tensors under `language_model.model.*`.
    HF_CONVERSION_MAP_REVERSED = (
        SimpleNamespace(
            source_patterns=[r"model\.language_model"],
            target_patterns=[r"^language_model.model"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"lm_head"],
            target_patterns=[r"^language_model.lm_head"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.vision_model"],
            target_patterns=[r"^vision_model"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.multi_modal_projector"],
            target_patterns=[r"^multi_modal_projector"],
            operations=[],
        ),
    )

    module_tree = [
        [
            "model",
            "language_model",
            "layers",
            "#",
            {
                "input_layernorm": ("input_layernorm:!",),
                "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
                "post_attention_layernorm": ("post_attention_layernorm:!",),
                "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        ],
        [
            "language_model",
            "model",
            "layers",
            "#",
            {
                "input_layernorm": ("input_layernorm:!",),
                "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
                "post_attention_layernorm": ("post_attention_layernorm:!",),
                "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        ],
    ]

<<<<<<< HEAD

class MLlamaTextQModel(MLlamaQModel):
    loader = AutoModelForCausalLM

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]


__all__ = ["MLlamaQModel", "MLlamaTextQModel"]
=======
    @classmethod
    def _resolve_language_model(cls, model):
        for prefix in ("model.language_model", "language_model.model"):
            language_model = get_module(model, prefix)
            if language_model is not None:
                return language_model

        raise AttributeError("Unable to resolve an Mllama language model layout.")

    def _core_language_model(self):
        return self._resolve_language_model(self.model)

    def _materialize_language_module(self, language_model, attr_name: str):
        module = getattr(language_model, attr_name, None)
        if module is None:
            return

        if "_turtle_lock" not in self.__dict__ and "shell_module_materialize" not in self.__dict__:
            setattr(language_model, attr_name, move_to(module, device=self.quantize_config.device))
            return

        setattr(
            language_model,
            attr_name,
            self.shell_module_materialize(module, self.quantize_config.device),
        )

    def pre_quantize_generate_hook_start(self):
        language_model = self._core_language_model()
        self._materialize_language_module(language_model, "embed_tokens")
        self._materialize_language_module(language_model, "rotary_emb")

    def pre_quantize_generate_hook_end(self):
        language_model = self._core_language_model()

        for attr_name in ("embed_tokens", "rotary_emb"):
            module = getattr(language_model, attr_name, None)
            if module is None:
                continue

            if self.quantize_config.offload_to_disk:
                offload_to_disk(
                    model=language_model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            else:
                setattr(language_model, attr_name, move_to(module, device=CPU))

    @staticmethod
    def _prepare_first_layer_attention_mask(attention_mask):
        if attention_mask is None or not torch.is_tensor(attention_mask):
            return attention_mask

        if attention_mask.ndim <= 2 and bool(attention_mask.to(dtype=torch.bool).all().item()):
            return None

        return attention_mask

    def run_input_capture(self, example, use_cache: bool, data_device):
        input_ids = example.get("input_ids")
        if input_ids is None:
            return super().run_input_capture(example, use_cache=use_cache, data_device=data_device)

        language_model = self._core_language_model()
        attention_mask = example.get("attention_mask")
        position_ids = example.get("position_ids")
        past_key_values = example.get("past_key_values")

        embedding_weight = getattr(language_model.embed_tokens, "weight", None)
        if torch.is_tensor(embedding_weight) and input_ids.device != embedding_weight.device:
            input_ids = input_ids.to(device=embedding_weight.device)

        # Input capture only needs the tensors entering the first decoder layer.
        # Calling it directly avoids the multimodal wrapper's mask construction
        # path, which can inspect meta tensors during lazy quantization.
        inputs_embeds = language_model.embed_tokens(input_ids)
        if getattr(inputs_embeds, "is_meta", False):
            raise RuntimeError("Mllama input capture produced meta inputs_embeds after materializing embed_tokens.")

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        elif position_ids.device != inputs_embeds.device:
            position_ids = position_ids.to(device=inputs_embeds.device)

        position_embeddings = language_model.rotary_emb(inputs_embeds, position_ids=position_ids)
        first_layer_attention_mask = self._prepare_first_layer_attention_mask(attention_mask)

        return language_model.layers[0](
            inputs_embeds,
            attention_mask=first_layer_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
>>>>>>> main
