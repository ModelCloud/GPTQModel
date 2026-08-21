# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from ...utils.calibration import batched
from ...utils.device import get_device
from ...utils.model import MODALITY, get_module_by_name_prefix, move_to, nested_move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


_RAW_ATTENTION_MASK = "__gptqmodel_cohere_compass_attention_mask"
_POSITION_IDS_BATCH_FIRST = "__gptqmodel_cohere_compass_position_ids_batch_first"
_DEEPSTACK_VISUAL_MASK = "__gptqmodel_cohere_compass_visual_pos_masks"
_DEEPSTACK_VISUAL_EMBEDS = "__gptqmodel_cohere_compass_deepstack_visual_embeds"
_MISSING = object()


class CohereCompassQModel(BaseQModel):
    """Cohere Compass VLM adapter used by North Micro Vision.

    The vision tower stays in its native dtype. Quantization targets the text
    decoder, whose attention and MLP consume the same normalized layer input.
    """

    loader = AutoModelForImageTextToText

    require_load_processor = True
    require_trust_remote_code = False
    require_pkgs = ["transformers>=5.16.0.dev0"]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    # Cohere Compass uses GQA, so o_proj does not match v_proj's output shape.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]
    awq_preserve_explicit_position_embeddings = True

    # Decoder forward order is:
    #   normalized = input_layernorm(hidden_states)
    #   hidden_states + self_attn(normalized) + mlp(normalized)
    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
        },
    ]

    def _language_model(self):
        core_model = getattr(self.model, "model", self.model)
        return getattr(core_model, "language_model", core_model)

    def _start_outer_input_capture(self) -> None:
        """Capture state owned by the outer text loop before first-layer early stop."""

        self._stop_outer_input_capture()
        language_model = self._language_model()

        def capture_outer_inputs(module, args, kwargs):
            del args
            setattr(module, _RAW_ATTENTION_MASK, kwargs.get("attention_mask"))
            position_ids = kwargs.get("position_ids")
            if torch.is_tensor(position_ids) and position_ids.ndim == 3:
                position_ids = position_ids.transpose(0, 1)
            setattr(module, _POSITION_IDS_BATCH_FIRST, position_ids)
            setattr(module, _DEEPSTACK_VISUAL_MASK, kwargs.get("visual_pos_masks"))
            setattr(module, _DEEPSTACK_VISUAL_EMBEDS, kwargs.get("deepstack_visual_embeds"))

        self._cohere_compass_input_capture_handle = language_model.register_forward_pre_hook(
            capture_outer_inputs,
            with_kwargs=True,
        )

    def _stop_outer_input_capture(self) -> None:
        handle = getattr(self, "_cohere_compass_input_capture_handle", None)
        if handle is not None:
            handle.remove()
            delattr(self, "_cohere_compass_input_capture_handle")

        language_model = self._language_model()
        for name in (
            _RAW_ATTENTION_MASK,
            _POSITION_IDS_BATCH_FIRST,
            _DEEPSTACK_VISUAL_MASK,
            _DEEPSTACK_VISUAL_EMBEDS,
        ):
            if hasattr(language_model, name):
                delattr(language_model, name)

    @staticmethod
    def _layer_type(layer, text_config):
        self_attn = getattr(layer, "self_attn", None)
        layer_index = getattr(self_attn, "layer_idx", None)
        layer_types = getattr(text_config, "layer_types", ())
        if layer_index is None or not 0 <= layer_index < len(layer_types):
            return None
        return layer_types[layer_index]

    @staticmethod
    def _position_ids_for_replay(position_ids_batch_first, hidden_states, target_device):
        batch_size = hidden_states.shape[0] if hidden_states.dim() >= 3 else 1
        seq_len = hidden_states.shape[1] if hidden_states.dim() >= 3 else hidden_states.shape[0]

        if position_ids_batch_first is None:
            position_ids = torch.arange(seq_len, device=target_device, dtype=torch.long)
            position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)
        else:
            position_ids = move_to(position_ids_batch_first, device=target_device)
            if position_ids.ndim == 3:
                position_ids = position_ids.transpose(0, 1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            return position_ids[0], position_ids[1:]
        return None, position_ids

    def _prepare_layer_forward_kwargs(
        self,
        layer,
        layer_input,
        module_kwargs: Dict,
        target_device,
    ) -> Dict:
        """Rebuild kwargs normally selected by Cohere's outer sliding/full decoder loop."""

        prepared_kwargs = dict(module_kwargs)
        raw_attention_mask = prepared_kwargs.pop(_RAW_ATTENTION_MASK, _MISSING)
        position_ids_batch_first = prepared_kwargs.pop(_POSITION_IDS_BATCH_FIRST, _MISSING)
        prepared_kwargs.pop(_DEEPSTACK_VISUAL_MASK, None)
        prepared_kwargs.pop(_DEEPSTACK_VISUAL_EMBEDS, None)

        text_config = getattr(self.model.config, "text_config", self.model.config)
        layer_type = self._layer_type(layer, text_config)
        if layer_type is None:
            return prepared_kwargs

        rope_parameters = getattr(text_config, "rope_parameters", {})
        if layer_type in rope_parameters and rope_parameters[layer_type] is None:
            prepared_kwargs["position_embeddings"] = None

        if not layer_input or not torch.is_tensor(layer_input[0]):
            return prepared_kwargs

        hidden_states = layer_input[0]
        target_device = torch.device(target_device)
        text_position_ids = None
        rotary_position_ids = None
        if position_ids_batch_first is not _MISSING:
            text_position_ids, rotary_position_ids = self._position_ids_for_replay(
                position_ids_batch_first,
                hidden_states,
                target_device,
            )

            if (
                layer_type in rope_parameters
                and rope_parameters[layer_type] is not None
                and rotary_position_ids is not None
            ):
                rotary, _ = get_module_by_name_prefix(self.model, [self.rotary_embedding])
                if rotary is not None:
                    rotary_device = get_device(rotary)
                    rotary_input = torch.empty(1, device=rotary_device, dtype=hidden_states.dtype)
                    rotary_position_ids = move_to(rotary_position_ids, device=rotary_device)
                    prepared_kwargs["position_embeddings"] = nested_move_to(
                        rotary(rotary_input, rotary_position_ids, layer_type),
                        device=target_device,
                    )

        if raw_attention_mask is not _MISSING:
            if isinstance(raw_attention_mask, dict):
                prepared_kwargs["attention_mask"] = nested_move_to(
                    raw_attention_mask[layer_type],
                    device=target_device,
                )
            else:
                raw_attention_mask = nested_move_to(raw_attention_mask, device=target_device)
                mask_builder = (
                    create_sliding_window_causal_mask
                    if layer_type == "sliding_attention"
                    else create_causal_mask
                )
                prepared_kwargs["attention_mask"] = mask_builder(
                    config=text_config,
                    inputs_embeds=hidden_states,
                    attention_mask=raw_attention_mask,
                    past_key_values=None,
                    position_ids=text_position_ids,
                )

        return prepared_kwargs

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Keep isolated decoder replay aligned with Cohere's outer text loop."""

        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )
        return self._prepare_layer_forward_kwargs(layer, layer_input, additional_inputs, target_device)

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Persist outer-loop masks, positions, and DeepStack inputs for isolated replay."""

        layer_input_kwargs = super().capture_first_layer_input_kwargs(
            args,
            kwargs,
            batch_device,
            layer_input_kwargs,
        )
        language_model = self._language_model()
        for name in (_RAW_ATTENTION_MASK, _POSITION_IDS_BATCH_FIRST):
            value = getattr(language_model, name, None)
            layer_input_kwargs[name] = nested_move_to(value, device=batch_device)

        visual_pos_masks = getattr(language_model, _DEEPSTACK_VISUAL_MASK, None)
        deepstack_visual_embeds = getattr(language_model, _DEEPSTACK_VISUAL_EMBEDS, None)
        if visual_pos_masks is not None and deepstack_visual_embeds is not None:
            layer_input_kwargs[_DEEPSTACK_VISUAL_MASK] = nested_move_to(
                visual_pos_masks,
                device=batch_device,
            )
            layer_input_kwargs[_DEEPSTACK_VISUAL_EMBEDS] = nested_move_to(
                deepstack_visual_embeds,
                device=batch_device,
            )

        return layer_input_kwargs

    def update_layer_replay_kwargs_from_output(self, layer, layer_output, layer_input_kwargs, target_device):
        """Apply the DeepStack residual that the outer text loop adds after early decoder layers."""

        del target_device
        visual_pos_masks = layer_input_kwargs.get(_DEEPSTACK_VISUAL_MASK)
        deepstack_visual_embeds = layer_input_kwargs.get(_DEEPSTACK_VISUAL_EMBEDS)
        layer_index = getattr(getattr(layer, "self_attn", None), "layer_idx", None)
        if (
            visual_pos_masks is None
            or deepstack_visual_embeds is None
            or layer_index is None
            or not 0 <= layer_index < len(deepstack_visual_embeds)
        ):
            return layer_input_kwargs

        primary_output = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        if not torch.is_tensor(primary_output):
            return layer_input_kwargs

        updated_output = self._language_model()._deepstack_process(
            primary_output,
            visual_pos_masks,
            deepstack_visual_embeds[layer_index],
        )
        primary_output.copy_(updated_output)
        return layer_input_kwargs

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Apply the same outer-loop kwargs to AWQ attention replays."""

        fallback_input = next(
            (value for value in input_feat.values() if torch.is_tensor(value) and value.numel() > 0),
            None,
        )
        layer_input = [fallback_input] if fallback_input is not None else []
        target_device = fallback_input.device if fallback_input is not None else get_device(module)
        prepared_kwargs = self._prepare_layer_forward_kwargs(
            module,
            layer_input,
            module_kwargs,
            target_device,
        )
        feature_kwargs = prepared_kwargs.get("_awq_feature_kwargs")
        if isinstance(feature_kwargs, dict):
            prepared_feature_kwargs = {}
            for name, kwargs in feature_kwargs.items():
                if not isinstance(kwargs, dict):
                    prepared_feature_kwargs[name] = kwargs
                    continue

                feature_input = input_feat.get(name, fallback_input)
                feature_layer_input = [feature_input] if feature_input is not None else []
                feature_device = feature_input.device if feature_input is not None else target_device
                prepared_feature_kwargs[name] = self._prepare_layer_forward_kwargs(
                    module,
                    feature_layer_input,
                    kwargs,
                    feature_device,
                )
            prepared_kwargs["_awq_feature_kwargs"] = prepared_feature_kwargs
        return super().awq_get_modules_for_scaling(module, input_feat, prepared_kwargs)

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.norm, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(core_model.visual, self.quantize_config.device)
        self._start_outer_input_capture()

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self._stop_outer_input_capture()
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=language_model,
                module=language_model.embed_tokens,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.norm,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.rotary_emb,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=core_model,
                module=core_model.visual,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.norm = move_to(language_model.norm, device=CPU)
        language_model.rotary_emb = move_to(language_model.rotary_emb, device=CPU)
        core_model.visual = move_to(core_model.visual, device=CPU)

    def preprocess_dataset(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=False)

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        return processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(self.prepare_inputs_for_conversations(processor, batch))
        del processor
        return calib_data


__all__ = ["CohereCompassQModel"]
