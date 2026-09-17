# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPT-QModel support for the ZDTaichu-5.0 vision-language model."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from inspect import signature
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, ProcessorMixin

from ...utils.attn_mask import normalize_seq_mask
from ...utils.calibration import batched_conversations
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from .qwen3_5_text import Qwen3_5TextQModel

_PADDING_MASK_KEY = "_zdtaichu5_padding_mask"
_CAPTURE_PADDING_MASK_ATTR = "_zdtaichu5_capture_padding_mask"


class ZDTaichu5QModel(Qwen3_5TextQModel):
    """Quantization definition for ``TaichuAI/ZDTaichu5.0-9B``.

    ZDTaichu mounts a Qwen3.5 causal LM below ``language_model`` and keeps a
    RADIO vision tower plus ``mlp1`` projector at the wrapper root.  The
    Qwen3.5 hybrid decoder tree is reused verbatim so both linear-attention
    and full-attention blocks retain their auxiliary norms and projections.
    """

    loader = AutoModelForCausalLM

    lm_head = "language_model.lm_head"
    pre_lm_head_norm_module = "language_model.model.norm"
    rotary_embedding = "language_model.model.rotary_emb"

    require_trust_remote_code = True
    require_load_processor = True
    layer_modules_strict = False
    support_batch_quantize = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT, MODALITY.VIDEO_TO_TEXT]

    # The wrapper's decoder is the text-only Qwen3.5 tree under one additional
    # root.  Copy it so quantization-method overrides cannot mutate the parent.
    module_tree = deepcopy(Qwen3_5TextQModel.module_tree)
    module_tree.insert(0, "language_model")

    @classmethod
    def extract_layers_node(cls):
        return ["language_model.model.layers"]

    @classmethod
    def get_base_modules(cls, model):
        base_modules = super().get_base_modules(model)
        seen = set(base_modules)
        for name, _ in model.named_children():
            if name == "language_model":
                continue
            if name not in seen:
                base_modules.append(name)
                seen.add(name)
        return base_modules

    def _materialize_module(self, parent, name: str, module_path: str) -> None:
        module = getattr(parent, name)
        setattr(
            parent,
            name,
            self.shell_module_materialize(
                module,
                self.quantize_config.device,
                module_path=module_path,
            ),
        )

    def pre_quantize_generate_hook_start(self):
        language_model = self.model.language_model.model
        self._materialize_module(
            language_model, "embed_tokens", "language_model.model.embed_tokens"
        )
        self._materialize_module(
            language_model, "rotary_emb", "language_model.model.rotary_emb"
        )
        self._materialize_module(self.model, "vision_model", "vision_model")
        self._materialize_module(self.model, "mlp1", "mlp1")

    def pre_quantize_generate_hook_end(self):
        language_model = self.model.language_model.model
        modules = (
            (language_model, "embed_tokens"),
            (language_model, "rotary_emb"),
            (self.model, "vision_model"),
            (self.model, "mlp1"),
        )

        if self.quantize_config.offload_to_disk:
            for parent, name in modules:
                offload_to_disk(
                    model=parent,
                    module=getattr(parent, name),
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        for parent, name in modules:
            setattr(parent, name, move_to(getattr(parent, name), device=CPU))

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(
            self.model_local_path,
            trust_remote_code=True,
            use_fast=False,
        )

    def run_input_capture(self, example, use_cache: bool, data_device):
        """Keep the original 2-D padding mask for hybrid-layer replay."""

        previous_mask = self.__dict__.get(_CAPTURE_PADDING_MASK_ATTR)
        attention_mask = example.get("attention_mask")
        setattr(
            self,
            _CAPTURE_PADDING_MASK_ATTR,
            attention_mask if torch.is_tensor(attention_mask) and attention_mask.ndim == 2 else None,
        )
        try:
            return super().run_input_capture(example, use_cache=use_cache, data_device=data_device)
        finally:
            setattr(self, _CAPTURE_PADDING_MASK_ATTR, previous_mask)

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Retain padding metadata hidden by the first Qwen3.5 mask mapping."""

        layer_input_kwargs = super().capture_first_layer_input_kwargs(
            args,
            kwargs,
            batch_device,
            layer_input_kwargs,
        )
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        seq_len = (
            hidden_states.shape[1]
            if torch.is_tensor(hidden_states) and hidden_states.ndim >= 2
            else None
        )
        padding_mask = self.__dict__.get(_CAPTURE_PADDING_MASK_ATTR)
        captured_mask = kwargs.get("attention_mask")
        if not torch.is_tensor(padding_mask) and torch.is_tensor(captured_mask):
            padding_mask = normalize_seq_mask(captured_mask, seq_len=seq_len)
        if torch.is_tensor(padding_mask):
            layer_input_kwargs[_PADDING_MASK_KEY] = move_to(
                padding_mask,
                device=batch_device,
            )
        return layer_input_kwargs

    @staticmethod
    def _recurrent_attention_mask(mask_kwargs):
        try:
            from transformers.masking_utils import create_recurrent_attention_mask
        except ImportError:
            # Transformers versions that first exposed Qwen3.5 may not yet
            # expose the generic builder. Linear attention consumes the raw
            # [batch, sequence] keep-mask in that compatibility path. An
            # all-ones mask is equivalent to ``None`` for the decoder.
            mask = mask_kwargs["attention_mask"]
            if torch.is_tensor(mask) and torch.all(mask):
                return None
            return mask
        return create_recurrent_attention_mask(**mask_kwargs)

    @staticmethod
    def _causal_attention_mask(mask_kwargs, cache_position=None):
        from transformers.masking_utils import create_causal_mask

        kwargs = dict(mask_kwargs)
        parameters = signature(create_causal_mask).parameters
        if "cache_position" in parameters:
            if cache_position is None:
                inputs_embeds = kwargs["inputs_embeds"]
                cache_position = torch.arange(
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
            kwargs["cache_position"] = cache_position
        return create_causal_mask(**kwargs)

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Rebuild the mask for each Qwen3.5 linear/full attention block."""

        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )
        padding_mask = additional_inputs.pop(_PADDING_MASK_KEY, None)
        if not layer_input or not torch.is_tensor(layer_input[0]):
            return additional_inputs

        hidden_states = layer_input[0]
        if not torch.is_tensor(padding_mask):
            captured_mask = additional_inputs.get("attention_mask")
            if torch.is_tensor(captured_mask):
                padding_mask = normalize_seq_mask(
                    captured_mask,
                    seq_len=hidden_states.shape[1],
                )

        language_model = self.model.language_model.model
        layer_config = getattr(language_model, "config", None)
        if layer_config is None:
            return additional_inputs

        position_ids = additional_inputs.get("position_ids")
        if not torch.is_tensor(position_ids) or position_ids.ndim != 2:
            position_ids = None
        mask_kwargs = {
            "config": layer_config,
            "inputs_embeds": hidden_states,
            "attention_mask": padding_mask,
            "past_key_values": additional_inputs.get("past_key_values"),
            "position_ids": position_ids,
        }
        block_type = getattr(layer, "block_type", None) or getattr(layer, "layer_type", None)
        linear_attn = getattr(layer, "linear_attn", None)
        if block_type == "linear_attention" or linear_attn is not None:
            update_mask = getattr(language_model, "_update_linear_attn_mask", None)
            if callable(update_mask):
                cache_position = additional_inputs.get("cache_position")
                if cache_position is None:
                    cache_position = torch.arange(
                        hidden_states.shape[1],
                        device=hidden_states.device,
                    )
                additional_inputs["attention_mask"] = update_mask(
                    padding_mask,
                    cache_position,
                )
                return additional_inputs
            additional_inputs["attention_mask"] = self._recurrent_attention_mask(mask_kwargs)
        elif block_type == "full_attention" or getattr(layer, "self_attn", None) is not None:
            additional_inputs["attention_mask"] = self._causal_attention_mask(
                mask_kwargs,
                cache_position=additional_inputs.get("cache_position"),
            )
        return additional_inputs

    @staticmethod
    def _is_conversation_sample(sample: Any) -> bool:
        if isinstance(sample, Mapping):
            return "messages" in sample
        if not isinstance(sample, (list, tuple)) or not sample:
            return False
        first = sample[0]
        if isinstance(first, Mapping) and "messages" in first:
            return True
        if isinstance(first, Mapping) and "role" in first and "content" in first:
            return True
        return isinstance(first, (list, tuple)) and ZDTaichu5QModel._is_conversation_sample(first)

    @classmethod
    def _conversation_dataset(cls, calibration_dataset):
        if isinstance(calibration_dataset, Mapping):
            return [calibration_dataset]
        if isinstance(calibration_dataset, (list, tuple)):
            first = calibration_dataset[0] if calibration_dataset else None
            if isinstance(first, Mapping) and "role" in first:
                return [calibration_dataset]
            return calibration_dataset
        return list(calibration_dataset)

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        if isinstance(calibration_dataset, Mapping):
            samples = [calibration_dataset]
        elif isinstance(calibration_dataset, (str, bytes, bytearray)):
            return super().prepare_dataset(calibration_dataset, batch_size=batch_size, **kwargs)
        else:
            # Inspecting once also allows generators and HF iterable datasets to
            # use the processor path when they contain structured conversations.
            samples = list(calibration_dataset)

        conversation_input = self._is_conversation_sample(samples) or any(
            self._is_conversation_sample(sample) for sample in samples
        )
        if not conversation_input:
            return super().prepare_dataset(samples, batch_size=batch_size, **kwargs)

        processor = self.load_processor()
        calibration_data = []
        for batch in batched_conversations(
            self._conversation_dataset(samples),
            1,
        ):
            conversation = batch[0]
            if not isinstance(conversation, (list, tuple)):
                raise ValueError("ZDTaichu5 calibration conversation must be a message list.")
            calibration_data.append(processor.from_messages(conversation, return_tensors="pt"))
        del processor
        return calibration_data

    def move_input_capture_example(self, example: Dict[str, Any], data_device):
        example = super().move_input_capture_example(example, data_device)
        vision_model = getattr(self.model, "vision_model", None)
        first_parameter = next(vision_model.parameters(), None) if vision_model is not None else None
        vision_dtype = getattr(vision_model, "dtype", None)
        if not isinstance(vision_dtype, torch.dtype) and first_parameter is not None:
            vision_dtype = first_parameter.dtype
        if first_parameter is not None:
            vision_device = first_parameter.device
        else:
            vision_device = data_device

        if isinstance(vision_dtype, torch.dtype):
            for key in ("pixel_values", "pixel_values_videos"):
                value = example.get(key)
                if torch.is_tensor(value) and value.is_floating_point():
                    example[key] = value.to(device=vision_device, dtype=vision_dtype)
        return example


__all__ = ["ZDTaichu5QModel"]
