# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any

import torch
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)

from ...utils.device import get_device
from ...utils.model import get_module_by_name_prefix, move_to, nested_move_to
from ..base import BaseQModel


_RAW_ATTENTION_MASK = "__gptqmodel_lfm3_attention_mask"
_POSITION_IDS = "__gptqmodel_lfm3_position_ids"


class LFM3QModel(BaseQModel):
    """LiquidAI LFM3 hybrid short-conv / gated-attention decoder.

    LFM3 alternates `conv` (depthwise short convolution), `full_attention`, and
    `sliding_attention` decoder layers.  Only the linear projections are
    quantized: the depthwise convolution kernel, every RMSNorm weight, and the
    tiny per-head sigmoid gate projection keep their source dtype.

    Every decoder layer receives `position_embeddings` for its own `layer_type`
    plus a mask built for that `layer_type` (conv/full/sliding).  The outer
    `Lfm3Model.forward` builds both once per batch, so isolated decoder replay
    (`prepare_layer_replay_kwargs`) has to rebuild them from the captured
    `position_ids` and raw `attention_mask`.

    The checkpoint is a remote-code (`auto_map`) Transformers model, therefore
    the loader requires `trust_remote_code=True`.
    """

    pre_lm_head_norm_module = "model.embedding_norm"
    rotary_embedding = "model.rotary_emb"

    require_trust_remote_code = True

    # Conv layers have no `self_attn`, and attention layers have no `conv`;
    # the per-layer module sets therefore differ.
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "operator_norm": ("operator_norm:!",),
            "post_operator_norm": ("post_operator_norm:!",),
            "conv": ("in_proj:0", "out_proj:1"),
            "self_attn": (
                "q_proj:0",
                "q_norm:0:!",
                "k_proj:0",
                "k_norm:0:!",
                "v_proj:0",
                # `gate_proj` is a per-head sigmoid gate (`num_attention_heads`
                # outputs); keep it in its source dtype.
                "gate_proj:!",
                "o_proj:1",
            ),
            "ffn_norm": ("ffn_norm:!",),
            "post_ffn_norm": ("post_ffn_norm:!",),
            "feed_forward": ("w1:0", "w3:0", "w2:1"),
        },
    ]

    def _language_model(self):
        return self.model.model

    def _text_config(self) -> Any:
        config = getattr(self._language_model(), "config", None)
        return getattr(config, "get_text_config", lambda: config)() if config is not None else None

    @staticmethod
    def _layer_type(layer, text_config) -> str | None:
        # Conv layers own `conv.layer_idx`; attention layers own
        # `self_attn.layer_idx`. A decoder layer has exactly one of the two.
        layer_index = None
        for name in ("self_attn", "conv"):
            layer_index = getattr(getattr(layer, name, None), "layer_idx", None)
            if layer_index is not None:
                break
        layer_types = getattr(text_config, "layer_types", ())
        if layer_index is None or not 0 <= layer_index < len(layer_types):
            return None
        return layer_types[layer_index]

    def _start_outer_input_capture(self) -> None:
        """Capture the outer text-loop mask/positions before first-layer early stop."""

        self._stop_outer_input_capture()
        language_model = self._language_model()

        def capture_outer_inputs(module, args, kwargs):
            del args
            setattr(module, _RAW_ATTENTION_MASK, kwargs.get("attention_mask"))
            setattr(module, _POSITION_IDS, kwargs.get("position_ids"))

        self._lfm3_input_capture_handle = language_model.register_forward_pre_hook(
            capture_outer_inputs,
            with_kwargs=True,
        )

    def _stop_outer_input_capture(self) -> None:
        handle = getattr(self, "_lfm3_input_capture_handle", None)
        if handle is not None:
            handle.remove()
            delattr(self, "_lfm3_input_capture_handle")

        language_model = self._language_model()
        for name in (_RAW_ATTENTION_MASK, _POSITION_IDS):
            if hasattr(language_model, name):
                delattr(language_model, name)

    def _materialize_rotary(self):
        rotary, _ = get_module_by_name_prefix(self.model, [self.rotary_embedding])
        if rotary is None:
            return None
        if get_device(rotary).type == "meta":
            return self.shell_module_materialize(rotary, self.quantize_config.device)
        return rotary

    def _position_embeddings_for_replay(self, layer_type, hidden_states, position_ids, target_device):
        rotary = self._materialize_rotary()
        if rotary is None or hidden_states is None:
            return None

        rotary_device = get_device(rotary)
        rotary_input = torch.empty(1, device=rotary_device, dtype=hidden_states.dtype)
        position_ids = move_to(position_ids, device=rotary_device)
        return nested_move_to(
            rotary(rotary_input, position_ids=position_ids, layer_type=layer_type),
            device=target_device,
        )

    def _attention_mask_for_replay(self, layer_type, text_config, hidden_states, raw_mask, position_ids, target_device):
        if raw_mask is None:
            return None
        if isinstance(raw_mask, dict):
            return nested_move_to(raw_mask.get(layer_type), device=target_device)

        raw_mask = nested_move_to(raw_mask, device=target_device)
        layer_types = getattr(text_config, "layer_types", ())
        if layer_type == "conv":
            return create_recurrent_attention_mask(
                config=text_config,
                inputs_embeds=hidden_states,
                attention_mask=raw_mask,
                past_key_values=None,
                position_ids=position_ids,
            )

        # Mirror `create_masks_by_layer_type` in the checkpoint's modeling code:
        # it names the first layer of each `layer_type` so hybrid caches size the
        # mask from a layer of the same type instead of the first non-sliding one.
        builder = (
            create_sliding_window_causal_mask
            if layer_type == "sliding_attention"
            else create_causal_mask
        )
        mask_kwargs = {
            "config": text_config,
            "inputs_embeds": hidden_states,
            "attention_mask": raw_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        if layer_type in layer_types:
            mask_kwargs["layer_idx"] = layer_types.index(layer_type)
        return builder(**mask_kwargs)

    def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
        """Rebuild the per-`layer_type` rotary/mask state that the outer loop builds."""

        additional_inputs = super().prepare_layer_replay_kwargs(
            layer,
            layer_input,
            additional_inputs,
            target_device,
        )

        text_config = self._text_config()
        layer_type = self._layer_type(layer, text_config)
        if layer_type is None:
            return additional_inputs

        raw_mask = additional_inputs.pop(_RAW_ATTENTION_MASK, None)
        position_ids = additional_inputs.pop(_POSITION_IDS, None)
        if raw_mask is None:
            # The capture hook only fires when the outer text loop ran this
            # batch; fall back to the 2D padding mask the looper recorded.
            raw_mask = additional_inputs.get("attention_mask")

        if not layer_input or not torch.is_tensor(layer_input[0]):
            return additional_inputs

        hidden_states = layer_input[0]
        if position_ids is None:
            seq_len = hidden_states.shape[1] if hidden_states.dim() >= 3 else hidden_states.shape[0]
            batch_size = hidden_states.shape[0] if hidden_states.dim() >= 3 else 1
            position_ids = (
                torch.arange(seq_len, device=target_device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        if layer_type != "conv":
            position_embeddings = self._position_embeddings_for_replay(
                layer_type,
                hidden_states,
                position_ids,
                target_device,
            )
            if position_embeddings is not None:
                additional_inputs["position_embeddings"] = position_embeddings
            additional_inputs["position_ids"] = position_ids

        attention_mask = self._attention_mask_for_replay(
            layer_type,
            text_config,
            hidden_states,
            raw_mask,
            position_ids,
            target_device,
        )
        if attention_mask is not None:
            additional_inputs["attention_mask"] = attention_mask

        return additional_inputs

    def pre_quantize_generate_hook_start(self):
        language_model = self._language_model()
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(language_model.embedding_norm, self.quantize_config.device)
        self._start_outer_input_capture()

    def pre_quantize_generate_hook_end(self):
        self._stop_outer_input_capture()

    def capture_first_layer_input_kwargs(self, args, kwargs, batch_device, layer_input_kwargs):
        """Persist the outer-loop mask/positions for isolated replay."""

        layer_input_kwargs = super().capture_first_layer_input_kwargs(
            args,
            kwargs,
            batch_device,
            layer_input_kwargs,
        )
        language_model = self._language_model()
        for name in (_RAW_ATTENTION_MASK, _POSITION_IDS):
            value = getattr(language_model, name, None)
            layer_input_kwargs[name] = nested_move_to(value, device=batch_device)
        return layer_input_kwargs


__all__ = ["LFM3QModel"]
