# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from typing import List

import torch

from ..base import BaseQModel


class ChatGLMQModel(BaseQModel):
    require_trust_remote_code = True
    require_monkeypatch = True

    pre_lm_head_norm_module = "transformer.encoder.final_layernorm"

    module_tree = [
        "transformer",
        "encoder",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attention": ("query_key_value:0", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("dense_h_to_4h:0", "dense_4h_to_h:1"),
        }
    ]

    lm_head = "transformer.output_layer"

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.tensor]]) -> List[List[torch.tensor]]:
        if not self.config.post_layer_norm:
            return inputs

        return super().lm_head_pre_quantize_generate_hook(inputs)


    # fix: chatglm-3 and glm-4 have broken transformer (latest) compat due to missing/wrong type-hints
    def monkey_patch(self):
        from typing import Optional

        import torch

        def forward(
                self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
                use_cache: Optional[bool] = True,
                output_hidden_states: Optional[bool] = False,
        ):
            if not kv_caches:
                kv_caches = [None for _ in range(self.num_layers)]
            presents = () if use_cache else None
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    use_cache = False

            all_self_attentions = None
            all_hidden_states = () if output_hidden_states else None
            for index in range(self.num_layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer = self._get_layer(index)
                if self.gradient_checkpointing and self.training:
                    layer_ret = torch.utils.checkpoint.checkpoint(
                        layer,
                        hidden_states,
                        attention_mask=attention_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        kv_caches=kv_caches[index],
                        use_cache=use_cache,
                        use_reentrant=False
                    )
                else:
                    layer_ret = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        kv_cache=kv_caches[index],
                        use_cache=use_cache
                    )
                hidden_states, kv_cache = layer_ret
                if use_cache:
                    # token by token decoding, use tuple format
                    if kv_caches[0] is not None:
                        presents = presents + (kv_cache,)
                    # prefilling in decoding, use tensor format to save cuda memory
                    else:
                        if len(presents) == 0:
                            presents = kv_cache
                        else:
                            presents = torch.cat((presents, kv_cache.to(presents.device)), dim=0)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Final layer norm.
            if self.post_layer_norm:
                hidden_states = self.final_layernorm(hidden_states)

            return hidden_states, presents, all_hidden_states, all_self_attentions

        if not self.load_quantized_model:
            chatglm_transformer = type(self.model.transformer.encoder)
            chatglm_transformer.forward = forward
