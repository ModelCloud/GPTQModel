# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class GraniteMoeHybridQModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"
    require_monkeypatch = True

    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "mamba": ("in_proj:0", "out_proj:1"),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "shared_mlp": ("input_linear:0", "output_linear:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
        }
    ]

    def monkey_patch(self):
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear

        mamba_layer_cls = type(self.model.model.layers[0].mamba)
        original_forward = mamba_layer_cls.forward

        def granitemoehybrid_mamba_forward(
            layer_self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
            seq_idx=None,
            **kwargs,
        ):
            if isinstance(layer_self.in_proj, BaseQuantLinear) or isinstance(layer_self.out_proj, BaseQuantLinear):
                if seq_idx is not None:
                    raise NotImplementedError(
                        "`seq_idx` support requires fast path support. Please install `mamba_ssm` and `causal_conv1d`"
                    )
                dtype = hidden_states.dtype
                if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
                return layer_self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)

            return original_forward(
                layer_self,
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
                seq_idx=seq_idx,
                **kwargs,
            )

        mamba_layer_cls.forward = granitemoehybrid_mamba_forward
