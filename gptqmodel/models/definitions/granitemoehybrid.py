# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from inspect import signature

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class GraniteMoeHybridQModel(BaseQModel):
    # Quantized Granite Mamba layers must use the eager path because the fast kernels
    # expect dense projections instead of GPTQ quantized linear modules.
    fusable = FusableConfig("granitemoehybrid").merge(LlamaFusableConfig(), profiles=("qkv",))

    pre_lm_head_norm_module = "model.norm"
    require_monkeypatch = True

    layer_modules_strict = False
    dynamic_expert_index = "num_local_experts"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "mamba": ("in_proj:0", "out_proj:1"),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "block_sparse_moe:moe:?": {
                # Defuser expands each packed parallel projection into
                # `<projection>.<expert>.gate_proj/up_proj/down_proj` modules.
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
            "shared_mlp": ("input_linear:0", "output_linear:1"),
        },
    ]

    def monkey_patch(self):
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear

        mamba_layer_cls = type(self.model.model.layers[0].mamba)
        if getattr(mamba_layer_cls.forward, "_gptqmodel_granite_patch", False):
            return
        original_forward = mamba_layer_cls.forward
        # Older Transformers releases exposed a separate eager `torch_forward`
        # next to the fast `forward`. Newer releases folded the eager
        # implementation into `forward` and removed the symbol, so probe the
        # installed class instead of assuming either layout.
        torch_forward = getattr(mamba_layer_cls, "torch_forward", original_forward)
        torch_forward_params = frozenset(signature(torch_forward).parameters)

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
                # The eager forward dropped cache_position in newer Transformers
                # releases, so only pass kwargs the installed callable explicitly
                # supports.
                torch_forward_kwargs = {}
                if "cache_params" in torch_forward_params:
                    torch_forward_kwargs["cache_params"] = cache_params
                if "cache_position" in torch_forward_params:
                    torch_forward_kwargs["cache_position"] = cache_position
                if "attention_mask" in torch_forward_params:
                    torch_forward_kwargs["attention_mask"] = attention_mask
                return torch_forward(layer_self, hidden_states, **torch_forward_kwargs)

            return original_forward(
                layer_self,
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
                seq_idx=seq_idx,
                **kwargs,
            )

        granitemoehybrid_mamba_forward._gptqmodel_granite_patch = True
        mamba_layer_cls.forward = granitemoehybrid_mamba_forward
