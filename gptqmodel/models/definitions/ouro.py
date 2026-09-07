# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class OuroQModel(BaseQModel):
    """GPTQ definition for ByteDance's looped Ouro decoder."""

    require_trust_remote_code = True

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    # Ouro's sandwich RMSNorms are part of the forward path, but only the
    # linear projections are quantized. The recurrent loop reuses these
    # decoder layers, so the tree describes one shared layer topology.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0:in=x",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "input_layernorm_2": ("input_layernorm_2:!",),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": (
                "gate_proj:0:in=x",
                "up_proj:0:in=x",
                "down_proj:1",
            ),
            "post_attention_layernorm_2": ("post_attention_layernorm_2:!",),
        },
    ]

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        """Bridge Ouro's legacy rotary class to the Transformers 5.x loader API."""

        del load_quantized_model
        from functools import wraps
        from importlib import import_module

        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        rotary_cls = get_class_from_dynamic_module(
            "modeling_ouro.OuroRotaryEmbedding",
            model_local_path,
        )
        if not hasattr(rotary_cls, "compute_default_rope_parameters"):

            def compute_default_rope_parameters(self, config):
                del self
                return ROPE_INIT_FUNCTIONS["default"](config)

            rotary_cls.compute_default_rope_parameters = compute_default_rope_parameters

        remote_module = import_module(rotary_cls.__module__)
        for mask_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
            mask_fn = getattr(remote_module, mask_name, None)
            if not callable(mask_fn) or getattr(mask_fn, "_gptqmodel_ouro_mask_compat", False):
                continue

            @wraps(mask_fn)
            def mask_compat(*args, _mask_fn=mask_fn, **kwargs):
                if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                # Ouro's remote code also forwards cache_position, which was
                # removed from the Transformers 5.x mask helper signatures.
                kwargs.pop("cache_position", None)
                return _mask_fn(*args, **kwargs)

            mask_compat._gptqmodel_ouro_mask_compat = True
            setattr(remote_module, mask_name, mask_compat)


__all__ = ["OuroQModel"]
