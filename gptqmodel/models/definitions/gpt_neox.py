# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class GPTNeoXQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )
    pre_lm_head_norm_module = "gpt_neox.final_layer_norm"
    # Transformers 5.22 renamed the runtime module to ``lm_head``; the
    # checkpoint still stores ``embed_out.weight`` and is bridged by the
    # class-keyed conversion mapping handled in LazyTurtle.
    lm_head = "lm_head"

    module_tree= [
        "gpt_neox",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "attention": ("query_key_value:0", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # GELU between the two projections is not positively homogeneous.
            "mlp": ("dense_h_to_4h:0:nosrc", "dense_4h_to_h:1"),
        }
    ]
