# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader


class LFM2QModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )
    shared_input_verified_model_types = frozenset({"lfm2"})

    fusable = FusableConfig("lfm2").merge(LlamaFusableConfig())

    pre_lm_head_norm_module = "model.embedding_norm"

    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "operator_norm": ("operator_norm:!",),
            # `out_proj` consumes the convolution output produced from
            # `in_proj`, so it must not share the producer's subset.
            "conv": ("in_proj:0", "out_proj:1"),
            "self_attn": (
                "q_proj:0:in=x",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "out_proj:1",
            ),
            "ffn_norm": ("ffn_norm:!",),
            "feed_forward": ("w1:0:in=x", "w3:0:in=x", "w2:1"),
        },
    ]
