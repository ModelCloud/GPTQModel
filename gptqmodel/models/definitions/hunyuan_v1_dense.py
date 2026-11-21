# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from .llama import LlamaQModel


class HunYuanDenseV1QModel(LlamaQModel):
    """
    Hunyuan Dense V1 follows a Llama-style decoder layout with per-head Q/K
    RMSNorm modules inside attention. Those norms are metadata/base modules for
    quantization and should not be replaced by quantized linear kernels.
    """
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )
    fusable = FusableConfig("hunyuan_v1_dense").merge(LlamaFusableConfig())

    # q/k/v share the post-norm hidden state and gate/up share the MLP input.
    # Verified against a real (tiny, CPU) forward in
    # tests/module_tree/test_shared_input_cpu_forward.py.
    shared_input_verified_model_types = frozenset({"hunyuan_v1_dense"})

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "query_layernorm:!",
                "q_proj:0:in=x",
                "key_layernorm:!",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
        },
    ]
