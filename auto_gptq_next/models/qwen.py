from ._base import BaseGPTQForCausalLM


class QwenGPTQ(BaseGPTQForCausalLM):
    layer_type = "QWenBlock"
    layers_node = "transformer.h"
    non_layer_modules = [
        "transformer.wte",
        "transformer.wpe",
        "transformer.ln_f",
        "transformer.visual",
    ]
    layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.w1", "mlp.w2"],
        ["mlp.c_proj"],
    ]
