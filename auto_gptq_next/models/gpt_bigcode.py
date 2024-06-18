from ._base import BaseGPTQForCausalLM


class GPTBigCodeGPTQ(BaseGPTQForCausalLM):
    non_layer_modules = ["transformer.wpe", "transformer.wte", "transformer.ln_f"]

    layers_node = "transformer.h"
    layer_type = "GPTBigCodeBlock"
    layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"],
    ]
