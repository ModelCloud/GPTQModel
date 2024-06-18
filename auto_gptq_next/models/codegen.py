from ._base import BaseGPTQForCausalLM


class CodeGenGPTQ(BaseGPTQForCausalLM):
    non_layer_modules = ["transformer.wte", "transformer.ln_f"]

    layers_node = "transformer.h"
    layer_type = "CodeGenBlock"
    layer_modules = [
        ["attn.qkv_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"],
    ]
