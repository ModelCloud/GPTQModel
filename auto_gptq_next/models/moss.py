from ._base import BaseGPTQForCausalLM


class MOSSGPTQ(BaseGPTQForCausalLM):
    non_layer_modules = ["transformer.wte", "transformer.ln_f"]

    layers_node = "transformer.h"
    layer_type = "MossBlock"
    layer_modules = [
        ["attn.qkv_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"],
    ]
