from .base import BaseGPTQModel


class GPT2GPTQ(BaseGPTQModel):
    base_modules = ["transformer.wte", "transformer.wpe", "transformer.ln_f"]

    layers_node = "transformer.h"
    layer_type = "GPT2Block"
    layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"],
    ]
