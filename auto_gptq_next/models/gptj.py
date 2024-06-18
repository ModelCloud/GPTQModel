from ._base import BaseGPTQModel


class GPTJGPTQ(BaseGPTQModel):
    non_layer_modules = ["transformer.wte", "transformer.ln_f"]

    layers_node = "transformer.h"
    layer_type = "GPTJBlock"
    layer_modules = [
        ["attn.k_proj", "attn.v_proj", "attn.q_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"],
    ]
