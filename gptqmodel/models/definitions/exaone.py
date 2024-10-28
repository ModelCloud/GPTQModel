from ..base import BaseGPTQModel


class ExaoneGPTQ(BaseGPTQModel):
    # exaone requires custom model code
    require_trust_remote_code = True

    base_modules = ["transformer.ln_f", "transformer.wte"]

    layers_node = "transformer.h"
    layer_type = "ExaoneBlock"
    layer_modules = [
        ["attn.attention.k_proj", "attn.attention.v_proj", "attn.attention.q_proj"],
        ["attn.attention.out_proj"],
        ["mlp.c_fc_0", "mlp.c_fc_1"],
        ["mlp.c_proj"],
    ]
