from .base import BaseGPTQModel


class MiniCPMGPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens",]

    layers_node = "model.layers"
    layer_type = "MiniCPMDecoderLayer"
    layer_modules = [
        ["self_attn.q_proj"],
        ["self_attn.k_proj"],
        ["self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj","mlp.down_proj"],
        ["mlp.c_proj"],
    ]
