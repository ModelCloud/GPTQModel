from .base import BaseGPTQModel


class Phi3GPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "embed_dropout", "model.norm"]

    layers_node = "model.layers"
    layer_type = "Phi3DecoderLayer"
    layer_modules = [
        ["self_attn.qkv_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_up_proj"],
        ["mlp.down_proj"],
    ]
