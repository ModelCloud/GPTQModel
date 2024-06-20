from .base import BaseGPTQModel


class PhiGPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "model.final_layernorm"]

    layers_node = "model.layers"
    layer_type = "PhiDecoderLayer"
    layer_modules = [
        ["self_attn.q_proj"],
        ["self_attn.k_proj"],
        ["self_attn.v_proj"],
        ["self_attn.dense"],
        ["mlp.fc1"],
        ["mlp.fc2"],
    ]
