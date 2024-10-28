from ..base import BaseGPTQModel


class MiniCPM3GPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens",]

    layers_node = "model.layers"
    layer_type = "MiniCPM3DecoderLayer"
    layer_modules = [
        ["self_attn.q_a_proj","self_attn.kv_a_proj_with_mqa"],
        ["self_attn.q_b_proj","self_attn.kv_b_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj","mlp.up_proj"],
        ["mlp.down_proj"],
    ]
