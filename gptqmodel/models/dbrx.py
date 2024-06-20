from .base import BaseGPTQModel


class DbrxGPTQ(BaseGPTQModel):
    base_modules = ["transformer.wte", "transformer.norm_f"]

    layers_node = "transformer.blocks"
    layer_type = "DbrxBlock"
    layer_modules = [
        ["norm_attn_norm.attn.Wqkv"],
        ["norm_attn_norm.attn.o_proj"],
        ["ffn.experts.mlp.w1", "ffn.experts.mlp.v1"],
        ["ffn.experts.mlp.w2"],
    ]
