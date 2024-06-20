from .base import BaseGPTQModel


class DbrxGPTQ(BaseGPTQModel):
    base_modules = ["transformer.wte", "transformer.norm_f"]

    layers_node = "transformer.blocks"
    layer_type = "DbrxBlock"
    layer_modules = [
        ["norm_attn_norm.attn.Wqkv"],
        ["norm_attn_norm.attn.out_proj"],
    ]
