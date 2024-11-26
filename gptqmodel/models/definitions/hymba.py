from ..base import BaseGPTQModel


class HymbaGPTQ(BaseGPTQModel):
    require_trust_remote_code = True

    base_modules = ["model.embed_tokens", "model.final_layernorm"]

    layers_node = "model.layers"
    layer_type = "HymbaDecoderLayer"
    layer_modules = [
        ["mamba.in_proj"],
        ["mamba.out_proj"],
        # ["mamba.x_proj.0"],
        # ["mamba.dt_proj.0"],
        ["moe.experts.0.up_proj", "moe.experts.0.gate_proj"],
        ["moe.experts.0.down_proj"],
    ]
