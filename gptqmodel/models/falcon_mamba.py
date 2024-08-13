from .base import BaseGPTQModel


class FalconMambaGPTQ(BaseGPTQModel):
    base_modules = ["backbone.embeddings", "backbone.norm_f"]

    layers_node = "backbone.layers"
    layer_type = "FalconMambaBlock"
    layer_modules = [
        ["mixer.in_proj"],
        ["mixer.x_proj"],
        ["mixer.dt_proj"],
        ["mixer.out_proj"],
    ]
