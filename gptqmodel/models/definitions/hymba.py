from ..base import BaseGPTQModel


class HymbaGPTQ(BaseGPTQModel):
    require_trust_remote_code = True
    require_monkeypatch = True

    base_modules = ["model.embed_tokens", "model.final_layernorm"]

    layers_node = "model.layers"
    layer_type = "HymbaDecoderLayer"
    layer_modules = [
        ["mamba.in_proj"],
        ["mamba.out_proj"],
        # ["mamba.x_proj.0"],
        # ["mamba.dt_proj.0"], TODO We need to add auto pad to TritonV2QuantLinear before we can quantify the Module.
        ["moe.experts.0.up_proj", "moe.experts.0.gate_proj"],
        ["moe.experts.0.down_proj"],
    ]

    def monkey_patch(self):
        if hasattr(self.config, 'conv_dim'):
            new_conv_dim = dict()
            try:
                for k, v in self.config.conv_dim.items():
                    if isinstance(k, str):
                        new_conv_dim[int(k)] = v
                self.config.conv_dim = new_conv_dim
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError("The key of HymbaConfig.conv_dim should be a string of numbers.")
