from ..base import BaseGPTQModel


class Olmo2GPTQ(BaseGPTQModel):
    require_pkgs_version = ["transformers>=4.47.0"]
    base_modules = ["model.embed_tokens", "model.norm"]

    layers_node = "model.layers"
    layer_type = "Olmo2DecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
