from .base import BaseGPTQModel


class OPTGPTQ(BaseGPTQModel):
    base_modules = [
        "model.decoder.embed_tokens",
        "model.decoder.embed_positions",
        "model.decoder.project_out",
        "model.decoder.project_in",
        "model.decoder.final_layer_norm",
    ]

    layers_node = "model.decoder.layers"
    layer_type = "OPTDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        # ["fc1"], disabled: not a good candidate for quantization
        # ["fc2"], disabled: not a good candidate for quantization
    ]
