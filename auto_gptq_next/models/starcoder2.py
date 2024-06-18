from ._base import BaseGPTQForCausalLM


class Starcoder2GPTQ(BaseGPTQForCausalLM):
    non_layer_modules = ["model.embed_tokens", "model.norm"]

    layers_node = "model.layers"
    layer_type = "Starcoder2DecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"],
    ]
