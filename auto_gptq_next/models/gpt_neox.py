from ._base import BaseGPTQModel


class GPTNeoXGPTQ(BaseGPTQModel):
    non_layer_modules = ["gpt_neox.embed_in", "gpt_neox.final_layer_norm"]
    lm_head = "embed_out"

    layers_node = "gpt_neox.layers"
    layer_type = "GPTNeoXLayer"
    layer_modules = [
        ["attention.query_key_value"],
        ["attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]

