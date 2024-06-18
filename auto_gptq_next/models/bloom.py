from ._base import BaseGPTQModel


class BloomGPTQ(BaseGPTQModel):
    # non-layer (root) modules
    non_layer_modules = [
        "transformer.word_embeddings",
        "transformer.word_embeddings_layernorm",
        "transformer.ln_f",
    ]

    # repeating layers
    layers_node = "transformer.h"
    layer_type = "BloomBlock"
    layer_modules = [
        ["self_attention.query_key_value"],
        ["self_attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]
