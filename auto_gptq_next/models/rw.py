from ._base import BaseGPTQModel


class RWGPTQ(BaseGPTQModel):
    non_layer_modules = ["transformer.word_embeddings", "transformer.ln_f"]

    layers_node = "transformer.h"
    layer_type = "DecoderLayer"
    layer_modules = [
        ["self_attention.query_key_value"],
        ["self_attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]
