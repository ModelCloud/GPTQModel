from ._base import BaseGPTQForCausalLM


class ChatGLM(BaseGPTQForCausalLM):
    non_layer_modules = ["transformer.embedding.word_embeddings", "transformer.output_layer"]

    layers_node = "transformer.encoder.layers"
    layer_type = "GLMBlock"
    layer_modules = [
        ["self_attention.query_key_value"],
        ["self_attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
    ]
