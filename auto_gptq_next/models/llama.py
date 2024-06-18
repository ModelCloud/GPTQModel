from ._base import BaseGPTQForCausalLM


class LlamaGPTQ(BaseGPTQForCausalLM):
    # non-repeating layers at the root level
    non_layer_modules = ["model.embed_tokens", "model.norm"]

    # below holds describes all the repeating layers in a transformer model
    # `model.layers` is a node in the weights that hold all the repeating layers
    layers_node = "model.layers"
    # each repeating layer in `model.layers` is of type `LlamaDecoderLayer`
    layer_type = "LlamaDecoderLayer"
    # inside each `LlamaDecoderLayer` layer are many internal modules
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
