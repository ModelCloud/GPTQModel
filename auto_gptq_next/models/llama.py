from ._base import BaseGPTQForCausalLM


class LlamaGPTQ(BaseGPTQForCausalLM):
    # non-repeating layers at the root level (same level as layers_node)
    non_layer_modules = ["model.embed_tokens", "model.norm"]

    # below describes all the repeating layers in this transformer model
    # `model.layers` is a node/module that hold all the repeating layers. The parent node for all n-layers.
    layers_node = "model.layers"
    # each repeating layer in `model.layers` is of type `LlamaDecoderLayer`
    layer_type = "LlamaDecoderLayer"
    # inside each `LlamaDecoderLayer` layer are many internal modules
    # list them in the order executed in model forward() code
    # many models have same execution order of: attention (q_k_v) projection, attention (output) projection, mlp (n) projections
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
