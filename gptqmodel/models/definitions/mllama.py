from transformers import AutoModelForPreTraining

from ..base import BaseGPTQModel


# TODO FIXME: we currently do not support quantizing cross attention layer (pixel_values)
class MLlamaGPTQ(BaseGPTQModel):
    # AutoModelForPreTraining return a correct MLlamaForConditionalGeneration for mllama.
    loader = AutoModelForPreTraining

    # Non-repeating layers at the root level: same level as `layers_node`
    # Excluding `layers_node`.
    base_modules = ["language_model.model.embed_tokens", "language_model.model.norm"]

    # Below describes all the repeating layers in this transformer model
    # `model.layers` is a node/module that hold all the repeating layers. The parent node for all n-layers.
    layers_node = "language_model.model.layers"
    # MLllama has two types of repeating layers. Repeats in groups of 4 layers: 0-2 (first 3 layers) is text layers, 3 (4th) is cross-attention layer for vision
    layer_type = "MllamaSelfAttentionDecoderLayer"
    # Inside each `LlamaDecoderLayer` layer are many internal modules
    # List them in the order executed in model forward() code
    # Many models have same execution order of: attention (q_k_v) projection, attention (output) projection, mlp (n) projections
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
