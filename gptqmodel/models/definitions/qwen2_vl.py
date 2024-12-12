from ..base import BaseGPTQModel
from transformers import Qwen2VLForConditionalGeneration

class Qwen2VLGPTQ(BaseGPTQModel):
    # use correct Qwen2VLForConditionalGeneration for qwen2_vl.
    loader = Qwen2VLForConditionalGeneration

    base_modules = ["model.embed_tokens", "model.norm"]

    layers_node = "model.layers"
    layer_type = "Qwen2VLDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
