from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel


# Both DeepSeek-v2 and DeepSeek-v2-lite are supported in this model def
class OlmoeGPTQ(BaseGPTQModel):

    dynamic_expert_index = "num_experts"

    base_modules = ["model.embed_tokens", "model.norm"]

    layers_node = "model.layers"
    layer_type = "OlmoeDecoderLayer"

    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],

        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj", f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],
    ]
