from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel


class GrinMOEGPTQ(BaseGPTQModel):
    # grin moe requires custom model code
    require_trust_remote_code = True

    dynamic_expert_index = "num_local_experts"

    base_modules = ["model.embed_tokens", "model.norm"]

    layers_node = "model.layers"
    layer_type = "GRINMoEDecoderLayer"
    layer_modules = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],

        # uses dynamic_expert_index
        [f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.w1", f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.w3"],
        [f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.w2"],
    ]
