from ._const import EXPERT_INDEX_PLACEHOLDER
from .base import BaseGPTQModel


class Qwen2MoeGPTQ(BaseGPTQModel):
    # qwen2moe requires true_sequential = False
    require_true_sequential = False

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"

    base_modules = ["model.embed_tokens", "model.norm"]

    layers_node = "model.layers"
    layer_type = "Qwen2DecoderLayer"
    layer_modules = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.shared_expert.up_proj", "mlp.shared_expert.gate_proj"],
        ["mlp.shared_expert.down_proj"],

        # uses dynamic_expert_index
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj", f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],
    ]
