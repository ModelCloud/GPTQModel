from .base import BaseGPTQModel


class Qwen2MoeGPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "model.norm"]

    layers_node = "model.layers"
    layer_type = "Qwen2DecoderLayer"
    layer_modules = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.shared_expert.up_proj", "mlp.shared_expert.gate_proj"],
        ["mlp.shared_expert.down_proj"],
        ["mlp.experts.{expert_idx}.up_proj", "mlp.experts.{expert_idx}.gate_proj"],
        ["mlp.experts.{expert_idx}.down_proj"],
    ]