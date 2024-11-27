from ..base import BaseGPTQModel


class OvisGPTQ(BaseGPTQModel):
    base_modules = ["llm.model.embed_tokens", "llm.model.norm", "visual_tokenizer", "vte"]

    layers_node = "llm.model.layers"
    layer_type = ["LlamaDecoderLayer", "Gemma2DecoderLayer"]
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
