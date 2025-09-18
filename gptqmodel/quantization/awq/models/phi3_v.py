from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer as OldPhi3DecoderLayer

from .base import BaseAWQForCausalLM


class Phi3VAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Phi3DecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["vision_embed_tokens"]

    @staticmethod
    def get_model_layers(model):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldPhi3DecoderLayer):
        return {"is_scalable": False}

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldPhi3DecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            {
                "prev_op": module.input_layernorm,
                "layers": [module.self_attn.qkv_proj],
                "inp": input_feat["self_attn.qkv_proj"],
                "module2inspect": module.self_attn,
                "kwargs": module_kwargs,
            }
        )

        # attention out
        layers.append(
            {
                "prev_op": module.self_attn.qkv_proj,
                "layers": [module.self_attn.o_proj],
                "inp": input_feat["self_attn.o_proj"],
            }
        )

        # linear 1
        layers.append(
            {
                "prev_op": module.post_attention_layernorm,
                "layers": [module.mlp.gate_up_proj],
                "inp": input_feat["mlp.gate_up_proj"],
                "module2inspect": module.mlp,
            }
        )

        # linear 2
        layers.append(
            {
                "prev_op": module.mlp.gate_up_proj,
                "layers": [module.mlp.down_proj],
                "inp": input_feat["mlp.down_proj"],
            }
        )

        return layers
