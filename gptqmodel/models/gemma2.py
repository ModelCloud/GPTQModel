from .base import BaseGPTQModel


class Gemma2GPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "model.norm"]

    layers_node = "model.layers"
    layer_type = "Gemma2DecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # There is an issue with duplicate outputs in the quantized gemma-2 model 27b.
        # Until this issue is fixed, quantize gemma-2 27b model is not supported.
        num_hidden_layers = getattr(self.model.config, "num_hidden_layers")
        # The gemma-2 model 9b has 42 hidden layers, while the gemma-2 model 27b has 46 hidden layers.
        if num_hidden_layers > 42:
            raise ValueError("Only Gemma-2 9B models are supported at the moment. For Gemma-2 27B models please following the github issue at https://github.com/ModelCloud/GPTQModel/issues/140 .")

