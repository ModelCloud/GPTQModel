from .base import BaseGPTQModel
import logging
from logging import getLogger

logger = getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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

        # There is an issue with duplicate outputs in the quantized gemma-2 model 27b with transformers.
        # Until this issue is fixed, quantized gemma-2 27b model only support vLLM load.
        if hasattr(self.model.config, "num_hidden_layers"):
            num_hidden_layers = getattr(self.model.config, "num_hidden_layers")
            # The gemma-2 model 9b has 42 hidden layers, while the gemma-2 model 27b has 46 hidden layers.
            if num_hidden_layers > 42:
                if self.quantized:
                    raise ValueError("Currently, only vllm can load the quantized gemma2-27b for proper inference. https://huggingface.co/ModelCloud/gemma-2-27b-it-gptq-4bit is a quantized gemma-2-27b-it model, along with an example of loading it using vLLM.")
                else:
                    logger.warning("Currently, only vllm can load the quantized gemma2-27b for proper inference. https://huggingface.co/ModelCloud/gemma-2-27b-it-gptq-4bit is a quantized gemma-2-27b-it model, along with an example of loading it using vLLM.")

