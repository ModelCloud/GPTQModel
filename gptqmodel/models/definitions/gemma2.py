from ...utils import BACKEND
from ...utils.logger import setup_logger
from ..base import BaseGPTQModel

logger = setup_logger()

SUPPORT_ERR = "Currently, only vLLM/SGLang with flashinfer enabled can correctly inference a quantized Gemma2-27B model. Pre-quantized model with sample vLLM code: https://huggingface.co/ModelCloud/gemma-2-27b-it-gptq-4bit ."

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
        if hasattr(self.model.config, "num_hidden_layers"):
            num_hidden_layers = getattr(self.model.config, "num_hidden_layers")
            # The gemma-2 model 9b has 42 hidden layers, while the gemma-2 model 27b has 46 hidden layers.
            if num_hidden_layers > 42:
                if not self.quantized:
                    logger.warning(SUPPORT_ERR)
                    return

                # quantized gemma-2 27b model only support vLLM/SGLang load.
                from ...utils.vllm import VLLM_AVAILABLE
                if VLLM_AVAILABLE:
                    from vllm import LLM
                    if isinstance(self.model, LLM):
                        backend = BACKEND.VLLM

                from ...utils.sglang import SGLANG_AVAILABLE
                if SGLANG_AVAILABLE:
                    from sglang.srt.server import Runtime
                    if isinstance(self.model, Runtime):
                        backend = BACKEND.SGLANG

                if backend not in [BACKEND.VLLM,  BACKEND.SGLANG]:
                    raise ValueError(SUPPORT_ERR)

