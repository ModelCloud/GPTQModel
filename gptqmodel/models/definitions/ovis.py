import torch

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

    # hack so one can prepare examples outside
    def _prepare_dataset_for_quantization(
            self,
            calibration_dataset,
            batch_size: int = 1,
            tokenizer=None, ):
        return calibration_dataset

    def generate(self, inputs, **kwargs):
        """shortcut for model.generate"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(inputs, **kwargs)
