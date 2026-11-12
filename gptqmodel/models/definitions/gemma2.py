# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...utils import BACKEND
from ...utils.logger import setup_logger
from ..fusable import FusableConfig, LlamaFusableConfig
from . import LlamaQModel


log = setup_logger()

SUPPORT_ERR = "Currently, only vLLM/SGLang with flashinfer enabled can correctly inference a quantized Gemma2-27B model. Pre-quantized model with sample vLLM code: https://huggingface.co/ModelCloud/gemma-2-27b-it-gptq-4bit ."


class Gemma2QModel(LlamaQModel):
    fusable = FusableConfig("gemma2").merge(LlamaFusableConfig())

    shared_input_verified_model_types = frozenset({"gemma2"})

    # Gemma 2 wraps the MLP in two additional RMSNorm layers. They own
    # checkpoint weights but are never quantized, so they must be declared as
    # excluded leaves; the generic Llama tree cannot resolve them.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!:norm",),
            "self_attn": ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1:o"),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "pre_feedforward_layernorm": ("pre_feedforward_layernorm:!",),
            "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
            "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # There is an issue with duplicate outputs in the quantized gemma-2 model 27b with transformers.
        if hasattr(self.model.config, "num_hidden_layers"):
            num_hidden_layers = getattr(self.model.config, "num_hidden_layers")
            # The gemma-2 model 9b has 42 hidden layers, while the gemma-2 model 27b has 46 hidden layers.
            if num_hidden_layers > 42:
                if not self.quantized:
                    log.warn(SUPPORT_ERR)
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

