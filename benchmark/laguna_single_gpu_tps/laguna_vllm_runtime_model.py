"""Laguna single-GPU benchmark compatibility adapter for vLLM."""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.laguna import (
    LagunaForCausalLM as _MainLagunaForCausalLM,
)


class LagunaForCausalLM(_MainLagunaForCausalLM):
    """Map the checkpoint's plural shared-expert prefix to main's singular prefix."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        quant_config = vllm_config.quant_config
        quantized_layers = getattr(
            quant_config,
            "modules_in_block_to_quantize",
            None,
        )
        if quantized_layers:
            quant_config.modules_in_block_to_quantize = [
                name.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
                for name in quantized_layers
            ]
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        remapped_weights = (
            (
                name.replace(".mlp.shared_experts.", ".mlp.shared_expert."),
                loaded_weight,
            )
            for name, loaded_weight in weights
        )
        return super().load_weights(remapped_weights)
