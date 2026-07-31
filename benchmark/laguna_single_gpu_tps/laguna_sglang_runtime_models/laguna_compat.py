"""Laguna single-GPU benchmark compatibility adapter for SGLang."""

from collections.abc import Iterable

import torch

from sglang.srt.models.laguna import LagunaForCausalLM as _MainLagunaForCausalLM


class LagunaForCausalLM(_MainLagunaForCausalLM):
    """Map the checkpoint's plural shared-expert prefix to main's singular prefix."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        remapped_weights = (
            (
                name.replace(".mlp.shared_experts.", ".mlp.shared_expert."),
                loaded_weight,
            )
            for name, loaded_weight in weights
        )
        return super().load_weights(remapped_weights)


EntryClass = LagunaForCausalLM
