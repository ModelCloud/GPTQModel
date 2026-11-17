# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from transformers.models.minimax.configuration_minimax import MiniMaxConfig


class MiniMaxText01Config(MiniMaxConfig):
    """Native hybrid MiniMax config retaining the released partial-RoPE width."""

    model_type = "minimax"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        parameters = dict(self.rope_parameters or {})
        rotary_dim = getattr(self, "rotary_dim", None)
        if rotary_dim is None:
            factor = parameters.get("partial_rotary_factor", 1.0)
            if not isinstance(factor, (int, float)) or not 0 < factor <= 1:
                raise ValueError("MiniMax partial_rotary_factor must be in (0, 1]")
            rotary_dim = int(head_dim * factor)
        if not isinstance(rotary_dim, int) or rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError("MiniMax rotary_dim must be a positive even integer no larger than head_dim")
        self.rotary_dim = rotary_dim
        if parameters.get("rope_type", "default") != "default":
            parameters["partial_rotary_factor"] = rotary_dim / head_dim
        self.rope_parameters = parameters
