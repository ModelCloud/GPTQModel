# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# GPT-QModel adaptation: package-owned loading and Transformers 5 execution.
# Pinned source, modifications and qualification limits: README.md and NOTICE.
from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import auto_docstring


@auto_docstring(checkpoint="LiquidAI/LFM3-1.7B-Preview")
@strict
class Lfm3Config(PreTrainedConfig):
    r"""
    rope_parameters (`dict[str, RopeParameters]`, *optional*):
        RoPE parameters keyed by attention layer type, for example `{"full_attention": {"rope_type": "default",
        "rope_theta": 1000000.0, "partial_rotary_factor": 0.5}, "sliding_attention": {"rope_type": "default",
        "rope_theta": 10000.0}}`. `conv` layers do not use RoPE and have no entry. Full-attention layers rotate the
        first `partial_rotary_factor * head_dim` dimensions and leave the rest unrotated.
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the conv layers.
    conv_L_cache (`int`, *optional*, defaults to 7):
        L_cache dim in the conv layers.
    layer_types (`list[str]`, *optional*):
        Per-layer operator type, each one of `"full_attention"`, `"sliding_attention"` or `"conv"`. Defaults to the
        LFM3 pattern of a full-attention layer every fourth layer starting from the third, a sliding-attention layer
        in between, and gated short convolutions elsewhere.
    sliding_window (`int`, *optional*, defaults to 256):
        Sliding-window size of the `sliding_attention` layers.
    use_sandwich_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply an RMSNorm to the operator and feed-forward outputs before they enter the residual stream.

    ```python
    >>> from transformers import Lfm3Model, Lfm3Config

    >>> # Initializing a LFM3 model
    >>> configuration = Lfm3Config()

    >>> # Initializing a model from the LFM3-1.7B style configuration
    >>> model = Lfm3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "lfm3"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    vocab_size: int = 128_000
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 128_000
    initializer_range: float = 0.02
    norm_eps: float = 0.00001
    use_cache: bool = True
    pad_token_id: int | None = 124_893
    bos_token_id: int | None = 124_894
    eos_token_id: int | list[int] | None = 124_900
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    conv_bias: bool = False
    conv_L_cache: int = 7
    layer_types: list[str] | None = None
    _attention_layer_types = ("full_attention", "sliding_attention")
    head_dim: int | None = None
    sliding_window: int | None = 256
    use_sandwich_norm: bool = True

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            }
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"
                for i in range(self.num_hidden_layers)
            ]

        self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of ``@strict``-powered validation."""
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` has {len(self.layer_types)} entries but `num_hidden_layers` is {self.num_hidden_layers}."
            )
        allowed_layer_types = (*self._attention_layer_types, "conv")
        invalid_layer_types = [layer_type for layer_type in self.layer_types if layer_type not in allowed_layer_types]
        if invalid_layer_types:
            raise ValueError(f"`layer_types` entries must be one of {allowed_layer_types}; got {invalid_layer_types}.")
        if "sliding_attention" in self.layer_types and self.sliding_window is None:
            raise ValueError("`layer_types` has `sliding_attention` layers but `sliding_window` is None.")
        missing_rope = [
            layer_type
            for layer_type in self._attention_layer_types
            if layer_type in self.layer_types and layer_type not in self.rope_parameters
        ]
        if missing_rope:
            raise ValueError(f"`rope_parameters` has no entry for the {missing_rope} layers in `layer_types`.")
        for layer_type, rope_params in self.rope_parameters.items():
            if not isinstance(rope_params, dict):
                continue
            if (
                rope_params.get("rope_type", "default") in ("dynamic", "longrope")
                and rope_params.get("partial_rotary_factor", 1.0) < 1
            ):
                raise ValueError(
                    f"`{rope_params['rope_type']}` RoPE recomputes its frequencies during the forward pass over the "
                    f"rotated dimension only and cannot keep LFM3's partial rotation on `{layer_type}` layers."
                )


__all__ = ["Lfm3Config"]
