# Copyright 2026 Agnes AI. All rights reserved.
# Adapted for package-owned Transformers 5 loading by ModelCloud.ai, 2026.
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
"""Agnes 3.0 Flash model configuration."""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import auto_docstring


LAYER_GLOBAL = "agnes_global_attention"
LAYER_DELTA = "agnes_delta_attention"
LAYER_TYPES = (LAYER_GLOBAL, LAYER_DELTA)


@auto_docstring
@strict
class AgnesTextConfig(PreTrainedConfig):
    r"""
    linear_num_key_heads (`int`, *optional*, defaults to 16):
        Number of key heads in the delta-rule (recurrent) attention layers.
    linear_num_value_heads (`int`, *optional*, defaults to 32):
        Number of value heads in the delta-rule attention layers.
    linear_key_head_dim (`int`, *optional*, defaults to 128):
        Per-head key width in the delta-rule attention layers.
    linear_value_head_dim (`int`, *optional*, defaults to 128):
        Per-head value width in the delta-rule attention layers.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel width of the causal depthwise convolution feeding the delta-rule layers.
    parallel_ffn_intermediate_size (`int`, *optional*, defaults to 0):
        Width of the dense SwiGLU branch that runs beside the main MLP in every
        decoder layer and is summed into the same residual. 0 disables the branch.
    """

    model_type = "agnes_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    # core dimensions
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 12288
    parallel_ffn_intermediate_size: int = 0
    vocab_size: int = 248320

    # layer plan
    layer_types: list[str] | None = None

    # global attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    attention_dropout: float | int = 0.0

    # delta-rule attention
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # ffn / norm / init
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # positions
    rope_parameters: RopeParameters | dict | None = None
    max_position_embeddings: int = 32768

    # tokens / runtime
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False
    use_cache: bool = True

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_tp_plan = {
        "layers.*.global_attn.q_proj": "colwise",
        "layers.*.global_attn.k_proj": "colwise",
        "layers.*.global_attn.v_proj": "colwise",
        "layers.*.global_attn.o_proj": "rowwise",
        "layers.*.global_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.global_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.parallel_ffn.gate_proj": "colwise",
        "layers.*.mlp.parallel_ffn.up_proj": "colwise",
        "layers.*.mlp.parallel_ffn.down_proj": "rowwise",
    }

    def __post_init__(self, **kwargs):
        # rotary embedding covers a quarter of each head unless told otherwise
        kwargs.setdefault("partial_rotary_factor", 0.25)
        if self.layer_types is None:
            period = kwargs.pop("global_attention_interval", kwargs.pop("full_attention_interval", 4))
            plan = []
            for i in range(self.num_hidden_layers):
                is_global = (i + 1) % period == 0
                plan.append(LAYER_GLOBAL if is_global else LAYER_DELTA)
            self.layer_types = plan
        super().__post_init__(**kwargs)

    def validate_layer_type(self):
        """The layer plan uses this model's own type names, so the generic
        allow-list in the base class does not apply."""
        plan = self.layer_types
        if plan is None:
            return
        unknown = sorted({t for t in plan if t not in LAYER_TYPES})
        if unknown:
            raise ValueError(f"`layer_types` entries must be in {LAYER_TYPES}, got {unknown}")
        if self.num_hidden_layers is not None and self.num_hidden_layers != len(plan):
            raise ValueError(
                f"`num_hidden_layers` ({self.num_hidden_layers}) must equal the number of `layer_types` ({len(plan)})"
            )


@auto_docstring
@strict
class AgnesVisionConfig(PreTrainedConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 3584):
        Width of the merged visual tokens handed to the language model.
    num_position_embeddings (`int`, *optional*, defaults to 2304):
        Size of the learned 2-D position table (a square grid, side = sqrt of this).
    """

    model_type = "agnes_vision"
    base_config_key = "vision_config"
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "depth"}

    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    out_hidden_size: int = 3584
    patch_size: int | list[int] | tuple[int, int] = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    in_channels: int = 3
    num_position_embeddings: int = 2304
    hidden_act: str = "gelu_pytorch_tanh"
    initializer_range: float = 0.02


@auto_docstring
@strict
class AgnesConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from gptqmodel.models.ports.agnes.configuration_agnes import AgnesConfig

    >>> configuration = AgnesConfig.from_pretrained("<model dir>")
    >>> configuration.text_config.num_hidden_layers
    72
    ```"""

    model_type = "agnes"
    sub_configs = {"text_config": AgnesTextConfig, "vision_config": AgnesVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        for key, cls in self.sub_configs.items():
            value = getattr(self, key)
            if isinstance(value, dict):
                setattr(self, key, cls(**value))
            elif value is None:
                setattr(self, key, cls())
        super().__post_init__(**kwargs)


__all__ = ["AgnesConfig", "AgnesTextConfig", "AgnesVisionConfig", "LAYER_GLOBAL", "LAYER_DELTA", "LAYER_TYPES"]
