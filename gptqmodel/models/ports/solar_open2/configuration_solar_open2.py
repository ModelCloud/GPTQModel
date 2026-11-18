# Modified by ModelCloud: package-local Transformers 5.23 port with validated layout metadata.
# Copyright 2026 Upstage AI and the HuggingFace Inc. team. All rights reserved.
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
from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class SolarOpen2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SolarOpen2Model`]. It is used to instantiate a
    SolarOpen2 model according to the specified arguments, defining the model architecture.

    Args:
        use_rope (`bool`, *optional*, defaults to `False`):
            Whether the grouped-query full-attention layers apply rotary position embeddings. When `False` the model
            is NoPE (positional information is carried by the linear-attention layers).
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply per-head RMS normalization to the query and key projections of the full-attention layers.
        use_gqa_gate (`bool`, *optional*, defaults to `True`):
            Whether the full-attention layers apply an output sigmoid gate (`attn_output * sigmoid(g_proj(x))`).
        use_gqa_gate_bias (`bool`, *optional*, defaults to `False`):
            Whether the output gate projection uses a bias.
        gqa_interval (`int`, *optional*, defaults to 4):
            When `gqa_layers` is not provided, every layer `i` with `(i + 1) % gqa_interval == 0` is a full-attention
            layer; all other layers use Kimi-Delta linear attention.
        gqa_layers (`list[int]`, *optional*):
            Explicit list of layer indices that use full attention. Takes priority over `gqa_interval`.
        linear_attn_config (`dict`, *optional*):
            Configuration of the Kimi-Delta-Attention layers, with keys `short_conv_kernel_size`, `head_dim`,
            `num_heads` and (optionally) `num_kv_heads`.
        kda_use_full_proj (`bool`, *optional*, defaults to `False`):
            When `True`, the KDA gate/output-gate use single full-rank projections (`f_proj`, `g_proj`) instead of the
            factored low-rank `f_a_proj`/`f_b_proj` and `g_a_proj`/`g_b_proj`.
        kda_gate_lower_bound (`float`, *optional*, defaults to -5.0):
            Lower bound on the KDA log-decay gate.
        kda_allow_neg_eigval (`bool`, *optional*, defaults to `True`):
            When `True`, the KDA `beta` gate is scaled by 2 (allowing negative eigenvalues of the state transition).
        layer_types (`list[str]`, *optional*):
            Per-layer attention type (`"full_attention"` / `"linear_attention"`). Derived from `gqa_layers` /
            `gqa_interval` when not given.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts (grouped top-k routing).
        topk_group (`int`, *optional*, defaults to 1):
            Number of groups selected per token for grouped top-k routing.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor applied to the routed-expert contribution.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of leading dense (non-MoE) decoder layers.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the top-k routing probabilities.
    """

    model_type = "solar_open2"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Routed experts are a per-expert ModuleList (released checkpoint layout), not fused
    # `gate_up_proj` tensors, so no packed/grouped-GEMM expert plans apply.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 196608
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 1_048_576
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 1280
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_routed_experts: int = 320
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    # Released checkpoints store `rope_theta`; configs without it keep the 10k base.
    default_theta = 10_000.0
    head_dim: int = 128
    intermediate_size: int = 10240
    first_k_dense_replace: int = 0

    # SolarOpen2-specific
    use_rope: bool = False
    use_qk_norm: bool = False
    use_gqa_gate: bool = True
    use_gqa_gate_bias: bool = False
    gqa_interval: int = 4
    gqa_layers: list | None = None
    linear_attn_config: dict | None = None
    kda_use_full_proj: bool = False
    kda_gate_lower_bound: float = -5.0
    kda_allow_neg_eigval: bool = True
    layer_types: list | None = None

    def __post_init__(self, **kwargs):
        # Legacy `rope_theta`/`rope_scaling` are folded into `rope_parameters` by PreTrainedConfig.
        # Full attention is NoPE by default; keep the full rotary dim available when use_rope=True.
        kwargs.setdefault("partial_rotary_factor", 1.0)

        if self.linear_attn_config is None:
            self.linear_attn_config = {
                "short_conv_kernel_size": 4,
                "head_dim": self.head_dim,
                "num_heads": self.num_attention_heads,
                "num_kv_heads": None,
            }

        # Grouped-topk degenerates to plain top-k when n_group/topk_group are unset.
        if getattr(self, "n_group", None) is None:
            self.n_group = 1
        if getattr(self, "topk_group", None) is None:
            self.topk_group = 1

        # Derive the per-layer attention pattern consumed by the model and the hybrid cache.
        if self.layer_types is None:
            if self.gqa_layers is not None:
                full = set(self.gqa_layers)
                self.layer_types = [
                    "full_attention" if i in full else "linear_attention" for i in range(self.num_hidden_layers)
                ]
            else:
                self.layer_types = [
                    "full_attention" if (i + 1) % self.gqa_interval == 0 else "linear_attention"
                    for i in range(self.num_hidden_layers)
                ]

        super().__post_init__(**kwargs)
        self.layer_types = list(self.layer_types)
        self.linear_attn_config = dict(self.linear_attn_config)
        if len(self.layer_types) != self.num_hidden_layers or any(
            kind not in ("full_attention", "linear_attention") for kind in self.layer_types
        ):
            raise ValueError("layer_types must contain one full_attention/linear_attention entry per decoder layer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        linear_heads = self.linear_attn_config["num_heads"]
        kv_heads = self.linear_attn_config.get("num_kv_heads") or linear_heads
        if linear_heads % kv_heads or self.linear_attn_config["short_conv_kernel_size"] < 1:
            raise ValueError("Invalid KDA head grouping or convolution size")
        if self.first_k_dense_replace < self.num_hidden_layers:
            if self.n_routed_experts < 2 or not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
                raise ValueError("Invalid routed expert count or top-k")
            if self.n_routed_experts % self.n_group or self.n_routed_experts // self.n_group < 2:
                raise ValueError("Expert groups must divide the expert count and contain at least two experts")
            if not 1 <= self.topk_group <= self.n_group:
                raise ValueError("Invalid topk_group")
            if self.num_experts_per_tok > self.topk_group * (self.n_routed_experts // self.n_group):
                raise ValueError("Selected groups cannot supply the requested expert top-k")


__all__ = ["SolarOpen2Config"]
