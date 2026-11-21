# SPDX-License-Identifier: Apache-2.0
# Adapted from ricursiveAI/Ripple-65B-Preview@2f007aabfcedd0f8763d86d08f9aa0afbc344cea.
"""HuggingFace PretrainedConfig for the Ricursive transformer architecture."""

from copy import copy

from transformers import PretrainedConfig


class RicursiveConfig(PretrainedConfig):
    """
    Configuration for the Ricursive transformer.

    Key architectural features:
    - PeriNorm blocks (4 layer norms per block instead of 2)
    - Gated attention with elementwise sigmoid gate
    - Mixed local/global layers: local layers use sliding window + RoPE,
      global layers (every 5th) use full attention with no RoPE
    - Per-head Q/K RMSNorm
    """

    model_type = "ricursive"

    def __init__(
        self,
        vocab_size=100352,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=50,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=2048,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        sliding_window=1024,
        layer_types=None,
        use_gated_attention=True,
        gate_full_precision=True,
        use_valres=False,
        ffn_clamp_threshold=None,
        embed_scale=None,
        tie_word_embeddings=False,
        mlp_chunk_size=2048,
        **kwargs,
    ):
        if not isinstance(mlp_chunk_size, int) or mlp_chunk_size < 0:
            raise ValueError("mlp_chunk_size must be a non-negative integer")
        self.mlp_chunk_size = mlp_chunk_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.use_gated_attention = use_gated_attention
        self.gate_full_precision = gate_full_precision
        self.use_valres = use_valres
        self.ffn_clamp_threshold = ffn_clamp_threshold
        # Hybrid-SWA metadata is derived below. Serialized copies must not override the
        # architecture encoded by layer_types.
        kwargs.pop("is_hybrid_swa", None)
        kwargs.pop("hybrid_layer_pattern", None)
        # Embedding scale convention (see modeling_ricursive.RicursiveModel.forward):
        #   - non-folded checkpoints set embed_scale (e.g. sqrt(hidden_size)) and the model
        #     scales the embeddings at runtime;
        #   - folded checkpoints bake the scale into embed_tokens.weight and leave this None.
        # Either way the SGLang native port (sglang_ricursive) reads this same key, so the
        # served and trained forwards agree.
        self.embed_scale = embed_scale

        # Per-layer type: "sliding_attention" (window + RoPE) or "full_attention" (no RoPE)
        # These names follow the HuggingFace PretrainedConfig layer_types validation.
        if layer_types is not None:
            self.layer_types = layer_types
        else:
            # Default: every 5th layer is full_attention (indices 4, 9, 14, ...)
            self.layer_types = [
                "full_attention" if i % 5 == 4 else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        if len(self.layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, expected {num_hidden_layers}"
            )
        invalid_layer_types = set(self.layer_types) - {
            "sliding_attention",
            "full_attention",
        }
        if invalid_layer_types:
            raise ValueError(f"unsupported Ricursive layer_types: {sorted(invalid_layer_types)}")

        # SGLang's compact hybrid KV allocator uses 1 for sliding attention and 0 for full
        # attention. Keep these fields fully derived so they cannot drift from model semantics.
        self.hybrid_layer_pattern = [
            1 if layer_type == "sliding_attention" else 0 for layer_type in self.layer_types
        ]
        self.is_hybrid_swa = 0 in self.hybrid_layer_pattern and 1 in self.hybrid_layer_pattern

        # Standard HF config fields expected by modeling code
        self.use_cache = kwargs.pop("use_cache", True)

        self.base_model_tp_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.self_attn.g_proj": "colwise",
            "layers.*.mlp.gate_proj": "colwise",
            "layers.*.mlp.up_proj": "colwise",
            "layers.*.mlp.down_proj": "rowwise",
        }
        # Checkpoint Python is never part of a package-owned config.
        kwargs.pop("auto_map", None)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        # Must be set AFTER super().__init__ — HF's PretrainedConfig.__init__
        # pops "architectures" from kwargs (default None) and assigns it to
        # self.architectures, which would otherwise overwrite our value.
        self.architectures = ["RicursiveForCausalLM"]

    def get_text_config(self, decoder=False, **kwargs):
        # Native TF5 caches interpret `layer_types` as storage policy. Ripple
        # also uses these names to select RoPE. A disabled window therefore
        # needs full cache storage while retaining RoPE in the model itself.
        # Return a view: never mutate the saved/caller-owned architecture.
        if decoder and self.sliding_window is None and "sliding_attention" in self.layer_types:
            cache_config = copy(self)
            cache_config.layer_types = ["full_attention"] * self.num_hidden_layers
            return cache_config
        return self
