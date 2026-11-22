# SPDX-License-Identifier: Apache-2.0
"""raon-1-mini configuration for the Hugging Face ``transformers`` implementation.

The attribute surface matches the release ``config.json``, which the Raon vLLM runtime's
``Raon1MiniConfig`` also reads. Only ``model_type``
matters for ``AutoConfig``; every architectural knob defaults to the released
28.8B checkpoint so ``Raon1MiniConfig()`` alone describes the production model.
"""

from __future__ import annotations

from typing import ClassVar

from transformers.configuration_utils import PretrainedConfig


_RELEASED_RDA_LAYERS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
_RELEASED_MLA_LAYERS = [4, 8, 12, 16, 20, 24, 25]


class Raon1MiniConfig(PretrainedConfig):
    """Configuration of a raon-1-mini model.

    Layer indices inside ``rda_config`` (``rda_layers``: Raon Delta Attention, ``mla_layers``:
    MLA) are 1-based; ``layer_types`` converts.
    """

    model_type = "raon1_mini"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 259840,
        hidden_size: int = 2560,
        intermediate_size: int = 12032,
        num_hidden_layers: int = 25,
        num_attention_heads: int = 40,
        head_dim: int | None = 128,
        hidden_act: str = "situ",
        activation_situ_beta: float = 4.0,
        activation_situ_linear_beta: float = 25.0,
        rms_norm_eps: float = 1e-5,
        mla_latent_norm_eps: float = 1e-6,
        raon_mhc_selector_eps: float | None = None,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        max_position_embeddings: int = 262144,
        # MoE
        num_experts: int | None = 128,
        num_experts_per_token: int = 6,
        moe_intermediate_size: int = 1088,
        num_shared_experts: int = 1,
        moe_router_activation_func: str = "sigmoid",
        moe_renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 1,
        moe_layer_freq: int = 1,
        # MLA
        q_lora_rank: int = 576,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        mla_use_nope: bool = True,
        mla_use_output_gate: bool = True,
        # RDA
        rda_config: dict | None = None,
        # Raon_mHC
        raon_mhc_streams: int | None = None,
        raon_mhc_post_factor: float | None = None,
        rope_parameters: dict | None = None,
        pad_token_id: int = 259200,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        **kwargs,
    ) -> None:
        compressed = kwargs.pop("raon_mxfp8_config", None)
        quantization = kwargs.get("quantization_config")
        if isinstance(quantization, dict) and quantization.get("quant_method") == "mxfp8":
            if compressed is not None and compressed != quantization:
                raise ValueError("Conflicting Raon MXFP8 storage metadata.")
            compressed = kwargs.pop("quantization_config")
        if compressed is not None:
            if compressed.get("quant_method") != "mxfp8" or compressed.get("weight_block_size") != [1, 32]:
                raise ValueError("Raon routed MXFP8 storage requires weight_block_size=[1, 32].")
            if compressed.get("activation_scheme", "dynamic") != "dynamic" or compressed.get("dequantize", False):
                raise ValueError("Raon keeps MXFP8 storage; load-time dequantization is not supported.")
            if hidden_size % 32 or moe_intermediate_size % 32:
                raise ValueError("Raon MXFP8 projection widths must be divisible by 32.")
        self.raon_mxfp8_config = compressed
        if hidden_act != "situ" or not mla_use_nope:
            raise ValueError("The pinned Raon implementation supports SiTU and NoPE MLA only.")
        if raon_mhc_streams is not None and raon_mhc_streams < 2:
            raise ValueError("Raon mHC requires at least two residual streams.")
        if num_experts and not 0 < num_experts_per_token <= num_experts:
            raise ValueError("num_experts_per_token must fit the routed expert count.")
        self.raon_mhc_streams = 4 if raon_mhc_streams is None else raon_mhc_streams
        self.raon_mhc_post_factor = 2.0 if raon_mhc_post_factor is None else raon_mhc_post_factor
        self.raon_mhc_selector_eps = 1e-5 if raon_mhc_selector_eps is None else raon_mhc_selector_eps
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.rms_norm_eps = rms_norm_eps
        self.mla_latent_norm_eps = mla_latent_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_intermediate_size = moe_intermediate_size
        self.num_shared_experts = num_shared_experts
        self.moe_router_activation_func = moe_router_activation_func
        self.moe_renormalize = moe_renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        if rda_config is None:
            rda_config = {
                "num_heads": num_attention_heads,
                "head_dim": self.head_dim,
                "conv_kernel_size": 4,
                "full_rank_gate": True,
                "gate_lower_bound": -5.0,
                "rda_layers": list(_RELEASED_RDA_LAYERS),
                "mla_layers": list(_RELEASED_MLA_LAYERS),
            }
        self.rda_config = dict(rda_config)
        rda = self.rda_config.get("rda_layers", [])
        mla = self.rda_config.get("mla_layers", [])
        if sorted(rda + mla) != list(range(1, num_hidden_layers + 1)):
            raise ValueError("rda_layers and mla_layers must partition the 1-based decoder layers.")
        if self.rda_conv_kernel_size < 1:
            raise ValueError("Raon causal convolution requires a positive kernel size.")
        # NoPE: nothing reads this, kept so tools that expect a mapping do not trip.
        self.rope_parameters = rope_parameters
        # Decay rank is fixed by the released checkpoint (f_a_proj is 128 wide); it is only read to build shapes.
        self.rda_decay_rank = kwargs.pop("rda_decay_rank", 128)
        if eos_token_id is None:
            eos_token_id = [259202, 259200]
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ---- derived views -------------------------------------------------------------------
    @property
    def rda_num_heads(self) -> int:
        return int(self.rda_config.get("num_heads", self.num_attention_heads))

    @property
    def rda_head_dim(self) -> int:
        return int(self.rda_config.get("head_dim", self.head_dim))

    @property
    def rda_conv_kernel_size(self) -> int:
        return int(self.rda_config.get("conv_kernel_size", 4))

    @property
    def rda_gate_lower_bound(self) -> float | None:
        return self.rda_config.get("gate_lower_bound", -5.0)

    def is_rda_layer(self, layer_idx: int) -> bool:
        """0-based layer index -> True for an RDA (linear attention) layer."""
        return (layer_idx + 1) in self.rda_config["rda_layers"]

    def is_moe_layer(self, layer_idx: int) -> bool:
        """0-based layer index -> True when the MLP is the routed MoE block."""
        if not self.num_experts:
            return False
        return layer_idx >= self.first_k_dense_replace and layer_idx % self.moe_layer_freq == 0

    @property
    def layer_types(self) -> list[str]:
        return [
            "linear_attention" if self.is_rda_layer(i) else "full_attention" for i in range(self.num_hidden_layers)
        ]


__all__ = ["Raon1MiniConfig"]
