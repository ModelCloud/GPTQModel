# SPDX-License-Identifier: Apache-2.0
"""Allowlisted selection, never checkpoint AutoConfig/AutoModel dispatch."""

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, PretrainedConfig, StableLmConfig

from .configuration_longllama import LongLlamaConfig


NATIVE_TYPES = frozenset(
    {
        "cohere",
        "cohere2",
        "gemma",
        "granite",
        "helium",
        "llama",
        "mistral",
        "olmo",
        "seed_oss",
        "smollm3",
        "stablelm",
    }
)


class YiConfig(LlamaConfig):
    """Original ``model_type: "Yi"`` checkpoints, before 01-ai's Llama-format conversion.

    The math is native Llama; defaults follow the Yi source config and the
    decoder norms are serialized as ``ln1``/``ln2`` (see ``YI_KEY_MAPPING``).
    """

    model_type = "yi"
    default_theta = 5_000_000.0
    vocab_size: int = 64000
    num_key_value_heads: int | None = 4
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    pad_token_id: int | None = 0


YI_KEY_MAPPING = {r"\.ln1\.": ".input_layernorm.", r"\.ln2\.": ".post_attention_layernorm."}


class StableLMEpochConfig(StableLmConfig):
    """The original sequential StableLM layout, implemented natively in TF5.

    Stability's Epoch source uses LayerNorm, partial RoPE, GQA with optional
    q/k/v bias and SiLU. Its rotary_scaling_factor/num_heads metadata is not read.
    """

    model_type = "stablelm_epoch"

    def __init__(self, rope_pct=0.25, norm_eps=1e-5, use_qkv_bias=False, **kwargs):
        self.rope_pct = rope_pct
        self.norm_eps = norm_eps
        kwargs["partial_rotary_factor"] = rope_pct
        kwargs["layer_norm_eps"] = norm_eps
        kwargs.setdefault("eos_token_id", 2)
        # Later Epoch sources (StableLM-2 release) read use_qkv_bias from config;
        # earlier ones hardcode bias-free q/k/v and their configs omit the field.
        kwargs["use_qkv_bias"] = use_qkv_bias
        # These fields did not select alternate execution in the Epoch source.
        kwargs.update(
            hidden_act="silu",
            qk_layernorm=False,
            use_parallel_residual=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        super().__init__(**kwargs)


def _config_class(model_type):
    # Original Yi configs spell it "Yi"; registry lookup is case-insensitive.
    model_type = model_type.lower() if isinstance(model_type, str) else model_type
    if model_type == "longllama":
        return LongLlamaConfig
    if model_type == "stablelm_epoch":
        return StableLMEpochConfig
    if model_type == "yi":
        return YiConfig
    if model_type in NATIVE_TYPES:
        return type(AutoConfig.for_model(model_type))
    raise ValueError(f"Unsupported Llama definition model_type: {model_type!r}")


def _model_class(config):
    expected = _config_class(config.model_type)
    if type(config) is not expected:
        raise ValueError(f"Expected package-selected {expected.__name__}, got {type(config).__name__}")
    if expected is LongLlamaConfig:
        from .modeling_longllama import LongLlamaForCausalLM

        return LongLlamaForCausalLM
    if expected is StableLMEpochConfig:
        from transformers import StableLmForCausalLM

        return StableLmForCausalLM
    if expected is YiConfig:
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM
    return AutoModelForCausalLM._model_mapping[expected]


class LlamaAliasConfig:
    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        payload, remaining = PretrainedConfig.get_config_dict(model_id_or_path, **kwargs)
        config_class = _config_class(payload.get("model_type"))
        payload["model_type"] = config_class.model_type
        return config_class.from_dict(payload, **remaining)


class LlamaAliasForCausalLM:
    @classmethod
    def _from_config(cls, config, **kwargs):
        return _model_class(config)._from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id_or_path, *, config, **kwargs):
        if type(config) is YiConfig:
            kwargs.setdefault("key_mapping", YI_KEY_MAPPING)
        return _model_class(config).from_pretrained(model_id_or_path, config=config, **kwargs)
