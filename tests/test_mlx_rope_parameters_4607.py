# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""transformers-5 RoPE configuration must survive the MLX translation (#4607).

transformers 5 saves the rotary setup as ``rope_parameters`` and omits the
top-level ``rope_theta`` / ``rope_scaling`` pair. Most mlx-lm architectures only
read the flat pair, so forwarding the config unchanged silently runs the
exported model with ``rope_theta=10000`` and no scaling. No MLX runtime is
required here: the translation is pure config manipulation.
"""

import json
from pathlib import Path

import pytest
from transformers import Gemma2Config, LlamaConfig, Qwen2Config

from gptqmodel.utils.mlx_rope import _mlx_model_config, normalize_mlx_rope_config


ROOT = Path(__file__).resolve().parents[1]


def _llama_3_2() -> dict:
    """Llama-3.2 geometry: theta 500000 with llama3 rope scaling."""

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
    )
    return config.to_dict()


def _qwen2() -> dict:
    return Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
    ).to_dict()


def _gemma2() -> dict:
    return Gemma2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        head_dim=16,
        max_position_embeddings=8192,
        rope_theta=10000.0,
    ).to_dict()


def test_llama3_rope_theta_and_scaling_reach_the_flat_mlx_keys():
    source = _llama_3_2()
    assert "rope_theta" not in source and source["rope_parameters"]["rope_theta"] == 500000.0

    exported = _mlx_model_config(source)

    assert exported["rope_theta"] == 500000.0
    assert exported["rope_theta"] != 10000
    assert exported["rope_scaling"] == {
        "rope_type": "llama3",
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    }
    # mlx-lm models that read the nested form keep seeing it.
    assert exported["rope_parameters"] == source["rope_parameters"]
    # The source config is not mutated: export must not rewrite the checkpoint.
    assert "rope_theta" not in source


def test_qwen2_keeps_its_non_default_theta():
    source = _qwen2()
    exported = _mlx_model_config(source)

    assert exported["rope_theta"] == 1000000.0
    assert exported["rope_theta"] != 10000
    assert "rope_scaling" not in exported


def test_gemma2_theta_is_the_configured_value_not_a_fallback():
    source = _gemma2()
    assert source["rope_parameters"]["rope_theta"] == 10000.0

    exported = _mlx_model_config(source)

    assert exported["rope_theta"] == 10000.0
    # A plain rotary setup carries no scaling marker for mlx-lm to trip over.
    assert "rope_scaling" not in exported


def test_gemma3_text_config_is_normalized_too():
    config = {
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        },
    }

    exported = _mlx_model_config(config)

    assert exported["text_config"]["rope_theta"] == 1000000.0
    assert exported["text_config"]["rope_parameters"]["rope_theta"] == 1000000.0


def test_uniform_per_layer_rope_parameters_collapse():
    rope = {
        "full_attention": {"rope_theta": 500000.0, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 500000.0, "rope_type": "default"},
    }
    exported = normalize_mlx_rope_config({"model_type": "gemma3_text", "rope_parameters": rope})

    assert exported["rope_theta"] == 500000.0
    assert exported["rope_parameters"] == rope


def test_conflicting_per_layer_rope_parameters_are_rejected():
    """An architecture with no local/global split still cannot be represented."""

    rope = {
        "full_attention": {"rope_theta": 500000.0},
        "sliding_attention": {"rope_theta": 10000.0},
    }

    with pytest.raises(ValueError, match="per-layer-type RoPE"):
        normalize_mlx_rope_config({"model_type": "mistral", "rope_parameters": rope})


def _gemma3_rope(*, full_scaling=None) -> dict:
    full = {"rope_type": "default", "rope_theta": 1000000}
    if full_scaling:
        full.update(full_scaling)
    return {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
        "full_attention": full,
    }


def test_gemma3_per_layer_rope_maps_to_mlx_lm_fields():
    """Gemma 3's global/local split is representable: theta + local base freq.

    mlx-lm's ``gemma3_text.ModelArgs`` reads ``rope_theta`` for the
    ``full_attention`` layers, ``rope_local_base_freq`` for the
    ``sliding_attention`` layers and applies ``rope_scaling`` to the global rope
    only. Refusing this layout broke every Gemma 3 export (#4678, regression of
    #4617).
    """

    exported = normalize_mlx_rope_config(
        {"model_type": "gemma3_text", "rope_parameters": _gemma3_rope()}
    )

    assert exported["rope_theta"] == 1000000.0
    assert exported["rope_local_base_freq"] == 10000.0
    # A plain rotary setup carries no scaling marker for mlx-lm to trip over.
    assert "rope_scaling" not in exported


def test_gemma3_full_attention_scaling_lands_on_rope_scaling():
    """4B+ models scale only the global rope; that is exactly mlx-lm's contract."""

    rope = _gemma3_rope(full_scaling={"rope_type": "linear", "factor": 8.0})
    exported = normalize_mlx_rope_config({"model_type": "gemma3_text", "rope_parameters": rope})

    assert exported["rope_theta"] == 1000000.0
    assert exported["rope_local_base_freq"] == 10000.0
    assert exported["rope_scaling"] == {"rope_type": "linear", "factor": 8.0}


def test_gemma3_vlm_text_config_is_mapped_through_the_recursion():
    config = {
        "model_type": "gemma3",
        "text_config": {"model_type": "gemma3_text", "rope_parameters": _gemma3_rope()},
    }

    exported = _mlx_model_config(config)["text_config"]

    assert exported["rope_theta"] == 1000000.0
    assert exported["rope_local_base_freq"] == 10000.0


def test_gemma3_local_scaling_is_refused_not_silently_dropped():
    """mlx-lm never scales the sliding layers, so a local scaling must not vanish."""

    rope = _gemma3_rope()
    rope["sliding_attention"] = {
        "rope_type": "linear",
        "factor": 2.0,
        "rope_theta": 10000,
    }

    with pytest.raises(ValueError, match="applies `rope_scaling` to the global rope only"):
        normalize_mlx_rope_config({"model_type": "gemma3_text", "rope_parameters": rope})


def test_missing_theta_warns_instead_of_silently_defaulting(caplog):
    config = {"model_type": "llama", "rope_parameters": {"rope_type": "linear", "factor": 2.0}}

    with caplog.at_level("WARNING"):
        exported = normalize_mlx_rope_config(config)

    assert "rope_theta" not in exported
    assert any("without rope_theta" in record.message for record in caplog.records)


def test_qwen3_keeps_the_native_nested_form():
    config = {
        "model_type": "qwen3",
        "rope_parameters": {"rope_theta": 5000000.0, "rope_type": "default"},
    }

    assert normalize_mlx_rope_config(config) == {
        "model_type": "qwen3",
        "rope_parameters": {"rope_theta": 5000000.0, "rope_type": "default"},
    }


def test_normalization_is_idempotent_and_serializable():
    once = _mlx_model_config(_llama_3_2())
    twice = normalize_mlx_rope_config(dict(once))

    assert twice == once
    json.dumps(twice)


def test_both_mlx_config_builders_use_the_translation():
    """Export and the Apple-Silicon runtime share one config builder."""

    source = (ROOT / "gptqmodel/utils/mlx.py").read_text(encoding="utf-8")
    assert source.count("mlx_config = _mlx_model_config(config)") == 2
    assert "quantize_model(mlx_model, mlx_config," in source


class _MlxLlamaArgs:
    """``mlx_lm.models.llama.ModelArgs`` reduced to its RoPE contract.

    mlx-lm builds its args with ``cls(**{k: v for k, v in params.items() if k in
    signature})`` and defaults ``rope_theta`` to 10000 with no scaling, so an
    unread key is a silent fallback rather than an error.
    """

    _FIELDS = ("rope_theta", "rope_scaling")

    def __init__(self, rope_theta=10000.0, rope_scaling=None):
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

    @classmethod
    def from_dict(cls, params):
        return cls(**{key: params[key] for key in cls._FIELDS if key in params})


def test_mlx_lm_consumes_the_translated_config():
    """The consumer-level invariant: the flat keys are what mlx-lm reads."""

    raw = _llama_3_2()
    # This is the defect: the transformers-5 config alone falls back silently.
    assert _MlxLlamaArgs.from_dict(raw).rope_theta == 10000.0
    assert _MlxLlamaArgs.from_dict(raw).rope_scaling is None

    args = _MlxLlamaArgs.from_dict(_mlx_model_config(raw))

    assert args.rope_theta == 500000.0
    assert args.rope_scaling == {
        "rope_type": "llama3",
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    }


class _MlxGemma3Args:
    """``mlx_lm.models.gemma3_text.ModelArgs`` reduced to its RoPE contract.

    Verified against upstream: the sliding layers are built with
    ``base=args.rope_local_base_freq`` and no scaling, the full-attention layers
    with ``base=args.rope_theta`` and ``scaling_config=args.rope_scaling``.
    """

    _FIELDS = ("rope_theta", "rope_local_base_freq", "rope_scaling")

    def __init__(self, rope_theta=1_000_000.0, rope_local_base_freq=10_000.0, rope_scaling=None):
        self.rope_theta = rope_theta
        self.rope_local_base_freq = rope_local_base_freq
        self.rope_scaling = rope_scaling

    @classmethod
    def from_dict(cls, params):
        return cls(**{key: params[key] for key in cls._FIELDS if key in params})

    def base_for_layer(self, *, sliding: bool):
        return self.rope_local_base_freq if sliding else self.rope_theta

    def scaling_for_layer(self, *, sliding: bool):
        return None if sliding else self.rope_scaling


def test_gemma3_config_reaches_the_mlx_lm_consumer():
    """Before the fix every Gemma 3 config raised; mlx-lm's defaults masked it.

    The un-translated config leaves mlx-lm on its 1e6/1e4 defaults, which happens
    to match a 1B checkpoint but silently drops the 4B+ global linear factor.
    """

    raw = {"model_type": "gemma3_text", "rope_parameters": _gemma3_rope(full_scaling={"rope_type": "linear", "factor": 8.0})}

    # The nested form carries no top-level key mlx-lm reads, so it silently keeps
    # its own defaults: a matching theta, and no global scaling at all.
    untouched = _MlxGemma3Args.from_dict(raw)
    assert untouched.base_for_layer(sliding=False) == 1_000_000.0
    assert untouched.scaling_for_layer(sliding=False) is None

    args = _MlxGemma3Args.from_dict(_mlx_model_config(raw))

    assert args.base_for_layer(sliding=True) == 10000.0
    assert args.base_for_layer(sliding=False) == 1000000.0
    assert args.scaling_for_layer(sliding=False) == {"rope_type": "linear", "factor": 8.0}
    # mlx-lm applies no scaling to the sliding layers.
    assert args.scaling_for_layer(sliding=True) is None
