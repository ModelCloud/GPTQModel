# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Translate transformers-5 RoPE configuration for mlx-lm.

transformers 5 stores the rotary setup as ``rope_parameters`` and omits the
top-level ``rope_theta`` / ``rope_scaling`` pair. Most mlx-lm architectures only
read the flat pair, so forwarding the nested dict unchanged silently changes the
positional encoding (issue #4607). This module owns the translation and has no
mlx dependency, so it is importable and testable on any host.
"""

from copy import deepcopy

from .logger import setup_logger


log = setup_logger()

#: mlx-lm architectures whose ``ModelArgs`` read transformers-5
#: ``rope_parameters`` directly. Everything else only knows the flat
#: ``rope_theta`` / ``rope_scaling`` pair, so a nested-only config silently runs
#: with theta 10000 and no scaling (issue #4607).
_MLX_NATIVE_ROPE_PARAMETERS = frozenset({"qwen3"})

_ROPE_TYPE_KEYS = ("rope_type", "type")


def _flatten_rope_parameters(rope: dict, *, model_type: str | None):
    """Return ``(theta, scaling)`` from one transformers-5 rope mapping.

    Per-layer-type mappings (Gemma's ``full_attention`` / ``sliding_attention``)
    collapse when every layer type asks for the same rotary setup; otherwise the
    difference cannot be represented in mlx-lm's single ``rope_theta`` /
    ``rope_scaling`` pair and is rejected instead of silently defaulted.
    """

    nested = {key: value for key, value in rope.items() if isinstance(value, dict)}
    if nested:
        first = next(iter(nested.values()))
        if len(nested) != len(rope) or any(value != first for value in nested.values()):
            raise ValueError(
                f"MLX: `{model_type or 'this model'}` uses per-layer-type RoPE "
                "parameters that mlx-lm cannot represent; refusing to export a "
                "config that would silently fall back to rope_theta=10000. "
                f"Got {sorted(nested)}."
            )
        rope = first

    theta = rope.get("rope_theta")
    scaling = {key: value for key, value in rope.items() if key != "rope_theta"}
    rope_type = scaling.get("rope_type", scaling.get("type"))
    if rope_type in (None, "default"):
        for key in _ROPE_TYPE_KEYS:
            scaling.pop(key, None)
    return theta, scaling


def normalize_mlx_rope_config(config: dict, *, where: str = "config") -> dict:
    """Translate transformers-5 ``rope_parameters`` into what mlx-lm reads.

    transformers 5 stores the rotary setup as ``rope_parameters`` and omits the
    top-level ``rope_theta`` / ``rope_scaling`` pair. Most mlx-lm architectures
    only read the flat pair, so forwarding the dict unchanged silently changes
    the positional encoding (issue #4607). This copies the theta and the
    non-default scaling fields to the flat keys, keeps ``rope_parameters`` for
    the architectures that read it, and warns when no theta is available.

    The mapping is applied in place and the same dict is returned; callers that
    must not mutate the source use :func:`_mlx_model_config`.
    """

    if not isinstance(config, dict):
        return config

    model_type = config.get("model_type")
    text_config = config.get("text_config")
    if not model_type and isinstance(text_config, dict):
        model_type = text_config.get("model_type")
    if model_type in _MLX_NATIVE_ROPE_PARAMETERS:
        return config

    rope = config.get("rope_parameters")
    if isinstance(rope, dict) and rope:
        theta, scaling = _flatten_rope_parameters(rope, model_type=model_type)
        if theta is None:
            log.warning(
                "MLX: %s has rope_parameters without rope_theta; mlx-lm will use "
                "its own default (10000). Set rope_theta explicitly to keep the "
                "checkpoint's positional encoding.",
                where,
            )
        else:
            config["rope_theta"] = float(theta)
        if scaling:
            config["rope_scaling"] = scaling

    if isinstance(text_config, dict) and isinstance(text_config.get("rope_parameters"), dict):
        config["text_config"] = normalize_mlx_rope_config(
            text_config, where=f"{where}.text_config"
        )
    return config


def _mlx_model_config(config: dict) -> dict:
    """Deep-copy ``config`` with the RoPE fields mlx-lm reads (issue #4607)."""

    return normalize_mlx_rope_config(deepcopy(config))


