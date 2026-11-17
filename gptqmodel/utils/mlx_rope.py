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
#: ``rope_parameters`` directly, including its per-layer-type entries.
#: Everything else only knows the flat ``rope_theta`` / ``rope_scaling`` pair,
#: so a nested-only config silently runs with theta 10000 and no scaling (issue
#: #4607). ``gemma4_text`` belongs here because mlx-lm selects
#: ``rope_parameters[layer_key]`` itself; flattening it would lose the split.
_MLX_NATIVE_ROPE_PARAMETERS = frozenset({"qwen3", "gemma4_text"})

#: Architectures whose mlx-lm ``ModelArgs`` express a per-layer-type rotary
#: split as a dedicated global/local base pair instead of ``rope_parameters``.
#: Maps ``model_type`` to ``(global_layer_key, local_layer_key)``: the global
#: key feeds ``rope_theta`` / ``rope_scaling`` and the local key feeds
#: ``rope_local_base_freq``. Gemma 3 and Gemma 3n are the transformers-5
#: families with that split (issue #4678).
_PER_LAYER_ROPE_SPLIT = {
    "gemma3_text": ("full_attention", "sliding_attention"),
    "gemma3n_text": ("full_attention", "sliding_attention"),
}

_ROPE_TYPE_KEYS = ("rope_type", "type")


def _drop_default_rope_type(rope: dict) -> dict:
    """Return ``rope`` without the no-op markers mlx-lm treats as "no scaling"."""

    scaling = dict(rope)
    if scaling.get("rope_type", scaling.get("type")) in (None, "default"):
        for key in _ROPE_TYPE_KEYS:
            scaling.pop(key, None)
    return scaling


def _split_local_global_rope(rope: dict, *, model_type: str):
    """Map a per-layer-type rotary split onto mlx-lm's local/global pair.

    mlx-lm builds the global layers from ``rope_theta`` plus ``rope_scaling``
    and the local layers from ``rope_local_base_freq`` alone, so a local scaling
    config has no destination there and is rejected rather than dropped. Returns
    ``(theta, scaling, local_theta)``.
    """

    global_key, local_key = _PER_LAYER_ROPE_SPLIT[model_type]
    unexpected = sorted(set(rope) - {global_key, local_key})
    if unexpected:
        raise ValueError(
            f"MLX: `{model_type}` uses per-layer-type RoPE parameters with layer "
            f"types {unexpected} that mlx-lm's `{global_key}` / `{local_key}` "
            "split cannot represent; refusing to export a config that would "
            "silently fall back to rope_theta=10000."
        )
    global_rope = rope.get(global_key) or {}
    local_rope = rope.get(local_key) or {}
    if not isinstance(global_rope, dict) or not isinstance(local_rope, dict):
        raise ValueError(
            f"MLX: `{model_type}` has a non-mapping `{global_key}` / `{local_key}` "
            "rope_parameters entry; expected a per-layer-type mapping."
        )
    local_scaling = _drop_default_rope_type(
        {key: value for key, value in local_rope.items() if key != "rope_theta"}
    )
    if local_scaling:
        raise ValueError(
            f"MLX: `{model_type}` applies rope scaling {sorted(local_scaling)} to "
            f"its `{local_key}` layers, but mlx-lm builds those layers from "
            "`rope_local_base_freq` alone; refusing to export a config that would "
            "silently drop the local scaling."
        )
    theta = global_rope.get("rope_theta")
    scaling = _drop_default_rope_type(
        {key: value for key, value in global_rope.items() if key != "rope_theta"}
    )
    return theta, scaling, local_rope.get("rope_theta")


def _flatten_rope_parameters(rope: dict, *, model_type: str | None):
    """Return ``(theta, scaling)`` from one transformers-5 rope mapping.

    Per-layer-type mappings (Gemma's ``full_attention`` / ``sliding_attention``)
    collapse when every layer type asks for the same rotary setup; otherwise the
    difference cannot be represented in mlx-lm's single ``rope_theta`` /
    ``rope_scaling`` pair and is rejected instead of silently defaulted.
    Architectures with a dedicated local/global pair are handled by
    :func:`_split_local_global_rope` before this function runs.
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
    scaling = _drop_default_rope_type(
        {key: value for key, value in rope.items() if key != "rope_theta"}
    )
    return theta, scaling


def normalize_mlx_rope_config(config: dict, *, where: str = "config") -> dict:
    """Translate transformers-5 ``rope_parameters`` into what mlx-lm reads.

    transformers 5 stores the rotary setup as ``rope_parameters`` and omits the
    top-level ``rope_theta`` / ``rope_scaling`` pair. Most mlx-lm architectures
    only read the flat pair, so forwarding the dict unchanged silently changes
    the positional encoding (issue #4607). This copies the theta and the
    non-default scaling fields to the flat keys, keeps ``rope_parameters`` for
    the architectures that read it, and warns when no theta is available.

    A per-layer-type split is mapped per architecture instead of being refused:
    Gemma 3 and Gemma 3n get the global base in ``rope_theta`` (plus its scaling
    in ``rope_scaling``) and the sliding base in ``rope_local_base_freq``, which
    is the pair their mlx-lm ``ModelArgs`` reads (issue #4678). Layer types or
    local scaling a mapping cannot represent are still rejected.

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
        local_theta = None
        per_layer = any(isinstance(value, dict) for value in rope.values())
        if per_layer and model_type in _PER_LAYER_ROPE_SPLIT:
            theta, scaling, local_theta = _split_local_global_rope(
                rope, model_type=model_type
            )
        else:
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
        if local_theta is not None:
            config["rope_local_base_freq"] = float(local_theta)

    if isinstance(text_config, dict) and isinstance(text_config.get("rope_parameters"), dict):
        config["text_config"] = normalize_mlx_rope_config(
            text_config, where=f"{where}.text_config"
        )
    return config


def _mlx_model_config(config: dict) -> dict:
    """Deep-copy ``config`` with the RoPE fields mlx-lm reads (issue #4607)."""

    return normalize_mlx_rope_config(deepcopy(config))


