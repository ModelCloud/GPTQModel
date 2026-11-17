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

#: Architectures whose per-layer-type RoPE mlx-lm *can* represent, mapped to the
#: ``(global, local, local_field)`` triple mlx-lm reads.
#:
#: ``mlx_lm.models.gemma3_text.ModelArgs`` exposes ``rope_theta`` (used by the
#: ``full_attention`` layers, with ``rope_scaling`` applied), ``rope_local_base_freq``
#: (used by the ``sliding_attention`` layers, never scaled) and ``rope_scaling``.
#: Refusing this layout made every Gemma 3 export and MLX load fail (issue #4678,
#: a regression of #4617).
_PER_LAYER_ROPE_MAPPINGS = {
    "gemma3_text": ("full_attention", "sliding_attention", "rope_local_base_freq"),
    "gemma3": ("full_attention", "sliding_attention", "rope_local_base_freq"),
}


def _split_scaling(rope: dict) -> dict:
    """Return the non-theta, non-default fields of one rope mapping."""

    scaling = {key: value for key, value in rope.items() if key != "rope_theta"}
    rope_type = scaling.get("rope_type", scaling.get("type"))
    if rope_type in (None, "default"):
        for key in _ROPE_TYPE_KEYS:
            scaling.pop(key, None)
    return scaling


def _flatten_per_layer_rope(
    nested: dict, *, model_type: str | None
) -> tuple[float | None, dict, dict] | None:
    """Map a local/global per-layer rope layout onto mlx-lm's flat keys.

    Returns ``(theta, scaling, extra)`` when ``model_type`` has a known mapping
    and ``nested`` carries exactly that architecture's layer types, else ``None``
    so the caller can fall back to the uniform-collapse / refusal path.
    """

    mapping = _PER_LAYER_ROPE_MAPPINGS.get(model_type or "")
    if mapping is None:
        return None
    global_key, local_key, local_field = mapping
    if set(nested) != {global_key, local_key}:
        return None

    global_rope = nested[global_key] or {}
    local_rope = nested[local_key] or {}

    # mlx-lm builds the sliding layers with no scaling config at all, so a
    # local scaling would be silently dropped. Refuse instead of mis-encoding.
    local_scaling = _split_scaling(local_rope)
    if local_scaling:
        raise ValueError(
            f"MLX: `{model_type}` asks for RoPE scaling on its "
            f"`{local_key}` layers ({sorted(local_scaling)}), which mlx-lm cannot "
            "represent; it applies `rope_scaling` to the global rope only."
        )

    local_theta = local_rope.get("rope_theta")
    extra = {}
    if local_theta is not None:
        extra[local_field] = float(local_theta)

    return global_rope.get("rope_theta"), _split_scaling(global_rope), extra


def _flatten_rope_parameters(rope: dict, *, model_type: str | None):
    """Return ``(theta, scaling, extra)`` from one transformers-5 rope mapping.

    Per-layer-type mappings collapse when every layer type asks for the same
    rotary setup. Architectures with a known local/global split (Gemma 3) are
    mapped onto mlx-lm's dedicated fields; anything else that cannot be
    represented in mlx-lm's single ``rope_theta`` / ``rope_scaling`` pair is
    rejected instead of silently defaulted.
    """

    nested = {key: value for key, value in rope.items() if isinstance(value, dict)}
    if nested:
        mapped = _flatten_per_layer_rope(nested, model_type=model_type)
        if mapped is not None:
            return mapped

        first = next(iter(nested.values()))
        if len(nested) != len(rope) or any(value != first for value in nested.values()):
            raise ValueError(
                f"MLX: `{model_type or 'this model'}` uses per-layer-type RoPE "
                "parameters that mlx-lm cannot represent; refusing to export a "
                "config that would silently fall back to rope_theta=10000. "
                f"Got {sorted(nested)}."
            )
        rope = first

    return rope.get("rope_theta"), _split_scaling(rope), {}


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
        theta, scaling, extra = _flatten_rope_parameters(rope, model_type=model_type)
        if theta is None:
            log.warning(
                "MLX: %s has rope_parameters without rope_theta; mlx-lm will use "
                "its own default (10000). Set rope_theta explicitly to keep the "
                "checkpoint's positional encoding.",
                where,
            )
        else:
            config["rope_theta"] = float(theta)
        for key, value in extra.items():
            config[key] = value
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


