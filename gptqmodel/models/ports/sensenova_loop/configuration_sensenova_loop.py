# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""SenseNova-Loop configuration: native Qwen3.5 plus a fixed Loop plan and optional mHC.

Adapted from `modeling_sensenova_loop.py` of sensenova/SenseNova-Loop-9B at
fa381f7c466739475e172a7d3234e77351e4f613 (Apache-2.0). The validation rules
are the upstream `execution_plan()` rules; this module only moves them to
config construction so an invalid checkpoint fails before any allocation.
"""

import math

from huggingface_hub.dataclasses import strict

from ..native.qwen3_5.configuration_qwen3_5 import Qwen3_5Config


_MHC_FIELDS = frozenset({"start", "end", "streams", "iterations", "fixed_identity", "pre_scale"})


def _is_int(value):
    return type(value) is int


def execution_plan(config):
    """Physical layer indices in execution order: prefix, `eval_count` loop visits, suffix."""

    loop = config.loop_settings
    if not isinstance(loop, dict):
        raise ValueError("SenseNova-Loop requires explicit loop_settings")
    text = config.text_config
    num_layers = text.num_hidden_layers
    start, end, count = loop.get("start"), loop.get("end"), loop.get("eval_count")
    if not all(_is_int(value) for value in (num_layers, start, end, count)) or not (
        0 <= start < end <= num_layers and count >= 1
    ):
        raise ValueError("Invalid SenseNova-Loop interval or eval_count")
    if loop.get("counts") != [count]:
        raise ValueError("Only fixed-count SenseNova-Loop checkpoints are supported")
    layer_types = list(text.layer_types)
    if len(layer_types) != num_layers or set(layer_types) - {"full_attention", "linear_attention"}:
        raise ValueError("Unexpected SenseNova-Loop physical layer types")
    if "full_attention" not in layer_types:
        raise ValueError("SenseNova-Loop requires a full-attention layer for sequence-length tracking")

    mhc = config.mhc
    if mhc is not None:
        if not isinstance(mhc, dict) or set(mhc) - _MHC_FIELDS:
            raise ValueError("Unknown mHC configuration fields")
        mhc_start, mhc_end, streams, iterations = (mhc.get(key) for key in ("start", "end", "streams", "iterations"))
        if (
            not all(_is_int(value) for value in (mhc_start, mhc_end, streams, iterations))
            or not 0 <= mhc_start < mhc_end <= num_layers
            or streams not in (2, 4)
            or iterations < 1
        ):
            raise ValueError("Invalid mHC range, stream count or iterations")
        if type(mhc.get("fixed_identity")) is not bool:
            raise ValueError("mHC fixed_identity must be a boolean")
        scale = mhc.get("pre_scale", 1.0)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid mHC pre_scale")
        boundaries = (0, start, end, num_layers)
        if not any(left <= mhc_start < mhc_end <= right for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("mHC cannot cross a SenseNova-Loop phase boundary")

    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))


@strict
class SenseNovaLoopConfig(Qwen3_5Config):
    """Qwen3.5 VL config with the released `loop_settings` and `mhc` blocks.

    Construction without `loop_settings` is allowed (Transformers builds a
    default instance to compute config diffs); the model rejects it.
    """

    model_type = "sensenova_loop"

    loop_settings: dict | None = None
    mhc: dict | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.loop_settings is not None:
            execution_plan(self)


__all__ = ["SenseNovaLoopConfig", "execution_plan"]
