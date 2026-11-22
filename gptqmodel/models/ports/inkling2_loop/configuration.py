# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Configuration objects for Inkling 2 Mini (``inkling2_loop``).

Written from the released checkpoint: every field name, default and alias below
matches ``thinkingmachines/Inkling-2-Mini/config.json`` so the architecture can
be instantiated straight from the checkpoint config, and the alias table keeps
the short ecosystem names (``num_local_experts``, ``intermediate_size``, ...)
working for callers that use them.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from transformers.configuration_utils import PreTrainedConfig


#: File the released checkpoint ships next to ``config.json``. It is the only
#: place that records which routed-expert layers are NVFP4 and which are MXFP8.
QUANT_MANIFEST = "hf_quant_config.json"

#: ``output_dtype`` value the manifest uses for a packed NVFP4 weight.
_NVFP4_OUTPUT = "uint8"
#: ``output_dtype`` value the manifest uses for an MXFP8 weight.
_MXFP8_OUTPUT = "float8_e4m3fn"


class Inkling2LoopVisionConfig(PreTrainedConfig):
    """Vision backbone: a ViT with axial RoPE, layer scale and a merger MLP."""

    model_type = "inkling2_loop_vision"

    attribute_map = {
        "native_patch_size": "patch_size",
        "adapter_merge_size": "spatial_merge_size",
    }

    hidden_size: int = 1024
    intermediate_size: Optional[int] = None
    num_hidden_layers: int = 23
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 20
    spatial_merge_size: int = 2
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    mlp_ratio: float = 4.0
    layer_scale_init_value: float = 0.1
    posemb_grid_size: int = 32
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    text_hidden_size: int = 2304
    rope_theta: float = 10000.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * self.mlp_ratio)
        super().__post_init__(**kwargs)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class Inkling2LoopTextConfig(PreTrainedConfig):
    """Language model: fixed loop core, hybrid sliding/full attention, routed MoE.

    ``layer_types`` marks the layers that attend over the whole context; every
    other layer only sees ``sliding_window_size`` tokens back. Layers inside
    ``[loop_layer_index_start, loop_layer_index_end)`` are visited
    ``loop_num_passes`` times.
    """

    model_type = "inkling2_loop_text"

    attribute_map = {
        "sliding_window": "sliding_window_size",
        "num_local_experts": "n_routed_experts",
        "sconv_kernel_size": "conv_kernel_size",
        "model_max_length": "max_position_embeddings",
        "intermediate_size": "moe_intermediate_size",
        "routed_scaling_factor": "route_scale",
    }

    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None
    swa_num_attention_heads: Optional[int] = None
    swa_num_key_value_heads: Optional[int] = None
    swa_head_dim: Optional[int] = None
    sliding_window_size: int = 1024
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    layer_types: Optional[list[str]] = None
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    conv_kernel_size: int = 4
    hidden_act: str = "silu"
    moe_intermediate_size: int = 1152
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    route_scale: float = 8.0
    scoring_func: str = "sigmoid"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = 200006
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    n_layers_per_attn_res_block: int = 6
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float = 30.0
    use_cache: bool = True
    number_of_conv_states: int = 4
    loop_layer_index_start: int = 6
    loop_layer_index_end: int = 18
    loop_num_passes: int = 3
    loop_gate_rank: int = 64
    # ``"nvfp4"`` / ``"mxfp8"`` per layer for the routed experts, or None for a
    # dense (bf16) checkpoint. Filled from the checkpoint manifest on load.
    experts_formats: Optional[list[Optional[str]]] = None

    def __post_init__(self, **kwargs):
        loop_config = kwargs.pop("loop_config", None)
        if isinstance(loop_config, dict):
            self.loop_layer_index_start = int(
                loop_config.get("core_start", self.loop_layer_index_start)
            )
            self.loop_layer_index_end = int(loop_config.get("core_end", self.loop_layer_index_end))
            self.loop_num_passes = int(loop_config.get("num_passes", self.loop_num_passes))
            self.loop_gate_rank = int(loop_config.get("gate_rank", self.loop_gate_rank))

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.swa_num_attention_heads is None:
            self.swa_num_attention_heads = self.num_attention_heads
        if self.swa_num_key_value_heads is None:
            self.swa_num_key_value_heads = self.num_key_value_heads
        if self.swa_head_dim is None:
            self.swa_head_dim = self.head_dim
        if self.layer_types is None:
            block = self.n_layers_per_attn_res_block
            self.layer_types = [
                "hybrid" if (index + 1) % block == 0 else "hybrid_sliding"
                for index in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

        if self.loop_num_passes < 1:
            raise ValueError(f"`loop_num_passes` ({self.loop_num_passes}) must be at least 1")
        if self.loop_layer_index_start <= 0:
            raise ValueError("`loop_layer_index_start` must be greater than 0")
        for name in ("loop_layer_index_start", "loop_layer_index_end"):
            value = getattr(self, name)
            if value % self.n_layers_per_attn_res_block:
                raise ValueError(
                    f"`{name}` ({value}) must be a multiple of `n_layers_per_attn_res_block` "
                    f"({self.n_layers_per_attn_res_block})"
                )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` has {len(self.layer_types)} entries for "
                f"{self.num_hidden_layers} layers"
            )

    @property
    def total_num_hidden_layers(self) -> int:
        """Decoder layers visited per forward pass, looped layers counted once per pass."""

        looped = self.loop_layer_index_end - self.loop_layer_index_start
        return self.num_hidden_layers + (self.loop_num_passes - 1) * looped

    def is_sliding_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "hybrid_sliding"

    def is_loop_layer(self, layer_idx: int) -> bool:
        return self.loop_layer_index_start <= layer_idx < self.loop_layer_index_end

    def experts_format(self, layer_idx: int) -> Optional[str]:
        formats = self.experts_formats
        if not formats or layer_idx >= len(formats):
            return None
        return formats[layer_idx]


class Inkling2LoopConfig(PreTrainedConfig):
    """Wrapper config: a vision backbone plus the looped language model."""

    model_type = "inkling2_loop"

    sub_configs_defaults = {
        "text_config": Inkling2LoopTextConfig,
        "vision_config": Inkling2LoopVisionConfig,
    }

    text_config: Inkling2LoopTextConfig | dict | None = None
    vision_config: Inkling2LoopVisionConfig | dict | None = None
    image_token_id: int = 200054

    def __post_init__(self, **kwargs):
        # The released checkpoint carries a modelopt `quantization_config` that
        # Transformers does not understand. It describes the already-packed
        # routed experts; keeping it would only make the loader warn and skip.
        kwargs.pop("quantization_config", None)
        super().__post_init__(**kwargs)
        if isinstance(self.vision_config, dict):
            self.vision_config = Inkling2LoopVisionConfig(
                **{k: v for k, v in self.vision_config.items() if k != "model_type"}
            )
        if isinstance(self.text_config, dict):
            self.text_config = Inkling2LoopTextConfig(
                **{k: v for k, v in self.text_config.items() if k != "model_type"}
            )
        if self.vision_config.text_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"`vision_config.text_hidden_size` ({self.vision_config.text_hidden_size}) "
                f"must match `text_config.hidden_size` ({self.text_config.hidden_size})"
            )
        # Callers constructing the local config through `from_dict` need the
        # same manifest lookup as callers using `from_pretrained`.
        if self.text_config.experts_formats is None and self.name_or_path:
            formats = read_expert_formats(
                self.name_or_path, num_layers=self.text_config.num_hidden_layers
            )
            if formats is not None:
                self.text_config.experts_formats = formats

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        loaded = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config = loaded[0] if isinstance(loaded, tuple) else loaded
        formats = read_expert_formats(
            pretrained_model_name_or_path,
            subfolder=kwargs.get("subfolder"),
            num_layers=config.text_config.num_hidden_layers,
        )
        if formats is not None and config.text_config.experts_formats is None:
            config.text_config.experts_formats = formats
        return loaded


def _manifest_path(model_path: str, subfolder: Optional[str] = None) -> Optional[str]:
    if not isinstance(model_path, (str, os.PathLike)) or not os.path.isdir(model_path):
        return None
    root = os.path.join(model_path, subfolder) if subfolder else model_path
    candidate = os.path.join(root, QUANT_MANIFEST)
    return candidate if os.path.isfile(candidate) else None


def read_expert_formats(model_path: str, num_layers: int, **kwargs) -> Optional[list[Optional[str]]]:
    """Read the routed-expert storage format per layer from the checkpoint manifest.

    Returns ``None`` when the checkpoint has no manifest (dense experts). Each
    entry is ``"nvfp4"``, ``"mxfp8"`` or ``None`` for a layer whose experts are
    stored in full precision.
    """

    path = _manifest_path(model_path, kwargs.get("subfolder"))
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    formats: list[Optional[str]] = [None] * num_layers
    for group in manifest.get("model", {}).get("parameter_groups", []):
        output_dtype = group.get("output_dtype")
        if output_dtype == _NVFP4_OUTPUT:
            kind = "nvfp4"
        elif output_dtype == _MXFP8_OUTPUT:
            kind = "mxfp8"
        else:
            continue
        for tensor in group.get("tensors", []):
            parts = tensor.split(".")
            if len(parts) < 6 or parts[-1] != "w13_weight" or "experts" not in parts:
                continue
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            if 0 <= layer_idx < num_layers:
                formats[layer_idx] = kind
    return formats


__all__ = [
    "Inkling2LoopConfig",
    "Inkling2LoopTextConfig",
    "Inkling2LoopVisionConfig",
    "read_expert_formats",
]
