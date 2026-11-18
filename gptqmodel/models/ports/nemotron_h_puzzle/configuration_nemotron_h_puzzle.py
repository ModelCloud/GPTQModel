# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Puzzle's per-block geometry, adapted to the native Transformers 5 primitives."""

from copy import deepcopy
from typing import Any

from ..native.nemotron_h.configuration_nemotron_h import NemotronHConfig


_BLOCK_TYPES = {"mamba": "linear_attention", "attention": "full_attention", "moe": "moe"}
_MOE_FIELDS = ("moe_intermediate_size", "num_experts_per_tok")


def _blocks(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = deepcopy(values)
    for block in result:
        if not isinstance(block, dict) or block.get("block_type") not in _BLOCK_TYPES:
            raise ValueError("Puzzle block_configs must contain mamba, attention or moe dictionaries")
        allowed = {"block_type", *_MOE_FIELDS} if block["block_type"] == "moe" else {"block_type"}
        if set(block) - allowed:
            raise ValueError(f"Unknown Puzzle block fields: {set(block) - allowed}")
        for field in _MOE_FIELDS:
            if block.get(field) is not None and (not isinstance(block[field], int) or block[field] < 1):
                raise ValueError(f"{field} must be a positive integer")
    return result


class NemotronHPuzzleConfig(NemotronHConfig):
    model_type = "nemotron_h_puzzle"

    def __init__(self, block_configs=None, mtp_block_configs=None, **kwargs):
        if block_configs is None:
            legacy = kwargs.get("layers_block_type", kwargs.get("layer_types"))
            if legacy is None:
                pattern = kwargs.get("hybrid_override_pattern", "ME*E")
                legacy = [{"M": "mamba", "E": "moe", "*": "attention"}[c] for c in pattern]
            reverse = {v: k for k, v in _BLOCK_TYPES.items()}
            block_configs = [{"block_type": reverse.get(t, t)} for t in legacy]
        self.block_configs = _blocks(block_configs)
        self.mtp_block_configs = _blocks(mtp_block_configs or [])
        if not self.block_configs:
            raise ValueError("Puzzle requires at least one decoder block")
        if self.mtp_block_configs and [b["block_type"] for b in self.mtp_block_configs] != ["attention", "moe"]:
            raise ValueError("Puzzle MTP blocks must be attention followed by moe")
        for field in _MOE_FIELDS:
            statuses = {b.get(field) is not None for b in self.block_configs + self.mtp_block_configs
                        if b["block_type"] == "moe"}
            if len(statuses) > 1:
                raise ValueError(f"Block field '{field}' must be set for all MoE blocks or left global")
        kwargs.pop("blockwise_members", None)
        kwargs.pop("block_type", None)
        kwargs.pop("hybrid_override_pattern", None)
        kwargs.pop("layers_block_type", None)
        kwargs.pop("layer_types", None)
        kwargs.pop("num_hidden_layers", None)
        kwargs["layers_block_type"] = [_BLOCK_TYPES[b["block_type"]] for b in self.block_configs]
        if self.mtp_block_configs:
            kwargs["num_nextn_predict_layers"] = 1
        super().__init__(**kwargs)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.mamba_num_heads % self.n_groups:
            raise ValueError("mamba_num_heads must be divisible by n_groups")
        if self.n_routed_experts % self.n_group or self.n_routed_experts // self.n_group < 2:
            raise ValueError("Puzzle routing requires at least two experts per group")
        if not 1 <= self.topk_group <= self.n_group:
            raise ValueError("topk_group must be within n_group")
        for i, block in enumerate(self.block_configs + self.mtp_block_configs):
            if block["block_type"] == "moe":
                k = block.get("num_experts_per_tok") or self.num_experts_per_tok
                if not 1 <= k <= self.n_routed_experts:
                    raise ValueError(f"Invalid num_experts_per_tok for block {i}")

    def get_nemotron_h_config_for_layer(self, layer_idx):
        blocks = self.block_configs + self.mtp_block_configs
        if not 0 <= layer_idx < len(blocks):
            raise IndexError(f"Puzzle layer {layer_idx} is out of bounds")
        values = super().to_dict()
        for key in ("block_configs", "mtp_block_configs", "blockwise_members", "model_type", "auto_map"):
            values.pop(key, None)
        values.update({k: v for k, v in blocks[layer_idx].items() if k != "block_type" and v is not None})
        config = NemotronHConfig.from_dict(values)
        config._attn_implementation = self._attn_implementation
        return config

    def to_dict(self):
        values = super().to_dict()
        values["block_configs"] = deepcopy(self.block_configs)
        values["mtp_block_configs"] = deepcopy(self.mtp_block_configs)
        # Preserve the published schema, not native Nemotron-H's homogeneous metadata.
        values["layers_block_type"] = [b["block_type"] for b in self.block_configs]
        values.pop("layer_types", None)
        for field in _MOE_FIELDS:
            if any(b.get(field) is not None for b in self.block_configs + self.mtp_block_configs):
                values.pop(field, None)
        return values
