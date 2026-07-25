# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import dataclasses
from typing import Any

import torch

from .. import dtypes
from .base import BaseWeightSchema
from .humming import HummingWeightSchema


@dataclasses.dataclass(kw_only=True)
class GPTQWeightSchema(BaseWeightSchema):
    quant_method: str = "gptq"
    bits: int
    group_size: int
    desc_act: bool = False
    sym: bool = True

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        num_bits = self.bits
        group_size = self.group_size
        tensor_meta = {
            "qweight": {
                "shape": (shape_k * num_bits // 32, shape_n),
                "dtype": torch.int32,
                "extra_attrs": {
                    "output_dim": 1,
                    "input_dim": 0,
                    "packed_dim": 0,
                    "packed_factor": 32 / num_bits,
                },
            },
            "scales": {
                "shape": (shape_k // group_size, shape_n),
                "dtype": param_dtype,
                "extra_attrs": {"output_dim": 1, "input_dim": 0, "scale_type": "group"},
            },
            "qzeros": {
                "shape": (shape_k // group_size, shape_n * num_bits // 32),
                "dtype": torch.int32,
                "extra_attrs": {
                    "output_dim": 1,
                    "input_dim": 0,
                    "packed_dim": 1,
                    "packed_factor": 32 / num_bits,
                    "scale_type": "group",
                },
            },
            "g_idx": {
                "shape": (shape_k,),
                "dtype": torch.int32,
                "extra_attrs": {"input_dim": 0},
            },
        }

        if has_bias:
            tensor_meta["bias"] = {
                "shape": (shape_n,),
                "dtype": param_dtype,
                "extra_attrs": {"output_dim": 0},
            }

        self.may_add_expert_dim(tensor_meta, num_experts)
        return tensor_meta

    def infer_shape(self, tensors: dict[str, torch.Tensor]) -> tuple[int, int, int | None, bool]:
        shape_n = tensors["qweight"].size(-1)
        shape_k = tensors["qweight"].size(-2) * 32 // self.bits
        has_bias = "bias" in tensors
        return shape_n, shape_k, None, has_bias

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple[HummingWeightSchema, dict[str, torch.Tensor]]:
        assert not self.desc_act
        schema = HummingWeightSchema(
            b_dtype=dtypes.DataType.from_str(f"uint{self.bits}"),
            weight_scale_group_size=self.group_size,
            has_zero_point=not self.sym,
        )

        weight = tensors["qweight"]
        weight = weight.transpose(-1, -2).contiguous()
        weight_scale = tensors["scales"].to(param_dtype)
        weight_scale = weight_scale.transpose(-1, -2).contiguous()

        output_tensors = {"weight": weight, "weight_scale": weight_scale}

        if not self.sym:
            zero_point = tensors["qzeros"]
            zero_point = zero_point.transpose(-1, -2).contiguous()
            output_tensors["zero_point"] = zero_point

        if "bias" in tensors:
            output_tensors["bias"] = tensors["bias"]

        return schema, output_tensors
