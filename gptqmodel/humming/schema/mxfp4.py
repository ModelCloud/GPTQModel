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
class Mxfp4WeightSchema(BaseWeightSchema):
    quant_method: str = "mxfp4"

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        tensor_meta = {
            "weight": {
                "shape": (shape_n, shape_k // 2),
                "dtype": torch.uint8,
                "extra_attrs": {"output_dim": 0, "input_dim": 1},
            },
            "weight_scale": {
                "shape": (shape_n, shape_k // 32),
                "dtype": torch.uint8,
                "extra_attrs": {"output_dim": 0, "input_dim": 1, "scale_type": "group"},
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
        shape_n = tensors["weight"].size(-2)
        shape_k = tensors["weight"].size(-1) * 2
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
        schema = HummingWeightSchema(
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e8m0,
            weight_scale_group_size=32,
        )

        weight = tensors["weight"].view(torch.int32)
        weight_scale = tensors["weight_scale"]
        weight_scale = weight_scale.view(torch.float8_e8m0fnu)
        output_tensors = {"weight": weight, "weight_scale": weight_scale}

        if "bias" in tensors:
            output_tensors["bias"] = tensors["bias"]

        return schema, output_tensors
