# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import dataclasses
from typing import Any

import torch

from .. import dtypes, ops
from .base import BaseWeightSchema
from .humming import HummingWeightSchema


@dataclasses.dataclass(kw_only=True)
class BitnetWeightSchema(BaseWeightSchema):
    quant_method: str = "bitnet"

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
                "shape": (shape_n // 4, shape_k),
                "dtype": torch.uint8,
                "extra_attrs": {
                    "output_dim": 0,
                    "input_dim": 1,
                    "packed_dim": 0,
                    "packed_factor": 4,
                },
            },
            "weight_scale": {
                "shape": (stack_size,),
                "dtype": param_dtype,
                "extra_attrs": {"scale_type": "tensor"},
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
        shape_n = tensors["weight"].size(-2) * 4
        shape_k = tensors["weight"].size(-1)
        has_bias = "bias" in tensors
        return shape_n, shape_k, None, has_bias

    def process_loaded_weight(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if name == "weight":
            is_moe = tensor.ndim == 3
            e = tensor.size(0) if is_moe else 1
            tensor = tensor.cuda()
            tensor = tensor.transpose(-1, -2).contiguous().view(torch.int32)
            tensor = ops.unpack_weight(tensor, 2)
            tensor = tensor.view(e, tensor.size(-2), -1, 4)
            tensor = tensor.transpose(-1, -2).contiguous()
            tensor = tensor.view(e, tensor.size(-3), -1)
            tensor = ops.pack_weight(tensor, 2).view(torch.uint8)
            tensor = tensor.transpose(-1, -2).contiguous()
            tensor = tensor if is_moe else tensor.squeeze(0)

        return tensor

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple[HummingWeightSchema, dict[str, torch.Tensor]]:
        schema = HummingWeightSchema(
            b_dtype=dtypes.uint2,
            weight_scale_group_size=0,
        )

        weight = tensors["weight"]
        weight = weight.transpose(-1, -2).contiguous().view(torch.int32)
        weight = ops.unpack_weight(weight, 2) + 1
        weight = weight.transpose(-1, -2).contiguous()
        weight = ops.pack_weight(weight, 2)
        weight = weight.squeeze(0) if num_experts is None else weight

        weight_scale = tensors["weight_scale"]
        weight_scale = self._may_process_global_scale(
            1 / weight_scale,
            shape_n_stacks=shape_n_stacks,
            shape_k_stacks=shape_k_stacks,
            num_experts=num_experts,
            force_repeat=True,
        ).to(param_dtype)

        output_tensors = {"weight": weight, "weight_scale": weight_scale}

        if "bias" in tensors:
            output_tensors["bias"] = tensors["bias"]

        return schema, output_tensors
