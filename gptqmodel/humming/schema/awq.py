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
class AWQWeightSchema(BaseWeightSchema):
    quant_method: str = "awq"
    bits: int
    group_size: int
    zero_point: bool = True

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
                "shape": (shape_k, shape_n * num_bits // 32),
                "dtype": torch.int32,
                "extra_attrs": {
                    "input_dim": 0,
                    "output_dim": 1,
                    "packed_dim": 1,
                    "packed_factor": 8,
                },
            },
            "scales": {
                "shape": (shape_k // group_size, shape_n),
                "dtype": param_dtype,
                "extra_attrs": {"input_dim": 0, "output_dim": 1, "scale_type": "group"},
            },
            "qzeros": {
                "shape": (shape_k // group_size, shape_n * num_bits // 32),
                "dtype": torch.int32,
                "extra_attrs": {
                    "input_dim": 0,
                    "output_dim": 1,
                    "packed_dim": 1,
                    "packed_factor": 8,
                },
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
        shape_n = tensors["qweight"].size(-1) * 32 // self.bits
        shape_k = tensors["qweight"].size(-2)
        has_bias = "bias" in tensors
        return shape_n, shape_k, None, has_bias

    def _unpack_and_uninterleave(self, tensor: torch.Tensor):
        num_experts = tensor.size(0) if tensor.ndim == 3 else None
        tensor = tensor.cuda()
        tensor = ops.unpack_weight(tensor, self.bits)
        tensor = tensor.view(*tensor.shape[:-1], -1, 8)[..., [0, 4, 1, 5, 2, 6, 3, 7]]
        tensor = tensor.view(*tensor.shape[:-2], -1)
        if num_experts is not None:
            tensor = tensor.view(num_experts, -1, tensor.size(-1))

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
            b_dtype=dtypes.DataType.from_str(f"uint{self.bits}"),
            weight_scale_group_size=self.group_size,
            has_zero_point=self.zero_point,
        )

        weight = tensors["qweight"]
        weight = self._unpack_and_uninterleave(weight)
        weight = weight.transpose(-1, -2).contiguous()
        weight = ops.pack_weight(weight, self.bits)

        weight_scale = tensors["scales"].to(param_dtype)
        weight_scale = weight_scale.transpose(-1, -2).contiguous()

        output_tensors = {"weight": weight, "weight_scale": weight_scale}

        if self.zero_point:
            zero_point = tensors["qzeros"]
            zero_point = self._unpack_and_uninterleave(zero_point)
            zero_point = ops.pack_weight(zero_point, self.bits)
            zero_point = zero_point.transpose(-1, -2).contiguous()
            output_tensors["zero_point"] = zero_point

        if "bias" in tensors:
            output_tensors["bias"] = tensors["bias"]

        return schema, output_tensors
