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
from .base import BaseInputSchema, BaseWeightSchema
from .humming import HummingInputSchema, HummingWeightSchema


@dataclasses.dataclass(kw_only=True)
class Fp8WeightSchema(BaseWeightSchema):
    quant_method: str = "fp8"
    weight_block_size: tuple[int, int] | None = None

    def __post_init__(self):
        if isinstance(self.weight_block_size, list):
            self.weight_block_size = tuple(self.weight_block_size)
        self.weight_scale_key = "weight_scale" if self.weight_block_size is None else "weight_scale_inv"

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
                "shape": (shape_n, shape_k),
                "dtype": torch.float8_e4m3fn,
                "extra_attrs": {"output_dim": 0, "input_dim": 1},
            },
        }

        if self.weight_block_size is not None:
            scale_block_shape_n, scale_block_shape_k = self.weight_block_size
            scale_shape = (shape_n // scale_block_shape_n, shape_k // scale_block_shape_k)
            tensor_meta[self.weight_scale_key] = {
                "shape": scale_shape,
                "dtype": torch.float32,
                "extra_attrs": {"input_dim": 1, "output_dim": 0, "scale_type": "block"},
            }
        else:
            tensor_meta[self.weight_scale_key] = {
                "shape": (stack_size,),
                "dtype": torch.float32,
                "extra_attrs": {"scale_type": "tensor"},
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
        shape_k = tensors["weight"].size(-1)
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
        group_size = 0
        if self.weight_block_size is not None:
            group_size = self.weight_block_size[1]

        schema = HummingWeightSchema(
            b_dtype=dtypes.float8e4m3,
            weight_scale_group_size=group_size,
        )

        weight = tensors["weight"].view(torch.int32)
        weight_scale = tensors[self.weight_scale_key]

        if self.weight_block_size is None:
            weight_scale = self._may_process_global_scale(
                weight_scale,
                shape_n_stacks=shape_n_stacks,
                shape_k_stacks=shape_k_stacks,
                num_experts=num_experts,
                force_repeat=True,
            ).to(param_dtype)
        else:
            shape_n = sum(shape_n_stacks)
            scale_block_shape_n = self.weight_block_size[0]
            weight_scale = weight_scale.repeat_interleave(scale_block_shape_n, -2)
            weight_scale = weight_scale[..., :shape_n, :]
            weight_scale = weight_scale.to(param_dtype).contiguous()

        output_tensors = {"weight": weight, "weight_scale": weight_scale}

        if "bias" in tensors:
            output_tensors["bias"] = tensors["bias"]

        return schema, output_tensors


@dataclasses.dataclass(kw_only=True)
class Fp8InputSchema(BaseInputSchema):
    quant_method: str = "fp8"
    activation_scheme: str = "dynamic"

    def get_activation_bits(self):
        return 8

    def get_tensors_attrs(
        self,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        if self.activation_scheme == "static":
            return self._get_input_scale_attrs(num_experts, stack_size)
        return {}

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        sm_version: int | tuple[int, int] | None = None,
    ) -> tuple[HummingInputSchema, dict[str, torch.Tensor]]:
        a_dtype = self.get_fallback_input_dtype(dtypes.float8e4m3, sm_version)
        schema = HummingInputSchema(a_dtype=a_dtype, input_scale_group_size=0)
        return schema, {}
