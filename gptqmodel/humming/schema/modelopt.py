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
from ..config.enum import WeightScale2Type, WeightScaleType
from .base import BaseInputSchema, BaseWeightSchema
from .humming import HummingInputSchema, HummingWeightSchema


@dataclasses.dataclass(kw_only=True)
class ModeloptWeightSchema(BaseWeightSchema):
    quant_method: str = "modelopt"
    quant_algo: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModeloptWeightSchema":
        algo_cls = cls
        if cls is ModeloptWeightSchema:
            quant_algo = config["quant_algo"].lower()
            algo_cls: type[ModeloptWeightSchema]
            if quant_algo == "nvfp4":
                algo_cls = ModeloptNvfp4WeightSchema
            elif quant_algo == "mxfp8":
                algo_cls = ModeloptMxfp8WeightSchema
            else:
                raise ValueError(f"unsupported modelopt algo: {quant_algo}")

        kwargs = {}
        for field in dataclasses.fields(cls):
            name = field.name
            if name in config:
                kwargs[name] = config[name]

        return algo_cls(**kwargs)


@dataclasses.dataclass(kw_only=True)
class ModeloptNvfp4WeightSchema(ModeloptWeightSchema):
    quant_method: str = "modelopt"
    quant_algo: str = "nvfp4"

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
                "shape": (shape_n, shape_k // 16),
                "dtype": torch.float8_e4m3fn,
                "extra_attrs": {"output_dim": 0, "input_dim": 1, "scale_type": "group"},
            },
            "weight_scale_2": {
                "shape": (stack_size,),
                "dtype": torch.float32,
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
        shape_n, shape_k = tensors["weight"].shape[-2:]
        shape_k = shape_k * 2
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
        weight = tensors["weight"].view(torch.int32)
        weight_scale = tensors["weight_scale"].view(torch.float8_e4m3fn)

        target_group_size = 16 if len(shape_k_stacks) > 1 else None
        weight_scale_2 = tensors["weight_scale_2"].view(num_experts or 1, -1)

        weight_scale_2 = self._may_process_global_scale(
            weight_scale_2,
            shape_n_stacks=shape_n_stacks,
            shape_k_stacks=shape_k_stacks,
            num_experts=num_experts,
            target_group_size=target_group_size,
        )

        output_tensors = {"weight": weight, "weight_scale": weight_scale}
        has_tensor_scale = weight_scale_2.nelement() == (num_experts or 1)
        if has_tensor_scale:
            weight_scale_2_type = WeightScale2Type.TENSOR
            output_tensors["weight_scale_2"] = weight_scale_2.view((num_experts or 1)).float()
        else:
            weight_scale_2_type = WeightScale2Type.NONE
            weight_scale = weight_scale.float() * weight_scale_2.float()
            output_tensors["weight_scale"] = weight_scale.to(param_dtype)

        if "bias" in tensors:
            output_tensors["bias"] = tensors["bias"]

        schema = HummingWeightSchema(
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e4m3 if has_tensor_scale else None,
            weight_scale_group_size=16,
            weight_scale_type=WeightScaleType.GROUP,
            weight_scale_2_type=weight_scale_2_type,
        )

        return schema, output_tensors


@dataclasses.dataclass(kw_only=True)
class ModeloptMxfp8WeightSchema(ModeloptWeightSchema):
    quant_method: str = "modelopt"
    quant_algo: str = "mxfp8"

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
        shape_n, shape_k = tensors["weight"].shape[:-2]
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
            b_dtype=dtypes.float8e4m3,
            bs_dtype=dtypes.float8e8m0,
            weight_scale_group_size=32,
        )

        weight = tensors["weight"].view(torch.int32)
        weight_scale = tensors["weight_scale"].view(torch.float8_e8m0fnu)
        output_tensors = {"weight": weight, "weight_scale": weight_scale}

        if "bias" in tensors:
            output_tensors["bias"] = tensors["bias"]

        return schema, output_tensors


@dataclasses.dataclass(kw_only=True)
class ModeloptInputSchema(BaseInputSchema):
    quant_method: str = "modelopt"
    quant_algo: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModeloptInputSchema":
        if cls is ModeloptInputSchema:
            quant_algo = config["quant_algo"].lower()
            if quant_algo == "nvfp4":
                algo_cls = ModeloptNvfp4InputSchema
            else:
                raise ValueError(f"unsupported modelopt algo: {quant_algo}")

        kwargs = {}
        for field in dataclasses.fields(cls):
            name = field.name
            if name in config:
                kwargs[name] = config[name]

        return algo_cls(**kwargs)


@dataclasses.dataclass(kw_only=True)
class ModeloptNvfp4InputSchema(ModeloptInputSchema):
    dynamic: bool = False
    num_bits: int = 4
    type: str = "float"
    group_size: int = 16

    def get_activation_bits(self):
        return 4

    def get_tensors_attrs(
        self,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        if not self.dynamic:
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
        group_size = 16 if a_dtype == dtypes.float4e2m1 else 0
        schema = HummingInputSchema(a_dtype=a_dtype, input_scale_group_size=group_size)
        return schema, {}
