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
class CompressedTensorsWeightSchema(BaseWeightSchema):
    quant_method: str = "compressed-tensors"

    format: str
    type: str
    num_bits: int
    strategy: str
    symmetric: bool = True
    block_structure: tuple[int, int] | None = None
    group_size: int | None = None
    actorder: str | None = None

    def __post_init__(self):
        assert self.format in [
            "int-quantized",
            "float-quantized",
            "naive-quantized",
            "pack-quantized",
            "nvfp4-pack-quantized",
            "mxfp4-pack-quantized",
        ]
        msg = "actorder is not supported by humming"
        assert self.actorder is None or self.actorder in ["weight", "static"], msg
        self.weight_key = "weight_packed" if "pack" in self.format else "weight"
        if isinstance(self.block_structure, list):
            self.block_structure = tuple(self.block_structure)

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        weight_dtype = None
        if self.format == "pack-quantized":
            weight_dtype = torch.int32
        elif "pack-" not in self.format and "-quantized" in self.format:
            assert self.num_bits == 8
            if self.type == "float":
                weight_dtype = torch.float8_e4m3fn
            else:
                weight_dtype = torch.int8
        elif self.format in ["nvfp4-pack-quantized", "mxfp4-pack-quantized"]:
            weight_dtype = torch.uint8
        else:
            raise ValueError(f"unsupported compressed-tensors format: {self.format}")

        weight_shape = (shape_n, shape_k * self.num_bits // (weight_dtype.itemsize * 8))
        scale_dtype = param_dtype
        if "nvfp4" in self.format:
            scale_dtype = torch.float8_e4m3fn
        elif "mxfp4" in self.format:
            scale_dtype = torch.uint8

        scale_shape: tuple[int, ...]
        if "group" in self.strategy:
            scale_type = "group"
            assert self.group_size is not None
            scale_shape = (shape_n, shape_k // self.group_size)
        elif self.strategy == "block":
            scale_type = "group"
            assert self.block_structure is not None
            block_shape = self.block_structure
            scale_shape = (shape_n // block_shape[0], shape_k // block_shape[1])
            pass
        elif self.strategy == "channel":
            scale_shape = (shape_n, 1)
        elif self.strategy == "tensor":
            scale_shape = (stack_size,)
        else:
            raise ValueError(f"unsupported compressed-tensors strategy: {self.strategy}")

        scale_type = self.strategy
        if self.strategy == "tensor_group":
            scale_type = "group"

        tensor_meta: dict[str, Any] = {
            self.weight_key: {
                "shape": weight_shape,
                "dtype": weight_dtype,
                "extra_attrs": {"input_dim": 1, "output_dim": 0},
            },
            "weight_scale": {
                "shape": scale_shape,
                "dtype": scale_dtype,
                "extra_attrs": {"scale_type": scale_type},
            },
        }

        if "pack" in self.format:
            tensor_meta["weight_scale"]["extra_attrs"]["packed_dim"] = 1
            packed_factor = weight_dtype.itemsize * 8 / self.num_bits
            tensor_meta["weight_scale"]["extra_attrs"]["packed_factor"] = packed_factor

        if scale_type == "channel":
            tensor_meta["weight_scale"]["extra_attrs"]["output_dim"] = 0
        elif scale_type != "tensor":
            tensor_meta["weight_scale"]["extra_attrs"]["input_dim"] = 1
            tensor_meta["weight_scale"]["extra_attrs"]["output_dim"] = 0

        if not self.symmetric:
            assert self.strategy in ["group", "channel"]
            assert len(scale_shape) > 1
            num_groups = scale_shape[1]
            tensor_meta["weight_zero_point"] = {
                "shape": (shape_n * self.num_bits // 32, num_groups),
                "dtype": torch.int32,
                "extra_attrs": {
                    "input_dim": 1,
                    "output_dim": 0,
                    "packed_dim": 0,
                    "packed_factor": 32 / self.num_bits,
                },
            }

        if "nvfp4" in self.format:
            tensor_meta["weight_global_scale"] = {
                "shape": (stack_size,),
                "dtype": torch.float32,
                "extra_attrs": {"scale_type": "tensor"},
            }

        if self.format == "pack-quantized":
            tensor_meta["weight_shape"] = {
                "shape": (2,),
                "dtype": torch.int64,
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
        weight = tensors[self.weight_key]
        shape_n = weight.size(-2)
        shape_k = weight.size(-1) * weight.dtype.itemsize * 8 // self.num_bits
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
        weight = tensors[self.weight_key]
        if self.format == "int-quantized":
            assert weight.dtype == torch.int8
            weight = weight + 128
        weight = weight.view(torch.int32)
        weight_scale = tensors["weight_scale"]
        if self.format.startswith("mxfp"):
            weight_scale = weight_scale.view(torch.float8_e8m0fnu)
        elif self.format.startswith("nvfp4"):
            weight_scale = weight_scale.view(torch.float8_e4m3fn)
        else:
            weight_scale = weight_scale.to(param_dtype)

        if self.strategy in ["group", "tensor_group"]:
            assert self.group_size is not None
            group_size = self.group_size
        elif self.strategy == "block":
            assert self.block_structure is not None
            weight_scale = weight_scale.repeat_interleave(self.block_structure[0], -2)
            group_size = self.block_structure[1]
        elif self.strategy == "tensor":
            weight_scale = self._may_process_global_scale(
                weight_scale,
                shape_n_stacks=shape_n_stacks,
                shape_k_stacks=shape_k_stacks,
                num_experts=num_experts,
                force_repeat=True,
            ).to(param_dtype)
            group_size = 0
        else:
            assert self.strategy == "channel"
            group_size = 0

        output_tensors = {"weight": weight, "weight_scale": weight_scale}

        weight_scale_type = None
        weight_scale_2_type = None
        if self.format == "nvfp4-pack-quantized":
            weight_scale_2 = tensors["weight_global_scale"]
            target_group_size = 16 if len(shape_k_stacks) > 1 else None
            weight_scale_2 = self._may_process_global_scale(
                1 / weight_scale_2,
                shape_n_stacks=shape_n_stacks,
                shape_k_stacks=shape_k_stacks,
                num_experts=num_experts,
                target_group_size=target_group_size,
            )
            has_tensor_scale = weight_scale_2.nelement() == (num_experts or 1)
            weight_scale_type = WeightScaleType.GROUP
            if has_tensor_scale:
                weight_scale_2_type = WeightScale2Type.TENSOR
                output_tensors["weight_scale_2"] = weight_scale_2.float()
            else:
                weight_scale = weight_scale.float() * weight_scale_2.float()
                output_tensors["weight_scale"] = weight_scale.to(param_dtype)

        bs_dtype = None
        if self.format in ["int-quantized", "float-quantized", "naive-quantized"]:
            assert self.num_bits == 8
            b_dtype = dtypes.uint8 if self.type == "int" else dtypes.float8e4m3
        elif "nvfp4" in self.format:
            b_dtype = dtypes.float4e2m1
            bs_dtype = dtypes.float8e4m3 if has_tensor_scale else None
        elif "mxfp4" in self.format:
            b_dtype = dtypes.float4e2m1
            bs_dtype = dtypes.float8e8m0
        else:
            assert self.format == "pack-quantized"
            assert self.type == "int"
            b_dtype = dtypes.DataType.from_str(f"uint{self.num_bits}")

        if not self.symmetric:
            zero_point = tensors["weight_zero_point"]
            output_tensors["zero_point"] = zero_point

        if "bias" in tensors:
            output_tensors["bias"] = tensors["bias"]

        schema = HummingWeightSchema(
            b_dtype=b_dtype,
            bs_dtype=bs_dtype,
            weight_scale_group_size=group_size,
            weight_scale_type=weight_scale_type,
            weight_scale_2_type=weight_scale_2_type,
            has_zero_point=not self.symmetric,
        )

        return schema, output_tensors


@dataclasses.dataclass(kw_only=True)
class CompressedTensorsInputSchema(BaseInputSchema):
    quant_method: str = "compressed-tensors"

    format: str
    type: str
    num_bits: int
    dynamic: bool | str
    group_size: int
    symmetric: bool = True

    def __post_init__(self):
        assert self.format in [
            "int-quantized",
            "float-quantized",
            "naive-quantized",
            "pack-quantized",
            "nvfp4-pack-quantized",
            "mxfp4-pack-quantized",
        ]
        self.input_scale_key = "input_global_scale" if "nvfp4" in self.format else "input_scale"

    def get_activation_bits(self):
        return self.num_bits

    def get_tensors_attrs(
        self,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        tensors_attrs: dict[str, Any] = {}
        if self.dynamic is False or self.dynamic == "local":
            tensors_attrs |= self._get_input_scale_attrs(
                num_experts=num_experts,
                stack_size=stack_size,
                dtype=torch.float32,
                input_scale_name=self.input_scale_key,
            )
            if not self.symmetric:
                assert self.type == "int" and self.num_bits == 8
                tensors_attrs |= self._get_input_scale_attrs(
                    num_experts=num_experts,
                    stack_size=stack_size,
                    dtype=torch.int8,
                    input_scale_name="input_zero_point",
                )

        return tensors_attrs

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        sm_version: int | tuple[int, int] | None = None,
    ) -> tuple[HummingInputSchema, dict[str, torch.Tensor]]:
        if self.type == "float" and self.num_bits == 8:
            origin_a_dtype = dtypes.float8e4m3
        elif self.type == "float" and self.num_bits == 4:
            origin_a_dtype = dtypes.float4e2m1
        elif self.type == "int" and self.num_bits == 8:
            origin_a_dtype = dtypes.int8
        elif self.type == "int" and self.num_bits == 4:
            origin_a_dtype = dtypes.int4
        else:
            raise ValueError(f"unsupported {self.type}{self.num_bits}")

        a_dtype = self.get_fallback_input_dtype(origin_a_dtype, sm_version)
        group_size = self.group_size if a_dtype == dtypes.float4e2m1 else 0
        schema = HummingInputSchema(a_dtype=a_dtype, input_scale_group_size=group_size)
        return schema, {}
