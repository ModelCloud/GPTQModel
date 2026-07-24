# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import dataclasses
from typing import Any, ClassVar

import torch

from .. import dtypes
from ..config.enum import WeightScale2Type, WeightScaleType
from .base import BaseInputSchema, BaseWeightSchema
from ..utils.weight import dequantize_weight, quantize_weight


@dataclasses.dataclass(kw_only=True)
class HummingWeightSchema(BaseWeightSchema):
    quant_method: str = "humming"
    b_dtype: dtypes.DataType
    bs_dtype: dtypes.DataType | None = None
    weight_scale_group_size: int = 0
    weight_scale_group_size_n: int = 0
    weight_scale_type: WeightScaleType | str | None = None
    weight_scale_2_type: WeightScale2Type | str | None = None
    has_zero_point: bool = False
    is_fp_zero_point: bool = False
    hadamard_block_size: int = 0

    KWARGS_ALIAS: ClassVar[dict[str, list[str]]] = {
        "b_dtype": ["weight_dtype", "dtype"],
        "weight_scale_group_size": ["group_size"],
        "weight_scale_group_size_n": ["group_size_n"],
        "weight_scale_type": ["scale_type"],
        "weight_scale_2_type": ["scale_2_type"],
        "bs_dtype": ["weight_scale_dtype", "scale_dtype"],
    }

    def __post_init__(self):
        assert self.quant_method == "humming"
        if isinstance(self.b_dtype, str):
            self.b_dtype = dtypes.DataType.from_str(str(self.b_dtype))
        if isinstance(self.b_dtype, dtypes.IntegerType) and self.b_dtype.is_signed:
            self.b_dtype = dtypes.DataType.from_str("u" + str(self.b_dtype))
        if isinstance(self.bs_dtype, str):
            self.bs_dtype = dtypes.DataType.from_str(str(self.bs_dtype))

        if isinstance(self.weight_scale_type, str):
            self.weight_scale_type = WeightScaleType(self.weight_scale_type)
        elif self.weight_scale_type is None:
            if self.weight_scale_group_size_n > 1:
                self.weight_scale_type = WeightScaleType.BLOCK
            elif self.weight_scale_group_size == 0:
                self.weight_scale_type = WeightScaleType.CHANNEL
            elif self.weight_scale_group_size > 0:
                self.weight_scale_type = WeightScaleType.GROUP

        if isinstance(self.weight_scale_2_type, str):
            self.weight_scale_2_type = WeightScale2Type(self.weight_scale_2_type)
        if self.weight_scale_2_type is None:
            self.weight_scale_2_type = WeightScale2Type.NONE
        if self.weight_scale_2_type != WeightScale2Type.NONE:
            assert self.weight_scale_type == WeightScaleType.GROUP, (
                "weight_scale_2_type requires weight_scale_type='group'"
            )

        if self.weight_scale_type == WeightScaleType.BLOCK:
            self.bs_dtype = dtypes.float32

    @property
    def has_tensor_weight_scale(self) -> bool:
        return (
            self.weight_scale_type == WeightScaleType.TENSOR
            or self.weight_scale_2_type == WeightScale2Type.TENSOR
        )

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        num_bits = self.b_dtype.num_bits
        group_size = self.weight_scale_group_size or shape_k

        scale_torch_dtype = param_dtype
        if self.bs_dtype == dtypes.float8e8m0:
            scale_torch_dtype = torch.float8_e8m0fnu
        elif self.bs_dtype == dtypes.float8e4m3:
            scale_torch_dtype = torch.float8_e4m3fn
        elif self.bs_dtype == dtypes.float8e5m2:
            scale_torch_dtype = torch.float8_e5m2

        tensor_meta: dict[str, Any] = {
            "weight": {
                "shape": (shape_n, shape_k * num_bits // 32),
                "dtype": torch.int32,
                "extra_attrs": {
                    "input_dim": 1,
                    "output_dim": 0,
                    "packed_factor": 32 / num_bits,
                    "packed_dim": 1,
                },
            }
        }

        if self.weight_scale_type == WeightScaleType.GROUP:
            tensor_meta["weight_scale"] = {
                "shape": (shape_n, shape_k // group_size),
                "dtype": scale_torch_dtype,
                "extra_attrs": {"input_dim": 1, "output_dim": 0, "scale_type": "group"},
            }
        elif self.weight_scale_type == WeightScaleType.CHANNEL:
            tensor_meta["weight_scale"] = {
                "shape": (shape_n, 1),
                "dtype": scale_torch_dtype,
                "extra_attrs": {"output_dim": 0, "scale_type": "channel"},
            }
        elif self.weight_scale_type == WeightScaleType.BLOCK:
            group_size_n = self.weight_scale_group_size_n
            tensor_meta["weight_scale"] = {
                "shape": (shape_n // group_size_n, shape_k // group_size),
                "dtype": torch.float32,
                "extra_attrs": {"input_dim": 1, "output_dim": 0, "scale_type": "block"},
            }
        elif self.weight_scale_type == WeightScaleType.TENSOR:
            tensor_meta["weight_scale"] = {
                "shape": (1,),
                "dtype": torch.float32,
                "extra_attrs": {"scale_type": "tensor"},
            }

        if self.has_zero_point and not self.is_fp_zero_point:
            tensor_meta["zero_point"] = {
                "shape": (shape_n * num_bits // 32, shape_k // group_size),
                "dtype": torch.int32,
                "extra_attrs": {
                    "input_dim": 1,
                    "output_dim": 0,
                    "packed_factor": 32 / num_bits,
                    "packed_dim": 0,
                },
            }

            assert self.weight_scale_type == WeightScaleType.GROUP

        if self.has_zero_point and self.is_fp_zero_point:
            tensor_meta["zero_point"] = {
                "shape": (shape_n, shape_k // group_size),
                "dtype": param_dtype,
                "extra_attrs": {"input_dim": 1, "output_dim": 0},
            }

        if self.weight_scale_2_type == WeightScale2Type.TENSOR:
            tensor_meta["weight_scale_2"] = {
                "shape": (1,),
                "dtype": torch.float32,
                "extra_attrs": {"scale_type": "tensor"},
            }
        elif self.weight_scale_2_type == WeightScale2Type.CHANNEL:
            tensor_meta["weight_scale_2"] = {
                "shape": (shape_n,),
                "dtype": param_dtype,
                "extra_attrs": {"output_dim": 0, "scale_type": "channel"},
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
        num_bits = self.b_dtype.num_bits
        shape_n = tensors["weight"].size(-2)
        shape_k = tensors["weight"].size(-1) * 32 // num_bits
        has_bias = "bias" in tensors
        return shape_n, shape_k, None, has_bias

    @classmethod
    def quant_tensor(
        cls,
        tensor: torch.Tensor,
        schema: "HummingWeightSchema",
        param_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        f16_dtype = dtypes.DataType.from_torch_dtype(param_dtype)
        shape_n = tensor.size(-2)
        shape_k = tensor.size(-1)
        num_experts = tensor.size(0) if tensor.ndim == 3 else None
        if schema.hadamard_block_size > 1:
            from .. import ops

            tensor = ops.hadamard_transform(tensor, schema.hadamard_block_size)
        is_tensor_only = schema.weight_scale_type == WeightScaleType.TENSOR
        if schema.weight_scale_2_type == WeightScale2Type.TENSOR:
            scale_2_type = "tensor"
        elif schema.weight_scale_2_type == WeightScale2Type.CHANNEL:
            scale_2_type = "channel"
        else:
            scale_2_type = "tensor" if is_tensor_only else None

        weight, weight_scale, zero_point, scale_2 = quantize_weight(
            weight=tensor,
            dtype=schema.b_dtype,
            scale_dtype=None if is_tensor_only else (schema.bs_dtype or f16_dtype),
            group_size=schema.weight_scale_group_size,
            group_size_n=schema.weight_scale_group_size_n or None,
            has_zero_point=schema.has_zero_point,
            weight_scale_2_type=scale_2_type,
            is_fp_zero_point=schema.is_fp_zero_point,
            pack=True,
        )

        tensors = {"weight": weight}
        if weight_scale is not None and weight_scale.nelement() > 0:
            tensors["weight_scale"] = weight_scale
        if zero_point is not None and zero_point.nelement() > 0:
            tensors["zero_point"] = zero_point
        if scale_2 is not None and scale_2.nelement() > 0:
            if is_tensor_only:
                tensors["weight_scale"] = scale_2.float().view(-1)
            elif scale_2_type == "channel":
                tensors["weight_scale_2"] = scale_2.to(param_dtype)
            else:
                tensors["weight_scale_2"] = scale_2.float().view(-1)

        schema.validate_tensors(tensors, shape_n, shape_k, param_dtype, num_experts)

        return tensors

    def dequant_tensors(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        zero_point = None if not self.has_zero_point else tensors["zero_point"]
        if self.weight_scale_type == WeightScaleType.TENSOR:
            weight_scale = None
            weight_scale_2 = tensors["weight_scale"]
        else:
            weight_scale = tensors["weight_scale"]
            weight_scale_2 = tensors.get("weight_scale_2")
        return dequantize_weight(
            tensors["weight"],
            weight_scale=weight_scale,
            zero_point=zero_point,
            weight_scale_2=weight_scale_2,
            dtype=self.b_dtype,
            packed=True,
        )

    def requant_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        target_weight_schema: "HummingWeightSchema",
        param_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        dequanted_weight = self.dequant_tensors(tensors)
        return self.quant_tensor(dequanted_weight, target_weight_schema, param_dtype)

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple["HummingWeightSchema", dict[str, torch.Tensor]]:
        schema = dataclasses.replace(self)
        is_tensor_only = self.weight_scale_type == WeightScaleType.TENSOR
        if self.has_tensor_weight_scale:
            scale_key = "weight_scale" if is_tensor_only else "weight_scale_2"
            tensor_scale = tensors[scale_key].view(num_experts or 1, -1)
            tensor_scale = self._may_process_global_scale(
                tensor_scale,
                shape_n_stacks=shape_n_stacks,
                shape_k_stacks=shape_k_stacks,
                num_experts=num_experts,
                target_group_size=schema.weight_scale_group_size,
            )

            if tensor_scale.nelement() == (num_experts or 1):
                tensors[scale_key] = tensor_scale.to(torch.float32)
                if is_tensor_only:
                    schema.bs_dtype = dtypes.float32
            elif is_tensor_only:
                schema.weight_scale_type = WeightScaleType.CHANNEL
                schema.bs_dtype = dtypes.DataType.from_torch_dtype(param_dtype)
                tensors["weight_scale"] = tensor_scale.to(param_dtype)
            else:
                schema.weight_scale_2_type = WeightScale2Type.NONE
                weight_scale = tensors["weight_scale"].float() * tensor_scale.float()
                tensors["weight_scale"] = weight_scale.to(param_dtype)
                schema.bs_dtype = dtypes.DataType.from_torch_dtype(param_dtype)
                del tensors["weight_scale_2"]
        elif self.weight_scale_type in [WeightScaleType.GROUP, WeightScaleType.CHANNEL]:
            if tensors["weight_scale"].dtype == torch.float32:
                tensors["weight_scale"] = tensors["weight_scale"].to(param_dtype)
            if self.bs_dtype == dtypes.float32:
                schema.bs_dtype = dtypes.DataType.from_torch_dtype(param_dtype)
            if self.weight_scale_2_type == WeightScale2Type.CHANNEL:
                if tensors["weight_scale_2"].dtype == torch.float32:
                    tensors["weight_scale_2"] = tensors["weight_scale_2"].to(param_dtype)
        elif self.weight_scale_type == WeightScaleType.BLOCK:
            tensors["weight_scale"] = tensors["weight_scale"].to(torch.float32)
            schema.bs_dtype = dtypes.float32

        return schema, tensors


@dataclasses.dataclass(kw_only=True)
class HummingInputSchema(BaseInputSchema):
    quant_method: str = "humming"
    a_dtype: dtypes.DataType | None = None
    input_scale_group_size: int = 0
    input_scale_dtype: dtypes.DataType | None = None

    KWARGS_ALIAS: ClassVar[dict[str, list[str]]] = {
        "a_dtype": ["input_dtype", "dtype"],
        "input_scale_group_size": ["group_size"],
        "input_scale_dtype": ["scale_dtype"],
    }

    def __post_init__(self):
        if isinstance(self.a_dtype, str):
            self.a_dtype = dtypes.DataType.from_str(str(self.a_dtype))
        if isinstance(self.input_scale_dtype, str):
            self.input_scale_dtype = dtypes.DataType.from_str(str(self.input_scale_dtype))

    def get_activation_bits(self):
        if self.a_dtype is None:
            return 16
        return self.a_dtype.num_bits

    def get_tensors_attrs(
        self,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        return {}

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        sm_version: int | tuple[int, int] | None = None,
    ) -> tuple["HummingInputSchema", dict[str, torch.Tensor]]:
        return self, {}
