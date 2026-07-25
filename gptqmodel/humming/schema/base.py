# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import dataclasses
import math
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from .. import dtypes

if TYPE_CHECKING:
    from .humming import HummingInputSchema, HummingWeightSchema


@dataclasses.dataclass(kw_only=True)
class BaseWeightSchema:
    quant_method: str

    WEIGHT_SCHEMA_MAP: ClassVar[dict[str, type["BaseWeightSchema"]]]
    KWARGS_ALIAS: ClassVar[dict[str, list[str]]] = {}

    @classmethod
    def may_add_expert_dim(
        cls,
        tensors_attrs: dict[str, dict[str, Any]],
        num_experts: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        if num_experts is None or num_experts == 0:
            return tensors_attrs
        for _, attr in tensors_attrs.items():
            if attr["shape"] == (1,):
                attr["shape"] = (num_experts,)
            else:
                attr["shape"] = (num_experts,) + attr["shape"]
            for key, value in attr.get("extra_attrs", {}).items():
                if not key.endswith("dim"):
                    continue
                attr["extra_attrs"][key] = value + 1
        return tensors_attrs

    def infer_shape(self, tensors: dict[str, torch.Tensor]) -> tuple[int, int, int | None, bool]:
        raise NotImplementedError

    def get_padded_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        pad_n_to_multiple: int = 1,
        pad_k_to_multiple: int = 1,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        padded_shape_n = math.ceil(shape_n / pad_n_to_multiple) * pad_n_to_multiple
        padded_shape_k = math.ceil(shape_k / pad_k_to_multiple) * pad_k_to_multiple

        tensors_attrs = self.get_tensors_attrs(
            shape_n=shape_n,
            shape_k=shape_k,
            param_dtype=param_dtype,
            num_experts=num_experts,
            has_bias=has_bias,
            stack_size=stack_size,
        )

        if shape_n == padded_shape_n and shape_k == padded_shape_k:
            return tensors_attrs

        padded_tensors_attrs = self.get_tensors_attrs(
            shape_n=padded_shape_n,
            shape_k=padded_shape_k,
            param_dtype=param_dtype,
            num_experts=num_experts,
            has_bias=has_bias,
            stack_size=stack_size,
        )

        for key in padded_tensors_attrs:
            shape1 = tensors_attrs[key]["shape"]
            shape2 = padded_tensors_attrs[key]["shape"]
            pad_shape = tuple(y - x for x, y in zip(shape1, shape2, strict=True))
            padded_tensors_attrs[key]["extra_attrs"]["pad_shape"] = pad_shape

        return padded_tensors_attrs

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    def process_loaded_weight(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return tensor

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple["HummingWeightSchema", dict[str, torch.Tensor]]:
        raise NotImplementedError

    def validate_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
    ):
        tensors_attrs = self.get_tensors_attrs(
            shape_n=shape_n,
            shape_k=shape_k,
            param_dtype=param_dtype,
            num_experts=num_experts,
            has_bias=has_bias,
        )
        for key, attr in tensors_attrs.items():
            assert key in tensors, f"{key} is required by {self.quant_method}"
            tensor = tensors[key]
            msg = f"{key} dtype mismatched: {attr['dtype']=}, {tensor.dtype=}"
            assert attr["dtype"] == tensor.dtype, msg
            msg = f"{key} shape mismatched: {attr['shape']=}, {tensor.shape=}"
            assert attr["shape"] == tensor.shape, msg

    def _may_process_global_scale(
        self,
        scale: torch.Tensor,
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        num_experts: int | None = None,
        target_group_size: int | None = None,
        force_repeat: int = False,
    ):
        assert len(shape_n_stacks) == 1 or len(shape_k_stacks) == 1

        scale = scale.view(num_experts or 1, -1)

        if (scale == scale[:, [0]]).all() and not force_repeat:
            return scale[:, 0]
        elif len(shape_n_stacks) > 1:
            shape_n_repeats = torch.tensor(shape_n_stacks)
            target_group_size = target_group_size or sum(shape_k_stacks)
            shape_k_repeats = torch.tensor([sum(shape_k_stacks) // target_group_size])
            scale = scale.view(-1, len(shape_n_stacks), 1)
            shape_n_repeats = shape_n_repeats.to(scale.device)
            shape_k_repeats = shape_k_repeats.to(scale.device)
            scale = scale.repeat_interleave(shape_n_repeats, 1)
            scale = scale.repeat_interleave(shape_k_repeats, 2)
        elif len(shape_k_stacks) > 1:
            shape_n_repeats = torch.tensor([sum(shape_n_stacks)])
            gcd = math.gcd(*shape_k_stacks)
            max_group_size = gcd & -gcd
            target_group_size = target_group_size or max_group_size
            assert target_group_size <= max_group_size
            shape_k_repeats = torch.tensor([x // target_group_size for x in shape_k_stacks])
            scale = scale.view(-1, 1, len(shape_k_stacks))
            scale = scale.repeat_interleave(shape_n_repeats, 1)
            scale = scale.repeat_interleave(shape_k_repeats, 2)
        elif force_repeat:
            shape_n = sum(shape_n_stacks)
            scale = scale.view(-1, 1, 1).repeat_interleave(shape_n, 1)
        else:
            return scale.view(num_experts or 1)

        assert scale.size(0) == (num_experts or 1)
        return scale.squeeze(0) if num_experts is None else scale

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseWeightSchema":
        from . import WEIGHT_SCHEMA_MAP

        if cls is BaseWeightSchema:
            quant_method = config["quant_method"]
            if quant_method in WEIGHT_SCHEMA_MAP:
                schema_cls = WEIGHT_SCHEMA_MAP[quant_method]
                return schema_cls.from_config(config)
            else:
                raise ValueError(f"unsupported weight quant_method: {quant_method}")

        kwargs = {}
        for field in dataclasses.fields(cls):
            name = field.name
            if name in config:
                kwargs[name] = config[name]
            elif name in cls.KWARGS_ALIAS:
                for alias_name in cls.KWARGS_ALIAS[name]:
                    if alias_name in config:
                        kwargs[name] = config[alias_name]
                        break

        return cls(**kwargs)


@dataclasses.dataclass(kw_only=True)
class BaseInputSchema:
    quant_method: str

    INPUT_SCHEMA_MAP: ClassVar[dict[str, type["BaseInputSchema"]]]
    KWARGS_ALIAS: ClassVar[dict[str, list[str]]] = {}

    may_add_expert_dim = BaseWeightSchema.may_add_expert_dim

    def get_activation_bits(self):
        raise NotImplementedError

    def get_fallback_input_dtype(
        self,
        a_dtype: dtypes.DataType | None,
        sm_version: int | tuple[int, int] | None = None,
    ) -> dtypes.DataType | None:
        if sm_version is None:
            sm_version = torch.cuda.get_device_capability()
        if isinstance(sm_version, tuple):
            sm_version = sm_version[0] * 10 + sm_version[1]
        assert isinstance(sm_version, int)

        a_dtype_order: list[dtypes.DataType] = []
        if a_dtype is None or a_dtype in [dtypes.float16, dtypes.bfloat16]:
            return a_dtype
        elif a_dtype == dtypes.float8e4m3:
            a_dtype_order = [dtypes.float8e4m3]
        elif a_dtype == dtypes.float8e5m2:
            a_dtype_order = [dtypes.float8e5m2]
        elif a_dtype == dtypes.float4e2m1:
            # NOTE: float4e2m1 isn't fully tested now, so disable it
            a_dtype_order = [dtypes.float8e4m3]
        elif a_dtype == dtypes.int8:
            a_dtype_order = [dtypes.int8]
        elif a_dtype == dtypes.int4:
            a_dtype_order = [dtypes.int4, dtypes.float8e4m3, dtypes.int8]
        else:
            raise ValueError(f"unsupported a_dtype: {a_dtype}")

        for dtype in a_dtype_order:
            if dtype == dtypes.float8e4m3 and sm_version >= 89:
                return dtype
            elif dtype == dtypes.float8e5m2 and sm_version >= 89:
                return dtype
            elif dtype == dtypes.float4e2m1 and sm_version >= 120:
                return dtype
            elif dtype == dtypes.int8:
                return dtype
            elif dtype == dtypes.int4 and sm_version >= 80:
                return dtype

        return None

    def _get_input_scale_attrs(
        self,
        num_experts: int | None = None,
        stack_size: int = 1,
        dtype: torch.dtype = torch.float32,
        input_scale_name: str = "input_scale",
    ) -> dict[str, dict[str, Any]]:
        tensor_meta = {
            input_scale_name: {
                "shape": (stack_size,),
                "dtype": dtype,
                "extra_attrs": {"scale_type": "input_scale"},
            }
        }

        self.may_add_expert_dim(tensor_meta, num_experts)
        return tensor_meta

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
    ) -> tuple["HummingInputSchema", dict[str, torch.Tensor]]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseInputSchema":
        from . import INPUT_SCHEMA_MAP

        if cls is BaseInputSchema:
            quant_method = config["quant_method"]
            if quant_method in INPUT_SCHEMA_MAP:
                schema_cls = INPUT_SCHEMA_MAP[quant_method]
                return schema_cls.from_config(config)
            else:
                raise ValueError(f"unsupported input quant_method: {quant_method}")

        kwargs = {}
        for field in dataclasses.fields(cls):
            name = field.name
            if name in config:
                kwargs[name] = config[name]
            elif name in cls.KWARGS_ALIAS:
                for alias_name in cls.KWARGS_ALIAS[name]:
                    if alias_name in config:
                        kwargs[name] = config[alias_name]
                        break

        return cls(**kwargs)
