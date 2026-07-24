# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import dataclasses
import re

import torch


@dataclasses.dataclass(kw_only=True, repr=False)
class DataType:
    num_bits: int
    is_signed: bool = True
    is_integer_type: bool = False
    is_floating_point_type: bool = False

    def __post_init__(self):
        assert self.__class__ is not DataType
        assert self.is_integer_type or self.is_floating_point_type

    @classmethod
    def from_str(cls, s):
        if isinstance(s, DataType):
            return s
        if "float" in s:
            return FloatingPointType.from_str(s)
        elif "int" in s:
            return IntegerType.from_str(s)
        else:
            raise NotImplementedError

    @classmethod
    def from_torch_dtype(cls, torch_dtype):
        if "float" in str(torch_dtype):
            return FloatingPointType.from_torch_dtype(torch_dtype)
        elif "int" in str(torch_dtype):
            return IntegerType.from_torch_dtype(torch_dtype)
        else:
            raise NotImplementedError

    def to_str(self):
        raise NotImplementedError

    def to_cpp_str(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __repr__(self):
        return self.to_str()

    def id(self):
        raise NotImplementedError


@dataclasses.dataclass(kw_only=True, repr=False)
class IntegerType(DataType):
    is_integer_type: bool = True

    __hash__ = DataType.__hash__

    @classmethod
    def from_str(cls, s):
        if isinstance(s, IntegerType):
            return s
        s = s.lower()
        re_res = re.findall("^(u*)int(\\d+)$", s)
        if not re_res:
            raise ValueError(f"invalid integer dtype: {s}")
        re_res = re_res[0]
        return cls(is_signed=re_res[0] == "", num_bits=int(re_res[1]))

    def to_str(self):
        s = "Int" if self.is_signed else "UInt"
        s += str(self.num_bits)
        return s.lower()

    def to_cpp_str(self):
        return "IntegerType<{is_signed}, {num_bits}>".format(
            is_signed=str(self.is_signed).lower(),
            num_bits=str(self.num_bits),
        )

    @classmethod
    def from_torch_dtype(cls, torch_dtype):
        assert isinstance(torch_dtype, torch.dtype)
        dtype_str = str(torch_dtype)[6:]
        assert "int" in dtype_str
        return cls.from_str(dtype_str)

    def id(self):
        dtype_id = 1 * 1e7  # int type
        dtype_id += self.num_bits * 1e5  # num_bits
        dtype_id += self.is_signed * 1e4  # is_sign
        return int(dtype_id)


@dataclasses.dataclass(kw_only=True, repr=False)
class FloatingPointType(DataType):
    exponent_bits: int
    mantissa_bits: int
    is_floating_point_type: bool = True

    __hash__ = DataType.__hash__

    def __post_init__(self):
        self.sign_bits = self.num_bits - self.exponent_bits - self.mantissa_bits
        assert self.sign_bits in (0, 1)
        self.is_signed = self.sign_bits != 0

    @classmethod
    def from_str(cls, s):
        if isinstance(s, FloatingPointType):
            return s
        s = s.lower()
        if s in ("float16", "half"):
            return cls(num_bits=16, exponent_bits=5, mantissa_bits=10)
        elif s == "bfloat16":
            return cls(num_bits=16, exponent_bits=8, mantissa_bits=7)
        elif s == "float32":
            return cls(num_bits=32, exponent_bits=8, mantissa_bits=23)

        re_res = re.findall("^float(\\d+)_*e(\\d+)m(\\d+)$", s)
        if not re_res:
            raise ValueError(f"invalid floating point dtype: {s}")

        re_res = re_res[0]
        return cls(
            num_bits=int(re_res[0]),
            exponent_bits=int(re_res[1]),
            mantissa_bits=int(re_res[2]),
        )

    def to_str(self):
        s = f"Float{self.num_bits}E{self.exponent_bits}M{self.mantissa_bits}"
        if s == "Float16E5M10":
            s = "Float16"
        elif s == "Float16E8M7":
            s = "BFloat16"
        elif s == "Float32E8M23":
            s = "Float32"

        return s.lower()

    def to_cpp_str(self):
        return "FloatingPointType<{num_bits}, {exponent_bits}, {mantissa_bits}>".format(
            num_bits=self.num_bits,
            exponent_bits=self.exponent_bits,
            mantissa_bits=self.mantissa_bits,
        )

    @classmethod
    def from_torch_dtype(cls, torch_dtype):
        assert isinstance(torch_dtype, torch.dtype)
        dtype_str = str(torch_dtype)[6:]
        # note that fnuz format / packed format / padded format are not supported
        dtype_str = dtype_str.replace("fnu", "").replace("fn", "")
        return cls.from_str(dtype_str)

    def id(self):
        dtype_id = 2 * 1e7  # int type
        dtype_id += self.num_bits * 1e5  # num_bits
        dtype_id += self.is_signed * 1e4  # is_sign
        dtype_id += self.exponent_bits * 1e2  # exp_bits
        dtype_id += self.mantissa_bits  # num_bits
        return int(dtype_id)


uint1 = IntegerType.from_str("uint1")
uint2 = IntegerType.from_str("uint2")
uint3 = IntegerType.from_str("uint3")
uint4 = IntegerType.from_str("uint4")
uint5 = IntegerType.from_str("uint5")
uint6 = IntegerType.from_str("uint6")
uint7 = IntegerType.from_str("uint7")
uint8 = IntegerType.from_str("uint8")

int2 = IntegerType.from_str("int2")
int3 = IntegerType.from_str("int3")
int4 = IntegerType.from_str("int4")
int6 = IntegerType.from_str("int6")
int8 = IntegerType.from_str("int8")
int32 = IntegerType.from_str("int32")

float4e0m3 = FloatingPointType.from_str("float4e0m3")
float4e2m1 = FloatingPointType.from_str("float4e2m1")
float6e2m3 = FloatingPointType.from_str("float6e2m3")
float6e3m2 = FloatingPointType.from_str("float6e3m2")
float8e3m4 = FloatingPointType.from_str("float8e3m4")
float8e4m3 = FloatingPointType.from_str("float8e4m3")
float8e5m2 = FloatingPointType.from_str("float8e5m2")
float8e8m0 = FloatingPointType.from_str("float8e8m0")

float16 = FloatingPointType.from_str("float16")
bfloat16 = FloatingPointType.from_str("bfloat16")
float32 = FloatingPointType.from_str("float32")


torch_dtype_map = {
    float8e8m0: torch.float8_e8m0fnu,
    float8e4m3: torch.float8_e4m3fn,
    float8e5m2: torch.float8_e5m2,
    float16: torch.float16,
    bfloat16: torch.bfloat16,
    float32: torch.float32,
}
