# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

from .. import dtypes
from ..config import GemmType
from .base import DeviceHeuristics


class Sm80Heuristics(DeviceHeuristics):
    max_smem_size: int = 163 * 1024
    sm_version: int = 80
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8]
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.int4]

    @classmethod
    def get_base_config(
        cls,
        a_dtype: dtypes.DataType,
        b_dtype: dtypes.DataType,
        group_size: int,
        use_f16_accum: bool,
        use_fused_e8m0_scale: bool,
        gemm_type: GemmType,
        shape_k: int,
    ):
        if a_dtype.num_bits == 16 or group_size == 0 and a_dtype == b_dtype:
            if use_f16_accum:
                return {
                    "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                    "warp_shape": (128, 64, 512 // a_dtype.num_bits),
                }
            else:
                return {
                    "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                    "warp_shape": (64, 64, 1024 // a_dtype.num_bits),
                }
        elif group_size == 0 and a_dtype != b_dtype:
            return {
                "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                "warp_shape": (128, 32, 1024 // a_dtype.num_bits),
            }
        elif group_size > 0:
            return {
                "block_shape": (64, 128, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 32, 1024 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }


class Sm86Heuristics(DeviceHeuristics):
    max_smem_size: int = 99 * 1024
    sm_version: int = 86
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8]
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.int4]

    @classmethod
    def get_base_config(
        cls,
        a_dtype: dtypes.DataType,
        b_dtype: dtypes.DataType,
        group_size: int,
        use_f16_accum: bool,
        use_fused_e8m0_scale: bool,
        gemm_type: GemmType,
        shape_k: int,
    ):
        if a_dtype.num_bits == 16 and use_f16_accum:
            num_stages = 3 if b_dtype.num_bits < 8 else 2
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (64, 64, 64),
                "num_stages": num_stages,
            }
        elif a_dtype.num_bits == 16 and not use_f16_accum:
            num_stages = 3 if b_dtype.num_bits < 8 else 2
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (64, 64, 64),
            }
        elif use_fused_e8m0_scale:
            return {
                "block_shape": (128, 128, 512 // a_dtype.num_bits),
                "warp_shape": (128, 32, 512 // a_dtype.num_bits),
                "num_stages": 2,
                "num_warps_per_sm": 2,
            }
        elif group_size == 0 and a_dtype == b_dtype:
            return {
                "block_shape": (128, 256, 512 // a_dtype.num_bits),
                "warp_shape": (64, 64, 512 // a_dtype.num_bits),
                "num_stages": 2,
            }
        elif group_size == 0 and a_dtype != b_dtype:
            return {
                "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                "warp_shape": (128, 32, 1024 // a_dtype.num_bits),
                "num_stages": 2,
            }
        else:
            return {
                "block_shape": (64, 256, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 32, 1024 // a_dtype.num_bits),
                "num_stages": 2,
            }


class Sm87Heuristics(Sm80Heuristics):
    sm_version: int = 87


class Sm89Heuristics(Sm86Heuristics):
    sm_version: int = 89
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]
