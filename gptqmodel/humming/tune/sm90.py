# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import math

import numpy as np

from .. import dtypes
from ..config import GemmType, LayerConfig
from .base import DeviceHeuristics


class Sm90Heuristics(DeviceHeuristics):
    max_smem_size: int = 227 * 1024
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    ]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 90

    @classmethod
    def get_config1(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if layer_config.use_packed_k_layout:
            max_block_m = 128
        elif use_f16_accum:
            max_block_m = 256
        else:
            max_block_m = 176

        num_blocks_list = cls.calc_num_block_list(layer_config, shape_m, max_block_m)
        block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8
        warp_shape_n = 32
        warp_shape_k = 1024 // layer_config.a_dtype.num_bits

        if layer_config.shape_n <= 4096 and not use_batch_invariant and block_shape_m <= 64:
            block_shape_n = 128
            block_shape_k = warp_shape_k * 2
            if block_shape_m <= 32:
                block_shape_k = block_shape_k * 2
            if block_shape_k > 256:
                block_shape_k = block_shape_k // 2
                warp_shape_k = warp_shape_k // 2

            while layer_config.shape_k % block_shape_k != 0:
                block_shape_k = block_shape_k // 2
        else:
            block_shape_n = 256
            block_shape_k = warp_shape_k
            if block_shape_m <= 32 and layer_config.b_dtype.num_bits <= 6:
                block_shape_k = block_shape_k * 2
            elif block_shape_m <= 32:
                warp_shape_k = warp_shape_k // 2

        while layer_config.shape_k % block_shape_k != 0:
            warp_shape_k = 512 // layer_config.a_dtype.num_bits
            block_shape_k = block_shape_k // 2
            assert block_shape_k >= warp_shape_k

        config = {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (block_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": not use_batch_invariant,
            "use_f16_accum": use_f16_accum,
            "num_stages": 4,
        }

        if gemm_type != GemmType.INDEXED:
            config["use_warp_spec"] = True
            config["use_tma"] = True
            config["use_mbarrier"] = True

            if layer_config.shape_n % (block_shape_n * 2) == 0 and shape_m / block_shape_m >= 4:
                if gemm_type == GemmType.DENSE:
                    config["multi_cast_size_a"] = 2

        return config

    @classmethod
    def get_config2(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if use_f16_accum:
            max_block_m = 256
        elif layer_config.input_scale_group_size > 0:
            max_block_m = 160
        elif layer_config.weight_scale_group_size < 128:
            max_block_m = 192
        else:
            max_block_m = 200

        num_blocks_list = cls.calc_num_block_list(layer_config, shape_m, max_block_m)
        block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8

        block_shape_k = 256 if block_shape_m <= 32 else 128
        if layer_config.shape_k % 256 != 0:
            block_shape_k = 128

        config = {
            "block_shape": (block_shape_m, 128, block_shape_k),
            "warp_shape": (block_shape_m, 16, 128),
            "use_stream_k": not use_batch_invariant,
            "use_f16_accum": use_f16_accum,
            "num_stages": 4,
        }

        if gemm_type != GemmType.INDEXED:
            config["use_warp_spec"] = True
            config["use_tma"] = True
            config["use_mbarrier"] = True

            if shape_m / block_shape_m >= 4 and gemm_type == GemmType.DENSE:
                config["multi_cast_size_a"] = 2

        return config

    @classmethod
    def calc_num_block_list(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        max_block_m: int,
    ):
        num_blocks_list = []
        if not layer_config.num_experts:
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks_list.append(math.ceil(shape_m / block_m))
        else:
            random_state = np.random.RandomState(seed=0)
            samples = random_state.randint(0, layer_config.num_experts, size=shape_m)
            counts = np.bincount(samples)
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks = int(np.ceil(counts * 1.1 / block_m).sum().item())
                num_blocks_list.append(num_blocks)

        for i in range(max_block_m // 8):
            num_blocks = num_blocks_list[i]
            block_m = i * 8 + 8
            if layer_config.a_dtype == dtypes.int8 and block_m % 16 == 8 and block_m > 32:
                num_blocks_list[i] = 1000000

        return num_blocks_list

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if layer_config.a_dtype.num_bits == 16:
            func = cls.get_config1
        elif layer_config.use_packed_k_layout:
            func = cls.get_config1
        elif layer_config.input_scale_group_size == 0 and layer_config.weight_scale_group_size == 0:
            func = cls.get_config1
        elif layer_config.use_fused_e8m0_scale and layer_config.input_scale_group_size == 0:
            func = cls.get_config1
        else:
            func = cls.get_config2

        return func(layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type)
