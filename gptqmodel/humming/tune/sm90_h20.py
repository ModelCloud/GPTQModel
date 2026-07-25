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
from ..utils.smem import estimate_smem_size_layer


class Sm90H20Heuristics(DeviceHeuristics):
    max_smem_size: int = 227 * 1024
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 90

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
        is_moe = gemm_type != GemmType.DENSE
        if a_dtype.num_bits == 16:
            return {
                "block_shape": (64, 256, 512 // a_dtype.num_bits),
                "warp_shape": (64, 64, 512 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }
        elif use_fused_e8m0_scale and not is_moe:
            return {
                "block_shape": (128, 128, 1024 // a_dtype.num_bits),
                "warp_shape": (128, 32, 1024 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }
        elif use_fused_e8m0_scale and is_moe:
            return {
                "block_shape": (64, 128, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 32, 1024 // a_dtype.num_bits),
                "num_ctas_per_sm": 3,
            }
        elif group_size == 0 and not is_moe:
            return {
                "block_shape": (64, 256, 512 // a_dtype.num_bits),
                "warp_shape": (64, 64, 512 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }
        elif group_size == 0 and is_moe:
            return {
                "block_shape": (64, 128, 512 // a_dtype.num_bits),
                "warp_shape": (64, 32, 512 // a_dtype.num_bits),
                "num_ctas_per_sm": 3,
            }
        elif group_size >= 128 and shape_k > 512:
            return {
                "block_shape": (64, 128, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 16, 1024 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }
        else:
            return {
                "block_shape": (64, 128, 512 // a_dtype.num_bits),
                "warp_shape": (64, 32, 512 // a_dtype.num_bits),
                "num_ctas_per_sm": 3 if is_moe else 2,
            }

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        # 1. base config
        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        is_moe = gemm_type != GemmType.DENSE
        config = cls.get_base_config(
            layer_config.a_dtype,
            layer_config.b_dtype,
            group_size,
            use_f16_accum,
            layer_config.use_fused_e8m0_scale,
            gemm_type,
            layer_config.shape_k,
        )
        block_shape_m, block_shape_n, block_shape_k = config["block_shape"]
        num_ctas_per_sm = config.get("num_ctas_per_sm", 1)
        warp_shape_m, warp_shape_n, warp_shape_k = config["warp_shape"]
        if layer_config.use_packed_k_layout:
            warp_shape_n = max(warp_shape_n, 32)
        num_stages = 3
        assert layer_config.shape_n % block_shape_n == 0

        # 2. block_shape_m and warp_shape_m
        if not layer_config.num_experts:
            if shape_m <= block_shape_m:
                block_shape_m = math.ceil(shape_m / 8) * 8
            else:
                blocks = [math.ceil(shape_m / ((i + 1) * 8)) for i in range(block_shape_m // 8)]
                block_shape_m = np.argmin(blocks).item() * 8 + 8
            if layer_config.a_dtype == dtypes.int8 and block_shape_m > 32 and block_shape_m % 16 != 0:
                block_shape_m = math.ceil(block_shape_m / 16) * 16
        else:
            block_size_configs = [(8, 0.7), (16, 0.8), (32, 0.9), (48, 0.9), (64, 0.9)]
            for moe_block_size, threshold in block_size_configs:
                if shape_m / layer_config.num_experts / moe_block_size < threshold:
                    break

            new_shape_m = int(shape_m / layer_config.num_experts / 0.9)
            new_shape_m = max(new_shape_m, 1)
            if block_shape_m == 128:
                if np.ceil(new_shape_m / 96) * 96 < np.ceil(new_shape_m / 64) * 64:
                    block_shape_m = 96
                elif np.ceil(new_shape_m / 128) * 128 < np.ceil(new_shape_m / 64) * 64 * 1.05:
                    block_shape_m = 128
                else:
                    block_shape_m = moe_block_size
            elif new_shape_m >= 64 and new_shape_m < 96:
                block_shape_m = 48
            else:
                block_shape_m = moe_block_size

        warp_shape_m = block_shape_m
        num_blocks_n = layer_config.shape_n // block_shape_n
        num_blocks_m = cls.estimate_num_blocks_m(layer_config, shape_m, block_shape_m)

        num_sms = cls.get_num_sms()
        while num_blocks_n * num_blocks_m * 2 < num_sms * num_ctas_per_sm:
            if warp_shape_n == 64:
                warp_shape_n = warp_shape_n // 2
                block_shape_n = block_shape_n // 2
                num_blocks_n = num_blocks_n * 2
                if num_ctas_per_sm == 2:
                    num_ctas_per_sm = 3
                continue
            elif num_ctas_per_sm > 1:
                num_ctas_per_sm = num_ctas_per_sm - 1
                continue
            else:
                break

        num_warps_m = block_shape_m // warp_shape_m
        num_warps_n = block_shape_n // warp_shape_n
        num_warps_k = block_shape_k // warp_shape_k
        num_warps = num_warps_m * num_warps_n * num_warps_k * num_ctas_per_sm

        if num_warps == 4:
            warp_shape_k = 512 // layer_config.a_dtype.num_bits
            block_shape_k = warp_shape_k * 2

        if num_warps <= 8 and block_shape_m <= 32:
            if is_moe and warp_shape_n == 64:
                warp_shape_n = warp_shape_n // 2
            else:
                num_warps_k = block_shape_k // warp_shape_k
                warp_shape_k = 512 // layer_config.a_dtype.num_bits
                block_shape_k = warp_shape_k * num_warps_k * 2

        if is_moe and layer_config.shape_k <= 512 and layer_config.shape_n >= 2048 and block_shape_m <= 32:
            if block_shape_n == 256:
                warp_shape_n = 32
                block_shape_n = 128
                num_blocks_n = num_blocks_n * 2

            if num_blocks_n * num_blocks_m >= num_sms * 4:
                num_ctas_per_sm = 4

        if warp_shape_k == block_shape_k and warp_shape_k == 512 // layer_config.a_dtype.num_bits:
            block_shape = (block_shape_m, block_shape_n, block_shape_k * 2)
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages)
            if smem_size * num_ctas_per_sm < cls.max_smem_size:
                block_shape_k = block_shape_k * 2
                warp_shape_k = warp_shape_k * 2

        max_num_stages = 4
        for num_stages_new in range(num_stages + 1, max_num_stages + 1):
            block_shape = (block_shape_m, block_shape_n, block_shape_k)
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages_new)
            if smem_size * num_ctas_per_sm < cls.max_smem_size:
                num_stages = num_stages_new

        if num_ctas_per_sm == 1:
            factor = min(4.5, layer_config.shape_k / (3 * block_shape_k))
            num_sms = min(num_sms, math.ceil(num_blocks_n * num_blocks_m * factor))

        while layer_config.shape_k % block_shape_k != 0:
            warp_shape_k = 512 // layer_config.a_dtype.num_bits
            block_shape_k = block_shape_k // 2
            assert block_shape_k >= warp_shape_k

        if (
            layer_config.a_dtype.num_bits == 8
            and layer_config.input_scale_group_size > 0
            and gemm_type != GemmType.GROUPED_MASKED
            and shape_m >= 6144
        ):
            num_ctas_per_sm = min(num_ctas_per_sm, 2)

        config = {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": layer_config.shape_k > 1024,
            "use_f16_accum": use_f16_accum,
            "num_sms": num_sms,
            "num_stages": num_stages,
            "num_ctas_per_sm": num_ctas_per_sm,
        }

        if layer_config.shape_k <= 512 and is_moe and shape_m >= 2048:
            config["use_tma"] = True
            config["use_mbarrier"] = True
            if gemm_type == GemmType.INDEXED:
                config["use_tma_a"] = False
                config["use_tma_c"] = False

            # Small-K MoE down-gemm: size the persistent grid for ~5 output tiles
            # per CTA — here num_sms is the grid factor (a launch param, GEMM
            # bit-identical) and is intentionally set above the physical SM count.
            if config["num_ctas_per_sm"] > 1 and shape_m >= 24576:
                tiles_per_cta = 5
                block_m, block_n, _ = config["block_shape"]
                num_tiles = (layer_config.shape_n // block_n) * (shape_m // block_m)
                sms_target = num_tiles / (config["num_ctas_per_sm"] * tiles_per_cta)
                config["num_sms"] = max(config["num_sms"], 1 << round(math.log2(sms_target)))

        if block_shape_m >= 48 and num_ctas_per_sm <= 2 and num_warps <= 8 and not is_moe:
            config["use_tma"] = True
            config["use_warp_spec"] = True
            config["use_mbarrier"] = True
            config["num_stages"] = 3
        elif config["num_stages"] == 4 and block_shape_m <= 32:
            block_shape = (block_shape_m, block_shape_n, block_shape_k)
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, 5)
            if smem_size * num_ctas_per_sm < cls.max_smem_size:
                config["num_stages"] = 5

        if use_batch_invariant:
            warp_shape_k = 512 // layer_config.a_dtype.num_bits
            block_shape_k = 512 // layer_config.a_dtype.num_bits
            # TODO: check if TMA / cp.async affect batch invariance
            config["use_tma"] = False
            config["use_warp_spec"] = False
            config["use_mbarrier"] = False
            config["use_stream_k"] = False

        return config
