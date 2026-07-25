# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

from .. import dtypes
from ..config import GemmType
from .sm8x import Sm89Heuristics
from ..utils.smem import estimate_smem_size_layer


class Sm120Heuristics(Sm89Heuristics):
    sm_version: int = 120
    max_smem_size: int = 99 * 1024
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
        dtypes.float8e3m4,
    ]
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.float4e2m1, dtypes.float4e0m3]

    @classmethod
    def _is_mxmma(cls, a_dtype, group_size, use_fused_e8m0_scale) -> bool:
        return (
            a_dtype.is_floating_point_type
            and a_dtype.num_bits <= 8
            and group_size > 0
            and not use_fused_e8m0_scale
        )

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
        if cls._is_mxmma(a_dtype, group_size, use_fused_e8m0_scale):
            block_k = 512 // a_dtype.num_bits
            if a_dtype.num_bits == 8 and b_dtype.num_bits < a_dtype.num_bits:
                return {
                    "block_shape": (112, 256, block_k),
                    "warp_shape": (112, 32, block_k),
                    "num_stages": 2,
                }
            return {
                "block_shape": (256, 128, block_k),
                "warp_shape": (128, 32, block_k),
                "num_stages": 2,
            }
        if a_dtype.is_floating_point_type and a_dtype.num_bits == 16 and not use_f16_accum:
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (128, 32, 64),
                "num_stages": 2,
            }
        return super().get_base_config(
            a_dtype, b_dtype, group_size, use_f16_accum, use_fused_e8m0_scale, gemm_type, shape_k
        )

    @classmethod
    def get_config(
        cls,
        layer_config,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        config = super().get_config(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        if use_batch_invariant:
            return config

        a = layer_config.a_dtype
        is_wna16 = a.is_floating_point_type and a.num_bits == 16 and not use_f16_accum
        if a.is_floating_point_type and a.num_bits <= 8 and not layer_config.use_fused_e8m0_scale:
            config["use_tma"] = True
            config["use_warp_spec"] = True
        elif is_wna16:
            config["use_tma"] = True
            config["use_warp_spec"] = True
            config["num_stages"] = cls._fit_num_stages(layer_config, config, gemm_type, reduce_overlap=False)

        if gemm_type == GemmType.INDEXED:
            config["use_tma_a"] = False
            config["use_tma_c"] = False
            config["use_warp_spec"] = False

        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        if cls._is_mxmma(layer_config.a_dtype, group_size, layer_config.use_fused_e8m0_scale):
            config["reduce_overlap_last_stage_only"] = True
            config["num_stages"] = cls._fit_num_stages(layer_config, config, gemm_type, reduce_overlap=True)

        return config

    @classmethod
    def _fit_num_stages(cls, layer_config, config, gemm_type, reduce_overlap: bool) -> int:
        best = 2
        for num_stages in (3, 4):
            smem = estimate_smem_size_layer(
                layer_config,
                config["block_shape"],
                gemm_type,
                num_stages,
                warp_shape=config["warp_shape"],
                reduce_overlap_last_stage_only=reduce_overlap,
                use_mbarrier=True,
                use_warp_spec=True,
                num_write_splits=config.get("num_write_splits", 1),
            )
            if smem <= cls.max_smem_size:
                best = num_stages
        return best
