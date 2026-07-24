# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import dataclasses
import math
from typing import ClassVar

import torch

from .. import dtypes
from .base import BaseHummingConfig
from .enum import GemmType, MmaType, WeightScale2Type, WeightScaleType


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class LayerConfig(BaseHummingConfig):
    # shape config
    shape_n: int
    shape_k: int
    pad_shape_n: int = 0
    pad_shape_k: int = 0
    num_experts: int = 0

    # datatype config
    b_dtype: dtypes.DataType
    a_dtype: dtypes.DataType
    c_dtype: dtypes.DataType
    bs_dtype: dtypes.DataType | None = None
    as_dtype: dtypes.DataType | None = None

    # quant param config
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 0
    weight_scale_group_size_n: int = 0
    weight_scale_type: WeightScaleType | None = None
    weight_scale_2_type: WeightScale2Type | None = None
    use_int_weight_scale: bool | None = None
    use_fused_e8m0_scale: bool | None = None
    has_zero_point: bool = False
    is_fp_zero_point: bool = False

    # bias config
    has_bias: bool = False

    # mma config
    mma_type: MmaType | None = None

    # packed-K layout (wgmma + 8-bit activation only)
    use_packed_k_layout: bool | None = None

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "mma_type_id",
        "is_channel_weight_scale",
        "is_block_weight_scale",
        "is_group_weight_scale",
        "is_tensor_weight_scale",
        "is_channel_weight_scale_2",
        "is_tensor_weight_scale_2",
        "has_channel_weight_scale",
        "has_tensor_weight_scale",
        "has_input_scale",
    )

    @property
    def mxmma_supported(self):
        if torch.cuda.get_device_capability()[0] != 12:
            return False
        if not self.is_group_weight_scale:
            return False
        if self.a_dtype in (dtypes.float8e4m3, dtypes.float8e5m2, dtypes.float8e3m4):
            return self.weight_scale_group_size == 32 and self.bs_dtype == dtypes.float8e8m0
        if self.a_dtype in (dtypes.float4e2m1, dtypes.float4e0m3):
            if self.a_dtype == dtypes.float4e0m3 and self.weight_scale_group_size == 32:
                return False

            return (
                self.weight_scale_group_size == 16
                and self.bs_dtype in (dtypes.float8e8m0, dtypes.float8e4m3)
                or self.weight_scale_group_size == 32
                and self.bs_dtype == dtypes.float8e8m0
            )

        return False

    def __post_init__(self):
        self.problem_shape = (0, self.shape_n, self.shape_k)
        self.pad_shape = (0, self.pad_shape_n, self.pad_shape_k)

        if isinstance(self.weight_scale_type, str):
            self.weight_scale_type = WeightScaleType(self.weight_scale_type)
        if self.weight_scale_type is None:
            if self.weight_scale_group_size_n > 1:
                self.weight_scale_type = WeightScaleType.BLOCK
            elif self.weight_scale_group_size == 0:
                self.weight_scale_type = WeightScaleType.CHANNEL
            else:
                self.weight_scale_type = WeightScaleType.GROUP

        if isinstance(self.weight_scale_2_type, str):
            self.weight_scale_2_type = WeightScale2Type(self.weight_scale_2_type)
        if self.weight_scale_2_type is None:
            self.weight_scale_2_type = WeightScale2Type.NONE
        if self.weight_scale_2_type != WeightScale2Type.NONE:
            assert self.weight_scale_type == WeightScaleType.GROUP, (
                "weight_scale_2_type requires weight_scale_type='group'"
            )

        if self.weight_scale_type in [WeightScaleType.CHANNEL, WeightScaleType.TENSOR]:
            assert self.weight_scale_group_size == 0, (
                f"{self.weight_scale_type} requires weight_scale_group_size=0"
            )
        elif self.weight_scale_type in [WeightScaleType.GROUP, WeightScaleType.BLOCK]:
            assert self.weight_scale_group_size > 0, (
                f"{self.weight_scale_type} requires weight_scale_group_size>0"
            )

        for name in ["a", "b", "c", "bs", "as"]:
            value = getattr(self, f"{name}_dtype")
            if isinstance(value, str):
                value = dtypes.DataType.from_str(value)
            setattr(self, f"{name}_dtype", value)

        self.has_input_scale = self.a_dtype.num_bits != 16
        self.bs_dtype = self.bs_dtype or self.c_dtype

        if isinstance(self.b_dtype, dtypes.IntegerType):
            if isinstance(self.a_dtype, dtypes.FloatingPointType):
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=False)
            elif self.a_dtype.num_bits == self.b_dtype.num_bits:
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=True)
            else:
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=False)

        self._update_weight_scale_flags()

        if isinstance(self.mma_type, str):
            self.mma_type = MmaType(self.mma_type)
        elif self.mma_type is None:
            sm_version = torch.cuda.get_device_capability()[0]
            if sm_version == 9:
                self.mma_type = MmaType.WGMMA
            elif self.mxmma_supported:
                self.mma_type = MmaType.MXMMA
            else:
                self.mma_type = MmaType.MMA

        if not self.has_input_scale:
            self.as_dtype = None
        elif self.as_dtype is None:
            self.as_dtype = self.bs_dtype if self.mma_type == MmaType.MXMMA else dtypes.float32

        is_channel_scale_2 = self.weight_scale_2_type == WeightScale2Type.CHANNEL

        if self.use_fused_e8m0_scale is None:
            has_native_mxf8f6f4 = self.mma_type == MmaType.MXMMA and self.a_dtype == dtypes.float8e4m3
            self.use_fused_e8m0_scale = (
                not has_native_mxf8f6f4
                and self.a_dtype in [dtypes.float8e4m3, dtypes.int8]
                and self.b_dtype in [dtypes.float4e2m1]
                and self.bs_dtype in [dtypes.float8e8m0]
                and self.weight_scale_group_size > 0
            )

        if self.use_int_weight_scale is None:
            self.use_int_weight_scale = (
                not self.use_fused_e8m0_scale
                and self.a_dtype in [dtypes.int8, dtypes.int4]
                and not is_channel_scale_2
                and self.input_scale_group_size == 0
                and self.weight_scale_group_size > 0
                and self.weight_scale_group_size_n == 1
            )

        if self.use_int_weight_scale:
            assert not is_channel_scale_2, "use_int_weight_scale is incompatible with channel weight_scale_2"
            self.bs_dtype = self.c_dtype

        if self.use_int_weight_scale or self.use_fused_e8m0_scale:
            self.weight_scale_type = WeightScaleType.GROUP
            # the extracted min-exponent factor becomes the secondary scale:
            # per-channel when channel2 is requested, per-tensor otherwise
            if not is_channel_scale_2:
                self.weight_scale_2_type = WeightScale2Type.TENSOR
            self._update_weight_scale_flags()

        if self.use_packed_k_layout is None:
            self.use_packed_k_layout = (
                self.mma_type == MmaType.WGMMA
                and self.a_dtype.num_bits == 8
                and not self.use_fused_e8m0_scale
                and self.weight_scale_group_size == 128
            )
        elif self.use_packed_k_layout:
            assert self.mma_type == MmaType.WGMMA, "use_packed_k_layout requires wgmma"
            assert self.a_dtype.num_bits == 8, "use_packed_k_layout requires 8-bit activation"
            assert not self.use_fused_e8m0_scale, "packed_k_layout is incompatible with fused-e8m0"

        if type(self) is LayerConfig:
            self._config_str = self.to_str()

    def _update_weight_scale_flags(self):
        self.is_group_weight_scale = self.weight_scale_type == WeightScaleType.GROUP
        self.is_block_weight_scale = self.weight_scale_type == WeightScaleType.BLOCK
        self.is_channel_weight_scale = self.weight_scale_type == WeightScaleType.CHANNEL
        self.is_tensor_weight_scale = self.weight_scale_type == WeightScaleType.TENSOR
        self.is_channel_weight_scale_2 = self.weight_scale_2_type == WeightScale2Type.CHANNEL
        self.is_tensor_weight_scale_2 = self.weight_scale_2_type == WeightScale2Type.TENSOR
        self.has_channel_weight_scale = self.is_channel_weight_scale or self.is_channel_weight_scale_2  # noqa
        self.has_tensor_weight_scale = self.is_tensor_weight_scale or self.is_tensor_weight_scale_2

    def to_str(self) -> str:
        if hasattr(self, "_config_str"):
            return self._config_str
        return super().to_str()

    def __setattr__(self, name, value):
        if hasattr(self, "_config_str"):
            raise AttributeError(f"Instance is frozen, cannot set {name}")
        super().__setattr__(name, value)

    def estimate_bound_min_shape_m(self, use_f16_accum: bool = False):
        from ..utils.device import estimate_compute_bound_threshold

        return estimate_compute_bound_threshold(
            weight_nbytes=self.weight_nbytes // (self.num_experts or 1),
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            dtype=str(self.a_dtype),
            use_f16_accum=use_f16_accum,
        )

    @property
    def mma_type_id(self):
        assert self.mma_type is not None
        value = self.mma_type.value.lower()
        return ["mma", "wgmma", "umma_placeholder", "mxmma"].index(value)

    @property
    def mxmma_native_mixed(self) -> bool:
        if self.mma_type != MmaType.MXMMA:
            return False
        if self.a_dtype not in (dtypes.float8e4m3, dtypes.float8e5m2):
            return False
        return self.b_dtype in (dtypes.float4e2m1, dtypes.float6e3m2, dtypes.float6e2m3)

    @property
    def weight_nbytes(self):
        nbytes1 = self.shape_n * self.shape_k * self.b_dtype.num_bits // 8
        num_groups = self.shape_k / (self.weight_scale_group_size or self.shape_k)
        assert self.bs_dtype is not None
        nbytes2 = self.shape_n * num_groups * self.bs_dtype.num_bits // 8
        nbytes3 = self.shape_n * num_groups * (math.ceil(self.b_dtype.num_bits / 4) * 4) // 8
        nbytes = nbytes1 + nbytes2
        if self.has_zero_point and self.is_fp_zero_point:
            nbytes = nbytes + nbytes2
        elif self.has_zero_point:
            nbytes = nbytes + nbytes3
        return nbytes * (self.num_experts or 1)

    @property
    def param_dtype(self):
        if self.c_dtype == dtypes.float16:
            return torch.float16
        elif self.c_dtype == dtypes.bfloat16:
            return torch.bfloat16
        else:
            raise ValueError(f"unsupported c_dtype: {self.c_dtype}")

    @property
    def should_apply_bs_on_c(self):
        if self.use_fused_e8m0_scale:
            return False
        elif self.mma_type == MmaType.MMA:
            return self.weight_scale_group_size == 0 or self.a_dtype.num_bits != 16
        elif self.mma_type == MmaType.WGMMA:
            return self.weight_scale_group_size == 0
        elif self.mma_type == MmaType.MXMMA:
            return False
        else:
            raise ValueError(f"unsupported mma_type: {self.mma_type}")


@dataclasses.dataclass(kw_only=True)
class ComputeConfig(BaseHummingConfig):
    use_f16_accum: bool = False
    use_batch_invariant: bool = False
    use_m_major_input_scale: bool = False
    gemm_type: GemmType | None = None

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "gemm_type_id",
        "is_indexed_gemm",
        "is_grouped_gemm",
        "is_grouped_contiguous_gemm",
        "is_grouped_masked_gemm",
    )

    def __post_init__(self):
        if isinstance(self.gemm_type, str):
            self.gemm_type = GemmType(self.gemm_type)
        self.is_indexed_gemm = self.gemm_type == GemmType.INDEXED
        self.is_grouped_contiguous_gemm = self.gemm_type == GemmType.GROUPED_CONTIGUOUS
        self.is_grouped_masked_gemm = self.gemm_type == GemmType.GROUPED_MASKED
        self.is_grouped_gemm = self.is_grouped_contiguous_gemm or self.is_grouped_masked_gemm

    @property
    def gemm_type_id(self):
        assert self.gemm_type is not None
        value = self.gemm_type.value.lower()
        return ["dense", "indexed", "grouped_contiguous", "grouped_masked"].index(value)


@dataclasses.dataclass(kw_only=True)
class TuningConfig(BaseHummingConfig):
    block_shape: tuple[int, int, int]
    warp_shape: tuple[int, int, int]

    use_stream_k: bool = True

    num_stages: int = 2
    num_ctas_per_sm: int = 1

    use_warp_spec: bool | None = None
    use_mbarrier: bool | None = None
    use_cp_async: bool | None = None

    use_tma: bool | None = None
    use_tma_a: bool | None = None
    use_tma_as: bool | None = None
    use_tma_b: bool | None = None
    use_tma_c: bool | None = None
    use_tma_bs: bool | None = None
    use_tma_bs2: bool | None = None
    use_tma_bzp: bool | None = None
    use_tma_bias: bool | None = None

    reduce_overlap_last_stage_only: bool = False

    num_write_splits: int = 1
    multi_cast_size_a: int = 1
    multi_cast_size_b: int = 1

    raster_group_m: int = 1

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "num_threads",
        "num_math_threads",
        "num_load_threads",
    )

    _name_map = {
        "use_mbarrier": "kUseMBarrier",
        "use_tma_as": "kUseTmaAS",
        "use_tma_bs": "kUseTmaBS",
        "use_tma_bs2": "kUseTmaBS2",
        "use_tma_bzp": "kUseTmaBZP",
    }

    def __post_init__(self):
        if self.use_warp_spec is None:
            self.use_warp_spec = False

        if self.use_tma is None:
            self.use_tma = False

        if self.use_mbarrier is None:
            self.use_mbarrier = self.use_tma or self.use_warp_spec

        if self.use_cp_async is None:
            sm_version = torch.cuda.get_device_capability()
            self.use_cp_async = sm_version[0] >= 8

        self.num_math_threads = math.prod(self.block_shape) // math.prod(self.warp_shape) * 32
        if self.use_warp_spec:
            self.num_load_threads = 128
            self.num_threads = self.num_math_threads + 128
        else:
            self.num_load_threads = self.num_math_threads
            self.num_threads = self.num_math_threads

        if self.use_tma_as is None:
            self.use_tma_as = False

        for name in dir(self):
            if not name.startswith("use_tma_"):
                continue
            if not self.use_tma:
                assert getattr(self, name) is not True
            if getattr(self, name) is None:
                setattr(self, name, self.use_tma)
