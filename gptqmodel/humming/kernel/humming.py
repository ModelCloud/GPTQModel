# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import dataclasses
import functools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import jinja2
import torch

from ..utils import jit as jit_utils
from .. import dtypes
from ..config import (
    ComputeConfig,
    GemmType,
    LayerConfig,
    MmaOpClass,
    MmaType,
    TuningConfig,
)
from ..jit.runtime import KernelRuntime
from ..tune import get_heuristics_config

CODE_TEMPLATE = jinja2.Template("""

{{layer_config_macro}}

{{compute_config_macro}}

{{tuning_config_macro}}

#if {{use_warp_spec}}
#include <humming/kernel/humming_ws.cuh>
#else
#include <humming/kernel/humming.cuh>
#endif

class MmaOpClass {
public:
{{mma_op_class}}
};

class LayerConfig {
public:
{{layer_config}}
};

class ComputeConfig {
public:
{{compute_config}}
};

class TuningConfig {
public:
{{tuning_config}}
};

using SharedStorageType = SharedStorage<
    MmaOpClass,
    Shape<{{block_shape[0]}}, {{block_shape[1]}}, {{block_shape[2]}}>,
    Shape<{{warp_shape[0]}}, {{warp_shape[1]}}, {{warp_shape[2]}}>,
    {{a_dtype}},
    {{b_dtype}},
    {{bs_dtype}},
    LayerConfig,
    ComputeConfig,
    TuningConfig>;



extern "C" __constant__ uint32_t SMEM_SIZE = sizeof(SharedStorageType);
extern "C" __constant__ uint32_t SMEM_SIZE_A = 
    SharedStorageType::kNumStages * SharedStorageType::kStageSizeA * sizeof(int4);
extern "C" __constant__ uint32_t SMEM_SIZE_B = 
    SharedStorageType::kNumStages * SharedStorageType::kStageSizeB * sizeof(int4);
extern "C" __constant__ uint32_t SMEM_SIZE_REDUCE = sizeof(SharedStorageType::reduce);

extern "C" __constant__ uint32_t PROBLEM_SHAPE_N = {{problem_shape[1]}};
extern "C" __constant__ uint32_t PROBLEM_SHAPE_K = {{problem_shape[2]}};

extern "C" __constant__ uint32_t BLOCK_SHAPE_M = {{block_shape[0]}};
extern "C" __constant__ uint32_t BLOCK_SHAPE_N = {{block_shape[1]}};
extern "C" __constant__ uint32_t BLOCK_SHAPE_K = {{block_shape[2]}};

extern "C" __constant__ uint32_t WARP_SHAPE_M = {{warp_shape[0]}};
extern "C" __constant__ uint32_t WARP_SHAPE_N = {{warp_shape[1]}};
extern "C" __constant__ uint32_t WARP_SHAPE_K = {{warp_shape[2]}};

extern "C" __constant__ uint32_t A_DTYPE_ID = {{a_dtype}}::kId;
extern "C" __constant__ uint32_t B_DTYPE_ID = {{b_dtype}}::kId;
extern "C" __constant__ uint32_t C_DTYPE_ID = {{c_dtype}}::kId;
extern "C" __constant__ uint32_t BS_DTYPE_ID = {{bs_dtype}}::kId;

{{layer_config_extern}}

{{compute_config_extern}}

{{tuning_config_extern}}

""")


@dataclasses.dataclass(kw_only=True)
class HummingKernel(KernelRuntime, LayerConfig, ComputeConfig, TuningConfig):
    name: ClassVar[str] = "humming"
    _str2kernel_cache: ClassVar[dict[tuple[str, str, str], int | list[int]]] = {}
    _id2kernel: ClassVar[dict[int, "HummingKernel"]] = {}

    def __post_init__(self):
        LayerConfig.__post_init__(self)
        ComputeConfig.__post_init__(self)
        TuningConfig.__post_init__(self)
        KernelRuntime.__post_init__(self)

    def init_kernel(self) -> None:
        self.check_shape()
        self.check_dtype()
        self.check_scale()
        self.check_config()
        self.mma_op_class = self.select_mma_op_class()

        assert self.bs_dtype is not None
        self.code = CODE_TEMPLATE.render(
            use_warp_spec=int(self.use_warp_spec or False),
            mma_op_class=self.mma_op_class.to_cpp_str(),
            problem_shape=self.problem_shape,
            pad_shape=self.pad_shape,
            block_shape=self.block_shape,
            warp_shape=self.warp_shape,
            layer_config=self.to_cpp_str(LayerConfig),
            compute_config=self.to_cpp_str(ComputeConfig),
            tuning_config=self.to_cpp_str(TuningConfig),
            layer_config_extern=self.to_extern_cpp_str(LayerConfig),
            compute_config_extern=self.to_extern_cpp_str(ComputeConfig),
            tuning_config_extern=self.to_extern_cpp_str(TuningConfig),
            layer_config_macro=self.to_macro_cpp_str(LayerConfig),
            compute_config_macro=self.to_macro_cpp_str(ComputeConfig),
            tuning_config_macro=self.to_macro_cpp_str(TuningConfig),
            a_dtype=self.a_dtype.to_cpp_str(),
            b_dtype=self.b_dtype.to_cpp_str(),
            c_dtype=self.c_dtype.to_cpp_str(),
            bs_dtype=self.bs_dtype.to_cpp_str(),
        )
        self.kernel_expr = (
            f"humming<\n"
            f"    MmaOpClass,\n"
            f"    Shape<0, {self.problem_shape[1]}, {self.problem_shape[2]}>,\n"
            f"    Shape<{self.block_shape[0]}, {self.block_shape[1]}, {self.block_shape[2]}>,\n"
            f"    Shape<{self.warp_shape[0]}, {self.warp_shape[1]}, {self.warp_shape[2]}>,\n"
            f"    Shape<0, {self.pad_shape[1]}, {self.pad_shape[2]}>,\n"
            f"    {self.a_dtype.to_cpp_str()},\n"
            f"    {self.b_dtype.to_cpp_str()},\n"
            f"    {self.c_dtype.to_cpp_str()},\n"
            f"    {self.bs_dtype.to_cpp_str()},\n"
            f"    LayerConfig,\n"
            f"    ComputeConfig,\n"
            f"    TuningConfig>"
        )

        self.prepare()

    def load_cubin(self):
        from .. import ops

        if self.cubin_loaded:
            return None
        kernel_filename = self.kernel_filename
        self.kernel_id, self.kernel_name = ops.register_kernel(kernel_filename)
        self._id2kernel[self.kernel_id] = self
        self.kernel_dirname = os.path.dirname(kernel_filename)
        self.cubin_loaded = True

    @functools.cached_property
    def get_kernel_id(self):
        module = jit_utils.make_humming_module("get_kernel_id", self.kernel_id)
        return module.get_kernel_id

    def postprocess_cubin(self, cubin_path: str):
        mode = ""
        if dtypes.float8e3m4 in (self.mma_a_dtype, self.mma_b_dtype):
            if self.mma_a_dtype != dtypes.float8e3m4:
                mode = "mma_e3m4_b"
            elif self.mma_b_dtype != dtypes.float8e3m4:
                mode = "mma_e3m4_a"
            else:
                mode = "mma_e3m4_ab"
        elif dtypes.float4e0m3 in (self.mma_a_dtype, self.mma_b_dtype):
            if self.mma_a_dtype != dtypes.float4e0m3:
                mode = "mma_e0m3_b"
            elif self.mma_b_dtype != dtypes.float4e0m3:
                mode = "mma_e0m3_a"
            else:
                mode = "mma_e0m3_ab"

        if mode:
            from ..utils.cubin import patch_cubin

            patch_cubin(cubin_path=cubin_path, mode=mode)

    def select_mma_op_class(self):
        self.mma_a_dtype = self.a_dtype
        self.mma_b_dtype = self.a_dtype
        if self.mma_type == MmaType.MXMMA:
            mma_shape_m, mma_shape_n = 16, 8
            mma_shape_k = 256 // self.a_dtype.num_bits
            assert self.warp_shape[0] % mma_shape_m == 0
            assert self.warp_shape[1] % mma_shape_n == 0
            assert self.warp_shape[2] % mma_shape_k == 0
            group = self.weight_scale_group_size or mma_shape_k
            assert mma_shape_k % group == 0
            scale_vec = mma_shape_k // group

            self.mma_b_dtype = self.b_dtype if self.mxmma_native_mixed else self.a_dtype
            return MmaOpClass.from_config(
                self.mma_type,
                mma_shape_m,
                mma_shape_n,
                mma_shape_k,
                self.mma_a_dtype,
                self.mma_b_dtype,
                dtypes.float32,
                sf_dtype=self.bs_dtype,
                scale_vec=scale_vec,
            )

        if self.a_dtype in [dtypes.int4, dtypes.int8]:
            mma_cd_dtype = dtypes.int32
        elif self.use_f16_accum:
            mma_cd_dtype = self.c_dtype
        else:
            mma_cd_dtype = dtypes.float32

        mma_shape_m = self.warp_shape[0] if self.mma_type == MmaType.WGMMA else 16
        mma_shape_n = 64 if self.mma_type == MmaType.WGMMA else 8
        mma_shape_k = 256 // self.a_dtype.num_bits
        if self.sm_version == 75 and self.a_dtype == dtypes.int8:
            mma_shape_m = 8

        if self.mma_type == MmaType.MMA and self.warp_shape[0] % 16 == 8:
            mma_shape_m = 8

        if self.mma_type == MmaType.MMA and mma_shape_m == 8:
            mma_shape_k = mma_shape_k // 2

        if self.mma_type == MmaType.WGMMA:
            assert self.warp_shape[0] % mma_shape_m == 0
            assert self.warp_shape[1] % (mma_shape_n // 4) == 0
        else:
            assert self.warp_shape[0] % mma_shape_m == 0
            assert self.warp_shape[1] % mma_shape_n == 0
        assert self.warp_shape[2] % mma_shape_k == 0

        mma_native_mixed = (
            self.mma_type == MmaType.MMA
            and not self.use_fused_e8m0_scale
            and self.sm_version // 10 == 12
            and mma_shape_m == 16
            and self.a_dtype in (dtypes.float8e4m3, dtypes.float8e5m2, dtypes.float8e3m4)
            and self.b_dtype in (dtypes.float4e2m1, dtypes.float6e3m2, dtypes.float6e2m3)
        )
        self.mma_b_dtype = self.b_dtype if mma_native_mixed else self.a_dtype

        return MmaOpClass.from_config(
            self.mma_type,
            mma_shape_m,
            mma_shape_n,
            mma_shape_k,
            self.mma_a_dtype,
            self.mma_b_dtype,
            mma_cd_dtype,
        )

    def check_shape(self):
        assert self.problem_shape[1] % self.block_shape[1] == 0
        assert self.problem_shape[2] % self.block_shape[2] == 0
        assert self.block_shape[0] % self.warp_shape[0] == 0
        assert self.block_shape[1] % self.warp_shape[1] == 0
        assert self.block_shape[2] % self.warp_shape[2] == 0

        assert self.warp_shape[1] % 16 == 0
        assert jit_utils.is_power_of_two(self.block_shape[1])
        assert jit_utils.is_power_of_two(self.block_shape[2])
        assert jit_utils.is_power_of_two(self.warp_shape[1])
        assert jit_utils.is_power_of_two(self.warp_shape[2])
        assert jit_utils.is_power_of_two(self.block_shape[0] // self.warp_shape[0])
        assert jit_utils.is_power_of_two(self.block_shape[1] // self.warp_shape[1])
        assert jit_utils.is_power_of_two(self.block_shape[2] // self.warp_shape[2])
        assert self.problem_shape[1] > self.pad_shape[1]
        assert self.problem_shape[2] > self.pad_shape[2]
        assert self.pad_shape[1] % 8 == 0
        assert self.pad_shape[2] % (128 // self.a_dtype.num_bits) == 0

        assert self.warp_shape[1] <= 64
        if self.a_dtype.num_bits == 16:
            assert self.warp_shape[1] >= 32
            assert self.warp_shape[2] >= 32
        elif self.a_dtype.num_bits == 8:
            assert self.warp_shape[1] >= 16
            assert self.warp_shape[2] >= 64
        elif self.a_dtype.num_bits == 4:
            assert self.warp_shape[1] >= 16
            assert self.warp_shape[2] >= 128

    def check_scale(self):
        if self.mma_type == MmaType.MXMMA:
            mma_k = 256 // self.a_dtype.num_bits
            for gs in (self.input_scale_group_size, self.weight_scale_group_size):
                if gs > 0:
                    assert mma_k % gs == 0
                    assert mma_k // gs in (1, 2, 4)
            if self.input_scale_group_size > 0 and self.weight_scale_group_size > 0:
                assert self.input_scale_group_size == self.weight_scale_group_size
            return

        if self.input_scale_group_size > 0:
            assert self.input_scale_group_size >= 256 // self.a_dtype.num_bits
        if self.weight_scale_group_size > 0:
            assert self.weight_scale_group_size >= 256 // self.a_dtype.num_bits
        if self.weight_scale_group_size_n > 1:
            assert self.weight_scale_group_size_n >= 64

        if self.is_block_weight_scale:
            if self.input_scale_group_size > 0:
                assert self.input_scale_group_size == self.weight_scale_group_size
            assert self.weight_scale_group_size_n > 0
            assert not self.has_zero_point
        if self.is_tensor_weight_scale:
            self.bs_dtype = self.c_dtype
        if self.is_channel_weight_scale_2:
            assert self.is_group_weight_scale

    def check_dtype(self):
        dtype_map = {
            dtypes.int4: 80,
            dtypes.int8: 75,
            dtypes.float4e0m3: 120,
            dtypes.float4e2m1: 120,
            dtypes.float8e3m4: 120,
            dtypes.float8e4m3: 89,
            dtypes.float8e5m2: 89,
            dtypes.bfloat16: 80,
            dtypes.float16: 75,
        }
        assert self.a_dtype in dtype_map
        assert self.sm_version >= dtype_map[self.a_dtype]
        assert self.b_dtype.num_bits <= 8
        assert self.b_dtype.num_bits <= self.a_dtype.num_bits
        if self.b_dtype.is_integer_type and self.a_dtype.is_integer_type:
            if self.a_dtype.num_bits == self.b_dtype.num_bits:
                assert self.a_dtype == self.b_dtype
            else:
                assert not self.b_dtype.is_signed
        elif self.b_dtype.is_integer_type and self.a_dtype.is_floating_point_type:
            assert not self.b_dtype.is_signed
            if self.has_zero_point:
                assert self.b_dtype.num_bits <= self.a_dtype.mantissa_bits + 1
            else:
                assert self.b_dtype.num_bits <= self.a_dtype.mantissa_bits + 2
        elif self.b_dtype.is_floating_point_type and self.a_dtype.is_floating_point_type:
            assert self.b_dtype.is_signed
            assert self.b_dtype.exponent_bits <= self.a_dtype.exponent_bits
            assert self.b_dtype.mantissa_bits <= self.a_dtype.mantissa_bits
            assert self.a_dtype.exponent_bits == 0 or self.b_dtype.exponent_bits >= 1
        elif self.b_dtype.is_floating_point_type and self.a_dtype.is_integer_type:
            assert self.use_fused_e8m0_scale
            raise NotImplementedError

        if self.use_f16_accum:
            if self.a_dtype == dtypes.float8e4m3:
                assert self.b_dtype.is_integer_type or self.b_dtype.exponent_bits <= 4
            else:
                assert self.a_dtype == dtypes.float16

    def check_config(self):
        if self.use_warp_spec or self.use_tma:
            assert self.use_mbarrier
        is_channel_weight_scale = self.is_channel_weight_scale
        is_group_weight_scale = self.is_group_weight_scale
        if not (is_channel_weight_scale or is_group_weight_scale):
            self.use_tma_bs = False
        if not self.is_channel_weight_scale_2:
            self.use_tma_bs2 = False
        if not self.has_zero_point:
            self.use_tma_bzp = False
        if not self.has_bias:
            self.use_tma_bias = False
        if self.gemm_type is None and self.num_experts == 0:
            self.gemm_type = GemmType.DENSE
        assert self.gemm_type is not None, "gemm_type must be specify for MoE GEMM"

        if self.has_input_scale and self.input_scale_group_size == 0:
            self.use_m_major_input_scale = True

        if self.use_tma_as and self.is_indexed_gemm:
            self.use_tma_as = False
            self.use_m_major_input_scale = True

        if self.use_tma_as:
            assert self.use_m_major_input_scale, "use_tma_as requires use_m_major_input_scale=True"

        if self.use_packed_k_layout:
            warp_k = self.warp_shape[2]
            for gs in (self.input_scale_group_size, self.weight_scale_group_size):
                assert gs == 0 or gs >= warp_k

    def __call__(self):
        msg = (
            "don't call HummingKernel object directly, "
            "please use humming.ops.launch_kernel([kernel.kernel_id], ...) instead."
        )
        raise NotImplementedError(msg)

    @classmethod
    def prepare_kernels(
        cls,
        layer_config: str | dict,
        compute_config: str | dict | None = None,
        tuning_config: str | dict | list | None = None,
    ) -> "torch.Tensor":
        def prepare_config_str(config: str | dict | list | None):
            if config is None:
                return "{}"
            elif isinstance(config, str):
                return config
            else:
                return str(config)

        def prepare_config_obj(config: str | dict | list | None):
            if config is None:
                return {}
            elif not isinstance(config, str):
                return config
            else:
                return json.loads(config)

        layer_config_str = prepare_config_str(layer_config)
        compute_config_str = prepare_config_str(compute_config)
        tuning_config_str = prepare_config_str(tuning_config)
        cache_key = (layer_config_str, compute_config_str, tuning_config_str)
        if cache_key in cls._str2kernel_cache:
            return cls._str2kernel_cache[cache_key]

        layer_config_obj = prepare_config_obj(layer_config)
        compute_config_obj = prepare_config_obj(compute_config)
        tuning_config_obj = prepare_config_obj(tuning_config)
        layer_config_obj.pop("sublayer_name", None)

        if not tuning_config_obj:
            tuning_config_obj = get_heuristics_config(LayerConfig(**layer_config_obj), **compute_config_obj)

        if isinstance(tuning_config_obj, dict):
            config = layer_config_obj | compute_config_obj | tuning_config_obj
            num_sms = config.pop("num_sms", 0)
            kernel = HummingKernel(**config)
            res = torch.tensor(
                [0, 1 << 30, kernel.kernel_id, num_sms],
                dtype=torch.int64,
                device="cpu",
            )
            cls._str2kernel_cache[cache_key] = res
            return res

        def prepare_kernel(data):
            _, _, tuning_config_obj_single = data
            kernel_config = layer_config_obj | compute_config_obj | tuning_config_obj_single
            num_sms = kernel_config.pop("num_sms", 0)
            kernel = HummingKernel(**kernel_config)
            return data, kernel, num_sms

        res = []
        if os.environ.get("HUMMING_DISABLE_PARALLEL_BUILD", "0") != "1":
            _current_device = torch.cuda.current_device()
            with ThreadPoolExecutor(
                max_workers=16,
                initializer=torch.cuda.set_device,
                initargs=(_current_device,),
            ) as executor:
                for config, kernel, num_sms in executor.map(prepare_kernel, tuning_config_obj):
                    kernel.load_cubin()
                    res += [config[0], config[1], kernel.kernel_id, num_sms]
        else:
            for config in tuning_config_obj:
                _, kernel, num_sms = prepare_kernel(config)
                kernel.load_cubin()
                res += [config[0], config[1], kernel.kernel_id, num_sms]

        res = torch.tensor(res, dtype=torch.int64, device="cpu")
        cls._str2kernel_cache[cache_key] = res
        return res
