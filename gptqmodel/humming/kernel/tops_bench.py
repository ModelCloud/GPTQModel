# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import jinja2
import torch

from .. import dtypes
from ..config import MmaOpClass, MmaType
from ..jit.runtime import KernelRuntime
from ..utils.device import get_device_num_sms

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/tops_bench.cuh>

class MmaOpClass {
public:
{{mma_op_class}}
};

""")


@dataclasses.dataclass(kw_only=True)
class TopsBenchKernel(KernelRuntime):
    name: ClassVar[str] = "tops_bench"
    mma_type: str | MmaType
    mma_shape_m: int
    mma_shape_n: int
    mma_shape_k: int
    ab_dtype: str | dtypes.DataType
    cd_dtype: str | dtypes.DataType
    repeat_count: int
    unroll_count: int
    sf_dtype: str | dtypes.DataType | None = None

    def init_kernel(self):
        if isinstance(self.mma_type, str):
            self.mma_type = MmaType(self.mma_type)
        if isinstance(self.ab_dtype, str):
            self.ab_dtype = dtypes.DataType.from_str(self.ab_dtype)
        if isinstance(self.cd_dtype, str):
            self.cd_dtype = dtypes.DataType.from_str(self.cd_dtype)
        if isinstance(self.sf_dtype, str):
            self.sf_dtype = dtypes.DataType.from_str(self.sf_dtype)

        self.mma_op_class = MmaOpClass.from_config(
            self.mma_type,
            self.mma_shape_m,
            self.mma_shape_n,
            self.mma_shape_k,
            self.ab_dtype,
            self.ab_dtype,
            self.cd_dtype,
            sf_dtype=self.sf_dtype,
        )

        self.code = CODE_TEMPLATE.render(
            mma_op_class=self.mma_op_class.to_cpp_str(),
            repeat_count=self.repeat_count,
            unroll_count=self.unroll_count,
        )
        self.kernel_expr = f"tops_bench<MmaOpClass, {self.repeat_count}, {self.unroll_count}>"
        self.arg_types = (ctypes.c_void_p,)
        self.prepare()

        self.num_warps = 32
        self.ops_per_mma_per_warp = self.mma_shape_m * self.mma_shape_n * self.mma_shape_k * 2
        if self.mma_type == MmaType.WGMMA:
            self.ops_per_mma_per_warp = self.ops_per_mma_per_warp // 4
            self.num_warps = self.num_warps // 4
        self.sm_count = get_device_num_sms()
        self.num_ctas = self.sm_count * 2
        self.ops_per_call = self.ops_per_mma_per_warp * self.num_warps * self.num_ctas

    def __call__(self):
        self.check_context()
        config = cbd.CUlaunchConfig()
        config.gridDimX = self.num_ctas
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = self.num_warps * 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream().cuda_stream

        tensor = torch.empty((1,), dtype=torch.uint32, device="cuda:0")
        arg_values = (tensor.data_ptr(),)

        cbd.cuLaunchKernelEx(config, self.func, (arg_values, self.arg_types), 0)
