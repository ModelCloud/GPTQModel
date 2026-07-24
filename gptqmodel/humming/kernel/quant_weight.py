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
from ..jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/quant_weight.cuh>

""")


@dataclasses.dataclass(kw_only=True)
class QuantWeightKernel(KernelRuntime):
    disable_fast_math: ClassVar[bool] = True
    name: ClassVar[str] = "quant_weight"
    source_dtype: dtypes.DataType
    target_dtype: dtypes.DataType
    group_size: int
    has_scale: bool
    use_e8m0_scale: bool
    has_zero_point: bool = False
    is_fp_zero_point: bool = False
    allow_negative_scale: bool = True

    def init_kernel(self):
        self.code = CODE_TEMPLATE.render(
            source_dtype=self.source_dtype.to_cpp_str(),
            target_dtype=self.target_dtype.to_cpp_str(),
            group_size=self.group_size,
            has_scale=int(self.has_scale),
            use_e8m0_scale=int(self.use_e8m0_scale),
            has_zero_point=int(self.has_zero_point),
            is_fp_zero_point=int(self.is_fp_zero_point),
            allow_negative_scale=int(self.allow_negative_scale),
        )
        self.kernel_expr = (
            f"quant_weight<\n"
            f"    {self.source_dtype.to_cpp_str()},\n"
            f"    {self.target_dtype.to_cpp_str()},\n"
            f"    {self.group_size},\n"
            f"    {int(self.has_scale)},\n"
            f"    {int(self.use_e8m0_scale)},\n"
            f"    {int(self.has_zero_point)},\n"
            f"    {int(self.is_fp_zero_point)},\n"
            f"    {int(self.allow_negative_scale)}>"
        )
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        scales: torch.Tensor | None,
        zero_point: torch.Tensor | None,
    ):
        self.check_context()
        group_size = self.group_size
        group_size = inputs.size(-1) if group_size <= 0 else group_size

        device = inputs.device
        config = cbd.CUlaunchConfig()
        config.gridDimX = inputs.nelement() // group_size
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            0 if scales is None else scales.data_ptr(),
            0 if zero_point is None else zero_point.data_ptr(),
        )

        cbd.cuLaunchKernelEx(config, self.func, (arg_values, self.arg_types), 0)
