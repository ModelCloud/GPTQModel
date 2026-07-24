# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import ctypes
import dataclasses
import math
from typing import ClassVar

import cuda.bindings.driver as cbd
import jinja2
import torch

from ..jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/process.cuh>

""")


@dataclasses.dataclass(kw_only=True)
class RepackWeightKernel(KernelRuntime):
    name: ClassVar[str] = "weight_repack_nk"
    weight_bits: int
    activation_bits: int
    is_weight_packed: bool
    should_preprocess_for_int2fp: bool = False
    should_preprocess_with_zp: bool = False
    use_wgmma: bool = False
    use_fused_e8m0_scale: bool = False
    group_size_zp: int = 0
    use_packed_k_layout: bool = False

    def init_kernel(self):
        if self.should_preprocess_with_zp:
            assert self.should_preprocess_for_int2fp

        should_transpose_mini_block = self.use_wgmma and not self.use_fused_e8m0_scale

        self.code = CODE_TEMPLATE.render(
            weight_bits=self.weight_bits,
            activation_bits=self.activation_bits,
            is_weight_packed=int(self.is_weight_packed),
            should_preprocess_for_int2fp=int(self.should_preprocess_for_int2fp),
            should_preprocess_with_zp=int(self.should_preprocess_with_zp),
            use_wgmma=int(should_transpose_mini_block),
            group_size_zp=self.group_size_zp,
        )
        self.kernel_expr = (
            f"weight_repack_nk<\n"
            f"    {self.weight_bits},\n"
            f"    {self.activation_bits},\n"
            f"    {int(self.is_weight_packed)},\n"
            f"    {int(self.should_preprocess_for_int2fp)},\n"
            f"    {int(self.should_preprocess_with_zp)},\n"
            f"    {int(should_transpose_mini_block)},\n"
            f"    {self.group_size_zp},\n"
            f"    {int(self.use_packed_k_layout)}>"
        )
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        zero_point: torch.Tensor | None,
        padded_shape_n: int | None = None,
        padded_shape_k: int | None = None,
        interleave_mode: int = 3,
    ):
        self.check_context()
        num_experts = 1 if inputs.ndim == 2 else inputs.size(0)
        shape_n = inputs.size(-2)
        shape_k = inputs.size(-1)
        if self.is_weight_packed:
            assert shape_k * 32 % self.weight_bits == 0
            shape_k = shape_k * 32 // self.weight_bits

        device = inputs.device

        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(shape_n / 64)
        config.gridDimY = math.ceil(shape_k / 64)
        config.gridDimZ = num_experts
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            0 if zero_point is None else zero_point.data_ptr(),
            shape_n,
            shape_k,
            padded_shape_n or shape_n,
            padded_shape_k or shape_k,
            interleave_mode,
        )

        cbd.cuLaunchKernelEx(config, self.func, (arg_values, self.arg_types), 0)
