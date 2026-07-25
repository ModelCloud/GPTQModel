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
import torch

from ..jit.runtime import KernelRuntime

CODE_TEMPLATE = """
#include <humming/kernel/dequant_weight.cuh>

"""


@dataclasses.dataclass(kw_only=True)
class DequantKernel(KernelRuntime):
    disable_fast_math: ClassVar[bool] = True
    name: ClassVar[str] = "dequant_unpacked_fp_type"

    def init_kernel(self):
        self.code = CODE_TEMPLATE
        self.kernel_expr = "dequant_unpacked_fp_type"
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_bool,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        exponent_bits: int,
        mantissa_bits: int,
        is_signed: bool,
    ):
        self.check_context()
        device = inputs.device
        total_size = inputs.nelement()
        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(total_size / (32 * 32))
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            total_size,
            exponent_bits,
            mantissa_bits,
            is_signed,
        )

        cbd.cuLaunchKernelEx(config, self.func, (arg_values, self.arg_types), 0)
        return outputs
