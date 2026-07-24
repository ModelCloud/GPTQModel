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
#include <humming/kernel/pack_weight.cuh>

""")


@dataclasses.dataclass(kw_only=True)
class PackWeightKernel(KernelRuntime):
    name: ClassVar[str] = "pack_weight"
    num_bits: int

    def init_kernel(self):
        self.code = CODE_TEMPLATE.render(num_bits=self.num_bits)
        self.kernel_expr = f"pack_weight_kernel<{self.num_bits}>"
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
        self.prepare()

    def __call__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        self.check_context()
        device = inputs.device
        config = cbd.CUlaunchConfig()
        num_elements = inputs.nelement()
        config.gridDimX = math.ceil(num_elements / (32 * 32))
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (inputs.data_ptr(), outputs.data_ptr(), num_elements)

        cbd.cuLaunchKernelEx(config, self.func, (arg_values, self.arg_types), 0)
