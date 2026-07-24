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
#include <humming/kernel/process_mxfp4.cuh>
"""


@dataclasses.dataclass(kw_only=True)
class ProcessMxfp4W4A8Kernel(KernelRuntime):
    disable_fast_math: ClassVar[bool] = True
    name: ClassVar[str] = "process_mxfp4_w4a8"

    def init_kernel(self):
        self.code = CODE_TEMPLATE
        self.kernel_expr = "process_mxfp4_w4a8"
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        delta_scale_offsets: torch.Tensor,
    ):
        assert inputs.dtype == torch.int32
        assert outputs.dtype == torch.int32
        assert delta_scale_offsets.dtype == torch.uint8
        assert inputs.nelement() == outputs.nelement()
        assert inputs.nelement() == delta_scale_offsets.nelement() * 4

        self.check_context()
        device = inputs.device
        num_groups = delta_scale_offsets.nelement()
        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(num_groups / 128)
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 128
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            delta_scale_offsets.data_ptr(),
            num_groups,
        )

        cbd.cuLaunchKernelEx(config, self.func, (arg_values, self.arg_types), 0)
        return outputs
