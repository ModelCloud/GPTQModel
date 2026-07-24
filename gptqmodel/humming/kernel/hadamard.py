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

from ..jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/hadamard.cuh>
""")


_TORCH_TO_CPP_TYPE = {
    torch.float16: "__half",
    torch.bfloat16: "__nv_bfloat16",
    torch.float32: "float",
}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _pick_launch_params(block_size: int) -> tuple[int, int]:
    assert _is_pow2(block_size) and block_size >= 2
    threads_per_tile = max(1, block_size // 8)
    if threads_per_tile > 32:
        tiles_per_block = 1 if threads_per_tile >= 256 else 2
    else:
        tiles_per_block = max(1, 256 // threads_per_tile)
    return threads_per_tile, tiles_per_block


@dataclasses.dataclass(kw_only=True)
class HadamardKernel(KernelRuntime):
    name: ClassVar[str] = "hadamard_transform"
    torch_dtype: torch.dtype
    block_size: int
    has_scale: bool = False

    def init_kernel(self):
        cpp_dtype = _TORCH_TO_CPP_TYPE[self.torch_dtype]
        threads_per_tile, tiles_per_block = _pick_launch_params(self.block_size)
        self.threads_per_tile = threads_per_tile
        self.tiles_per_block = tiles_per_block
        self.threads_per_block = threads_per_tile * tiles_per_block

        self.code = CODE_TEMPLATE.render()
        self.kernel_expr = (
            f"hadamard_transform<\n"
            f"    {cpp_dtype},\n"
            f"    {self.block_size},\n"
            f"    {threads_per_tile},\n"
            f"    {tiles_per_block},\n"
            f"    {int(self.has_scale)}>"
        )
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_uint32,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        extra_scale: float = 1.0,
    ):
        self.check_context()
        assert inputs.is_contiguous() and outputs.is_contiguous()
        assert inputs.size(-1) % self.block_size == 0
        assert inputs.dtype == self.torch_dtype
        assert outputs.dtype == self.torch_dtype
        assert inputs.shape == outputs.shape

        num_tiles = inputs.numel() // self.block_size
        device = inputs.device

        config = cbd.CUlaunchConfig()
        grid_x = (num_tiles + self.tiles_per_block - 1) // self.tiles_per_block
        config.gridDimX = grid_x
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = self.threads_per_block
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            float(extra_scale),
            num_tiles,
        )

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
