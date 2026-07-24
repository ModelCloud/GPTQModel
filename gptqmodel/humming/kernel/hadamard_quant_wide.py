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
from .hadamard import _TORCH_TO_CPP_TYPE
from .hadamard_quant import _cvt_patch_mode

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/hadamard_quant_wide.cuh>
""")

_SCALE_DTYPE_CODE = {"float32": 0, "float8e4m3": 1, "float8e8m0": 2}
_SCALE_DTYPE_TORCH = {
    "float32": torch.float32,
    "float8e4m3": torch.float8_e4m3fn,
    "float8e8m0": torch.uint8,
}


def _pick_wide_launch_params(block_size: int, group_size: int) -> tuple[int, int]:
    assert block_size <= 512, "wide kernel requires block_size (FHT N) <= 512"
    assert group_size > block_size
    tiles_per_group = group_size // block_size
    target_elems_per_thread = 16

    best: tuple[int, int, int] | None = None
    for elems_per_lane in (16, 8, 4, 2):
        if block_size % elems_per_lane != 0:
            continue
        threads_per_tile = block_size // elems_per_lane
        if threads_per_tile > 32:
            continue
        for tiles_per_thread in range(1, tiles_per_group + 1):
            if tiles_per_group % tiles_per_thread != 0:
                continue
            tpb = (tiles_per_group // tiles_per_thread) * threads_per_tile
            if tpb % 32 != 0 or not (32 <= tpb <= 1024):
                continue
            elems_per_thread = tiles_per_thread * elems_per_lane
            score = (abs(elems_per_thread - target_elems_per_thread), tiles_per_thread)
            if best is None or score < best[0]:
                best = (score, threads_per_tile, tiles_per_thread)
        if best is not None:
            return best[1], best[2]

    raise AssertionError(
        f"no valid wide-kernel launch params for block_size={block_size}, group_size={group_size}"
    )


@dataclasses.dataclass(kw_only=True)
class HadamardQuantInputWideKernel(KernelRuntime):
    name: ClassVar[str] = "hadamard_quant_input_wide"
    source_torch_dtype: torch.dtype
    target_dtype: dtypes.DataType
    block_size: int
    group_size: int
    has_extra_scale: bool = False
    m_major: bool = False
    scale_dtype: str = "float32"
    has_global_scale: bool = False

    def init_kernel(self):
        assert self.group_size > self.block_size, (
            "wide kernel only for group_size > block_size; use HadamardQuantInputKernel otherwise"
        )
        threads_per_tile, tiles_per_thread = _pick_wide_launch_params(self.block_size, self.group_size)
        self.threads_per_tile = threads_per_tile
        self.tiles_per_thread = tiles_per_thread
        tiles_per_group = self.group_size // self.block_size
        self.threads_per_block = (tiles_per_group // tiles_per_thread) * threads_per_tile

        cpp_source = _TORCH_TO_CPP_TYPE[self.source_torch_dtype]
        cpp_target = self.target_dtype.to_cpp_str()
        self.code = CODE_TEMPLATE.render()
        self.kernel_expr = (
            f"hadamard_quant_input_wide<\n"
            f"    {cpp_source},\n"
            f"    {cpp_target},\n"
            f"    {self.block_size},\n"
            f"    {self.group_size},\n"
            f"    {threads_per_tile},\n"
            f"    {tiles_per_thread},\n"
            f"    {int(self.has_extra_scale)},\n"
            f"    {int(self.m_major)},\n"
            f"    {_SCALE_DTYPE_CODE[self.scale_dtype]},\n"
            f"    {int(self.has_global_scale)}>"
        )
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        self.prepare()

    def postprocess_cubin(self, cubin_path: str):
        mode = _cvt_patch_mode(self.target_dtype)
        if mode:
            from ..utils.cubin import patch_cubin

            patch_cubin(cubin_path=cubin_path, mode=mode)

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        scales: torch.Tensor,
        extra_scale: float = 1.0,
        global_scale: torch.Tensor | None = None,
    ):
        self.check_context()
        assert inputs.is_contiguous() and outputs.is_contiguous() and scales.is_contiguous()
        assert inputs.size(-1) % self.group_size == 0
        assert inputs.dtype == self.source_torch_dtype
        assert scales.dtype == _SCALE_DTYPE_TORCH[self.scale_dtype]
        assert (global_scale is not None) == self.has_global_scale

        num_groups = inputs.numel() // self.group_size
        shape_m = inputs.numel() // inputs.size(-1)
        if self.m_major:
            shape_m = (shape_m + 3) // 4 * 4
        groups_per_row = inputs.size(-1) // self.group_size
        device = inputs.device

        config = cbd.CUlaunchConfig()
        config.gridDimX = num_groups
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = self.threads_per_block
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            scales.data_ptr(),
            float(extra_scale),
            num_groups,
            shape_m,
            groups_per_row,
            global_scale.data_ptr() if global_scale is not None else 0,
        )

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
