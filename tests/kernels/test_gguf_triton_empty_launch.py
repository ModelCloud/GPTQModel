# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Direct GGUF Triton launchers return empty outputs without autotuning."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import gguf_triton


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not gguf_triton.triton_available(),
    reason="CUDA or ROCm GPU with Triton required",
)


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_direct_fused_q4_k_skips_zero_row_autotuning(output_dtype):
    x = torch.empty((0, 256), device="cuda", dtype=torch.float16)
    qs = torch.empty((1, 128, 64), device="cuda", dtype=torch.uint8)
    scale = torch.empty((1, 8, 64), device="cuda", dtype=torch.float16)
    minimum = torch.empty_like(scale)

    output = gguf_triton.fused_q4_k_matmul(
        x, qs, scale, minimum, output_dtype=output_dtype,
    )
    assert output.shape == (0, 64)
    assert output.dtype == (output_dtype or x.dtype)
    assert output.device == x.device

    with pytest.raises(RuntimeError, match="requires CUDA/ROCm tensors"):
        gguf_triton.fused_q4_k_matmul(
            x.cpu(), qs.cpu(), scale.cpu(), minimum.cpu(), output_dtype=output_dtype,
        )


@requires_gpu
@pytest.mark.cuda
def test_direct_fixed_launch_skips_zero_rows_and_columns():
    config = {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
        "num_warps": 4, "num_stages": 2,
    }
    x = torch.empty((0, 256), device="cuda", dtype=torch.float16)
    qs = torch.empty((1, 128, 64), device="cuda", dtype=torch.uint8)
    scale = torch.empty((1, 8, 64), device="cuda", dtype=torch.float16)
    minimum = torch.empty_like(scale)

    output = gguf_triton._launch_q4_k_fixed_matmul(
        x, qs, scale, minimum, fixed_config=config, output_dtype=torch.float32,
    )
    assert output.shape == (0, 64)
    assert output.dtype == torch.float32

    no_columns = gguf_triton._launch_q4_k_fixed_matmul(
        torch.empty((1, 256), device="cuda", dtype=torch.float16),
        qs[:, :, :0], scale[:, :, :0], minimum[:, :, :0],
        fixed_config=config,
    )
    assert no_columns.shape == (1, 0)
