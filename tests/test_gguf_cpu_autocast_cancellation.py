# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


def _q8_0_cancellation_case(dtype: torch.dtype):
    module = GGUFTorchLinear(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=1,
        bias=True,
    ).eval()
    if dtype == torch.float16:
        scale, code, activation = 0.125, 25, 1000.0
    else:
        scale, code, activation = 1.0, 1, 3120.0
    module.qweight[:, :2] = torch.tensor([scale], dtype=torch.float16).view(torch.uint8)
    module.qweight[:, 2:] = code
    module.bias.fill_(-100000.0)
    x = torch.full((1, 32), activation, dtype=dtype)

    # Decode the specified Q8_0 scale/code independently of either GGUF
    # dequantizer, then calculate the full dot and bias in FP64 and FP32.
    weight64 = torch.full((32, 1), code * scale, dtype=torch.float64)
    bias64 = torch.tensor([-100000.0], dtype=torch.float64)
    oracle64 = x.double() @ weight64 + bias64
    oracle32 = x.float() @ weight64.float() + bias64.float()
    torch.testing.assert_close(oracle32, oracle64.float(), rtol=0, atol=0)
    return module, x, oracle64.to(dtype)


@pytest.mark.parametrize("path", ["dense", "fused"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_public_gguf_cpu_bias_cancellation_ignores_outer_autocast(path, dtype, autocast_enabled):
    module, x, expected = _q8_0_cancellation_case(dtype)
    module.autotune_enabled = False
    module.gguf_fused_cpu_max_rows = 1 if path == "fused" else 0
    module.gguf_fused_cpu_min_matrix_elements = 0
    assert module._is_fused_k_forward_candidate(x) == (path == "fused")
    original_buffers = {name: tensor.clone() for name, tensor in module.named_buffers()}
    original_state_keys = set(module.state_dict())

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        for _ in range(2):
            actual = module(x)
            assert actual.dtype == dtype
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert set(module.state_dict()) == original_state_keys
    for name, original in original_buffers.items():
        torch.testing.assert_close(dict(module.named_buffers())[name], original, rtol=0, atol=0)
    assert not any(isinstance(value, torch.Tensor) for value in vars(module).values())
