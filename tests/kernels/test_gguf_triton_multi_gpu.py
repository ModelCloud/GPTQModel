# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUF Triton launches must bind the device that owns their operands.

A `device_map` shards a GGUF checkpoint over several GPUs. Triton launches into
the process's *current* CUDA context, so a layer on ``cuda:1`` used to fail with
``ValueError: Pointer argument cannot be accessed from Triton`` while ``cuda:0``
was current.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear, _dequantize_gguf_tensor_numpy
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="two CUDA/ROCm GPUs required",
    ),
    pytest.mark.cuda,
]


def _packed_kernel(
    bits: str, in_features: int, out_features: int = 64
) -> tuple[GGUFTorchLinear, GGUFTritonKernel]:
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(1.0)

    source = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=False,
    )
    source.pack_original(linear, scales=None, zeros=None)
    source.post_init()

    module = GGUFTritonKernel(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    module.load_state_dict(source.state_dict(), strict=True)
    return source, module


@pytest.mark.parametrize(
    ("bits", "in_features"),
    [
        ("q4_k", 256),
        ("q5_k", 256),
        ("q6_k", 256),
        ("q1_0", 1024),
    ],
)
def test_forward_on_a_non_current_gpu_matches_the_current_gpu(bits, in_features):
    source, module = _packed_kernel(bits, in_features)
    module = module.to("cuda:1").eval()
    torch.cuda.set_device(0)

    x = torch.full((2, in_features), 1.0, device="cuda:1", dtype=torch.float16)
    # NumPy unpacks the original GGUF bytes independently of the Triton
    # scales/codes; FP64 Torch then supplies the matmul oracle.
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(
            source.qweight.detach().cpu().numpy(), module.gguf_tensor_qtype
        )
    ).to(device="cuda:1", dtype=torch.float64)
    expected = x.to(torch.float16).to(torch.float64) @ dense.T

    # The operand lives on cuda:1 while cuda:0 is current. Without the device
    # guard this raises instead of launching.
    actual = module(x)

    assert torch.cuda.current_device() == 0
    torch.testing.assert_close(
        actual.to(torch.float64).cpu(), expected.cpu(), rtol=1e-2, atol=1e-2
    )
