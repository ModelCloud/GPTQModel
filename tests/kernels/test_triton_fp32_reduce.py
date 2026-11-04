# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The Triton dequant path must round once, after an FP32 reduction."""

import pytest
import torch

from gptqmodel.nn_modules.triton_utils.dequant import quant_matmul
from tests.test_torch_weight_storage import build_case


def _require_triton_gpu():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    pytest.importorskip("triton")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 8, 16, 64])
def test_quant_matmul_matches_fp64_oracle(dtype, rows):
    """Triton output must track the FP64 oracle as closely as the Torch backend.

    ``quant_matmul`` used to compute ``input @ dequantized_weight``. On CUDA
    that accumulates in the activation dtype, so every output drifted by about
    one bf16 ulp versus the Torch backend's FP32-reduce epilogue. Measured
    against an independent FP64 oracle, the plain matmul is strictly worse.
    """
    _require_triton_gpu()

    module, _ = build_case(
        bits=4, dtype=dtype, group_size=128, in_features=1024, out_features=512
    )
    module = module.to("cuda")
    x = torch.randn((rows, 1024), device="cuda", dtype=dtype) / 8

    with torch.inference_mode():
        actual = quant_matmul(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            module.bits, module.pack_dtype_bits, module.maxq,
        )
        weight = module.dequantize_weight().double()
        expected = x.double() @ weight

    assert actual.dtype == dtype

    # Visible error: the final output is one rounding of the FP64 result.
    torch.testing.assert_close(actual, expected.to(dtype), rtol=1e-3, atol=1e-3)

    # Arithmetic error: the kernel must accumulate in FP32 and round once at
    # the epilogue, so its result has to reproduce a wide matmul followed by a
    # single dtype rounding. Compare before that rounding to isolate arithmetic
    # error from the visible low-precision output error.
    wide = torch.mm(x, module.dequantize_weight(), out_dtype=torch.float32)
    torch.testing.assert_close(
        wide.double(), expected, rtol=0, atol=2e-3
    )
    torch.testing.assert_close(actual, wide.to(dtype), rtol=0, atol=0)

    # The plain low-precision matmul is measurably worse than the FP32 path on
    # at least some rows; that drift was the original failure.
    narrow = (x @ module.dequantize_weight()).double()
    assert (wide.double() - expected).abs().max() <= (narrow - expected).abs().max()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quant_matmul_width_reduction_matches_torch_backend(dtype):
    """Triton and Torch dequant backends must agree bit-for-bit on CUDA."""
    _require_triton_gpu()

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
    from gptqmodel.quantization import FORMAT

    source, _ = build_case(
        bits=4, dtype=dtype, group_size=128, in_features=1024, out_features=512
    )
    modules = []
    for cls in (TorchLinear, TritonV2Linear):
        module = cls(bits=4, group_size=128, sym=False, desc_act=True,
                     in_features=1024, out_features=512, bias=False, format=FORMAT.GPTQ_V2)
        module.load_state_dict(source.state_dict())
        module.post_init()
        module.eval()
        module.to("cuda", dtype)
        modules.append(module)

    x = torch.randn((16, 1024), device="cuda", dtype=dtype) / 8
    with torch.inference_mode():
        torch_out = modules[0](x)
        triton_out = modules[1](x)

    torch.testing.assert_close(triton_out, torch_out, rtol=0, atol=0)
