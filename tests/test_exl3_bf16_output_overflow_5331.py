# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for #5331: BF16 EXL3 must not store the GEMM in FP16.

The native EXL3 kernels are FP16-only.  ``_native_bf16_input`` bounds the
activation row so the input Hadamard rotation cannot overflow, but that bound
does not bound the dot product: a row with ``amax`` well below
``FP16_SAFE_AMAX`` can still produce an output above 65504.  The pre-fix
forward requested an FP16 result, so such a row became ``inf``/``NaN`` with no
error.  The BF16 path now asks the native GEMM for FP32 output, which both the
``hgemm`` and small-batch kernels support, and keeps the value until the BF16
re-scale.
"""

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from tests.accelerator_gates import requires_nvidia_cuda


requires_cuda = pytest.mark.skipif(
    not requires_nvidia_cuda(),
    reason="the native EXL3 dispatch needs an NVIDIA CUDA device",
)


def _quantized_layer(device: torch.device) -> ExllamaV3Linear:
    """A real W4 layer whose output can exceed FP16's range from a small row."""
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    in_features, out_features = 4096, 128
    torch.manual_seed(5331)
    weight = torch.full(
        (in_features, out_features), 0.5, dtype=torch.float32, device=device
    )
    activations = torch.randn((64, in_features), dtype=torch.float32, device=device)
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    _, _, tensors = quantize_exl3(
        weight,
        {"H": hessian, "count": activations.shape[0], "finalized": False},
        {
            "K": 4,
            "devices": [device],
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": 5331,
            "mcg": True,
        },
        return_weight_q=False,
    )
    module = ExllamaV3Linear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name="layer.down_proj",
        tensors=dict(tensors),
    ).to(device)
    module.post_init()
    return module


@requires_cuda
def test_bf16_native_path_requests_fp32_output():
    """The BF16 dispatch must not allocate the GEMM result in FP16."""
    module = object.__new__(ExllamaV3Linear)
    module.out_features = 128
    module.bias = None
    calls = []

    class Native:
        def forward(self, x, params, out_dtype):
            calls.append((x.dtype, out_dtype))
            return torch.zeros(
                (*x.shape[:-1], 128), dtype=out_dtype, device=x.device
            )

    module._ensure_inner = lambda: Native()
    module._ensure_torch_inner = lambda: pytest.fail(
        "BF16 must keep the native path"
    )
    x = torch.ones((2, 128), dtype=torch.bfloat16, device="cuda")

    with torch.inference_mode():
        output = module.forward(x)

    assert calls == [(torch.float16, torch.float32)]
    assert output.dtype == torch.bfloat16


@requires_cuda
@pytest.mark.parametrize("rows", [2, 33])
def test_bf16_output_above_fp16_range_stays_finite(rows):
    """A row under ``FP16_SAFE_AMAX`` whose dot product exceeds 65504.

    ``rows=2`` takes the small-batch BC kernel; ``rows=33`` takes the
    reconstruct + ``hgemm`` path. Both must keep the result out of FP16.
    """
    device = torch.device("cuda:0")
    module = _quantized_layer(device)

    # Row 0: amax 64, far inside the input bound; each output is ~130000.
    # Row 1: an ordinary row that must stay accurate alongside it.
    x = torch.zeros(rows, 4096, dtype=torch.bfloat16, device=device)
    x[0] = 64.0
    x[1:] = 1.0

    prepared = module._native_bf16_input(x)
    assert prepared is not None
    _, output_scale = prepared
    # The input pre-scale must be the identity: this is an output overflow,
    # not the wide-input case that #4870 already covers.
    assert torch.equal(output_scale, torch.ones_like(output_scale))

    with torch.inference_mode():
        native = module(x)
        oracle = module._ensure_torch_inner()(x)

    assert torch.isfinite(native).all()
    assert float(native[0].abs().max()) > torch.finfo(torch.float16).max
    torch.testing.assert_close(
        native[0].float(), oracle[0].float(), rtol=2e-2, atol=0
    )
