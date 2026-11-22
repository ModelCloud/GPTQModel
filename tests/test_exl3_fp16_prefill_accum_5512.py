# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for #5512: the FP16 EXL3 prefill GEMM reduces in FP32.

``ExllamaV3Linear.forward`` sends an FP16 batch of more than 32 rows to the
reconstruct + ``hgemm`` path; 32 rows or fewer take the decode ``bc`` kernel.
``hgemm`` used ``cublasHgemm`` for its FP16 output, and that call's compute type
is FP16, so its K reduction accumulated in half precision. At K=8192 the same
rows were 2-3x less accurate than the ``bc`` kernel and the FP32-output GEMM,
which is the prefill quality regression in #5512.

The kernel now uses ``cublasGemmEx`` with ``CUBLAS_COMPUTE_32F`` and keeps the
FP16 output buffer. This module pins that the 33-row and 64-row paths stay as
accurate as the 32-row decode path on an ordinary N(0,1) activation, without
the FP32 output allocation.
"""

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from tests.accelerator_gates import requires_nvidia_cuda


requires_cuda = pytest.mark.skipif(
    not requires_nvidia_cuda(),
    reason="the native EXL3 dispatch needs an NVIDIA CUDA device",
)

# Fixed: ~5.0e-4; cublasHgemm: ~1.14e-3 at K=8192, for every seed measured.
_MAX_PREFILL_REL_ERROR = 8e-4
# The prefill error must stay close to the decode kernel's own error.
_MAX_PREFILL_OVER_BC = 1.5


@pytest.fixture(scope="module")
def layer():
    from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension

    if not prewarm_exllamav3_extension():
        pytest.skip("the EXL3 torch.ops extension is not available in this build")

    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    device = torch.device("cuda", 0)
    in_features, out_features = 8192, 256
    seed = 5512
    torch.manual_seed(seed)
    weight = torch.randn((in_features, out_features), dtype=torch.float32, device=device) * 0.02
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
            "seed": seed,
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
def test_fp16_prefill_accumulates_in_fp32(layer):
    device = torch.device("cuda", 0)
    torch.manual_seed(5512)
    x = torch.randn(64, 8192, device=device).half()

    with torch.inference_mode():
        oracle = layer._ensure_torch_inner().forward(x.float()).double()[:32]

        def rel_error(y):
            return ((y.double()[:32] - oracle).norm() / oracle.norm()).item()

        assert layer(x[:33].contiguous()).dtype == torch.float16
        bc = rel_error(layer(x[:32].contiguous()))
        prefill = rel_error(layer(x[:33].contiguous()))
        wide = rel_error(layer(x))

    assert prefill <= _MAX_PREFILL_REL_ERROR
    assert wide <= _MAX_PREFILL_REL_ERROR
    assert prefill <= _MAX_PREFILL_OVER_BC * bc
