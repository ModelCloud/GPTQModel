# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""ctypes/native GEMM backends must receive a densely packed activation.

Batched decoding feeds ``hidden_states[:, -1:, :]``. ``reshape(-1, K)`` returns
a strided view for that input, and BitBLAS/Machete pass the tensor's
``data_ptr()`` to a kernel that assumes packed rows, so the operand must be
made contiguous before the launch.
"""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear import bitblas as bitblas_module
from gptqmodel.nn_modules.qlinear import machete as machete_module


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _strided_activation(in_features: int = 16, rows: int = 4) -> torch.Tensor:
    base = torch.randn(rows, 7, in_features, device="cuda", dtype=torch.float16)
    x = base[:, -1:, :]
    assert not x.is_contiguous()
    return x


@requires_cuda
@pytest.mark.cuda
def test_bitblas_forward_packs_a_strided_activation():
    module = bitblas_module.BitblasBaseQuantLinear.__new__(
        bitblas_module.BitblasBaseQuantLinear
    )
    torch.nn.Module.__init__(module)
    module._compute_dtype = torch.float16
    module.SUPPORTS_DTYPES = [torch.float16]
    module.out_features = 8
    module.bias = None
    module.adapter = None
    module.quant_config = SimpleNamespace(with_zeros=False)
    module.qweight = torch.zeros((1,), dtype=torch.int32, device="cuda")
    module.scales = torch.zeros((1,), dtype=torch.float16, device="cuda")

    seen = {}

    def _fake_matmul(x_2d, qweight, scales):
        seen["contiguous"] = x_2d.is_contiguous()
        return torch.zeros((x_2d.shape[0], module.out_features), device="cuda", dtype=torch.float16)

    module.bitblas_matmul = _fake_matmul
    out = module(_strided_activation())

    assert seen["contiguous"] is True
    assert out.shape == (4, 1, 8)


@requires_cuda
@pytest.mark.cuda
def test_machete_forward_packs_a_strided_activation(monkeypatch):
    module = machete_module.MacheteLinear.__new__(machete_module.MacheteLinear)
    torch.nn.Module.__init__(module)
    module.out_features = 8
    module.input_perm = torch.zeros((0,), dtype=torch.int32, device="cuda")
    module.scales = torch.zeros((1,), dtype=torch.float16, device="cuda")
    module.qzeros = torch.zeros((0,), dtype=torch.int32, device="cuda")
    module.has_zero_points = False
    module.qweight = torch.zeros((1,), dtype=torch.int32, device="cuda")
    module.weight_type = 0
    module.group_size = 128
    module.bias = None
    module.adapter = None

    seen = {}

    def _fake_mm(*, a, **kwargs):
        seen["contiguous"] = a.is_contiguous()
        return torch.zeros((a.shape[0], module.out_features), device="cuda", dtype=torch.float16)

    monkeypatch.setattr(machete_module, "machete_mm", _fake_mm)
    out = module(_strided_activation())

    assert seen["contiguous"] is True
    assert out.shape == (4, 1, 8)
