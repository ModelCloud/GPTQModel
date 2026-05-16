# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.utils.eora_marlin import apply_eora_marlin_fused_lora


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eora_marlin_addmm_tail_matches_dense(dtype):
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    rows, in_features, rank, out_features = 5, 96, 32, 128
    x = torch.randn(rows, in_features, device=device, dtype=dtype)
    base = torch.randn(rows, out_features, device=device, dtype=dtype)
    lora_a = torch.randn(in_features, rank, device=device, dtype=dtype) * 0.05
    lora_b = torch.randn(rank, out_features, device=device, dtype=dtype) * 0.05
    adapter = Lora(rank=rank, lora_A=lora_a, lora_B=lora_b)

    expected = base + (x @ lora_a) @ lora_b
    actual = apply_eora_marlin_fused_lora(adapter, x=x, out=base.clone())

    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eora_marlin_cuda_up_add_matches_dense(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_EORA_MARLIN_CUDA_UP_ADD", "1")
    torch.manual_seed(1)
    device = torch.device("cuda:0")
    rows, in_features, rank, out_features = 3, 64, 16, 96
    x = torch.randn(rows, in_features, device=device, dtype=dtype)
    base = torch.randn(rows, out_features, device=device, dtype=dtype)
    lora_a = torch.randn(in_features, rank, device=device, dtype=dtype) * 0.05
    lora_b = torch.randn(rank, out_features, device=device, dtype=dtype) * 0.05
    adapter = Lora(rank=rank, lora_A=lora_a, lora_B=lora_b)

    expected = base + (x @ lora_a) @ lora_b
    actual = apply_eora_marlin_fused_lora(adapter, x=x, out=base.clone())

    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_eora_marlin_fused_lora_falls_back_on_cpu():
    adapter = Lora(rank=4, lora_A=torch.randn(8, 4), lora_B=torch.randn(4, 16))
    x = torch.randn(2, 8)
    out = torch.randn(2, 16)

    assert apply_eora_marlin_fused_lora(adapter, x=x, out=out) is None
