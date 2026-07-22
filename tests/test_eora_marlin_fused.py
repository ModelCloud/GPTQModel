# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel import extension
from gptqmodel.adapter.adapter import Lora
from gptqmodel.utils.eora_marlin import apply_eora_marlin_fused_lora


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eora_marlin_addmm_tail_matches_dense(dtype):
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    rows, in_features, rank, out_features = 5, 96, 32, 128
    x = torch.randn(1, rows, in_features, device=device, dtype=dtype)
    base = torch.randn(1, rows, out_features, device=device, dtype=dtype)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("rank", [37, 64, 128, 320])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("in_features,out_features", [(97, 130), (257, 97)])
def test_eora_marlin_cooperative_fused_tail_matches_fp32_reference(
    dtype, rows, rank, prepared, in_features, out_features, monkeypatch
):
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) != (8, 0):
        pytest.skip("The cooperative EoRA kernel is currently enabled only for sm_80")

    monkeypatch.setenv("GPTQMODEL_EORA_MARLIN_COOPERATIVE", "1")
    torch.manual_seed(2)
    x = torch.randn(rows, in_features, device=device, dtype=dtype)
    base = torch.randn(rows, out_features, device=device, dtype=dtype)
    lora_a = torch.randn(in_features, rank, device=device, dtype=dtype) * 0.025
    lora_b = torch.randn(rank, out_features, device=device, dtype=dtype) * 0.025
    adapter = Lora(rank=rank, lora_A=lora_a, lora_B=lora_b)
    workspace = torch.empty((rows, rank), device=device, dtype=torch.float32)

    expected = base.float() + (x.float() @ lora_a.float()) @ lora_b.float()
    if prepared:
        op = extension.op("eora_marlin", "lora_fused_add_prepared")
        actual = op(x, lora_a, lora_b, base.clone(), workspace)
    else:
        actual = apply_eora_marlin_fused_lora(
            adapter,
            x=x,
            out=base.clone(),
            cooperative_buffer=workspace,
        )

    assert actual is not None
    assert actual.shape == base.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
@pytest.mark.parametrize("device_index", [0, 1])
def test_eora_marlin_cooperative_uses_current_device_and_stream(device_index, monkeypatch):
    device = torch.device(f"cuda:{device_index}")
    if torch.cuda.get_device_capability(device) != (8, 0):
        pytest.skip("The cooperative EoRA kernel is currently enabled only for sm_80")

    monkeypatch.setenv("GPTQMODEL_EORA_MARLIN_COOPERATIVE", "1")
    generator = torch.Generator(device=device)
    generator.manual_seed(10 + device_index)
    rows, in_features, rank, out_features = 2, 95, 48, 129
    x = torch.randn(rows, in_features, device=device, dtype=torch.float16, generator=generator)
    base = torch.randn(rows, out_features, device=device, dtype=torch.float16, generator=generator)
    lora_a = torch.randn(in_features, rank, device=device, dtype=torch.float16, generator=generator) * 0.02
    lora_b = torch.randn(rank, out_features, device=device, dtype=torch.float16, generator=generator) * 0.02
    adapter = Lora(rank=rank, lora_A=lora_a, lora_B=lora_b)
    workspace = torch.empty((rows, rank), device=device, dtype=torch.float32)
    stream = torch.cuda.Stream(device=device)

    expected = base.float() + (x.float() @ lora_a.float()) @ lora_b.float()
    with torch.cuda.stream(stream):
        actual = apply_eora_marlin_fused_lora(
            adapter,
            x=x,
            out=base.clone(),
            cooperative_buffer=workspace,
        )
    stream.synchronize()

    assert actual is not None
    assert actual.device == device
    torch.testing.assert_close(actual.float(), expected, rtol=5e-2, atol=5e-2)


def test_eora_marlin_fused_lora_falls_back_on_cpu():
    adapter = Lora(rank=4, lora_A=torch.randn(8, 4), lora_B=torch.randn(4, 16))
    x = torch.randn(2, 8)
    out = torch.randn(2, 16)

    assert apply_eora_marlin_fused_lora(adapter, x=x, out=out) is None
