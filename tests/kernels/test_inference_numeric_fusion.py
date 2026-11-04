# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from .test_inference_numeric_range import check


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("columns", [129, 1024, 4096, 32768])
@pytest.mark.parametrize("power", [0, 8, 120, -20, -130])
def test_qqq_fused_quantization_thresholds(dtype, columns, power):
    import gptqmodel.nn_modules.qlinear.qqq as qqq

    if dtype == torch.float16 and power in (-130, 120):
        pytest.skip("FP16 cannot represent this input range")
    # Exact half-steps and one input ULP either side, including both signs.
    midpoints = torch.arange(-126.5, 127., dtype=torch.float64).to(dtype)
    neighbors = torch.stack((
        torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf"))),
        midpoints,
        torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
    ), -1).flatten()
    values = neighbors.repeat((columns + neighbors.numel() - 1) // neighbors.numel())[:columns]
    values[:2] = torch.tensor([-127., 127.], dtype=dtype)
    source = torch.stack((values, -values, torch.zeros_like(values)))
    source = (source.double() * 2.**power).to(dtype)
    # Independent arithmetic with explicit FP32 scale and quotient rounding.
    scale = (source.double().abs().amax(-1, keepdim=True) / 127).float()
    scale[scale == 0] = 1
    expected = (source.double() / scale.double()).float().double().round().clamp(-128, 127).to(torch.int8)
    x = source.cuda()
    before = x.clone()
    codes, actual_scale = qqq.QQQLinear.dynamic_quant(None, x)
    torch.testing.assert_close(codes.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_scale.cpu(), scale, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qqq_fused_random_and_fallback(dtype, monkeypatch):
    import gptqmodel.nn_modules.qlinear.qqq as qqq

    x = torch.randn((7, 4096), generator=torch.Generator().manual_seed(67), dtype=dtype).cuda()
    fused_q, fused_s = qqq.QQQLinear.dynamic_quant(None, x)
    monkeypatch.setattr(qqq, "_qqq_gpu_quantizer", lambda: None)
    eager_q, eager_s = qqq.QQQLinear.dynamic_quant(None, x)
    torch.testing.assert_close(fused_q, eager_q, rtol=0, atol=0)
    torch.testing.assert_close(fused_s, eager_s, rtol=1e-6, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_qqq_quantization_layout_and_empty(monkeypatch):
    import gptqmodel.nn_modules.qlinear.qqq as qqq

    codes, scales = qqq.QQQLinear.dynamic_quant(None, torch.empty((0, 128), device="cuda"))
    assert codes.shape == (0, 128) and scales.shape == (0, 1)

    def unexpected_fusion():
        raise AssertionError("strided and oversized rows must use the Torch path")

    monkeypatch.setattr(qqq, "_qqq_gpu_quantizer", unexpected_fusion)
    for x in (torch.ones((128, 3), device="cuda").t(), torch.ones((1, 32769), device="cuda")):
        codes, scales = qqq.QQQLinear.dynamic_quant(None, x)
        torch.testing.assert_close(codes, torch.full_like(codes, 127), rtol=0, atol=0)
        torch.testing.assert_close(scales, torch.full_like(scales, 1 / 127), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits", ["q4_0", "q8_0", "q4_k", "q5_k", "q6_k"])
def test_gguf_fused_chunk_accumulation(dtype, bits):
    from .test_gguf_torch_cpu import _build_module, _oracle_dequantize

    _, module = _build_module(bits, in_features=513, out_features=65, seed=73)
    weight = torch.from_numpy(_oracle_dequantize(module.qweight, bits.upper())[:, :513]).double()
    source = torch.randn((7, 513), generator=torch.Generator().manual_seed(91), dtype=dtype)
    expected = source.double() @ weight.t() + module.bias.double()
    module = module.cuda()
    module.gguf_fused_chunk_blocks = 1
    for _ in range(2):
        actual = module._forward_fused_k(source.cuda(), bias=module.bias)
        assert actual.dtype == dtype and actual.isfinite().all()
        torch.testing.assert_close(actual.cpu(), expected.to(dtype), rtol=2e-3, atol=2e-3)
    assert set(dict(module.named_buffers())) == {"qweight", "bias"}


@pytest.mark.skipif(not torch.cuda.is_available() or bool(torch.version.hip), reason="Native AWQ requires CUDA")
@pytest.mark.parametrize("rows", [32, 33, 64, 65, 128, 129, 192, 193, 256])
@pytest.mark.parametrize("columns", [256, 512])
def test_awq_prefill_fused_store(rows, columns):
    from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear

    k = 2048
    generator = torch.Generator().manual_seed(41)
    codes = torch.randint(0, 16, (columns, k), generator=generator)
    # Powers of two make source weights and their packed reconstruction exact.
    weights = (codes - 8).double() / 128
    linear = torch.nn.Linear(k, columns, bias=False, dtype=torch.float16)
    linear.weight.data.copy_(weights)
    module = AwqGEMVFastLinear(bits=4, group_size=128, sym=False, desc_act=False,
                             in_features=k, out_features=columns, bias=False,
                             register_buffers=True, pack_dtype=torch.int16)
    module.pack(linear, torch.full((columns, k // 128), 1 / 128, dtype=torch.float16),
                torch.full((columns, k // 128), 8, dtype=torch.float16))
    module = module.cuda().eval()
    module.post_init()
    source = torch.randn((rows, k), generator=generator, dtype=torch.float16)
    expected = source.double() @ weights.t()
    check(module, source.cuda(), expected.cuda())
