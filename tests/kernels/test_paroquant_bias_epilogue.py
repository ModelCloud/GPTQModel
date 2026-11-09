# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import weakref

import pytest
import torch

import gptqmodel.nn_modules.qlinear.paroquant_triton as paro_module
from gptqmodel.quantization.awq.utils.packing_utils import AWQ_ORDER, pack_awq_columns
from gptqmodel.quantization.paroquant.modules.triton.gemm import (
    paroquant_dequantize_triton,
    paroquant_gemm_triton_decode,
    paroquant_gemm_triton_prefill,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA ParoQuant Triton epilogue test",
)


def _case(dtype: torch.dtype, rows: int):
    qweight = torch.full((128, 16), 0x11111111, device="cuda", dtype=torch.int32)
    qzeros = torch.zeros((1, 16), device="cuda", dtype=torch.int32)
    scales = torch.ones((1, 128), device="cuda", dtype=dtype)
    if dtype == torch.float16:
        x = torch.full((rows, 128), 516, device="cuda", dtype=dtype)
        bias = torch.full((128,), -1024, device="cuda", dtype=dtype)
    else:
        x = torch.ones((rows, 128), device="cuda", dtype=dtype)
        x[:, -1] = 1.25
        bias = torch.full((128,), -0.5, device="cuda", dtype=dtype)
    # An independent dense weight oracle: 0x11111111 encodes eight ones per word.
    weight = torch.ones((128, 128), device="cuda", dtype=dtype)
    expected = (torch.mm(x, weight, out_dtype=torch.float32) + bias.float()).to(dtype)
    return x, qweight, scales, qzeros, bias, expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 16])
def test_fused_paroquant_bias_precedes_output_rounding(dtype, rows):
    x, qweight, scales, qzeros, bias, expected = _case(dtype, rows)
    kernel = paroquant_gemm_triton_decode if rows == 1 else paroquant_gemm_triton_prefill
    with torch.inference_mode():
        actual = kernel(x, qweight, scales, qzeros, bias)
        unbiased = kernel(x, qweight, scales, qzeros)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert expected[0, 0].item() == (65024 if dtype == torch.float16 else 128)
    weight = torch.ones((128, 128), device=x.device, dtype=dtype)
    torch.testing.assert_close(
        unbiased, torch.mm(x, weight, out_dtype=torch.float32).to(dtype), rtol=0, atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 16])
def test_fused_paroquant_bias_matches_independent_packed_code_oracle(dtype, rows):
    torch.manual_seed(816)
    codes = torch.randint(0, 16, (128, 128), device="cuda", dtype=torch.int32)
    zero_codes = torch.full((1, 128), 7, device="cuda", dtype=torch.int32)
    qweight = pack_awq_columns(codes, 4, order=AWQ_ORDER)
    qzeros = pack_awq_columns(zero_codes, 4, order=AWQ_ORDER)
    scales = torch.full((1, 128), 0.125, device="cuda", dtype=dtype)
    bias_storage = ((torch.arange(256, device="cuda") % 17) - 8).to(dtype) / 8
    bias = bias_storage[::2]
    assert not bias.is_contiguous()
    x = torch.randint(-8, 9, (rows, 128), device="cuda").to(dtype)

    kernel = paroquant_gemm_triton_decode if rows == 1 else paroquant_gemm_triton_prefill
    with torch.inference_mode():
        actual = kernel(x, qweight, scales, qzeros, bias)
    weight_oracle = (codes.float() - zero_codes.float()) * scales.float()
    expected = (x.float() @ weight_oracle + bias.float()).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,operand", [
    (1, "input_columns"), (16, "input_rows"), (16, "input_columns"),
    (16, "qweight"), (16, "scales"), (16, "qzeros"),
])
def test_fused_paroquant_accepts_strided_operands(dtype, rows, operand):
    columns = 128
    x = torch.randint(-2, 3, (rows, columns), device="cuda").to(dtype)
    qweight = torch.full((columns, columns // 8), 0x11111111, device="cuda", dtype=torch.int32)
    scales = torch.ones((1, columns), device="cuda", dtype=dtype)
    qzeros = torch.zeros((1, columns // 8), device="cuda", dtype=torch.int32)
    bias = torch.zeros((columns,), device="cuda", dtype=dtype)
    arguments = {"input": x, "qweight": qweight, "scales": scales, "qzeros": qzeros}

    key = "input" if operand.startswith("input") else operand
    source = arguments[key]
    if operand == "input_rows":
        storage = torch.empty((rows, columns + 8), device="cuda", dtype=dtype)
        view = storage[:, :columns]
    else:
        storage = torch.empty((source.shape[0], source.shape[1] * 2), device="cuda", dtype=source.dtype)
        view = storage[:, ::2]
    storage.fill_(-1 if operand == "qzeros" else 0)
    view.copy_(source)
    assert not view.is_contiguous()
    arguments[key] = view

    kernel = paroquant_gemm_triton_decode if rows == 1 else paroquant_gemm_triton_prefill
    with torch.inference_mode():
        actual = kernel(
            arguments["input"], arguments["qweight"], arguments["scales"], arguments["qzeros"], bias,
        )
        dequantized = paroquant_dequantize_triton(
            arguments["qweight"], arguments["scales"], arguments["qzeros"],
        )
    # 0x11111111 is eight ones, qzeros=0, and scales=1: the logical weight is all ones.
    expected = x.float().sum(dim=-1, keepdim=True).expand(rows, columns).to(dtype)
    torch.cuda.synchronize()
    torch.testing.assert_close(dequantized, torch.ones_like(dequantized), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fused_paroquant_rejects_malformed_bias_before_launch():
    x, qweight, scales, qzeros, bias, _ = _case(torch.float16, 1)
    with pytest.raises(ValueError, match="ParoQuant Triton bias must be"):
        paroquant_gemm_triton_decode(x, qweight, scales, qzeros, bias[:64])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 16])
def test_paroquant_module_fused_and_dense_bias_paths(dtype, rows, monkeypatch):
    x, qweight, scales, qzeros, bias, expected = _case(dtype, rows)
    module = paro_module.ParoQuantTritonLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=128, bias=True, register_buffers=True, krot=8,
    ).to(device="cuda", dtype=dtype).eval()
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    module.post_init()
    module.paroquant_triton_autotune_enabled = False

    dequant_refs = []
    original_dequant = paro_module.paroquant_dequantize_triton

    def tracked_dequant(*args, **kwargs):
        weight = original_dequant(*args, **kwargs)
        dequant_refs.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(paro_module, "paroquant_dequantize_triton", tracked_dequant)
    with torch.inference_mode():
        fused = module(x)
        dense = module._run_plan("dense", x)
    torch.testing.assert_close(fused, expected, rtol=0, atol=0)
    torch.testing.assert_close(dense, expected, rtol=0, atol=0)
    assert all(ref() is None for ref in dequant_refs)

    # The fused path has no backward for differentiable inputs or bias.
    module.bias.requires_grad_(True)
    kind = "decode" if rows == 1 else "prefill"
    assert module._select_plan(kind, x) == "dense"
    output = module(x)
    output.float().sum().backward()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(module.bias.grad, torch.full_like(bias, rows), rtol=0, atol=0)

    module.bias.grad = None
    direct = module._run_plan("decode_fused", x)
    direct.float().sum().backward()
    torch.testing.assert_close(direct, expected, rtol=0, atol=0)
    torch.testing.assert_close(module.bias.grad, torch.full_like(bias, rows), rtol=0, atol=0)

    module.bias.requires_grad_(False)
    input_with_grad = x.detach().clone().requires_grad_(True)
    assert module._select_plan(kind, input_with_grad) == "dense"
    differentiable = module(input_with_grad)
    differentiable.float().sum().backward()
    torch.testing.assert_close(differentiable, expected, rtol=0, atol=0)
    torch.testing.assert_close(input_with_grad.grad, torch.full_like(x, 128), rtol=0, atol=0)

    direct_input = x.detach().clone().requires_grad_(True)
    direct = module._run_plan("decode_fused", direct_input)
    direct.float().sum().backward()
    torch.testing.assert_close(direct, expected, rtol=0, atol=0)
    torch.testing.assert_close(direct_input.grad, torch.full_like(x, 128), rtol=0, atol=0)


def test_paroquant_dense_preserves_stored_bias_precision_before_bf16_rounding():
    x, qweight, scales, qzeros, _, _ = _case(torch.bfloat16, 1)
    module = paro_module.ParoQuantTritonLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=128, bias=True, register_buffers=True, krot=8,
    ).to(device="cuda", dtype=torch.bfloat16).eval()
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias = torch.full((128,), 0.251, device="cuda", dtype=torch.float16)
    module.post_init()
    weight = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
    expected = (torch.mm(x, weight, out_dtype=torch.float32) + module.bias.float()).to(x.dtype)

    with torch.inference_mode():
        fused = module._run_plan("decode_fused", x)
        dense = module._run_plan("dense", x)
    assert expected[0, 0].item() == 129
    torch.testing.assert_close(fused, expected, rtol=0, atol=0)
    torch.testing.assert_close(dense, expected, rtol=0, atol=0)
