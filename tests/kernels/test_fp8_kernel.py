# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import weakref

import pytest
import torch
import torch.nn as nn

import gptqmodel.nn_modules.qlinear.fp8 as fp8_module
from gptqmodel.quantization.dtype import available_float8_dtype_names


def _available_fp8_quant_formats():
    return [fmt for fmt in available_float8_dtype_names() if fmt != "float8_e8m0fnu"]


def _make_fp8_kernel(*, in_features: int = 64, out_features: int = 64) -> fp8_module.TorchFP8Linear:
    torch.manual_seed(3)
    linear = nn.Linear(in_features, out_features, bias=True).eval()
    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
        format="float8_e4m3fn",
        weight_scale_method="row",
        weight_block_size=None,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    return kernel.eval()


def _weight_only_oracle(kernel: fp8_module.TorchFP8Linear, x: torch.Tensor) -> torch.Tensor:
    """Decode the packed FP8 payload with independent FP64 arithmetic.

    This does not call ``dequantize_weight`` or any other code under test.
    For range-safe row/tensor scales the native W8A16 GEMM casts the exact FP8
    codes and applies the scale to an FP32 output (#5146), so the oracle keeps
    the decoded weight un-narrowed. Block scales still divide the exact codes
    into the activation dtype (FP32 opmath, one rounding), so that oracle keeps
    the same weight narrowing as the kernel.
    """

    codes = kernel.weight.to(torch.float64)
    scale = kernel.weight_scale_inv.to(torch.float64)

    def apply_scale(weight: torch.Tensor, scale_tensor: torch.Tensor) -> torch.Tensor:
        if kernel.weight_scale_semantics == "multiplier":
            return weight * scale_tensor
        return weight / scale_tensor

    if kernel.weight_scale_method == "tensor":
        decoded = apply_scale(codes, scale)
    elif kernel.weight_scale_method == "row":
        decoded = apply_scale(codes, scale.reshape(-1, 1))
    else:
        block_rows, block_cols = kernel.weight_block_size
        decoded = apply_scale(
            codes.reshape(
                kernel.out_features // block_rows,
                block_rows,
                kernel.in_features // block_cols,
                block_cols,
            ),
            scale[:, None, :, None],
        ).reshape(kernel.out_features, kernel.in_features)

    if kernel.weight_scale_method == "block":
        decoded = decoded.to(x.dtype).to(torch.float64)
    output = x.to(torch.float64) @ decoded.t()
    if kernel.bias is not None:
        output = output + kernel.bias.to(torch.float64)
    return output


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA FP8 scaled matmul required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("prefix", [(2,), (2, 3)])
def test_fp8_empty_cuda_forward_preserves_scaled_mm_for_later_tokens(monkeypatch, dtype, biased, prefix):
    linear = nn.Linear(64, 64, bias=biased).eval()
    module = fp8_module.TorchFP8Linear(
        bits=8, group_size=-1, desc_act=False, sym=True,
        in_features=64, out_features=64, bias=biased,
        format="float8_e4m3fn", weight_scale_method="tensor",
        activation_quant_method="per_token",
    )
    module.pack_original(linear, scales=None, zeros=None)
    module = module.eval().to(device="cuda")
    x = torch.randn(*prefix, 64, device="cuda", dtype=dtype)
    empty = x[:0]
    x_flat = x.reshape(-1, 64)
    if not module._can_use_scaled_mm(x_flat):
        pytest.skip("Native FP8 scaled matmul unavailable on this GPU")
    try:
        module._forward_scaled_mm(x_flat)
    except Exception as exc:
        pytest.skip(f"Native FP8 scaled matmul unavailable: {exc}")

    native_calls = []
    dequant_refs = []
    native_forward = module._forward_scaled_mm
    dequantize_weight = module.dequantize_weight

    def counted_native(tensor):
        native_calls.append(tuple(tensor.shape))
        return native_forward(tensor)

    def tracked_dequant(*args, **kwargs):
        weight = dequantize_weight(*args, **kwargs)
        dequant_refs.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(module, "_forward_scaled_mm", counted_native)
    monkeypatch.setattr(module, "dequantize_weight", tracked_dequant)
    with torch.inference_mode():
        empty_output = module(empty)
        assert empty_output.shape == empty.shape[:-1] + (64,)
        assert empty_output.dtype == dtype
        assert not module._scaled_mm_hard_disabled
        assert native_calls == []
        assert all(ref() is None for ref in dequant_refs)

        output = module(x)
        assert output.shape == x.shape
        assert output.dtype == dtype
        assert torch.isfinite(output).all()
        assert native_calls == [tuple(x_flat.shape)]
        assert not module._scaled_mm_hard_disabled


@pytest.mark.parametrize("fmt", _available_fp8_quant_formats())
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
def test_fp8_pack_matches_reference_quantizer(fmt: str, weight_scale_method: str):
    torch.manual_seed(0)
    linear = nn.Linear(64, 64, bias=True).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None

    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)

    expected_weight, expected_scale_inv = fp8_module.quantize_fp8_weight(
        linear.weight.detach().to(torch.float32),
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
    )

    assert torch.equal(kernel.weight, expected_weight)
    assert torch.equal(kernel.weight_scale_inv, expected_scale_inv)


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="float8_e8m0fnu unavailable")
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
def test_fp8_pack_rejects_e8m0fnu(weight_scale_method: str):
    linear = nn.Linear(64, 64, bias=True).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None

    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        format="float8_e8m0fnu",
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
    )

    with pytest.raises(ValueError, match="dequantization of existing checkpoints"):
        kernel.pack_original(linear=linear, scales=None, zeros=None)


@pytest.mark.parametrize("fmt", _available_fp8_quant_formats())
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_fp8_forward_matches_dequantized_reference(fmt: str, weight_scale_method: str, device: str):
    torch.manual_seed(1)
    linear = nn.Linear(64, 64, bias=True).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None

    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel = kernel.to(device=device).eval()

    x_dtype = torch.float32 if device == "cpu" else torch.float16
    x = torch.randn(5, 64, device=device, dtype=x_dtype)

    with torch.inference_mode():
        out = kernel(x)
        if device == "cuda":
            # The shipped default is W8A16: it must not quantize activations.
            assert not kernel._can_use_scaled_mm(x.reshape(-1, 64))
        expected = torch.matmul(
            x,
            kernel.dequantize_weight(device=device, dtype=x_dtype),
        )
        if kernel.bias is not None:
            expected = expected + kernel.bias.to(device=device, dtype=x_dtype)

    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def _packed_gpu_kernel(
    *,
    fmt: str = "float8_e4m3fn",
    weight_scale_method: str = "tensor",
    bias: bool = True,
    activation_quant_method: str | None = None,
) -> fp8_module.TorchFP8Linear:
    torch.manual_seed(3)
    linear = nn.Linear(64, 64, bias=bias).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None
    kwargs = {}
    if activation_quant_method is not None:
        kwargs["activation_quant_method"] = activation_quant_method
    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=bias,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
        **kwargs,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    return kernel.eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("fmt", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_default_forward_is_w8a16(fmt: str, weight_scale_method: str, dtype: torch.dtype):
    """The shipped default must dequantize the weight and keep activations wide."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")

    kernel = _packed_gpu_kernel(fmt=fmt, weight_scale_method=weight_scale_method).to("cuda")
    x = torch.randn(5, 64, device="cuda", dtype=dtype)

    # No activation quantization unless it was explicitly requested.
    assert not kernel._can_use_scaled_mm(x)

    with torch.inference_mode():
        out = kernel(x)

    oracle = _weight_only_oracle(kernel, x)
    assert out.dtype == dtype
    # With the weight narrowing matched, only the FP32-vs-FP64 reduction and
    # the final output rounding remain. Half a bf16 ulp at unity is
    # 2^-9 ~= 2e-3, which is the bound this tolerance allows; an oracle built
    # from the un-narrowed FP32 weight differs by a full bf16 ulp (2^-8) and
    # still fails here.
    torch.testing.assert_close(out, oracle.to(dtype), rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("fmt", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_default_forward_is_batch_invariant(fmt: str):
    """Rows must not depend on the largest activation batched beside them."""
    kernel = _packed_gpu_kernel(fmt=fmt, weight_scale_method="tensor").to("cuda")
    rows = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    outlier = torch.full((1, 64), 2000.0, device="cuda", dtype=torch.float16)

    assert not kernel._can_use_scaled_mm(rows)

    with torch.inference_mode():
        alone = kernel(rows)
        batched = kernel(torch.cat([rows, outlier], dim=0))[:4]

    torch.testing.assert_close(batched, alone, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fp8_w8a8_is_opt_in_and_uses_per_token_scales():
    """W8A8 is reachable only through the explicit method, and per token."""
    default = _packed_gpu_kernel(bias=False).to("cuda")
    opted = _packed_gpu_kernel(
        bias=False, activation_quant_method="per_token"
    ).to("cuda")

    rows = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    outlier = torch.full((1, 64), 2000.0, device="cuda", dtype=torch.float16)

    assert not default._can_use_scaled_mm(rows)
    assert opted._can_use_scaled_mm(rows)

    with torch.inference_mode():
        codes_alone, scale_alone = opted._quantize_input_for_scaled_mm(rows)
        codes_batched, scale_batched = opted._quantize_input_for_scaled_mm(
            torch.cat([rows, outlier], dim=0)
        )

    assert scale_alone.shape == (4, 1)
    assert scale_batched.shape == (5, 1)
    # One scale per token: a neighbour's 2000.0 cannot move these rows.
    assert torch.equal(scale_alone, scale_batched[:4])
    assert torch.equal(codes_alone, codes_batched[:4])

    # The method is the switch; the default module never takes the path.
    default.set_activation_quant_method("per_token")
    assert default._can_use_scaled_mm(rows)
    default.set_activation_quant_method("none")
    assert not default._can_use_scaled_mm(rows)
    with pytest.raises(ValueError, match="activation quantization"):
        default.set_activation_quant_method("per_tensor")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("fmt", ["float8_e4m3fn"])
def test_fp8_w8a8_forward_is_batch_invariant(fmt: str):
    kernel = _packed_gpu_kernel(
        fmt=fmt, bias=False, activation_quant_method="per_token"
    ).to("cuda")
    rows = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    outlier = torch.full((1, 64), 2000.0, device="cuda", dtype=torch.float16)

    if not kernel._can_use_scaled_mm(rows):
        pytest.skip("Native FP8 rowwise scaled matmul unavailable on this GPU")

    with torch.inference_mode():
        alone = kernel(rows)
    assert not kernel._scaled_mm_hard_disabled, "W8A8 opt-in silently fell back"

    with torch.inference_mode():
        batched = kernel(torch.cat([rows, outlier], dim=0))[:4]

    torch.testing.assert_close(batched, alone, rtol=2e-3, atol=2e-3)


def test_fp8_forward_dequantizes_per_call_without_dense_cache(monkeypatch):
    kernel = _make_fp8_kernel()
    x = torch.randn(5, 64)

    calls = 0
    dequantize_weight = kernel.dequantize_weight

    def counted_dequantize_weight(*args, **kwargs):
        nonlocal calls
        calls += 1
        return dequantize_weight(*args, **kwargs)

    monkeypatch.setattr(kernel, "dequantize_weight", counted_dequantize_weight)

    with torch.inference_mode():
        first = kernel(x)
        second = kernel(x)

    assert calls == 2
    assert not hasattr(kernel, "_dequant_weight_cache")
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_fp8_optimize_compiles_cpu_forward(monkeypatch):
    kernel = _make_fp8_kernel()
    x = torch.randn(7, 64)
    with torch.inference_mode():
        eager = kernel(x)

    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return lambda value, bias=None: fn(value, bias=bias)

    monkeypatch.setattr(fp8_module, "TORCH_HAS_COMPILE", True)
    monkeypatch.setattr(fp8_module, "torch_compile", fake_compile)

    assert kernel.optimize(backend="inductor") is kernel
    assert kernel.optimized
    assert kernel._compiled_forward is not None
    assert calls == [{"backend": "inductor", "mode": None, "fullgraph": False}]

    with torch.inference_mode():
        compiled = kernel(x)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)

    kernel.optimize(backend="inductor")
    assert len(calls) == 1


def test_fp8_compiled_forward_invalidates_on_move_and_state_load(monkeypatch):
    kernel = _make_fp8_kernel()
    monkeypatch.setattr(fp8_module, "TORCH_HAS_COMPILE", True)
    monkeypatch.setattr(
        fp8_module,
        "torch_compile",
        lambda fn, **_kwargs: lambda value, bias=None: fn(value, bias=bias),
    )

    kernel.optimize()
    assert kernel._compiled_forward is not None
    kernel._apply(lambda tensor: tensor)
    assert kernel._compiled_forward is None
    assert not kernel.optimized

    kernel.optimize()
    assert kernel._compiled_forward is not None
    with torch.inference_mode():
        kernel.load_state_dict(kernel.state_dict())
    assert kernel._compiled_forward is None
    assert not kernel.optimized


def test_fp8_compiled_forward_falls_back_to_eager(monkeypatch):
    kernel = _make_fp8_kernel()
    x = torch.randn(7, 64)
    with torch.inference_mode():
        expected = kernel(x)

    def fail(_value, bias=None):
        del bias
        raise RuntimeError("compiled graph failed")

    kernel._compiled_forward = fail
    kernel.optimized = True

    with torch.inference_mode():
        actual = kernel(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert kernel._compiled_forward is None
    assert not kernel.optimized


def test_fp8_compiled_forward_reads_live_weight_without_dense_cache(monkeypatch):
    kernel = _make_fp8_kernel()
    x = torch.randn(7, 64)
    monkeypatch.setattr(fp8_module, "TORCH_HAS_COMPILE", True)
    monkeypatch.setattr(
        fp8_module,
        "torch_compile",
        lambda fn, **_kwargs: lambda value, bias=None: fn(value, bias=bias),
    )
    kernel.optimize()

    with torch.inference_mode():
        before = kernel(x)
        kernel.weight.zero_()
        after = kernel(x)
        expected = torch.matmul(
            x,
            kernel.dequantize_weight(device=x.device, dtype=x.dtype),
        ) + kernel.bias

    assert not torch.equal(before, after)
    torch.testing.assert_close(after, expected, rtol=0, atol=0)
    assert not hasattr(kernel, "_dequant_weight_cache")


def test_fp8_optimize_skips_non_cpu_weights(monkeypatch):
    kernel = _make_fp8_kernel()
    kernel = kernel.to("meta")
    calls = []
    monkeypatch.setattr(fp8_module, "TORCH_HAS_COMPILE", True)
    monkeypatch.setattr(
        fp8_module,
        "torch_compile",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    kernel.optimize()

    assert calls == []
    assert not kernel.optimized


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fmt", _available_fp8_quant_formats())
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row"])
def test_fp8_scaled_mm_matches_dense_reference(fmt: str, weight_scale_method: str):
    torch.manual_seed(2)
    linear = nn.Linear(64, 64, bias=False).eval()

    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=False,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        activation_quant_method="per_token",
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel = kernel.to(device="cuda").eval()

    x = torch.randn(7, 64, device="cuda", dtype=torch.float16)
    if not kernel._can_use_scaled_mm(x):
        pytest.skip("scaled_mm is not available for this environment.")

    try:
        with torch.inference_mode():
            scaled_mm_out = kernel._forward_scaled_mm(x)
    except Exception as exc:
        pytest.skip(f"scaled_mm path unavailable: {exc}")

    with torch.inference_mode():
        x_q, scale_a = kernel._quantize_input_for_scaled_mm(x)
        x_q_dequant = x_q.to(torch.float16) * scale_a.to(torch.float16)
        dense_out = torch.matmul(
            x_q_dequant,
            kernel.dequantize_weight(device="cuda", dtype=torch.float16),
        )

    torch.testing.assert_close(scaled_mm_out, dense_out, rtol=5e-2, atol=5e-2)
