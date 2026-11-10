# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NF12-AD codec and inference-oracle tests.

The codec is a lossless BF16 encoding, so decode must reproduce every input
bit. The CUDA kernels share the same format; these tests validate the Torch
oracle used for quantization and CPU inference.

Native HIP cases were validated on ROCm 10.1.0 / HIP 7.16.26385 with R9700
(gfx1201); see gptqmodel_ext/nf12_hip/README.md for the full toolchain.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.utils import nf12_codec


def _adversarial_bits(n: int, k: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 1 << 16, (n, k), dtype=torch.int32, generator=generator)
    # Cover signed zero, subnormals, Inf, NaN, and the smallest/largest normals.
    edge = torch.tensor(
        [
            0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F,
            0x7F80, 0xFF80, 0x7FC0, 0x7FFF, 0xFFC1, 0x3F80, 0xBF80,
        ],
        dtype=torch.int32,
    )
    bits.view(-1)[: edge.numel()] = edge
    return bits.to(torch.int16).view(torch.bfloat16)


@pytest.mark.parametrize("shape", [(16, 64), (32, 128), (48, 192)])
def test_roundtrip_is_bit_exact(shape):
    n, k = shape
    weight = _adversarial_bits(n, k)
    packed, escapes, exp_base = nf12_codec.encode(weight)
    assert packed.dtype == torch.uint8
    assert packed.shape == (n, k * 3 // 2)
    assert escapes.dtype == torch.int16
    assert escapes.shape[1] == 8
    assert 0 <= exp_base <= 120

    decoded = nf12_codec.decode(packed, escapes, exp_base)
    assert decoded.dtype == torch.bfloat16
    assert torch.equal(decoded.view(torch.int16), weight.contiguous().view(torch.int16))


def test_forced_exponent_base_roundtrips():
    weight = _adversarial_bits(16, 64, seed=3)
    for exp_base in (0, 60, 120):
        packed, escapes, base = nf12_codec.encode(weight, exp_base=exp_base)
        assert base == exp_base
        decoded = nf12_codec.decode(packed, escapes, exp_base)
        assert torch.equal(decoded.view(torch.int16), weight.view(torch.int16))


def test_bf16t_twin_roundtrips():
    weight = torch.randn(32, 128, dtype=torch.bfloat16)
    twin = nf12_codec.pack_bf16t(weight)
    assert twin.shape == (32, 256)
    restored = nf12_codec.unpack_bf16t(twin, 32, 128)
    assert torch.equal(restored.view(torch.int16), weight.view(torch.int16))


def test_linear_matches_fp32_oracle():
    torch.manual_seed(0)
    weight = torch.randn(32, 128, dtype=torch.bfloat16)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    bias = torch.randn(32, dtype=torch.bfloat16)
    packed, escapes, exp_base = nf12_codec.encode(weight)
    decoded = nf12_codec.decode(packed, escapes, exp_base)

    actual = nf12_codec.linear(x, decoded, bias)
    expected = (x.float() @ weight.float().T + bias.float()).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_encode_weight_casts_to_bf16():
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    weight = torch.randn(16, 64, dtype=torch.float32)
    result = quantize_nf12_ad_weight(weight)
    decoded = nf12_codec.decode(result.packed, result.escapes, result.exp_base)
    assert torch.equal(decoded, weight.to(torch.bfloat16))
    assert result.storage_bytes == result.packed.numel() + result.escapes.numel() * 2 + 1


def test_module_saves_nf12_packed_and_retains_no_dense_cache():
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.utils.backend import BACKEND

    torch.manual_seed(0)
    linear = torch.nn.Linear(128, 32, bias=True).to(torch.bfloat16)
    module = NF12ADLinear(
        bits=12,
        in_features=128,
        out_features=32,
        bias=True,
        backend=BACKEND.NF12_AD,
        pack_dtype=torch.int32,
        exp_base=54,
    )
    module.pack_original(linear)

    # A quantized checkpoint must expose its packed weight as `nf12_packed`,
    # not as a dense `weight` parameter that a loader could mistake for BF16.
    keys = set(module.state_dict())
    assert "nf12_packed" in keys
    assert module.nf12_packed.dtype == torch.uint8
    assert module.nf12_packed.shape == (32, 192)
    assert "weight" not in module._parameters
    assert int(module.nf12_exp_base.item()) == 54

    x = torch.randn(4, 128, dtype=torch.bfloat16)
    expected = linear(x).float()
    for _ in range(3):
        actual = module(x)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual.float(), expected, rtol=2e-3, atol=2e-3)

    # Repeated inference must not leave a decoded dense weight behind.
    assert set(module._buffers) == {"nf12_packed", "nf12_escapes", "nf12_exp_base", "bias"}
    decoded = module.dequantize_weight()
    assert torch.equal(
        decoded.view(torch.int16),
        linear.weight.detach().to(torch.bfloat16).view(torch.int16),
    )


def test_accelerate_buffer_replacement_refreshes_cached_exponent_base():
    """A pre-load repr must not pin the placeholder exponent for inference."""
    from accelerate.utils.modeling import set_module_tensor_to_device

    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    module = NF12ADLinear(bits=12, in_features=64, out_features=16)
    assert "exp_base=0" in repr(module)

    weight = torch.full((16, 64), 2.0**-12, dtype=torch.bfloat16)
    weight[:, ::2] = -weight[:, ::2]
    result = quantize_nf12_ad_weight(weight, exp_base=54)
    assert result.escapes.shape == (0, 8)
    set_module_tensor_to_device(module, "nf12_packed", "cpu", value=result.packed)
    set_module_tensor_to_device(module, "nf12_escapes", "cpu", value=result.escapes)
    set_module_tensor_to_device(
        module, "nf12_exp_base", "cpu",
        value=torch.tensor([result.exp_base], dtype=torch.uint8),
    )

    assert torch.equal(module.dequantize_weight().view(torch.int16), weight.view(torch.int16))
    x = torch.ones((1, 64), dtype=torch.bfloat16)
    x[:, 1::2] = 0
    expected = (x.float() @ weight.float().T).to(torch.bfloat16)
    for _ in range(2):
        assert torch.equal(module(x), expected)
    assert "exp_base=54" in repr(module)
    assert "weight" not in module.state_dict()


def test_exponent_cache_tracks_versioned_updates_and_explicit_clear():
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear

    module = NF12ADLinear(bits=12, in_features=64, out_features=16)
    module.nf12_exp_base = torch.tensor([54], dtype=torch.uint8)
    assert module._exp_base() == 54
    module.nf12_exp_base.fill_(55)
    assert module._exp_base() == 55

    with torch.inference_mode():
        module.nf12_exp_base = torch.tensor([56], dtype=torch.uint8)
        assert module._exp_base() == 56
        module.nf12_exp_base.fill_(57)
        module.clear_weight_cache()
        assert module._exp_base() == 57


def test_repr_handles_meta_and_unallocated_exponent_buffers():
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear

    with torch.device("meta"):
        meta = NF12ADLinear(bits=12, in_features=64, out_features=16)
    assert meta.nf12_exp_base.device.type == "meta"
    assert "exp_base=unavailable" in repr(meta)

    unallocated = NF12ADLinear(bits=12, in_features=64, out_features=16,
                               register_buffers=False)
    assert "exp_base=unavailable" in repr(unallocated)


def test_forward_on_meta_returns_metadata_output():
    """A meta module must answer forward without materializing any value.

    Lazy-load and offload shells build quant-linear layers on the meta device.
    Decoding or reading `exp_base` there raises `Tensor.item() cannot be
    called on meta tensors`, so forward returns the shape/dtype-correct empty
    output that the zero-row path already uses.
    """
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear

    with torch.device("meta"):
        linear = NF12ADLinear(bits=12, in_features=64, out_features=16, bias=False)
    x = torch.zeros((2, 3, 64), dtype=torch.bfloat16, device="meta")

    out = linear(x)

    assert out.shape == (2, 3, 16)
    assert out.device.type == "meta"
    assert out.dtype == torch.bfloat16
    assert "weight" not in linear.state_dict()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_pack_and_reload_preserve_source_bias_precision(dtype):
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear

    source = torch.nn.Linear(64, 16, bias=True).to(dtype)
    with torch.no_grad():
        source.weight.fill_(2.0**-12)
        source.weight[:, ::2].neg_()
        source.bias.fill_(100.25)

    packed = NF12ADLinear(bits=12, in_features=64, out_features=16,
                          bias=True, dtype=dtype, exp_base=54)
    packed.pack_original(source)
    x = torch.zeros((2, 64), dtype=dtype)
    expected = source(x)
    assert torch.equal(packed(x), expected)
    assert packed.bias.dtype == dtype
    assert torch.equal(packed.bias, source.bias)

    reloaded = NF12ADLinear(bits=12, in_features=64, out_features=16,
                            bias=True, dtype=dtype)
    reloaded.load_state_dict(packed.state_dict())
    assert reloaded.bias.dtype == dtype
    assert torch.equal(reloaded(x), expected)


def test_loader_resizes_data_dependent_escape_buffer(tmp_path):
    """The escape table has a checkpoint-dependent row count, not a config size."""

    from safetensors.torch import save_file

    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.model import _prepare_variable_quant_buffers

    module = NF12ADLinear(
        bits=12,
        in_features=64,
        out_features=16,
        bias=False,
        backend=BACKEND.NF12_AD,
        pack_dtype=torch.int32,
    )
    assert module.nf12_escapes.shape == (0, 8)

    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {
            "blk.nf12_packed": torch.zeros((16, 96), dtype=torch.uint8),
            "blk.nf12_escapes": torch.zeros((3, 8), dtype=torch.int16),
            "blk.nf12_exp_base": torch.zeros((1,), dtype=torch.uint8),
        },
        str(checkpoint),
    )
    model = torch.nn.Module()
    model.add_module("blk", module)
    _prepare_variable_quant_buffers(model, str(checkpoint))
    assert module.nf12_escapes.shape == (3, 8)


# ---------------------------------------------------------------------------
# Accuracy criterion
# ---------------------------------------------------------------------------
def _drift(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """Elementwise drift, computed in float64.

    ``max_rel`` over *all* elements is deliberately not a gate: it diverges on
    near-zero outputs by construction, where the absolute error is far below one
    BF16 ULP. The gate is ``|got - ref| <= atol + rtol * |ref|``.
    """
    g, r = got.double(), ref.double()
    err = (g - r).abs()
    atol, rtol = 2e-3, 2e-3
    violations = int((err > atol + rtol * r.abs()).sum().item())
    return {
        "max_abs": err.max().item(),
        "violations": violations,
        "exact": (got.view(torch.int16) == ref.view(torch.int16)).float().mean().item()
        if got.dtype == ref.dtype == torch.bfloat16
        else float("nan"),
    }


def test_near_zero_outputs_do_not_trip_the_relative_bound():
    """A tiny absolute error on a near-zero reference is 100% 'relative' error.

    This is the regression the naive ``max_rel == 0`` gate produced: a correctly
    rounded output of 1e-9 against a reference of 2e-9 has 50% relative error
    and zero real inaccuracy.
    """
    ref = torch.tensor([0.0, 1e-9, -3e-8, 1e-30], dtype=torch.bfloat16)
    got = ref.clone()
    got[1] = 2e-9  # one BF16 ULP at this magnitude
    drift = _drift(got, ref)
    assert drift["violations"] == 0
    assert drift["max_abs"] <= 2e-3


def test_linear_e2e_matches_fp32_oracle_on_cpu():
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    torch.manual_seed(0)
    n, k, m = 256, 256, 8
    weight = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(n) * 0.1).to(torch.bfloat16)
    linear = NF12ADLinear(bits=12, in_features=k, out_features=n, bias=True)
    linear.load_quant_result(quantize_nf12_ad_weight(weight), bias=bias)
    x = (torch.randn(m, k) * 0.05).to(torch.bfloat16)

    expected = (torch.matmul(x.float(), weight.float().t()) + bias.float()).to(torch.bfloat16)
    drift = _drift(linear(x), expected)
    assert drift["violations"] == 0
    # Persistent storage must stay quantized: no dense weight buffer survives the
    # forward. A bias is an ordinary additive parameter, not an expanded weight.
    state = set(linear.state_dict())
    assert "weight" not in state
    assert {"nf12_packed", "nf12_escapes", "nf12_exp_base"} <= state


@pytest.mark.parametrize(
    ("weight_value", "activation_value"),
    [
        (1.00117e-8, 10.0),
        (99840.0, 0.064),
    ],
)
def test_linear_fallback_keeps_bf16_weight_range_for_fp16_input(
    weight_value,
    activation_value,
):
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    weight = torch.zeros((256, 256), dtype=torch.bfloat16)
    weight[0, 0] = weight_value
    bias = torch.zeros(256, dtype=torch.bfloat16)
    linear = NF12ADLinear(
        bits=12,
        in_features=256,
        out_features=256,
        bias=True,
    )
    linear.load_quant_result(quantize_nf12_ad_weight(weight), bias=bias)
    x = torch.zeros((1, 256), dtype=torch.float16)
    x[0, 0] = activation_value
    expected = (
        x.float() @ weight.float().t() + bias.float()
    ).to(torch.float16)

    assert torch.equal(linear(x), expected)


_GPU_OR_CPU = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
    ),
]


@pytest.mark.parametrize("device", _GPU_OR_CPU)
@pytest.mark.parametrize(
    ("weight_value", "activation_value"),
    [
        # A BF16 weight outside the FP16 range: narrowing it to the FP16
        # activation dtype yields `inf`, although the exact product is 1.
        (65536.0, 2.0**-16),
        (-65536.0, 2.0**-16),
        # A BF16 weight below the FP16 subnormal range would be flushed to 0.
        (1.00117e-8, 10.0),
        (99840.0, 0.064),
    ],
)
def test_linear_fallback_preserves_bf16_weight_range(
    device, weight_value, activation_value, monkeypatch
):
    """The decoded fallback must not narrow a BF16 weight to the FP16 range.

    NF12-AD checkpoints are lossless BF16. Converting the decoded matrix to the
    activation dtype before the product loses representable final results on
    every accelerator, so the fallback multiplies wide and rounds the output
    only once.
    """
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    weight = torch.zeros((256, 256), dtype=torch.bfloat16)
    weight[0, 0] = weight_value
    linear = NF12ADLinear(bits=12, in_features=256, out_features=256, bias=False)
    linear.load_quant_result(quantize_nf12_ad_weight(weight))
    linear = linear.to(device)
    monkeypatch.setattr(linear, "_native_forward", lambda x, bias: None)

    x = torch.zeros((1, 256), dtype=torch.float16, device=device)
    x[0, 0] = activation_value
    oracle = (x.double() @ weight.double().to(device).t()).to(torch.float16)
    assert torch.isfinite(oracle).all()

    with torch.inference_mode():
        first = linear(x)
        repeated = linear(x)

    assert first.dtype == torch.float16
    assert first.shape == oracle.shape
    assert torch.isfinite(first).all()
    assert torch.equal(first, oracle)
    assert torch.equal(first, repeated)
    # The wide scratch is local to the forward; only packed weights persist.
    assert "weight" not in linear.state_dict()
    assert {"nf12_packed", "nf12_escapes", "nf12_exp_base"} <= set(linear.state_dict())


@pytest.mark.parametrize("device", _GPU_OR_CPU)
def test_linear_fallback_rounds_only_the_final_bf16_output(device, monkeypatch):
    """A BF16 activation keeps the BF16 weight range and the BF16 output dtype."""
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    weight = torch.zeros((64, 128), dtype=torch.bfloat16)
    weight[0, 0] = 65536.0
    weight[1, 1] = -65536.0
    bias = torch.zeros(64, dtype=torch.bfloat16)
    bias[0] = -65536.0
    linear = NF12ADLinear(bits=12, in_features=128, out_features=64, bias=True)
    linear.load_quant_result(quantize_nf12_ad_weight(weight), bias=bias)
    linear = linear.to(device)
    monkeypatch.setattr(linear, "_native_forward", lambda x, bias: None)

    x = torch.zeros((2, 3, 128), dtype=torch.bfloat16, device=device)
    x[0, 0, 0] = 1.0
    x[1, 2, 1] = 1.0
    oracle = (
        x.double() @ weight.double().to(device).t() + bias.double().to(device)
    ).to(torch.bfloat16)

    with torch.inference_mode():
        actual = linear(x)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == oracle.shape
    assert torch.equal(actual, oracle)
    assert "weight" not in linear.state_dict()


@pytest.mark.parametrize("device", _GPU_OR_CPU)
def test_linear_fallback_preserves_bf16_range_under_autocast(device, monkeypatch):
    """Autocast must not narrow the wide fallback GEMM back to FP16.

    CPU autocast narrows an FP32 matmul exactly like the CUDA path, so the
    guard is required on every device that the wide branch can run on.
    """
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    weight = torch.zeros((256, 256), dtype=torch.bfloat16)
    weight[0, 0] = 65536.0
    linear = NF12ADLinear(bits=12, in_features=256, out_features=256, bias=False)
    linear.load_quant_result(quantize_nf12_ad_weight(weight))
    linear = linear.to(device)
    monkeypatch.setattr(linear, "_native_forward", lambda x, bias: None)

    x = torch.zeros((1, 256), dtype=torch.float16, device=device)
    x[0, 0] = 2.0**-16
    oracle = (x.double() @ weight.double().to(device).t()).to(torch.float16)
    with torch.autocast(device_type=device, dtype=torch.float16), torch.inference_mode():
        actual = linear(x)

    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, oracle)
    assert "weight" not in linear.state_dict()


@pytest.mark.parametrize("device", _GPU_OR_CPU)
@pytest.mark.parametrize(
    "activation_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
def test_linear_fallback_preserves_bf16_range_for_lossless_activation_under_autocast(
    device, activation_dtype, monkeypatch
):
    """A lossless activation is not enough when the autocast context narrows.

    `bfloat16` and `float32` can hold every BF16 weight, but an FP16 autocast
    context rewrites the GEMM operands to FP16 before the product. The narrow
    branch must therefore widen when the autocast dtype cannot hold the weight
    range, not only when the activation dtype cannot.
    """
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    weight = torch.zeros((256, 256), dtype=torch.bfloat16)
    weight[0, 0] = 65536.0
    linear = NF12ADLinear(bits=12, in_features=256, out_features=256, bias=False)
    linear.load_quant_result(quantize_nf12_ad_weight(weight))
    linear = linear.to(device)
    monkeypatch.setattr(linear, "_native_forward", lambda x, bias: None)

    x = torch.zeros((1, 256), dtype=activation_dtype, device=device)
    x[0, 0] = 2.0**-16
    oracle = (x.double() @ weight.double().to(device).t()).to(activation_dtype)
    assert torch.isfinite(oracle).all()
    assert oracle.flatten()[0].item() == 1.0

    with torch.autocast(device_type=device, dtype=torch.float16), torch.inference_mode():
        actual = linear(x)

    assert actual.dtype == activation_dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, oracle)
    assert "weight" not in linear.state_dict()


@pytest.mark.parametrize("device", _GPU_OR_CPU)
def test_linear_fallback_keeps_float64_activation_precision(device, monkeypatch):
    """A float64 activation must not have its product rounded down to float32.

    `float64` is outside `SUPPORTS_DTYPES`, but a caller that already holds
    float64 activations expects the fallback to stay in float64. The lossless
    predicate therefore admits float64 so the decoded weight is not narrowed to
    float32 before the product.
    """
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    torch.manual_seed(0)
    weight = (torch.randn((256, 256), dtype=torch.float32) * 1e-3).to(torch.bfloat16)
    linear = NF12ADLinear(bits=12, in_features=256, out_features=256, bias=False)
    linear.load_quant_result(quantize_nf12_ad_weight(weight))
    linear = linear.to(device)
    monkeypatch.setattr(linear, "_native_forward", lambda x, bias: None)

    x = torch.randn((4, 256), dtype=torch.float64, device=device) * 1e-3
    oracle = x @ weight.double().to(device).t()

    with torch.inference_mode():
        actual = linear(x)

    assert actual.dtype == torch.float64
    if device == "cpu":
        # CPU deliberately decodes and multiplies in FP32 for every activation
        # dtype, so only the accelerator must stay bitwise exact in float64.
        torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=1e-9)
    else:
        assert torch.equal(actual, oracle)


@pytest.mark.parametrize("device", _GPU_OR_CPU)
def test_linear_fallback_keeps_float64_multiply_wide_for_mixed_bias(device, monkeypatch):
    """A float64 activation stays float64 in the wide GEMM, including the bias.

    The wide branch used to cast ``x`` with ``.float()`` before the product, so
    a float64 activation silently re-entered the GEMM as float32 and rounded
    every partial sum. The bias is also sent to the scratch dtype rather than
    ``.float()``, so a float64 bias is not narrowed by the epilogue. Bitwise
    equality with the float64 oracle is the contract on an accelerator; CPU
    keeps its documented FP32 decode.
    """
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    torch.manual_seed(1)
    weight = (torch.randn((256, 256), dtype=torch.float32) * 1e-3).to(torch.bfloat16)
    bias = torch.randn(256, dtype=torch.bfloat16) * 1e-3
    linear = NF12ADLinear(bits=12, in_features=256, out_features=256, bias=True)
    linear.load_quant_result(quantize_nf12_ad_weight(weight), bias=bias)
    linear = linear.to(device)
    monkeypatch.setattr(linear, "_native_forward", lambda x, bias: None)

    x = torch.randn((4, 256), dtype=torch.float64, device=device) * 1e-3
    oracle = x @ weight.double().to(device).t() + bias.double().to(device)

    with torch.inference_mode():
        actual = linear(x)

    assert actual.dtype == torch.float64
    # Bitwise for the accelerator; CPU keeps its documented FP32 decode.
    if device == "cpu":
        torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=1e-9)
    else:
        assert torch.equal(actual, oracle)
        assert "weight" not in linear.state_dict()


def test_forward_on_meta_applies_the_adapter_exactly_once():
    """`empty_linear_output` applies the adapter itself; do not apply it again.

    The meta and zero-row branches are separate; the zero-row case runs on a
    real device so it cannot be satisfied by the meta branch alone.
    """

    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    calls = []

    class CountingAdapter:
        def apply(self, x, out):
            calls.append((out.shape, out.device.type))
            return out

    with torch.device("meta"):
        meta_linear = NF12ADLinear(bits=12, in_features=64, out_features=16, bias=False)
    meta_linear.adapter = CountingAdapter()

    meta_linear(torch.zeros((2, 3, 64), dtype=torch.bfloat16, device="meta"))
    assert len(calls) == 1, calls

    calls.clear()
    linear = NF12ADLinear(bits=12, in_features=64, out_features=16, bias=False)
    linear.load_quant_result(quantize_nf12_ad_weight(torch.zeros((16, 64), dtype=torch.bfloat16)))
    linear.adapter = CountingAdapter()

    out = linear(torch.zeros((0, 64), dtype=torch.bfloat16))
    assert out.shape == (0, 16)
    assert len(calls) == 1, calls


def test_forward_on_meta_with_online_rotation_returns_metadata_output():
    """A rotation-enabled meta shell must answer forward from metadata.

    The online Hadamard arm runs before the fallback, and its native operator
    has no meta kernel. Answering the meta case first keeps a rotation-enabled
    lazy-load shell from raising where a plain one already worked.
    """

    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear

    with torch.device("meta"):
        linear = NF12ADLinear(bits=12, in_features=64, out_features=16, bias=False)
    linear.online_full_had = True
    x = torch.zeros((2, 3, 64), dtype=torch.bfloat16, device="meta")

    out = linear(x)

    assert out.shape == (2, 3, 16)
    assert out.device.type == "meta"
    assert out.dtype == torch.bfloat16
    assert "weight" not in linear.state_dict()


@pytest.mark.parametrize(
    ("activation_dtype", "bias_dtype"),
    [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float32)],
)
def test_linear_fallback_adds_mixed_dtype_bias_before_rounding(
    activation_dtype, bias_dtype
):
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    weight = torch.zeros((16, 64), dtype=torch.bfloat16)
    weight[0, 0] = -99840.0
    bias = torch.zeros(16, dtype=bias_dtype)
    bias[0] = 99840.0
    linear = NF12ADLinear(bits=12, in_features=64, out_features=16, bias=True)
    linear.load_quant_result(quantize_nf12_ad_weight(weight), bias=bias)
    x = torch.zeros((1, 64), dtype=activation_dtype)
    x[0, 0] = 1.0

    expected = (x.float() @ weight.float().t() + bias.float()).to(x.dtype)
    actual = linear(x)

    assert torch.equal(actual, expected)
    assert actual[0, 0].item() == 0.0
    assert linear.bias.dtype == bias_dtype


def test_hip_kernel_is_bit_exact_when_available():
    """RDNA4 HIP decode must reproduce every BF16 bit the Torch codec encodes."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires an AMD ROCm device")
    from gptqmodel.utils import nf12_ad_hip

    if not nf12_ad_hip.prewarm_nf12_ad_hip_extension():
        pytest.skip(f"NF12-AD HIP kernel unavailable: {nf12_ad_hip.last_error()}")

    for shape in [(16, 64), (256, 256), (1024, 512)]:
        weight = _adversarial_bits(*shape, seed=5)
        packed, escapes, exp_base = nf12_codec.encode(weight)
        decoded = nf12_ad_hip.decode(
            packed.to("cuda"), escapes.to("cuda"), int(exp_base)
        ).cpu()
        assert torch.equal(decoded.view(torch.int16), weight.contiguous().view(torch.int16))


def test_hip_wmma_uses_weight_device_when_current_device_differs():
    """WMMA dispatch and stream must follow the packed weight on multi-GPU HIP."""
    if torch.version.hip is None or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two AMD ROCm devices")

    from gptqmodel.utils import nf12_ad_hip

    wmma_devices = [
        index
        for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").split(":", 1)[0]
        in {"gfx1200", "gfx1201"}
    ]
    if len(wmma_devices) < 2:
        pytest.skip("requires two RDNA4 WMMA devices")
    target, other = wmma_devices[:2]
    with torch.cuda.device(target):
        if not nf12_ad_hip.prewarm_nf12_ad_hip_extension():
            pytest.skip(f"NF12-AD HIP kernel unavailable: {nf12_ad_hip.last_error()}")

    torch.manual_seed(17)
    weight = (torch.randn(256, 256) * 0.02).to(torch.bfloat16)
    x = (torch.randn(16, 256) * 0.05).to(torch.bfloat16)
    bias = (torch.randn(256) * 0.1).to(torch.bfloat16)
    packed, escapes, exp_base = nf12_codec.encode(weight)
    expected = (x.float() @ weight.float().t() + bias.float()).to(torch.bfloat16)
    target_device = torch.device(f"cuda:{target}")
    x = x.to(target_device)
    packed = packed.to(target_device)
    escapes = escapes.to(target_device)
    bias = bias.to(target_device)
    torch.cuda.synchronize(target_device)
    with torch.cuda.device(other):
        actual = nf12_ad_hip.linear(x, packed, escapes, exp_base, bias)
        assert torch.cuda.current_device() == other

    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-3, atol=2e-3)


def test_hip_split_k_calls_on_distinct_streams_keep_their_own_workspace():
    """In-flight split-K calls must not overwrite each other's FP32 partials."""
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("requires an AMD ROCm device")

    from gptqmodel.utils import nf12_ad_hip

    device = torch.device("cuda", torch.cuda.current_device())
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":", 1)[0]
    if arch not in {"gfx1200", "gfx1201"}:
        pytest.skip("requires an RDNA4 WMMA device")
    if not nf12_ad_hip.prewarm_nf12_ad_hip_extension():
        pytest.skip(f"NF12-AD HIP kernel unavailable: {nf12_ad_hip.last_error()}")

    torch.manual_seed(23)
    # N=512 launches four CTAs, so the current RDNA4 planner uses eight K
    # splits and allocates the partial planes exercised by this test.
    weight = (torch.randn(512, 512) * 0.02).to(torch.bfloat16)
    x_a = (torch.randn(32, 512) * 0.05).to(torch.bfloat16)
    x_b = (torch.randn(32, 512) * 0.05).to(torch.bfloat16)
    packed, escapes, exp_base = nf12_codec.encode(weight)
    expected_a = (x_a.float() @ weight.float().t()).to(torch.bfloat16)
    expected_b = (x_b.float() @ weight.float().t()).to(torch.bfloat16)
    packed = packed.to(device)
    escapes = escapes.to(device)
    x_a = x_a.to(device)
    x_b = x_b.to(device)
    torch.cuda.synchronize(device)
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)

    for _ in range(4):
        with torch.cuda.stream(stream_a):
            actual_a = nf12_ad_hip.linear(x_a, packed, escapes, exp_base)
        with torch.cuda.stream(stream_b):
            actual_b = nf12_ad_hip.linear(x_b, packed, escapes, exp_base)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual_a.cpu().float(), expected_a.float(), rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(actual_b.cpu().float(), expected_b.float(), rtol=2e-3, atol=2e-3)


def _hip_runtime_or_skip():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires an AMD ROCm device")
    from gptqmodel.utils import nf12_ad_hip

    if not nf12_ad_hip.prewarm_nf12_ad_hip_extension():
        pytest.skip(f"NF12-AD HIP kernel unavailable: {nf12_ad_hip.last_error()}")
    return nf12_ad_hip


def test_hip_linear_preserves_leading_dimensions():
    """A 3-D activation keeps its leading dims; only the last dim becomes N."""
    hip = _hip_runtime_or_skip()
    torch.manual_seed(7)
    n, k = 256, 256
    weight = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = nf12_codec.encode(weight.cpu())
    packed, escapes = packed.cuda(), escapes.cuda()
    for shape in [(1, 8, k), (2, 8, k), (2, 32, k)]:
        x = (torch.randn(shape, device="cuda") * 0.05).to(torch.bfloat16)
        got = hip.linear(x, packed, escapes, int(exp_base))
        assert tuple(got.shape) == (*shape[:-1], n), shape
        expected = torch.nn.functional.linear(x, weight)
        torch.testing.assert_close(got.float(), expected.float(), rtol=2e-3, atol=2e-3)


def test_hip_linear_accepts_padded_activation_rows():
    """A sliced activation is a legal torch layout; the op normalizes it.

    The HIP kernels index contiguous rows, so forwarding a wider row stride
    silently read the wrong rows (3562/4096 elements out of tolerance at M=16).
    """
    hip = _hip_runtime_or_skip()
    torch.manual_seed(8)
    n, k = 256, 256
    weight = (torch.randn(n, k) * 0.02).to(torch.bfloat16).cuda()
    packed, escapes, exp_base = nf12_codec.encode(weight.cpu())
    packed, escapes = packed.cuda(), escapes.cuda()
    storage = (torch.randn(64, k + 8, device="cuda") * 0.05).to(torch.bfloat16)
    for m in (2, 16, 64):
        x = storage[:m, :k]
        assert x.stride(0) == k + 8 and not x.is_contiguous()
        got = hip.linear(x, packed, escapes, int(exp_base))
        expected = torch.nn.functional.linear(x, weight)
        torch.testing.assert_close(got.float(), expected.float(), rtol=2e-3, atol=2e-3)


def test_hip_linear_preserves_finite_overflow_cancellation():
    """Finite BF16 operands may overflow FP32 before they cancel.

    ``+bf16_max`` and ``-bf16_max`` multiplied by 2 are finite in BF16 but
    overflow FP32. The packed dot keeps a wider intermediate and returns 0;
    a naive two-FMA gfx1250 fallback returned ``-inf``. The fallback must
    detect the non-finite intermediate and recompute in FP64.
    """
    hip = _hip_runtime_or_skip()
    weight = torch.zeros(16, 64, dtype=torch.bfloat16)
    weight[:, 0] = torch.finfo(torch.bfloat16).max
    weight[:, 1] = -torch.finfo(torch.bfloat16).max
    x = torch.ones(1, 64, dtype=torch.bfloat16)
    x[:, :2] = 2

    packed, escapes, exp_base = nf12_codec.encode(weight)
    actual = hip.linear(x.cuda(), packed.cuda(), escapes.cuda(), int(exp_base))
    expected = x.double() @ weight.double().T

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-3, atol=2e-3)


def test_nf12_module_preserves_leading_dimensions_on_hip():
    """The quant-linear module must not flatten a multi-dim activation."""
    _hip_runtime_or_skip()
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.quantization.nf12_ad import quantize_nf12_ad_weight

    torch.manual_seed(9)
    n, k = 256, 256
    weight = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    layer = NF12ADLinear(bits=12, in_features=k, out_features=n)
    layer.load_quant_result(quantize_nf12_ad_weight(weight))
    layer = layer.cuda()
    for shape in [(1, 8, k), (2, 32, k)]:
        x = (torch.randn(shape, device="cuda") * 0.05).to(torch.bfloat16)
        got = layer(x)
        assert tuple(got.shape) == (*shape[:-1], n), shape
