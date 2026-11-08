# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NF12-AD codec and inference-oracle tests.

The codec is a lossless BF16 encoding, so decode must reproduce every input
bit. The CUDA kernels share the same format; these tests validate the Torch
oracle used for quantization and CPU inference.
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


def test_module_saves_qweight_and_retains_no_dense_cache():
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

    # A quantized checkpoint must expose its packed weight as `qweight`, not as
    # a dense `weight` parameter that a loader could mistake for BF16.
    keys = set(module.state_dict())
    assert "qweight" in keys
    assert module.qweight.dtype == torch.uint8
    assert module.qweight.shape == (32, 192)
    assert "weight" not in module._parameters
    assert int(module.nf12_exp_base.item()) == 54

    x = torch.randn(4, 128, dtype=torch.bfloat16)
    expected = linear(x).float()
    for _ in range(3):
        actual = module(x)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual.float(), expected, rtol=2e-3, atol=2e-3)

    # Repeated inference must not leave a decoded dense weight behind.
    assert set(module._buffers) == {"qweight", "nf12_escapes", "nf12_exp_base", "bias"}
    decoded = module.dequantize_weight()
    assert torch.equal(
        decoded.view(torch.int16),
        linear.weight.detach().to(torch.bfloat16).view(torch.int16),
    )


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
            "blk.qweight": torch.zeros((16, 96), dtype=torch.uint8),
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
    assert {"qweight", "nf12_escapes", "nf12_exp_base"} <= state


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
