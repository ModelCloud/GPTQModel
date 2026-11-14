# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant kernel tests adapted from the ParoQuant paper and public project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""Kernel-focused tests for ParoQuant runtime behavior and backend parity."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import paroquant as paroquant_module
from gptqmodel.nn_modules.qlinear._numerics import linear_wide_bias
from gptqmodel.nn_modules.qlinear.paroquant import (
    ParoLinear,
    awq_gemm_bf16_supported,
    awq_gemm_shape_supported,
)
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.awq.utils.packing_utils import (
    dequantize_gemm,
    reverse_awq_order,
    unpack_awq,
)
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import get_kernel_for_backend, select_quant_linear
from gptqmodel.utils.paroquant import (
    apply_paroquant_rotation_reference,
    apply_paroquant_rotation_vectorized,
    build_identity_rotation_buffers,
)
from gptqmodel.utils.rocm import IS_ROCM


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unpacked integer weights into the AWQ bit layout used by the kernels."""
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    assert unpacked.shape[1] % pack_factor == 0
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


def _make_packed_buffers(bits: int, in_features: int, out_features: int, group_size: int):
    """Build synthetic packed AWQ tensors for ParoQuant runtime tests."""
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 2.0) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)

    return (
        _pack_awq_tensor(int_weight, bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
        bias,
    )


def _upstream_transformers_contract_reference(
    x: torch.Tensor,
    *,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    channel_scales: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
    rotated: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference for upstream RotateQuantizedLinear.forward().

    Upstream ParoQuant applies per-projection rotation to the input and then
    feeds the rotated activations into the AWQ GEMM path. We reproduce that
    contract with dense dequantization here to assess kernel accuracy without
    importing or copying upstream implementation code.
    """

    if rotated is None:
        rotated = apply_paroquant_rotation_reference(
            x,
            pairs,
            theta,
            scales=channel_scales,
            group_size=group_size,
        )
    dense_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        # Reconstruct in FP32 so the oracle describes the exact grid value the
        # kernel multiplies, not a same-dtype rounding of it.
        scales=scales.float(),
        bits=bits,
        group_size=group_size,
    ).to(device=x.device)
    # ``F.linear`` accumulates in FP32 and applies the bias before the single
    # visible cast; a same-dtype matmul plus add rounds twice and no longer
    # describes the contract this reference is meant to model.
    out = torch.matmul(rotated.reshape(-1, x.shape[-1]).float(), dense_weight.float())
    out = out.reshape(*x.shape[:-1], out_features)
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def _fp32_awq_oracle(
    rotated: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Independent FP32 dequantization oracle for the AWQ/ParoQuant layout."""
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)
    mask = (1 << bits) - 1
    iweight = torch.bitwise_and(iweight, mask).to(torch.float32)
    izeros = torch.bitwise_and(izeros, mask).to(torch.float32)
    group_index = torch.arange(iweight.shape[0]) // group_size
    weight = (iweight - izeros[group_index]) * scales.float()[group_index]
    out = rotated.float() @ weight
    if bias is not None:
        out = out + bias.float()
    return out


def test_paroquant_identity_forward_matches_awq_torch(monkeypatch):
    """Guard that identity ParoQuant is behaviorally equivalent to plain AWQ."""
    bits = 4
    in_features = 128
    out_features = 64
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    awq_module = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    )
    awq_module.qweight.copy_(qweight)
    awq_module.qzeros.copy_(qzeros)
    awq_module.scales.copy_(scales)
    awq_module.bias.copy_(bias)
    awq_module.post_init()
    awq_module.eval()

    paro_module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    )
    paro_module.qweight.copy_(qweight)
    paro_module.qzeros.copy_(qzeros)
    paro_module.scales.copy_(scales)
    paro_module.bias.copy_(bias)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        dtype=torch.float16,
    )
    paro_module.pairs.copy_(pairs)
    paro_module.theta.copy_(theta)
    paro_module.channel_scales.copy_(channel_scales)
    paro_module.post_init()
    paro_module.eval()
    monkeypatch.setattr(paro_module, "_forward_cpu_aten", lambda _x: None)

    x = torch.randn(4, in_features, dtype=torch.float16)
    torch.testing.assert_close(paro_module(x), awq_module(x), atol=5e-3, rtol=5e-3)


def test_paroquant_fp32_dense_fallback_preserves_checkpoint_scale(monkeypatch):
    """The FP32 fallback must not overflow a valid FP32 scale through FP16."""
    module = ParoLinear(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=8,
        bias=False,
        register_buffers=True,
        krot=1,
    )
    module.qweight.copy_(_pack_awq_tensor(torch.ones((16, 8), dtype=torch.int32), 4))
    module.qzeros.zero_()
    module.scales = torch.full((1, 8), 65536.0, dtype=torch.float32)
    module.post_init()
    module.eval()
    monkeypatch.setattr(module, "_forward_cpu_aten", lambda _x: None)

    actual = module(torch.ones((1, 16), dtype=torch.float32))

    assert actual.dtype is torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, torch.full((1, 8), 16 * 65536.0, dtype=torch.float32), rtol=0, atol=0
    )
    assert module.scales.dtype is torch.float32
    retained = list(vars(module).values()) + list(module.buffers()) + list(module.parameters())
    assert not any(
        isinstance(value, torch.Tensor) and value.shape == (16, 8)
        for value in retained
    )


def _cancelling_bias_module(in_features: int, out_features: int, group_size: int) -> ParoLinear:
    """A layer whose finite FP16 result needs the FP32 reduction and bias."""

    module = ParoLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=1,
    ).eval()
    # Packed code 1 in every position: the reconstructed grid is 1 * 8 = 8.
    module.qweight.fill_(0x11111111)
    module.qzeros.zero_()
    module.scales.fill_(8)
    module.bias.fill_(-1024)
    module.post_init()
    return module


def test_paroquant_dense_fallback_accumulates_reduction_and_bias_in_fp32(monkeypatch):
    """Regression for #3866: the FP16 dot overflowed before the bias cancelled it."""

    in_features, out_features, group_size = 16, 8, 16
    module = _cancelling_bias_module(in_features, out_features, group_size)
    # group_size 16 already skips the ATen path; pin it so a future eligibility
    # change cannot hide the dense fallback under test.
    monkeypatch.setattr(module, "_forward_cpu_aten", lambda _x: None)

    x = torch.full((1, in_features), 516, dtype=torch.float16)
    with torch.no_grad():
        actual = module(x)

    expected = _fp32_awq_oracle(
        x.reshape(-1, in_features),
        module.qweight,
        module.qzeros,
        module.scales,
        module.bias,
        bits=4,
        group_size=group_size,
    ).to(torch.float16)

    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native CUDA ParoQuant kernel required (unavailable on ROCm)",
)
def test_paroquant_cuda_kernel_adds_bias_after_the_fp32_reduction(monkeypatch):
    """The native AWQ epilogue must not round the sum before the bias cancels it."""

    in_features, out_features, group_size = 32, 64, 32
    module = _cancelling_bias_module(in_features, out_features, group_size).cuda()
    x = torch.full((1, in_features), 258, dtype=torch.float16, device="cuda")

    # The dense fallback has its own FP32 epilogue, so fail loudly if the
    # native kernel silently delegates to it instead of being exercised.
    monkeypatch.setattr(
        module,
        "_forward_dense",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Expected the ParoQuant CUDA kernel path, got the dense fallback.")
        ),
    )

    with torch.no_grad():
        actual = module(x)

    expected = _fp32_awq_oracle(
        x.reshape(-1, in_features),
        module.qweight,
        module.qzeros,
        module.scales,
        module.bias,
        bits=4,
        group_size=group_size,
    ).to(torch.float16)

    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("bits", "in_features", "out_features", "groups", "expected"),
    [
        # The issue's advertised configuration: group_size 16 is not a native group size.
        (4, 16, 8, 1, False),
        (4, 128, 64, 8, False),  # group_size 16
        (4, 32, 64, 1, True),  # group_size 32, 64-column output tile
        (4, 64, 64, 2, True),
        (4, 128, 128, 4, True),
        (4, 128, 128, 1, True),  # group_size -1 packs one scale row per channel
        (4, 128, 64, 1, False),  # output width not divisible by group_size 128
        (4, 256, 128, 2, True),
        (4, 64, 32, 2, False),  # output width below the 64-column tile
        (4, 64, 96, 2, False),  # 96 is not a multiple of 64
        (8, 64, 64, 2, False),  # the native kernel decodes uint4b only
        (4, 64, 64, 0, False),  # no scale rows
        (4, 96, 64, 3, True),  # group_size 32
        (4, 100, 64, 3, False),  # in_features not divisible by groups
    ],
)
def test_awq_gemm_shape_supported_matches_native_validator(
    bits, in_features, out_features, groups, expected
):
    """Pin the native kernel's shape contract without needing a GPU.

    ``validate_awq_gemm_args`` rejects a group size that is not a multiple of
    32, an output width that is not a multiple of the 64-column CTA tile, and an
    output width that is not divisible by the group size. ParoQuant advertises
    configurations outside that set, so the predicate has to be exact.
    """

    assert (
        awq_gemm_shape_supported(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            groups=groups,
        )
        is expected
    )


def test_cuda_awq_kernel_gate_is_false_off_cuda():
    """A CPU module must never be routed into the CUDA extension."""
    module = _cancelling_bias_module(32, 64, 32)

    assert module.qweight.device.type == "cpu"
    assert module._cuda_awq_kernel_supported(torch.float16) is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native CUDA ParoQuant kernel required (unavailable on ROCm)",
)
def test_paroquant_cuda_falls_back_for_native_unsupported_shape(monkeypatch):
    """Regression for #3919: an advertised group_size 16 / narrow output must not raise."""

    in_features, out_features, group_size = 16, 8, 16
    module = _cancelling_bias_module(in_features, out_features, group_size).cuda()

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("native AWQ GEMM must not be attempted for an unsupported shape")

    monkeypatch.setattr(paroquant_module, "_awq_cuda_gemm_forward", _unexpected)
    monkeypatch.setattr(paroquant_module, "_awq_cuda_gemm_forward_fp32_reduce", _unexpected)

    x = torch.full((1, in_features), 516, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        actual = module(x)

    expected = _fp32_awq_oracle(
        x.reshape(-1, in_features),
        module.qweight,
        module.qzeros,
        module.scales,
        module.bias,
        bits=4,
        group_size=group_size,
    ).to(torch.float16)

    assert module._cuda_awq_kernel_supported(torch.float16) is False
    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    assert actual[0, 0].item() == 65024.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native CUDA ParoQuant kernel required (unavailable on ROCm)",
)
def test_paroquant_cuda_fallback_promotes_bf16_activations(monkeypatch):
    """The dense fallback pairs BF16 activations with its reconstructed FP32 grid."""

    in_features, out_features, group_size = 16, 8, 16
    module = _cancelling_bias_module(in_features, out_features, group_size).cuda()
    monkeypatch.setattr(module, "_forward_cuda_awq_kernel", lambda _x: None)

    x = torch.full((1, in_features), 516, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        actual = module(x)

    expected = _fp32_awq_oracle(
        x.reshape(-1, in_features),
        module.qweight,
        module.qzeros,
        module.scales,
        module.bias,
        bits=4,
        group_size=group_size,
    ).to(torch.bfloat16)

    assert actual.dtype is torch.bfloat16
    assert torch.isfinite(actual).all()
    assert actual[0, 0].item() == 65024.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native CUDA ParoQuant kernel required (unavailable on ROCm)",
)
def test_paroquant_cuda_gate_admits_the_native_supported_shape():
    """The qualifying shape stays on the native path, including the BF16 rule."""

    module = _cancelling_bias_module(32, 64, 32).cuda()

    assert module._cuda_awq_kernel_supported(torch.float16) is True
    assert module._cuda_awq_kernel_supported(torch.bfloat16) is awq_gemm_bf16_supported(
        module.qweight.device
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native CUDA ParoQuant kernel required (unavailable on ROCm)",
)
def test_paroquant_cuda_fallback_retains_only_packed_storage():
    """Repeated fallback forwards must not install a dense weight cache."""

    in_features, out_features, group_size = 16, 8, 16
    module = _cancelling_bias_module(in_features, out_features, group_size).cuda()
    x = torch.full((1, in_features), 516, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        for _ in range(2):
            module(x)

    retained = list(vars(module).values()) + list(module.buffers()) + list(module.parameters())
    assert not any(
        isinstance(value, torch.Tensor) and value.shape == (in_features, out_features)
        for value in retained
    )
    assert module.qweight.dtype is torch.int32
    assert module.qweight.numel() == in_features * (out_features // 8)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_ROCM,
    reason="ROCm device required",
)
def test_paroquant_rocm_skips_the_native_cuda_kernel():
    """A HIP build must use the portable path, not attempt the CUDA-only extension."""

    in_features, out_features, group_size = 32, 64, 32
    module = _cancelling_bias_module(in_features, out_features, group_size).cuda()

    assert module._cuda_awq_kernel_supported(torch.float16) is False

    x = torch.full((1, in_features), 258, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        actual = module(x)

    expected = _fp32_awq_oracle(
        x.reshape(-1, in_features),
        module.qweight,
        module.qzeros,
        module.scales,
        module.bias,
        bits=4,
        group_size=group_size,
    ).to(torch.float16)

    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA operand promotion")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_wide_bias_promotes_mismatched_cuda_dtypes(dtype):
    """Regression for #3919: FP32 reconstructed weights beside FP16/BF16 activations."""

    x = torch.full((1, 16), 516, dtype=dtype, device="cuda")
    weight = torch.full((8, 16), 8.0, dtype=torch.float32, device="cuda")
    bias = torch.full((8,), -1024.0, dtype=torch.float32, device="cuda")

    actual = linear_wide_bias(x, weight, bias)
    expected = (x.double() @ weight.double().t() + bias.double()).to(dtype)

    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    assert actual[0, 0].item() == 65024.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_paroquant_forward_matches_explicit_rotated_reference(monkeypatch):
    """Guard the dense reference contract for non-identity ParoQuant rotations."""
    bits = 4
    in_features = 128
    out_features = 64
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=1,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=1,
        dtype=torch.float16,
    )
    theta.fill_(0.2)
    channel_scales.mul_(0.75)
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()
    monkeypatch.setattr(module, "_forward_cpu_aten", lambda _x: None)

    x = torch.randn(3, in_features, dtype=torch.float16)
    rotated = apply_paroquant_rotation_reference(
        x,
        module.pairs,
        module.theta,
        scales=module.channel_scales,
        group_size=group_size,
    )
    dequant_weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales.float(),
        bits=bits,
        group_size=group_size,
    )
    expected = (rotated.float() @ dequant_weight.float() + module.bias.float()).to(x.dtype)

    torch.testing.assert_close(module(x), expected, atol=5e-3, rtol=5e-3)


def test_awq_torch_forward_dequantizes_per_call_without_dense_cache(monkeypatch):
    from gptqmodel.nn_modules.qlinear import torch_awq as torch_awq_module

    bits = 4
    in_features = 128
    out_features = 64
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(
        bits,
        in_features,
        out_features,
        group_size,
    )

    module = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    module.post_init()
    module.eval()

    calls = 0
    dequantize_gemm = torch_awq_module.dequantize_gemm

    def counted_dequantize_gemm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return dequantize_gemm(*args, **kwargs)

    monkeypatch.setattr(torch_awq_module, "dequantize_gemm", counted_dequantize_gemm)

    x = torch.randn(3, in_features, dtype=torch.float16)
    with torch.inference_mode():
        first = module(x)
        second = module(x)

    assert calls == 2
    assert not hasattr(module, "_dense_weight_cache")
    torch.testing.assert_close(second, first, rtol=0, atol=0)


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paroquant_rotation_vectorized_matches_reference_exactly(group_size, dtype):
    torch.manual_seed(17)
    in_features = 512
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=4,
        dtype=dtype,
    )
    theta.normal_()
    channel_scales.uniform_(0.5, 1.5)
    x = torch.randn(7, 3, in_features, dtype=dtype)

    expected = apply_paroquant_rotation_reference(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=group_size,
    )
    actual = apply_paroquant_rotation_vectorized(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=group_size,
    )

    assert actual.dtype == x.dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paroquant_cpu_optimized_forward_matches_fp32_oracle(group_size, dtype):
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import _has_local_int4pack_cpu_ops
    from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear

    if dtype == torch.float16 and not _has_local_int4pack_cpu_ops():
        pytest.skip("ATen CPU int4pack ops are unavailable")
    if dtype == torch.bfloat16 and not TorchCpuAvx512Linear.cached_validate_once()[0]:
        pytest.skip("AVX512-BF16 CPU kernel is unavailable")

    torch.manual_seed(23)
    bits = 4
    in_features = 512
    out_features = 256
    qweight, qzeros, scales, bias = _make_packed_buffers(
        bits,
        in_features,
        out_features,
        group_size,
    )
    scales.mul_(0.01)

    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=2,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=2,
        dtype=dtype,
    )
    theta.normal_()
    channel_scales.uniform_(0.75, 1.25)
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()

    x = torch.randn(7, in_features, dtype=dtype)
    rotated = apply_paroquant_rotation_reference(
        x,
        module.pairs,
        module.theta,
        scales=module.channel_scales,
        group_size=group_size,
    )
    expected = _fp32_awq_oracle(
        rotated,
        module.qweight,
        module.qzeros,
        module.scales,
        module.bias,
        bits=bits,
        group_size=group_size,
    )

    with torch.inference_mode():
        actual = module(x)

    assert actual.dtype == x.dtype
    # The public output is BF16, so compare the kernel's visible result with
    # the oracle rounded to the same dtype. Keep a separate normalized gate
    # against the unrounded FP32 oracle so BF16 output rounding cannot hide
    # arithmetic drift.
    torch.testing.assert_close(actual, expected.to(dtype=dtype), rtol=2e-3, atol=2e-3)
    normalized_error = (actual.float() - expected).norm() / expected.norm()
    assert normalized_error.item() <= 2e-3
    if dtype == torch.float16:
        assert module._cpu_aten_layer is not None
    else:
        assert module._cpu_packed_key is not None


def test_paroquant_cpu_caches_invalidate_on_rotation_and_weight_mutation():
    from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear

    if not TorchCpuAvx512Linear.cached_validate_once()[0]:
        pytest.skip("AVX512-BF16 CPU kernel is unavailable")

    torch.manual_seed(24)
    bits = 4
    in_features = 512
    out_features = 256
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(
        bits,
        in_features,
        out_features,
        group_size,
    )
    scales.mul_(0.01)
    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=1,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=1,
        dtype=torch.bfloat16,
    )
    theta.normal_()
    channel_scales.uniform_(0.75, 1.25)
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()

    x = torch.randn(7, in_features, dtype=torch.bfloat16)
    with torch.inference_mode():
        module(x)
    rotation_key = module._rotation_runtime_key
    packed_key = module._cpu_packed_key
    assert rotation_key is not None
    assert packed_key is not None

    with torch.no_grad():
        module.theta.add_(0.01)
    with torch.inference_mode():
        module(x)
    assert module._rotation_runtime_key != rotation_key

    with torch.no_grad():
        module.qweight.add_(1)
    with torch.inference_mode():
        module(x)
    assert module._cpu_packed_key != packed_key

    module.to(dtype=torch.float16)
    assert module._rotation_runtime_key is None
    assert module._cpu_packed_key is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native CUDA ParoQuant kernel required (unavailable on ROCm)",
)
def test_paroquant_cuda_matches_upstream_transformers_contract():
    """Compare the internal CUDA kernel path to an upstream-style contract.

    The official implementation rotates activations and then runs an AWQ-style
    packed matmul. This test reproduces that contract without importing upstream
    code and checks our fused CUDA path stays within a bounded numerical error.
    """
    bits = 4
    in_features = 128
    out_features = 128
    group_size = 128
    torch.manual_seed(0)

    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.full((groups, out_features), 2 ** (bits - 1), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 0.75) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)
    qweight = _pack_awq_tensor(int_weight, bits)
    qzeros = _pack_awq_tensor(zero_points, bits)

    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda()

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        device="cuda",
        dtype=torch.float16,
    )
    theta.uniform_(-0.25, 0.25)
    channel_scales.uniform_(0.7, 1.3)

    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.cuda())
    module.bias.copy_(bias.cuda())
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()

    x = torch.randn(3, 7, in_features, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        # Isolate the GEMM epilogue: the fused CUDA rotation has its own
        # exact-match contract, and using the same rotated input here keeps
        # this comparison about the kernel, not about rotation arithmetic.
        rotated = module._rotate_inputs(x.reshape(-1, in_features)).reshape(
            *x.shape[:-1], in_features
        )
        expected = _upstream_transformers_contract_reference(
            x,
            qweight=module.qweight,
            qzeros=module.qzeros,
            scales=module.scales,
            bias=module.bias,
            pairs=module.pairs,
            theta=module.theta,
            channel_scales=module.channel_scales,
            bits=bits,
            group_size=group_size,
            out_features=out_features,
            rotated=rotated,
        )

        original_forward_dense = module._forward_dense
        try:
            module._forward_dense = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Expected ParoQuant CUDA kernel path, but dense fallback was used.")
            )
            actual = module(x)
        finally:
            module._forward_dense = original_forward_dense

    diff = (actual - expected).abs().float()
    assert diff.max().item() <= 0.25
    assert diff.mean().item() <= 0.03


def test_paroquant_backend_selection():
    """Guard user-facing backend selection for the default CUDA runtime."""
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=None,
        backend=BACKEND.PARO,
        format=FORMAT.PAROQUANT,
        quant_method=METHOD.PARO,
        pack_dtype=torch.int32,
    )
    assert qlinear_cls is ParoLinear


def test_paroquant_triton_backend_mapping():
    """Guard registry lookup for the Triton ParoQuant runtime class."""
    assert (
        get_kernel_for_backend(BACKEND.PAROQUANT_TRITON, METHOD.PARO, FORMAT.PAROQUANT)
        is ParoQuantTritonLinear
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native CUDA ParoQuant baseline required (unavailable on ROCm)",
)
def test_paroquant_triton_matches_existing_cuda_kernel():
    """Guard Triton runtime accuracy against the established CUDA implementation."""
    pytest.importorskip("triton")

    bits = 4
    in_features = 128
    out_features = 128
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    baseline = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda()
    candidate = ParoQuantTritonLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda()

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        device="cuda",
        dtype=torch.float16,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)

    for module in (baseline, candidate):
        module.qweight.copy_(qweight.cuda())
        module.qzeros.copy_(qzeros.cuda())
        module.scales.copy_(scales.cuda())
        module.bias.copy_(bias.cuda())
        module.pairs.copy_(pairs)
        module.theta.copy_(theta)
        module.channel_scales.copy_(channel_scales)
        module.post_init()
        module.eval()

    x = torch.randn(2, 8, in_features, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        rotated = apply_paroquant_rotation_reference(
            x,
            baseline.pairs,
            baseline.theta,
            scales=baseline.channel_scales,
            group_size=group_size,
        )
        dense_weight = dequantize_gemm(
            qweight=baseline.qweight,
            qzeros=baseline.qzeros,
            scales=baseline.scales,
            bits=bits,
            group_size=group_size,
        ).to(dtype=x.dtype)
        dense_reference = torch.matmul(rotated.reshape(-1, in_features), dense_weight).reshape(2, 8, out_features)
        dense_reference = dense_reference + baseline.bias

        baseline_out = baseline(x)
        candidate_out = candidate(x)

    baseline_max_abs = (baseline_out - dense_reference).abs().max().item()
    baseline_mean_abs = (baseline_out - dense_reference).abs().mean().item()
    candidate_max_abs = (candidate_out - dense_reference).abs().max().item()
    candidate_mean_abs = (candidate_out - dense_reference).abs().mean().item()

    assert candidate_max_abs <= baseline_max_abs + 0.1
    assert candidate_mean_abs <= baseline_mean_abs + 0.02


def test_paroquant_output_width_declaration_matches_the_packer():
    """#3939: the declared output divisibility must equal the packer's contract.

    ``pack_awq`` stores eight INT4 codes per int32, so a narrower output width
    cannot be packed at all. Declaring 1 let backend selection accept such a
    layer and fail later inside the packer with an unrelated message.
    """

    assert paroquant_module.PAROQUANT_OUT_FEATURES_ALIGN == 8
    assert ParoLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY == [8]

    shared = {
        "bits": 4,
        "group_size": 16,
        "desc_act": False,
        "sym": True,
        "pack_dtype": torch.int32,
        "dtype": torch.float16,
        "device": torch.device("cpu"),
        "in_features": 16,
    }
    # `validate` is the entry point backend selection uses.
    ok, err = ParoLinear.validate(out_features=9, **shared)
    assert ok is False
    assert isinstance(err, NotImplementedError)
    assert "divisible by [8]" in str(err)

    for width in (8, 16, 64):
        ok, err = ParoLinear.validate(out_features=width, **shared)
        assert ok is True, (width, err)


def test_paroquant_constructor_rejects_an_unpackable_output_width():
    """A directly constructed layer must fail before packing, not inside it."""

    with pytest.raises(ValueError, match="multiple of 8"):
        ParoLinear(
            bits=4,
            group_size=16,
            sym=True,
            desc_act=False,
            in_features=16,
            out_features=9,
            bias=False,
            register_buffers=True,
            krot=1,
        )

    # The smallest packable width is still accepted.
    module = ParoLinear(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=8,
        bias=False,
        register_buffers=True,
        krot=1,
    )
    assert module.out_features == 8


def test_paroquant_pack_and_forward_round_trips_at_the_packer_alignment(monkeypatch):
    """Positive control: the narrowest packable width packs and forwards exactly."""

    in_features, out_features, group_size = 16, 8, 16
    torch.manual_seed(3939)
    source = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    with torch.no_grad():
        source.weight.copy_(torch.randn(out_features, in_features))

    module = ParoLinear(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
        krot=1,
    )
    scales = torch.rand(out_features, 1, dtype=torch.float16) * 0.5 + 0.25
    zeros = torch.zeros(out_features, 1, dtype=torch.float16)

    # `pack` used to be the first place the width constraint surfaced.
    module.pack(source, scales, zeros)
    module.post_init()
    module.eval()
    monkeypatch.setattr(module, "_forward_cpu_aten", lambda _x: None)

    assert module.qweight.numel() == in_features * (out_features // 8)

    x = torch.randn(2, in_features, dtype=torch.float16)
    with torch.no_grad():
        actual = module(x)

    expected = _fp32_awq_oracle(
        x,
        module.qweight,
        module.qzeros,
        module.scales,
        None,
        bits=4,
        group_size=group_size,
    ).to(torch.float16)

    assert actual.shape == (2, out_features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
