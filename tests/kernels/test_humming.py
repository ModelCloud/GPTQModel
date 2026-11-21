# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Registration and numeric tests for the Humming (W<8-bit) sub-kernels.

Humming is an optional external package, so the numeric cases skip unless a
usable CUDA device and ``humming-kernels`` are both present.
"""

from __future__ import annotations

import json

import pytest
import torch

from gptqmodel.nn_modules.qlinear.humming import AwqHummingLinear, HummingLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.humming import (
    HUMMING_AWQ_BITS,
    HUMMING_GPTQ_BITS,
    humming_apply_linear,
    humming_compute_config,
    humming_import_exception,
    humming_supported_device_indices,
)
from tests.accelerator_gates import cuda_only_skip_reason


AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _skip_reason() -> str | None:
    rocm_reason = cuda_only_skip_reason("Humming")
    if rocm_reason is not None:
        return rocm_reason
    if humming_import_exception is not None:
        return f"humming-kernels is not installed: {humming_import_exception}"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if not humming_supported_device_indices():
        return "Humming requires SM75+; no visible device is supported"
    return None


_SKIP_REASON = _skip_reason()
requires_humming = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


def _device() -> torch.device:
    return torch.device("cuda", humming_supported_device_indices()[0])


def _pack_gptq(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """[out, in] uint codes -> continuous GPTQ qweight [in * bits / 32, out]."""

    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    out_features, in_features = codes.shape
    words = torch.zeros((in_features // pack_factor, out_features), dtype=torch.int64)
    for index in range(pack_factor):
        field = codes[:, index::pack_factor].T.to(torch.int64) & mask
        words |= field << (bits * index)
    return words.to(torch.int32)


def _pack_gptq_zeros(zeros: torch.Tensor, bits: int) -> torch.Tensor:
    """[out, groups] uint zero points -> GPTQ qzeros [groups, out * bits / 32]."""

    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    out_features, groups = zeros.shape
    words = torch.zeros((groups, out_features // pack_factor), dtype=torch.int64)
    for index in range(pack_factor):
        field = zeros.T[:, index::pack_factor].to(torch.int64) & mask
        words |= field << (bits * index)
    return words.to(torch.int32)


def _pack_awq(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """[rows, columns] uint codes -> classic interleaved AWQ words."""

    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    rows, columns = codes.shape
    grouped = (codes.to(torch.int64) & mask).reshape(rows, columns // pack_factor, pack_factor)
    grouped = grouped[:, :, AWQ_ORDER]
    words = torch.zeros((rows, columns // pack_factor), dtype=torch.int64)
    for index in range(pack_factor):
        words |= grouped[:, :, index] << (bits * index)
    return words.to(torch.int32)


def _quantize(weight: torch.Tensor, bits: int, group_size: int, sym: bool):
    """Return deterministic codes/scales/zero points for one dense weight."""

    out_features, in_features = weight.shape
    levels = (1 << bits) - 1
    half_range = 1 << (bits - 1)
    if group_size == -1:
        group_size = in_features
    groups = in_features // group_size
    codes = torch.empty_like(weight)
    scales = torch.empty((out_features, groups), dtype=weight.dtype)
    zeros = torch.empty((out_features, groups), dtype=weight.dtype)
    for group in range(groups):
        block = weight[:, group * group_size : (group + 1) * group_size]
        if sym:
            scale = block.abs().amax(dim=1, keepdim=True) / (half_range - 1)
            zero = torch.full_like(scale, float(half_range))
        else:
            low = block.amin(dim=1, keepdim=True)
            high = block.amax(dim=1, keepdim=True)
            scale = (high - low) / levels
            zero = torch.round(-low / scale.clamp_min(torch.finfo(weight.dtype).eps))
            zero = zero.clamp(0, levels)
        scale = scale.clamp_min(torch.finfo(weight.dtype).eps)
        codes[:, group * group_size : (group + 1) * group_size] = torch.round(
            block / scale + zero
        ).clamp_(0, levels)
        scales[:, group] = scale.squeeze(1)
        zeros[:, group] = zero.squeeze(1)
    return codes, scales, zeros


def _assert_matches_reference(
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    atol: float,
    normalized_limit: float = 2e-3,
) -> None:
    """Compare with the fp32 oracle after its mandatory output-dtype rounding.

    The elementwise check uses ``rtol=2e-3``; ``atol`` only absorbs one output
    ulp. A normalized Frobenius bound then measures the arithmetic drift without
    letting a single near-zero logit dominate it. bf16 dequantizes weights with
    an 8-bit mantissa, so its bound is wider than fp16's.
    """

    rounded = ref.to(dtype=got.dtype).float()
    assert torch.allclose(got.float(), rounded, atol=atol, rtol=2e-3), (
        f"max abs err {(got.float() - rounded).abs().max().item():.6f}"
    )
    normalized = torch.linalg.vector_norm(got.float() - rounded) / torch.linalg.vector_norm(
        rounded
    ).clamp_min(1e-6)
    assert normalized <= normalized_limit, f"normalized error {normalized.item():.6f}"


def _reference_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    device = x.device
    if group_size == -1:
        group_size = x.shape[-1]
    codes = codes.to(device)
    scales = scales.to(device)
    zeros = zeros.to(device)
    weight = (codes.float() - zeros.float().repeat_interleave(group_size, dim=1)) * (
        scales.float().repeat_interleave(group_size, dim=1)
    )
    # Humming dequantizes to the compute dtype before the MMA, so the oracle
    # uses the same operand precision and only accumulation order differs.
    weight = weight.to(dtype=x.dtype).float()
    out = x.float() @ weight.T
    if bias is not None:
        out = out + bias.float()
    return out


def _make_gptq(
    *,
    in_features: int,
    out_features: int,
    bits: int,
    group_size: int,
    sym: bool,
    dtype: torch.dtype,
    bias: bool,
    qzero_format: int = 2,
) -> tuple[HummingLinear, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    torch.manual_seed(11 + in_features + out_features + bits + group_size + int(sym))
    weight = torch.randn((out_features, in_features), dtype=dtype) * 0.5
    codes, scales, zeros = _quantize(weight, bits=bits, group_size=group_size, sym=sym)

    module = HummingLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=sym,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        pack_dtype=torch.int32,
        dtype=dtype,
    )
    module.qweight = torch.nn.Parameter(_pack_gptq(codes, bits), requires_grad=False)
    module.scales = torch.nn.Parameter(scales.T.contiguous(), requires_grad=False)
    # GPTQ v1 checkpoints store (zero_point - 1) in each field.
    stored_zeros = zeros - 1 if (not sym and qzero_format == 1) else zeros
    module.qzeros = torch.nn.Parameter(_pack_gptq_zeros(stored_zeros, bits), requires_grad=False)
    module.g_idx = torch.nn.Parameter(
        torch.arange(in_features, dtype=torch.int32) // group_size, requires_grad=False
    )
    module.qzero_format(format=qzero_format)
    if bias:
        module.bias = torch.randn(out_features, dtype=dtype)
    module = module.to(device=_device())
    # post_init() replaces `bias` with humming's C-layout permutation, so keep
    # the checkpoint bias for the dequantized reference.
    raw_bias = None if not bias else module.bias.detach().clone()
    module.post_init()
    return module, codes, scales, zeros, raw_bias


def _make_awq(
    *,
    in_features: int,
    out_features: int,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
    bias: bool,
) -> tuple[AwqHummingLinear, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    torch.manual_seed(29 + in_features + out_features + group_size)
    weight = torch.randn((out_features, in_features), dtype=dtype) * 0.5
    codes, scales, zeros = _quantize(weight, bits=bits, group_size=group_size, sym=False)

    module = AwqHummingLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        pack_dtype=torch.int32,
        dtype=dtype,
    )
    module.qweight = torch.nn.Parameter(
        _pack_awq(codes.T.contiguous(), bits), requires_grad=False
    )
    module.scales = torch.nn.Parameter(scales.T.contiguous(), requires_grad=False)
    module.qzeros = torch.nn.Parameter(
        _pack_awq(zeros.T.contiguous(), bits), requires_grad=False
    )
    if bias:
        module.bias = torch.randn(out_features, dtype=dtype)
    module = module.to(device=_device())
    # post_init() permutes `bias` into humming's C layout; keep the checkpoint
    # bias for the dequantized reference.
    raw_bias = None if not bias else module.bias.detach().clone()
    module.post_init()
    return module, codes, scales, zeros, raw_bias


@requires_humming
@pytest.mark.parametrize("bits", HUMMING_GPTQ_BITS)
@pytest.mark.parametrize("group_size", [32, 64, 128, -1])
def test_gptq_humming_matches_dequantized_reference(bits: int, group_size: int):
    in_features, out_features = 512, 256
    effective_group = in_features if group_size == -1 else group_size
    module, codes, scales, zeros, bias = _make_gptq(
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        group_size=group_size,
        sym=True,
        dtype=torch.float16,
        bias=True,
    )

    x = torch.randn(13, in_features, dtype=torch.float16, device=_device())
    got = module(x)
    ref = _reference_linear(
        x, codes.to(_device()), scales, zeros, effective_group, bias
    )
    assert got.shape == (13, out_features)
    _assert_matches_reference(got, ref, atol=0.05)


@requires_humming
@pytest.mark.parametrize("qzero_format", [1, 2])
def test_gptq_humming_asymmetric_zero_points(qzero_format: int):
    in_features, out_features, group_size, bits = 512, 256, 128, 4
    module, codes, scales, zeros, bias = _make_gptq(
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        group_size=group_size,
        sym=False,
        dtype=torch.float16,
        bias=False,
        qzero_format=qzero_format,
    )

    x = torch.randn(5, in_features, dtype=torch.float16, device=_device())
    got = module(x)
    ref = _reference_linear(
        x, codes.to(_device()), scales, zeros, group_size, bias
    )
    _assert_matches_reference(got, ref, atol=0.05)


@requires_humming
@pytest.mark.parametrize("bits", HUMMING_AWQ_BITS)
@pytest.mark.parametrize("group_size", [32, 64, 128, -1])
def test_awq_humming_matches_dequantized_reference(bits: int, group_size: int):
    in_features, out_features = 512, 256
    effective_group = in_features if group_size == -1 else group_size
    module, codes, scales, zeros, bias = _make_awq(
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        group_size=group_size,
        dtype=torch.float16,
        bias=True,
    )

    x = torch.randn(9, in_features, dtype=torch.float16, device=_device())
    got = module(x)
    ref = _reference_linear(
        x, codes.to(_device()), scales, zeros, effective_group, bias
    )
    _assert_matches_reference(got, ref, atol=0.05)


@requires_humming
def test_humming_pads_tile_misaligned_shapes():
    """in/out features that are not humming tile multiples still round-trip."""

    in_features, out_features, group_size = 384, 200, 128
    module, codes, scales, zeros, bias = _make_gptq(
        in_features=in_features,
        out_features=out_features,
        bits=4,
        group_size=group_size,
        sym=True,
        dtype=torch.bfloat16,
        bias=True,
    )

    x = torch.randn(3, 7, in_features, dtype=torch.bfloat16, device=_device())
    got = module(x)
    ref = _reference_linear(
        x.reshape(-1, in_features),
        codes.to(_device()),
        scales,
        zeros,
        group_size,
        bias,
    ).reshape(3, 7, out_features)
    assert got.shape == (3, 7, out_features)
    # One bf16 ulp is 0.125 at |x| in [16, 32) and 0.25 at [32, 64); the wider
    # normalized bound covers bf16's 8-bit weight-dequantization mantissa.
    _assert_matches_reference(got, ref, atol=0.25, normalized_limit=4e-3)


@requires_humming
def test_humming_empty_rows():
    in_features, out_features = 512, 256
    module, _, _, _, _ = _make_gptq(
        in_features=in_features,
        out_features=out_features,
        bits=4,
        group_size=128,
        sym=True,
        dtype=torch.float16,
        bias=False,
    )

    empty = torch.empty(0, in_features, dtype=torch.float16, device=_device())
    assert module(empty).shape == (0, out_features)


@requires_humming
def test_humming_wider_activation_round_trips_its_dtype():
    """One activation dtype per module: a fresh layer may take fp32."""

    in_features, out_features = 512, 256
    module, _, _, _, _ = _make_gptq(
        in_features=in_features,
        out_features=out_features,
        bits=4,
        group_size=128,
        sym=True,
        dtype=torch.float16,
        bias=False,
    )

    # The kernel consumes the layer's compute dtype and returns the public
    # activation dtype (contracts/kernel/forward-dtype.md).
    wide = torch.randn(4, in_features, dtype=torch.float32, device=_device())
    out = module(wide)
    assert out.dtype == torch.float32
    assert out.shape == (4, out_features)


def test_humming_compute_config_uses_explicit_arguments():
    assert json.loads(humming_compute_config()) == {
        "use_batch_invariant": False,
        "use_f16_accum": False,
        "gemm_type": "dense",
    }

    tuned = json.loads(humming_compute_config(batch_invariant=True, f16_accum=True))
    assert tuned["use_batch_invariant"] is True
    assert tuned["use_f16_accum"] is True
    assert tuned["gemm_type"] == "dense"


@requires_humming
def test_humming_batch_invariant_is_reproducible():
    """The batch-invariant policy pins humming's split-K reduction order."""

    module, _, _, _, _ = _make_gptq(
        in_features=512,
        out_features=256,
        bits=4,
        group_size=128,
        sym=True,
        dtype=torch.float16,
        bias=False,
    )
    x = torch.randn(17, 512, dtype=torch.float16, device=_device())
    policy = humming_compute_config(batch_invariant=True)

    def run() -> torch.Tensor:
        return humming_apply_linear(
            config=module.humming_config,
            inputs=x,
            weight=module.qweight,
            weight_scale=module.scales,
            zero_point=module.qzeros if module.has_zero_point else None,
            bias=module.bias,
            locks=module.locks,
            compute_config=policy,
        )

    assert torch.equal(run(), run())


def test_humming_backend_registration():
    assert HummingLinear.SUPPORTS_BACKENDS == [BACKEND.HUMMING]
    assert AwqHummingLinear.SUPPORTS_BACKENDS == [BACKEND.HUMMING]
    assert HummingLinear.SUPPORTS_METHODS == [METHOD.GPTQ]
    assert AwqHummingLinear.SUPPORTS_METHODS == [METHOD.AWQ]
    assert FORMAT.GPTQ in HummingLinear.SUPPORTS_FORMATS
    assert FORMAT.GPTQ_V2 in HummingLinear.SUPPORTS_FORMATS
    assert FORMAT.GEMM in AwqHummingLinear.SUPPORTS_FORMATS
    # Priority 0 = format support without changing AUTO's default kernel.
    assert all(priority == 0 for priority in HummingLinear.SUPPORTS_FORMATS.values())
    assert all(priority == 0 for priority in AwqHummingLinear.SUPPORTS_FORMATS.values())


def test_humming_is_registered_but_not_auto_selected(monkeypatch):
    importer.clear_validation_cache()
    auto_map, support_map = importer.build_kernel_support_maps()

    assert BACKEND.HUMMING in support_map[METHOD.GPTQ][FORMAT.GPTQ]
    assert BACKEND.HUMMING in support_map[METHOD.AWQ][FORMAT.GEMM]
    # Opt-in only: AUTO keeps its existing priority order.
    assert BACKEND.HUMMING not in auto_map[METHOD.GPTQ][FORMAT.GPTQ]
    assert BACKEND.HUMMING not in auto_map[METHOD.AWQ][FORMAT.GEMM]

    assert (
        importer.get_kernel_for_backend(BACKEND.HUMMING, METHOD.GPTQ, FORMAT.GPTQ)
        is HummingLinear
    )
    assert (
        importer.get_kernel_for_backend(BACKEND.HUMMING, METHOD.AWQ, FORMAT.GEMM)
        is AwqHummingLinear
    )


def test_humming_rejects_unsupported_layouts():
    with pytest.raises(Exception):
        # 3/5/6/7-bit GPTQ use word layouts humming cannot read.
        HummingLinear(
            bits=5,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=512,
            out_features=256,
            pack_dtype=torch.int32,
            dtype=torch.float16,
        )

    with pytest.raises(Exception):
        # Humming's GPTQ schema asserts against act-order checkpoints.
        HummingLinear(
            bits=4,
            group_size=128,
            desc_act=True,
            sym=True,
            in_features=512,
            out_features=256,
            pack_dtype=torch.int32,
            dtype=torch.float16,
        )


def test_humming_validate_once_reports_missing_runtime(monkeypatch):
    monkeypatch.setattr(
        "gptqmodel.nn_modules.qlinear.humming.humming_import_exception",
        ImportError("humming is missing"),
    )
    ok, err = HummingLinear.validate_once()
    assert ok is False
    assert isinstance(err, ImportError)
    assert "humming" in str(err).lower()
