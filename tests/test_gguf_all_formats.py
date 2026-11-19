# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Full-format GGUF coverage: pack -> dequantize -> linear forward.

Every alias in ``_GGUF_BITS_ALIAS_TO_TENSOR_QTYPE`` must quantize a weight
matrix, dequantize back to float, and run through both the torch and the
native ggml CPU kernels without leaving the quantized representation.
"""

from __future__ import annotations

import ctypes
import functools
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

import gptqmodel.nn_modules.qlinear.gguf_cpp as gguf_cpp
from gptqmodel.nn_modules.qlinear.gguf import (
    _GGUF_AVAILABLE,
    _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE,
    _GGUF_TYPE_INFO,
    GGUFTorchLinear,
    _dequantize_gguf_tensor_numpy,
    _quantize_gguf_tensor_numpy,
)
from gptqmodel.nn_modules.qlinear.gguf_cpp import (
    GGUFCppKernel,
    _get_ggml_bridge,
    _require_llama_cpp,
)
from gptqmodel.quantization.config import (
    FORMAT,
    GGUFBits,
    GGUFConfig,
    QuantizeConfig,
    _gguf_bits_from_components,
    _normalize_gguf_public_format,
)
from gptqmodel.utils import internal_gguf as ig
from gptqmodel.utils.kernel import KernelConfig, configure_kernel


# llama-cpp-python is imported on demand by the kernel now, so load it here
# before the module-level skip guards below read `_llama_cpp_lib`.
_require_llama_cpp()
from gptqmodel.nn_modules.qlinear.gguf_cpp import _llama_cpp_lib  # noqa: E402


if _GGUF_AVAILABLE:
    import gguf as gguf_lib
else:
    gguf_lib = None


ALL_ALIASES = sorted(_GGUF_BITS_ALIAS_TO_TENSOR_QTYPE)
# KV-cache/auxiliary types have no upstream mul_mat kernel; the torch path is
# the reference implementation for them.
_GGML_MATMUL_SKIP_QTYPES = {"Q8_1", "Q8_K"}
# Formats whose dequantizer is only in our internal table (gguf_lib lacks them).
_GGUF_LIB_SKIP_QTYPES = {"Q1_0", "Q1_0_g128", "Q2_0", "Q8_1", "Q8_K"}
# Forward tolerance vs a native fp32 Linear, keyed on effective bits/weight.
# <=2bpw formats (1-bit, ternary, iq1/iq2, q2_0) quantize aggressively.
def _forward_rel_bound(alias: str) -> float:
    if alias.startswith(("iq1", "q1_0", "tq")) or alias == "q2_0":
        return 1.5
    if alias.startswith(("iq2", "iq3", "q2_k", "q3_k")):
        return 1.0
    return 0.5


def _visible_one_ulp(dtype: torch.dtype, expected: torch.Tensor) -> float:
    """Largest one-step rounding error of ``expected`` at ``dtype``.

    ``expected`` is already the once-rounded visible result, so the kernel is
    allowed at most the adjacent representable step (which also covers the
    FP16/BF16 subnormal range). Kernels may not accumulate in the activation
    dtype and then round again.
    """
    upper = torch.nextafter(expected, torch.full_like(expected, float("inf")))
    # ``nextafter`` runs in ``dtype``, so the step is that dtype's ulp (it also
    # reports the fixed step of zero and subnormal values).
    step = (upper.float() - expected.float()).abs()
    return float(max(step.max().item(), torch.finfo(dtype).tiny))


@functools.lru_cache(maxsize=None)
def _cpp_kernel_unavailable_reason() -> str:
    """Why the llama.cpp-backed CPU kernel cannot run on this host, if it cannot.

    ``llama_cpp`` can import while its shared library still fails to load (for
    example a wheel built against a different CUDA runtime). The bridge probe is
    the honest check: a kernel that cannot construct is unavailable, not broken.
    """

    try:
        _get_ggml_bridge()
        GGUFCppKernel(
            bits="q4_0", group_size=-1, sym=True, desc_act=False,
            in_features=32, out_features=16,
        )
    except Exception as exc:  # noqa: BLE001 - optional native dependency
        return f"{type(exc).__name__}: {exc}"
    return ""


@functools.lru_cache(maxsize=None)
def _quantizer_available(qtype: str) -> bool:
    """Whether this host can *produce* ``qtype`` (not merely decode it).

    ``_quantize_gguf_tensor_numpy`` prefers ``gguf``'s reference quantizer, then
    the internal numpy fallback, then llama.cpp's ggml runtime. Formats outside
    that coverage would otherwise fail with a dependency ``NotImplementedError``
    instead of skipping, which hides real regressions behind an unrelated
    environment gap.
    """

    block = _GGUF_TYPE_INFO[qtype]["block_size"]
    probe = np.zeros((1, block), dtype=np.float32)
    probe[0, 0] = 1.0
    try:
        _quantize_gguf_tensor_numpy(probe, qtype)
    except NotImplementedError:
        return False
    return True


def _dequantizer_unavailable(qtype: str) -> bool:
    """Whether this host cannot decode ``qtype`` at all."""

    type_size = _GGUF_TYPE_INFO[qtype]["type_size"]
    try:
        _dequantize_gguf_tensor_numpy(np.zeros((1, type_size), np.uint8), qtype)
    except NotImplementedError:
        return True
    return False


def _skip_without_quantizer(qtype: str) -> None:
    if not _quantizer_available(qtype):
        pytest.skip(
            f"no quantizer for {qtype} on this host: the `gguf` reference package "
            f"is {'installed' if _GGUF_AVAILABLE else 'missing'}, the numpy fallback "
            "does not implement it, and llama.cpp's ggml runtime is "
            f"{'available' if _llama_cpp_lib is not None else 'unavailable'}"
        )


def _weight_rows(qtype: str) -> int:
    return max(1, _GGUF_TYPE_INFO[qtype]["block_size"] // 64)


def _lin_and_x(in_features):
    torch.manual_seed(0)
    lin = torch.nn.Linear(in_features, 16, bias=False)
    lin.weight.data = torch.randn(16, in_features) * 0.05
    x = torch.randn(4, in_features)
    return lin, x


def _reference_dequantize(packed, qtype):
    if qtype not in _GGUF_LIB_SKIP_QTYPES:
        if gguf_lib is None:
            pytest.skip("gguf package required for the independent oracle")
        return torch.from_numpy(gguf_lib.dequantize(packed, gguf_lib.GGMLQuantizationType[qtype]))
    blocks = torch.from_numpy(packed.copy()).reshape(-1, _GGUF_TYPE_INFO[qtype]["type_size"])
    if qtype in {"Q8_1", "Q8_K"}:
        scale_bytes = 2 if qtype == "Q8_1" else 4
        scale_dtype = torch.float16 if qtype == "Q8_1" else torch.float32
        scale = blocks[:, :scale_bytes].contiguous().view(scale_dtype).float()
        width = _GGUF_TYPE_INFO[qtype]["block_size"]
        codes = blocks[:, 4:4 + width].contiguous().view(torch.int8).float()
    else:
        # Sign-only and Q2_0 pack consecutive codes into each byte, low bits first.
        scale = blocks[:, :2].contiguous().view(torch.float16).float()
        shifts = torch.arange(8) if qtype.startswith("Q1_0") else torch.arange(0, 8, 2)
        mask = 1 if qtype.startswith("Q1_0") else 3
        codes = ((blocks[:, 2:, None].int() >> shifts[None, None, :]) & mask).flatten(1).float()
        codes = codes * 2 - 1 if mask == 1 else codes - 1
    return (scale * codes).reshape(packed.shape[0], -1)


def _cpp_from_torch(module):
    cpp = GGUFCppKernel(
        bits=module.bits, group_size=-1, sym=True, desc_act=False,
        in_features=module.in_features, out_features=module.out_features,
        bias=module.bias is not None,
    )
    cpp.load_state_dict(module.state_dict())
    return cpp.eval()


_Q8_1_HEADER_BYTES = 4
_Q8_1_OVERFLOW_BLOCKS = [
    # 32 x 3000: d = 23.62 is finite in FP16, but s = 96000 is not.
    pytest.param(np.full(32, 3000.0, dtype=np.float32), "auxiliary scale-sum s", id="scale-sum-3000"),
    # FP16 rounds 65536 up to inf, while its neighbouring 65504 is still finite.
    pytest.param(np.full(32, 2048.0, dtype=np.float32), "auxiliary scale-sum s", id="scale-sum-2048"),
    pytest.param(np.full(32, -2048.0, dtype=np.float32), "auxiliary scale-sum s", id="scale-sum-neg-2048"),
    # 1e7 makes the block scale itself leave the FP16 range.
    pytest.param(np.full(32, 1e7, dtype=np.float32), "block scale d", id="block-scale-1e7"),
]
_Q8_1_ACCEPTED_BLOCKS = [
    # 2047 * 32 = 65504 exactly: the largest finite FP16 scale-sum.
    pytest.param(np.full(32, 2047.0, dtype=np.float32), id="scale-sum-max-finite"),
    pytest.param(np.full(32, -2047.0, dtype=np.float32), id="scale-sum-max-finite-negative"),
    pytest.param(np.zeros(32, dtype=np.float32), id="zero-block"),
    pytest.param(np.linspace(-1.0, 1.0, 32, dtype=np.float32), id="ordinary-block"),
]


def _q8_1_fields(packed: np.ndarray):
    blocks = np.asarray(packed, dtype=np.uint8).reshape(-1, _GGUF_TYPE_INFO["Q8_1"]["type_size"])
    scale = blocks[:, :2].copy().view(np.float16).reshape(-1)
    scale_sum = blocks[:, 2:4].copy().view(np.float16).reshape(-1)
    codes = blocks[:, _Q8_1_HEADER_BYTES:].copy().view(np.int8).reshape(-1, 32)
    return scale, scale_sum, codes


@pytest.mark.parametrize("block,reason", _Q8_1_OVERFLOW_BLOCKS)
def test_q8_1_rejects_auxiliary_fields_that_overflow_fp16(block, reason):
    """Q8_1 must not publish an FP16 ``d``/``s`` header that rounds to infinity.

    ``d = max|x| / 127`` is finite for all of these blocks, so checking only the
    decoded weight or ``d`` accepts them. The auxiliary ``s = d * sum(qs)`` is
    what leaves FP16 range, and any reader that folds ``s`` into its dot product
    then produces a non-finite result. The format's header dtype is fixed, so
    the packer fails closed with the offending field named instead of emitting
    an unreadable block.
    """

    _skip_without_quantizer("Q8_1")
    weight = block.reshape(1, -1)

    with pytest.raises(ValueError) as excinfo:
        _quantize_gguf_tensor_numpy(weight, "Q8_1")

    message = str(excinfo.value)
    assert "Q8_1" in message
    assert reason in message
    assert "block 0 of 1" in message
    with np.errstate(over="ignore", invalid="ignore"):
        expected_scale = np.abs(block).max() / np.float32(127.0)
        expected_sum = np.float32(sum(np.copysign(
            np.trunc(np.abs(block / expected_scale))
            + (np.abs(block / expected_scale) - np.trunc(np.abs(block / expected_scale)) >= 0.5),
            block,
        ).astype(np.int32))) * expected_scale
    assert np.isfinite(expected_scale)
    if reason == "auxiliary scale-sum s":
        # The reported trigger is the auxiliary field: d stays finite while the
        # independently computed scale-sum does not.
        assert np.isfinite(np.float16(expected_scale))
        assert not np.isfinite(np.float16(expected_sum))
    else:
        assert not np.isfinite(np.float16(expected_scale))


def test_q8_1_reports_the_first_unrepresentable_block():
    """A mixed matrix must name the failing block, not the first accepted one."""

    _skip_without_quantizer("Q8_1")
    weight = np.zeros((1, 64), dtype=np.float32)
    weight[0, :32] = 1.0
    weight[0, 32:] = 3000.0

    with pytest.raises(ValueError) as excinfo:
        _quantize_gguf_tensor_numpy(weight, "Q8_1")

    message = str(excinfo.value)
    assert "block 1 of 2" in message
    assert "scale-sum s" in message


def test_q8_1_rejects_packed_bytes_with_a_nonfinite_header():
    """The shared entry point validates bytes from any packer, not just the fallback."""

    from gptqmodel.nn_modules.qlinear.gguf import _check_packed_q8_1_fields

    packed = np.zeros((1, _GGUF_TYPE_INFO["Q8_1"]["type_size"]), dtype=np.uint8)
    # block_q8_1 stores fp16 d first, then fp16 s; 0x7C00 is positive infinity.
    packed[0, :4] = np.array([0x0000, 0x7C00], dtype=np.uint16).view(np.uint8)
    with pytest.raises(ValueError, match=r"auxiliary scale-sum s.*block 0 of 1"):
        _check_packed_q8_1_fields(packed)

    packed[0, :4] = np.array([0x7C00, 0x0000], dtype=np.uint16).view(np.uint8)
    with pytest.raises(ValueError, match=r"block scale d.*block 0 of 1"):
        _check_packed_q8_1_fields(packed)

    _check_packed_q8_1_fields(
        _quantize_gguf_tensor_numpy(np.full((1, 32), 2047.0, dtype=np.float32), "Q8_1")
    )


@pytest.mark.parametrize("block", _Q8_1_ACCEPTED_BLOCKS)
def test_q8_1_in_range_fields_stay_bit_exact(block):
    """Blocks that fit FP16 must keep the existing ggml bit layout and values."""

    _skip_without_quantizer("Q8_1")
    weight = block.reshape(1, -1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        packed = _quantize_gguf_tensor_numpy(weight, "Q8_1")

    overflow_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, RuntimeWarning) and "overflow" in str(item.message)
    ]
    assert not overflow_warnings, overflow_warnings

    scale, scale_sum, codes = _q8_1_fields(packed)
    assert np.isfinite(scale).all()
    assert np.isfinite(scale_sum).all()
    float32_scale = np.float32(np.abs(block).max() / np.float32(127.0))
    assert scale[0] == np.float16(float32_scale)
    # ``s`` is accumulated in FP32 from the unrounded scale and rounded once to
    # FP16, so comparing against a product of the stored FP16 scale would hide
    # the rounding ``ggml`` performs.
    np.testing.assert_array_equal(
        scale_sum[0],
        np.float16(np.float32(codes[0].astype(np.float32).sum()) * float32_scale),
    )
    # The scale-sum is a stored field, so the dequantized weight alone cannot
    # prove it was written correctly.
    expected = (scale[0].astype(np.float64) * codes[0].astype(np.float64))
    np.testing.assert_allclose(
        _dequantize_gguf_tensor_numpy(packed, "Q8_1")[0],
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("alias", ALL_ALIASES)
def test_gguf_quantize_dequantize_roundtrip(alias):
    """Pack every alias and verify dequant stays inside its precision budget."""
    qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE[alias]
    _skip_without_quantizer(qtype)
    info = _GGUF_TYPE_INFO[qtype]
    rows = _weight_rows(qtype)
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((rows, info["block_size"])).astype(np.float32)

    packed = _quantize_gguf_tensor_numpy(weight, qtype)
    assert packed.dtype == np.uint8
    assert packed.shape == (
        rows,
        weight.shape[1] // info["block_size"] * info["type_size"],
    )

    restored = _dequantize_gguf_tensor_numpy(packed, qtype)
    np.testing.assert_array_equal(restored.shape, weight.shape)
    np.testing.assert_allclose(
        np.abs(restored - weight).max(),
        0.0,
        atol=max(0.05, 0.75 * np.abs(weight).max()),
    )


@pytest.mark.parametrize("alias", ALL_ALIASES)
def test_gguf_dequantize_matches_native_and_reference(alias):
    """Our dequantizer must agree bit-exactly with ggml and the `gguf` package."""
    qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE[alias]
    _skip_without_quantizer(qtype)
    info = _GGUF_TYPE_INFO[qtype]
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((2, info["block_size"])).astype(np.float32)
    packed = _quantize_gguf_tensor_numpy(weight, qtype)
    ours = _dequantize_gguf_tensor_numpy(packed, qtype)

    expected = _reference_dequantize(packed, qtype).numpy()

    np.testing.assert_allclose(ours, expected, rtol=1e-6, atol=1e-6)

    bridge = None
    try:
        bridge = _get_ggml_bridge()
    except (ImportError, OSError, RuntimeError):
        # llama-cpp-python is optional: when absent, the ggml-native
        # assertions below are skipped and only the `gguf` oracle runs.
        pass
    if bridge is not None and qtype != "Q8_1":
        ggml_out = bridge.dequantize_rows(
            packed,
            qtype,
            (2, packed.shape[1] // info["type_size"] * info["block_size"]),
        )
        np.testing.assert_allclose(ggml_out, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("alias", ALL_ALIASES)
def test_gguf_torch_linear_forward(alias):
    """GGUFTorchLinear packs a Linear and stays close to the fp32 reference."""
    qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE[alias]
    _skip_without_quantizer(qtype)
    block = _GGUF_TYPE_INFO[qtype]["block_size"]
    lin, x = _lin_and_x(block)

    module = GGUFTorchLinear(
        bits=alias,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=block,
        out_features=16,
        bias=False,
        register_buffers=False,
    )
    module.pack_original(linear=lin, scales=None, zeros=None, g_idx=None)
    module.post_init()

    # Weights must stay quantized between forwards (AGENTS.md storage rule):
    # the buffer is uint8-packed, never a dense float weight.
    assert module.qweight.dtype == torch.uint8
    packed_bytes = module.qweight.numel()
    dense_bytes = 16 * block * 4
    assert packed_bytes < dense_bytes

    ref = lin(x)
    out = module(x)
    assert out.shape == ref.shape
    err = (out - ref).abs().mean()
    assert err <= max(0.02, _forward_rel_bound(alias) * ref.abs().mean())


@pytest.mark.parametrize("alias", ALL_ALIASES)
def test_gguf_cpp_kernel_forward(alias):
    """GGUFCppKernel agrees with the torch path on the same packed weight."""
    if _llama_cpp_lib is None:
        pytest.skip("llama-cpp-python not installed")

    qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE[alias]
    block = _GGUF_TYPE_INFO[qtype]["block_size"]
    lin, x = _lin_and_x(block)

    torch_mod = GGUFTorchLinear(
        bits=alias,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=block,
        out_features=16,
        bias=False,
        register_buffers=False,
    )
    torch_mod.pack_original(linear=lin, scales=None, zeros=None, g_idx=None)
    torch_mod.post_init()

    cpp_mod = GGUFCppKernel(
        bits=alias,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=block,
        out_features=16,
        bias=False,
        register_buffers=False,
    )
    cpp_mod._buffers.update(torch_mod._buffers)
    cpp_mod._parameters.update(torch_mod._parameters)
    for name in (
        "padded_in_features",
        "gguf_tensor_qtype",
        "gguf_block_size",
        "gguf_type_size",
        "scales",
        "qzeros",
        "g_idx",
        "bias",
    ):
        if hasattr(torch_mod, name):
            setattr(cpp_mod, name, getattr(torch_mod, name))

    torch_out = torch_mod(x)
    cpp_out = cpp_mod(x)
    assert cpp_out.shape == torch_out.shape
    # Native ggml also quantizes activations; this measures that extra loss,
    # separately from the exact-activation kernel oracle below.
    diff = (cpp_out - torch_out).abs().max().item()
    assert diff <= 0.05

    if qtype in _GGML_MATMUL_SKIP_QTYPES:
        # ggml mul_mat has no F32-source vec_dot for these types and would
        # segfault; the kernel must select the torch path instead.
        assert cpp_mod._use_ggml_cpu_matmul(x.reshape(-1, block)) is False


@pytest.mark.parametrize("alias", ALL_ALIASES)
def test_gguf_bits_alias_round_trip(alias):
    """Alias -> GGUFBits -> string/public-format is a closed loop."""
    bits = GGUFBits.from_alias(alias)
    assert bits.to_string() == alias

    if bits.version == "mxfp":
        assert bits.to_public_format() == "mxfp4"
    elif bits.version == "nvfp":
        assert bits.to_public_format() == "nvfp4"
    else:
        public = bits.to_public_format()
        assert public.startswith(f"{bits.version}_")
        resolved = _gguf_bits_from_components(bits.bits, public)
        assert resolved.to_string() == alias

    config = GGUFConfig(bits=alias)
    restored = QuantizeConfig.from_quant_config(config.to_dict())
    assert restored.runtime_bits == bits


def test_gguf_public_format_normalization():
    assert _normalize_gguf_public_format("q4_0") == "q_0"
    assert _normalize_gguf_public_format("iq2_xxs") == "iq_xxs"
    assert _normalize_gguf_public_format("mxfp4") == "mxfp4"
    assert _normalize_gguf_public_format(FORMAT.GGUF) is None
    assert _normalize_gguf_public_format(None) is None


@pytest.mark.parametrize("alias", ALL_ALIASES)
def test_internal_gguf_dequantize_and_torch(alias):
    """The shared internal_gguf helpers cover every alias in both numpy and torch."""
    qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE[alias]
    # The decode table is independent of the quantizer, but ``internal_gguf``
    # delegates exotic types to the same llama.cpp/`gguf` chain.
    if _dequantizer_unavailable(qtype):
        pytest.skip(f"no dequantizer for {qtype} on this host")
    enum = getattr(ig.GGMLQuantizationType, qtype)
    block_size, type_size = ig.GGML_QUANT_SIZES[enum]
    data = np.zeros((2, type_size), dtype=np.uint8)

    out = ig.dequantize(data, enum)
    assert out.shape[-1] == block_size
    assert out.dtype == np.float32

    tensor = ig.dequantize_to_torch(data, enum)
    assert tensor.shape[-1] == block_size


def test_gguf_bits_default_alias_map():
    from gptqmodel.quantization.config import _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH

    assert _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH[3] == "q3_k"
    assert _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH[4] == "q4_0"


@pytest.mark.parametrize("qtype", ["Q8_1", "Q8_K"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_q8_pack_boundaries_match_torch_and_ggml(qtype, dtype):
    width = _GGUF_TYPE_INFO[qtype]["block_size"]
    ties = torch.tensor([-126.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 126.5], dtype=dtype)
    values = torch.cat((
        torch.tensor([127.0, -127.0, 0.0, -0.0], dtype=dtype),
        torch.nextafter(ties, torch.full_like(ties, -torch.inf)),
        ties,
        torch.nextafter(ties, torch.full_like(ties, torch.inf)),
    )).float()
    row = torch.zeros(width)
    row[:values.numel()] = values
    weight = torch.stack((row, -row, row * 0.003, row * 3.1, row * 0))
    if qtype == "Q8_1":
        scale = weight.abs().amax(1, keepdim=True) / 127
        inverse = torch.where(scale == 0, 0, scale.reciprocal())
        scaled = weight * inverse
        integral = scaled.abs().trunc()
        codes = ((integral + (scaled.abs() - integral >= 0.5)) * scaled.sign()).to(torch.int8)
        expected = torch.cat((
            scale.half().view(torch.uint8),
            (codes.float().sum(1, keepdim=True) * scale).half().view(torch.uint8),
            codes.view(torch.uint8),
        ), dim=1)
    else:
        peak = weight.gather(1, weight.abs().argmax(1, keepdim=True))
        inverse = torch.where(peak == 0, 0, torch.div(torch.full_like(peak, -127), peak))
        scale = torch.where(inverse == 0, 0, inverse.reciprocal())
        codes = (weight * inverse).round().clamp(-127, 127).to(torch.int8)
        sums = codes.reshape(-1, 16, 16).sum(2).to(torch.int16)
        expected = torch.cat((
            scale.view(torch.uint8), codes.view(torch.uint8), sums.view(torch.uint8),
        ), dim=1)
    packed = _quantize_gguf_tensor_numpy(weight.numpy(), qtype)
    np.testing.assert_array_equal(packed, expected.numpy())
    torch.testing.assert_close(
        torch.from_numpy(_dequantize_gguf_tensor_numpy(packed, qtype)),
        _reference_dequantize(packed, qtype), rtol=1e-6, atol=1e-6,
    )
    if _llama_cpp_lib is not None:
        name = "quantize_row_q8_1_ref" if qtype == "Q8_1" else "quantize_row_q8_K_ref"
        fn = getattr(_get_ggml_bridge()._ggml_base, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
        native = np.zeros_like(packed)  # Q8_K leaves zero-block auxiliary sums unwritten.
        fn(weight.numpy().ctypes.data, native.ctypes.data, weight.numel())
        np.testing.assert_array_equal(packed, native)


def _q8_k_parts(packed: np.ndarray):
    blocks = np.asarray(packed, dtype=np.uint8).reshape(-1, _GGUF_TYPE_INFO["Q8_K"]["type_size"])
    scale = blocks[:, :4].copy().view(np.float32).reshape(-1)
    codes = blocks[:, 4:260].copy().view(np.int8).reshape(-1, 256)
    sums = blocks[:, 260:].copy().view(np.int16).reshape(-1, 16)
    return scale, codes, sums


_Q8_K_OVERFLOW_PEAKS = [
    pytest.param(np.float32(127.0) / np.finfo(np.float32).max, id="reciprocal-threshold"),
    pytest.param(
        np.nextafter(np.float32(127.0) / np.finfo(np.float32).max, np.float32(0)),
        id="reciprocal-overflows",
    ),
    pytest.param(np.float32(1e-36), id="subnormal-control"),
    pytest.param(np.float32(1e-37), id="positive-1e-37"),
    pytest.param(np.float32(-1e-37), id="negative-1e-37"),
    pytest.param(np.float32(1e-42), id="deep-subnormal"),
    pytest.param(np.float32(0.0), id="zero"),
    pytest.param(np.nextafter(np.float32(0.0), np.float32(1.0)), id="scale-unrepresentable"),
]


@pytest.mark.parametrize("peak", _Q8_K_OVERFLOW_PEAKS)
def test_q8_k_overflowing_reciprocal_keeps_representable_scale(peak):
    """A subnormal Q8_K scale must not become zero through a reciprocal overflow.

    This is the CPU NumPy packer, so the assertions pin IEEE-754 gradual
    underflow for the stored fp32 scale instead of assuming a device's
    flush-to-zero mode. The direct quotient ``-peak / 127`` is the independent
    FP64 oracle, rounded once to fp32.
    """

    ratios = np.array([1.0, -1.0, 0.5, -0.5, 0.9, -0.9, 0.75, -0.75], dtype=np.float32)
    block = (np.resize(ratios, 256) * peak).astype(np.float32)
    block[0] = peak  # Keep the requested peak even when a ratio underflows.
    weight = block.reshape(1, -1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        packed = _quantize_gguf_tensor_numpy(weight, "Q8_K")

    overflow_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, RuntimeWarning) and "overflow" in str(item.message)
    ]
    assert not overflow_warnings, overflow_warnings

    scale, codes, sums = _q8_k_parts(packed)
    np.testing.assert_array_equal(sums, codes.reshape(1, 16, 16).sum(axis=2).astype(np.int16))
    assert np.isfinite(scale).all()

    expected_scale = np.float32(-float(peak) / 127.0)
    uniform = np.full((1, 256), peak, dtype=np.float32)
    uniform_scale, uniform_codes, _ = _q8_k_parts(
        _quantize_gguf_tensor_numpy(uniform, "Q8_K")
    )
    assert uniform_scale[0] == expected_scale
    if expected_scale == 0:
        np.testing.assert_array_equal(uniform_codes, 0)
    else:
        np.testing.assert_allclose(
            uniform_scale[0] * uniform_codes[0].astype(np.float64),
            uniform[0],
            rtol=0.02,
            atol=0,
        )

    if peak == 0 or expected_scale == 0:
        # No nonzero fp32 scale represents this block. Pack the canonical zero
        # block instead of deriving codes from an Inf reciprocal.
        assert scale[0] == 0
        np.testing.assert_array_equal(codes, 0)
        np.testing.assert_array_equal(scale[0] * codes[0].astype(np.float64), np.zeros(256))
        return

    assert scale[0] == expected_scale
    if abs(float(expected_scale)) < np.finfo(np.float32).tiny:
        assert scale[0] != 0  # Stored as a subnormal, not flushed to zero.
    assert codes[0, 0] < 0  # The peak and its opposite-signed scale stay negative.
    assert np.unique(codes).size > 2  # Codes are not all saturated by an Inf reciprocal.
    np.testing.assert_allclose(
        scale[0] * codes[0].astype(np.float64),
        block,
        rtol=0.02,
        atol=0,
    )


@pytest.mark.parametrize("alias", ["q8_1", "q8_k"])
@pytest.mark.parametrize("override", ["1", "0", None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpp_unsupported_native_types_always_fall_back(alias, override, dtype, monkeypatch):
    unavailable = _cpp_kernel_unavailable_reason()
    if unavailable:
        pytest.skip(f"GGUF C++ kernel unavailable: {unavailable}")
    configure_kernel(
        KernelConfig(
            gguf_cpp_native_matmul=None if override is None else override == "1"
        )
    )
    qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE[alias]
    width = _GGUF_TYPE_INFO[qtype]["block_size"] + 3
    lin, x = _lin_and_x(width)
    module = GGUFTorchLinear(
        bits=alias, group_size=-1, sym=True, desc_act=False,
        in_features=width, out_features=16,
    )
    module.pack_original(lin, None, None)
    cpp = _cpp_from_torch(module)
    x = x.to(dtype).reshape(2, 2, width)
    weight = _reference_dequantize(module.qweight.numpy(), qtype)[:, :width].to(dtype)
    # CPU fused execution accumulates the full-precision reconstructed weight.
    monkeypatch.setattr(cpp, "_should_use_fused_k_forward", lambda _: False)
    expected = x @ weight.T
    # The fused reference rounds once at the output dtype. A bf16 output has a
    # one-ULP step of 2**-7 at unit magnitude, so fp16/fp32 keep the tighter
    # 2e-3 absolute bound while bf16 allows exactly one dtype ULP.
    atol = max(2e-3, torch.finfo(dtype).eps)
    for _ in range(2):
        torch.testing.assert_close(cpp(x), expected, rtol=2e-3, atol=atol)
        assert cpp._use_ggml_cpu_matmul(x.reshape(-1, width)) is False
        assert cpp.qweight.dtype == torch.uint8
        assert all(not isinstance(v, torch.Tensor) for v in vars(cpp).values())
        assert cpp._parameters == {}
        assert set(cpp._buffers) == {"qweight"}
    assert cpp(x[:0]).shape == (0, 2, 16)


@pytest.mark.parametrize("shape", [(2, 255), (2, 257), (1, 512)])
def test_native_dequant_rejects_wrong_output_shape(shape):
    if _llama_cpp_lib is None:
        pytest.skip("llama-cpp-python not installed")
    with pytest.raises(ValueError, match="dequantized shape"):
        _get_ggml_bridge().dequantize_rows(np.zeros((2, 66), np.uint8), "IQ2_XXS", shape)


def test_native_dequant_rejects_truncated_block():
    if _llama_cpp_lib is None:
        pytest.skip("llama-cpp-python not installed")
    with pytest.raises(ValueError, match="complete packed rows"):
        _get_ggml_bridge().dequantize_rows(np.zeros((2, 65), np.uint8), "IQ2_XXS", (2, 256))


def test_cpp_rejects_wrong_input_width_and_clears_autotune():
    unavailable = _cpp_kernel_unavailable_reason()
    if unavailable:
        pytest.skip(f"GGUF C++ kernel unavailable: {unavailable}")
    cpp = GGUFCppKernel(bits="q4_0", group_size=-1, sym=True, desc_act=False, in_features=33, out_features=16)
    with pytest.raises(ValueError, match="last dimension"):
        cpp(torch.zeros(1, 34))
    cpp._gguf_cpu_autotune_cache[(1, torch.float32, "Q4_0")] = True
    cpp.clear_autotune()
    assert not cpp._gguf_cpu_autotune_cache


def test_native_matmul_rejects_malformed_weight_and_handles_empty_rows():
    if _llama_cpp_lib is None:
        pytest.skip("llama-cpp-python not installed")
    kwargs = {"x": torch.zeros(0, 32), "gguf_tensor_qtype": "Q4_0", "padded_in_features": 32, "out_features": 2}
    bridge = _get_ggml_bridge()
    with pytest.raises(ValueError, match="packed uint8"):
        bridge.quantized_matmul_cpu(qweight=torch.zeros(2, 17, dtype=torch.uint8), **kwargs)
    assert bridge.quantized_matmul_cpu(qweight=torch.zeros(2, 18, dtype=torch.uint8), **kwargs).shape == (0, 2)


@pytest.mark.parametrize("alias", ALL_ALIASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("path", ["native", "dense", "fused"])
def test_gguf_kernel_independent_packed_oracle(alias, dtype, path, monkeypatch):
    qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE[alias]
    if path == "native" and (qtype in _GGML_MATMUL_SKIP_QTYPES or _llama_cpp_lib is None):
        pytest.skip("native ggml matmul unavailable for this format")
    if not _quantizer_available(qtype):
        pytest.skip(
            f"no quantizer for {qtype} on this host: `gguf` is "
            f"{'installed' if _GGUF_AVAILABLE else 'missing'}, the numpy fallback "
            "does not implement it, and llama.cpp's ggml runtime is "
            f"{'available' if _llama_cpp_lib is not None else 'unavailable'}"
        )
    block = _GGUF_TYPE_INFO[qtype]["block_size"]
    width = block * 2 - 3
    lin, _ = _lin_and_x(width)
    module = GGUFTorchLinear(
        bits=alias, group_size=-1, sym=True, desc_act=False,
        in_features=width, out_features=16,
    ).eval()
    module.pack_original(lin, None, None)
    packed = module.qweight.numpy()
    weight = _reference_dequantize(packed, qtype)[:, :width]
    torch.testing.assert_close(module.dequantize_weight(dtype=dtype).T, weight.to(dtype), rtol=1e-6, atol=1e-6)
    # A power-of-two Q8 activation scale separates kernel arithmetic from
    # ggml's additional activation-quantization error (tested above).
    generator = torch.Generator().manual_seed(72)
    x = torch.randint(-126, 127, (4, width), generator=generator).float()
    x[:, ::32] = 127
    x = (x / 128).to(dtype)
    # Every path computes the exact FP32 product and rounds once, so compare
    # against that single rounding. Rounding the reference weight to the
    # activation dtype first (the old ``dense`` expectation) double-rounds:
    # for wide block formats it differs from the kernel by more than the
    # 2e-3 tolerance even though both are exact.
    expected = (x.float() @ weight.T).to(dtype)
    if path == "native":
        module = _cpp_from_torch(module)
        configure_kernel(KernelConfig(gguf_cpp_native_matmul=True))
    elif path == "dense":
        monkeypatch.setattr(module, "_should_use_fused_k_forward", lambda _: False)
    else:
        monkeypatch.setattr(module, "_should_use_fused_k_forward", lambda _: True)
    for _ in range(2):
        result = module(x)
        assert result.dtype == dtype
        # A single visible rounding step of the exact product is the tightest
        # device-independent bound: the fused BF16/FP16 intermediate the dense
        # path used before this assertion cost up to one extra output ulp. FP32
        # output has no such cast, so it keeps the strict kernel tolerance.
        if dtype == torch.float32:
            torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)
        else:
            torch.testing.assert_close(
                result.float(),
                expected.float(),
                rtol=0,
                atol=_visible_one_ulp(dtype, expected),
            )
        assert module.qweight.dtype == torch.uint8
        assert set(module._buffers) == {"qweight"}
        assert module._parameters == {}
        pending = list(vars(module).values())
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, (tuple, list)):
                pending.extend(item)
            elif isinstance(item, (torch.Tensor, np.ndarray)):
                assert item is module.qweight


def test_native_bridge_initialized_once_across_threads(monkeypatch):
    calls = []
    sentinel = object()
    barrier = threading.Barrier(4)

    def initialize():
        calls.append(1)
        time.sleep(0.01)
        return sentinel

    def get_bridge(_):
        barrier.wait(timeout=5)
        return gguf_cpp._get_ggml_bridge()

    monkeypatch.setattr(gguf_cpp, "_GGML_BRIDGE", None)
    monkeypatch.setattr(gguf_cpp, "_GGMLBridge", initialize)
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(result is sentinel for result in executor.map(get_bridge, range(4)))
    assert len(calls) == 1


@pytest.mark.parametrize("alias", ALL_ALIASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_random_activations_match_independent_a8_oracle(alias, dtype):
    qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE[alias]
    if qtype in _GGML_MATMUL_SKIP_QTYPES or _llama_cpp_lib is None:
        pytest.skip("native ggml matmul unavailable for this format")
    block = _GGUF_TYPE_INFO[qtype]["block_size"]
    lin, x = _lin_and_x(block)
    x = x.to(dtype)
    module = GGUFTorchLinear(
        bits=alias, group_size=-1, sym=True, desc_act=False,
        in_features=block, out_features=16,
    )
    module.pack_original(lin, None, None)
    weight = _reference_dequantize(module.qweight.numpy(), qtype)
    # ggml pairs K/IQ/TQ weights with Q8_K activations, other formats with
    # Q8_0 or Q8_1. The latter store an FP16 activation scale.
    activation_block = 256 if block == 256 else 32
    blocks = x.float().reshape(-1, activation_block)
    if activation_block == 256:
        peak = blocks.gather(1, blocks.abs().argmax(1, keepdim=True))
        inverse = torch.where(peak == 0, 0, torch.div(torch.full_like(peak, -127), peak))
        scale = torch.where(inverse == 0, 0, inverse.reciprocal())
    else:
        scale = blocks.abs().amax(1, keepdim=True) / 127
        inverse = torch.where(scale == 0, 0, scale.reciprocal())
    codes = (blocks * inverse).round().clamp(-127, 127)
    if activation_block == 32:
        scale = scale.half().float()
    activation = (codes * scale).reshape_as(x)
    expected = activation @ weight.T
    result = _cpp_from_torch(module)._forward_ggml_cpu(x)
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)


def test_native_cpu_backend_serializes_shared_scratch(monkeypatch):
    if _llama_cpp_lib is None:
        pytest.skip("llama-cpp-python not installed")
    bridge = _get_ggml_bridge()
    original = bridge._run_quantized_matmul
    active = 0
    max_active = 0
    lock = threading.Lock()
    barrier = threading.Barrier(4)
    packed = _quantize_gguf_tensor_numpy(np.ones((16, 32), np.float32), "Q4_0")
    weight = _reference_dequantize(packed, "Q4_0")
    qweight = torch.from_numpy(packed)

    def run(**kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.01)
            return original(**kwargs)
        finally:
            with lock:
                active -= 1

    def forward(i):
        x = torch.full((1, 32), 127 / 128 * (i + 1))
        barrier.wait(timeout=5)
        output = bridge.quantized_matmul_cpu(
            qweight=qweight, x=x, gguf_tensor_qtype="Q4_0",
            padded_in_features=32, out_features=16,
        )
        torch.testing.assert_close(output, x @ weight.T, rtol=2e-3, atol=2e-3)

    monkeypatch.setattr(bridge, "_run_quantized_matmul", run)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(forward, range(4)))
    assert max_active == 1
