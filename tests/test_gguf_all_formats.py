# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Full-format GGUF coverage: pack -> dequantize -> linear forward.

Every alias in ``_GGUF_BITS_ALIAS_TO_TENSOR_QTYPE`` must quantize a weight
matrix, dequantize back to float, and run through both the torch and the
native ggml CPU kernels without leaving the quantized representation.
"""

from __future__ import annotations

import ctypes
import threading
import time
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
    _llama_cpp_lib,
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


@pytest.mark.parametrize("alias", ALL_ALIASES)
def test_gguf_quantize_dequantize_roundtrip(alias):
    """Pack every alias and verify dequant stays inside its precision budget."""
    qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE[alias]
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


@pytest.mark.parametrize("alias", ["q8_1", "q8_k"])
@pytest.mark.parametrize("override", ["1", "0", None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpp_unsupported_native_types_always_fall_back(alias, override, dtype, monkeypatch):
    if override is None:
        monkeypatch.delenv("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL", raising=False)
    else:
        monkeypatch.setenv("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL", override)
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
    for _ in range(2):
        torch.testing.assert_close(cpp(x), expected, rtol=2e-3, atol=2e-3)
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
    if path == "native":
        module = _cpp_from_torch(module)
        monkeypatch.setenv("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL", "1")
        expected = (x.float() @ weight.T).to(dtype)
    elif path == "dense":
        monkeypatch.setattr(module, "_should_use_fused_k_forward", lambda _: False)
        expected = x @ weight.to(dtype).T
    else:
        monkeypatch.setattr(module, "_should_use_fused_k_forward", lambda _: True)
        expected = (x.float() @ weight.T).to(dtype)
    for _ in range(2):
        result = module(x)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)
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
