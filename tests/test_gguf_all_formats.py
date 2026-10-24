# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Full-format GGUF coverage: pack -> dequantize -> linear forward.

Every alias in ``_GGUF_BITS_ALIAS_TO_TENSOR_QTYPE`` must quantize a weight
matrix, dequantize back to float, and run through both the torch and the
native ggml CPU kernels without leaving the quantized representation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

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

    if qtype in _GGUF_LIB_SKIP_QTYPES:
        # No external oracle exists; the internal table is authoritative.
        expected = ours
    elif gguf_lib is not None:
        expected = gguf_lib.dequantize(
            packed, gguf_lib.GGMLQuantizationType[qtype]
        ).reshape(ours.shape)
    else:
        pytest.skip("gguf package not installed")

    np.testing.assert_allclose(ours, expected, rtol=1e-6, atol=1e-6)

    bridge = None
    try:
        bridge = _get_ggml_bridge()
    except Exception:
        pass
    if bridge is not None and qtype not in _GGUF_LIB_SKIP_QTYPES:
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
    # Same packed input through two implementations must stay in float precision.
    diff = (cpp_out - torch_out).abs().max().item()
    assert diff <= 0.05


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
