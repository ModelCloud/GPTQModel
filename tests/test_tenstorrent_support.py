# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts and optional hardware tests for Tenstorrent GPTQ inference."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.tt_xla import XLALinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.tt_packing import (
    dequantize_tt_packed_4bit,
    repack_gptq_for_tt,
    unpack_tt_byte_planes_4bit,
    unpack_tt_word_4bit,
)


BOUNDARY_WORDS = [4294967295, 305419896, 2141965807, 2147483648]


def _signed_words(values):
    return torch.tensor(values, dtype=torch.uint32).view(torch.int32)


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Independent checkpoint packer for [out, in] codes, low nibble first."""
    out_features, in_features = codes.shape
    assert in_features % 8 == 0
    words = torch.zeros((out_features, in_features // 8), dtype=torch.int64)
    for lane in range(8):
        words |= codes[:, lane::8].to(torch.int64) << (lane * 4)
    return words.transpose(0, 1).to(torch.int32).contiguous()


def _pack_output_nibbles(values: torch.Tensor) -> torch.Tensor:
    """Pack [groups, out] zero points into checkpoint orientation."""
    groups, out_features = values.shape
    words = torch.zeros((groups, out_features // 8), dtype=torch.int64)
    for lane in range(8):
        words |= values[:, lane::8].to(torch.int64) << (lane * 4)
    return words.to(torch.int32).contiguous()


def _fixture(*, qzero_format=2, bias=False, in_features=256, out_features=128):
    torch.manual_seed(17)
    group_size = 128
    groups = in_features // group_size
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.int32)
    # GPTQ-v1 stores the logical zero point directly. GPTQ-v2 stores
    # (zero_point + 1), so a v2 stored nibble zero decodes to logical -1.
    raw_zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    logical_zeros = raw_zeros - 1 if qzero_format == 2 else raw_zeros
    # Use scales representative of GPTQ checkpoints. Very large synthetic
    # scales push the output into a BF16 binade where a single BF16 output ULP
    # exceeds the repository's visible-output tolerance.
    scales = (torch.rand((groups, out_features), dtype=torch.float32) * 0.004 + 0.001).to(torch.bfloat16)
    g_idx = (torch.arange(in_features, dtype=torch.int32) // group_size).contiguous()
    fixture = {
        "qweight": _pack_nibbles(codes),
        "qzeros": _pack_output_nibbles(raw_zeros),
        "scales": scales,
        "g_idx": g_idx,
        "bias": (torch.randn(out_features, dtype=torch.float32) * 0.01).to(torch.bfloat16) if bias else None,
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "codes": codes,
        "logical_zeros": logical_zeros,
        "format": FORMAT.GPTQ_V2 if qzero_format == 2 else FORMAT.GPTQ,
        "qzero_format": qzero_format,
    }
    return fixture


def _layout(fixture, *, device=None):
    return repack_gptq_for_tt(
        qweight=fixture["qweight"],
        qzeros=fixture["qzeros"],
        scales=fixture["scales"],
        g_idx=fixture["g_idx"],
        bias=fixture["bias"],
        in_features=fixture["in_features"],
        out_features=fixture["out_features"],
        group_size=fixture["group_size"],
        bits=4,
        pack_dtype=torch.int32,
        sym=True,
        planar=False,
        qzero_format=fixture["qzero_format"],
        device=device,
    )


@pytest.mark.parametrize("word", BOUNDARY_WORDS)
def test_byte_plane_unpack_is_bit_exact_for_high_word(word):
    words = _signed_words([word])
    words_u32 = words.view(torch.uint32)
    layout = _layout(_fixture())
    packed = torch.tensor(
        words_u32.numpy().view("uint8").reshape(1, 1, 4).copy(),
    )
    expected = torch.tensor(
        [(word >> shift) & 0xF for shift in range(0, 32, 4)],
        dtype=torch.int64,
    ).reshape(1, 8)
    assert torch.equal(unpack_tt_byte_planes_4bit(packed), expected)
    assert torch.equal(unpack_tt_word_4bit(words_u32), expected)
    # Retain one real layout construction so this test also exercises the
    # checkpoint-to-TT transpose and persistent packed representation.
    assert layout.weight_bytes.shape == (128, 32, 4)


@pytest.mark.parametrize("qzero_format", [1, 2])
def test_gptq_v1_v2_zero_points_decode_to_same_logical_values(qzero_format):
    fixture = _fixture(qzero_format=qzero_format)
    expected_zeros = fixture["logical_zeros"]
    layout = _layout(fixture)
    zeros = unpack_tt_byte_planes_4bit(layout.zero_bytes)[:, : fixture["out_features"]]
    zeros = torch.where(
        layout.zero_neg_one.bool()[:, : fixture["out_features"]],
        torch.full_like(zeros, -1, dtype=torch.int32),
        zeros.to(torch.int32),
    )
    assert torch.equal(zeros, expected_zeros)


def test_packed_transpose_preserves_checkpoint_codes():
    fixture = _fixture()
    layout = _layout(fixture)
    codes = unpack_tt_byte_planes_4bit(layout.weight_bytes)[:, : fixture["in_features"]]
    assert torch.equal(codes, fixture["codes"])


def test_cpu_dequant_matches_independent_formula():
    fixture = _fixture(bias=True)
    layout = _layout(fixture)
    actual = dequantize_tt_packed_4bit(layout)
    scales = fixture["scales"].float()
    transforms = layout.g_idx.long()
    group_zeros = fixture["logical_zeros"][transforms]
    expected = (fixture["codes"].T.float() - group_zeros.float()) * scales[transforms]
    assert torch.equal(actual, expected)
    assert layout.scales.dtype == torch.bfloat16
    assert layout.bias.dtype == torch.bfloat16


def test_layout_rejects_contract_violations():
    fixture = _fixture()
    with pytest.raises(NotImplementedError, match="symmetric"):
        repack_gptq_for_tt(
            qweight=fixture["qweight"], qzeros=fixture["qzeros"],
            scales=fixture["scales"], g_idx=fixture["g_idx"], bias=None,
            in_features=256, out_features=128, group_size=128, bits=4,
            pack_dtype=torch.int32, sym=False, planar=False, qzero_format=2,
        )
    with pytest.raises(NotImplementedError, match="int32"):
        repack_gptq_for_tt(
            qweight=fixture["qweight"], qzeros=fixture["qzeros"],
            scales=fixture["scales"], g_idx=fixture["g_idx"], bias=None,
            in_features=256, out_features=128, group_size=128, bits=4,
            pack_dtype=torch.int16, sym=True, planar=False, qzero_format=2,
        )
    with pytest.raises(ValueError, match="divisible by 8"):
        repack_gptq_for_tt(
            qweight=fixture["qweight"], qzeros=fixture["qzeros"],
            scales=fixture["scales"], g_idx=fixture["g_idx"], bias=None,
            in_features=252, out_features=128, group_size=128, bits=4,
            pack_dtype=torch.int32, sym=True, planar=False, qzero_format=2,
        )


def test_tt_dispatch_is_explicit_and_fail_closed(monkeypatch):
    assert importer.get_kernel_for_backend(
        BACKEND.GPTQ_TT_XLA, fmt=FORMAT.GPTQ, quant_method=METHOD.GPTQ
    ) is XLALinear

    class _Runtime:
        @staticmethod
        def addressable_device_count():
            return 4

    tt_xla_module = importlib.import_module(XLALinear.__module__)
    monkeypatch.setattr(
        tt_xla_module,
        "require_xla_runtime",
        lambda: type("TorchXLA", (), {"runtime": _Runtime})(),
    )
    assert XLALinear.validate_once() == (True, None)
    ok, error = XLALinear._validate(
        bits=4, group_size=128, desc_act=False, sym=True,
        in_features=256, out_features=128, pack_dtype=torch.int32,
        dtype=torch.bfloat16, device=DEVICE.XLA,
    )
    assert ok, error
    for kwargs in (
        {"device": DEVICE.CUDA},
        {"sym": False},
        {"bits": 8},
        {"dtype": torch.float16},
        {"pack_dtype": torch.int16},
        {"in_features": 252},
    ):
        args = {
            "bits": 4, "group_size": 128, "desc_act": False, "sym": True,
            "in_features": 256, "out_features": 128, "pack_dtype": torch.int32,
            "dtype": torch.bfloat16, "device": DEVICE.XLA,
        }
        args.update(kwargs)
        ok, _ = XLALinear._validate(**args)
        assert not ok, kwargs
    torch_kernel = importer.get_kernel_for_backend(
        BACKEND.GPTQ_TORCH, fmt=FORMAT.GPTQ, quant_method=METHOD.GPTQ
    )
    with pytest.raises(NotImplementedError, match="only supports"):
        torch_kernel.validate_device(DEVICE.XLA)


def test_plain_import_does_not_initialize_xla(monkeypatch):
    # gptqmodel's own imports may inspect package presence. The external
    # accelerate import can initialize torch_xla on TT hosts, so this test pins
    # only the package-local policy: module-level device resolution never probes.
    from gptqmodel.utils import torch as torch_utils

    monkeypatch.setattr(
        torch_utils,
        "require_xla_runtime",
        lambda: pytest.fail("plain imports must not initialize PJRT"),
    )
    assert torch_utils._xla_package_installed() == (
        importlib.util.find_spec("torch_xla") is not None
    )


def test_explicit_device_discovery_returns_all_xla_cards(monkeypatch):
    from gptqmodel.utils import torch as torch_utils

    class _Runtime:
        @staticmethod
        def addressable_device_count():
            return 4

    class _TorchXLA:
        runtime = _Runtime()

    monkeypatch.setattr(torch_utils, "HAS_CUDA", False)
    monkeypatch.setattr(torch_utils, "HAS_XPU", False)
    monkeypatch.setattr(torch_utils, "HAS_NPU", False)
    monkeypatch.setattr(torch_utils, "HAS_MPS", False)
    monkeypatch.setattr(torch_utils, "is_xla_available", lambda: True)
    monkeypatch.setattr(torch_utils, "require_xla_runtime", lambda: _TorchXLA)

    assert torch_utils.torch_devices() == [torch.device(f"xla:{index}") for index in range(4)]
    assert torch_utils.auto_select_torch_device(index=9) == torch.device("xla:0")


def test_ttxla_persistent_layout_and_invalidation(monkeypatch):
    fixture = _fixture(qzero_format=2)
    module = XLALinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=256, out_features=128, bias=False,
        pack_dtype=torch.int32, format=fixture["format"],
    )
    module.qweight.copy_(fixture["qweight"])
    module.qzeros.copy_(fixture["qzeros"])
    module.scales.copy_(fixture["scales"].float())
    module.g_idx.copy_(fixture["g_idx"])
    module.qzero_format(2)
    module.post_init()
    first = module._tt_layout
    assert first is not None
    assert module._build_tt_layout() is first
    assert first.weight_bytes.shape == (128, 32, 4)
    assert first.zero_bytes.shape == (2, 16, 4)
    assert module._tt_layout.scales.dtype == torch.bfloat16
    module.clear_weight_cache()
    assert module._tt_layout is None
    module._load_from_state_dict({}, "", {}, True, [], [], [])
    assert module._tt_layout is None


def test_tt_layout_tracks_device_moves_and_weight_updates():
    fixture = _fixture(qzero_format=2)
    module = XLALinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=256, out_features=128, bias=False,
        pack_dtype=torch.int32, format=fixture["format"],
    )
    module.qweight.copy_(fixture["qweight"])
    module.qzeros.copy_(fixture["qzeros"])
    module.scales.copy_(fixture["scales"].float())
    module.g_idx.copy_(fixture["g_idx"])
    module.qzero_format(2)
    module.post_init()
    assert module._tt_layout is not None
    assert module._tt_layout.weight_bytes.device.type == "cpu"
    module._apply(lambda tensor: tensor)
    assert module._tt_layout is None
    rebuilt = module._build_tt_layout()
    assert rebuilt.zero_neg_one.device.type == "cpu"
    module.scales.add_(1)
    assert module._build_tt_layout() is not rebuilt


def test_ttxla_forward_rejects_cpu_activation_fail_closed():
    fixture = _fixture(qzero_format=2)
    module = XLALinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=256, out_features=128, bias=False,
        pack_dtype=torch.int32, format=fixture["format"],
    )
    with pytest.raises(RuntimeError, match="XLA activation"):
        module.forward(torch.zeros(4, 256, dtype=torch.bfloat16))


def _xla_runtime_available():
    if importlib.util.find_spec("torch_xla") is None:
        return False
    try:
        import torch_xla
        return int(torch_xla.runtime.addressable_device_count()) >= 4
    except Exception:
        return False


def _bf16_ordered_bits(values: torch.Tensor) -> np.ndarray:
    """Map BF16 values to monotonic integer keys for ULP-distance checks."""
    bits = (
        values.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        .view(torch.int16).numpy().view(np.uint16)
    )
    magnitude = (bits & 0x7FFF).astype(np.int64)
    return np.where(bits & 0x8000, 0x7FFF - magnitude, 0x8000 + bits.astype(np.int64))


@pytest.mark.skipif(os.environ.get("GPTQMODEL_TEST_TT_XLA") != "1", reason="explicit hardware gate")
def test_actual_ttxla_forward_on_all_four_cards():
    import torch_xla

    assert _xla_runtime_available(), "requires 4 addressable Tenstorrent PJRT devices"

    fixture = _fixture(qzero_format=2, bias=True)
    oracle_layout = _layout(fixture)
    cpu_weights = dequantize_tt_packed_4bit(oracle_layout)
    torch.manual_seed(29)
    activation_cpu = torch.randn(4, fixture["in_features"], dtype=torch.float32).to(torch.bfloat16)
    # This is the visible BF16 contract: calculate independently in FP32, add
    # bias there, then round once at the output. The repository's numerical
    # gate separately covers packed-code exactness in the CPU oracle tests.
    expected = (
        activation_cpu.float() @ cpu_weights.to(torch.bfloat16).float() + fixture["bias"].float()
    ).to(torch.bfloat16).float()

    for ordinal in range(4):
        device = torch.device(f"xla:{ordinal}")
        module = XLALinear(
            bits=4, group_size=128, sym=True, desc_act=False,
            in_features=256, out_features=128, bias=True,
            pack_dtype=torch.int32, format=fixture["format"],
        )
        module.qweight.copy_(fixture["qweight"])
        module.qzeros.copy_(fixture["qzeros"])
        module.scales.copy_(fixture["scales"].float())
        module.g_idx.copy_(fixture["g_idx"])
        module.bias.copy_(fixture["bias"])
        module.qzero_format(2)
        module.post_init()
        module.eval()
        module.to(device)
        device_weights = module.dequantize_weight()
        torch_xla.sync(wait=True)
        assert torch.equal(device_weights.cpu(), cpu_weights.to(torch.bfloat16))
        output = module(activation_cpu.to(device))
        torch_xla.sync(wait=True)
        actual = output.cpu().float()
        pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1]
        normalized_error = (actual - expected).abs().sum() / expected.abs().sum()
        assert pcc >= 0.999
        assert normalized_error <= 2e-3
        # Near zero, BF16 ULP distance is dominated by cancellation and the
        # absolute tolerance is the relevant contract. At material magnitudes,
        # TT's BF16 GEMM accumulation can round to the adjacent BF16 value from
        # Torch's CPU reduction order. Packed codes and dequantized weights were
        # already asserted bit-exact above.
        ulp_distance = np.abs(_bf16_ordered_bits(output) - _bf16_ordered_bits(expected))
        material = expected.abs() >= 0.05
        cancellation = ~material
        assert torch.allclose(
            actual[cancellation], expected[cancellation], rtol=2e-3, atol=2e-3,
        )
        assert int(ulp_distance[material.numpy()].max()) <= 1
        assert module._tt_layout is not None
        assert module._tt_layout.weight_bytes.device == device
        assert module._tt_layout.scales.dtype == torch.bfloat16
        assert module._tt_layout.bias.dtype == torch.bfloat16
        first_layout = module._tt_layout
        assert module._build_tt_layout() is first_layout
        del module
        torch_xla.sync(wait=True)


def test_forward_never_retains_expanded_weights(monkeypatch):
    fixture = _fixture(qzero_format=2)
    module = XLALinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=256, out_features=128, bias=False,
        pack_dtype=torch.int32, format=fixture["format"],
    )
    module.qweight.copy_(fixture["qweight"])
    module.qzeros.copy_(fixture["qzeros"])
    module.scales.copy_(fixture["scales"].float())
    module.g_idx.copy_(fixture["g_idx"])
    module.qzero_format(2)
    module.post_init()
    # The only persistent derived state is the packed byte layout. Forward's
    # dense/expanded values are locals; assert no other tensor-valued state.
    tensor_attrs = {
        name: value for name, value in vars(module).items()
        if isinstance(value, torch.Tensor) and name not in {"qweight", "qzeros", "scales", "g_idx", "bias"}
    }
    assert all(not name.startswith(("weight_", "dequant", "codes")) for name in tensor_attrs)
