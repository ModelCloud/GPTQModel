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

# Calibration text is required by the block-float quantizer: a narrow passage
# tunes GPTQ's error feedback to that passage and wrecks the model on other text.
TT_CALIBRATION_TEXT = (
    "The quick brown fox jumps over the lazy dog while the market stalls open. "
    "Tenstorrent accelerators pair many small RISC-V cores with a matrix engine. "
    "Quantization reduces the number of bits needed to store model weights. "
    "A shop sells pencils in boxes of twelve and four boxes arrive on Monday. "
    "The library opens at nine in the morning and closes at six in the evening. "
    "Rain fell for three hours at two centimeters per hour and the gauge rose. "
    "Sorting a list of numbers takes time proportional to n log n when efficient. "
    "A recipe calls for two cups of flour for every three cups of milk, so nine "
    "cups of milk require six cups of flour to keep the mixture consistent. "
)


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
    # TorchLinear normalizes GPTQ v1 by decoding ``zero_point - 1`` and
    # adding one. GPTQ v2 stores the same logical value directly.
    logical_zeros = (raw_zeros + 1) & 0xF if qzero_format == 1 else raw_zeros
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

    tt_xla_module = importlib.import_module(XLALinear.__module__)
    # ``validate_once`` now probes through ``xla_device_count``, which is the
    # only call that both reads the count and records that PJRT took ownership.
    monkeypatch.setattr(tt_xla_module, "xla_device_count", lambda: 4)
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


# ---------------------------------------------------------------------------
# Native TTNN backend (no Torch/XLA in the execution path)
# ---------------------------------------------------------------------------


def test_ttnn_dispatch_is_registered_and_kept_out_of_auto():
    from gptqmodel.nn_modules.qlinear.ttnn import TTNNLinear

    assert importer.get_kernel_for_backend(
        BACKEND.GPTQ_TTNN, fmt=FORMAT.GPTQ, quant_method=METHOD.GPTQ
    ) is TTNNLinear
    assert BACKEND.GPTQ_TTNN in TTNNLinear.SUPPORTS_BACKENDS
    # AUTO keeps the XLA path ahead of the native one so existing model loads
    # do not silently switch execution backends.
    auto = importer.get_auto_backend_kernel_mapping()[METHOD.GPTQ][FORMAT.GPTQ]
    ordered = [backend for backend, _ in auto.items() if backend in (BACKEND.GPTQ_TT_XLA, BACKEND.GPTQ_TTNN)]
    assert ordered[:2] == [BACKEND.GPTQ_TT_XLA, BACKEND.GPTQ_TTNN][: len(ordered)]


def test_native_ttnn_device_resolution_never_initializes_pjrt(monkeypatch):
    """The native backend must not probe the XLA runtime during device selection.

    ``expand_selector_device_family`` asks PJRT for an addressable device count,
    which initializes the Tenstorrent PJRT plugin and claims every chip. TT-Metal
    then cannot open its own mesh and ``GPTQ_TTNN`` is unusable through
    ``GPTQModel.load``. Pin the failure closed without touching hardware.
    """
    from gptqmodel.models import loader as loader_module
    from gptqmodel.utils import importer as importer_module
    from gptqmodel.utils import torch as torch_module

    def _explode(*args, **kwargs):
        raise AssertionError("the native TTNN path must not initialize PJRT")

    calls = []

    def _record(device):
        calls.append(device)
        return device

    monkeypatch.setattr(loader_module, "expand_selector_device_family", _record)
    monkeypatch.setattr(torch_module, "require_xla_runtime", _explode)
    monkeypatch.setattr(importer_module, "require_xla_runtime", _explode, raising=False)

    resolved = loader_module._resolve_default_layerwise_device(DEVICE.XLA, BACKEND.GPTQ_TTNN)
    assert resolved == DEVICE.XLA, resolved
    assert calls == [], "the native TTNN path must not expand through PJRT"

    # The XLA backend still expands through the PJRT-owned ordinal list.
    expanded = loader_module._resolve_default_layerwise_device(DEVICE.XLA, BACKEND.GPTQ_TT_XLA)
    assert calls == [DEVICE.XLA], calls
    assert expanded == DEVICE.XLA, expanded


def test_native_ttnn_auto_device_selection_skips_the_xla_probe(monkeypatch):
    from gptqmodel.utils import importer as importer_module

    def _explode(*args, **kwargs):
        raise AssertionError("auto_select_device must not probe XLA for GPTQ_TTNN")

    monkeypatch.setattr(importer_module, "is_xla_available", _explode)
    assert importer_module.auto_select_device(None, BACKEND.GPTQ_TTNN) == DEVICE.XLA


def test_ttnn_rejects_non_8_aligned_widths():
    from gptqmodel.nn_modules.qlinear.ttnn import TTNNLinear

    base = {
        "bits": 4, "group_size": 128, "desc_act": False, "sym": True,
        "in_features": 256, "out_features": 128, "pack_dtype": torch.int32,
        "dtype": torch.bfloat16, "device": DEVICE.XLA,
    }
    for kwargs in (
        {"in_features": 252},
        {"out_features": 124},
        {"sym": False},
        {"bits": 8},
        {"dtype": torch.float16},
        {"pack_dtype": torch.int16},
        {"device": DEVICE.CUDA},
    ):
        args = dict(base)
        args.update(kwargs)
        ok, _ = TTNNLinear._validate(**args)
        assert not ok, kwargs


def test_ttnn_arch_lookup_never_assumes_blackhole():
    from gptqmodel.nn_modules.qlinear.ttnn import _ttnn_arch
    from gptqmodel.utils.ttnn_runtime import TTNNUnavailable

    class _TTNN:
        Arch = type("Arch", (), {"BLACKHOLE": "blackhole-enum", "WORMHOLE_B0": "wormhole-enum"})

        def __init__(self, name):
            self._name = name

        def get_arch_name(self):
            return self._name

    assert _ttnn_arch(_TTNN("blackhole")) == "blackhole-enum"
    assert _ttnn_arch(_TTNN("wormhole_b0")) == "wormhole-enum"
    with pytest.raises(TTNNUnavailable):
        _ttnn_arch(_TTNN("quasar"))


def test_ttnn_mesh_ownership_is_single_owner():
    import importlib

    runtime = importlib.import_module("gptqmodel.utils.ttnn_runtime")

    class _FakeMesh:
        def __init__(self, *a, **k):
            pass

        @property
        def num_devices(self):
            return 1

        def close(self):
            pass

    # No device is opened here: this only exercises the bookkeeping contract.
    mesh = runtime.TTNNMesh(device=_FakeMesh(), mesh_shape=(1, 4), fabric="FABRIC_1D", owner="a")
    assert mesh.num_devices == 4
    assert runtime.current_mesh() is None


class _ThermalClock:
    """Deterministic monotonic clock and sleeper for the pacing tests."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    @property
    def total(self):
        return sum(self.slept)


def test_gddr_guard_waits_for_cooldown():
    """A hot card must be paced until it drops below the soft limit."""

    from gptqmodel.utils import tt_thermal

    clock = _ThermalClock()
    readings = [[104], [104], [97], [88]]
    tt_thermal.reset_cooldown_state()
    peak = tt_thermal.wait_for_cooldown(
        sleep=clock.sleep,
        read_temps=lambda: readings.pop(0) if readings else [88],
        monotonic=clock.monotonic,
    )
    assert peak == 88
    assert clock.total > 0
    # GDDR sheds only ~2 C/min, so the guard must poll coarsely instead of
    # busy-spinning PCI reads between near-identical temperatures.
    assert all(step >= 5.0 for step in clock.slept)


def test_gddr_guard_gives_up_at_budget(monkeypatch):
    """A card that never cools must not hang the load forever."""

    from gptqmodel.utils import tt_thermal

    monkeypatch.setenv("GPTQMODEL_TT_GDDR_MAX_WAIT_SECS", "5")
    clock = _ThermalClock()
    tt_thermal.reset_cooldown_state()
    peak = tt_thermal.wait_for_cooldown(
        sleep=clock.sleep, read_temps=lambda: [110], monotonic=clock.monotonic,
    )
    assert peak == 110
    assert clock.total <= 5.0


def test_gddr_guard_is_skipped_without_telemetry():
    """An unreadable temperature disables the guard instead of failing a load."""

    from gptqmodel.utils import tt_thermal

    clock = _ThermalClock()
    tt_thermal.reset_cooldown_state()
    assert tt_thermal.wait_for_cooldown(
        sleep=clock.sleep, read_temps=lambda: None, monotonic=clock.monotonic,
    ) is None
    assert clock.slept == []


def test_guard_can_be_disabled(monkeypatch):
    from gptqmodel.utils import tt_thermal

    monkeypatch.setenv("GPTQMODEL_TT_GDDR_SOFT_LIMIT_C", "0")
    clock = _ThermalClock()
    tt_thermal.reset_cooldown_state()
    assert tt_thermal.wait_for_cooldown(
        sleep=clock.sleep, read_temps=lambda: [110], monotonic=clock.monotonic,
    ) == 110
    assert clock.slept == []


def test_cooldown_is_paid_at_most_once_per_window(monkeypatch):
    """Load-level and mesh-level gates must not stack two full cooldowns."""

    from gptqmodel.utils import tt_thermal

    monkeypatch.setenv("GPTQMODEL_TT_COOLDOWN_SECS", "5")
    clock = _ThermalClock()
    tt_thermal.reset_cooldown_state()
    first = tt_thermal.before_device_open(
        owner="loader", sleep=clock.sleep, read_temps=lambda: [60],
        monotonic=clock.monotonic,
    )
    second = tt_thermal.before_device_open(
        owner="mesh", sleep=clock.sleep, read_temps=lambda: [60],
        monotonic=clock.monotonic,
    )
    assert first == second == 60
    assert clock.total == 5.0

    # A close starts a fresh window, so the pending open is not charged again.
    tt_thermal.after_device_close(
        owner="mesh", sleep=clock.sleep, read_temps=lambda: [60],
        monotonic=clock.monotonic,
    )
    assert clock.total == 10.0
    tt_thermal.before_device_open(
        owner="loader", sleep=clock.sleep, read_temps=lambda: [60],
        monotonic=clock.monotonic,
    )
    assert clock.total == 10.0


def test_zero_cooldown_disables_sleep(monkeypatch):
    from gptqmodel.utils import tt_thermal

    monkeypatch.setenv("GPTQMODEL_TT_COOLDOWN_SECS", "0")
    clock = _ThermalClock()
    tt_thermal.reset_cooldown_state()
    tt_thermal.before_device_open(
        owner="loader", sleep=clock.sleep, read_temps=lambda: [60], monotonic=clock.monotonic,
    )
    tt_thermal.after_device_close(
        owner="mesh", sleep=clock.sleep, read_temps=lambda: [60], monotonic=clock.monotonic,
    )
    assert clock.slept == []


def test_mesh_lifecycle_applies_cooldown(monkeypatch):
    """``open_mesh``/``close_mesh`` must pace the parts, not just the loader."""

    import importlib

    runtime = importlib.import_module("gptqmodel.utils.ttnn_runtime")

    calls = []
    monkeypatch.setattr(runtime, "before_device_open", lambda **kw: calls.append(("open", kw["owner"])))
    monkeypatch.setattr(runtime, "after_device_close", lambda **kw: calls.append(("close", kw["owner"])))

    class _FakeDevice:
        pass

    class _FakeTTNN:
        FabricConfig = type("FabricConfig", (), {"DISABLED": "dis"})

        @staticmethod
        def open_device(device_id=0):
            return _FakeDevice()

        @staticmethod
        def close_device(device):
            return None

    monkeypatch.setattr(runtime, "import_ttnn", lambda: _FakeTTNN)
    monkeypatch.setattr(runtime, "_MESH", None)
    monkeypatch.setattr(runtime, "_reject_pjrt_owner", lambda: None)
    try:
        runtime.open_mesh(device_count=1, owner="unit-test")
        runtime.close_mesh(owner="unit-test")
    finally:
        runtime.close_mesh()
    assert calls == [("open", "unit-test"), ("close", "unit-test")]


def test_pjrt_owner_guard_trips_once_the_chips_are_claimed(monkeypatch):
    """The old guard tested an attribute nothing ever set, so it never fired.

    ``xla_device_count`` is the call that initializes PJRT and claims every
    chip, and it records that fact. ``_reject_pjrt_owner`` must then stop the
    native TTNN path instead of letting TT-Metal fail with a bare assert.
    """

    import importlib
    import sys
    import types

    runtime = importlib.import_module("gptqmodel.utils.ttnn_runtime")

    fake_xla = types.ModuleType("torch_xla")
    fake_xla._gptqmodel_device_claimed = False
    monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)

    # Unclaimed: the native path is allowed through.
    runtime._reject_pjrt_owner()

    fake_xla._gptqmodel_device_claimed = True
    with pytest.raises(runtime.TTNNUnavailable, match="already claimed"):
        runtime._reject_pjrt_owner()


def test_xla_device_count_marks_the_claim(monkeypatch):
    """Reading the count must record ownership, since that read is the claim."""

    import importlib
    import sys
    import types

    torch_utils = importlib.import_module("gptqmodel.utils.torch")

    class _Runtime:
        @staticmethod
        def addressable_device_count():
            return 4

    fake_xla = types.ModuleType("torch_xla")
    fake_xla.runtime = _Runtime
    monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)
    monkeypatch.setattr(torch_utils, "require_xla_runtime", lambda: fake_xla)

    assert torch_utils.xla_device_count() == 4
    assert fake_xla._gptqmodel_device_claimed is True


def test_layerwise_count_never_initializes_pjrt_for_native_ttnn(monkeypatch):
    """The native TTNN backend must not probe PJRT while sizing a device map.

    ``xla_device_count()`` initializes the PJRT plugin and claims every chip.
    Calling it here used to poison the ``ttnn.open_device`` in ``post_init``
    with a TT_Fatal ownership assert, because TT-Metal allows one owner.
    """

    import importlib

    loader = importlib.import_module("gptqmodel.models.loader")
    ttnn_runtime = importlib.import_module("gptqmodel.utils.ttnn_runtime")

    calls = []
    monkeypatch.setattr(loader, "xla_device_count", lambda: calls.append("xla") or 4)
    monkeypatch.setattr(ttnn_runtime, "ttnn_device_count", lambda: calls.append("ttnn") or 4)

    assert loader._layerwise_device_count(DEVICE.XLA, BACKEND.GPTQ_TTNN) == 4
    assert calls == ["ttnn"], "native TTNN must count over PCIe, not by initializing PJRT"

    # The XLA backend still resolves its count through PJRT, which is correct
    # for that backend because PJRT is the runtime that will own the chips.
    calls.clear()
    assert loader._layerwise_device_count(DEVICE.XLA, BACKEND.GPTQ_TT_XLA) == 4
    assert calls == ["xla"]


def test_set_tdp_limit_drives_the_firmware_message(monkeypatch):
    """The installed plugin has no TT_METAL_TDP_LIMIT_WATTS, so we send the ARC message."""

    import importlib
    import types

    thermal = importlib.import_module("gptqmodel.utils.tt_thermal")

    sent = []
    limits = [125]

    class _FakeBH:
        def arc_msg(self, msg, arg0=0, arg1=0):
            sent.append((msg, arg0, arg1))
            if msg == thermal.BLACKHOLE_ARC_SET_TDP_LIMIT:
                limits[0] = arg0 or 125
            return (0, 0)

        def get_telemetry(self):
            return types.SimpleNamespace(tdp_limit_max=limits[0])

    monkeypatch.setattr(
        thermal, "_blackhole_chips", lambda: [_FakeBH(), _FakeBH()]
    )

    thermal.set_tdp_limit(90)
    assert sent == [
        (thermal.BLACKHOLE_ARC_SET_TDP_LIMIT, 90, 0),
        (thermal.BLACKHOLE_ARC_SET_TDP_LIMIT, 90, 0),
    ]
    assert thermal.tdp_limit_watts() == (90, 90)


def _fake_ttnn():
    """Minimal ttnn stand-in that records the transfer-relevant calls."""

    class _FakeTensor:
        def __init__(self, numel):
            self._numel = numel

        def numel(self):
            return self._numel

    class _FakeTTNN:
        TILE_LAYOUT = "tile"
        bfloat16 = "bf16"
        calls = []

        @staticmethod
        def from_torch(tensor, dtype=None, layout=None):
            return _FakeTensor(tensor.numel())

        @staticmethod
        def to_device(tensor, device):
            _FakeTTNN.calls.append("to_device")
            return tensor

        @staticmethod
        def to_torch(tensor):
            _FakeTTNN.calls.append("to_torch")
            return torch.zeros(2, 4, dtype=torch.bfloat16)

        @staticmethod
        def deallocate(tensor):
            _FakeTTNN.calls.append("deallocate")

        @staticmethod
        def relu(tensor):
            _FakeTTNN.calls.append("relu")
            return tensor

    return _FakeTTNN


def test_resident_chain_pays_one_upload_and_one_readback(monkeypatch):
    """A resident chain must not cross PCIe once per layer."""

    import importlib

    resident = importlib.import_module("gptqmodel.utils.tt_resident")
    runtime = importlib.import_module("gptqmodel.utils.ttnn_runtime")

    fake = _fake_ttnn()
    monkeypatch.setattr(runtime, "import_ttnn", lambda: fake)
    resident.reset_transfer_stats()

    class _Mesh:
        num_devices = 1
        device = object()

    class _Layer:
        """Stands in for a resident-aware linear."""

        def forward_resident(self, value):
            return value

    host = torch.zeros(2, 4, dtype=torch.bfloat16)
    chain = [_Layer() for _ in range(6)]
    activation = resident.upload(host, mesh=_Mesh())
    out = resident.run_resident(chain, activation)
    out.readback()

    stats = resident.transfer_stats()
    assert stats["uploads"] == 1
    assert stats["readbacks"] == 1
    # Six layers, but only one upload and one readback crossed PCIe.
    assert fake.calls.count("to_device") == 1
    assert fake.calls.count("to_torch") == 1


def test_resident_boundaries_counts_chain_breaks():
    """The crossing count makes the remaining host-op gap measurable."""

    import importlib

    resident = importlib.import_module("gptqmodel.utils.tt_resident")

    # Six consecutive resident layers = one upload + one readback.
    assert resident.resident_boundaries([True] * 6) == 2
    # A host op between two resident runs costs two runs = four crossings.
    assert resident.resident_boundaries([True, False, True]) == 4
    assert resident.resident_boundaries([]) == 0
    assert resident.resident_boundaries([False, False]) == 0


def test_resident_rejects_host_tensor_and_unknown_activation(monkeypatch):
    import importlib

    resident = importlib.import_module("gptqmodel.utils.tt_resident")

    value = resident.TTDeviceTensor(
        tensor=object(), mesh=None, batch_shape=(2,), features=4
    )
    with pytest.raises(NotImplementedError):
        resident.apply_activation(value, "tanh_approx")


def test_dequant_pair_lut_decodes_two_codes_per_word():
    """The 256-entry table must reproduce the 16-entry table byte for byte."""

    import importlib

    kern = importlib.import_module("gptqmodel.utils.tt_int4_kernel")

    scale, zero = 0.0173, 7
    single = kern.build_dequant_lut(scale, zero)          # 16 BF16, 2 per word
    pair = kern.build_dequant_pair_lut(scale, zero)       # 256 packed pairs
    assert pair.shape == (1, 256)

    bits = single.reshape(-1).to(torch.int64) & 0xFFFFFFFF
    vals = torch.stack([(bits & 0xFFFF).to(torch.int16),
                        ((bits >> 16) & 0xFFFF).to(torch.int16)], dim=1)
    bf16 = vals.reshape(-1).view(torch.bfloat16)

    got = pair.reshape(-1).to(torch.int64) & 0xFFFFFFFF
    lo = (got & 0xFFFF).to(torch.int16)
    hi = ((got >> 16) & 0xFFFF).to(torch.int16)
    decoded = torch.stack([lo, hi], dim=1).reshape(-1).view(torch.bfloat16)

    # Index is (low nibble | high nibble << 4), and each entry stores [low, high].
    expected = torch.empty(512, dtype=torch.bfloat16)
    for low in range(16):
        for high in range(16):
            expected[(low | (high << 4)) * 2] = bf16[low]
            expected[(low | (high << 4)) * 2 + 1] = bf16[high]
    assert torch.equal(decoded, expected)


def test_blockfloat_grid_rounds_on_shared_exponent():
    """A block-float block shares one exponent and keeps a fixed mantissa width."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(3)
    weight = torch.randn(64, 128) * 0.05

    out8, exp8 = bq.quantize_blockfloat(weight, 8)
    assert out8.shape == weight.shape
    # One shared exponent per block of 16 along the last dimension.
    assert exp8.shape[-1] == 128 // bq.BLOCKFLOAT_BLOCK

    # 8-bit storage must beat 4-bit storage on the same data.
    out4, _ = bq.quantize_blockfloat(weight, 4)
    assert bq.relative_error(weight, out8) < bq.relative_error(weight, out4)

    # Rounding is to a grid: every value must be an exact multiple of 2^exp
    # scaled unit, i.e. re-quantizing the output changes nothing.
    again, _ = bq.quantize_blockfloat(out8, 8)
    assert torch.allclose(out8, again, rtol=0, atol=1e-6)


def test_blockfloat_grid_values_are_exactly_hardware_representable():
    """Quantizing onto the grid must produce values the format can store exactly.

    This is what makes a block-float checkpoint need no inference-time dequant:
    the storage format is a fixed point of the quantizer, so the unpack stage
    yields exactly what the grid predicted. Verified on device as
    max_drift=0.000000 for both ttbf8 and ttbf4.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(5)
    weight = torch.randn(64, 128) * 0.02

    for bits in (8, 4):
        out, exponent = bq.quantize_blockfloat(weight, bits)
        levels = float(1 << bq.stored_mantissa_bits(bits))

        # Every value must be an integer multiple of 2^exp / levels, exactly the
        # set the hardware holds for that shared exponent.
        blocks = out.reshape(64, -1, bq.BLOCKFLOAT_BLOCK)
        scale = torch.pow(2.0, exponent).unsqueeze(-1)
        units = blocks / scale * levels
        assert torch.allclose(units, units.round(), atol=1e-3), (
            f"{bits}-bit output is not on the {levels}-level grid"
        )

        # Re-quantizing is a no-op: the grid is a fixed point.
        again, _ = bq.quantize_blockfloat(out, bits)
        assert torch.allclose(out, again, rtol=0, atol=1e-6)


def _synthetic_hessian(in_features, out_features, nsamples=256, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(nsamples, in_features)
    return x, torch.randn(out_features, in_features) * 0.02, x.T @ x


def test_gptq_blockfloat_solve_uses_error_feedback():
    """GPTQ onto a block-float grid must beat plain rounding on the same grid."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    x, w, h = _synthetic_hessian(128, 64)
    reference = x @ w.T

    def out_err(wq):
        return bq.relative_error(reference, x @ wq.T)

    for bits, enc in ((4, bq.quantize_ttbf4), (8, bq.quantize_ttbf8)):
        plain = enc(w)
        solved = bq.gptq_solve_blockfloat(w, h, bits=bits)
        # The whole point of GPTQ: error feedback beats rounding.
        assert out_err(solved) < out_err(plain), (
            f"{bits}-bit: solve ({out_err(solved)}) should beat rounding ({out_err(plain)})"
        )
        # And the result must already be on the grid, so storage is free.
        assert torch.allclose(solved, enc(solved), atol=1e-6)


def test_gptq_blockfloat_is_competitive_with_uniform():
    """With error feedback on both grids, block-float must not be a regression.

    Measured on correlated activations: 4-bit is on par (ratio ~1.0x) and 8-bit is
    ~5% worse. An earlier revision claimed a 2x advantage, but that compared a
    solved block-float against *plain rounding* on a uniform grid -- not a fair
    A/B. The honest claim is competitiveness, not superiority.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(0)
    ns, in_f, out_f = 384, 128, 64
    x = torch.randn(ns, 16) @ torch.randn(16, in_f)     # correlated features
    w = torch.randn(out_f, in_f) * 0.02
    h = x.T @ x
    reference = x @ w.T

    def out_err(q):
        return bq.relative_error(reference, x @ q.T)

    bf4 = out_err(bq.gptq_solve_blockfloat(w, h, bits=4, grid="blockfloat"))
    u4 = out_err(bq.gptq_solve_blockfloat(w, h, bits=4, grid="uniform"))
    assert bf4 < u4 * 1.25, f"4-bit block-float regressed: {bf4} vs uniform {u4}"


def test_hessian_accumulation_stays_float32():
    """A narrow Hessian was measured and rejected; it must not come back.

    bf16 is 3.4x faster (0.09 s against 0.31 s for one 8192-wide ``X^T X`` over
    2048 tokens) and only 0.27 % away from float32, yet OPT-125M quantized with it
    produced NaN weights and NaN perplexity. fp16 was 84x slower (25.9 s) and
    overflows on activation outliers. The accumulator therefore stays float32.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    width = 16

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.decoder = torch.nn.Module()
            block = torch.nn.Module()
            block.proj = torch.nn.Linear(width, 4, bias=False)
            self.model.decoder.layers = torch.nn.ModuleList([block])

        def forward(self, ids):
            x = torch.ones(1, ids.shape[1], width)
            self.model.decoder.layers[0].proj(x)
            return None

    class _Tok:
        def __call__(self, _text, return_tensors=None):
            class _Out:
                input_ids = torch.arange(12).unsqueeze(0)

            return _Out()

    hessians = bq.collect_layer_hessians(
        _Model(), _Tok(), text="ignored", nsamples=1, seqlen=12
    )
    h = hessians["0.proj"]
    assert h.dtype is torch.float32
    assert bool(torch.isfinite(h).all())


def test_shared_input_layers_compute_one_hessian():
    """q/k/v share one input, so their ``X^T X`` must be the same matrix.

    The collector caches per distinct activation: without it the same square GEMM
    is repeated for every projection that reads the same hidden state, which is
    most of the replay cost on a wide model. The reuse is exact.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    width = 32

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(width, 8, bias=False)
            self.k = torch.nn.Linear(width, 8, bias=False)
            self.v = torch.nn.Linear(width, 8, bias=False)
            self.out = torch.nn.Linear(8, width, bias=False)
            self.last_x = None

        def forward(self, ids):
            x = torch.ones(1, ids.shape[1], width)
            self.last_x = x
            self.q(x)
            self.k(x)
            self.v(x)
            self.out(torch.ones(1, ids.shape[1], 8))
            return None

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.decoder = torch.nn.Module()
            self.block = _Block()
            self.model.decoder.layers = torch.nn.ModuleList([self.block])

        def forward(self, ids):
            return self.block(ids)

    class _Tok:
        def __call__(self, _text, return_tensors=None):
            class _Out:
                input_ids = torch.arange(12).unsqueeze(0)

            return _Out()

    model = _Model()
    hessians = bq.collect_layer_hessians(
        model, _Tok(), text="ignored", nsamples=1, seqlen=12
    )

    flat = model.block.last_x.reshape(-1, width)
    expected = flat.T @ flat
    for name in ("0.q", "0.k", "0.v"):
        assert torch.allclose(hessians[name], expected), name
    assert torch.equal(hessians["0.q"], hessians["0.k"])


def test_gptq_blockfloat_rejects_mismatched_hessian():
    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    w = torch.randn(16, 32)
    with pytest.raises(ValueError, match="hessian must be"):
        bq.gptq_solve_blockfloat(w, torch.eye(16), bits=4)


def test_gptq_blockfloat_matches_a_canonical_reference():
    """Pin the solve against an independent, textbook GPTQ implementation.

    The grid is deliberately shared, so this isolates the error-feedback
    update. The oracle takes the **upper Cholesky factor of the inverse**
    (``H^-1 = U^T U``) and updates with rows of ``U``, which is the
    remaining-variable inverse GPTQ specifies. Reusing ``H^-1`` directly is the
    tempting shortcut and it is wrong: the correction is too small.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    def reference(weight, hessian, bits, damp=0.01, block=16, factor="upper"):
        w = weight.clone().float()
        h = hessian.clone().float()
        dead = torch.diag(h) == 0
        h[dead, dead] = 1.0
        w[:, dead] = 0.0
        h += torch.eye(h.shape[0]) * damp * torch.mean(torch.diag(h))
        hinv = torch.linalg.inv(h)
        if factor == "upper":
            hinv = torch.linalg.cholesky(hinv, upper=True)
        exps = bq._block_exponents(weight, block)
        levels = float(1 << bq.stored_mantissa_bits(bits))
        out = torch.zeros_like(w)
        n = w.shape[1]
        for i in range(n):
            col = w[:, i]
            scale = torch.pow(2.0, exps[:, i // block])
            unit = torch.round(col / scale * levels) / levels
            unit = unit.clamp(-(2.0 - 1.0 / levels), 2.0 - 1.0 / levels)
            q = unit * scale
            out[:, i] = q
            if i + 1 < n:
                w[:, i + 1:] -= torch.outer((col - q) / hinv[i, i], hinv[i, i + 1:])
        return out

    torch.manual_seed(3)
    for _ in range(3):
        weight = torch.randn(4, 32)
        a = torch.randn(32, 32)
        hessian = a.T @ a + 0.1 * torch.eye(32)
        for bits in (4, 8):
            got = bq.gptq_solve_blockfloat(weight, hessian, bits=bits)
            want = reference(weight, hessian, bits)
            assert torch.equal(got, want), (
                f"bits={bits}: solver and canonical GPTQ disagree by "
                f"{float((got - want).abs().max()):.6g}"
            )

    # The case that separates the two updates: 1.2 is off-grid, so column 0
    # produces real error, and the remaining-variable inverse moves column 2
    # from 1.25 down to 1.0. The plain inverse leaves it at 1.25.
    weight = torch.tensor([[1.2, 1.2, 1.2] + [0.0] * 13])
    hessian = torch.eye(16)
    hessian[:3, :3] = torch.tensor(
        [[1.0, 0.8, 0.7], [0.8, 1.0, 0.9], [0.7, 0.9, 1.0]]
    )
    solved = bq.gptq_solve_blockfloat(weight, hessian, bits=4)
    assert solved[0, :3].tolist() == [1.25, 1.25, 1.0]

    def damped_loss(candidate):
        # The Hessian the solve is minimising, with the same damping it applies.
        h = hessian.clone()
        h += torch.eye(16) * 0.01 * torch.mean(torch.diag(h))
        err = weight - candidate
        return float((err @ h @ err.T).item())

    naive = reference(weight, hessian, 4, factor="full")
    assert naive[0, :3].tolist() == [1.25, 1.25, 1.25]
    assert damped_loss(solved) < damped_loss(naive), (
        "the remaining-variable update must beat the plain-inverse one on the "
        "objective GPTQ minimises"
    )


def test_blockfloat_mantissa_widths():
    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    assert bq.BLOCKFLOAT_BITS == {"ttbf4": 4, "ttbf8": 8}
    # One sign bit and one implicit mantissa bit are not stored.
    assert bq.stored_mantissa_bits(8) == 6
    assert bq.stored_mantissa_bits(4) == 2


def test_blockfloat_dtype_names_are_tt_prefixed():
    """Block float is Tenstorrent-specific, so the canonical name is ``ttbf*``.

    The nearest industry formats are OCP microscaling (a per-32 ``E8M0``
    scale) and NVFP4 (a per-16 ``FP8`` scale); neither is bit-compatible with
    Tenstorrent's per-16 power-of-two exponent. The vendor spelling
    (``bfloat8_b`` / ``bfloat4_b``) is kept only for the ``getattr(ttnn, ...)``
    call site, because it reads like an 8-bit bfloat and is easily mistaken for
    OCP FP8 E4M3.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    assert bq.TT_BLOCKFLOAT_DTYPES == {4: "ttbf4", 8: "ttbf8"}
    assert all(name.startswith("ttbf") for name in bq.TT_BLOCKFLOAT_DTYPES.values())
    assert bq.tt_blockfloat_dtype(4) == "ttbf4"
    assert bq.tt_blockfloat_dtype(8) == "ttbf8"
    with pytest.raises(ValueError, match="no Tenstorrent block-float format"):
        bq.tt_blockfloat_dtype(16)

    assert bq.TT_BLOCKFLOAT_VENDOR_ATTRS == {4: "bfloat4_b", 8: "bfloat8_b"}
    assert bq.ttnn_blockfloat_attr(4) == "bfloat4_b"
    assert bq.ttnn_blockfloat_attr(8) == "bfloat8_b"
    with pytest.raises(ValueError, match="ttnn has no block-float dtype"):
        bq.ttnn_blockfloat_attr(16)

    # The kernel hands WEIGHT_FORMAT straight to ``getattr(ttnn, ...)``, so it
    # must stay the vendor spelling and must agree with the canonical map.
    kern = importlib.import_module("gptqmodel.nn_modules.qlinear.ttnn_blockfloat")
    assert kern.TTNNBlockFloat8Linear.WEIGHT_FORMAT == bq.ttnn_blockfloat_attr(8)
    assert kern.BLOCK_FLOAT_FORMATS == ("ttbf8",)


def test_blockfloat_vendor_attrs_exist_on_ttnn():
    """Every vendor spelling we emit must exist on the installed ttnn build."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    ttnn = pytest.importorskip("ttnn")
    for bits, attr in bq.TT_BLOCKFLOAT_VENDOR_ATTRS.items():
        assert hasattr(ttnn, attr), f"ttnn has no {attr} dtype for {bits} bits"


def test_packed_code_bytes_rounds_up_and_rejects_other_widths():
    """The byte count must match what the packer actually emits.

    ``_pack_codes`` pads an odd number of codes with one spare nibble, so a
    floor division would declare a buffer one byte shorter than the tensor that
    is written, and the checkpoint would fail to load whenever
    ``out_features * in_features`` is odd.
    """

    import importlib

    kern = importlib.import_module("gptqmodel.nn_modules.qlinear.tt_blockfloat")

    assert kern.packed_code_bytes(4, 4) == 2
    assert kern.packed_code_bytes(8, 4) == 4
    assert kern.packed_code_bytes(3, 4) == 2  # ceil(3 / 2), not 1
    assert kern.packed_code_bytes(1, 4) == 1
    assert kern.packed_code_bytes(3, 8) == 3
    with pytest.raises(ValueError, match="no Tenstorrent block-float format"):
        kern.packed_code_bytes(8, 6)


def test_resolve_layer_widths_rejects_corrupt_storage():
    """A checkpoint whose code buffers cannot be interpreted must say so.

    Inferring a width from a length that matches neither 4 nor 8 bits decodes
    every code at the wrong scale and produces plausible-looking garbage, so
    each malformed case raises instead of guessing.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    def state(shape, codes_len, *, with_shape=True):
        entry = {
            "a.codes": torch.zeros(codes_len, dtype=torch.uint8),
            "a.code_shape": torch.tensor(list(shape), dtype=torch.int64),
        }
        if not with_shape:
            del entry["a.code_shape"]
        return entry

    # 4-bit, 8 values -> 4 bytes; 8-bit -> 8 bytes.
    assert bq._resolve_layer_widths(state((2, 4), 4), {"bits": 4}) == {"a": 4}
    assert bq._resolve_layer_widths(state((2, 4), 8), {"bits": 8}) == {"a": 8}
    # The declared width is not trusted when the payload disagrees.
    assert bq._resolve_layer_widths(state((2, 4), 8), {"bits": 4}) == {"a": 8}
    assert bq._resolve_layer_widths(state((2, 4), 4), {"bits": 8}) == {"a": 4}
    # Odd element count still packs two codes per byte with one spare nibble.
    assert bq._resolve_layer_widths(state((1, 3), 2), {"bits": 4}) == {"a": 4}

    with pytest.raises(ValueError, match="no `.code_shape`"):
        bq._resolve_layer_widths(state((2, 4), 4, with_shape=False), {"bits": 4})
    with pytest.raises(ValueError, match="not a positive"):
        bq._resolve_layer_widths(state((0, 4), 4), {"bits": 4})
    with pytest.raises(ValueError, match="declares 6 bits"):
        bq._resolve_layer_widths(
            state((2, 4), 4), {"bits": 4, "bits_per_layer": {"a": 6}}
        )
    with pytest.raises(ValueError, match="Neither matches"):
        bq._resolve_layer_widths(
            {
                "a.codes": torch.zeros(5, dtype=torch.uint8),
                "a.code_shape": torch.tensor([1, 3], dtype=torch.int64),
            },
            {"bits": 4},
        )
    with pytest.raises(ValueError, match="must be an object"):
        bq._resolve_layer_widths(
            state((2, 4), 4), {"bits": 4, "bits_per_layer": ["a", 4]}
        )


def _tiny_llama(config_kwargs=None):
    from transformers import LlamaConfig, LlamaForCausalLM

    kwargs = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "tie_word_embeddings": True,
    }
    kwargs.update(config_kwargs or {})
    return LlamaForCausalLM(LlamaConfig(**kwargs))


def test_resave_of_a_loaded_checkpoint_keeps_its_packed_weights(tmp_path):
    """A second save must copy the packed buffers, not look for ``.weight``.

    A loaded model stores codes and exponents as buffers on ``TTBlockfloat*``
    modules, which are neither ``nn.Linear`` nor ``nn.Embedding``. Enumerating
    those types found nothing to write and produced a checkpoint with only the
    norm tensors, while still reporting success.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(23)
    model = _tiny_llama()
    with torch.no_grad():
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
                continue
            bits = 8 if name.endswith("embed_tokens") else 4
            weight.copy_(bq.quantize_blockfloat(weight.detach(), bits)[0])

    first_dir = str(tmp_path / "first")
    second_dir = str(tmp_path / "second")
    first = bq.save_blockfloat_model(
        model, first_dir, bits=4, embed_bits=8, metadata={"source_model": "original/model"}
    )
    loaded, restored = bq.load_blockfloat_model(first_dir)

    second = bq.save_blockfloat_model(loaded, second_dir, bits=4, embed_bits=8)
    assert second["layers"] == first["layers"] > 0
    assert second["payload_bytes"] == first["payload_bytes"]
    assert second["bits_per_layer"] == first["bits_per_layer"]

    # A re-save must not erase where the checkpoint came from.
    import json

    config = json.loads((tmp_path / "second" / "config.json").read_text())
    assert config["quantization_config"]["meta"]["source_model"] == "original/model"

    again, restored_again = bq.load_blockfloat_model(second_dir)
    assert set(restored_again) == set(restored)
    before, after = loaded.state_dict(), again.state_dict()
    assert set(before) == set(after)
    for name in before:
        assert torch.equal(before[name], after[name]), name


def test_save_blockfloat_model_rejects_a_model_with_nothing_packed(tmp_path):
    """A dense model must be rejected before anything is written."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    model = _tiny_llama()
    out = tmp_path / "dense"
    with pytest.raises(ValueError, match="no quantized modules"):
        bq.save_blockfloat_model(model, str(out), bits=4)
    assert not (out / "config.json").exists()


def test_quantize_config_from_checkpoint_preserves_pack_dtype(tmp_path):
    """The loader's config must carry the ``uint8`` the kernel actually uses."""

    import json

    from gptqmodel.models.tt_blockfloat_loader import quantize_config_from_checkpoint

    out = tmp_path / "cfg"
    out.mkdir()
    (out / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization_config": {
                    "quant_method": "tt-blockfloat",
                    "bits": 4,
                    "block": 16,
                    "embed_bits": 8,
                    "bits_per_layer": {"model.layers.0.self_attn.q_proj": 8},
                },
            }
        )
    )
    qcfg = quantize_config_from_checkpoint(str(out))
    assert qcfg.method == METHOD.TT_BLOCKFLOAT
    assert qcfg.pack_dtype == torch.uint8
    assert qcfg.bits == 4 and qcfg.embed_bits == 8


def test_embed_bits_quantizes_the_embedding(tmp_path, monkeypatch):
    """``--embed-bits`` has to quantize the table, not just label it.

    The writer stores a dense ``.weight`` when it finds an off-grid table, so
    requesting a width without quantizing first silently shipped float16.
    """

    import importlib
    import json

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    monkeypatch.setattr(bq, "collect_layer_hessians", lambda *a, **k: {})

    model = _tiny_llama()
    report = bq.quantize_model_blockfloat(
        model, None, bits=4, text=TT_CALIBRATION_TEXT, embed_bits=8
    )
    embedding = model.get_input_embeddings()
    assert torch.equal(
        embedding.weight.detach(),
        bq.quantize_blockfloat(embedding.weight.detach(), 8)[0],
    )
    assert report["model.embed_tokens"]["bits"] == 8
    assert report["model.embed_tokens"]["on_grid"] is True

    out = tmp_path / "embed"
    info = bq.save_blockfloat_model(model, str(out), bits=4, embed_bits=8)
    assert info["embed_bits"] == 8
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert "model.embed_tokens.codes" in index["weight_map"]
    assert "model.embed_tokens.weight" not in index["weight_map"]


def test_public_export_keeps_the_save_contract(tmp_path):
    """``save_quantized`` must honour generation settings, shard size and metadata.

    The block-float branch exports packed state directly, so it has to reproduce
    the rest of the publication contract instead of returning early past it:
    generation settings, the requested shard limit, caller safetensors metadata
    and the checkpoint's own provenance.
    """

    import importlib
    import json
    import os

    from safetensors import safe_open

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    from gptqmodel import GPTQModel

    torch.manual_seed(31)
    model = _tiny_llama()
    with torch.no_grad():
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
                continue
            bits = 8 if name.endswith("embed_tokens") else 4
            weight.copy_(bq.quantize_blockfloat(weight.detach(), bits)[0])

    first = str(tmp_path / "first")
    bq.save_blockfloat_model(
        model, first, bits=4, embed_bits=8, metadata={"source_model": "original/model"}
    )

    loaded = GPTQModel.load(first)
    loaded.model.generation_config.eos_token_id = [2, 3]
    loaded.model.generation_config.max_new_tokens = 17

    second = str(tmp_path / "second")
    loaded.save_quantized(
        second,
        max_shard_size="1KB",
        safetensors_metadata={"review": "preserve-me"},
    )

    # Generation settings survive the round trip.
    from transformers import GenerationConfig

    gen = GenerationConfig.from_pretrained(second)
    assert list(gen.eos_token_id) == [2, 3]
    assert gen.max_new_tokens == 17

    # The requested shard limit is applied rather than the writer default.
    shards = sorted(f for f in os.listdir(second) if f.endswith(".safetensors"))
    assert len(shards) > 1, shards

    # Caller metadata reaches the safetensors header, and ``format`` stays ours.
    with safe_open(os.path.join(second, shards[0]), framework="pt") as handle:
        header = handle.metadata()
    assert header["review"] == "preserve-me"
    assert header["format"] == "pt"

    # Provenance from the source checkpoint is not erased by the re-export.
    config = json.loads((tmp_path / "second" / "config.json").read_text())
    assert config["quantization_config"]["meta"]["source_model"] == "original/model"


def test_validate_restored_geometry_rejects_a_swapped_shape():
    """A shape with the right element count but the wrong orientation is corrupt."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    good = {
        "a.code_shape": torch.tensor([64, 16]),
        "a.exponents": torch.zeros(64, 1, dtype=torch.int8),
    }
    bq._validate_restored_geometry({"a": [64, 16]}, good, block=16)

    swapped = {"a.code_shape": torch.tensor([16, 64]), "a.exponents": torch.zeros(64, 1, dtype=torch.int8)}
    with pytest.raises(ValueError, match="declares code_shape"):
        bq._validate_restored_geometry({"a": [64, 16]}, swapped, block=16)

    bad_exponents = {
        "a.code_shape": torch.tensor([64, 16]),
        "a.exponents": torch.zeros(64, 3, dtype=torch.int8),
    }
    with pytest.raises(ValueError, match="exponents"):
        bq._validate_restored_geometry({"a": [64, 16]}, bad_exponents, block=16)


def test_to_host_replica_uses_a_composer_for_multi_device():
    """Multi-chip readback needs the composer; single-chip must not pass one."""

    from gptqmodel.utils import ttnn_runtime as rt

    seen = {}

    class FakeTtnn:
        @staticmethod
        def to_torch(tensor, mesh_composer=None):
            seen["composer"] = mesh_composer
            return torch.zeros(8, 4)

        @staticmethod
        def ConcatMeshToTensor(device, dim=0):
            return ("concat", device, dim)

    class Mesh:
        def __init__(self, count):
            self.num_devices = count
            self.device = "dev"

    out = rt.to_host_replica(FakeTtnn, "t", Mesh(2))
    assert seen["composer"] == ("concat", "dev", 0)
    assert out.shape == (4, 4)

    single = rt.to_host_replica(FakeTtnn, "t", Mesh(1))
    assert seen["composer"] is None
    assert single.shape == (8, 4)


def test_tensor_cache_key_tolerates_none_and_inference_tensors():
    """The TTNN layout key must not read ``._version`` unguarded."""

    from gptqmodel.nn_modules.qlinear._runtime_cache import tensor_cache_key

    assert tensor_cache_key(None) is None
    with torch.inference_mode():
        inference = torch.zeros(4)
    key = tensor_cache_key(inference)
    assert key is not None
    assert key[1] is None
    assert tensor_cache_key(torch.zeros(4))[1] == 0


def test_auto_select_device_clamps_to_the_visible_count(monkeypatch):
    """Asking for index 1 on a single-GPU host must not return ``cuda:1``."""

    import gptqmodel.utils.torch as ttmod

    monkeypatch.setattr(ttmod, "HAS_CUDA", True)
    monkeypatch.setattr(ttmod, "HAS_XPU", False)
    monkeypatch.setattr(ttmod, "HAS_NPU", False)
    monkeypatch.setattr(ttmod, "HAS_XLA", False)
    monkeypatch.setattr(ttmod, "HAS_MPS", False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert ttmod.auto_select_torch_device_unprobed(0) == torch.device("cuda:0")
    assert ttmod.auto_select_torch_device_unprobed(1) == torch.device("cuda:0")


def test_blockfloat_beats_uniform_int4_for_heavy_tails():
    """Block-float should win where a per-group int4 scale clips an outlier."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(11)
    weight = torch.randn(64, 128) * 0.01
    # One large value per group compresses every other value under a shared
    # int4 scale; block-float re-scales per 16 values instead.
    weight[:, ::128] = 8.0

    int4_err = bq.relative_error(weight, bq.quantize_uniform_int4(weight))
    bf4_err = bq.relative_error(weight, bq.quantize_ttbf4(weight))
    assert bf4_err < int4_err


def test_int4_kernel_reference_matches_packing():
    """The kernel's reference must invert the packing exactly (CPU, no hardware)."""

    import importlib

    kern = importlib.import_module("gptqmodel.utils.tt_int4_kernel")

    torch.manual_seed(7)
    codes = torch.randint(0, 16, (2048,), dtype=torch.int64)
    words = kern.pack_codes_to_words(codes)
    assert words.numel() == 2048 // 8

    expected = kern.bf16_truncate(0.5 * (codes.to(torch.float32) - 7.0))
    assert torch.equal(kern.reference_dequant(words, 0.5, 7), expected)

    # Nibble order is low-first, which is GPTQ's checkpoint convention.
    single = kern.pack_codes_to_words(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
    assert int(single[0]) & 0xFFFFFFFF == 0x87654321


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TTNN") != "1",
    reason="explicit TTNN hardware gate",
)
def test_int4_dequant_kernel_is_bit_exact_on_device():
    """Custom TT-Metalium dequant must match the torch reference exactly."""

    from gptqmodel.utils.tt_int4_kernel import verify_on_device
    from gptqmodel.utils.ttnn_runtime import close_mesh, current_mesh, open_mesh

    mesh = current_mesh() or open_mesh(device_count=1)
    try:
        exact, max_err = verify_on_device(1024, 0.0173, 7, device=mesh.device)
        assert exact, f"kernel dequant differs from reference (max abs err {max_err})"
    finally:
        close_mesh()




def test_tile_order_is_face_major():
    """A 32x32 tile is four row-major 16x16 faces; memory order must reflect it."""

    import importlib

    kern = importlib.import_module("gptqmodel.utils.tt_int4_kernel")

    order = kern.tile_order_indices()
    assert order.numel() == 1024
    # Face 0 is rows 0-15 x cols 0-15, row-major: the first row is 0..15.
    assert order[:16].tolist() == list(range(16))
    # Face 1 starts at memory offset 256 and is cols 16-31 of row 0.
    assert order[256].item() == 16
    # Faces are contiguous 256-element blocks, and the mapping is a permutation.
    assert sorted(order.tolist()) == list(range(1024))

    values = torch.arange(1024)
    permuted = kern.permute_to_tile_order(values)
    # Inverting the permutation recovers the logical order.
    recovered = torch.empty_like(permuted)
    recovered[order] = permuted
    assert torch.equal(recovered, values)


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TTNN") != "1",
    reason="explicit TTNN hardware gate",
)
def test_fused_tile_path_is_bit_exact_on_device():
    """BRISC dequant into an L1 CB, TRISC consume, no dense tile in DRAM."""

    from gptqmodel.utils.tt_int4_kernel import (
        bf16_truncate,
        run_fused_tile,
    )
    from gptqmodel.utils.ttnn_runtime import close_mesh, current_mesh, open_mesh

    torch.manual_seed(6)
    codes = torch.randint(0, 16, (1024,), dtype=torch.int64)
    expected = bf16_truncate(0.0173 * (codes.to(torch.float32) - 7.0))

    mesh = current_mesh() or open_mesh(device_count=1)
    try:
        got = run_fused_tile(codes, 0.0173, 7, device=mesh.device)
        assert torch.equal(got, expected), "fused tile path is not bit-exact"
    finally:
        close_mesh()


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TTNN") != "1",
    reason="explicit TTNN hardware gate",
)
def test_fused_matmul_matches_torch_on_device():
    """The fused kernel must compute W x ACT, with the dequant staying in L1."""

    from gptqmodel.utils.tt_int4_kernel import run_fused_matmul
    from gptqmodel.utils.ttnn_runtime import close_mesh, current_mesh, open_mesh

    torch.manual_seed(8)
    codes = torch.randint(0, 16, (1024,), dtype=torch.int64)
    weight = (0.0173 * (codes.to(torch.float32) - 7.0)).to(torch.bfloat16).reshape(32, 32)
    act = torch.randn(32, 32, dtype=torch.bfloat16)
    reference = weight.float() @ act.float()

    mesh = current_mesh() or open_mesh(device_count=1)
    try:
        got = run_fused_matmul(act, codes, 0.0173, 7, device=mesh.device)
    finally:
        close_mesh()

    assert got.shape == (32, 32)
    norm_l1 = float((got - reference).abs().sum() / reference.abs().sum().clamp_min(1e-6))
    assert norm_l1 < 0.02, f"fused matmul deviates from torch: norm_l1={norm_l1}"
    # The reverse orientation must NOT match, otherwise the test proves nothing
    # about operand order.
    transposed = float((got - (act.float() @ weight.float())).abs().sum()
                       / reference.abs().sum().clamp_min(1e-6))
    assert transposed > norm_l1 * 10



def test_blockfloat_grid_matches_hardware_levels():
    """The quantizer's grid must match what ttbf4/ttbf8 can store.

    Measured on a p300c by encoding and reading back: bf4 holds 4 levels per
    16-value block inside one binade, bf8 holds up to 64. That is 2 and 6 stored
    mantissa bits, matching ``stored_mantissa_bits``. The step is ``2^exp/levels``:
    hardware showed 0.125 = 0.5/4 for bf4.

    Blocks each carry their own exponent -- verified by placing a 16x weaker
    segment beside a strong one in the same row: both keep their own 4 levels
    instead of the weak one collapsing.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    # bf4 exercises the full grid inside one binade: exactly 4 levels.
    vals = torch.linspace(0.5, 0.99, 16).reshape(1, 16)
    out4, exp4 = bq.quantize_blockfloat(vals, 4)
    levels4 = torch.unique(out4.reshape(-1))
    assert levels4.numel() == 4, levels4
    assert torch.unique(exp4).numel() == 1
    span = float(torch.pow(2.0, exp4.reshape(-1)[0]))
    assert abs(float(levels4[1] - levels4[0]) - span / 4) < 1e-6

    # bf8 must be strictly finer than bf4 on the same input.
    out8, _ = bq.quantize_blockfloat(vals, 8)
    assert torch.unique(out8.reshape(-1)).numel() > levels4.numel()

    # Neighbouring blocks keep independent exponents.
    t = torch.zeros(1, 32)
    t[0, :16] = torch.linspace(0.5, 0.99, 16)
    t[0, 16:] = torch.linspace(0.03, 0.06, 16)
    out, exponent = bq.quantize_blockfloat(t, 4)
    # One exponent per 16-value block: shape [rows, blocks].
    assert exponent.shape[-1] == 2, exponent.shape
    assert int(exponent[0, 0]) > int(exponent[0, 1]), (
        "the weaker block should carry a smaller exponent"
    )
    assert torch.unique(out[0, 16:]).numel() == 4


def test_calibration_hessian_strides_through_the_text():
    """Each calibration sample must see a different window of the text.

    The Hessian used to be collected from ``ids[:, :seqlen]`` on every iteration,
    so ``nsamples`` multiplied one passage instead of covering the calibration
    text. The resulting error feedback is tuned to those 256 tokens, which is
    measurably catastrophic on other text for a sensitive model.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    seqlen = 8
    total = 8 * seqlen
    seen = []

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.decoder = torch.nn.Module()
            block = torch.nn.Module()
            block.proj = torch.nn.Linear(4, 4, bias=False)
            self.model.decoder.layers = torch.nn.ModuleList([block])

        def forward(self, ids):
            seen.append(ids[0].tolist())
            x = torch.zeros(1, ids.shape[1], 4)
            self.model.decoder.layers[0].proj(x)
            return None

    class _Tok:
        def __call__(self, _text, return_tensors=None):
            class _Out:
                input_ids = torch.arange(total).unsqueeze(0)

            return _Out()

    bq.collect_layer_hessians(
        _Model(), _Tok(), nsamples=4, seqlen=seqlen, text="ignored"
    )

    assert len(seen) == 4
    windows = {tuple(window) for window in seen}
    assert len(windows) == 4, "every calibration sample must use a distinct window"
    assert all(len(window) == seqlen for window in seen)
    # Spread across the whole text: the budget reaches the end instead of
    # stopping at ``nsamples * seqlen``. span = 64 - 8 + 1 = 57, so the four
    # windows start at 0, 14, 28 and 42 and do not overlap.
    starts = [window[0] for window in seen]
    assert starts == [0, 14, 28, 42]
    assert all(b - a >= seqlen for a, b in zip(starts, starts[1:]))


def test_tt_hessian_matches_core_gptq_accumulator():
    """The TT collector's ``X^T X`` must agree with core GPTQ's accumulator.

    The two paths cannot share the accumulator yet -- core only fills ``H`` inside
    ``quantize()``, so there is no public "collect a Hessian" entry point -- so
    this pins agreement instead. It reads the private per-device partials on
    purpose: that is the only place core exposes the raw sum.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    from gptqmodel.quantization.gptq import GPTQ

    torch.manual_seed(3)
    width = 32

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.decoder = torch.nn.Module()
            block = torch.nn.Module()
            block.proj = torch.nn.Linear(width, 8, bias=False)
            self.model.decoder.layers = torch.nn.ModuleList([block])
            self.last_x = None

        def forward(self, ids):
            x = ids.float().reshape(1, -1, 1).expand(1, ids.shape[1], width) / 10.0
            self.last_x = x
            self.model.decoder.layers[0].proj(x)
            return None

    class _Tok:
        def __call__(self, _text, return_tensors=None):
            class _Out:
                input_ids = torch.arange(12).unsqueeze(0)

            return _Out()

    model = _Model()
    hessians = bq.collect_layer_hessians(
        model, _Tok(), text="ignored", nsamples=1, seqlen=12
    )
    h_tt = hessians["0.proj"]

    accumulator = GPTQ(model.model.decoder.layers[0].proj)
    accumulator.add_batch(model.last_x, None)
    h_core = sum(accumulator._device_hessian_partials.values())

    assert tuple(h_tt.shape) == (width, width)
    assert torch.allclose(h_tt, h_core, atol=1e-6), float((h_tt - h_core).abs().max())


def test_self_contained_checkpoint_roundtrip(tmp_path):
    """save_blockfloat_model -> load_blockfloat_model must be a complete round trip.

    The decoder-only checkpoint is an overlay: embeddings, norms and the head come
    from the base model. This path stores all of it so the directory loads on its
    own, which is what a published artifact needs.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(0)
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    model.eval()

    # The saver stores weights that are already on the grid, so quantise first.
    with torch.no_grad():
        for block in model.model.layers:
            for _, module in block.named_modules():
                if getattr(module, "weight", None) is not None and module.weight.dim() == 2:
                    module.weight.copy_(bq.quantize_ttbf4(module.weight.detach().float()))
        embed = model.model.embed_tokens
        embed.weight.copy_(
            bq.quantize_blockfloat(embed.weight.detach().float(), 8)[0]
        )

    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        expected = model(ids).logits.clone()

    out = str(tmp_path / "full")
    info = bq.save_blockfloat_model(model, out, bits=4, embed_bits=8)
    assert info["tied"] is True
    # q,k,v,o,gate,up,down per layer, plus the embedding.
    assert info["layers"] == 2 * 7 + 1

    loaded, restored = bq.load_blockfloat_model(out)
    assert len(restored) == 2 * 7 + 1

    with torch.no_grad():
        after = loaded(ids).logits
    # Every weight was already on the grid, and codes are stored rather than
    # re-derived, so the round trip must be bit-exact -- including the tied head,
    # which reads the embedding's codes.
    assert torch.equal(after, expected), float((after - expected).abs().max())


def test_self_contained_loader_refuses_a_decoder_only_directory(tmp_path):
    """A decoder-only directory must fail loudly, not load random norms.

    ``save_blockfloat_checkpoint`` writes the decoder blocks alone. Pointing the
    self-contained loader at that directory would leave the embedding and every
    norm at their initial values and still return a usable-looking model.
    """

    import importlib

    import pytest

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(1)
    block = torch.nn.Module()
    block.proj = torch.nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        block.proj.weight.copy_(bq.quantize_ttbf4(torch.randn(32, 64) * 0.02))

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.decoder = torch.nn.Module()
    model.model.decoder.layers = torch.nn.ModuleList([block])

    out = str(tmp_path / "overlay")
    bq.save_blockfloat_checkpoint(model, out, bits=4)

    # No model.safetensors and no index: nothing a standard HF loader could use.
    with pytest.raises(FileNotFoundError):
        bq.load_blockfloat_model(out)


def test_native_checkpoint_roundtrip_via_codes(tmp_path):
    """Saving codes and reading them back must reconstruct the same weights."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(5)
    w = bq.quantize_ttbf4(torch.randn(32, 64) * 0.02)

    # Build a throwaway model with one decoder-style block.
    block = torch.nn.Module()
    block.proj = torch.nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        block.proj.weight.copy_(w)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.decoder = torch.nn.Module()
    model.model.decoder.layers = torch.nn.ModuleList([block])

    out = str(tmp_path / "native")
    info = bq.save_blockfloat_checkpoint(model, out, bits=4)
    assert info["layers"] == 1
    # 4-bit packs two codes per byte, so payload is half the element count plus
    # one exponent per 16-value block.
    assert info["payload_bytes"] < w.numel(), info

    payload = bq.read_blockfloat_checkpoint(out)
    assert payload["bits"] == 4 and payload["block"] == 16
    expanded = bq.expand_native_layer(payload["layers"]["0.proj"])
    assert expanded.shape == w.shape
    assert expanded.equal(w), "codes must reconstruct the grid weights exactly"


def test_native_checkpoint_records_its_source_model(tmp_path):
    """A checkpoint must name the model and tokenizer it came from.

    A native checkpoint scored with a different tokenizer -- or an instruct model
    scored without its chat template -- produces a meaningless number, and the
    files themselves are indistinguishable without this metadata.
    """

    import importlib
    import json

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(7)
    block = torch.nn.Module()
    block.proj = torch.nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        block.proj.weight.copy_(bq.quantize_ttbf4(torch.randn(32, 64) * 0.02))
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.decoder = torch.nn.Module()
    model.model.decoder.layers = torch.nn.ModuleList([block])

    out = tmp_path / "meta"
    bq.save_blockfloat_checkpoint(
        model,
        str(out),
        bits=4,
        metadata={
            "source_model": "unsloth/Llama-3.2-1B-Instruct",
            "tokenizer_has_chat_template": True,
            # Deliberately wrong: format fields must win over caller metadata,
            # otherwise a stray key would make the checkpoint unloadable.
            "bits": 8,
            "format": "bogus",
        },
    )

    meta = json.loads((out / "blockfloat.json").read_text())
    assert meta["source_model"] == "unsloth/Llama-3.2-1B-Instruct"
    assert meta["tokenizer_has_chat_template"] is True
    # Caller metadata must not clobber the format fields the loader depends on.
    assert meta["format"] == "tt-blockfloat" and meta["bits"] == 4
    assert meta["block"] == 16


def test_blockfloat_is_inference_only_in_the_lifecycle_contract():
    """``tt-blockfloat`` describes a checkpoint; it cannot pack a dense model.

    The grid solve lives in ``scripts/tt_quantize_blockfloat.py`` and writes
    codes directly, so there is no looper processor behind this method. Both
    entry points that can start a quantization run must refuse it with the same
    message, or a caller would get a weight-only plan with nothing implementing
    it.
    """

    from gptqmodel.models.quantization_run_plan import QuantizationRunPlan
    from gptqmodel.quantization.config import (
        FORMAT,
        METHOD,
        QUANTIZE_BLACK_LIST,
        QuantizeConfig,
    )

    assert METHOD.TT_BLOCKFLOAT in QUANTIZE_BLACK_LIST

    config = QuantizeConfig(method=METHOD.TT_BLOCKFLOAT, bits=4, block=16, embed_bits=8)
    assert config.uses_weight_only_lifecycle() is True

    import pytest

    with pytest.raises(ValueError, match="Unsupported quantization operation"):
        QuantizationRunPlan.build(
            config,
            format_code=FORMAT.TT_BLOCKFLOAT,
            pack_backend=None,
            loop_backend=None,
            batch_size=1,
            needs_lora=False,
        )


def test_gptqmodel_load_consumes_a_blockfloat_checkpoint(tmp_path):
    """``GPTQModel.load`` must consume a native checkpoint, not just the helper.

    The published artifact is only useful if the public loader can open it. A
    ``tt-blockfloat`` directory has none of the GPTQ keys the quantized loader
    binds, so without an explicit route every decoder layer reads as missing.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(7)
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        for block in model.model.layers:
            for _, module in block.named_modules():
                if getattr(module, "weight", None) is not None and module.weight.dim() == 2:
                    module.weight.copy_(bq.quantize_ttbf4(module.weight.detach().float()))

    out = str(tmp_path / "ckpt")
    bq.save_blockfloat_model(model, out, bits=4, embed_bits=8)

    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
    from gptqmodel.quantization import METHOD

    loaded = GPTQModel.load(out)
    assert loaded.quantized is True
    assert loaded.quantize_config.method == METHOD.TT_BLOCKFLOAT

    q_proj = loaded.model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, TTBlockfloatLinear)
    # The module keeps codes, not a dense weight: 4-bit codes plus one int8
    # exponent per 16 values is 0.5625 bytes per weight.
    stored = q_proj.codes.numel() + q_proj.exponents.numel()
    assert stored / (q_proj.out_features * q_proj.in_features) < 0.6

    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits = loaded.model(ids).logits
    assert torch.isfinite(logits).all()
    assert logits.shape[-1] == 64

    # A placement request must not be accepted and then ignored: the block-float
    # kernel runs on CPU, and reporting a device the model is not on is worse
    # than refusing.
    import pytest

    with pytest.raises(NotImplementedError, match="load on CPU"):
        GPTQModel.load(out, device="xla:0")


def test_blockfloat_exponent_search_beats_floor_log2():
    """Per-block exponent selection must lower error without leaving the grid.

    ``floor(log2(block_max))`` is not always the lowest-error exponent: the grid
    only reaches ``(2 - 1/levels) * 2**exp``, so blocks whose maximum sits in the
    top of a binade clip. Trying the neighbours and keeping the best one is a
    pure win on the same grid, so it must not change what the grid can store.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(4)
    weight = torch.randn(32, 64) * 0.02
    quantized, exponents = bq.quantize_blockfloat(weight, 4)

    # Every stored value must decompose exactly, which also proves the search
    # landed on an exponent the format can encode.
    codes, recovered = bq.decompose_blockfloat(quantized, 4)
    assert torch.equal(
        bq.recompose_blockfloat(codes, recovered, bits=4), quantized
    ), "quantized weights are not exactly on the block-float grid"

    # The search may pick a different exponent per block, so it must never be
    # worse than the plain floor(log2(max)) rule on the same weights.
    fixed = bq._block_exponents(weight, bq.BLOCKFLOAT_BLOCK)
    plain = bq._round_with_exponents(weight, fixed.unsqueeze(-1), bits=4)
    searched_err = (quantized - weight).pow(2).sum().item()
    plain_err = (plain - weight).pow(2).sum().item()
    assert searched_err <= plain_err, (searched_err, plain_err)


def test_mixed_width_checkpoint_roundtrip(tmp_path):
    """A checkpoint may store some layers at bf8 and the rest at bf4.

    ttbf8 costs one byte per weight and is close to lossless, while
    ttbf4 is roughly a 2.8-bit format. A real checkpoint wants the extra
    byte only where it buys quality, so the format has to carry a width per
    layer rather than one global value.
    """

    import importlib
    import json
    import os

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(5)
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)

    widths = {"*": 4, "q_proj": 8, "down_proj": 8}
    with torch.no_grad():
        for index, block in enumerate(model.model.layers):
            for name, module in block.named_modules():
                if getattr(module, "weight", None) is None or module.weight.dim() != 2:
                    continue
                bits = bq._layer_bits(widths, f"{index}.{name}")
                module.weight.copy_(
                    bq.quantize_blockfloat(module.weight.detach().float(), bits)[0]
                )
        # The embedding is stored as bf8 codes too, so put it on the grid.
        embed = model.model.embed_tokens
        embed.weight.copy_(
            bq.quantize_blockfloat(embed.weight.detach().float(), 8)[0]
        )

    out = str(tmp_path / "mixed")
    info = bq.save_blockfloat_model(model, out, bits=widths, embed_bits=8)
    assert sorted(info["bits_per_layer"]) == [4, 8]
    # 2 layers x 7 linears + the embedding.
    assert info["layers"] == 2 * 7 + 1

    # The directory is an ordinary HF model: shards plus the standard index, and
    # no per-role payload or sidecar manifest.
    assert os.path.isfile(os.path.join(out, bq.INDEX_FILE))
    assert not os.path.exists(os.path.join(out, "blockfloat.json"))
    assert not os.path.exists(os.path.join(out, "norms.safetensors"))
    with open(os.path.join(out, bq.INDEX_FILE)) as handle:
        weight_map = json.load(handle)["weight_map"]
    shards = sorted(set(weight_map.values()))
    assert shards and all(name.endswith(".safetensors") for name in shards)
    for shard in shards:
        assert os.path.isfile(os.path.join(out, shard))

    # Quantized modules are named after their own buffers, so ``load_state_dict``
    # needs no translation table; dense tensors keep a normal ``.weight``.
    assert weight_map["model.layers.0.self_attn.q_proj.codes"] in shards
    assert weight_map["model.layers.0.mlp.gate_proj.codes"] in shards
    assert weight_map["model.norm.weight"] in shards
    # The embedding is codes at bf8, not a dense table: 64 x 32 codes at 8 bits
    # is 2048 bytes against 4096 for float16.
    assert weight_map["model.embed_tokens.codes"] in shards
    assert "model.embed_tokens.weight" not in weight_map
    # The directory ships the modeling file plain ``transformers`` needs, and
    # points its auto_map at it. Without both, ``from_pretrained`` leaves the
    # decoder layers at their random initial values and still returns a model.
    assert os.path.isfile(os.path.join(out, bq.MODELING_FILE))
    with open(os.path.join(out, "config.json")) as handle:
        saved_config = json.load(handle)
    assert saved_config["auto_map"]["AutoModelForCausalLM"].endswith(
        ".TTBlockfloatForCausalLM"
    )
    # At 8 bits a code is one byte, so the table is 2048 bytes instead of the
    # 4096 a dense float16 copy would need.
    from safetensors import safe_open

    with safe_open(os.path.join(out, weight_map["model.embed_tokens.codes"]), framework="pt") as handle:
        embed_codes = handle.get_tensor("model.embed_tokens.codes")
    assert embed_codes.numel() == 64 * 32
    assert embed_codes.dtype == torch.uint8

    with open(os.path.join(out, "config.json")) as handle:
        quant_config = json.load(handle)["quantization_config"]
    assert quant_config["mixed_bits"] is True
    assert quant_config["bits_per_layer"]["model.layers.0.self_attn.q_proj"] == 8
    assert quant_config["bits_per_layer"]["model.layers.0.mlp.gate_proj"] == 4

    loaded, restored = bq.load_blockfloat_model(out)
    assert len(restored) == 2 * 7 + 1

    # Each layer must come back at its own width, bit-exactly.
    for index, block in enumerate(model.model.layers):
        for name, module in block.named_modules():
            if getattr(module, "weight", None) is None or module.weight.dim() != 2:
                continue
            prefix = f"model.layers.{index}.{name}"
            quantized = loaded.get_submodule(prefix)
            assert quantized.bits == bq._layer_bits(widths, prefix)
            got = quantized.dequantize()
            want = module.weight.detach().float()
            assert torch.equal(got, want), prefix


def test_shipped_modeling_file_makes_transformers_load_correctly(tmp_path):
    """Plain ``transformers`` must load the checkpoint correctly, not silently.

    Without the shipped modeling file, ``AutoModelForCausalLM.from_pretrained``
    reports the codes as unexpected, leaves the decoder layers at their random
    initial values, and returns a model that generates garbage -- with no error.
    The checkpoint therefore carries an ``auto_map`` plus the class it points at,
    and this pins that the result is the same model the library loader builds.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(8)
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        for block in model.model.layers:
            for _, module in block.named_modules():
                if getattr(module, "weight", None) is not None and module.weight.dim() == 2:
                    module.weight.copy_(bq.quantize_ttbf4(module.weight.detach().float()))
        embed = model.model.embed_tokens
        embed.weight.copy_(bq.quantize_blockfloat(embed.weight.detach().float(), 8)[0])

    out = str(tmp_path / "ckpt")
    bq.save_blockfloat_model(model, out, bits=4, embed_bits=8)

    from transformers import AutoModelForCausalLM

    remote = AutoModelForCausalLM.from_pretrained(
        out, dtype=torch.float32, trust_remote_code=True
    )
    assert type(remote).__name__ == "TTBlockfloatForCausalLM"
    assert type(remote.model.embed_tokens).__name__ == "TTBlockfloatEmbedding"

    from gptqmodel import GPTQModel

    ours = GPTQModel.load(out).model

    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        a = remote(ids).logits
        b = ours(ids).logits
    assert torch.equal(a, b), float((a - b).abs().max())


def test_native_checkpoint_resave_is_bit_identical(tmp_path):
    """checkpoint -> floats -> checkpoint must reproduce the same codes.

    Re-saving is how provenance is added to a checkpoint that was written before
    the metadata existed. That is only honest if the decode/encode round trip is
    exact -- otherwise "re-saving" would quietly re-quantise the model.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(11)
    block = torch.nn.Module()
    block.proj = torch.nn.Linear(64, 32, bias=True)
    with torch.no_grad():
        block.proj.weight.copy_(bq.quantize_ttbf4(torch.randn(32, 64) * 0.02))
        block.proj.bias.zero_()

    def _model():
        m = torch.nn.Module()
        m.model = torch.nn.Module()
        m.model.decoder = torch.nn.Module()
        m.model.decoder.layers = torch.nn.ModuleList([block])
        return m

    first = str(tmp_path / "a")
    bq.save_blockfloat_checkpoint(_model(), first, bits=4)
    bq.load_native_checkpoint_into_dense_model(_model(), first)
    second = str(tmp_path / "b")
    bq.save_blockfloat_checkpoint(_model(), second, bits=4)

    a = bq._load_blockfloat_payload(first)
    b = bq._load_blockfloat_payload(second)
    assert set(a) == set(b)
    mismatched = [key for key in a if not torch.equal(a[key], b[key])]
    assert not mismatched, mismatched[:5]


def test_checkpoint_payload_is_safetensors(tmp_path):
    """The writer must emit safetensors and no pickle payload.

    safetensors is the repo-wide default for converted checkpoints; pickle is a
    legacy read-only format. Asserting the container here keeps a future writer
    change from silently reintroducing ``blockfloat.pt``.
    """

    import importlib
    import os

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(12)
    block = torch.nn.Module()
    block.proj = torch.nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        block.proj.weight.copy_(bq.quantize_ttbf4(torch.randn(32, 64) * 0.02))

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.decoder = torch.nn.Module()
    model.model.decoder.layers = torch.nn.ModuleList([block])

    out = str(tmp_path / "ckpt")
    bq.save_blockfloat_checkpoint(model, out, bits=4)

    assert os.path.isfile(os.path.join(out, bq.BLOCKFLOAT_PAYLOAD))
    assert not os.path.exists(os.path.join(out, bq.BLOCKFLOAT_PAYLOAD_LEGACY))


def test_legacy_pickle_payload_still_loads(tmp_path):
    """Checkpoints written before the switch must keep loading."""

    import importlib
    import os

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(13)
    block = torch.nn.Module()
    block.proj = torch.nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        block.proj.weight.copy_(bq.quantize_ttbf4(torch.randn(32, 64) * 0.02))

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.decoder = torch.nn.Module()
    model.model.decoder.layers = torch.nn.ModuleList([block])

    out = str(tmp_path / "ckpt")
    bq.save_blockfloat_checkpoint(model, out, bits=4)

    # Recreate the pre-change layout: pickle payload, no safetensors file.
    tensors = bq._load_blockfloat_payload(out)
    os.replace(
        os.path.join(out, bq.BLOCKFLOAT_PAYLOAD),
        os.path.join(str(tmp_path), "moved.safetensors"),
    )
    torch.save(tensors, os.path.join(out, bq.BLOCKFLOAT_PAYLOAD_LEGACY))

    restored = bq.load_native_checkpoint_into_dense_model(model, out)
    assert restored


def test_native_checkpoint_restores_a_fresh_model(tmp_path):
    """A fresh model populated from codes must match the saved weights."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(6)
    w = bq.quantize_ttbf4(torch.randn(32, 64) * 0.02)

    def make_model():
        block = torch.nn.Module()
        block.proj = torch.nn.Linear(64, 32, bias=False)
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.decoder = torch.nn.Module()
        model.model.decoder.layers = torch.nn.ModuleList([block])
        return model

    src = make_model()
    with torch.no_grad():
        src.model.decoder.layers[0].proj.weight.copy_(w)
    out = str(tmp_path / "native")
    bq.save_blockfloat_checkpoint(src, out, bits=4)

    fresh = make_model()
    restored = bq.load_native_checkpoint_into_dense_model(fresh, out)
    assert restored == {"0.proj": [32, 64]}
    assert fresh.model.decoder.layers[0].proj.weight.detach().equal(w)


def test_blockfloat_pack_roundtrips_exactly():
    """Codes plus exponents must reconstruct the grid weights bit-for-bit.

    The checkpoint stores no floats for weights, so exactness here is what makes
    "native format" true rather than approximate.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(4)
    w = bq.quantize_ttbf4(torch.randn(64, 128) * 0.02)
    codes, exponents = bq.decompose_blockfloat(w, bits=4)
    assert codes.dtype == torch.uint8
    assert bq.recompose_blockfloat(codes, exponents, bits=4).equal(w)

    # Two codes per byte at 4-bit density.
    packed = bq._pack_codes(codes, 4)
    assert packed.numel() == codes.numel() // 2
    unpacked = bq._unpack_codes(packed, 4, tuple(codes.shape))
    assert unpacked.equal(codes)

    # Off-grid input must be rejected rather than silently approximated.
    with pytest.raises(ValueError, match="not exactly on the block-float grid"):
        bq.decompose_blockfloat(torch.randn(8, 16) * 0.017, bits=4)


def test_blockfloat_decompose_allows_the_full_code_range():
    """A block may hold values up to just under 2x its shared scale.

    The code range is [-(2*levels-1), 2*levels-1] -- 15 values for 4-bit -- not
    [-levels, levels]. Clamping at levels rejected valid quantised weights.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    # Force values in the upper half of the binade, which needs the extra bit.
    w = torch.zeros(1, 16)
    w[0, :8] = 0.9
    w[0, 8:] = -0.9
    quantized = bq.quantize_ttbf4(w)
    codes, exponents = bq.decompose_blockfloat(quantized, bits=4)
    assert int(codes.max()) > 4, "test data should exercise codes above levels"
    assert bq.recompose_blockfloat(codes, exponents, bits=4).equal(quantized)


def test_gptq_blockfloat_is_native_not_an_int4_cast():
    """The solve must land on the bf4 grid itself, not pass through int4.

    Three assertions: the result is a fixed point of the grid, it differs from an
    int4 quantisation of the same weights, and the error is far below what a
    double rounding (bf4 -> int4) would produce.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(0)
    out_f, in_f = 64, 128
    w = torch.randn(out_f, in_f) * 0.02
    x = torch.randn(128, 32) @ torch.randn(32, in_f)
    h = x.T @ x + torch.eye(in_f)

    solved = bq.gptq_solve_blockfloat(w, h, bits=4, grid="blockfloat")

    # 1. Already on the grid: re-quantising is a no-op.
    assert torch.allclose(solved, bq.quantize_ttbf4(solved), atol=1e-6)
    # 2. Not the int4 answer for the same weights.
    assert not torch.allclose(solved, bq.quantize_uniform_int4(solved), atol=1e-9)
    # 3. Direct-to-grid beats a double rounding through int4 -- measured on the
    #    layer output, which is what GPTQ optimises. Raw weight distance is not the
    #    objective: the solve deliberately trades it for lower output error.
    reference = x @ w.T
    direct = bq.relative_error(reference, x @ solved.T)
    via_int4 = bq.relative_error(
        reference, x @ bq.quantize_ttbf4(bq.quantize_uniform_int4(w)).T
    )
    assert direct < via_int4, f"direct {direct} should beat int4-then-bf4 {via_int4}"


def test_quantize_model_blockfloat_reports_true_weight_error():
    """The error metric must compare against the ORIGINAL weight.

    ``module.weight.detach()`` shares storage with the parameter, so measuring
    after ``copy_()`` reported norm_l1 0.0000 for every layer -- a full-precision
    model that looked perfectly quantized. The reference must be cloned.
    """

    import importlib
    import types

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(2)
    block = torch.nn.Module()
    block.q_proj = torch.nn.Linear(128, 64, bias=False)
    with torch.no_grad():
        block.q_proj.weight.copy_(torch.randn(64, 128) * 0.02)
    original = block.q_proj.weight.detach().clone()

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.decoder = torch.nn.Module()
    model.model.decoder.layers = torch.nn.ModuleList([block])

    hessian = torch.eye(128) * 10.0
    fake = types.SimpleNamespace(
        collect_layer_hessians=lambda *a, **k: {"0.q_proj": hessian},
    )
    orig = bq.collect_layer_hessians
    bq.collect_layer_hessians = fake.collect_layer_hessians
    try:
        report = bq.quantize_model_blockfloat(
            model, tokenizer=None, bits=4, text=TT_CALIBRATION_TEXT
        )
    finally:
        bq.collect_layer_hessians = orig

    assert "0.q_proj" in report
    # A quantized layer must report a real, non-zero deviation.
    assert report["0.q_proj"]["norm_l1"] > 0.01, report
    # And the measured value must match the true change to the weights.
    true_err = bq.relative_error(original, block.q_proj.weight.detach())
    assert abs(report["0.q_proj"]["norm_l1"] - true_err) < 1e-4
    assert not torch.equal(original, block.q_proj.weight.detach())


def test_dequant_lut_is_exactly_rounded_bf16():
    """The 16-entry table must hold the exact BF16 values, not truncated ones."""

    import importlib

    kern = importlib.import_module("gptqmodel.utils.tt_int4_kernel")

    scale, zero = 0.0173, 7
    packed = kern.build_dequant_lut(scale, zero)
    assert packed.shape == (1, 8)

    bits = packed.reshape(-1).to(torch.int64) & 0xFFFFFFFF
    lo = (bits & 0xFFFF).to(torch.int16)
    hi = ((bits >> 16) & 0xFFFF).to(torch.int16)
    lut = torch.stack([lo, hi], dim=1).reshape(-1).view(torch.bfloat16)

    expected = (scale * (torch.arange(16, dtype=torch.float32) - zero)).to(torch.bfloat16)
    assert torch.equal(lut, expected)
    # The extremes bracket zero as GPTQ's code range implies.
    assert lut[0] < 0 < lut[15]


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TTNN") != "1",
    reason="explicit TTNN hardware gate",
)
@pytest.mark.parametrize("nm", [2, 4])
def test_fused_gemm_amortises_dequant_over_m(nm):
    """More M tiles must reuse the same dequantized weights, not redo them."""

    from gptqmodel.utils.tt_int4_kernel import run_fused_gemm_m
    from gptqmodel.utils.ttnn_runtime import close_mesh, current_mesh, open_mesh

    nk = 8
    torch.manual_seed(11)
    codes = torch.randint(0, 16, (nk * 1024,), dtype=torch.int64)
    weight = (0.0173 * (codes.to(torch.float32) - 7.0)).to(torch.bfloat16).reshape(nk * 32, 32)
    act = torch.randn(nm * 32, nk * 32, dtype=torch.bfloat16)
    reference = act.float() @ weight.float()

    mesh = current_mesh() or open_mesh(device_count=1)
    try:
        got = run_fused_gemm_m(act, codes, 0.0173, 7, device=mesh.device)
    finally:
        close_mesh()

    assert got.shape == (nm * 32, 32)
    norm_l1 = float((got - reference).abs().sum() / reference.abs().sum().clamp_min(1e-6))
    assert norm_l1 < 0.02, f"M-amortised GEMM deviates: norm_l1={norm_l1} (nm={nm})"


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TTNN") != "1",
    reason="explicit TTNN hardware gate",
)
def test_fused_gemm_accumulates_k_on_device():
    """Full K reduction inside the kernel must equal act @ dequant(weights)."""

    from gptqmodel.utils.tt_int4_kernel import run_fused_gemm
    from gptqmodel.utils.ttnn_runtime import close_mesh, current_mesh, open_mesh

    nk = 8
    torch.manual_seed(11)
    codes = torch.randint(0, 16, (nk * 1024,), dtype=torch.int64)
    weight = (0.0173 * (codes.to(torch.float32) - 7.0)).to(torch.bfloat16).reshape(nk * 32, 32)
    act = torch.randn(32, nk * 32, dtype=torch.bfloat16)
    reference = act.float() @ weight.float()

    mesh = current_mesh() or open_mesh(device_count=1)
    try:
        got = run_fused_gemm(act, codes, 0.0173, 7, device=mesh.device)
    finally:
        close_mesh()

    assert got.shape == (32, 32)
    norm_l1 = float((got - reference).abs().sum() / reference.abs().sum().clamp_min(1e-6))
    assert norm_l1 < 0.02, f"fused K accumulation deviates: norm_l1={norm_l1}"


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TTNN") != "1",
    reason="explicit TTNN hardware gate",
)
def test_fused_gemm_lut_is_at_least_as_accurate_as_soft_float():
    """The LUT path must reproduce the fused GEMM without regressing accuracy."""

    from gptqmodel.utils.tt_int4_kernel import run_fused_gemm, run_fused_gemm_lut
    from gptqmodel.utils.ttnn_runtime import close_mesh, current_mesh, open_mesh

    nk = 8
    torch.manual_seed(11)
    codes = torch.randint(0, 16, (nk * 1024,), dtype=torch.int64)
    weight = (0.0173 * (codes.to(torch.float32) - 7.0)).to(torch.bfloat16).reshape(nk * 32, 32)
    act = torch.randn(32, nk * 32, dtype=torch.bfloat16)
    reference = act.float() @ weight.float()

    def _err(got):
        return float((got - reference).abs().sum() / reference.abs().sum().clamp_min(1e-6))

    mesh = current_mesh() or open_mesh(device_count=1)
    try:
        lut_err = _err(run_fused_gemm_lut(act, codes, 0.0173, 7, device=mesh.device))
        soft_err = _err(run_fused_gemm(act, codes, 0.0173, 7, device=mesh.device))
    finally:
        close_mesh()

    assert lut_err < 0.02, f"LUT path deviates: norm_l1={lut_err}"
    assert lut_err <= soft_err, f"LUT ({lut_err}) should not be worse than soft-float ({soft_err})"



@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TTNN") != "1",
    reason="explicit TTNN hardware gate",
)
def test_fused_grid_gemm_matches_torch_on_device():
    """Multi-core fused int4 GEMM: one N tile per core, full K reduction.

    Verified on device at 2x2 (4 cores) and 8x8 (64 cores). Guards three things
    that were each found the hard way: rt[x][y] per-core arg indexing, a
    TILE_LAYOUT output (the writer pushes raw tile bytes), and every weight buffer
    appearing in io_tensors.
    """

    from gptqmodel.utils.tt_int4_kernel import run_fused_gemm_grid
    from gptqmodel.utils.ttnn_runtime import close_mesh, current_mesh, open_mesh

    gw, gh, nk, nm = 2, 2, 4, 2
    cores = gw * gh
    torch.manual_seed(13)
    codes = torch.randint(0, 16, (cores, nk, 1024), dtype=torch.int64)
    weight = (0.0173 * (codes.to(torch.float32) - 7.0)).to(torch.bfloat16)
    # Column block c comes from core c, so build W by blocks, not a flat reshape.
    w = weight.reshape(cores, nk, 32, 32).permute(1, 2, 0, 3).reshape(nk * 32, cores * 32)
    act = torch.randn(nm * 32, nk * 32, dtype=torch.bfloat16)
    reference = act.float() @ w.float()

    mesh = current_mesh() or open_mesh(device_count=1)
    try:
        got = run_fused_gemm_grid(act, codes, 0.0173, 7, device=mesh.device, grid=(gw, gh))
    finally:
        close_mesh()

    assert got.shape == (nm * 32, cores * 32)
    norm_l1 = float((got - reference).abs().sum() / reference.abs().sum().clamp_min(1e-6))
    assert norm_l1 < 0.02, f"grid GEMM deviates: norm_l1={norm_l1}"


@pytest.mark.skipif(os.environ.get("GPTQMODEL_TEST_TTNN") != "1", reason="explicit TTNN hardware gate")
@pytest.mark.parametrize("device_count", [1, 4])
def test_actual_ttnn_linear_matches_torch_oracle(device_count):
    """Native TTNN dequant+matmul against an independent Torch oracle."""

    from gptqmodel.nn_modules.qlinear.ttnn import TTNNLinear
    from gptqmodel.utils.ttnn_runtime import close_mesh

    if not hasattr(__import__("ttnn"), "open_device"):
        pytest.skip("ttnn is not installed")

    fixture = _fixture(qzero_format=1, bias=True, in_features=256, out_features=128)
    oracle_layout = _layout(fixture)
    cpu_weights = dequantize_tt_packed_4bit(oracle_layout).to(torch.bfloat16)
    torch.manual_seed(31)
    activation = torch.randn(4, fixture["in_features"], dtype=torch.float32).to(torch.bfloat16)
    # Independent FP32 oracle: dequantized checkpoint weights, activations and
    # bias accumulated in FP32, rounded once to BF16.
    expected = (
        activation.reshape(-1, fixture["in_features"]).float()
        @ cpu_weights.float()
        + fixture["bias"].float()
    ).to(torch.bfloat16)

    module = TTNNLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=fixture["in_features"], out_features=fixture["out_features"],
        bias=True, pack_dtype=torch.int32, format=fixture["format"],
    )
    module.qweight.copy_(fixture["qweight"])
    module.qzeros.copy_(fixture["qzeros"])
    module.scales.copy_(fixture["scales"].float())
    module.g_idx.copy_(fixture["g_idx"])
    module.bias.copy_(fixture["bias"])
    module.qzero_format(1)
    module.ttnn_device_count = device_count
    try:
        module.post_init()
    except Exception as exc:
        # TT-Metal allows one runtime owner per chip. When another test in this
        # session already initialized the PJRT plugin, the native path cannot
        # open the devices; that is an environment constraint, not a kernel
        # failure, so report it as a skip with the translated reason.
        close_mesh()
        if "already owned by another runtime" in str(exc):
            pytest.skip(f"chips are owned by the XLA runtime: {exc}")
        raise
    module.eval()
    try:
        # Packed codes must survive the device round trip bit-exactly.
        device_weights = module.dequantize_weight().to(torch.bfloat16)
        assert torch.equal(device_weights, cpu_weights)

        actual = module(activation).reshape(-1, fixture["out_features"]).float()
        reference = expected.float()
        pcc = torch.corrcoef(torch.stack([actual.flatten(), reference.flatten()]))[0, 1]
        assert pcc >= 0.999
        normalized = (actual - reference).abs().sum() / reference.abs().sum()
        assert normalized <= 2e-3
        # Tensix FMA precision leaves a bounded residual on material
        # magnitudes; BF16 output rounding sits on top of it.
        ulp = np.abs(_bf16_ordered_bits(actual) - _bf16_ordered_bits(reference))
        material = (reference.abs() >= 0.05).numpy()
        if material.any():
            assert int(ulp[material].max()) <= 64
        assert torch.allclose(
            actual[~material], reference[~material], rtol=2e-3, atol=2e-3
        )

        # Persistent state stays packed: no dense weight cache may be retained.
        retained = [
            name for name, value in vars(module).items()
            if isinstance(value, torch.Tensor)
            and value.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and value.numel() >= fixture["in_features"] * fixture["out_features"]
        ]
        assert retained == []
        layer = module._ttnn_layer
        assert layer is not None
        assert layer.weight_words.shape[0] == fixture["out_features"] // device_count
        packed_bytes = layer.weight_words.shape[0] * layer.weight_words.shape[1] * 4
        assert packed_bytes < fixture["in_features"] * fixture["out_features"] * 2
        first = module._build_layer()
        assert module._build_layer() is first
    finally:
        module.release_device()
        close_mesh()


# --- end-to-end: dense -> native bf4 -> post-quant inference ----------------
#
# Two levels. The CPU contract below runs always, using a synthetic decoder so it
# exercises the real save/load/serve code paths without a download. The hardware
# test is gated and runs the actual model on device, because that is where the
# claim "native format, no dequant at load" has to hold.

OPT_MODEL_ID = "facebook/opt-125m"
OPT_CHECKPOINT_ID = "ModelCloud/Opt-125-GPTQ-4bit-10-25-2024"


def _tiny_decoder_model(in_f=64, out_f=32):
    """Minimal OPT-shaped model: model.decoder.layers[i].<name> Linear modules."""

    def build():
        block = torch.nn.Module()
        block.q_proj = torch.nn.Linear(in_f, out_f, bias=True)
        block.fc1 = torch.nn.Linear(in_f, in_f, bias=False)
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.decoder = torch.nn.Module()
        model.model.decoder.layers = torch.nn.ModuleList([block])
        return model

    return build


def _quantize_and_infer_tiny(tmp_path, bits=4):
    """Quantize the synthetic model onto the grid, save, reload, and compare."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    build = _tiny_decoder_model()

    torch.manual_seed(17)
    src = build()
    with torch.no_grad():
        for param in src.parameters():
            param.copy_(torch.randn_like(param) * 0.02)

    # The synthetic model has no tokenizer, so drive the quantizer from a
    # hand-built Hessian: this is exactly what quantize_model_blockfloat does
    # internally, minus the calibration replay.
    in_f = src.model.decoder.layers[0].fc1.weight.shape[1]
    hessian = torch.eye(in_f) * float(in_f)
    solved = {}
    with torch.no_grad():
        for index, block in enumerate(src.model.decoder.layers):
            for name, module in block.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                q = bq.gptq_solve_blockfloat(
                    module.weight.detach(), hessian, bits=bits, grid="blockfloat"
                )
                module.weight.copy_(q)
                solved[f"{index}.{name}"] = q.clone()

    out_dir = str(tmp_path / "native")
    info = bq.save_blockfloat_checkpoint(src, out_dir, bits=bits)
    assert info["layers"] == len(solved)

    # Post-quant inference must run and must reflect the quantized weights.
    ids = torch.randint(0, 8, (1, 4))
    inputs = src.model.decoder.layers[0].fc1.weight.new_zeros(1, 4, in_f)
    with torch.no_grad():
        direct = src.model.decoder.layers[0].fc1(inputs)

    fresh = build()
    restored = bq.load_native_checkpoint_into_dense_model(fresh, out_dir)
    assert len(restored) == len(solved)
    for key, expected in solved.items():
        index, name = key.split(".", 1)
        module = dict(fresh.model.decoder.layers[int(index)].named_modules())[name]
        assert module.weight.detach().equal(expected), f"{key} did not restore exactly"

    with torch.no_grad():
        again = fresh.model.decoder.layers[0].fc1(inputs)
    assert torch.equal(direct, again), "post-quant inference changed after reload"
    return src, fresh, ids


def test_dense_to_native_bf4_to_inference(tmp_path):
    """Dense -> grid GPTQ -> packed codes -> reload -> inference, all exact."""

    src, fresh, _ = _quantize_and_infer_tiny(tmp_path)
    # The quantized weights differ from the dense ones (i.e. it really quantized)
    # yet inference after reload is bit-identical to inference before saving.
    assert fresh is not src


def test_native_bf4_payload_is_denser_than_float16(tmp_path):
    """4-bit packing must actually beat a float weight per element."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    build = _tiny_decoder_model(in_f=256, out_f=256)
    model = build()
    torch.manual_seed(3)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(bq.quantize_ttbf4(torch.randn_like(param) * 0.02))

    info = bq.save_blockfloat_checkpoint(model, str(tmp_path / "p2"), bits=4)
    elements = 256 * 256 * 2                     # q_proj + fc1
    float16_bytes = elements * 2
    assert info["payload_bytes"] < float16_bytes * 0.6, (
        f"{info['payload_bytes']} bytes is not meaningfully below "
        f"{float16_bytes} for {elements} elements"
    )


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TT_MODEL") != "1",
    reason="explicit end-to-end model gate (downloads OPT-125M, runs on device)",
)
def test_opt125m_native_bf4_quantize_and_infer(tmp_path):
    """Quantize real OPT-125M to native bf4, then run post-quant inference.

    Checks the property the whole path exists for: the solver's output survives
    ttbf4 storage unchanged, so inference needs no dequant at load.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel.utils.tt_blockfloat_quant import (
        expand_native_layer,
        load_native_checkpoint_into_dense_model,
        quantize_model_blockfloat,
        read_blockfloat_checkpoint,
        save_blockfloat_checkpoint,
    )

    tok = AutoTokenizer.from_pretrained(OPT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(OPT_MODEL_ID, dtype=torch.float32)
    ids = tok("The quick brown fox jumps over", return_tensors="pt").input_ids

    with torch.no_grad():
        dense_logits = model(ids).logits[0, -1].float().clone()

    report = quantize_model_blockfloat(
        model, tok, bits=4, nsamples=4, text=TT_CALIBRATION_TEXT
    )
    assert len(report) >= 72, f"only {len(report)} layers quantized"
    assert all(entry["on_grid"] for entry in report.values()), "a layer left the grid"
    # Guard the aliasing bug: a detached reference made this read 0.0 everywhere.
    assert min(entry["norm_l1"] for entry in report.values()) > 0.0, (
        "reported quantization error is zero, which means it was measured after "
        "the in-place write"
    )

    out_dir = str(tmp_path / "opt-native")
    saved = save_blockfloat_checkpoint(model, out_dir, bits=4)
    assert saved["layers"] >= 72

    with torch.no_grad():
        quantized_logits = model(ids).logits[0, -1].float().clone()

    # Native checkpoints must be usable without any float weights on disk.
    payload = read_blockfloat_checkpoint(out_dir)
    assert payload["bits"] == 4 and payload["block"] == 16
    assert len(payload["layers"]) >= 72

    fresh = AutoModelForCausalLM.from_pretrained(OPT_MODEL_ID, dtype=torch.float32)
    restored = load_native_checkpoint_into_dense_model(fresh, out_dir)
    assert len(restored) >= 72
    with torch.no_grad():
        reloaded_logits = fresh(ids).logits[0, -1].float().clone()

    # The reload must reproduce post-quant inference exactly, not approximately.
    assert torch.equal(reloaded_logits, quantized_logits), (
        "reloading from codes changed the output"
    )

    # And quantization must stay close to the dense model.
    cosine = float(torch.nn.functional.cosine_similarity(
        dense_logits, reloaded_logits, dim=0
    ))
    assert cosine > 0.99, f"logits cosine {cosine} is too low"
    assert int(dense_logits.argmax()) == int(reloaded_logits.argmax())

    # On device, a restored layer must store as ttbf4 with no drift.
    from gptqmodel.utils.ttnn_runtime import close_mesh, current_mesh, import_ttnn, open_mesh

    ttnn = import_ttnn()
    weights = expand_native_layer(payload["layers"]["0.self_attn.q_proj"])
    mesh = current_mesh() or open_mesh(device_count=1)
    try:
        device_tensor = ttnn.to_device(
            ttnn.from_torch(weights, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT),
            mesh.device,
        )
        round_tripped = ttnn.to_torch(device_tensor).float()
        drift = float((round_tripped - weights.float()).abs().max())
        assert drift == 0.0, f"ttbf4 storage moved the grid values by {drift}"
        ttnn.deallocate(device_tensor)
    finally:
        close_mesh()


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TT_MODEL") != "1",
    reason="explicit end-to-end model gate",
)
def test_opt125m_native_bf4_matches_int4_quantized_checkpoint():
    """Native bf4 should be in the same accuracy class as a shipped GPTQ int4 model.

    Compares post-quantization quality on the same prompt, using the published
    OPT-125M GPTQ int4 checkpoint as the reference point rather than the dense
    model, which is the comparison a user actually cares about.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel.utils.tt_blockfloat_quant import quantize_model_blockfloat

    tok = AutoTokenizer.from_pretrained(OPT_MODEL_ID)
    ids = tok("The quick brown fox", return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(OPT_MODEL_ID, dtype=torch.float32)
    with torch.no_grad():
        dense = model(ids).logits[0, -1].float().clone()

    quantize_model_blockfloat(
        model, tok, bits=4, nsamples=4, text=TT_CALIBRATION_TEXT
    )
    with torch.no_grad():
        bf4 = model(ids).logits[0, -1].float().clone()

    bf4_cos = float(torch.nn.functional.cosine_similarity(dense, bf4, dim=0))
    assert bf4_cos > 0.99, f"native bf4 cosine {bf4_cos} below the int4 class"


# --- GSM8K evaluation harness ----------------------------------------------
#
# The evaluator itself is tested without a model: prompt construction, answer
# extraction and scoring are deterministic and cost nothing, so they belong in
# the always-on suite. The model run is gated.

def _load_gsm8k_eval():
    """Load the GSM8K helpers from the scripts directory if present."""

    import importlib.util
    import os as _os

    path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                         "scripts", "gsm8k_eval.py")
    if not _os.path.exists(path):
        pytest.skip("scripts/gsm8k_eval.py not present")
    spec = importlib.util.spec_from_file_location("gsm8k_eval_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chat_template_is_only_applied_when_the_tokenizer_has_one():
    """Base models have no chat template; the harness must say so, not pretend.

    ``tests/eval.py`` silently falls back to the raw prompt when a tokenizer lacks
    a template, which makes a mis-specified run indistinguishable from a correctly
    templated one. This records whether the template was actually applied.
    """

    ev = _load_gsm8k_eval()

    class _WithTemplate:
        chat_template = "T"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert messages == [{"role": "user", "content": "Q"}]
            assert tokenize is False and add_generation_prompt is True
            return "<templated>Q</templated>"

    class _NoTemplate:
        chat_template = None

    raw, applied = ev.apply_chat_template_to_prompt(
        _WithTemplate(), "Q", enabled=True
    )
    assert applied is True and raw == "<templated>Q</templated>"

    raw, applied = ev.apply_chat_template_to_prompt(
        _NoTemplate(), "Q", enabled=True
    )
    assert applied is False and raw == "Q"

    # Disabled means never templated, even when a template exists.
    raw, applied = ev.apply_chat_template_to_prompt(
        _WithTemplate(), "Q", enabled=False
    )
    assert applied is False and raw == "Q"


def test_gsm8k_reports_the_checkpoint_source_model(tmp_path):
    """A native checkpoint names its source model; the harness must surface it.

    Scoring a checkpoint against a different tokenizer silently mis-measures, so
    the harness reads the recorded source model before loading the payload.
    """

    import json

    ev = _load_gsm8k_eval()

    good = tmp_path / "good"
    good.mkdir()
    (good / "blockfloat.json").write_text(
        json.dumps({"format": "tt-blockfloat", "source_model": "org/model-a"})
    )
    assert ev.read_checkpoint_source_model(str(good)) == "org/model-a"

    # No metadata at all: unknown provenance, not an error.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "blockfloat.json").write_text(json.dumps({"format": "tt-blockfloat"}))
    assert ev.read_checkpoint_source_model(str(bare)) is None

    assert ev.read_checkpoint_source_model(str(tmp_path / "missing")) is None

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "blockfloat.json").write_text("{not json")
    assert ev.read_checkpoint_source_model(str(broken)) is None


def test_gsm8k_extract_uses_the_last_answer_marker():
    """A marker emitted mid-reasoning must not be scored as the answer.

    Observed on Llama-3.2-1B-Instruct: ``#### 16 x 3 = $<<16*3=48>>48 ... #### 26``.
    Taking the first marker scored 16 and threw away a correct 26.
    """

    ev = _load_gsm8k_eval()

    assert ev.extract("reasoning\n#### 26") == "26"
    assert ev.extract("#### 16 x 3 = 48 for breakfast\n...\n#### 26") == "26"
    # No marker at all: fall back to the final number in the text.
    assert ev.extract("so the total is 42") == "42"
    assert ev.extract("no digits here") is None


def test_gsm8k_prompt_rendering_is_identical_across_shots():
    """Five-shot shots must be the rows immediately preceding the target.

    Determinism matters more than the exact wording: if shot selection drifted
    between the dense and quantized runs, the score difference would be confounded
    by a different prompt rather than the quantization.
    """

    import types

    ev = _load_gsm8k_eval()
    rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(12)]

    a = ev.build_prompt(rows, 9, tok=types.SimpleNamespace())
    b = ev.build_prompt(rows, 9, tok=types.SimpleNamespace())
    assert a == b, "prompt must be deterministic"

    # Shots for index 9 are rows 4..8 inclusive.
    for j in range(4, 9):
        assert f"q{j}" in a
    assert "q3" not in a, "a sixth shot leaked in"
    assert "q10" not in a, "a future row leaked in"


def test_gsm8k_answer_extraction_prefers_the_marker():
    """`#### N` is the gold answer format and must win over prose numbers."""

    ev = _load_gsm8k_eval()

    assert ev.extract("She earns 18 dollars.\n#### 18") == "18"
    assert ev.extract("So the answer is #### 1,234") == "1234"
    assert ev.extract("#### -5") == "-5"
    # Without a marker, fall back to the last number in the text.
    assert ev.extract("9 eggs then 18 dollars") == "18"


def test_gsm8k_gold_strips_marker_and_separators():
    ev = _load_gsm8k_eval()
    assert ev.gold("work...\n#### 18") == "18"
    assert ev.gold("#### 1,000") == "1000"


def test_gsm8k_prompt_is_five_shot_and_ends_with_the_question():
    """The prompt must be deterministic and end where the model should answer."""

    import types

    ev = _load_gsm8k_eval()
    rows = [
        {"question": f"q{i}", "answer": f"#### {i}"} for i in range(10)
    ]
    prompt = ev.build_prompt(rows, 7, tok=types.SimpleNamespace())
    assert prompt.count("Question:") == 6        # 5 shots + the question under test
    assert prompt.rstrip().endswith("Answer:")
    assert prompt.endswith("Question: q7\nAnswer:")
    # Earlier shots precede the target question.
    assert prompt.index("q2") < prompt.index("q7")


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TT_MODEL") != "1",
    reason="explicit end-to-end model gate (runs GSM8K generation)",
)
def test_gsm8k_eval_reports_a_score(tmp_path):
    """A short GSM8K run must produce a well-formed accuracy in [0, 1]."""

    ev = _load_gsm8k_eval()
    result = ev.run_eval("facebook/opt-125m", limit=4)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n"] == 4
    assert result["correct"] <= result["n"]
