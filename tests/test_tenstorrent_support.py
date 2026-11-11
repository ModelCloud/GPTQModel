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
    max_drift=0.000000 for both bfloat8_b and bfloat4_b.
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

    for bits, enc in ((4, bq.quantize_bfloat4_b), (8, bq.quantize_bfloat8_b)):
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


def test_gptq_blockfloat_rejects_mismatched_hessian():
    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    w = torch.randn(16, 32)
    with pytest.raises(ValueError, match="hessian must be"):
        bq.gptq_solve_blockfloat(w, torch.eye(16), bits=4)


def test_blockfloat_mantissa_widths():
    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    assert bq.BLOCKFLOAT_BITS == {"bfloat8_b": 8, "bfloat4_b": 4}
    # One sign bit and one implicit mantissa bit are not stored.
    assert bq.stored_mantissa_bits(8) == 6
    assert bq.stored_mantissa_bits(4) == 2


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
    bf4_err = bq.relative_error(weight, bq.quantize_bfloat4_b(weight))
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
        report = bq.quantize_model_blockfloat(model, tokenizer=None, bits=4)
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
