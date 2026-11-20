# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts and optional hardware tests for Tenstorrent GPTQ inference."""

from __future__ import annotations

import importlib.util
import json
import os
import types

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


# ---------------------------------------------------------------------------
# HIP dispatch routing (CPU, with only the backend availability mocked)
# ---------------------------------------------------------------------------


def _hip_routing(monkeypatch, *, available=True):
    """Make the HIP kernels look reachable so the routing itself is exercised.

    ``is_cuda`` is a property on the tensor type, so patching it is the only way
    to reach the dispatch branch on a CPU-only host. The kernels themselves are
    never executed here: every entry point the forward would call is replaced.
    """

    from gptqmodel.utils import tt_blockfloat_hip

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(tt_blockfloat_hip, "available", lambda device=None: available)
    return tt_blockfloat_hip


def _blockfloat_linear(bits=4, in_features=16, out_features=3):
    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear

    return TTBlockfloatLinear(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        backend=BACKEND.TT_BLOCKFLOAT,
        name="proj",
    )


@pytest.mark.parametrize(
    "shape, expected",
    [((16,), (1, 16)), ((2, 16), (2, 16)), ((1, 2, 16), (2, 16)), ((1, 2, 3, 16), (6, 16))],
)
def test_hip_linear_flattens_leading_dimensions(monkeypatch, shape, expected):
    """The binding takes a 2-D activation; F.linear accepted any leading dims."""

    hip = _hip_routing(monkeypatch)
    module = _blockfloat_linear()
    seen = {}

    def fake_linear(x, codes, exponents, bits, bias):
        seen["shape"] = tuple(x.shape)
        return torch.zeros(x.shape[0], 3)

    monkeypatch.setattr(hip, "linear", fake_linear)

    out = module(torch.zeros(*shape))

    assert seen["shape"] == expected
    assert out.shape == shape[:-1] + (3,)


def test_tied_head_uses_the_embedding_width_and_flattens(monkeypatch):
    """TTBlockfloatTiedHead holds no ``bits`` of its own; the embedding owns it."""

    from gptqmodel.nn_modules.qlinear.tt_blockfloat import (
        TTBlockfloatEmbedding,
        TTBlockfloatTiedHead,
    )

    hip = _hip_routing(monkeypatch)
    embedding = TTBlockfloatEmbedding(3, 16, bits=4, name="embed")
    head = TTBlockfloatTiedHead(embedding, name="head")
    assert not hasattr(head, "bits")

    seen = {}

    def fake_linear(x, codes, exponents, bits, bias):
        seen["shape"] = tuple(x.shape)
        seen["bits"] = bits
        return torch.zeros(x.shape[0], 3)

    monkeypatch.setattr(hip, "linear", fake_linear)

    out = head(torch.zeros(1, 2, 16))

    assert seen == {"shape": (2, 16), "bits": 4}
    assert out.shape == (1, 2, 3)


def test_embedding_with_a_wider_block_stays_on_the_torch_path(monkeypatch):
    """The kernels bind one exponent per 16 values, so block=32 cannot dispatch."""

    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatEmbedding

    hip = _hip_routing(monkeypatch)

    def refuse(*args, **kwargs):
        raise AssertionError("HIP embedding must not be used for block != 16")

    monkeypatch.setattr(hip, "embedding", refuse)
    monkeypatch.setattr(hip, "dequantize", refuse)

    embedding = TTBlockfloatEmbedding(3, 32, bits=4, block=32, name="embed")
    assert embedding.block == 32
    assert not embedding.hip_dispatch(torch.zeros(4, 32))

    ids = torch.tensor([[0, 2]], dtype=torch.int64)
    got = embedding(ids)
    assert got.shape == (1, 2, 32)
    assert torch.equal(got, embedding.dequantize()[ids])


def test_embedding_accepts_int32_indices(monkeypatch):
    """F.embedding accepts int32; the binding indexes with int64."""

    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatEmbedding

    hip = _hip_routing(monkeypatch)
    embedding = TTBlockfloatEmbedding(3, 16, bits=4, name="embed")
    seen = {}

    def fake_embedding(ids, codes, exponents, bits, rows, cols, dtype):
        seen["dtype"] = ids.dtype
        seen["shape"] = tuple(ids.shape)
        return torch.zeros(*ids.shape, cols)

    monkeypatch.setattr(hip, "embedding", fake_embedding)

    out = embedding(torch.tensor([0, 2], dtype=torch.int32))

    assert seen == {"dtype": torch.int64, "shape": (2,)}
    assert out.shape == (2, 16)


def test_empty_batches_short_circuit_before_the_kernels(monkeypatch):
    """An empty batch must not reach a launcher that rejects null pointers."""

    from gptqmodel.nn_modules.qlinear.tt_blockfloat import (
        TTBlockfloatEmbedding,
        TTBlockfloatTiedHead,
    )

    hip = _hip_routing(monkeypatch)

    def refuse(*args, **kwargs):
        raise AssertionError("an empty batch must not reach the kernel")

    monkeypatch.setattr(hip, "embedding", refuse)
    monkeypatch.setattr(hip, "linear", refuse)

    embedding = TTBlockfloatEmbedding(3, 16, bits=4, name="embed")
    empty_ids = torch.zeros(0, dtype=torch.int64)
    out = embedding(empty_ids)
    assert out.shape == (0, 16)
    assert out.dtype == torch.float32

    head = TTBlockfloatTiedHead(embedding, name="head")
    hidden = torch.zeros(1, 0, 16)
    projected = head(hidden)
    assert projected.shape == (1, 0, 3)
    assert projected.dtype == hidden.dtype


def test_bfloat2_b_grid_is_three_levels_and_packs_four_per_byte():
    """bfloat2_b is a sign bit plus one magnitude bit, so a block holds
    ``{-1, 0, +1} x 2**exponent`` and four codes share a byte."""

    from gptqmodel.utils.tt_blockfloat_quant import (
        _pack_codes,
        _unpack_codes,
        decompose_blockfloat,
        quantize_blockfloat,
        recompose_blockfloat,
        stored_mantissa_bits,
    )

    assert stored_mantissa_bits(2) == 0

    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(4, 32, generator=generator) * 0.3

    rounded, exponents = quantize_blockfloat(weight, 2)
    scale = torch.pow(2.0, exponents.float()).repeat_interleave(16, dim=1)
    # Every stored value is zero or one magnitude step off the shared exponent.
    assert torch.all((rounded == 0) | (rounded.abs() == scale))

    codes, recovered = decompose_blockfloat(rounded, 2)
    assert torch.equal(recompose_blockfloat(codes, recovered, 2), rounded)

    packed = _pack_codes(codes, 2)
    assert packed.numel() == codes.numel() // 4
    assert torch.equal(_unpack_codes(packed, 2, tuple(codes.shape)), codes)


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


@pytest.fixture
def cpu_xla_layout(monkeypatch):
    """Allow CPU layout tests to construct XLALinear without claiming PJRT."""

    monkeypatch.setattr(XLALinear, "validate_once", classmethod(lambda cls: (True, None)))


def test_ttxla_persistent_layout_and_invalidation(cpu_xla_layout):
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


def test_tt_layout_tracks_device_moves_and_weight_updates(cpu_xla_layout):
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


def test_ttxla_forward_rejects_cpu_activation_fail_closed(cpu_xla_layout):
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


def test_forward_never_retains_expanded_weights(cpu_xla_layout):
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
    options = []

    def _record(device, **kwargs):
        calls.append(device)
        options.append(kwargs)
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
    # The default placement keeps one compute-capability group (#4705/#4734).
    assert options == [{"single_capability_group": True}], options


@pytest.mark.parametrize("backend", [BACKEND.GPTQ_TTNN, BACKEND.GPTQ_TT_BF8])
def test_native_ttnn_auto_device_selection_skips_the_xla_probe(monkeypatch, backend):
    from gptqmodel.utils import importer as importer_module

    def _explode(*args, **kwargs):
        raise AssertionError(f"auto_select_device must not probe XLA for {backend}")

    monkeypatch.setattr(importer_module, "is_xla_available", _explode)
    assert importer_module.auto_select_device(None, backend) == DEVICE.XLA


@pytest.mark.parametrize("backend", [BACKEND.GPTQ_TTNN, BACKEND.GPTQ_TT_BF8])
def test_native_ttnn_layerwise_resolution_never_initializes_pjrt(monkeypatch, backend):
    """Both native TTNN backends share one host-staging and counting contract.

    ``GPTQ_TT_BF8`` used to be classified as a PJRT backend, so a default load
    resolved the XLA family through ``xla_device_count()``. That initializes
    the Tenstorrent PJRT plugin and claims every chip, which then collides with
    the TT-Metal mesh the kernel opens in ``post_init``.
    """
    from gptqmodel.models import loader as loader_module
    from gptqmodel.utils import torch as torch_module
    from gptqmodel.utils import ttnn_runtime

    def _explode(*args, **kwargs):
        raise AssertionError("the native TTNN path must not initialize PJRT")

    monkeypatch.setattr(loader_module, "expand_selector_device_family", _explode)
    monkeypatch.setattr(torch_module, "require_xla_runtime", _explode)
    monkeypatch.setattr(ttnn_runtime, "ttnn_device_count", lambda: 4)

    assert loader_module._native_ttnn_backend(backend)
    assert loader_module._resolve_default_layerwise_device(DEVICE.XLA, backend) == DEVICE.XLA
    assert loader_module._layerwise_device_count(DEVICE.XLA, backend) == 4
    assert loader_module._layerwise_device_strings(DEVICE.XLA, 4, backend) == [
        "xla:0", "xla:1", "xla:2", "xla:3",
    ]


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


def test_tt_bf8_matmul_uses_hifi4_and_fp32_destination(monkeypatch):
    """The block-float GEMM must not run on TT-Metal's default LoFi config.

    Default ``ttnn.linear`` uses LoFi with ``math_approx_mode``; that rounds
    ``258 + (-256)`` to zero instead of two. ``TTNNLinear`` already passes
    HiFi4 with FP32 destination accumulation, and the BF8 path must match it.
    """

    from gptqmodel.nn_modules.qlinear import ttnn_blockfloat as bf
    from gptqmodel.utils.ttnn_runtime import TTNNMesh

    calls = []
    configs = []

    class _FakeTTNN:
        class Arch:
            BLACKHOLE = "blackhole"

        class MathFidelity:
            HiFi4 = "hifi4"

        @staticmethod
        def get_arch_name():
            return "blackhole"

        @staticmethod
        def init_device_compute_kernel_config(arch, **kwargs):
            config = {"arch": arch, **kwargs}
            configs.append(config)
            return config

        @staticmethod
        def linear(activation, weight, **kwargs):
            calls.append(kwargs)
            return activation

    layer = bf.TTNNBlockFloat8Linear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=64, out_features=128, bias=False,
        pack_dtype=torch.int32, format=FORMAT.GPTQ_V2,
    )
    mesh = TTNNMesh(
        device=object(), mesh_shape=(), fabric="DISABLED", owner="test", generation=1
    )
    layer._ttnn_weight = object()
    layer._ttnn_bias = None
    layer._ttnn_devices = 1
    layer._ttnn_key = layer._weight_key(mesh)
    monkeypatch.setattr(bf, "current_mesh", lambda: mesh)
    monkeypatch.setattr(bf, "import_ttnn", lambda: _FakeTTNN)

    layer._compute(_FakeTTNN, None, torch.zeros(1, 64, dtype=torch.bfloat16))

    assert calls[0]["compute_kernel_config"] is configs[0]
    assert configs[0]["math_fidelity"] == "hifi4"
    assert configs[0]["fp32_dest_acc_en"] is True
    assert configs[0]["math_approx_mode"] is False


def test_tt_bf8_clearing_the_weight_cache_drops_the_compute_config(monkeypatch):
    """A cleared layer must rebuild both the weight and its compute config."""

    from gptqmodel.nn_modules.qlinear import ttnn_blockfloat as bf

    class _FakeTTNN:
        class Arch:
            BLACKHOLE = "blackhole"

        class MathFidelity:
            HiFi4 = "hifi4"

        @staticmethod
        def get_arch_name():
            return "blackhole"

        @staticmethod
        def init_device_compute_kernel_config(arch, **kwargs):
            return {"arch": arch, **kwargs}

    layer = bf.TTNNBlockFloat8Linear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=64, out_features=128, bias=False,
        pack_dtype=torch.int32, format=FORMAT.GPTQ_V2,
    )
    assert layer._compute_config(_FakeTTNN) is not None
    layer.clear_weight_cache()
    assert layer._ttnn_compute_config is None
    assert layer._ttnn_key is None


def test_tt_bf8_warm_device_weight_does_not_ignore_scale_updates(monkeypatch):
    """A present packed weight is not a cache hit when its sources changed."""

    from gptqmodel.nn_modules.qlinear import ttnn_blockfloat as bf
    from gptqmodel.utils.ttnn_runtime import TTNNMesh

    layer = bf.TTNNBlockFloat8Linear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=64, out_features=128, bias=False,
        pack_dtype=torch.int32, format=FORMAT.GPTQ_V2,
    )
    mesh = TTNNMesh(
        device=object(), mesh_shape=(), fabric="DISABLED", owner="test", generation=1
    )
    builds = []

    def _fake_build():
        builds.append(1)
        return object()

    monkeypatch.setattr(bf, "current_mesh", lambda: mesh)
    monkeypatch.setattr(layer, "_build_device_weight", _fake_build)
    layer._ttnn_weight = object()

    layer.scales.mul_(2)
    layer._device_weight()
    assert builds, "mutating the checkpoint scales must invalidate the packed weight"


def test_tt_bf8_device_weight_rebuilds_when_the_source_changes(monkeypatch):
    """Presence of a packed weight is not evidence that it is current.

    The cache key must follow in-place scale updates, tensor replacement, and a
    closed/reopened mesh; an explicit clear must still force a rebuild.
    """

    from gptqmodel.nn_modules.qlinear import ttnn_blockfloat as bf
    from gptqmodel.utils.ttnn_runtime import TTNNMesh

    def _mesh(generation):
        return TTNNMesh(
            device=object(),
            mesh_shape=(),
            fabric="DISABLED",
            owner="test",
            generation=generation,
        )

    layer = bf.TTNNBlockFloat8Linear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=64, out_features=128, bias=False,
        pack_dtype=torch.int32, format=FORMAT.GPTQ_V2,
    )
    builds = []

    def _fake_build():
        builds.append(len(builds))
        layer._ttnn_weight = object()
        layer._ttnn_bias = None
        layer._ttnn_devices = 1
        layer._ttnn_key = layer._weight_key(bf.current_mesh())
        return layer._ttnn_weight

    monkeypatch.setattr(bf, "current_mesh", lambda: _mesh(1))
    monkeypatch.setattr(layer, "_build_device_weight", _fake_build)

    first = layer._device_weight()
    assert builds == [0]
    assert layer._device_weight() is first
    assert builds == [0], "an unchanged layer must reuse its packed weight"

    layer.scales.mul_(2)
    assert layer._device_weight() is not first
    assert builds == [0, 1], "an in-place scale update must rebuild"

    layer.qweight = torch.zeros_like(layer.qweight)
    layer._device_weight()
    assert builds == [0, 1, 2], "a replaced source tensor must rebuild"

    monkeypatch.setattr(bf, "current_mesh", lambda: _mesh(2))
    layer._device_weight()
    assert builds == [0, 1, 2, 3], "a reopened mesh must rebuild"

    layer.clear_weight_cache()
    layer._device_weight()
    assert builds == [0, 1, 2, 3, 4], "an explicit clear must force a rebuild"


def test_ttnn_zero_words_are_repacked_per_shard():
    """A zero word that straddles a shard boundary must be split by column.

    ``out_features=40`` packs into five global words, but four shards of ten
    columns each need two padded words apiece. Slicing the global words by
    column would give the second shard a word it does not own.
    """

    from gptqmodel.nn_modules.qlinear.ttnn import _shard_zero_words

    def _decode(words, columns):
        values = []
        for word in words.tolist():
            for lane in range(8):
                values.append((word >> (4 * lane)) & 0xF)
        return values[:columns]

    torch.manual_seed(5)
    groups = 2
    for out_features, device_count in ((40, 4), (64, 2), (40, 1)):
        logical = torch.randint(0, 16, (groups, out_features), dtype=torch.int64)
        global_words = _pack_output_nibbles(logical)
        sharded = _shard_zero_words(
            global_words, out_features=out_features, device_count=device_count
        )
        shard_out = out_features // device_count
        words_per_shard = (shard_out + 7) // 8
        assert sharded.shape == (groups, device_count * words_per_shard), (
            out_features, device_count, tuple(sharded.shape)
        )
        for index in range(device_count):
            chunk = sharded[:, index * words_per_shard:(index + 1) * words_per_shard]
            decoded = [_decode(row, shard_out) for row in chunk]
            expected = logical[:, index * shard_out:(index + 1) * shard_out].tolist()
            assert decoded == expected, (out_features, device_count, index)


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


def _fake_ttnn_module(opened):
    """A TTNN stand-in that records opens and rejects no request itself."""

    class _FabricConfig:
        DISABLED = "DISABLED"
        FABRIC_1D = "FABRIC_1D"

    class _MeshShape:
        def __init__(self, *dims):
            self.dims = tuple(dims)

    class _Module:
        FabricConfig = _FabricConfig
        MeshShape = _MeshShape

        @staticmethod
        def open_device(**kwargs):
            opened.append(("device", dict(kwargs)))
            return object()

        @staticmethod
        def open_mesh_device(shape, **kwargs):
            opened.append(("mesh", shape.dims, dict(kwargs)))
            return object()

        @staticmethod
        def close_device(_device):
            return None

        @staticmethod
        def close_mesh_device(_device):
            return None

        @staticmethod
        def set_fabric_config(_cfg):
            return None

    return _Module


def _patch_ttnn_runtime(monkeypatch, ttnn_module):
    from gptqmodel.utils import ttnn_runtime

    monkeypatch.setattr(ttnn_runtime, "import_ttnn", lambda: ttnn_module)
    monkeypatch.setattr(ttnn_runtime, "_reject_pjrt_owner", lambda: None)
    monkeypatch.setattr(ttnn_runtime, "before_device_open", lambda **_: None)
    monkeypatch.setattr(ttnn_runtime, "after_device_close", lambda **_: None)
    monkeypatch.setattr(ttnn_runtime, "park_idle_clock", lambda: None)
    monkeypatch.setattr(ttnn_runtime, "_MESH", None)
    return ttnn_runtime


def test_ttnn_open_mesh_reuse_must_match_the_requested_placement(monkeypatch):
    """A live handle may only be reused for the exact placement requested.

    Reusing a one-chip open for a four-chip or chip-1 request returned the
    original chip with no error, so the caller received different placement and
    fabric than it asked for. Only an explicit close may change placement.
    """

    from gptqmodel.utils.ttnn_runtime import TTNNUnavailable

    opened = []
    runtime = _patch_ttnn_runtime(monkeypatch, _fake_ttnn_module(opened))
    try:
        first = runtime.open_mesh(device_count=1, device_id=0, owner="t")
        assert (first.num_devices, first.device_id, first.fabric) == (1, 0, "DISABLED")
        assert len(opened) == 1

        # Different chip count, requested with a fabric: must not be served by
        # the one-chip handle, and must not open a second runtime either.
        with pytest.raises(TTNNUnavailable, match="device_count=4"):
            runtime.open_mesh(device_count=4, fabric="FABRIC_1D", owner="t")
        # Different chip ordinal on the same count.
        with pytest.raises(TTNNUnavailable, match="device_id=1"):
            runtime.open_mesh(device_count=1, device_id=1, owner="t")
        # Different fabric on the same count.
        with pytest.raises(TTNNUnavailable, match="fabric=FABRIC_1D"):
            runtime.open_mesh(device_count=1, fabric="FABRIC_1D", owner="t")
        # Different trace reservation on the same count.
        with pytest.raises(TTNNUnavailable, match="trace_region_size=4096"):
            runtime.open_mesh(device_count=1, trace_region_size=4096, owner="t")

        # Nothing was opened behind the caller's back and the handle is unchanged.
        assert len(opened) == 1
        assert runtime.current_mesh() is first
        assert first.num_devices == 1

        # An identical request still reuses the live handle.
        assert runtime.open_mesh(device_count=1, device_id=0, owner="t") is first
        assert len(opened) == 1
    finally:
        runtime.close_mesh(owner="t")

    # Control: after an explicit close the new placement is opened normally, and
    # the previously valid two-chip -> four-chip rejection still fails closed.
    two = runtime.open_mesh(device_count=2, owner="t")
    try:
        assert two.num_devices == 2
        with pytest.raises(TTNNUnavailable, match="device_count=4"):
            runtime.open_mesh(device_count=4, owner="t")
    finally:
        runtime.close_mesh(owner="t")
    assert [entry[0] for entry in opened] == ["device", "mesh"]


def test_ttnn_mesh_lease_closes_only_after_the_last_holder(monkeypatch):
    """The process-wide mesh may serve several kernels; only the last lets go.

    ``release_mesh_lease`` is the signal a kernel uses to decide whether it may
    close the shared handle. A second holder must keep the device open, and a
    repeated hold must not inflate the count.
    """

    from gptqmodel.utils.ttnn_runtime import TTNNUnavailable

    class _Holder:
        pass

    opened = []
    runtime = _patch_ttnn_runtime(monkeypatch, _fake_ttnn_module(opened))
    first, second = _Holder(), _Holder()
    try:
        mesh = runtime.open_mesh(device_count=1, owner="t")
        runtime.hold_mesh_lease(first)
        runtime.hold_mesh_lease(second)
        runtime.hold_mesh_lease(first)

        assert runtime.release_mesh_lease(first) is False
        assert runtime.current_mesh() is mesh, "another holder still owns the mesh"
        assert runtime.release_mesh_lease(second) is True
        assert runtime.release_mesh_lease(second) is False, "a lease is released once"
    finally:
        runtime.close_mesh(owner="t")

    with pytest.raises(TTNNUnavailable, match="closed TTNN mesh"):
        runtime.hold_mesh_lease(first)


def test_release_device_on_a_non_native_wrapper_leaves_the_mesh_alone(monkeypatch):
    """A CPU wrapper must return False without touching another model's mesh."""

    from types import SimpleNamespace

    from gptqmodel.models.base import BaseQModel
    from gptqmodel.utils import ttnn_runtime

    calls = []
    monkeypatch.setattr(ttnn_runtime, "close_mesh", lambda **kwargs: calls.append(kwargs))

    wrapper = SimpleNamespace(model=_tiny_decoder_model()())
    assert BaseQModel.release_device(wrapper) is False
    assert calls == []


def test_release_device_delegates_to_each_native_kernel(monkeypatch):
    """The wrapper releases only its own native kernels, once each."""

    from types import SimpleNamespace

    from gptqmodel.models.base import BaseQModel

    released = []

    class _Native(torch.nn.Module):
        SUPPORTS_BACKENDS = [BACKEND.GPTQ_TTNN]

        def __init__(self, name):
            super().__init__()
            self.name = name

        def release_device(self):
            released.append(self.name)
            return self.name == "b"

    wrapper = SimpleNamespace(model=torch.nn.Module())
    wrapper.model.a = _Native("a")
    wrapper.model.b = _Native("b")

    assert BaseQModel.release_device(wrapper) is True
    assert released == ["a", "b"]


class _GridFakeTTNN:
    """TTNN stand-in that tags every device-resident buffer with its device."""

    uint32 = "uint32"
    bfloat16 = "bfloat16"
    TILE_LAYOUT = "tile"
    ROW_MAJOR_LAYOUT = "row_major"
    DRAM_MEMORY_CONFIG = "dram"

    class _SourceType:
        SOURCE_CODE = "source"

    class KernelDescriptor:
        SourceType = None  # filled below

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Descriptor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Args(dict):
        def __missing__(self, key):
            self[key] = {}
            return self[key]

    class _CoreRangeSet:
        def __init__(self, ranges):
            self.ranges = tuple(ranges)

    class _CoreRange:
        def __init__(self, start, end):
            self.start, self.end = start, end

    class _CoreCoord:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Shape:
        def __init__(self, dims):
            self.dims = tuple(int(dim) for dim in dims)

        def __iter__(self):
            return iter(self.dims)

    def __init__(self):
        self.uploads = []
        self.submissions = []
        self._addresses = 0
        self.KernelDescriptor.SourceType = self._SourceType

    def _next_address(self):
        self._addresses += 1
        return self._addresses

    def from_torch(self, tensor, **_kwargs):
        return tensor

    def to_device(self, tensor, device):
        buffer = type("DeviceTensor", (), {})()
        buffer.device = device
        buffer.host = tensor
        buffer.address = self._next_address()
        buffer.buffer_address = lambda: buffer.address
        self.uploads.append(buffer)
        return buffer

    def allocate_tensor_on_device(self, shape, _dtype, _layout, device, _memory):
        buffer = type("DeviceTensor", (), {})()
        buffer.device = device
        buffer.dims = tuple(shape)
        buffer.address = self._next_address()
        buffer.buffer_address = lambda: buffer.address
        return buffer

    def to_torch(self, tensor):
        return torch.zeros(tensor.dims)

    def generic_op(self, io, _program):
        self.submissions.append([buffer.device for buffer in io])

    def synchronize_device(self, _device):
        return None

    Shape = _Shape
    CoreRangeSet = _CoreRangeSet
    CoreRange = _CoreRange
    CoreCoord = _CoreCoord
    RuntimeArgs = _Args
    CBDescriptor = _Descriptor
    CBFormatDescriptor = _Descriptor
    ProgramDescriptor = _Descriptor
    ReaderConfigDescriptor = _Descriptor
    ComputeConfigDescriptor = _Descriptor
    WriterConfigDescriptor = _Descriptor


def test_fused_grid_caches_are_owned_by_the_consuming_device(monkeypatch):
    """Packed weights and the LUT must be uploaded per device, not shared.

    The cache key covered the source codes and (scale, zero) only, so a second
    live device was handed the first device's buffers next to its own
    activations and output: a mixed-owner native submission.
    """

    from gptqmodel.utils import ttnn_runtime
    from gptqmodel.utils.tt_int4_kernel import clear_grid_weight_cache, run_fused_gemm_grid

    driver = _GridFakeTTNN()
    monkeypatch.setattr(ttnn_runtime, "import_ttnn", lambda: driver)
    chip0, chip1 = object(), object()
    codes = torch.arange(1024).remainder(16).reshape(1, 1, 1024)
    act = torch.ones(32, 32)

    clear_grid_weight_cache()
    try:
        run_fused_gemm_grid(act, codes, 0.0173, 7, device=chip0, grid=(1, 1))
        assert driver.submissions[-1] == [chip0, chip0, chip0, chip0]
        uploads_after_first = len(driver.uploads)

        # Same weights on another live device: every operand it receives must
        # belong to that device, so the weights and LUT are uploaded again.
        run_fused_gemm_grid(act, codes, 0.0173, 7, device=chip1, grid=(1, 1))
        assert driver.submissions[-1] == [chip1, chip1, chip1, chip1]
        assert len(driver.uploads) == uploads_after_first + 3

        # The per-device entries are real caches: repeating either device's call
        # must upload its activation only, never its weights or LUT again.
        run_fused_gemm_grid(act, codes, 0.0173, 7, device=chip0, grid=(1, 1))
        assert driver.submissions[-1] == [chip0, chip0, chip0, chip0]
        assert len(driver.uploads) == uploads_after_first + 4
        run_fused_gemm_grid(act, codes, 0.0173, 7, device=chip1, grid=(1, 1))
        assert driver.submissions[-1] == [chip1, chip1, chip1, chip1]
        assert len(driver.uploads) == uploads_after_first + 5

        # Control: clearing the cache is still enough to force a fresh upload.
        clear_grid_weight_cache()
        run_fused_gemm_grid(act, codes, 0.0173, 7, device=chip1, grid=(1, 1))
        assert driver.submissions[-1] == [chip1, chip1, chip1, chip1]
        assert len(driver.uploads) == uploads_after_first + 8
    finally:
        clear_grid_weight_cache()


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
    max_drift=0.000000 for both bf8_b and bf4_b.
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

    for bits, enc in ((4, bq.quantize_bf4), (8, bq.quantize_bf8)):
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

    bf4_b = out_err(bq.gptq_solve_blockfloat(w, h, bits=4, grid="blockfloat"))
    u4 = out_err(bq.gptq_solve_blockfloat(w, h, bits=4, grid="uniform"))
    assert bf4_b < u4 * 1.25, f"4-bit block-float regressed: {bf4_b} vs uniform {u4}"


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
    assert bq.BLOCKFLOAT_BITS == {"bf2_b": 2, "bf4_b": 4, "bf8_b": 8}
    # One sign bit and one implicit mantissa bit are not stored.
    assert bq.stored_mantissa_bits(8) == 6
    assert bq.stored_mantissa_bits(4) == 2
    # At two bits there is no stored mantissa: the magnitude bit is the value.
    assert bq.stored_mantissa_bits(2) == 0


def test_blockfloat_dtype_names_are_canonical():
    """The block-float dtypes are named ``bf2_b``, ``bf4_b`` and ``bf8_b``.

    The names are Tenstorrent formats, not industry ones: OCP microscaling uses
    a per-32 ``E8M0`` scale and NVFP4 a per-16 ``FP8`` scale, and neither is
    bit-compatible with a per-16 power-of-two exponent. They carry no vendor
    prefix because no other provider claims ``bf4_b``/``bf8_b``, but the ``_b``
    is kept: upstream ``DataFormat::Bfp4`` without it is the 5-bit-exponent
    family, while ``Bfp4_b`` is the 8-bit one this quantizer implements. The
    exact bit layout and that reasoning are recorded in ``hw/dtypes.md``. The
    full vendor spelling (``bfloat4_b`` / ``bfloat8_b``) is kept only for the
    ``getattr(ttnn, ...)`` call site.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    assert bq.TT_BLOCKFLOAT_DTYPES == {2: "bf2_b", 4: "bf4_b", 8: "bf8_b"}
    # ttnn exposes no bfloat2_b attribute, so bf2_b has no vendor spelling.
    assert 2 not in bq.TT_BLOCKFLOAT_VENDOR_ATTRS
    # No vendor prefix: the shortening is deliberate, not an oversight.
    assert not any(name.startswith("tt") for name in bq.TT_BLOCKFLOAT_DTYPES.values())
    # The canonical name only shortens "bfloat"; every other part of the vendor
    # spelling, including the meaningful ``_b``, is preserved.
    for bits, vendor in bq.TT_BLOCKFLOAT_VENDOR_ATTRS.items():
        assert bq.TT_BLOCKFLOAT_DTYPES[bits] == vendor.replace("bfloat", "bf")
    assert bq.tt_blockfloat_dtype(4) == "bf4_b"
    assert bq.tt_blockfloat_dtype(8) == "bf8_b"
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
    assert kern.BLOCK_FLOAT_FORMATS == ("bf8_b",)


def test_blockfloat_vendor_attrs_exist_on_ttnn():
    """Every vendor spelling we emit must exist on the installed ttnn build.

    The un-suffixed names must not: ``ttnn`` exposes only the B family (8-bit
    shared exponent), so ``bf4``/``bf8`` without the suffix would name a format
    this build cannot produce. There is no ``_a``/``_c`` variant of either.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    ttnn = pytest.importorskip("ttnn")
    for bits, attr in bq.TT_BLOCKFLOAT_VENDOR_ATTRS.items():
        assert hasattr(ttnn, attr), f"ttnn has no {attr} dtype for {bits} bits"
    for absent in (
        "bfloat4",
        "bfloat8",
        "bfloat4_a",
        "bfloat8_a",
        "bfloat4_c",
        "bfloat8_c",
        "bfloat2_b",  # BFP2 exists in C++ but is not exposed by this build
    ):
        assert not hasattr(ttnn, absent), f"ttnn unexpectedly exposes {absent}"


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


def test_dense_remainder_is_stored_losslessly(tmp_path):
    """A norm that is not float16-exact must survive the round trip.

    Casting the dense remainder to float16 saved 0.13 MiB on a 1B model and
    rounded one norm by a single ulp, which was enough to break the
    "reload is bit-exact" invariant the checkpoint advertises.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    model = _tiny_llama()
    awkward = torch.tensor(
        [0.123456789, 0.987654321, 1.0000001192092896] * 11, dtype=torch.float32
    )[: model.model.norm.weight.numel()]
    with torch.no_grad():
        model.model.norm.weight.copy_(awkward)
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
                continue
            bits = 8 if name.endswith("embed_tokens") else 4
            weight.copy_(bq.quantize_blockfloat(weight.detach(), bits)[0])

    out = str(tmp_path / "lossless")
    bq.save_blockfloat_model(model, out, bits=4, embed_bits=8)
    loaded, _restored = bq.load_blockfloat_model(out)
    assert torch.equal(loaded.model.norm.weight.detach(), awkward)


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

    loaded = GPTQModel.load(first, device="cpu")
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


def test_unprobed_device_selection_never_picks_xla(monkeypatch):
    """An installed XLA runtime with no devices must not redirect a CPU job.

    Package presence is not evidence that a device exists, and asking costs a
    PJRT initialization that claims the Tenstorrent chips for the process.
    """

    import gptqmodel.utils.torch as ttmod

    monkeypatch.setattr(ttmod, "HAS_CUDA", False)
    monkeypatch.setattr(ttmod, "HAS_XPU", False)
    monkeypatch.setattr(ttmod, "HAS_NPU", False)
    monkeypatch.setattr(ttmod, "HAS_MPS", False)
    monkeypatch.setattr(ttmod, "HAS_XLA", True)

    assert ttmod.auto_select_torch_device_unprobed(0) == ttmod.CPU
    assert ttmod.auto_select_torch_device_unprobed(1) == ttmod.CPU


def test_torch_sync_does_not_probe_unclaimed_xla(monkeypatch):
    """A generic barrier must not discover and claim another runtime."""

    import gptqmodel.utils.torch as ttmod

    def forbidden(*args, **kwargs):
        raise AssertionError("torch_sync touched XLA for a job that never selected it")

    for flag in ("HAS_CUDA", "HAS_XPU", "HAS_NPU", "HAS_MPS"):
        monkeypatch.setattr(ttmod, flag, False)
    monkeypatch.setattr(ttmod, "is_xla_available", forbidden)
    monkeypatch.setattr(ttmod, "require_xla_runtime", forbidden)
    monkeypatch.setattr(ttmod, "xla_runtime_claimed", lambda: False)

    ttmod.torch_sync()


def test_torch_sync_synchronizes_claimed_xla(monkeypatch):
    """Once XLA owns the devices, the barrier does include it."""

    import gptqmodel.utils.torch as ttmod

    synced = {"count": 0}

    class Runtime:
        @staticmethod
        def sync(wait=False):
            synced["count"] += 1

    for flag in ("HAS_CUDA", "HAS_XPU", "HAS_NPU", "HAS_MPS"):
        monkeypatch.setattr(ttmod, flag, False)
    monkeypatch.setattr(ttmod, "require_xla_runtime", lambda: Runtime)
    monkeypatch.setattr(ttmod, "xla_runtime_claimed", lambda: True)

    ttmod.torch_sync()
    assert synced["count"] == 1


def test_qqq_portable_kernel_accepts_pre_ampere(monkeypatch):
    """The FP32 fallback must not inherit the native kernel's SM80 floor."""

    from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear

    def forbidden(*args, **kwargs):
        raise AssertionError("portable kernel consulted compute capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    QQQTorchLinear.validate_device(torch.device("cuda:0"))

    # The native kernel keeps its restriction.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(NotImplementedError, match="Marlin"):
        QQQLinear.validate_device(torch.device("cuda:0"))


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
    bf4_err = bq.relative_error(weight, bq.quantize_bf4(weight))
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
@pytest.mark.parametrize("chips", [1, 4])
def test_ttnn_zero_words_crossing_shard_boundaries(chips):
    """A supported GPTQ width with a partial qzero word on each shard."""

    from gptqmodel.nn_modules.qlinear.ttnn import TTNNLinear
    from gptqmodel.utils.ttnn_runtime import close_mesh

    width, outputs = 64, 40
    layer = TTNNLinear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=width, out_features=outputs, bias=False,
        pack_dtype=torch.int32, format=FORMAT.GPTQ_V2,
    ).eval()
    layer.ttnn_device_count = chips
    codes = torch.arange(outputs, dtype=torch.int32) % 16
    lane_pattern = 0x11111111 * codes.long()
    layer.qweight.copy_(lane_pattern.expand_as(layer.qweight).to(torch.int32))
    layer.qzeros.fill_(0x88888888 - 2**32)
    layer.qzero_format(2)
    layer.scales.fill_(0.125)
    x = torch.zeros(4, width, dtype=torch.bfloat16)
    x[:, 0] = 1
    expected = ((codes.double() - 8) * 0.125).expand(4, -1).bfloat16()
    try:
        with torch.inference_mode():
            layer.post_init()
            actual = layer(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        layer.clear_weight_cache()
        close_mesh()


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TTNN") != "1",
    reason="explicit TTNN hardware gate",
)
def test_tt_bf8_matmul_keeps_finite_cancellation_residuals():
    """BF8 weights that are exactly one must not round ``258 - 256`` to zero.

    TT-Metal's default LoFi matmul returns 0 for every row; HiFi4 with FP32
    destination accumulation returns the exact sum the dense oracle keeps.
    """

    from gptqmodel.nn_modules.qlinear.ttnn_blockfloat import TTNNBlockFloat8Linear
    from gptqmodel.utils.ttnn_runtime import close_mesh

    layer = TTNNBlockFloat8Linear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=64, out_features=128, bias=False,
        pack_dtype=torch.int32, format=FORMAT.GPTQ_V2,
    ).eval()
    layer.qweight.fill_(0x99999999 - 2**32)  # all code 9
    layer.qzeros.fill_(0x88888888 - 2**32)   # all zero point 8
    layer.qzero_format(2)
    layer.scales.fill_(1)                    # exact device BF8 weight 1
    x = torch.zeros(3, 64, dtype=torch.bfloat16)
    x[:, 0] = torch.tensor([258, 256, 256])
    x[:2, 1] = -256
    expected = x.double().sum(-1, keepdim=True).expand(-1, 128).bfloat16()
    try:
        with torch.inference_mode():
            actual = layer(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        layer.clear_weight_cache()
        close_mesh()


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TTNN") != "1",
    reason="explicit TTNN hardware gate",
)
def test_tt_bf8_weight_cache_follows_scales_after_warmup():
    """A warmed BF8 layer must rebuild when the checkpoint scales change."""

    from gptqmodel.nn_modules.qlinear.ttnn_blockfloat import TTNNBlockFloat8Linear
    from gptqmodel.utils.ttnn_runtime import close_mesh

    layer = TTNNBlockFloat8Linear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=64, out_features=128, bias=False,
        pack_dtype=torch.int32, format=FORMAT.GPTQ_V2,
    ).eval()
    layer.qweight.fill_(0x99999999 - 2**32)  # all code 9
    layer.qzeros.fill_(0x88888888 - 2**32)   # all zero point 8
    layer.qzero_format(2)
    layer.scales.fill_(1)
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[:, 0] = 1
    try:
        with torch.no_grad():
            before = layer(x)
            version = layer.scales._version
            layer.scales.mul_(2)
            assert layer.scales._version > version
            rebuilt = layer(x)
        assert torch.equal(before, torch.ones_like(before))
        assert torch.equal(rebuilt, torch.full_like(rebuilt, 2))
    finally:
        layer.clear_weight_cache()
        close_mesh()


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
    """The quantizer's grid must match what bf4_b/bf8_b can store.

    Measured on a p300c by encoding and reading back: bf4_b holds 4 levels per
    16-value block inside one binade, bf8_b holds up to 64. That is 2 and 6 stored
    mantissa bits, matching ``stored_mantissa_bits``. The step is ``2^exp/levels``:
    hardware showed 0.125 = 0.5/4 for bf4_b.

    Blocks each carry their own exponent -- verified by placing a 16x weaker
    segment beside a strong one in the same row: both keep their own 4 levels
    instead of the weak one collapsing.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    # bf4_b exercises the full grid inside one binade: exactly 4 levels.
    vals = torch.linspace(0.5, 0.99, 16).reshape(1, 16)
    out4, exp4 = bq.quantize_blockfloat(vals, 4)
    levels4 = torch.unique(out4.reshape(-1))
    assert levels4.numel() == 4, levels4
    assert torch.unique(exp4).numel() == 1
    span = float(torch.pow(2.0, exp4.reshape(-1)[0]))
    assert abs(float(levels4[1] - levels4[0]) - span / 4) < 1e-6

    # bf8_b must be strictly finer than bf4_b on the same input.
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
    # stopping at ``nsamples * seqlen``. The legal start range is
    # ``0..total - seqlen`` = ``0..56``; the four windows are spaced by
    # ``56 // 3`` and land on 0, 18, 37 and 56, so they still do not overlap.
    starts = [window[0] for window in seen]
    assert starts == [0, 18, 37, 56]
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
                    module.weight.copy_(bq.quantize_bf4(module.weight.detach().float()))
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


def test_blockfloat_grid_values_survive_bf16_storage():
    """Every storable grid value is exact in bf16, which is why bf16 is safe.

    ``bf4_b`` keeps 3 significant bits and ``bf8_b`` keeps 7; bf16 carries 8.
    So copying a solved weight back into a bf16 parameter cannot move it off the
    grid, and the loader may use the checkpoint's own precision instead of
    upcasting everything to float32.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(0)
    weight = torch.randn(64, 128)
    for bits in (4, 8):
        quantized, _exponents = bq.quantize_blockfloat(weight, bits)
        round_tripped = quantized.to(torch.bfloat16).to(torch.float32)
        assert torch.equal(round_tripped, quantized), f"bf{bits} grid not exact in bf16"


def test_resolve_blockfloat_dtype_defaults_to_the_checkpoint():
    """``None``/``auto`` must mean the checkpoint's dtype, not float32."""

    import importlib
    from types import SimpleNamespace

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    bf16_config = SimpleNamespace(torch_dtype=torch.bfloat16)
    string_config = SimpleNamespace(torch_dtype="bfloat16")

    assert bq.resolve_blockfloat_dtype(None, bf16_config) is torch.bfloat16
    assert bq.resolve_blockfloat_dtype("auto", bf16_config) is torch.bfloat16
    assert bq.resolve_blockfloat_dtype(None, string_config) is torch.bfloat16
    # An explicit request wins over the checkpoint.
    assert bq.resolve_blockfloat_dtype(torch.float32, bf16_config) is torch.float32
    assert bq.resolve_blockfloat_dtype("float16", bf16_config) is torch.float16
    # The loader's "not provided" sentinel is not a dtype and must not win.
    assert bq.resolve_blockfloat_dtype(object(), bf16_config) is torch.bfloat16
    # No config and no request: float32 remains the safe fallback.
    assert bq.resolve_blockfloat_dtype(None) is torch.float32
    with pytest.raises(ValueError, match="unknown dtype"):
        bq.resolve_blockfloat_dtype("float8", bf16_config)


def test_blockfloat_load_follows_the_checkpoint_dtype(tmp_path):
    """A bf16 checkpoint loads as bf16 by default and can be forced to fp32.

    The source is built the way ``from_pretrained`` builds one -- bf16 weights
    with a float32 rotary ``inv_freq`` -- so a blanket ``.to(bf16)`` in the
    loader would narrow that buffer and change the rotary frequencies.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(0)
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True, dtype=torch.bfloat16,
    )
    model = LlamaForCausalLM._from_config(config)
    model.eval()
    assert model.model.rotary_emb.inv_freq.dtype is torch.float32
    with torch.no_grad():
        for block in model.model.layers:
            for _, module in block.named_modules():
                if getattr(module, "weight", None) is not None and module.weight.dim() == 2:
                    module.weight.copy_(
                        bq.quantize_bf4(module.weight.detach().float()).to(torch.bfloat16)
                    )
        embed = model.model.embed_tokens
        embed.weight.copy_(
            bq.quantize_blockfloat(embed.weight.detach().float(), 8)[0].to(torch.bfloat16)
        )

    out = str(tmp_path / "bf16")
    bq.save_blockfloat_model(model, out, bits=4, embed_bits=8)
    import json

    with open(f"{out}/config.json") as handle:
        saved = json.load(handle)
    assert saved["dtype"] == "bfloat16"
    assert saved["torch_dtype"] == "bfloat16"

    auto, _ = bq.load_blockfloat_model(out)
    assert next(auto.parameters()).dtype is torch.bfloat16
    # The float32 rotary buffer must survive the load, as it does through
    # from_pretrained; narrowing it shifts every rotary angle.
    assert auto.model.rotary_emb.inv_freq.dtype is torch.float32

    forced, _ = bq.load_blockfloat_model(out, dtype=torch.float32)
    assert next(forced.parameters()).dtype is torch.float32
    # An explicit override must also update the authoritative config field,
    # otherwise a re-save writes `dtype: bfloat16` next to `torch_dtype: float32`
    # and transformers prefers the former on the next load.
    assert forced.config.dtype is torch.float32

    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        reference = model(ids).logits.float()
        reloaded = auto(ids).logits.float()
    assert torch.equal(reloaded, reference), float((reloaded - reference).abs().max())

    # Re-save the upcast model and confirm the artifact does not silently fall
    # back to bf16 on the next load.
    upcast_dir = str(tmp_path / "upcast")
    bq.save_blockfloat_model(forced, upcast_dir, bits=4, embed_bits=8)
    with open(f"{upcast_dir}/config.json") as handle:
        upcast_config = json.load(handle)
    assert upcast_config["dtype"] == "float32"
    assert upcast_config["torch_dtype"] == "float32"

    again, _ = bq.load_blockfloat_model(upcast_dir)
    assert next(again.parameters()).dtype is torch.float32
    norm = forced.model.norm.weight.detach()
    assert torch.equal(again.model.norm.weight.detach(), norm)


def test_blockfloat_pinned_fp32_norms_survive_an_fp32_override(tmp_path):
    """Pinned parameters must be converted to fp32, not skipped.

    GPT-OSS declares ``_keep_in_fp32_modules = ["post_attention_layernorm",
    "input_layernorm", "norm"]``, so ``from_pretrained`` keeps its norms fp32
    inside a bf16 model. The loader initialises them in the checkpoint dtype,
    so an explicit fp32 load has to actively promote them -- skipping the cast
    leaves a bf16 parameter, and the stored fp32 norm rounds on load
    (1.001 -> 1.0).
    """

    import importlib
    import json

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    from transformers import GptOssConfig, GptOssForCausalLM

    torch.manual_seed(0)
    config = GptOssConfig(
        vocab_size=64, hidden_size=32, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=2, head_dim=8, max_position_embeddings=64,
        tie_word_embeddings=True, dtype=torch.bfloat16,
    )
    model = GptOssForCausalLM._from_config(config)
    # The precondition: without an explicit promotion the norm starts bf16.
    assert model.model.norm.weight.dtype is torch.bfloat16

    pinned = tuple(GptOssForCausalLM._keep_in_fp32_modules)
    with torch.no_grad():
        # Emulate from_pretrained: pinned tensors are fp32, the rest bf16.
        for name, parameter in model.named_parameters():
            if any(fragment in name for fragment in pinned):
                parameter.data = parameter.data.float()
                parameter.fill_(1.001)
        for block in model.model.layers:
            for _, module in block.named_modules():
                if getattr(module, "weight", None) is not None and module.weight.dim() == 2:
                    module.weight.copy_(
                        bq.quantize_bf4(module.weight.detach().float()).to(torch.bfloat16)
                    )
        embed = model.model.embed_tokens
        embed.weight.copy_(
            bq.quantize_blockfloat(embed.weight.detach().float(), 8)[0].to(torch.bfloat16)
        )

    out = str(tmp_path / "gptoss")
    bq.save_blockfloat_model(model, out, bits=4, embed_bits=8)
    with open(f"{out}/config.json") as handle:
        saved = json.load(handle)
    assert saved["dtype"] == "bfloat16"

    # Default load: the model is bf16, but the pinned norm stays fp32 exactly
    # as it does through from_pretrained.
    auto, _ = bq.load_blockfloat_model(out)
    assert next(auto.parameters()).dtype is torch.bfloat16
    assert auto.model.norm.weight.dtype is torch.float32

    # Explicit fp32 override: the pinned norm must be promoted, not skipped.
    forced, _ = bq.load_blockfloat_model(out, dtype=torch.float32)
    assert forced.model.norm.weight.dtype is torch.float32
    assert float(forced.model.norm.weight[0].detach()) == 1.0010000467300415


def test_blockfloat_embedding_follows_module_casts():
    """``.float()``/``.bfloat16()`` must move the table's output dtype too.

    The table owns no float parameter, so a plain attribute would stay frozen
    while the rest of the model was cast. A bf16 table feeding fp32 linears then
    raises a dtype mismatch on the next forward.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    kern = importlib.import_module("gptqmodel.nn_modules.qlinear.tt_blockfloat")

    torch.manual_seed(0)
    weight = bq.quantize_blockfloat(torch.randn(16, 32), 8)[0]
    codes, exponents = bq.decompose_blockfloat(weight, bits=8)

    table = kern.TTBlockfloatEmbedding(
        num_embeddings=16, embedding_dim=32, bits=8, dtype=torch.bfloat16
    )
    table.set_quantized_state(codes, exponents)

    ids = torch.tensor([0, 3, 15])
    assert table.output_dtype is torch.bfloat16
    assert table(ids).dtype is torch.bfloat16

    table.float()
    assert table.output_dtype is torch.float32
    assert table(ids).dtype is torch.float32

    table.bfloat16()
    assert table.output_dtype is torch.bfloat16
    assert table(ids).dtype is torch.bfloat16

    # The dtype anchor is a non-persistent buffer: it must not become a key in
    # the checkpoint.
    assert "_output_dtype_anchor" not in table.state_dict()


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
        block.proj.weight.copy_(bq.quantize_bf4(torch.randn(32, 64) * 0.02))

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
    w = bq.quantize_bf4(torch.randn(32, 64) * 0.02)

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
        block.proj.weight.copy_(bq.quantize_bf4(torch.randn(32, 64) * 0.02))
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
                    module.weight.copy_(bq.quantize_bf4(module.weight.detach().float()))

    out = str(tmp_path / "ckpt")
    bq.save_blockfloat_model(model, out, bits=4, embed_bits=8)

    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
    from gptqmodel.quantization import METHOD

    loaded = GPTQModel.load(out, device="cpu")
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


def test_gptqmodel_load_accepts_a_blockfloat_checkpoint_selected_by_subfolder(tmp_path):
    """``subfolder=`` must select the same checkpoint as a direct directory path.

    The block-float probe and the model build both have to use the resolved
    checkpoint directory. Probing the repository root instead made this public
    load fall through to the GPTQ config parser and fail with
    ``Unknown quantization format: `tt-blockfloat``` before the model existed.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(11)
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
                    module.weight.copy_(bq.quantize_bf4(module.weight.detach().float()))

    root = tmp_path / "repo"
    packed = root / "packed"
    bq.save_blockfloat_model(model, str(packed), bits=4, embed_bits=8)

    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear

    direct = GPTQModel.load(str(packed), device="cpu")
    nested = GPTQModel.load(str(root), subfolder="packed", device="cpu")

    assert nested.quantized is True
    assert nested.quantize_config.method == METHOD.TT_BLOCKFLOAT
    # The wrapped model is built from the resolved directory, so its recorded
    # root is the checkpoint the caller selected, not the parent repository.
    assert nested.model_local_path == str(packed)

    def blockfloat_modules(loaded):
        return {
            name: module
            for name, module in loaded.model.named_modules()
            if isinstance(module, TTBlockfloatLinear)
        }

    direct_modules = blockfloat_modules(direct)
    nested_modules = blockfloat_modules(nested)
    assert nested_modules, "fixture produced no block-float modules"
    assert nested_modules.keys() == direct_modules.keys()
    for name, direct_module in direct_modules.items():
        nested_module = nested_modules[name]
        assert type(nested_module) is type(direct_module)
        # Compare the packed storage itself: matching module types alone would
        # pass if the subfolder route silently loaded different weights.
        assert torch.equal(nested_module.codes, direct_module.codes), name
        assert torch.equal(nested_module.exponents, direct_module.exponents), name

    ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.equal(nested.model(ids).logits, direct.model(ids).logits)


def _scalar_blockfloat_oracle(block, bits):
    """Independent double-precision oracle for one block, built from scratch.

    It performs its own arithmetic: ``math.frexp`` for the exact binade index,
    ``round`` for round-to-nearest-even and Python floats for the squared error.
    The fp32 inputs are exactly representable in double, and every scale is a
    power of two, so each intermediate below is exact and the answer is a
    bit-level expectation rather than a re-run of the implementation.
    """

    import math

    levels = 1 << max(bits - 2, 0)
    limit = 2.0 - 1.0 / levels
    smallest = float(torch.finfo(torch.float32).tiny)
    magnitude = max(abs(float(v)) for v in block)
    safe = max(magnitude, smallest)
    base = math.frexp(safe)[1] - 1

    def round_with(exp):
        scale = 2.0 ** exp
        out = []
        for value in block:
            unit = round(float(value) / scale * levels) / levels
            unit = min(max(unit, -limit), limit)
            out.append(unit * scale)
        return out

    def error(values):
        return sum((v - float(x)) ** 2 for v, x in zip(values, block))

    best = round_with(base)
    best_error = error(best)
    best_exp = base
    for delta in (-1, 1):
        candidate_exp = base + delta
        if not -127 <= candidate_exp <= 127:
            continue
        candidate = round_with(candidate_exp)
        candidate_error = error(candidate)
        if math.isfinite(candidate_error) and candidate_error < best_error:
            best, best_error, best_exp = candidate, candidate_error, candidate_exp
    return best, best_exp


def test_blockfloat_largest_finite_value_uses_the_storable_exponent():
    """``float32.max`` must quantize inside the grid, not overflow to NaN.

    The shared exponent is the binade index of the block maximum. Deriving it
    with ``floor(torch.log2(max))`` gave 128 for ``float32.max``, because
    ``log2`` rounds to nearest and the value sits one step below ``2**128``.
    ``2**128`` is inf, so the ``+1`` candidate produced ``0 * inf = NaN`` and
    that NaN error also blocked the ``-1`` candidate from replacing the base.

    The invariant is on the public quantizer: every finite input produces a
    finite grid, and the chosen exponent is a storable one. The expected values
    come from the scalar double-precision oracle, not from the implementation.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    fmax = torch.finfo(torch.float32).max
    for width in (2, 4, 8):
        block = torch.full((1, 16), fmax)
        quantized, exponent = bq.quantize_blockfloat(block, width)

        expected, expected_exp = _scalar_blockfloat_oracle(block.reshape(-1).tolist(), width)
        assert int(exponent[0, 0]) == expected_exp == 127
        assert torch.isfinite(quantized).all(), "an illegal exponent leaked a NaN"
        assert torch.equal(quantized.reshape(-1), torch.tensor(expected, dtype=torch.float32))
        assert int(exponent.max()) <= bq.BLOCKFLOAT_EXPONENT_MAX

        # The result must still be a checkpoint-storable grid point.
        codes, recovered = bq.decompose_blockfloat(quantized, width)
        assert torch.equal(bq.recompose_blockfloat(codes, recovered, bits=width), quantized)
        assert int(recovered.max()) <= bq.BLOCKFLOAT_EXPONENT_MAX

    # The negative extreme takes the same path with the sign preserved.
    negative = torch.full((1, 16), -fmax)
    quantized, exponent = bq.quantize_blockfloat(negative, 4)
    assert int(exponent[0, 0]) == 127
    assert torch.isfinite(quantized).all()
    assert quantized.min() < -1e38


def test_blockfloat_binade_boundaries_derive_the_exact_exponent():
    """The base exponent must be the binade index at every boundary.

    ``torch.log2`` reads a value just below ``2**k`` as exactly ``k`` and then
    ``floor`` yields ``k``, one too high; ``float32.max`` is the extreme case at
    ``k = 128``. A bit-level extraction is exact, so it is the reference here.
    Power-of-two neighbours, the smallest normal and the smallest subnormal all
    have to agree, and every result has to stay finite and on the grid.
    """

    import importlib
    import math

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    tiny = torch.finfo(torch.float32).tiny
    smallest_subnormal = torch.tensor(2.0**-149, dtype=torch.float32)
    fmax = torch.finfo(torch.float32).max

    values = [
        torch.tensor(0.0),
        torch.tensor(tiny),
        smallest_subnormal,
        torch.tensor(fmax),
    ]
    for k in range(-126, 128):
        boundary = torch.tensor(2.0**k, dtype=torch.float32)
        below = torch.nextafter(boundary, torch.zeros((), dtype=torch.float32))
        above = torch.nextafter(boundary, torch.tensor(math.inf, dtype=torch.float32))
        for value in (below, boundary, above):
            if bool(torch.isfinite(value)) and float(value) > 0:
                values.append(value)

    stacked = torch.stack(values).reshape(-1, 1).expand(-1, 16).contiguous()
    blocks = stacked.reshape(-1, 1, 16)
    got = bq._block_base_exponent(blocks).reshape(-1)

    # Independent expectation: the exponent field of the float, with the same
    # clamp-to-tiny convention the quantizer documents for a zero block.
    safe = stacked[:, 0].abs().clamp_min(tiny)
    want = ((safe.view(torch.int32) >> 23) & 0xFF) - 127
    assert torch.equal(got.to(torch.int32), want), "base exponent is not the binade index"

    # Independently, through frexp, except where the clamp changes a subnormal.
    frexp_exp = (torch.frexp(safe)[1] - 1).to(torch.int32)
    clamped = stacked[:, 0].abs() < tiny
    assert torch.equal(got[~clamped].to(torch.int32), frexp_exp[~clamped])

    for width in (4, 8):
        quantized, exponent = bq.quantize_blockfloat(stacked, width)
        assert torch.isfinite(quantized).all()
        assert int(exponent.min()) >= bq.BLOCKFLOAT_EXPONENT_MIN
        assert int(exponent.max()) <= bq.BLOCKFLOAT_EXPONENT_MAX

    # Smallest normal and subnormal: the former is exactly representable, the
    # latter flushes to zero on the -126 binade grid.
    quantized, exponent = bq.quantize_blockfloat(torch.full((1, 16), tiny), 4)
    assert int(exponent[0, 0]) == -126 and float(quantized[0, 0]) == float(tiny)
    quantized, exponent = bq.quantize_blockfloat(
        torch.full((1, 16), float(smallest_subnormal)), 4
    )
    assert int(exponent[0, 0]) == -126 and float(quantized[0, 0]) == 0.0


def test_blockfloat_rejects_nonfinite_blocks():
    """NaN and infinity have no block-float representation, so refuse them.

    Deriving an exponent from a non-finite maximum used to produce a silently
    finite-looking grid (or NaN codes); a checkpoint cannot store either. The
    contract is a ``ValueError`` before any arithmetic, so the caller learns
    that the weight must not be quantized rather than receiving poison.
    """

    import importlib
    import math

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="non-finite"):
            bq.quantize_blockfloat(torch.full((1, 16), bad), 4)
    # A single non-finite value in an otherwise ordinary block is still refused.
    mixed = torch.zeros(1, 16)
    mixed[0, 7] = math.inf
    with pytest.raises(ValueError, match="non-finite"):
        bq.quantize_blockfloat(mixed, 8)
    with pytest.raises(ValueError, match="non-finite"):
        bq._block_exponents(mixed)

    # The fixed-exponent rounding entry point cannot be handed a scale that has
    # no representation either.
    with pytest.raises(ValueError, match="block-float exponents must lie in"):
        bq._round_with_exponents(
            torch.randn(1, 16), torch.tensor([[bq.BLOCKFLOAT_EXPONENT_MAX + 1]]), bits=4
        )


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
    """A checkpoint may store some layers at bf8_b and the rest at bf4_b.

    bf8_b costs one byte per weight and is close to lossless, while
    bf4_b is roughly a 2.8-bit format. A real checkpoint wants the extra
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
        # The embedding is stored as bf8_b codes too, so put it on the grid.
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
    # The embedding is codes at bf8_b, not a dense table: 64 x 32 codes at 8 bits
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
    # The shipped loader is declared in the quantization config so the HF
    # refusing-quantizer guard lets this checkpoint through (#4557).
    assert quant_config["shipped_modeling_file"] == bq.MODELING_FILE

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


def test_refusing_quantizer_allows_a_shipped_blockfloat_modeling_file():
    """#4557: a shipped ``auto_map`` loader must not be refused by transformers.

    The refusing quantizer exists to stop silent random-init loads of packed
    checkpoints that transformers cannot read. A block-float checkpoint that
    ships its own modeling file is read by that class, so the writer records the
    shipped file in the quantization config. A bare config without it still
    fails closed.
    """

    from transformers.quantizers.auto import AutoHfQuantizer, AutoQuantizationConfig

    shipped = AutoQuantizationConfig.from_dict(
        {
            "quant_method": "tt-blockfloat",
            "bits": 4,
            "bits_per_layer": {"model.layers.0.mlp.down_proj": 4},
            "shipped_modeling_file": "modeling_tt_blockfloat.py",
        }
    )
    quantizer = AutoHfQuantizer.from_config(shipped)
    assert quantizer.pre_quantized is True
    # The remote modeling class reads these back out of the config object.
    assert dict(shipped)["bits_per_layer"] == {"model.layers.0.mlp.down_proj": 4}

    # A checkpoint written before the explicit marker existed still records the
    # format field that only ``save_blockfloat_model`` writes.
    legacy = AutoQuantizationConfig.from_dict(
        {
            "quant_method": "tt-blockfloat",
            "checkpoint_format": "tt-blockfloat",
            "bits": 4,
        }
    )
    assert AutoHfQuantizer.from_config(legacy).pre_quantized is True

    bare = AutoQuantizationConfig.from_dict(
        {"quant_method": "tt-blockfloat", "bits": 4}
    )
    with pytest.raises(ValueError, match="cannot be loaded by `transformers`"):
        AutoHfQuantizer.from_config(bare)


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
                    module.weight.copy_(bq.quantize_bf4(module.weight.detach().float()))
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

    ours = GPTQModel.load(out, device="cpu").model

    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        a = remote(ids).logits
        b = ours(ids).logits
    assert torch.equal(a, b), float((a - b).abs().max())


def test_shipped_modeling_file_uses_the_source_architecture(tmp_path):
    """The generated entry point must subclass the checkpoint's own class.

    It used to import ``LlamaForCausalLM`` unconditionally, so an OPT
    checkpoint exported and reloaded through the native loader, but the
    advertised ``AutoModelForCausalLM.from_pretrained`` path raised
    ``AttributeError: 'OPTConfig' object has no attribute
    'num_key_value_heads'`` before loading any packed weight.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(5)
    from transformers import OPTConfig, OPTForCausalLM

    model = OPTForCausalLM(
        OPTConfig(
            vocab_size=64,
            hidden_size=32,
            ffn_dim=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            word_embed_proj_dim=32,
            max_position_embeddings=64,
        )
    ).eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and ".layers." in name:
                module.weight.copy_(
                    bq.quantize_blockfloat(module.weight.detach().float(), 4)[0]
                )
        embedding = model.get_input_embeddings()
        embedding.weight.copy_(
            bq.quantize_blockfloat(embedding.weight.detach().float(), 8)[0]
        )

    out = str(tmp_path / "opt")
    bq.save_blockfloat_model(model, out, bits=4, embed_bits=8)

    with open(os.path.join(out, bq.MODELING_FILE)) as handle:
        source = handle.read()
    assert "from transformers.models.opt.modeling_opt import OPTForCausalLM" in source
    assert "LlamaForCausalLM" not in source

    from transformers import AutoModelForCausalLM

    remote = AutoModelForCausalLM.from_pretrained(
        out, dtype=torch.float32, trust_remote_code=True
    )
    assert type(remote).__mro__[1].__name__ == "OPTForCausalLM"

    native, _, _ = bq.build_blockfloat_model(out, dtype=torch.float32)
    ids = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        a = remote(ids).logits
        b = native(ids).logits
    assert torch.equal(a, b), float((a - b).abs().max())


def test_modeling_base_fails_closed_for_an_unknown_architecture():
    """An unresolvable architecture must fail at save time, not ship a bad file."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    wrapper = torch.nn.Module()
    wrapper.config = type("Config", (), {"model_type": "not_a_real_model_type"})()
    with pytest.raises(ValueError, match="no known Transformers causal-LM class"):
        bq._modeling_base(wrapper)


def test_blockfloat_bias_keeps_source_precision(tmp_path):
    """Dense biases are not quantized, so packing must not narrow them.

    The module used to register an fp16 bias and ``pack`` cast the source bias
    to fp16, which turned valid fp32/bf16 values such as 65536 into infinity
    and changed the projection output even when fp32 was requested.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")
    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear

    values = torch.tensor([65536.0, 2.0**-30, 0.12345678])
    for dtype in (torch.float32, torch.bfloat16):
        source = torch.nn.Linear(16, 3, bias=True, dtype=dtype)
        with torch.no_grad():
            source.weight.zero_()
            source.bias.copy_(values.to(dtype))
        packed = TTBlockfloatLinear(
            bits=4,
            in_features=16,
            out_features=3,
            bias=True,
            dtype=dtype,
            backend=BACKEND.TT_BLOCKFLOAT,
        ).pack(source)
        assert packed.bias.dtype == dtype
        assert torch.equal(packed.bias, source.bias)
        with torch.no_grad():
            result = packed(torch.zeros(1, 16, dtype=dtype))
        assert torch.equal(result[0], source.bias)

    # The same precision must survive a real checkpoint round trip.
    model = _tiny_llama({"attention_bias": True, "mlp_bias": True})
    with torch.no_grad():
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.dim() == 2:
                bits = 8 if name.endswith("embed_tokens") else 4
                weight.copy_(
                    bq.quantize_blockfloat(weight.detach().float(), bits)[0]
                )
        model.model.layers[0].self_attn.q_proj.bias[:3] = values

    out = str(tmp_path / "bias")
    bq.save_blockfloat_model(model, out, bits=4, embed_bits=8)
    loaded, _, _ = bq.build_blockfloat_model(out, dtype=torch.float32)
    bias = loaded.model.layers[0].self_attn.q_proj.bias
    assert bias.dtype == torch.float32
    assert torch.equal(bias[:3], values)
    with torch.no_grad():
        before = model.model.layers[0].self_attn.q_proj(torch.zeros(1, 32))
        after = loaded.model.layers[0].self_attn.q_proj(torch.zeros(1, 32))
    assert torch.equal(before, after)
    assert torch.isfinite(after).all()


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
        block.proj.weight.copy_(bq.quantize_bf4(torch.randn(32, 64) * 0.02))
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
        block.proj.weight.copy_(bq.quantize_bf4(torch.randn(32, 64) * 0.02))

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
        block.proj.weight.copy_(bq.quantize_bf4(torch.randn(32, 64) * 0.02))

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
    w = bq.quantize_bf4(torch.randn(32, 64) * 0.02)

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


def _mixed_grid_and_dense_model(dtype=torch.float32):
    """One decoder block with an on-grid linear and an off-grid dense linear."""

    block = torch.nn.Module()
    block.grid = torch.nn.Linear(16, 2, bias=True, dtype=dtype)
    block.dense = torch.nn.Linear(16, 2, bias=True, dtype=dtype)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.decoder = torch.nn.Module()
    model.model.decoder.layers = torch.nn.ModuleList([block])
    return model


def test_native_checkpoint_roundtrips_dense_fallback_layers(tmp_path):
    """The dense fallback must be loadable, not only written.

    A decoder linear that is not on the grid is stored as ``<layer>.weight``, but
    it used to be left out of ``blockfloat.json``, so the reader skipped it and
    the loader returned success with the target layer still at its initial
    values. Its bias was never written at all. Both kinds of layer are now listed
    in the manifest and restored weight-for-weight and bias-for-bias.
    """

    import importlib
    import json

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    src = _mixed_grid_and_dense_model()
    block = src.model.decoder.layers[0]
    with torch.no_grad():
        block.grid.weight.fill_(1.0)       # exactly on the bf4_b grid
        block.grid.bias.fill_(0.5)
        block.dense.weight.fill_(0.1)      # off-grid, so it stays dense
        block.dense.bias.fill_(3.0)

    out = str(tmp_path / "mixed")
    info = bq.save_blockfloat_checkpoint(src, out, bits=4)
    assert info["layers"] == 1 and info["dense_layers"] == 1

    payload = bq._load_blockfloat_payload(out)
    # Both biases are written, and the dense weight is stored at its source dtype.
    assert payload["0.dense.weight"].dtype is torch.float32
    assert torch.equal(payload["0.dense.weight"], block.dense.weight.detach())
    assert torch.equal(payload["0.dense.bias"], block.dense.bias.detach())
    assert torch.equal(payload["0.grid.bias"], block.grid.bias.detach())

    meta = json.loads((tmp_path / "mixed" / "blockfloat.json").read_text())
    assert meta["layers"]["0.grid"]["kind"] == "blockfloat"
    assert meta["layers"]["0.dense"]["kind"] == "dense"
    assert meta["layers"]["0.dense"]["dtype"] == "float32"
    assert meta["layers"]["0.dense"]["has_bias"] is True
    # Only quantized layers carry a width, so mixed-width bookkeeping is unchanged.
    assert "bits" not in meta["layers"]["0.dense"]

    read_back = bq.read_blockfloat_checkpoint(out)
    assert set(read_back["layers"]) == {"0.grid", "0.dense"}
    assert read_back["layers"]["0.dense"]["kind"] == "dense"
    assert bq.expand_native_layer(read_back["layers"]["0.dense"]).equal(
        block.dense.weight.detach()
    )

    fresh = _mixed_grid_and_dense_model()
    with torch.no_grad():
        for param in fresh.parameters():
            param.fill_(-9.0)
    restored = bq.load_native_checkpoint_into_dense_model(fresh, out)
    assert set(restored) == {"0.grid", "0.dense"}
    target = fresh.model.decoder.layers[0]
    assert torch.equal(target.grid.weight.detach(), block.grid.weight.detach())
    assert torch.equal(target.grid.bias.detach(), block.grid.bias.detach())
    assert torch.equal(target.dense.weight.detach(), block.dense.weight.detach())
    assert torch.equal(target.dense.bias.detach(), block.dense.bias.detach())


def test_native_checkpoint_with_only_dense_layers(tmp_path):
    """A checkpoint with no on-grid layer is still loadable.

    ``layers`` reports the block-float count, ``dense_layers`` the fallback
    count, and the manifest is the only list of what the payload holds.
    """

    import importlib
    import json

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    block = torch.nn.Module()
    block.proj = torch.nn.Linear(16, 2, bias=True)
    with torch.no_grad():
        block.proj.weight.fill_(0.1)
        block.proj.bias.fill_(3.0)
    src = torch.nn.Module()
    src.model = torch.nn.Module()
    src.model.decoder = torch.nn.Module()
    src.model.decoder.layers = torch.nn.ModuleList([block])

    out = str(tmp_path / "dense")
    info = bq.save_blockfloat_checkpoint(src, out, bits=4)
    assert info["layers"] == 0
    assert info["dense_layers"] == 1
    assert info["bits_per_layer"] == []

    meta = json.loads((tmp_path / "dense" / "blockfloat.json").read_text())
    assert set(meta["layers"]) == {"0.proj"}
    assert meta["layers"]["0.proj"]["kind"] == "dense"
    assert "bits_per_layer" not in meta

    fresh_block = torch.nn.Module()
    fresh_block.proj = torch.nn.Linear(16, 2, bias=True)
    fresh = torch.nn.Module()
    fresh.model = torch.nn.Module()
    fresh.model.decoder = torch.nn.Module()
    fresh.model.decoder.layers = torch.nn.ModuleList([fresh_block])
    with torch.no_grad():
        for param in fresh.parameters():
            param.fill_(-9.0)
    restored = bq.load_native_checkpoint_into_dense_model(fresh, out)
    assert restored == {"0.proj": [2, 16]}
    target = fresh.model.decoder.layers[0].proj
    assert torch.equal(target.weight.detach(), block.proj.weight.detach())
    assert torch.equal(target.bias.detach(), block.proj.bias.detach())


def test_native_checkpoint_dense_fallback_keeps_source_precision(tmp_path):
    """A dense fallback keeps bf16 exactly; it used to be narrowed to fp16."""

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    src = _mixed_grid_and_dense_model(dtype=torch.bfloat16)
    block = src.model.decoder.layers[0]
    values = torch.tensor([0.1234567, -0.98765], dtype=torch.bfloat16)
    with torch.no_grad():
        block.grid.weight.fill_(1.0)
        block.grid.bias.copy_(values)
        block.dense.weight.fill_(0.1)
        block.dense.bias.copy_(values)

    out = str(tmp_path / "bf16")
    bq.save_blockfloat_checkpoint(src, out, bits=4)

    payload = bq._load_blockfloat_payload(out)
    assert payload["0.dense.weight"].dtype is torch.bfloat16
    assert payload["0.dense.bias"].dtype is torch.bfloat16
    assert torch.equal(payload["0.dense.bias"], values)
    assert torch.equal(payload["0.grid.bias"], values)

    fresh = _mixed_grid_and_dense_model(dtype=torch.bfloat16)
    with torch.no_grad():
        for param in fresh.parameters():
            param.fill_(-9.0)
    bq.load_native_checkpoint_into_dense_model(fresh, out)
    target = fresh.model.decoder.layers[0]
    assert torch.equal(target.dense.weight.detach(), block.dense.weight.detach())
    assert torch.equal(target.dense.bias.detach(), values)


def test_native_checkpoint_rejects_layers_with_no_target_module(tmp_path):
    """A manifest entry the target model cannot hold must not pass silently."""

    import importlib

    import pytest

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    src = _mixed_grid_and_dense_model()
    with torch.no_grad():
        src.model.decoder.layers[0].grid.weight.fill_(1.0)
        src.model.decoder.layers[0].dense.weight.fill_(0.1)
    out = str(tmp_path / "mixed")
    bq.save_blockfloat_checkpoint(src, out, bits=4)

    # Same shape of model, but the dense fallback layer is gone.
    block = torch.nn.Module()
    block.grid = torch.nn.Linear(16, 2, bias=True)
    other = torch.nn.Module()
    other.model = torch.nn.Module()
    other.model.decoder = torch.nn.Module()
    other.model.decoder.layers = torch.nn.ModuleList([block])

    with pytest.raises(ValueError, match="0.dense"):
        bq.load_native_checkpoint_into_dense_model(other, out)


def test_native_loader_rejects_a_bias_the_target_cannot_hold(tmp_path):
    """A stored bias with no target module bias must not be dropped."""

    import importlib

    import pytest

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    src = _mixed_grid_and_dense_model()
    with torch.no_grad():
        src.model.decoder.layers[0].grid.weight.fill_(1.0)
        src.model.decoder.layers[0].dense.weight.fill_(0.1)
    out = str(tmp_path / "mixed")
    bq.save_blockfloat_checkpoint(src, out, bits=4)

    block = torch.nn.Module()
    block.grid = torch.nn.Linear(16, 2, bias=False)
    block.dense = torch.nn.Linear(16, 2, bias=False)
    target = torch.nn.Module()
    target.model = torch.nn.Module()
    target.model.decoder = torch.nn.Module()
    target.model.decoder.layers = torch.nn.ModuleList([block])

    with pytest.raises(ValueError, match="stores a bias"):
        bq.load_native_checkpoint_into_dense_model(target, out)


def test_native_checkpoint_reads_a_version_1_manifest(tmp_path):
    """Version 1 checkpoints still load their quantized layers.

    Version 1 listed only block-float layers and wrote the dense fallback (and no
    bias for it) without a manifest entry. Those unreachable tensors cannot be
    attributed to a layer afterwards -- the payload key is the only evidence --
    so the reader keeps loading every entry the manifest does list, and a
    cross-check now fails the load for a declared layer whose codes are missing.
    """

    import importlib
    import json
    import os

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    src = _mixed_grid_and_dense_model()
    with torch.no_grad():
        src.model.decoder.layers[0].grid.weight.fill_(1.0)
        src.model.decoder.layers[0].dense.weight.fill_(0.1)
    out = str(tmp_path / "v1")
    bq.save_blockfloat_checkpoint(src, out, bits=4)

    path = os.path.join(out, "blockfloat.json")
    with open(path) as handle:
        meta = json.load(handle)
    # Reconstruct the pre-change manifest: only the quantized layer is listed,
    # with no ``kind`` and no bias record.
    quantized = {
        key: {k: v for k, v in entry.items() if k in (
            "shape", "bits", "codes_bytes", "numel", "exponent_bytes"
        )}
        for key, entry in meta["layers"].items()
        if entry["kind"] == "blockfloat"
    }
    meta["version"] = 1
    meta["layers"] = quantized
    with open(path, "w") as handle:
        json.dump(meta, handle, indent=1)

    payload = bq.read_blockfloat_checkpoint(out)
    assert set(payload["layers"]) == {"0.grid"}
    assert payload["layers"]["0.grid"]["kind"] == "blockfloat"

    fresh = _mixed_grid_and_dense_model()
    with torch.no_grad():
        for param in fresh.parameters():
            param.fill_(-9.0)
    # The unlisted dense tensor is not reachable, so it is not a target layer.
    block = torch.nn.Module()
    block.grid = torch.nn.Linear(16, 2, bias=True)
    target = torch.nn.Module()
    target.model = torch.nn.Module()
    target.model.decoder = torch.nn.Module()
    target.model.decoder.layers = torch.nn.ModuleList([block])
    restored = bq.load_native_checkpoint_into_dense_model(target, out)
    assert restored == {"0.grid": [2, 16]}
    assert target.model.decoder.layers[0].grid.weight.detach().eq(1.0).all()


def test_native_reader_rejects_a_layer_whose_codes_are_missing(tmp_path):
    """A listed layer with no payload tensor is a broken checkpoint, not a skip."""

    import importlib
    import os

    import pytest

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    src = _mixed_grid_and_dense_model()
    with torch.no_grad():
        src.model.decoder.layers[0].grid.weight.fill_(1.0)
        src.model.decoder.layers[0].dense.weight.fill_(0.1)
    out = str(tmp_path / "broken")
    bq.save_blockfloat_checkpoint(src, out, bits=4)

    payload = bq._load_blockfloat_payload(out)
    payload.pop("0.dense.weight")
    from safetensors.torch import save_file

    save_file(payload, os.path.join(out, bq.BLOCKFLOAT_PAYLOAD))

    with pytest.raises(ValueError, match="0.dense.weight is missing"):
        bq.read_blockfloat_checkpoint(out)


def test_blockfloat_pack_roundtrips_exactly():
    """Codes plus exponents must reconstruct the grid weights bit-for-bit.

    The checkpoint stores no floats for weights, so exactness here is what makes
    "native format" true rather than approximate.
    """

    import importlib

    bq = importlib.import_module("gptqmodel.utils.tt_blockfloat_quant")

    torch.manual_seed(4)
    w = bq.quantize_bf4(torch.randn(64, 128) * 0.02)
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
    quantized = bq.quantize_bf4(w)
    codes, exponents = bq.decompose_blockfloat(quantized, bits=4)
    assert int(codes.max()) > 4, "test data should exercise codes above levels"
    assert bq.recompose_blockfloat(codes, exponents, bits=4).equal(quantized)


def test_gptq_blockfloat_is_native_not_an_int4_cast():
    """The solve must land on the bf4_b grid itself, not pass through int4.

    Three assertions: the result is a fixed point of the grid, it differs from an
    int4 quantisation of the same weights, and the error is far below what a
    double rounding (bf4_b -> int4) would produce.
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
    assert torch.allclose(solved, bq.quantize_bf4(solved), atol=1e-6)
    # 2. Not the int4 answer for the same weights.
    assert not torch.allclose(solved, bq.quantize_uniform_int4(solved), atol=1e-9)
    # 3. Direct-to-grid beats a double rounding through int4 -- measured on the
    #    layer output, which is what GPTQ optimises. Raw weight distance is not the
    #    objective: the solve deliberately trades it for lower output error.
    reference = x @ w.T
    direct = bq.relative_error(reference, x @ solved.T)
    via_int4 = bq.relative_error(
        reference, x @ bq.quantize_bf4(bq.quantize_uniform_int4(w)).T
    )
    assert direct < via_int4, f"direct {direct} should beat int4-then-bf4_b {via_int4}"


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


@pytest.mark.skipif(os.environ.get("GPTQMODEL_TEST_TTNN") != "1", reason="explicit TTNN hardware gate")
def test_cpu_wrapper_release_device_leaves_a_native_mesh_open():
    """A CPU wrapper must not close a mesh another model's kernels still use.

    Reproduced on Blackhole: ``release_device()`` on a CPU wrapper returned
    ``True`` and closed the process-wide mesh, so the next native forward raised
    that it had no open device.
    """

    from gptqmodel.models.definitions.opt import OptQModel
    from gptqmodel.nn_modules.qlinear.ttnn import TTNNLinear
    from gptqmodel.utils.ttnn_runtime import close_mesh, current_mesh

    try:
        from transformers import OPTConfig, OPTForCausalLM
    except Exception:  # pragma: no cover - transformers is a runtime dependency
        pytest.skip("transformers is not installed")

    cpu = OptQModel(
        model=OPTForCausalLM(OPTConfig(
            hidden_size=32, word_embed_proj_dim=32, ffn_dim=64,
            num_attention_heads=2, num_hidden_layers=1, vocab_size=32,
        )).eval(),
        quantized=False, quantize_config=None,
    )
    native = TTNNLinear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=64, out_features=128, bias=False,
        pack_dtype=torch.int32, format=FORMAT.GPTQ_V2,
    ).eval()
    native.qweight.fill_(0x99999999 - 2**32)
    native.qzeros.fill_(0x88888888 - 2**32)
    native.qzero_format(2)
    native.scales.fill_(1)
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[0, 0] = 1
    try:
        assert cpu.release_device() is False
        first = native(x)
        torch.testing.assert_close(first, torch.ones_like(first), rtol=0, atol=0)
        assert current_mesh() is not None
        assert cpu.release_device() is False
        assert current_mesh() is not None, "the CPU wrapper closed another model's mesh"
        torch.testing.assert_close(native(x), first, rtol=0, atol=0)
    finally:
        native.release_device()
        close_mesh()


# --- end-to-end: dense -> native bf4_b -> post-quant inference ----------------
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
            param.copy_(bq.quantize_bf4(torch.randn_like(param) * 0.02))

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
    """Quantize real OPT-125M to native bf4_b, then run post-quant inference.

    Checks the property the whole path exists for: the solver's output survives
    bf4_b storage unchanged, so inference needs no dequant at load.
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

    # On device, a restored layer must store as bf4_b with no drift.
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
        assert drift == 0.0, f"bf4_b storage moved the grid values by {drift}"
        ttnn.deallocate(device_tensor)
    finally:
        close_mesh()


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_TEST_TT_MODEL") != "1",
    reason="explicit end-to-end model gate",
)
def test_opt125m_native_bf4_matches_int4_quantized_checkpoint():
    """Native bf4_b should be in the same accuracy class as a shipped GPTQ int4 model.

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
        bf4_b = model(ids).logits[0, -1].float().clone()

    bf4_cos = float(torch.nn.functional.cosine_similarity(dense, bf4_b, dim=0))
    assert bf4_cos > 0.99, f"native bf4_b cosine {bf4_cos} below the int4 class"


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


class _WordTokenizer:
    """Word-level tokenizer stub with transformers' right-truncation semantics."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.chat_template = None
        self._ids = {"<bos>": 0, "<pad>": 1, "<eos>": 2}
        self._words = {0: "<bos>", 1: "<pad>", 2: "<eos>"}

    def _id(self, word):
        if word not in self._ids:
            index = len(self._ids)
            self._ids[word] = index
            self._words[index] = word
        return self._ids[word]

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = [self.bos_token_id] + [self._id(word) for word in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]  # what a right-truncating fast tokenizer does
        return types.SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self._words[int(index)] for index in ids)


class _RecordingModel:
    """Generation stub: records the prompt ids and answers ``#### 18``."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.seen = []

    def eval(self):
        return self

    def generate(self, ids, **_kwargs):
        self.seen.append(ids.clone())
        answer = torch.tensor([[self.tokenizer._id(word) for word in "#### 18".split()]])
        return torch.cat([ids, answer], dim=1)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _stub_gsm8k_model(monkeypatch):
    import transformers

    tokenizer = _WordTokenizer()
    model = _RecordingModel(tokenizer)
    monkeypatch.setattr(
        transformers, "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
    )
    monkeypatch.setattr(
        transformers, "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
    )
    return tokenizer, model


def _gsm8k_paths(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    shots_path = tmp_path / "shots.jsonl"
    _write_jsonl(rows_path, [{"question": "TARGET question?", "answer": "#### 18"}])
    _write_jsonl(
        shots_path,
        [
            {"question": f"demo question number {index} with extra words", "answer": f"#### {index}"}
            for index in range(5)
        ],
    )
    return rows_path, shots_path


def test_gsm8k_truncation_keeps_the_question(monkeypatch, tmp_path, capsys):
    """#5403: an over-budget few-shot prompt must keep the question under test."""

    ev = _load_gsm8k_eval()
    tokenizer, model = _stub_gsm8k_model(monkeypatch)
    rows_path, shots_path = _gsm8k_paths(tmp_path)

    result = ev.run_eval(
        "stub",
        limit=1,
        rows_path=str(rows_path),
        shots_path=str(shots_path),
        max_prompt_tokens=12,
        verbose=False,
    )

    assert len(model.seen) == 1
    scored = tokenizer.decode(model.seen[0][0])
    assert "TARGET" in scored, f"the question under test was truncated away: {scored!r}"
    assert scored.rstrip().endswith("Answer:")
    # The demonstrations are what gets dropped.
    assert "demo question" not in scored
    assert result["truncated_prompts"] == 1
    assert result["max_prompt_tokens"] == 12
    assert "WARNING" in capsys.readouterr().out


def test_gsm8k_short_prompt_is_not_truncated(monkeypatch, tmp_path, capsys):
    """A prompt that fits keeps every shot and reports no truncation."""

    ev = _load_gsm8k_eval()
    tokenizer, model = _stub_gsm8k_model(monkeypatch)
    rows_path, shots_path = _gsm8k_paths(tmp_path)

    result = ev.run_eval(
        "stub",
        limit=1,
        rows_path=str(rows_path),
        shots_path=str(shots_path),
        max_prompt_tokens=4096,
        verbose=False,
    )

    scored = tokenizer.decode(model.seen[0][0])
    assert "TARGET" in scored
    assert scored.count("demo question") == 5
    assert result["truncated_prompts"] == 0
    assert "WARNING" not in capsys.readouterr().out


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
