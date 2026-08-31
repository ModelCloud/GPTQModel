# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reference coverage for the legacy QVQ LR32 layout."""

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.qvq import (
    QVQLinear,
    qvq_local_ring_dense_oracle_forward,
)
from gptqmodel.quantization.config import FORMAT, QVQConfig
from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_LR_RING_STEPS,
    QVQ_V2B2_P32_LR_RINGS_PER_TILE,
    decode_local_ring_states,
    decode_local_ring_tiles,
    local_ring_states_from_edges,
    pack_local_ring_states,
    pack_qvq_binary_bank_ids,
    reconstruct_local_ring_inner_weight,
    unpack_local_ring_edges,
    unpack_local_ring_states,
)
from gptqmodel.quantization.qvq_codecs import (
    pgc16_decode_states_v2_banked,
    pgc16_levels_for_version,
)
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import make_quant

LR_RATES = (1, 1.5, 2, 2.5, 3, 3.5)


def _random_lr_payload(bits: float, tiles: int = 2):
    transition_bits = int(bits * 2)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (tiles, QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS),
        dtype=torch.int64,
    )
    states = local_ring_states_from_edges(edges, bits=bits)
    trellis = pack_local_ring_states(states, bits=bits)
    selectors = torch.randint(
        0,
        2,
        (tiles * QVQ_V2B2_P32_LR_RINGS_PER_TILE,),
        dtype=torch.uint8,
    )
    return edges, states, trellis, selectors


@pytest.mark.parametrize("bits", LR_RATES)
def test_lr32_planar_pack_roundtrip_all_rates(bits):
    edges, states, trellis, _ = _random_lr_payload(bits)

    assert trellis.dtype == torch.int32
    assert torch.equal(unpack_local_ring_edges(trellis, bits=bits), edges)
    assert torch.equal(unpack_local_ring_states(trellis, bits=bits), states)
    assert torch.equal(decode_local_ring_states(trellis, bits=bits), states)


def test_lr32_rings_are_independent():
    edges, states, _, _ = _random_lr_payload(2, tiles=1)
    changed = edges.clone()
    changed[..., 1, :] ^= 3

    changed_states = local_ring_states_from_edges(changed, bits=2)
    assert torch.equal(changed_states[..., 0, :], states[..., 0, :])
    assert not torch.equal(changed_states[..., 1, :], states[..., 1, :])


@pytest.mark.parametrize("bits", (1, 2, 3, 3.5))
def test_lr32_decode_matches_banked_codec(bits):
    _, states, trellis, selectors = _random_lr_payload(bits, tiles=3)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)

    actual = decode_local_ring_tiles(
        trellis,
        bits=bits,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    bank_ids = selectors.reshape(3, 8).repeat_interleave(16, dim=1).mul(2)
    expected = pgc16_decode_states_v2_banked(
        states.reshape(3, 128),
        bank_ids,
        bits=bits,
        levels=pgc16_levels_for_version("pgc16-v1"),
    ).reshape(3, 8, 16, 2).to(torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_lr32_reconstruction_uses_k32_n8_logical_tiles():
    _, _, trellis, selectors = _random_lr_payload(2, tiles=4)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([1], dtype=torch.uint8)
    decoded = decode_local_ring_tiles(
        trellis,
        bits=2,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    expected = (
        decoded.reshape(2, 2, 8, 16, 2)
        .reshape(2, 2, 8, 32)
        .permute(0, 3, 1, 2)
        .reshape(64, 16)
    )
    actual = reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=64,
        out_features=16,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _make_torch_lr_layer(bits=2, in_features=64, out_features=16):
    _, _, trellis, selectors = _random_lr_payload(
        bits,
        tiles=(in_features // 32) * (out_features // 8),
    )
    return QVQLinear(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_count=2,
        v2b2_p32_lr=True,
        dtype=torch.float32,
        tensors={
            "trellis": trellis,
            "SU": torch.rand(in_features, dtype=torch.float32) + 0.5,
            "SV": torch.rand(out_features, dtype=torch.float32) + 0.5,
            "bank_ids": pack_qvq_binary_bank_ids(selectors),
            "bank_alt_id": torch.tensor([2], dtype=torch.uint8),
        },
    ).eval()


def test_lr32_qvq_linear_oracle_uses_torch_reference_path():
    layer = _make_torch_lr_layer()
    x = torch.randn(5, layer.in_features, dtype=torch.float32)
    oracle = qvq_local_ring_dense_oracle_forward(layer, x)

    assert layer.v2b2_p32_lr
    assert layer.trellis.shape == (4, 16)
    assert oracle.shape == (5, layer.out_features)
    torch.testing.assert_close(layer(x), oracle, rtol=0, atol=0)


def test_lr32_qvq_linear_state_dict_round_trip_preserves_oracle():
    layer = _make_torch_lr_layer(bits=2, in_features=64, out_features=8)
    reloaded = QVQLinear(
        bits=2,
        in_features=64,
        out_features=8,
        bank_count=2,
        v2b2_p32_lr=True,
        dtype=torch.float32,
    ).eval()
    reloaded.load_state_dict(layer.state_dict(), strict=True)
    x = torch.randn(3, 64, dtype=torch.float32)

    for name, tensor in layer.state_dict().items():
        torch.testing.assert_close(tensor, reloaded.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(
        qvq_local_ring_dense_oracle_forward(reloaded, x),
        qvq_local_ring_dense_oracle_forward(layer, x),
        rtol=0,
        atol=0,
    )


def test_lr32_rejects_legacy_or_multiple_format_flags():
    with pytest.raises(ValueError, match="mutually exclusive"):
        QVQLinear(
            bits=2,
            in_features=32,
            out_features=8,
            bank_count=2,
            v2b2_p32=True,
            v2b2_p32_lr=True,
        )


def test_lr32_config_and_format_metadata():
    config = QVQConfig(bits=2, format=FORMAT.QVQ_V2B2_P32_LR, bank_count=2)
    assert config.format == FORMAT.QVQ_V2B2_P32_LR
    assert config.quant_linear_init_kwargs()["v2b2_p32_lr"] is True
    assert config.quant_linear_init_kwargs()["v2b2_p32"] is False
    assert QVQLinear.supported_bits(FORMAT.QVQ_V2B2_P32_LR) == (1, 1.5, 2, 2.5, 3, 3.5)


def test_lr32_make_quant_preserves_layout_format_for_n8_modules():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32, 8, bias=False, dtype=torch.float16)

    model = TinyModel()
    config = QVQConfig(
        bits=2,
        format=FORMAT.QVQ_V2B2_P32_LR,
        bank_count=2,
        rounding="block_ldlq",
        offload_to_disk=False,
    )

    selected = make_quant(
        model,
        config,
        {"proj": {}},
        BACKEND.QVQ,
        "lm_head",
        device="cpu",
        dtype=torch.float16,
    )

    assert selected is QVQLinear
    assert model.proj.v2b2_p32_lr is True
    assert model.proj.trellis.shape == (1, 16)


def test_lr32_reconstruction_pins_k32_n8_abi_coordinates():
    # Every tile/ring gets a distinct constant edge pattern. The expected
    # values below are scalar decoder lookups, not a second reshape/permute
    # implementation, so this test pins the serialized K32 x N8 ABI.
    edge_count = 4
    edges = torch.empty((4, 8, 16), dtype=torch.int64)
    for tile in range(4):
        for ring in range(8):
            edges[tile, ring] = (tile * 8 + ring + 1) % (1 << edge_count)
    states = local_ring_states_from_edges(edges, bits=2)
    trellis = pack_local_ring_states(states, bits=2)
    selectors = torch.zeros(32, dtype=torch.uint8)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    decoded = pgc16_decode_states_v2_banked(
        states.reshape(4, 128),
        selectors.reshape(4, 8).repeat_interleave(16, dim=1),
        bits=2,
        levels=pgc16_levels_for_version("pgc16-v1"),
    ).reshape(4, 8, 16, 2)
    actual = reconstruct_local_ring_inner_weight(
        trellis,
        bits=2,
        in_features=64,
        out_features=16,
        bank_ids=packed_selectors,
        bank_alt_id=torch.tensor([1], dtype=torch.uint8),
    )

    torch.testing.assert_close(actual[0, 0], decoded[0, 0, 0, 0].to(torch.float32))
    torch.testing.assert_close(actual[31, 0], decoded[0, 0, 15, 1].to(torch.float32))
    torch.testing.assert_close(actual[0, 7], decoded[0, 7, 0, 0].to(torch.float32))
    torch.testing.assert_close(actual[32, 0], decoded[2, 0, 0, 0].to(torch.float32))
    torch.testing.assert_close(actual[0, 8], decoded[1, 0, 0, 0].to(torch.float32))


def test_legacy_v2_cpu_dispatch_does_not_pass_lr_keyword(monkeypatch):
    from gptqmodel.utils import qvq_cpu

    calls = {}

    def fake_gemv(x, trellis, bits, **kwargs):
        calls.update(kwargs)
        return torch.zeros((x.shape[0], kwargs["out_features"]), dtype=torch.float32)

    monkeypatch.setattr(qvq_cpu, "qvq_cpu_supported", lambda: True)
    monkeypatch.setattr(qvq_cpu, "qvq_cpu_gemv", fake_gemv)
    _, _, trellis, selectors = _random_lr_payload(2, tiles=2)
    layer = QVQLinear(
        bits=2,
        in_features=32,
        out_features=16,
        bank_count=2,
        v2b2_p32=True,
        dtype=torch.float32,
        tensors={
            "trellis": trellis,
            "SU": torch.ones(32),
            "SV": torch.ones(16),
            "bank_ids": pack_qvq_binary_bank_ids(selectors),
            "bank_alt_id": torch.tensor([1], dtype=torch.uint8),
        },
    ).eval()

    output = layer._inner_forward(torch.ones(1, 32))

    assert output.shape == (1, 16)
    assert "v2b2_p32_lr" not in calls
    assert calls["v2b2_p32"] is True


def test_lr32_mlx_conversion_rejects_abandoned_layout():
    pytest.importorskip("mlx.core")
    from gptqmodel.utils.mlx import _qvq_mlx_linear_from_torch

    with pytest.raises(ValueError, match="does not support the abandoned LR32 format"):
        _qvq_mlx_linear_from_torch(_make_torch_lr_layer())
