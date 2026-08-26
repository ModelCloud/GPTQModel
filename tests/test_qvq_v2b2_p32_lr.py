# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reference and MLX coverage for the GPU-oriented QVQ LR32 layout."""

import numpy as np
import pytest
import torch

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


def test_lr32_config_and_format_metadata():
    config = QVQConfig(bits=2, format=FORMAT.QVQ_V2B2_P32_LR, bank_count=2)
    assert config.format == FORMAT.QVQ_V2B2_P32_LR
    assert config.quant_linear_init_kwargs()["v2b2_p32_lr"] is True
    assert config.quant_linear_init_kwargs()["v2b2_p32"] is False


@pytest.mark.parametrize("output_fp32", (False, True))
@pytest.mark.parametrize("out_features", (8, 16, 40))
@pytest.mark.parametrize("bits", LR_RATES)
def test_lr32_mlx_gpu_kernel_matches_torch_reconstruction(
    output_fp32,
    out_features,
    bits,
):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

    in_features = 64
    tile_count = (in_features // 32) * (out_features // 8)
    _, _, trellis, selectors = _random_lr_payload(bits, tiles=tile_count)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)
    x = torch.randn(5, in_features, dtype=torch.float16)
    inner = reconstruct_local_ring_inner_weight(
        trellis,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    expected = x.to(torch.float32) @ inner

    actual = qvq_mlx_gemv(
        mx.array(x.numpy()),
        mx.array(trellis.numpy()),
        bits,
        out_features=out_features,
        bank_ids=mx.array(packed_selectors.numpy()),
        bank_alt_id=mx.array(bank_alt_id.numpy()),
        v2b2_p32_lr=True,
        output_fp32=output_fp32,
    )
    mx.eval(actual)
    actual_torch = torch.from_numpy(np.asarray(actual))
    expected = expected if output_fp32 else expected.to(torch.float16)
    torch.testing.assert_close(actual_torch, expected, rtol=0, atol=4e-3)


def test_lr32_mlx_linear_inference_matches_torch_oracle():
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import QVQMLXLinear

    torch_layer = _make_torch_lr_layer(bits=2, in_features=64, out_features=16)
    mlx_layer = QVQMLXLinear(
        bits=torch_layer.bits,
        in_features=torch_layer.in_features,
        out_features=torch_layer.out_features,
        trellis=mx.array(torch_layer.trellis.numpy()),
        SU=mx.array(torch_layer.SU.numpy()),
        SV=mx.array(torch_layer.SV.numpy()),
        codebook_version=torch_layer.codebook_version,
        vector_size=torch_layer.vector_size,
        trellis_window=torch_layer.trellis_window,
        bank_ids=mx.array(torch_layer.bank_ids.numpy()),
        v2b2_p32_lr=True,
        bank_alt_id=mx.array(torch_layer.bank_alt_id.numpy()),
    )
    x = torch.randn(3, 64, dtype=torch.float16)
    expected = qvq_local_ring_dense_oracle_forward(torch_layer, x)
    actual = mlx_layer(mx.array(x.numpy()))
    mx.eval(actual)
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(actual)).to(torch.float32),
        expected,
        rtol=0,
        atol=2e-2,
    )
