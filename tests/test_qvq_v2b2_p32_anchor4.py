# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exactness coverage for the lossless storage-neutral P32 Anchor-4 layout."""

import pytest
import torch

from gptqmodel.quantization.qvq import (
    decode_p32_anchor4_tiles,
    decode_trellis_tiles,
    pack_qvq_binary_bank_ids,
    reconstruct_p32_anchor4_inner_weight,
    reconstruct_qvq_inner_weight,
    repack_p32_anchor4_to_planar,
    repack_p32_planar_to_anchor4,
    unpack_p32_anchor4_states,
    unpack_trellis_states,
)
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile

P32_ANCHOR4_RATES = (2, 2.5, 3, 3.5)


def _random_planar_words(bits: float, *, tiles: int, device: torch.device | str = "cpu") -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(20260902 + int(bits * 10))
    words_per_tile = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    return torch.randint(
        0,
        1 << 32,
        (tiles, words_per_tile),
        generator=generator,
        device=device,
        dtype=torch.int64,
    ).to(torch.int32)


@pytest.mark.parametrize("bits", P32_ANCHOR4_RATES)
def test_p32_anchor4_repack_is_bit_exact_storage_neutral_and_state_exact(bits):
    planar = _random_planar_words(bits, tiles=17)
    anchor4 = repack_p32_planar_to_anchor4(planar, bits=bits)

    assert anchor4.dtype == torch.int32
    assert anchor4.is_contiguous()
    assert anchor4.shape == planar.shape
    assert anchor4.nbytes == planar.nbytes
    assert torch.equal(repack_p32_anchor4_to_planar(anchor4, bits=bits), planar)
    assert torch.equal(unpack_p32_anchor4_states(anchor4, bits=bits), unpack_trellis_states(planar, bits=bits))


@pytest.mark.parametrize("bits", P32_ANCHOR4_RATES)
def test_p32_anchor4_repack_accepts_noncontiguous_input(bits):
    planar = _random_planar_words(bits, tiles=5)
    noncontiguous = torch.stack((planar, planar ^ 0x55), dim=1)[:, 0, :]
    assert not noncontiguous.is_contiguous()

    actual = repack_p32_planar_to_anchor4(noncontiguous, bits=bits)
    expected = repack_p32_planar_to_anchor4(planar, bits=bits)

    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("bits", P32_ANCHOR4_RATES)
def test_p32_anchor4_tile_and_k16n16_matrix_match_canonical_p32(bits):
    in_features = out_features = 32
    planar = _random_planar_words(bits, tiles=4)
    anchor4 = repack_p32_planar_to_anchor4(planar, bits=bits)
    selectors = torch.tensor(
        [(tile + segment) & 1 for tile in range(4) for segment in range(8)],
        dtype=torch.uint8,
    )
    bank_ids = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([3], dtype=torch.uint8)

    expected_tiles = decode_trellis_tiles(
        planar,
        bits=bits,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    actual_tiles = decode_p32_anchor4_tiles(
        anchor4,
        bits=bits,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
    )
    assert torch.equal(actual_tiles, expected_tiles)

    expected_matrix = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    actual_matrix = reconstruct_p32_anchor4_inner_weight(
        anchor4,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
    )
    assert torch.equal(actual_matrix, expected_matrix)


def test_p32_anchor4_repack_rejects_invalid_contracts():
    planar = _random_planar_words(2, tiles=1)
    with pytest.raises(TypeError, match="torch.int32"):
        repack_p32_planar_to_anchor4(planar.to(torch.int64), bits=2)
    with pytest.raises(ValueError, match="words per 256-weight tile"):
        repack_p32_planar_to_anchor4(planar[:, :-1], bits=2)
    with pytest.raises(ValueError, match="W2 through W3.5"):
        repack_p32_planar_to_anchor4(torch.zeros((1, 8), dtype=torch.int32), bits=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bits", P32_ANCHOR4_RATES)
def test_p32_anchor4_cuda_matches_cpu_words_states_and_roundtrip(bits):
    planar_cpu = _random_planar_words(bits, tiles=257)
    expected_anchor4 = repack_p32_planar_to_anchor4(planar_cpu, bits=bits)
    expected_states = unpack_trellis_states(planar_cpu, bits=bits)

    planar_cuda = planar_cpu.cuda()
    actual_anchor4 = repack_p32_planar_to_anchor4(planar_cuda, bits=bits)
    actual_states = unpack_p32_anchor4_states(actual_anchor4, bits=bits)

    assert torch.equal(actual_anchor4.cpu(), expected_anchor4)
    assert torch.equal(actual_states.cpu(), expected_states)
    assert torch.equal(repack_p32_anchor4_to_planar(actual_anchor4, bits=bits), planar_cuda)
