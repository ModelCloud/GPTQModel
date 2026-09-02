# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exactness coverage for the lossless continuous-window P32 inference layout."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq import (
    decode_p32_window_tiles,
    decode_trellis_tiles,
    pack_qvq_binary_bank_ids,
    reconstruct_p32_window_inner_weight,
    reconstruct_qvq_inner_weight,
    repack_p32_planar_to_window,
    repack_p32_window_to_planar,
    unpack_p32_window_states,
    unpack_trellis_states,
)
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    pgc16_levels_for_version,
)
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
from gptqmodel.utils.qvq_wgmma_cuda import (
    qvq_h100_grouped_ordered_split_counts,
    qvq_h100_ordered_split_count,
    qvq_p32_window_wgmma_m16_tma,
    qvq_p32_window_wgmma_m16_tma_ordered_split,
)

P32_RATES = (1, 1.5, 2, 2.5, 3, 3.5)


@pytest.mark.parametrize("transition_bits", (4, 5, 6, 7))
@pytest.mark.parametrize("logical_rows", (1, 2, 4, 8, 16))
def test_h100_llama_down_uses_measured_ordered_split(transition_bits, logical_rows):
    assert (
        qvq_h100_ordered_split_count(
            device_name="NVIDIA H100 80GB HBM3",
            compute_capability=(9, 0),
            logical_rows=logical_rows,
            in_features=8192,
            out_features=2048,
            transition_bits=transition_bits,
        )
        == 16
    )


@pytest.mark.parametrize(
    ("overrides"),
    (
        {"device_name": "NVIDIA H200"},
        {"compute_capability": (8, 0)},
        {"logical_rows": 3},
        {"logical_rows": 17},
        {"in_features": 2048},
        {"out_features": 8192},
        {"transition_bits": 3},
    ),
)
def test_ordered_split_policy_fails_closed(overrides):
    arguments = {
        "device_name": "NVIDIA H100 80GB HBM3",
        "compute_capability": (9, 0),
        "logical_rows": 1,
        "in_features": 8192,
        "out_features": 2048,
        "transition_bits": 6,
    }
    arguments.update(overrides)
    assert qvq_h100_ordered_split_count(**arguments) == 0


@pytest.mark.parametrize("transition_bits", (4, 5, 6, 7))
def test_h100_llama_qkv_uses_measured_grouped_ordered_splits(transition_bits):
    assert qvq_h100_grouped_ordered_split_counts(
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability=(9, 0),
        in_features=2048,
        out_features=(2048, 512, 512),
        transition_bits=transition_bits,
    ) == (8, 8, 8)


@pytest.mark.parametrize(
    "overrides",
    (
        {"device_name": "NVIDIA H200"},
        {"compute_capability": (8, 0)},
        {"in_features": 4096},
        {"out_features": (2048, 512)},
        {"out_features": (2048, 512, 1024)},
        {"transition_bits": 3},
    ),
)
def test_grouped_ordered_split_policy_fails_closed(overrides):
    arguments = {
        "device_name": "NVIDIA H100 80GB HBM3",
        "compute_capability": (9, 0),
        "in_features": 2048,
        "out_features": (2048, 512, 512),
        "transition_bits": 6,
    }
    arguments.update(overrides)
    assert qvq_h100_grouped_ordered_split_counts(**arguments) is None


def _random_planar_words(
    bits: float, *, tiles: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(20260831 + int(bits * 10))
    words_per_tile = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    return torch.randint(
        0,
        1 << 32,
        (tiles, words_per_tile),
        generator=generator,
        device=device,
        dtype=torch.int64,
    ).to(torch.int32)


@pytest.mark.parametrize("bits", P32_RATES)
def test_p32_window_repack_is_bit_exact_and_storage_neutral(bits):
    planar = _random_planar_words(bits, tiles=17)
    window = repack_p32_planar_to_window(planar, bits=bits)

    assert window.dtype == torch.int32
    assert window.is_contiguous()
    assert window.shape == planar.shape
    assert window.nbytes == planar.nbytes
    assert torch.equal(repack_p32_window_to_planar(window, bits=bits), planar)
    assert torch.equal(
        unpack_p32_window_states(window, bits=bits),
        unpack_trellis_states(planar, bits=bits),
    )


@pytest.mark.parametrize("bits", P32_RATES)
def test_p32_window_repack_accepts_noncontiguous_planar_input(bits):
    planar = _random_planar_words(bits, tiles=5)
    noncontiguous = torch.stack((planar, planar ^ 0x55), dim=1)[:, 0, :]
    assert not noncontiguous.is_contiguous()

    actual = repack_p32_planar_to_window(noncontiguous, bits=bits)
    expected = repack_p32_planar_to_window(planar, bits=bits)

    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
def test_p32_window_decode_and_matrix_mapping_match_canonical_p32(bits):
    in_features = out_features = 32
    planar = _random_planar_words(bits, tiles=4)
    window = repack_p32_planar_to_window(planar, bits=bits)
    selectors = torch.tensor(
        [(tile + segment) & 1 for tile in range(4) for segment in range(8)],
        dtype=torch.uint8,
    )
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([3], dtype=torch.uint8)

    expected_tiles = decode_trellis_tiles(
        planar,
        bits=bits,
        bank_ids=packed_selectors,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    actual_tiles = decode_p32_window_tiles(
        window,
        bits=bits,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    assert torch.equal(actual_tiles, expected_tiles)

    expected_matrix = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    actual_matrix = reconstruct_p32_window_inner_weight(
        window,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=packed_selectors,
        bank_alt_id=bank_alt_id,
    )
    assert torch.equal(actual_matrix, expected_matrix)


def test_p32_window_repack_rejects_invalid_contracts():
    planar = _random_planar_words(2, tiles=1)
    with pytest.raises(TypeError, match="torch.int32"):
        repack_p32_planar_to_window(planar.to(torch.int64), bits=2)
    with pytest.raises(ValueError, match="words per 256-weight tile"):
        repack_p32_planar_to_window(planar[:, :-1], bits=2)
    with pytest.raises(ValueError, match="W1 through W3.5"):
        repack_p32_planar_to_window(torch.zeros((1, 32), dtype=torch.int32), bits=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
def test_p32_window_cuda_matches_cpu_words_and_states(bits):
    planar_cpu = _random_planar_words(bits, tiles=257)
    expected_window = repack_p32_planar_to_window(planar_cpu, bits=bits)
    expected_states = unpack_trellis_states(planar_cpu, bits=bits)

    planar_cuda = planar_cpu.cuda()
    actual_window = repack_p32_planar_to_window(planar_cuda, bits=bits)
    actual_states = unpack_p32_window_states(actual_window, bits=bits)

    assert torch.equal(actual_window.cpu(), expected_window)
    assert torch.equal(actual_states.cpu(), expected_states)
    assert torch.equal(
        repack_p32_window_to_planar(actual_window, bits=bits), planar_cuda
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
def test_p32_window_tma_wgmma_matches_exact_matrix(bits):
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0):
        pytest.skip("P32 TMA RS-WGMMA requires SM90")

    in_features = out_features = 256
    tile_count = (in_features // 16) * (out_features // 16)
    planar = _random_planar_words(bits, tiles=tile_count, device="cuda")
    window = repack_p32_planar_to_window(planar, bits=bits)
    generator = torch.Generator(device="cuda").manual_seed(20260901 + int(bits * 10))
    selectors = torch.randint(
        0,
        2,
        (tile_count * 8,),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    bank_ids = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor(3, dtype=torch.uint8, device="cuda")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    x = (
        torch.randn((16, in_features), generator=generator, device="cuda") * 0.1
    ).half()

    dense = reconstruct_p32_window_inner_weight(
        window,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
    )
    expected = x.float() @ dense
    actual = qvq_p32_window_wgmma_m16_tma(
        x,
        window,
        levels,
        bank_ids,
        bits,
        out_features=out_features,
        bank_alt_id=3,
    )

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bits", (2, 2.5, 3, 3.5))
def test_p32_window_tma_wgmma_ordered_split_is_repeatable(bits):
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0):
        pytest.skip("P32 TMA RS-WGMMA requires SM90")

    in_features = 512
    out_features = 256
    split_count = 2
    tile_count = (in_features // 16) * (out_features // 16)
    planar = _random_planar_words(bits, tiles=tile_count, device="cuda")
    window = repack_p32_planar_to_window(planar, bits=bits)
    generator = torch.Generator(device="cuda").manual_seed(20260921 + int(bits * 10))
    selectors = torch.randint(
        0,
        2,
        (tile_count * 8,),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    bank_ids = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor(3, dtype=torch.uint8, device="cuda")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    x = (
        torch.randn((16, in_features), generator=generator, device="cuda") * 0.1
    ).half()

    dense = reconstruct_p32_window_inner_weight(
        window,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
    )
    expected = x.float() @ dense

    def run():
        return qvq_p32_window_wgmma_m16_tma_ordered_split(
            x,
            window,
            levels,
            bank_ids,
            bits,
            out_features=out_features,
            bank_alt_id=3,
            split_count=split_count,
        )

    actual = run()
    torch.cuda.synchronize()
    for _ in range(5):
        assert torch.equal(run(), actual)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_qvq_linear_hopper_p32_dispatch_reuses_window_cache():
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or not (
        "H100" in properties.name or "H200" in properties.name
    ):
        pytest.skip("Hopper P32 QVQLinear dispatch requires H100 or H200 SM90")
    bits = 3.0
    in_features = out_features = 256
    tile_count = (in_features // 16) * (out_features // 16)
    planar = _random_planar_words(bits, tiles=tile_count, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    selectors = torch.randint(
        0, 2, (tile_count * 8,), generator=generator, device="cuda", dtype=torch.uint8
    )
    bank_ids = pack_qvq_binary_bank_ids(selectors)
    bank_alt_id = torch.tensor([3], dtype=torch.uint8, device="cuda")
    layer = QVQLinear(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        tensors={
            "trellis": planar,
            "SU": torch.ones(in_features, device="cuda"),
            "SV": torch.ones(out_features, device="cuda"),
            "bank_ids": bank_ids,
            "bank_alt_id": bank_alt_id,
        },
        bank_count=2,
        v2b2_p32=True,
    ).eval()
    x = torch.randn(
        (3, in_features), generator=generator, device="cuda", dtype=torch.float16
    )
    dense = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    expected = x.float() @ dense
    actual = layer._inner_forward(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=0.0)
    assert layer._qvq_cuda_window_cache is not None
    cached_window = layer._qvq_cuda_window_cache[3]
    layer._inner_forward(x)
    torch.cuda.synchronize()
    assert layer._qvq_cuda_window_cache[3] is cached_window
