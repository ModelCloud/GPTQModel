# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""R4NX, the RDNA4-native EXL3 NX codebook (issue #5709).

R4NX reuses Q4NX's trellis packing byte for byte and changes only the
state-to-weight map: ``product = state * 0xBC707477`` (mod 2**32) and the four
signed bytes of the product, scaled by ``7 / 512``, are the four weights of the
step in EXL3 fragment order.  These tests pin the portable reference
(``gptqmodel.exllamav3.r4nx``) that both the torch quantizer and the HIP
kernels are written against:

* the packing is Q4NX's (same bitstream, same unpacked states),
* the decode is an independent byte-extraction oracle, not a second copy of the
  implementation,
* the encoder really minimizes the per-step error it claims to minimize,
* encode -> pack -> unpack -> decode is a closed loop at all four rates,
* the C++ header constants agree with the Python reference, and
* the quantizer/checkpoint plumbing records ``r4nx`` as the codebook.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from gptqmodel.exllamav3.q4nx import unpack_q4nx_states
from gptqmodel.exllamav3.r4nx import (
    R4NX_ALPHA,
    R4NX_MEAN_SQUARE,
    R4NX_MULTIPLIER,
    R4NX_RATES,
    R4NX_STEPS,
    R4NX_TILE_WEIGHTS,
    decode_r4nx_states,
    encode_r4nx_states,
    pack_r4nx_states,
    r4nx_codebook,
    r4nx_step_bits,
    r4nx_supports_rate,
    unpack_r4nx_states,
)
from gptqmodel.exllamav3.trellis import validate_trellis_k


R4NX_KV = tuple(int(4 * rate) for rate in R4NX_RATES)


def _signed_bytes(product: int) -> list[int]:
    """The four signed bytes of a 32-bit product, independent of torch."""

    return [struct.unpack("<b", bytes([(product >> (8 * index)) & 0xFF]))[0] for index in range(4)]


def _pack_oracle(fields: torch.Tensor, kv: int) -> torch.Tensor:
    """MSB-first kv-bit field stream stored as little-endian uint16 pairs."""

    mask = (1 << kv) - 1
    rows = []
    for row in fields.tolist():
        stream = "".join(f"{int(field) & mask:0{kv}b}" for field in row)
        words = [int(stream[index:index + 16], 2) for index in range(0, 64 * kv, 16)]
        rows.append([words[index ^ 1] for index in range(len(words))])
    # The uint16 half-word bit patterns are the int16 values the kernels read.
    return torch.tensor(rows, dtype=torch.int64).to(torch.uint16).view(torch.int16)


def _window_oracle(fields: torch.Tensor, kv: int) -> torch.Tensor:
    """The 64 zero-prefixed (32 - kv)-bit states the packer's stream names."""

    mask = (1 << (32 - kv)) - 1
    state = torch.zeros((fields.shape[0],), dtype=torch.int64)
    states = []
    for step in range(R4NX_STEPS):
        state = ((state << kv) | fields[:, step]) & mask
        states.append(state)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize("kv", R4NX_KV)
def test_r4nx_unpack_is_the_q4nx_unpack(kv):
    """R4NX shares Q4NX's bitstream, so its unpacker must be byte identical."""

    generator = torch.Generator().manual_seed(5709 + kv)
    packed = torch.randint(
        -32768, 32767, (4, 4 * kv), dtype=torch.int16, generator=generator
    )
    torch.testing.assert_close(unpack_r4nx_states(packed), unpack_q4nx_states(packed), rtol=0, atol=0)


def test_r4nx_decode_matches_an_independent_byte_oracle():
    states = torch.tensor([0, 1, 2, 3, 0x1234, 0xABCD, 0x7FFF, 0x8000, 0xFFFF], dtype=torch.int64)
    expected = torch.tensor(
        [[level * R4NX_ALPHA for level in _signed_bytes((int(state) * R4NX_MULTIPLIER) & 0xFFFFFFFF)]
         for state in states.tolist()],
        dtype=torch.float32,
    )
    torch.testing.assert_close(decode_r4nx_states(states), expected, rtol=0, atol=0)


def test_r4nx_decode_masks_int16_state_rows():
    """Packed int16 state rows carry the uint16 bit pattern, not a value."""

    states = torch.tensor([0, 1, 0x7FFF, 0x8000, 0xBEEF, 0xFFFF], dtype=torch.int64)
    as_int16 = states.to(torch.int16)
    torch.testing.assert_close(
        decode_r4nx_states(as_int16), decode_r4nx_states(states), rtol=0, atol=0
    )


def test_r4nx_alpha_is_exact_in_binary16():
    """The HIP decode is bit-exact only if ``alpha`` survives the half cast."""

    assert R4NX_ALPHA == 7 / 512
    assert R4NX_ALPHA == 0.013671875
    assert struct.unpack("<e", struct.pack("<e", R4NX_ALPHA))[0] == R4NX_ALPHA
    # Every level * alpha is an integer of at most 10 significant bits times a
    # power of two, so it is exact in both binary32 and binary16.
    levels = torch.arange(-128, 128, dtype=torch.float32)
    values = levels * R4NX_ALPHA
    assert torch.equal(values.half().float(), values)
    assert torch.equal(values.double().float(), values)


def test_r4nx_four_bit_codebook_has_no_aliased_states():
    """An odd multiplier makes state -> four signed bytes injective."""

    codebook = r4nx_codebook(16)
    assert codebook.shape == (1 << 16, 4)
    unique = torch.unique(codebook, dim=0)
    assert unique.shape[0] == 1 << 16
    # 7 / 512 normalizes the codebook to (almost) unit RMS, which keeps the
    # shared g_scale search in the same range the other codebooks use.
    rms = codebook.square().mean().sqrt()
    assert 0.99 < float(rms) < 1.03


def test_r4nx_mean_square_matches_the_codebook():
    # ``R4NX_MEAN_SQUARE`` describes the unnormalized levels; the codebook
    # itself is already scaled by ``R4NX_ALPHA``.
    levels = r4nx_codebook(16) / R4NX_ALPHA
    assert abs(float(levels.square().mean()) - R4NX_MEAN_SQUARE) < 1e-3


@pytest.mark.parametrize("kv", R4NX_KV)
def test_r4nx_pack_matches_the_bitstream_oracle(kv):
    generator = torch.Generator().manual_seed(57 + kv)
    fields = torch.randint(0, 1 << kv, (3, R4NX_STEPS), generator=generator, dtype=torch.int64)
    packed = pack_r4nx_states(fields, kv)
    assert packed.shape == (3, 4 * kv)
    assert packed.dtype == torch.int16
    torch.testing.assert_close(packed, _pack_oracle(fields, kv), rtol=0, atol=0)


@pytest.mark.parametrize("kv", R4NX_KV)
def test_r4nx_pack_unpack_round_trip(kv):
    generator = torch.Generator().manual_seed(4709 + kv)
    fields = torch.randint(0, 1 << kv, (3, R4NX_STEPS), generator=generator, dtype=torch.int64)
    packed = pack_r4nx_states(fields, kv)

    states = unpack_r4nx_states(packed)
    torch.testing.assert_close(states, _window_oracle(fields, kv), rtol=0, atol=0)
    # The low kv bits of each decode state are exactly the packed field.
    torch.testing.assert_close(states & ((1 << kv) - 1), fields, rtol=0, atol=0)


def test_r4nx_encoder_is_exhaustively_optimal_at_four_bits():
    """At kv = 16 the state is the whole field, so the encoder must be exact.

    There is no trellis memory: every step independently picks the codepoint
    with the smallest squared error.  Compare against a brute-force argmin over
    the full 65536-codepoint codebook.
    """

    torch.manual_seed(5709)
    tiles = torch.randn(3, R4NX_TILE_WEIGHTS)
    values, suffixes = encode_r4nx_states(tiles, 16)
    assert suffixes.shape == (3, R4NX_STEPS)
    codebook = r4nx_codebook(16)

    target = tiles.reshape(3, R4NX_STEPS, 4) / R4NX_ALPHA
    achieved = ((values.reshape(3, R4NX_STEPS, 4) / R4NX_ALPHA - target) ** 2).sum(-1)
    for step in range(R4NX_STEPS):
        distances = ((codebook.unsqueeze(0) - target[:, step].unsqueeze(1)) ** 2).sum(-1)
        best = distances.min(dim=1).values
        assert torch.all(achieved[:, step] <= best + 1e-4 * (1.0 + best)), (
            f"step {step} is {float((achieved[:, step] - best).max()):g} above the "
            "brute-force minimum"
        )


@pytest.mark.parametrize("rate", R4NX_RATES)
def test_r4nx_encode_pack_unpack_decode_round_trip(rate):
    kv = r4nx_step_bits(rate)
    torch.manual_seed(6000 + kv)
    weights = torch.randn(2, R4NX_TILE_WEIGHTS)

    values, suffixes = encode_r4nx_states(weights, kv)
    packed = pack_r4nx_states(suffixes, kv)
    decoded = decode_r4nx_states(unpack_r4nx_states(packed)).reshape(2, R4NX_TILE_WEIGHTS)

    # The encoder reports the decode of the states it packed, so the loop must
    # close exactly: any packing or window bug shows up as a mismatch.
    torch.testing.assert_close(decoded, values, rtol=0, atol=0)
    assert suffixes.min() >= 0
    assert suffixes.max() < (1 << kv)


def test_r4nx_kernel_header_constants_match_the_reference():
    header = (
        Path(__file__).resolve().parents[1]
        / "gptqmodel_ext/exllamav3/quant/r4nx.cuh"
    )
    text = header.read_text()
    assert f"EXL3_R4NX_MULTIPLIER = 0x{R4NX_MULTIPLIER:08X}u" in text
    assert "EXL3_R4NX_ALPHA = 0.013671875f" in text
    assert "EXL3_R4NX_CODEBOOK = 4" in text
    # The __byte_perm selectors that gather the two low / two high bytes.
    assert "EXL3_R4NX_PAIR_01 = 0x5140u" in text
    assert "EXL3_R4NX_PAIR_23 = 0x5342u" in text


def test_r4nx_rates_and_step_bits():
    assert R4NX_RATES == (2.5, 3.0, 3.5, 4.0)
    for rate in R4NX_RATES:
        assert r4nx_supports_rate(rate)
        assert r4nx_step_bits(rate) in (10, 12, 14, 16)
    for rate in (1.0, 1.5, 2.0, 4.5, 5.0, True, "4.0"):
        assert not r4nx_supports_rate(rate)
        with pytest.raises(ValueError, match="R4NX supports"):
            r4nx_step_bits(rate)


def test_validate_trellis_k_accepts_the_r4nx_rates():
    for rate in R4NX_RATES:
        validate_trellis_k(rate, r4nx=True)


@pytest.mark.parametrize("rate", [1.0, 1.5, 2.0, 4.5, 5.0])
def test_validate_trellis_k_rejects_unsupported_r4nx_rates(rate):
    with pytest.raises(ValueError, match="r4nx supports"):
        validate_trellis_k(rate, r4nx=True)


@pytest.mark.parametrize(
    "markers",
    [
        {"mcg": True, "r4nx": True},
        {"mul1": True, "r4nx": True},
        {"q4nx": True, "r4nx": True},
        {"mcg": True, "mul1": True, "q4nx": True, "r4nx": True},
    ],
)
def test_validate_trellis_k_rejects_conflicting_r4nx_markers(markers):
    # mcg+mul1 is reported by its own, older check first.
    with pytest.raises(ValueError, match="codebook markers|mcg and mul1"):
        validate_trellis_k(3, **markers)


def test_trellis_rate_reads_the_r4nx_geometry():
    from gptqmodel.exllamav3.trellis import trellis_rate

    assert trellis_rate(4 * 10, r4nx=True) == 2.5
    assert trellis_rate(4 * 12, r4nx=True) == 3
    assert trellis_rate(4 * 14, r4nx=True) == 3.5
    assert trellis_rate(4 * 16, r4nx=True) == 4


@pytest.mark.parametrize("rate", R4NX_RATES)
def test_exl3_config_accepts_the_r4nx_rates(rate):
    from gptqmodel.quantization.config import EXL3Config

    config = EXL3Config(bits=rate, codebook="r4nx", offload_to_disk=False)
    assert config.codebook == "r4nx"
    assert config.bits == (int(rate) if float(rate).is_integer() else rate)


@pytest.mark.parametrize("rate", [1.0, 1.5, 2.0, 4.5, 5.0])
def test_exl3_config_rejects_unsupported_r4nx_rates(rate):
    from gptqmodel.quantization.config import EXL3Config

    with pytest.raises(ValueError, match="r4nx supports"):
        EXL3Config(bits=rate, codebook="r4nx", offload_to_disk=False)


def test_exl3_config_rejects_an_unknown_codebook():
    from gptqmodel.quantization.config import EXL3Config

    with pytest.raises(ValueError, match="`codebook` must be one of"):
        EXL3Config(bits=4, codebook="r4nx2", offload_to_disk=False)


def test_r4nx_module_records_the_codebook_marker():
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear

    generator = torch.Generator().manual_seed(5709)
    tensors = {
        "trellis": torch.randint(
            -32768, 32767, (2, 2, 4 * 16), dtype=torch.int16, generator=generator
        ),
        "suh": torch.ones(128, dtype=torch.float16),
        "svh": torch.ones(128, dtype=torch.float16),
        "r4nx": torch.tensor(R4NX_MULTIPLIER - (1 << 32), dtype=torch.int32),
    }
    module = ExllamaV3TorchLinear.from_tensors(
        in_features=128, out_features=128, name="proj", tensors=tensors
    )
    entry = module.tensor_storage_entry()
    assert entry["bits_per_weight"] == 4
    assert entry["r4nx_multiplier"] == R4NX_MULTIPLIER
    assert "proj.r4nx" in entry["stored_tensors"]
    assert module._codebook_name() == "r4nx"


def _r4nx_reference_inner(packed: torch.Tensor) -> torch.Tensor:
    """The Q4NX tile layout applied to the R4NX decode (independent oracle)."""

    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import tensor_core_perm_i

    tiles_k, tiles_n, _ = packed.shape
    values = decode_r4nx_states(unpack_r4nx_states(packed)).reshape(
        tiles_k, tiles_n, R4NX_TILE_WEIGHTS
    )
    tiles = values[..., tensor_core_perm_i(packed.device)]
    return tiles.reshape(tiles_k, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(
        tiles_k * 16, tiles_n * 16
    )


@pytest.mark.parametrize("rate", R4NX_RATES)
def test_reconstruct_r4nx_inner_uses_the_q4nx_tile_layout(rate):
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import reconstruct_r4nx_inner

    kv = r4nx_step_bits(rate)
    torch.manual_seed(7000 + kv)
    tiles_k, tiles_n = 2, 3
    suffixes = torch.randint(0, 1 << kv, (tiles_k * tiles_n, R4NX_STEPS))
    packed = pack_r4nx_states(suffixes, kv).reshape(tiles_k, tiles_n, 4 * kv)

    torch.testing.assert_close(
        reconstruct_r4nx_inner(packed), _r4nx_reference_inner(packed), rtol=0, atol=0
    )


@pytest.mark.parametrize("rate", R4NX_RATES)
def test_torch_module_decode_inner_weight_uses_the_q4nx_tile_layout(rate):
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear

    kv = r4nx_step_bits(rate)
    torch.manual_seed(8000 + kv)
    # EXL3 storage is 128-channel aligned, so use a full 128x128 tile block.
    tiles_k, tiles_n = 8, 8
    suffixes = torch.randint(0, 1 << kv, (tiles_k * tiles_n, R4NX_STEPS))
    packed = pack_r4nx_states(suffixes, kv).reshape(tiles_k, tiles_n, 4 * kv)
    tensors = {
        "trellis": packed.contiguous(),
        "suh": torch.ones(tiles_k * 16, dtype=torch.float16),
        "svh": torch.ones(tiles_n * 16, dtype=torch.float16),
        "r4nx": torch.tensor(R4NX_MULTIPLIER - (1 << 32), dtype=torch.int32),
    }
    module = ExllamaV3TorchLinear.from_tensors(
        in_features=tiles_k * 16,
        out_features=tiles_n * 16,
        name="proj",
        tensors=tensors,
    )

    torch.testing.assert_close(
        module._decode_inner_weight(), _r4nx_reference_inner(packed), rtol=0, atol=0
    )
