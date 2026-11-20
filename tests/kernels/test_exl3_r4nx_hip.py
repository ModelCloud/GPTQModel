# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""R4NX on RDNA4: the HIP decode is bit-exact against the torch reference.

R4NX reuses the Q4NX trellis packing byte for byte and changes only the
state-to-weight map: ``product = state * 0xBC707477`` (mod 2**32), the four
signed bytes of the product are the four weights of the step, scaled by
``7 / 512``.  The packed stream is written and read natively, the decode runs
in ``reconstruct_r4nx``, and the quantizer's torch Viterbi feeds the native
packer.  Each check below compares against an independent oracle (a bit-string
packer and byte arithmetic in Python) or against the portable reference module
that ships in ``gptqmodel/exllamav3/r4nx.py``.

The target is gfx1200/gfx1201/gfx1250.  ``EXL3_R4NX_ALPHA`` is exactly
representable in binary16 and every ``level * alpha`` product is exact there,
so the FP16 output must match the FP32 oracle *bit for bit*.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch


# Kept in sync with gptqmodel_ext/exllamav3/quant/r4nx.cuh; the reference
# module (gptqmodel/exllamav3/r4nx.py) is the oracle for the codebook.
R4NX_MULTIPLIER = 0xBC707477
R4NX_ALPHA = 7 / 512  # 0.013671875, exactly representable in binary16
R4NX_RATES = {10: 2.5, 12: 3.0, 14: 3.5, 16: 4.0}
R4NX_KV = tuple(R4NX_RATES)
R4NX_STEPS = 64


def _rdna4_hip() -> bool:
    from gptqmodel.utils import arch

    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and arch.is_rdna4(torch.cuda.current_device())
    )


pytestmark = pytest.mark.skipif(not _rdna4_hip(), reason="RDNA4 ROCm GPU required")


def _reference():
    from gptqmodel.exllamav3 import r4nx

    return r4nx


def _unpack_states(packed: torch.Tensor, kv: int) -> torch.Tensor:
    """Read the 64 zero-prefixed (32 - kv)-bit states from each packed tile."""

    halves = packed.to(torch.int64) & 0xFFFF
    words = halves[..., 0::2] | (halves[..., 1::2] << 16)
    end = kv * torch.arange(1, R4NX_STEPS + 1, device=packed.device)
    index = (end - 1) // 32
    shift = 32 * (index + 1) - end
    previous = torch.where(index > 0, words[..., (index - 1).clamp_min(0)], 0)
    window = (previous << 32) | words[..., index]
    return (window >> shift) & ((1 << (32 - kv)) - 1)


def _decode_states(states: torch.Tensor) -> torch.Tensor:
    """Signed product bytes scaled by alpha, flattened to 256 per tile."""

    product = (states.to(torch.int64) * R4NX_MULTIPLIER) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 8, device=states.device)
    octets = (product.unsqueeze(-1) >> shifts) & 0xFF
    levels = octets.to(torch.uint8).view(torch.int8).to(torch.float32)
    return levels.flatten(-2) * R4NX_ALPHA


def _row_major(values: torch.Tensor) -> torch.Tensor:
    """Undo the tensor-core fragment permutation, as reconstruct_r4nx_inner does."""

    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import tensor_core_perm_i

    tiles_k, tiles_n = values.shape[0], values.shape[1]
    tiles = values[..., tensor_core_perm_i(values.device)]
    return tiles.reshape(tiles_k, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(
        tiles_k * 16, tiles_n * 16
    )


def _pack_oracle(fields: torch.Tensor, kv: int) -> torch.Tensor:
    """MSB-first kv-bit field stream stored as little-endian uint16 pairs."""

    mask = (1 << kv) - 1
    rows = []
    for row in fields.tolist():
        stream = "".join(f"{int(field) & mask:0{kv}b}" for field in row)
        words = [int(stream[index:index + 16], 2) for index in range(0, R4NX_STEPS * kv, 16)]
        rows.append([words[index ^ 1] for index in range(len(words))])
    return torch.tensor(rows, dtype=torch.int64).to(torch.uint16).view(torch.int16).to(fields.device)


def _ops():
    from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension

    assert prewarm_exllamav3_extension()
    return torch.ops.gptqmodel_exllamav3


def test_r4nx_reference_constants_match_the_kernel():
    reference = _reference()
    assert int(reference.R4NX_MULTIPLIER) == R4NX_MULTIPLIER
    assert float(reference.R4NX_ALPHA) == R4NX_ALPHA
    assert struct.unpack("<e", struct.pack("<e", R4NX_ALPHA))[0] == R4NX_ALPHA

    header = (
        Path(__file__).resolve().parents[2] / "gptqmodel_ext/exllamav3/quant/r4nx.cuh"
    ).read_text()
    assert f"EXL3_R4NX_MULTIPLIER = 0x{R4NX_MULTIPLIER:08X}u" in header
    assert "EXL3_R4NX_ALPHA = 0.013671875f" in header


def test_r4nx_reference_decode_is_the_python_byte_oracle():
    reference = _reference()
    states = torch.arange(1 << 16, dtype=torch.int64, device="cuda")
    expected = torch.tensor(
        [
            [
                struct.unpack("<b", bytes([(product >> (8 * index)) & 0xFF]))[0] * R4NX_ALPHA
                for index in range(4)
            ]
            for product in ((int(state) * R4NX_MULTIPLIER) & 0xFFFFFFFF for state in states.tolist())
        ],
        dtype=torch.float32,
        device="cuda",
    )
    torch.testing.assert_close(
        reference.decode_r4nx_states(states), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("kv", R4NX_KV)
def test_r4nx_pack_writes_the_trellis_bitstream(kv):
    ops = _ops()
    reference = _reference()
    tiles = 5
    generator = torch.Generator(device="cuda").manual_seed(5709 + kv)
    fields = torch.randint(
        0, 1 << kv, (tiles, R4NX_STEPS), generator=generator, device="cuda", dtype=torch.int64
    )

    packed = torch.empty((tiles, 4 * kv), dtype=torch.int16, device="cuda")
    ops.pack_trellis_r4nx(packed, fields.to(torch.int16), kv)

    assert torch.equal(packed, _pack_oracle(fields, kv))
    assert torch.equal(packed, reference.pack_r4nx_states(fields, kv))


@pytest.mark.parametrize("kv", R4NX_KV)
def test_r4nx_unpack_recovers_the_packed_fields(kv):
    ops = _ops()
    reference = _reference()
    tiles = 5
    generator = torch.Generator(device="cuda").manual_seed(4709 + kv)
    fields = torch.randint(
        0, 1 << kv, (tiles, R4NX_STEPS), generator=generator, device="cuda", dtype=torch.int64
    )
    packed = reference.pack_r4nx_states(fields, kv)

    states = torch.zeros((tiles, R4NX_STEPS), dtype=torch.int16, device="cuda")
    ops.unpack_trellis_r4nx(states, packed, kv)

    # The native unpacker returns the 64 packed fields (the packer's input),
    # which fit int16 for every rate; the (32 - kv)-bit decode states are what
    # the reference's unpack_r4nx_states returns and are checked below.
    assert torch.equal(states, fields.to(torch.int16))
    windows = reference.unpack_r4nx_states(packed)
    recurrence = torch.zeros((tiles,), dtype=torch.int64, device="cuda")
    rebuilt = []
    for step in range(R4NX_STEPS):
        recurrence = ((recurrence << kv) | fields[:, step]) & ((1 << (32 - kv)) - 1)
        rebuilt.append(recurrence)
    torch.testing.assert_close(windows, torch.stack(rebuilt, dim=1), rtol=0, atol=0)


@pytest.mark.parametrize("kv", R4NX_KV)
def test_r4nx_hip_reconstruct_is_bit_exact_with_the_torch_reference(kv):
    ops = _ops()
    reference = _reference()
    tiles_k, tiles_n = 2, 8
    generator = torch.Generator(device="cuda").manual_seed(5709 * kv)
    fields = torch.randint(
        0,
        1 << kv,
        (tiles_k * tiles_n, R4NX_STEPS),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )

    packed = torch.empty((tiles_k * tiles_n, 4 * kv), dtype=torch.int16, device="cuda")
    ops.pack_trellis_r4nx(packed, fields.to(torch.int16), kv)
    packed = packed.reshape(tiles_k, tiles_n, 4 * kv)

    unpacked = torch.empty((tiles_k * 16, tiles_n * 16), dtype=torch.float16, device="cuda")
    ops.reconstruct_r4nx(unpacked, packed, R4NX_RATES[kv])

    states = reference.unpack_r4nx_states(packed).reshape(tiles_k, tiles_n, R4NX_STEPS)
    expected = _row_major(reference.decode_r4nx_states(states).reshape(tiles_k, tiles_n, 256)).half()
    assert torch.equal(unpacked, expected)
    # The independent byte oracle must agree with the reference as well.
    torch.testing.assert_close(
        _row_major(_decode_states(states)), expected.float(), rtol=0, atol=0
    )


@pytest.mark.parametrize("rate", (2.5, 3.0, 3.5, 4.0))
def test_r4nx_quantize_tiles_round_trips_through_the_native_packer(rate):
    """A tile that is already an R4NX decode must come back unchanged.

    Every step has a zero-error codepoint, so the encoder must find it and the
    native packer must reproduce the bitstream the fields came from.
    """

    ops = _ops()
    reference = _reference()
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import pack_trellis, quantize_tiles

    kv = int(4 * rate)
    generator = torch.Generator(device="cuda").manual_seed(671 + kv)
    fields = torch.randint(
        0, 1 << kv, (1, R4NX_STEPS), generator=generator, device="cuda", dtype=torch.int64
    )
    packed = reference.pack_r4nx_states(fields, kv)
    values = reference.decode_r4nx_states(reference.unpack_r4nx_states(packed)).reshape(1, 256)
    source = values.to(torch.float32)

    args = {"K": rate, "r4nx": True}
    reconstructed, codes = quantize_tiles(source, args)
    torch.testing.assert_close(reconstructed, source, rtol=0, atol=0)
    # The index buffer is int16, so kv = 16 fields read back as their
    # two's-complement bit pattern.
    assert (codes[0, :64].to(torch.int64) & 0xFFFF).tolist() == fields[0].tolist()
    assert torch.equal(codes[0, 64:], torch.zeros(192, dtype=torch.int16, device="cuda"))

    repacked = pack_trellis(codes.view(1, 1, 256), args).flatten()
    assert torch.equal(repacked, packed.flatten())
    del ops  # prewarm is what the facade needs; pack_trellis goes through it
