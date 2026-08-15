# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fixed PGC16-v1 production state decoder for planar QVQ checkpoints."""

from __future__ import annotations

from functools import lru_cache

import torch

from ..qvq_rates import qvq_transition_bits


PGC16_CODEBOOK_VERSION = "pgc16-v1"
PGC16_CODEBOOK_VERSIONS = (PGC16_CODEBOOK_VERSION,)
PGC16_STATE_COUNT = 1 << 16
PGC16_LEVEL_COUNT = 1 << 8
PGC16_V4_BANK_COUNT = 4
PGC18_V4_STATE_COUNT = 1 << 18
PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS = {
    # Experimental V2B4-P64 graph portfolio. Bank zero is the canonical V2
    # mapping. The other masks are deliberately rate-keyed because the local
    # successor neighborhoods contain 2**E states at transition width E.
    2: (0x0000, 0xA5A5, 0x5A5A, 0x3C3C),
    3: (0x0000, 0xA5A5, 0x9696, 0x6969),
    4: (0x0000, 0x5A5A, 0x3C3C, 0xC3C3),
    5: (0x0000, 0x9696, 0x3C3C, 0xC3C3),
    6: (0x0000, 0x6969, 0x5A5A, 0x3C3C),
    7: (0x0000, 0xC3C3, 0x9696, 0x5A5A),
}
PGC16_V4_BANK_XOR_MASKS = (0xA5A5, 0x5A5A, 0x3C3C, 0xC3C3)
PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS = {
    # Bank zero preserves canonical V4. The remaining permutations were
    # selected by exhaustive orthant/covering-radius screening of the fixed
    # PGC16 table at each transition width; wider transitions favor masks with
    # less-correlated high-bit pairs.
    4: (0xA5A5, 0x5A5A, 0x3C3C, 0xC3C3),
    6: (0xA5A5, 0x5A5A, 0x9696, 0x6969),
    8: (0xA5A5, 0x3C3C, 0x5A5A, 0xC3C3),
    10: (0xA5A5, 0x9696, 0x3C3C, 0xC3C3),
    12: (0xA5A5, 0x6969, 0x5A5A, 0x3C3C),
    14: (0xA5A5, 0xC3C3, 0x9696, 0x5A5A),
    16: (0xA5A5, 0x3C3C, 0x9696, 0x6969),
}
PGC16_MULTIPLIER = 40503
PGC16_INCREMENT = 17011
# RMS of the complete canonical FP32 PGC16-v1 codebook.
PGC16_NORMALIZATION_RMS = 0.9975093603134155
# Unit-RMS Gaussian scale multipliers selected before the full BlockLDLQ pass.
# W1 and W2 intentionally stay at the HYB control scale; higher rates need
# additional range to use their larger reconstruction alphabets instead of
# clipping tails. Half-step entries linearly interpolate the validated integer
# anchors, so adding a rate does not introduce a discontinuous search prior.
PGC16_SCALE_FACTORS = (
    1.0,  # W1
    1.0,  # W1.5
    1.0,  # W2
    1.05,  # W2.5
    1.1,  # W3
    1.15,  # W3.5
    1.2,  # W4
    1.25,  # W4.5
    1.3,  # W5
    1.4,  # W5.5
    1.5,  # W6
    1.5,  # W6.5
    1.5,  # W7
    1.5,  # W7.5
    1.5,  # W8
)

# Frozen FP16 bit patterns for the 256 midpoint Gaussian quantiles. Keeping
# physical bits in the format avoids platform-dependent erfinv results during
# checkpoint load and kernel initialization.
_PGC16_LEVEL_BITS = (
    0xC1C5,
    0xC10A,
    0xC0AC,
    0xC06A,
    0xC037,
    0xC00C,
    0xBFD0,
    0xBF91,
    0xBF58,
    0xBF24,
    0xBEF5,
    0xBEC9,
    0xBEA0,
    0xBE7A,
    0xBE56,
    0xBE33,
    0xBE13,
    0xBDF4,
    0xBDD6,
    0xBDBA,
    0xBD9E,
    0xBD84,
    0xBD6A,
    0xBD52,
    0xBD3A,
    0xBD23,
    0xBD0C,
    0xBCF6,
    0xBCE1,
    0xBCCC,
    0xBCB8,
    0xBCA4,
    0xBC90,
    0xBC7D,
    0xBC6B,
    0xBC58,
    0xBC46,
    0xBC35,
    0xBC24,
    0xBC13,
    0xBC02,
    0xBBE3,
    0xBBC3,
    0xBBA3,
    0xBB83,
    0xBB64,
    0xBB46,
    0xBB28,
    0xBB0A,
    0xBAED,
    0xBAD0,
    0xBAB3,
    0xBA97,
    0xBA7B,
    0xBA5F,
    0xBA44,
    0xBA29,
    0xBA0E,
    0xB9F3,
    0xB9D9,
    0xB9BF,
    0xB9A5,
    0xB98B,
    0xB972,
    0xB959,
    0xB940,
    0xB927,
    0xB90E,
    0xB8F6,
    0xB8DE,
    0xB8C6,
    0xB8AE,
    0xB896,
    0xB87F,
    0xB867,
    0xB850,
    0xB839,
    0xB822,
    0xB80B,
    0xB7E9,
    0xB7BB,
    0xB78F,
    0xB762,
    0xB735,
    0xB709,
    0xB6DD,
    0xB6B1,
    0xB685,
    0xB65A,
    0xB62F,
    0xB603,
    0xB5D9,
    0xB5AE,
    0xB583,
    0xB559,
    0xB52E,
    0xB504,
    0xB4DA,
    0xB4B0,
    0xB486,
    0xB45D,
    0xB433,
    0xB40A,
    0xB3C0,
    0xB36E,
    0xB31C,
    0xB2C9,
    0xB278,
    0xB226,
    0xB1D4,
    0xB183,
    0xB131,
    0xB0E0,
    0xB08F,
    0xB03E,
    0xAFDA,
    0xAF39,
    0xAE97,
    0xADF6,
    0xAD55,
    0xACB4,
    0xAC13,
    0xAAE6,
    0xA9A4,
    0xA863,
    0xA644,
    0xA385,
    0x9D03,
    0x1D03,
    0x2385,
    0x2644,
    0x2863,
    0x29A4,
    0x2AE6,
    0x2C13,
    0x2CB4,
    0x2D55,
    0x2DF6,
    0x2E97,
    0x2F39,
    0x2FDA,
    0x303E,
    0x308F,
    0x30E0,
    0x3131,
    0x3183,
    0x31D4,
    0x3226,
    0x3278,
    0x32C9,
    0x331C,
    0x336E,
    0x33C0,
    0x340A,
    0x3433,
    0x345D,
    0x3486,
    0x34B0,
    0x34DA,
    0x3504,
    0x352E,
    0x3559,
    0x3583,
    0x35AE,
    0x35D9,
    0x3603,
    0x362F,
    0x365A,
    0x3685,
    0x36B1,
    0x36DD,
    0x3709,
    0x3735,
    0x3762,
    0x378F,
    0x37BB,
    0x37E9,
    0x380B,
    0x3822,
    0x3839,
    0x3850,
    0x3867,
    0x387F,
    0x3896,
    0x38AE,
    0x38C6,
    0x38DE,
    0x38F6,
    0x390E,
    0x3927,
    0x3940,
    0x3959,
    0x3972,
    0x398B,
    0x39A5,
    0x39BF,
    0x39D9,
    0x39F3,
    0x3A0E,
    0x3A29,
    0x3A44,
    0x3A5F,
    0x3A7B,
    0x3A97,
    0x3AB3,
    0x3AD0,
    0x3AED,
    0x3B0A,
    0x3B28,
    0x3B46,
    0x3B64,
    0x3B83,
    0x3BA3,
    0x3BC3,
    0x3BE3,
    0x3C02,
    0x3C13,
    0x3C24,
    0x3C35,
    0x3C46,
    0x3C58,
    0x3C6B,
    0x3C7D,
    0x3C90,
    0x3CA4,
    0x3CB8,
    0x3CCC,
    0x3CE1,
    0x3CF6,
    0x3D0C,
    0x3D23,
    0x3D3A,
    0x3D52,
    0x3D6A,
    0x3D84,
    0x3D9E,
    0x3DBA,
    0x3DD6,
    0x3DF4,
    0x3E13,
    0x3E33,
    0x3E56,
    0x3E7A,
    0x3EA0,
    0x3EC9,
    0x3EF5,
    0x3F24,
    0x3F58,
    0x3F91,
    0x3FD0,
    0x400C,
    0x4037,
    0x406A,
    0x40AC,
    0x410A,
    0x41C5,
)


@lru_cache(maxsize=1)
def canonical_pgc16_levels() -> torch.Tensor:
    """Return the process-wide immutable-by-contract PGC16 FP16 levels."""

    signed_bits = [value if value < 0x8000 else value - 0x10000 for value in _PGC16_LEVEL_BITS]
    return torch.tensor(signed_bits, dtype=torch.int16).view(torch.float16)


def pgc16_levels_for_version(codebook_version: str) -> torch.Tensor:
    """Resolve the only production QVQ scalar table: fixed PGC16-v1."""

    version = str(codebook_version).strip().lower()
    if version == PGC16_CODEBOOK_VERSION:
        return canonical_pgc16_levels()
    raise ValueError(f"Unsupported production PGC16 codebook/codec `{codebook_version}`; expected `pgc16-v1`.")


def pgc16_mix_states(states: torch.Tensor) -> torch.Tensor:
    """Apply the versioned PGC16 bijection to unsigned 16-bit states."""

    if states.dtype == torch.bool or states.is_floating_point() or states.is_complex():
        raise TypeError("PGC16 states must use an integer dtype.")
    states_i64 = states.to(torch.int64)
    if torch.any((states_i64 < 0) | (states_i64 >= PGC16_STATE_COUNT)):
        raise ValueError("PGC16 states must be in `[0, 65535]`.")
    mixed = states_i64 ^ (states_i64 >> 8)
    mixed = (mixed * PGC16_MULTIPLIER + PGC16_INCREMENT) & 0xFFFF
    return (mixed ^ (mixed >> 7)).contiguous()


def validate_pgc16_levels(levels: torch.Tensor) -> None:
    """Validate a complete, finite, strictly ordered scalar compander."""

    if levels.ndim != 1 or levels.numel() != PGC16_LEVEL_COUNT:
        raise ValueError(f"PGC16 levels must have shape `({PGC16_LEVEL_COUNT},)`, got `{tuple(levels.shape)}`.")
    if not levels.is_floating_point():
        raise TypeError("PGC16 levels must use a floating-point dtype.")
    if not torch.isfinite(levels).all():
        raise ValueError("PGC16 levels must contain only finite values.")
    if not torch.all(levels[1:] > levels[:-1]):
        raise ValueError("PGC16 levels must be strictly increasing to preserve 65,536 unique vectors.")


def pgc16_decode_states(states: torch.Tensor, *, levels: torch.Tensor | None = None) -> torch.Tensor:
    """Decode PGC16 states with the canonical table or explicit research levels."""

    mixed = pgc16_mix_states(states)
    if levels is None:
        levels = canonical_pgc16_levels().to(device=states.device)
    else:
        if levels.device != states.device:
            raise ValueError("PGC16 states and levels must be on the same device.")
        validate_pgc16_levels(levels)
    indices = torch.stack((mixed >> 8, mixed & 0xFF), dim=-1)
    return levels[indices].contiguous()


def pgc16_decode_states_v2_banked(
    states: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    bits: float,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode the experimental rate-keyed V2B4 graph portfolio.

    Every bank remains an L16/V2 decoder. Bank zero is exactly canonical V2;
    banks one through three only change the bijective state labelling. The
    P64 format chooses one bank for each contiguous 64-weight segment.
    """

    if bank_ids.dtype == torch.bool or bank_ids.is_floating_point() or bank_ids.is_complex():
        raise TypeError("PGC16 V2B4 bank selectors must use an integer dtype.")
    if bank_ids.device != states.device:
        raise ValueError("PGC16 states and V2B4 bank selectors must be on the same device.")
    if bank_ids.ndim == 0:
        bank_ids = bank_ids.expand(states.shape)
    if tuple(bank_ids.shape) != tuple(states.shape):
        raise ValueError(f"PGC16 V2B4 selectors must have shape {tuple(states.shape)}, got {tuple(bank_ids.shape)}.")
    bank_ids_i64 = bank_ids.to(torch.int64)
    if torch.any((bank_ids_i64 < 0) | (bank_ids_i64 >= PGC16_V4_BANK_COUNT)):
        raise ValueError(f"PGC16 V2B4 selectors must be in `[0, {PGC16_V4_BANK_COUNT - 1}]`.")
    if levels is None:
        levels = canonical_pgc16_levels().to(device=states.device)
    else:
        if levels.device != states.device:
            raise ValueError("PGC16 states and levels must be on the same device.")
        validate_pgc16_levels(levels)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    masks_for_rate = PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS.get(transition_bits)
    if masks_for_rate is None:
        raise ValueError("PGC16 V2B4 decoding supports only rates W1 through W3.5.")
    masks = torch.tensor(masks_for_rate, dtype=torch.int64, device=states.device)
    mixed = pgc16_mix_states(states.to(torch.int64) ^ masks[bank_ids_i64])
    indices = torch.stack((mixed >> 8, mixed & 0xFF), dim=-1)
    return levels[indices].contiguous()


def pgc16_decode_states_v4(states: torch.Tensor, *, levels: torch.Tensor | None = None) -> torch.Tensor:
    """Decode the opt-in four-scalar PGC16 vector codec.

    The first pair is the canonical bijection, so the four-vector mapping is
    injective regardless of the second pair.  A fixed xor before the second
    bijection gives an independent-looking second pair without serialized
    codebook state.
    """

    if levels is None:
        levels = canonical_pgc16_levels().to(device=states.device)
    else:
        if levels.device != states.device:
            raise ValueError("PGC16 states and levels must be on the same device.")
        validate_pgc16_levels(levels)
    first = pgc16_mix_states(states)
    second = pgc16_mix_states(states.to(torch.int64) ^ PGC16_V4_BANK_XOR_MASKS[0])
    indices = torch.stack((first >> 8, first & 0xFF, second >> 8, second & 0xFF), dim=-1)
    return levels[indices].contiguous()


def pgc16_decode_states_v4_banked(
    states: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    bits: float,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode V4 states using one fixed bank per tile prefix."""

    if bank_ids.dtype == torch.bool or bank_ids.is_floating_point() or bank_ids.is_complex():
        raise TypeError("PGC16 V4 bank selectors must use an integer dtype.")
    if bank_ids.device != states.device:
        raise ValueError("PGC16 states and bank selectors must be on the same device.")
    if bank_ids.ndim == 0:
        bank_ids = bank_ids.expand(states.shape[:-1])
    if tuple(bank_ids.shape) != tuple(states.shape[:-1]):
        raise ValueError(
            f"PGC16 V4 bank selectors must have shape {tuple(states.shape[:-1])}, got {tuple(bank_ids.shape)}."
        )
    bank_ids_i64 = bank_ids.to(torch.int64)
    if torch.any((bank_ids_i64 < 0) | (bank_ids_i64 >= PGC16_V4_BANK_COUNT)):
        raise ValueError(f"PGC16 V4 bank selectors must be in `[0, {PGC16_V4_BANK_COUNT - 1}]`.")
    if levels is None:
        levels = canonical_pgc16_levels().to(device=states.device)
    else:
        if levels.device != states.device:
            raise ValueError("PGC16 states and levels must be on the same device.")
        validate_pgc16_levels(levels)
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    masks_for_rate = PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS.get(transition_bits)
    if masks_for_rate is None:
        raise ValueError("PGC16 V4 banked decoding supports only rates W1 through W4.")
    first = pgc16_mix_states(states)
    masks = torch.tensor(masks_for_rate, dtype=torch.int64, device=states.device)
    second_states = states.to(torch.int64) ^ masks[bank_ids_i64].unsqueeze(-1)
    second = pgc16_mix_states(second_states)
    indices = torch.stack((first >> 8, first & 0xFF, second >> 8, second & 0xFF), dim=-1)
    return levels[indices].contiguous()


def pgc18_decode_states_v4(states: torch.Tensor, *, bits: float, levels: torch.Tensor | None = None) -> torch.Tensor:
    """Decode the experimental L18/V4 contextual codec.

    The high two state bits select one of the four rate-keyed V4 manifolds;
    the low sixteen bits remain the canonical PGC16 state.  This makes the
    additional trellis history reconstruction-visible without adding a
    serialized bank selector or changing the planar transition payload.
    """

    if states.dtype == torch.bool or states.is_floating_point() or states.is_complex():
        raise TypeError("PGC18 V4 states must use an integer dtype.")
    states_i64 = states.to(torch.int64)
    if torch.any((states_i64 < 0) | (states_i64 >= PGC18_V4_STATE_COUNT)):
        raise ValueError(f"PGC18 V4 states must be in `[0, {PGC18_V4_STATE_COUNT - 1}]`.")
    local_states = (states_i64 & 0xFFFF).contiguous()
    context_banks = (states_i64 >> 16).contiguous()
    if levels is None:
        levels = canonical_pgc16_levels().to(device=states.device)
    else:
        if levels.device != states.device:
            raise ValueError("PGC18 V4 states and levels must be on the same device.")
        validate_pgc16_levels(levels)
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    if transition_bits > 10:
        raise ValueError("PGC18 V4 decoding supports only rates W1 through W2.5.")
    masks_for_rate = PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS.get(transition_bits)
    if masks_for_rate is None:
        raise ValueError("PGC18 V4 decoding supports only rates W1 through W2.5.")
    masks = torch.tensor(masks_for_rate, dtype=torch.int64, device=states.device)
    first = pgc16_mix_states(local_states)
    second = pgc16_mix_states(local_states ^ masks[context_banks])
    indices = torch.stack((first >> 8, first & 0xFF, second >> 8, second & 0xFF), dim=-1)
    return levels[indices].contiguous()


def pgc16_codebook(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize all 65,536 PGC16 vectors for Viterbi search."""

    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("PGC16 codebook dtype must be floating point.")
    target_device = torch.device("cpu" if device is None else device)
    states = torch.arange(PGC16_STATE_COUNT, dtype=torch.int64, device=target_device)
    target_levels = None if levels is None else levels.to(device=target_device)
    return pgc16_decode_states(states, levels=target_levels).to(dtype=dtype)


def pgc16_codebook_v4(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the opt-in 65,536-row, four-scalar PGC16 codebook."""

    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("PGC16 codebook dtype must be floating point.")
    target_device = torch.device("cpu" if device is None else device)
    states = torch.arange(PGC16_STATE_COUNT, dtype=torch.int64, device=target_device)
    target_levels = None if levels is None else levels.to(device=target_device)
    return pgc16_decode_states_v4(states, levels=target_levels).to(dtype=dtype)


def pgc16_codebook_v2_bank(
    bank: int,
    *,
    bits: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize one L16/V2 bank from the experimental P64 portfolio."""

    if isinstance(bank, bool) or not isinstance(bank, int) or not 0 <= bank < PGC16_V4_BANK_COUNT:
        raise ValueError(f"PGC16 V2B4 bank must be an integer in `[0, {PGC16_V4_BANK_COUNT - 1}]`.")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("PGC16 codebook dtype must be floating point.")
    target_device = torch.device("cpu" if device is None else device)
    states = torch.arange(PGC16_STATE_COUNT, dtype=torch.int64, device=target_device)
    selectors = torch.full((PGC16_STATE_COUNT,), bank, dtype=torch.uint8, device=target_device)
    target_levels = None if levels is None else levels.to(device=target_device)
    return pgc16_decode_states_v2_banked(states, selectors, bits=bits, levels=target_levels).to(dtype=dtype)


def pgc16_codebook_v4_bank(
    bank: int,
    *,
    bits: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize one fixed bank of the experimental V4 family."""

    if isinstance(bank, bool) or not isinstance(bank, int) or not 0 <= bank < PGC16_V4_BANK_COUNT:
        raise ValueError(f"PGC16 V4 bank must be an integer in `[0, {PGC16_V4_BANK_COUNT - 1}]`.")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("PGC16 codebook dtype must be floating point.")
    target_device = torch.device("cpu" if device is None else device)
    states = torch.arange(PGC16_STATE_COUNT, dtype=torch.int64, device=target_device)
    selectors = torch.tensor(bank, dtype=torch.uint8, device=target_device)
    target_levels = None if levels is None else levels.to(device=target_device)
    return pgc16_decode_states_v4_banked(states, selectors, bits=bits, levels=target_levels).to(dtype=dtype)


def pgc18_codebook_v4(
    *,
    bits: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the experimental 262,144-row L18/V4 codebook."""

    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("PGC18 V4 codebook dtype must be floating point.")
    target_device = torch.device("cpu" if device is None else device)
    states = torch.arange(PGC18_V4_STATE_COUNT, dtype=torch.int64, device=target_device)
    target_levels = None if levels is None else levels.to(device=target_device)
    return pgc18_decode_states_v4(states, bits=bits, levels=target_levels).to(dtype=dtype)


def pgc16_scale_factor(bits: float) -> float:
    """Return the Gaussian scale multiplier for half-step W1 through W8."""

    transition_bits = qvq_transition_bits(bits)
    return PGC16_SCALE_FACTORS[transition_bits - 2]


__all__ = [
    "PGC16_CODEBOOK_VERSION",
    "PGC16_CODEBOOK_VERSIONS",
    "PGC16_INCREMENT",
    "PGC16_LEVEL_COUNT",
    "PGC16_MULTIPLIER",
    "PGC16_NORMALIZATION_RMS",
    "PGC16_SCALE_FACTORS",
    "PGC16_STATE_COUNT",
    "PGC16_V4_BANK_COUNT",
    "PGC16_V2B4_BANK_XOR_MASKS_BY_TRANSITION_BITS",
    "PGC16_V4_BANK_XOR_MASKS",
    "PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS",
    "PGC18_V4_STATE_COUNT",
    "canonical_pgc16_levels",
    "pgc16_codebook",
    "pgc16_codebook_v2_bank",
    "pgc16_codebook_v4",
    "pgc16_codebook_v4_bank",
    "pgc16_decode_states",
    "pgc16_decode_states_v2_banked",
    "pgc16_decode_states_v4",
    "pgc16_decode_states_v4_banked",
    "pgc16_levels_for_version",
    "pgc16_mix_states",
    "pgc16_scale_factor",
    "pgc18_codebook_v4",
    "pgc18_decode_states_v4",
    "validate_pgc16_levels",
]
