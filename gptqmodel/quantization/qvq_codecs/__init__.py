# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Production QVQ state decoders.

Only fixed PGC16-v1 is a production checkpoint codec. HYB, uniform, and the
retired learned PGC16-v2 implementation live in explicit reference/deprecated
modules so a model loader cannot select them accidentally.
"""

from .pgc16 import (
    PGC16_CODEBOOK_VERSION,
    PGC16_CODEBOOK_VERSIONS,
    PGC16_INCREMENT,
    PGC16_LEVEL_COUNT,
    PGC16_MULTIPLIER,
    PGC16_NORMALIZATION_RMS,
    PGC16_SCALE_FACTORS,
    PGC16_STATE_COUNT,
    PGC16_V4_BANK_COUNT,
    PGC16_V4_BANK_XOR_MASKS,
    PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS,
    canonical_pgc16_levels,
    pgc16_codebook,
    pgc16_codebook_v4,
    pgc16_codebook_v4_bank,
    pgc16_decode_states,
    pgc16_decode_states_v4,
    pgc16_decode_states_v4_banked,
    pgc16_levels_for_version,
    pgc16_mix_states,
    pgc16_scale_factor,
    validate_pgc16_levels,
)


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
    "PGC16_V4_BANK_XOR_MASKS",
    "PGC16_V4_BANK_XOR_MASKS_BY_TRANSITION_BITS",
    "canonical_pgc16_levels",
    "pgc16_codebook",
    "pgc16_codebook_v4",
    "pgc16_codebook_v4_bank",
    "pgc16_decode_states",
    "pgc16_decode_states_v4",
    "pgc16_decode_states_v4_banked",
    "pgc16_levels_for_version",
    "pgc16_mix_states",
    "pgc16_scale_factor",
    "validate_pgc16_levels",
]
