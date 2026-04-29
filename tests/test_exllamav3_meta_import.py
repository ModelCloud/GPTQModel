# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression test for module-level import of exllamav3_torch under a
meta-device preload context.

Background
----------
``gptqmodel.nn_modules.exllamav3_torch`` evaluates two module-level constants
(``_EXL3_MUL1_INV``, ``_EXL3_MUL1_BIAS``) by calling ``_half_scalar_from_bits``
during import. The previous implementation built a torch tensor and called
``.item()`` on it, which fails on a meta tensor with::

    RuntimeError: Tensor.item() cannot be called on meta tensors

Transformers' AWQ pathway (``replace_with_awq_linear``) imports
``gptqmodel.quantization`` inside ``with torch.device('meta'):``, so any
GPTQ load path on transformers >= 5.6 would surface that error transitively.
This test guards against regression of that import path.
"""

from __future__ import annotations

import struct
import sys

import torch


_MOD = "gptqmodel.nn_modules.exllamav3_torch"


def _drop_module():
    """Drop the module from ``sys.modules`` so the next import re-runs the
    module body. The full ``gptqmodel`` package import (already completed by
    the test session) caches the module, masking import-time failures."""
    if _MOD in sys.modules:
        del sys.modules[_MOD]


def test_half_scalar_from_bits_matches_torch_view_path():
    """The ``struct``-based conversion must return the same float as the
    historical ``torch.tensor(...).view(torch.float16).item()`` path for
    every uint16 bit pattern of interest, including the two canonical
    EXL3 constants and a handful of edge cases (zero, +/-1.0, max
    finite, NaN)."""
    _drop_module()
    from gptqmodel.nn_modules.exllamav3_torch import _half_scalar_from_bits

    cases = [
        0x0000,  # +0.0
        0x3C00,  # +1.0
        0xBC00,  # -1.0
        0x1EEE,  # _EXL3_MUL1_INV
        0xC931,  # _EXL3_MUL1_BIAS
        0x7BFF,  # max finite
        0xFBFF,  # min finite
    ]
    for bits in cases:
        got = _half_scalar_from_bits(bits)
        want_struct = float(struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0])
        want_torch = float(
            torch.tensor([bits], dtype=torch.uint16).view(torch.float16).item()
        )
        assert got == want_struct == want_torch, (
            f"bits=0x{bits:04X} got={got!r} want_struct={want_struct!r} "
            f"want_torch={want_torch!r}"
        )

    # NaN bit patterns: torch and struct must both produce NaN; equality
    # via ``!= self`` because NaN != NaN.
    for bits in (0x7FFF, 0xFFFF):
        got = _half_scalar_from_bits(bits)
        assert got != got, f"bits=0x{bits:04X} expected NaN, got {got!r}"


def test_module_imports_under_meta_device():
    """Loading ``exllamav3_torch`` inside ``with torch.device('meta'):`` must
    not raise. This reproduces the transformers AWQ pathway that calls
    ``replace_with_awq_linear`` while a meta-device guard is active."""
    _drop_module()
    with torch.device("meta"):
        import gptqmodel.nn_modules.exllamav3_torch as exl  # noqa: F401

    # Module-level constants must still be plain Python floats (not tensors).
    assert isinstance(exl._EXL3_MUL1_INV, float)
    assert isinstance(exl._EXL3_MUL1_BIAS, float)


def test_module_constants_are_canonical_values():
    """Lock in the exact float64 values of the two module-level constants so
    any change to the bit-pattern math (or to the underlying conversion)
    is caught here rather than downstream in a kernel correctness test."""
    _drop_module()
    import gptqmodel.nn_modules.exllamav3_torch as exl

    # Float64-precise values reproduced from the IEEE-754 binary16 patterns.
    assert exl._EXL3_MUL1_INV == 0.00676727294921875
    assert exl._EXL3_MUL1_BIAS == -10.3828125
