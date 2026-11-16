# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""EXL3 trellis K bounds (issue #3971).

The native ``reconstruct`` op selects one of 24 kernel instantiations from
``K`` and the ``mcg``/``mul1`` variant. Out-of-range K indexed past the table
and launched a garbage function pointer.
"""

import pytest

from gptqmodel.exllamav3.trellis import validate_trellis_k


def test_exl3_trellis_k_accepts_supported_range():
    for k in range(1, 9):
        validate_trellis_k(k)
    validate_trellis_k(3, mcg=True)
    validate_trellis_k(3, mul1=True)


@pytest.mark.parametrize("bad", [0, -1, 9, 16, 24])
def test_exl3_trellis_k_rejects_out_of_table(bad):
    with pytest.raises(ValueError, match="K must be in"):
        validate_trellis_k(bad)


def test_exl3_trellis_k_rejects_mcg_and_mul1():
    with pytest.raises(ValueError, match="mcg and mul1"):
        validate_trellis_k(2, mcg=True, mul1=True)
