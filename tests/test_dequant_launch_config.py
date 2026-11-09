# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Per-architecture launch table for the Triton dequant kernel.

The dequant launch shape is a per-architecture decision.  RDNA4 used to inherit
the CDNA tile with an explicit "not measured yet" comment; it now carries a
measured tile.  These assertions pin both so a future edit cannot silently
re-introduce the unmeasured inheritance.
"""

from __future__ import annotations

import pytest

from gptqmodel.nn_modules.triton_utils.dequant import (
    DEQUANT_NUM_STAGES,
    DEQUANT_NUM_WARPS,
    DEQUANT_X_BLOCK,
    DequantLaunch,
    launch_config,
)


@pytest.mark.parametrize(
    ("arch_name", "expected"),
    [
        ("gfx950", DequantLaunch(x_block=1024, num_warps=1, num_stages=1)),
        ("gfx1200", DequantLaunch(x_block=2048, num_warps=4, num_stages=1)),
        ("gfx1201", DequantLaunch(x_block=2048, num_warps=4, num_stages=1)),
        ("gfx1201:sramecc+:xnack-", DequantLaunch(x_block=2048, num_warps=4, num_stages=1)),
        ("gfx1250", DequantLaunch(x_block=2048, num_warps=4, num_stages=1)),
        # Ada / RTX 4090 (SM 8.9): measured on a 4090; routed from the ``sm_*``
        # string and (equivalently) from a ``(8, 9)`` capability tuple.
        ("sm_89", DequantLaunch(x_block=512, num_warps=1, num_stages=2)),
    ],
)
def test_dequant_launch_config_is_arch_tabled(arch_name, expected):
    assert launch_config(arch_name) == expected


def test_rdna4_dequant_launch_differs_from_cdna():
    # The whole point of the entry: RDNA4 must not simply mirror CDNA.
    assert launch_config("gfx1201") != launch_config("gfx950")


def test_ada_dequant_launch_is_capability_tuple():
    # A capability tuple must reach the same entry as the sm_89 string.
    assert launch_config((8, 9)) == launch_config("sm_89")
    assert launch_config((8, 9)) != launch_config(None)


@pytest.mark.parametrize("arch_name", ["gfx942", "sm_80", "sm_90", "", "not-a-gfx-arch"])
def test_dequant_launch_config_falls_back_to_default(arch_name):
    # Unregistered architectures and non-ROCm identifiers keep the CUDA default.
    assert launch_config(arch_name) == DequantLaunch(
        x_block=DEQUANT_X_BLOCK,
        num_warps=DEQUANT_NUM_WARPS,
        num_stages=DEQUANT_NUM_STAGES,
    )


def test_dequant_launch_config_defaults_are_cdna_compatible():
    # Module-level symbols remain the compatibility contract.
    assert (DEQUANT_X_BLOCK, DEQUANT_NUM_WARPS, DEQUANT_NUM_STAGES) == (1024, 1, 1)
    assert launch_config("gfx950") == DequantLaunch(
        x_block=DEQUANT_X_BLOCK,
        num_warps=DEQUANT_NUM_WARPS,
        num_stages=DEQUANT_NUM_STAGES,
    )
