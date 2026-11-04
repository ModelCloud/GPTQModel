# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Hardware-parity contract: every tuned pipeline has a 4090 and an RDNA4 path.

This module is the single acceptance gate for "both RTX 4090 (Ada / SM 8.9) and
RDNA4 (gfx1200/gfx1201/gfx1250) have an optimized kernel path" across the
supported quantization pipelines and post-quantization inference.  It is
deliberately hardware-free: it inspects the dispatch tables and predicates, not
timings, so it runs in CI on CPU and fails loudly when a new quantization
pipeline is added without both paths.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.utils import arch


# Pipelines that ship tuned kernels and therefore owe a tuned launch entry for
# both hardware families.  Each case names the module that owns the launch
# table and the attribute that maps ``tuning_family()`` -> launch constants.
TUNED_PIPELINES = [
    ("gptq_gemv", "gptqmodel.nn_modules.triton_utils.gptq_gemv", "_GEMV_LAUNCH_CONFIGS"),
    ("gptq_gemm", "gptqmodel.nn_modules.triton_utils.gptq_gemm", "_LAUNCH_CONFIGS"),
    ("gptq_dequant", "gptqmodel.nn_modules.triton_utils.dequant", "_DEQUANT_LAUNCH_CONFIGS"),
]


def _families() -> tuple[str, str]:
    return "ada", "rdna4"


@pytest.mark.parametrize(("name", "module_name", "attr"), TUNED_PIPELINES)
def test_launch_table_has_both_families(name, module_name, attr):
    """Each fused Triton pipeline must register an Ada and an RDNA4 entry."""

    module = pytest.importorskip(module_name)
    table = getattr(module, attr, None)
    assert table is not None, f"{name}: missing launch table {attr}"
    for family in _families():
        assert family in table, (
            f"{name}: no tuned launch entry for `{family}` in {module_name}.{attr}; "
            f"found {sorted(table)}. Both RTX 4090 (ada) and RDNA4 (rdna4) must have a path."
        )


def test_cuda_family_classification():
    """The shared family vocabulary must resolve the two CUDA/AMD targets."""

    assert arch.tuning_family((8, 9)) == "ada"
    assert arch.tuning_family("sm_89") == "ada"
    assert arch.tuning_family((9, 0)) == "hopper"
    assert arch.tuning_family((10, 0)) == "blackwell"
    assert arch.tuning_family("gfx1201") == "rdna4"
    assert arch.tuning_family("gfx1200") == "rdna4"
    assert arch.tuning_family("gfx1250") == "rdna4"
    # Untuned targets must stay on the portable fallback rather than inherit
    # an SM90+ or CDNA tile.
    assert arch.tuning_family((8, 0)) == "other"
    assert arch.tuning_family("gfx942") == "other"


def test_ada_is_not_promoted_to_hopper():
    """RTX 4090 must not be routed to SM90 launch tables."""

    assert arch.is_ada((8, 9)) is True
    assert arch.is_ada("sm_89") is True
    assert arch.is_ada((9, 0)) is False
    assert arch.is_ada((10, 0)) is False
