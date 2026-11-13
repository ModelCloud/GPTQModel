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
    ("paroquant_gemm", "gptqmodel.quantization.paroquant.modules.triton.gemm", "_PAROQUANT_GEMM_LAUNCH_CONFIGS"),
    ("awq_gemv", "gptqmodel.quantization.awq.modules.triton.gemv", "_AWQ_GEMV_LAUNCH_CONFIGS"),
    ("bitsandbytes_4bit_gemm", "gptqmodel.nn_modules.qlinear.bitsandbytes_triton", "_BNB_GEMM_LAUNCH"),
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


def test_paroquant_launch_table_is_per_family_and_row_banded():
    """ParoQuant keeps its historical tile off-table and measured Ada/RDNA4 tiles."""

    from gptqmodel.quantization.paroquant.modules.triton import gemm as paro_gemm

    table = paro_gemm._PAROQUANT_GEMM_LAUNCH_CONFIGS
    assert "rdna4" in table, "ParoQuant: no measured RDNA4 launch entry"
    assert "ada" in table, "ParoQuant: no measured Ada launch entry"

    # Families with no measured entry keep the tile that shipped before the
    # table existed.  Ada and RDNA4 are measured now and are covered below.
    for family in ("other", "hopper", "blackwell", "cdna4"):
        assert (
            paro_gemm.paroquant_gemm_launch_config("decode", "gfx942")
            == paro_gemm._PAROQUANT_GEMM_DEFAULT_CONFIGS["decode"]
        )
    assert (
        paro_gemm.paroquant_gemm_launch_config("prefill", "gfx942")
        == paro_gemm._PAROQUANT_GEMM_DEFAULT_CONFIGS["prefill"]
    )

    # RDNA4 is row-banded: <=8 rows, 9..64 rows, >64 rows.
    small = paro_gemm.paroquant_gemm_launch_config("decode", "gfx1201", rows=8)
    medium = paro_gemm.paroquant_gemm_launch_config("prefill", "gfx1201", rows=9)
    large = paro_gemm.paroquant_gemm_launch_config("prefill", "gfx1201", rows=65)
    assert small == table["rdna4"]["decode"]
    assert medium == table["rdna4"]["prefill"]
    assert large == table["rdna4"]["prefill_large"]
    assert len({small, medium, large}) == 3
    # num_warps=8 lost the measured prefill band on wave32.
    for tile in (small, medium, large):
        assert tile[3] in (4, 8)
    assert medium[3] == 4 and large[3] == 4

    # Ada (RTX 4090) is row-banded the same way with its own measured tiles.
    ada_small = paro_gemm.paroquant_gemm_launch_config("decode", (8, 9), rows=8)
    ada_medium = paro_gemm.paroquant_gemm_launch_config("prefill", "sm_89", rows=9)
    ada_large = paro_gemm.paroquant_gemm_launch_config("prefill", (8, 9), rows=65)
    assert ada_small == table["ada"]["decode"]
    assert ada_medium == table["ada"]["prefill"]
    assert ada_large == table["ada"]["prefill_large"]
    assert len({ada_small, ada_medium, ada_large}) == 3


def test_gguf_qk_table_is_ada_only_and_rdna4_falls_through():
    """The GGUF Q4_K/Q5_K/Q6_K fixed tiles are keyed by qtype, not family.

    They are the Ada (SM 8.9) path: every entry must be reachable through the
    SM89 selector and must never fire on RDNA4/other capabilities, which keep
    the kernel's own autotuned generic launch.
    """

    from gptqmodel.nn_modules.qlinear import gguf_triton

    table = gguf_triton._GGUF_SM89_QK_SHAPES
    assert table, "GGUF SM89 table is empty"
    for qtype, bands in table.items():
        for band, shapes in bands.items():
            for (in_features, out_features), config in shapes.items():
                rows = 1 if band == 1 else 8 if band == "mid" else 64
                assert gguf_triton._select_gguf_qk_fixed_launch_config(
                    qtype, capability=(8, 9), rows=rows,
                    in_features=in_features, out_features=out_features,
                ) == config
                # RDNA4/other capability must fall through to the autotuner.
                assert gguf_triton._select_gguf_qk_fixed_launch_config(
                    qtype, capability=None, rows=rows,
                    in_features=in_features, out_features=out_features,
                ) is None


def test_awq_gemm_scheduler_has_both_measured_device_profiles():
    """The AWQ GEMM scheduler must ship a measured plan set for each target.

    Unlike the launch tables above, the AWQ scheduler keys its rules on an exact
    device profile rather than a family token, because neighbouring shapes were
    not stable enough to bucket.  That makes it easy to add one target and
    forget the other, so the contract pins both profiles and proves that a
    near-miss cannot inherit either.
    """

    from gptqmodel.quantization.awq.modules.triton import scheduler

    cuda_identity = ("cuda", 0, *scheduler._VERIFIED_4090_PROFILE)
    rdna4_name, rdna4_arch, rdna4_cus, rdna4_mem = scheduler._VERIFIED_RDNA4_PROFILE
    # A HIP host reports the six CUDA fields plus the gcnArchName last, with
    # ``capability`` always None (get_device_capability is meaningless on ROCm).
    rocm_identity = ("cuda", 0, rdna4_name, None, rdna4_cus, rdna4_mem, rdna4_arch)
    assert len(cuda_identity) == 6
    assert len(rocm_identity) == 7
    assert scheduler._VERIFIED_4090_PROFILE != scheduler._VERIFIED_RDNA4_PROFILE
    assert scheduler._VERIFIED_4090_FP16_RULES
    assert scheduler._VERIFIED_RDNA4_FP16_RULES

    def plan(identity, *, M, N, K, G, dtype=torch.float16):
        return scheduler._builtin_plan(
            identity, M=M, N=N, K=K, G=G,
            input_dtype=dtype, compute_dtype=dtype, output_dtype=dtype,
            fp32_accum=True,
        )

    shape = {"M": 32, "N": 8192, "K": 2048, "G": 128}
    assert plan(cuda_identity, **shape) == scheduler._VERIFIED_4090_FP16_RULES[(32, 8192, 2048, 128)]
    assert plan(rocm_identity, **shape) == scheduler._VERIFIED_RDNA4_FP16_RULES[(32, 8192, 2048, 128)]

    # Neither profile may match the other target's device, a different gfx
    # arch, or a different board/CU/memory count.
    wrong_arch = ("cuda", 0, rdna4_name, None, rdna4_cus, rdna4_mem, "gfx942")
    wrong_mem = ("cuda", 0 + 1, rdna4_name, None, rdna4_cus, 1, rdna4_arch)
    wrong_name = ("cuda", 0 + 1, "AMD Radeon AI PRO R9600", None, rdna4_cus, rdna4_mem, rdna4_arch)
    wrong_sms = ("cuda", 0 + 1, "NVIDIA GeForce RTX 4090", (8, 9), 96, 50_950_569_984)
    for identity in (wrong_arch, wrong_mem, wrong_name, wrong_sms):
        assert plan(identity, **shape) is None

    # The rules are FP16-only: the kernel rejects non-FP16 input/scales, so a
    # BF16 request must not inherit a tile.
    assert plan(cuda_identity, **shape, dtype=torch.bfloat16) is None
    assert plan(rocm_identity, **shape, dtype=torch.bfloat16) is None
