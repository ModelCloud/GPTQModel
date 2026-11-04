# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the identity/rule/gating cases (they use a synthetic identity); the
# real-device case carries ``pytest.mark.cuda`` and skips without a gfx1201.
"""AWQ Triton plan table for the RDNA4 (gfx1201, Radeon AI PRO R9700) profile.

On ROCm ``device_identity`` is a seven-tuple: the six CUDA fields plus the
normalized HIP architecture appended last, with ``capability`` set to ``None``
(``torch.cuda.get_device_capability`` is meaningless on a HIP device).  The
RDNA4 table is therefore guarded by an exact four-field profile
(name, gfx arch, compute-unit count, total_memory) drawn from indices
2/6/4/5.  These tests pin that guard, prove the FP16 rules fire only for the
measured profile and dtype triple, prove a near-miss cannot match, and confirm
the CUDA 4090 profile is still reachable only from a CUDA-shaped identity.
No BF16 rule set exists because ``awq_gemm_triton`` rejects non-FP16 input and
scales.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.awq.modules.triton import gemm as awq_gemm_mod
from gptqmodel.quantization.awq.modules.triton.scheduler import (
    _VERIFIED_4090_FP16_RULES,
    _VERIFIED_4090_PROFILE,
    _VERIFIED_RDNA4_FP16_RULES,
    _VERIFIED_RDNA4_PROFILE,
    _builtin_plan,
    clear_awq_triton_plan_cache,
    device_identity,
    legacy_plan,
    select_awq_triton_plan,
    validate_fused_config,
)
from gptqmodel.utils import arch


requires_rdna4 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip is not None and arch.is_rdna4(0)),
    reason="RDNA4 (gfx120*) GPU required",
)

FP16 = torch.float16
BF16 = torch.bfloat16


def _rocm_identity(*, name=None, gfx=None, cus=None, memory=None):
    """A synthetic ROCm seven-tuple; fields default to the verified profile."""
    p_name, p_gfx, p_cus, p_mem = _VERIFIED_RDNA4_PROFILE
    return (
        "cuda", 0,
        p_name if name is None else name,
        None,  # ROCm capability is always None
        p_cus if cus is None else cus,
        p_mem if memory is None else memory,
        p_gfx if gfx is None else gfx,
    )


def _cuda_identity(profile=_VERIFIED_4090_PROFILE):
    """A synthetic CUDA six-tuple with the given (name, cap, sm, mem) profile."""
    name, cap, sm, mem = profile
    return ("cuda", 0, name, cap, sm, mem)


def _plan(identity, *, M=32, N=8192, K=2048, G=128, dtype=FP16, accum=True):
    return _builtin_plan(
        identity, M=M, N=N, K=K, G=G,
        input_dtype=dtype, compute_dtype=dtype, output_dtype=dtype,
        fp32_accum=accum,
    )


# ---------------------------------------------------------------------------
# Profile guard
# ---------------------------------------------------------------------------


def test_profile_shape_matches_the_real_rocm_tuple_layout():
    # Four fields, taken from indices 2/6/4/5 of the ROCm seven-tuple, exactly
    # as the reference host reported them.
    name, gfx, cus, memory = _VERIFIED_RDNA4_PROFILE
    assert gfx.startswith("gfx12")
    assert isinstance(cus, int) and cus > 0
    assert isinstance(memory, int) and memory > 0
    assert _rocm_identity()[2:7] == (name, None, cus, memory, gfx)


@pytest.mark.parametrize("key", sorted(_VERIFIED_RDNA4_FP16_RULES))
def test_every_rdna4_rule_is_a_valid_fused_tile(key):
    M, N, K, G = key
    plan = _builtin_plan(
        _rocm_identity(), M=M, N=N, K=K, G=G,
        input_dtype=FP16, compute_dtype=FP16, output_dtype=FP16, fp32_accum=True,
    )
    assert plan is not None and plan.fused
    assert plan == _VERIFIED_RDNA4_FP16_RULES[key]
    ok, reason = validate_fused_config(
        M=M, N=N, K=K, group_size=G,
        block_size_m=plan.block_size_m, block_size_n=plan.block_size_n,
        block_size_k=plan.block_size_k, split_k_iters=plan.split_k_iters,
        num_warps=plan.num_warps, num_stages=plan.num_stages,
    )
    assert ok, reason


def test_rdna4_profile_matches_only_the_exact_identity():
    # The exact profile resolves to the measured rule.
    assert _plan(_rocm_identity()) == _VERIFIED_RDNA4_FP16_RULES[(32, 8192, 2048, 128)]


@pytest.mark.parametrize(
    "override",
    [
        {"gfx": "gfx1200"},
        {"gfx": "gfx1250"},
        {"gfx": "gfx942"},
        {"name": "AMD Radeon RX 9070 XT"},
        {"cus": 30},
        {"memory": 34_208_743_424 - 1},
    ],
)
def test_rdna4_near_miss_does_not_match(override):
    identity = _rocm_identity(**override)
    assert _plan(identity) is None


def test_rdna4_rejects_non_fp16_dtype_triple():
    for dtype in (BF16, torch.float32):
        assert _plan(_rocm_identity(), dtype=dtype) is None
    # fp32_accum=False is also rejected: the kernel's FP32 accumulation is part
    # of the measured contract.
    assert _plan(_rocm_identity(), accum=False) is None


def test_unmeasured_shape_falls_through_to_the_generic_policy():
    # A measured device but an unlisted (M, N, K, G) keeps the generic tile.
    assert _plan(_rocm_identity(), M=64, N=8192, K=2048, G=64) is None
    assert _plan(_rocm_identity(), M=33, N=8192, K=2048, G=128) is None


# ---------------------------------------------------------------------------
# CUDA 4090 profile must stay CUDA-only
# ---------------------------------------------------------------------------


def test_cuda_4090_profile_still_matches_a_cuda_identity_only():
    # CUDA-shaped identity: the 4090 table still resolves unchanged.
    assert _plan(_cuda_identity()) == _VERIFIED_4090_FP16_RULES[(32, 8192, 2048, 128)]

    # A ROCm identity whose name/CU/memory happen to equal the 4090 cannot
    # reach the CUDA table: the seventh field is a gfx string, so the ROCm
    # branch is taken and the name does not match the RDNA4 profile.
    name, cap, sm, mem = _VERIFIED_4090_PROFILE
    rocm_lookalike = ("cuda", 0, name, None, sm, mem, "gfx1201")
    assert _plan(rocm_lookalike) is None


def test_rdna4_profile_is_not_reachable_from_a_cuda_identity():
    # Without a seventh HIP-arch field the RDNA4 branch is never taken.
    name, _gfx, cus, memory = _VERIFIED_RDNA4_PROFILE
    cuda_lookalike = ("cuda", 0, name, None, cus, memory)
    assert _plan(cuda_lookalike) is None


# ---------------------------------------------------------------------------
# Selection wiring (auto mode) and the kernel's FP16-only contract
# ---------------------------------------------------------------------------


def test_auto_selects_installed_rdna4_rules_legacy_keeps_generic(monkeypatch):
    identity = _rocm_identity()
    import gptqmodel.quantization.awq.modules.triton.scheduler as sched

    monkeypatch.setattr(sched, "device_identity", lambda device=None: identity)
    clear_awq_triton_plan_cache()

    for key, expected in _VERIFIED_RDNA4_FP16_RULES.items():
        M, N, K, G = key
        plan = select_awq_triton_plan(
            M=M, N=N, K=K, group_size=G, device=torch.device("cuda", 0),
            input_dtype=FP16, compute_dtype=FP16, output_dtype=FP16,
            fp32_accum=True, mode="auto",
        )
        assert plan == expected, key
        legacy = select_awq_triton_plan(
            M=M, N=N, K=K, group_size=G, device=torch.device("cuda", 0),
            input_dtype=FP16, compute_dtype=FP16, output_dtype=FP16,
            fp32_accum=True, mode="legacy",
        )
        assert legacy == legacy_plan(M, K, N, G), key

    clear_awq_triton_plan_cache()


def test_rdna4_rule_table_is_fp16_only_and_dense_free():
    # Every installed rule is a fused tile (no dense fallback is installed,
    # because the fused kernel won every measured shape).
    assert _VERIFIED_RDNA4_FP16_RULES
    assert all(p.fused for p in _VERIFIED_RDNA4_FP16_RULES.values())
    # And no rule spans two quantization groups.
    for (M, N, K, G), plan in _VERIFIED_RDNA4_FP16_RULES.items():
        assert plan.block_size_k <= G and G % plan.block_size_k == 0


def test_kernel_rejects_non_fp16_input():
    # The reason the rule set is FP16-only: the kernel itself refuses BF16.
    qweight = torch.zeros(64, 8, dtype=torch.int32)
    scales = torch.zeros(1, 64, dtype=torch.bfloat16)
    qzeros = torch.zeros(1, 8, dtype=torch.int32)
    inputs = torch.zeros(4, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float16"):
        awq_gemm_mod.awq_gemm_triton(inputs, qweight, scales, qzeros, 1)


# ---------------------------------------------------------------------------
# Real device (skipped unless a gfx120* GPU is present)
# ---------------------------------------------------------------------------


@requires_rdna4
def test_real_device_identity_matches_the_rdna4_profile():
    identity = device_identity(torch.device("cuda", 0))
    assert len(identity) == 7
    assert identity[3] is None, "ROCm capability must stay None"
    assert (identity[2], identity[6], identity[4], identity[5]) == _VERIFIED_RDNA4_PROFILE


@requires_rdna4
def test_real_device_auto_selects_the_installed_rule():
    clear_awq_triton_plan_cache()
    plan = select_awq_triton_plan(
        M=32, N=8192, K=2048, group_size=128, device=torch.device("cuda", 0),
        input_dtype=FP16, compute_dtype=FP16, output_dtype=FP16,
        fp32_accum=True, mode="auto",
    )
    assert plan == _VERIFIED_RDNA4_FP16_RULES[(32, 8192, 2048, 128)]
    clear_awq_triton_plan_cache()
