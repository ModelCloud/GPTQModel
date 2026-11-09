# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the identity/rules cases (they use a synthetic identity); the
# real-device case carries ``pytest.mark.cuda`` and skips without an RTX 4090.
"""AWQ Triton plan table for the RTX 4090 (Ada, SM 8.9) profile.

The table is guarded by an *exact* four-field device profile
(name, capability, SM count, total_memory).  These tests pin that guard, prove
the FP16 rules only fire for the measured profile/dtype triple, and verify on a
real 4090 that ``device_identity`` matches the constant exactly.  No BF16 rule
set exists because ``awq_gemm_triton`` rejects non-FP16 input and scales.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.awq.modules.triton import gemm as awq_gemm_mod
from gptqmodel.quantization.awq.modules.triton.scheduler import (
    _VERIFIED_4090_FP16_RULES,
    _VERIFIED_4090_PROFILE,
    _builtin_plan,
    device_identity,
)
from gptqmodel.utils import arch


requires_ada = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip is None and arch.is_ada(0)),
    reason="Ada (SM 8.9) CUDA GPU required",
)

_IDENTITY = (torch.device("cuda", 0), "cuda:NVIDIA GeForce RTX 4090", *_VERIFIED_4090_PROFILE)
FP16 = torch.float16
BF16 = torch.bfloat16


def _plan(identity, *, M=32, N=8192, K=2048, G=128, dtype=FP16, accum=True):
    return _builtin_plan(
        identity, M=M, N=N, K=K, G=G,
        input_dtype=dtype, compute_dtype=dtype, output_dtype=dtype, fp32_accum=accum,
    )


def test_verified_profile_is_the_measured_4090_tuple():
    assert _VERIFIED_4090_PROFILE == (
        "NVIDIA GeForce RTX 4090", (8, 9), 128, 50_950_569_984,
    )


def test_builtin_plan_requires_the_exact_profile():
    assert _plan(_IDENTITY) is not None
    # Each profile field is an exact-match guard: a lone field change must
    # disable the measured table rather than fall back to a neighbouring tile.
    name, capability, sms, memory = _VERIFIED_4090_PROFILE
    for mutated in (
        ("NVIDIA GeForce RTX 4090 D", capability, sms, memory),
        (name, (8, 6), sms, memory),
        (name, capability, 64, memory),
        (name, capability, sms, memory - 1),
        (name, capability, sms, memory + 1024 ** 3),
    ):
        assert _plan((torch.device("cuda", 0), name, *mutated)) is None, mutated


def test_builtin_plan_is_fp16_only():
    # BF16 has no measured rule set and the kernel refuses BF16 outright.
    assert _plan(_IDENTITY, dtype=BF16) is None
    assert _plan(_IDENTITY, dtype=torch.float32) is None
    # fp32 accumulation is part of the measured contract.
    assert _plan(_IDENTITY, accum=False) is None
    # A ROCm identity (7th field = gfx arch) can never match the NVIDIA profile.
    assert _plan(_IDENTITY + ("gfx1201",)) is None


def test_builtin_plan_is_exact_shape_only():
    assert _plan(_IDENTITY, M=32, N=8192, K=2048, G=128) is not None
    # Neighbouring M and unmeasured shapes fall back to legacy policy.
    assert _plan(_IDENTITY, M=31) is None
    assert _plan(_IDENTITY, M=32, N=8192, K=2048, G=64) is None
    assert _plan(_IDENTITY, M=7, N=512, K=512, G=128) is None


def test_no_bf16_rule_set_is_installed():
    # ``awq_gemm_triton`` rejects BF16 input/scales, so a BF16 plan could not
    # be executed, let alone measured: keep it out of the table.
    assert all(plan is not None for plan in _VERIFIED_4090_FP16_RULES.values())
    with pytest.raises(ValueError, match="requires float16 input and scales"):
        awq_gemm_mod.awq_gemm_triton(
            torch.zeros(4, 128, device="cpu", dtype=torch.bfloat16),
            torch.zeros(128, 16, dtype=torch.int32),
            torch.zeros(1, 128, dtype=torch.bfloat16),
            torch.zeros(1, 16, dtype=torch.int32),
            1,
        )


@requires_ada
@pytest.mark.cuda
def test_real_4090_identity_matches_profile():
    identity = device_identity(torch.device("cuda:0"))
    assert identity[2:6] == _VERIFIED_4090_PROFILE, identity[2:6]
    # And the guard therefore accepts the real device for a measured shape.
    assert _plan(identity) is not None
