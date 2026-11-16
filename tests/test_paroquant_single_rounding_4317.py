# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""ParoQuant bf16 rotation must round once on every execution path (#4317).

c45c1b0 made only the Torch reference accumulate the Givens sweep in FP32.
The fused ``apply_paroquant_rotation`` op used by ``ParoLinear._rotate_inputs``
(and inherited unchanged by ``ParoQuantTritonLinear``) kept the running
activation in an FP16 shared-memory workspace, so it rounded after every
stage and drifted from the single-rounding reference on deep ``krot=8``
rotation stacks.

The fused bf16 kernel now keeps an FP32 workspace and narrows only on store.
These tests pin that contract against an FP64 Givens oracle; they skip when
the fused extension or Triton is unavailable.
"""

import pytest
import torch

from gptqmodel.utils.paroquant import (
    _load_rotation_extension,
    _rotation_kernel_ready,
    apply_paroquant_rotation,
    apply_paroquant_rotation_reference,
    apply_paroquant_rotation_vectorized,
)


def _rotation_case(
    *,
    rows: int = 256,
    hidden: int = 2048,
    group_size: int = 128,
    krot: int = 8,
    seed: int = 20261116,
    device: str = "cpu",
):
    """Build the issue's krot=8 / non-unit-scale / theta-in-±1 fixture.

    theta and channel_scales are BF16 because that is what the runtime caches
    for BF16 activations; both the reference and the fused op consume the same
    values, so the comparison isolates the workspace rounding contract.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = hidden // group_size
    half_group = group_size // 2
    pair_rows = []
    for _ in range(krot):
        per_group = []
        for _ in range(groups):
            permutation = torch.randperm(group_size, generator=generator)
            per_group.append(
                torch.stack([permutation[:half_group], permutation[half_group:]], dim=-1).reshape(-1)
            )
        pair_rows.append(torch.cat(per_group))

    pairs = torch.stack(pair_rows).to(torch.int16)
    theta = (torch.rand(krot, hidden // 2, generator=generator) * 2.0 - 1.0).to(torch.bfloat16)
    scales = (0.5 + torch.rand(hidden, generator=generator)).to(torch.bfloat16)
    x = torch.randn(rows, hidden, generator=generator, dtype=torch.bfloat16)
    return x.to(device), pairs.to(device), theta.to(device), scales.to(device)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def test_paroquant_bf16_reference_matches_fp64_givens_oracle():
    """The CPU reference/vectorized paths are the single-rounding floor."""

    x, pairs, theta, scales = _rotation_case(rows=64, hidden=512)
    oracle = apply_paroquant_rotation_reference(
        x.double(), pairs, theta.double(), scales=scales.double(), group_size=128
    )
    reference = apply_paroquant_rotation_reference(x, pairs, theta, scales=scales, group_size=128)
    vectorized = apply_paroquant_rotation_vectorized(x, pairs, theta, scales=scales, group_size=128)

    assert reference.dtype is torch.bfloat16
    assert vectorized.dtype is torch.bfloat16
    assert _relative_error(reference, oracle) < 4.0e-3
    torch.testing.assert_close(reference, vectorized, rtol=0, atol=0)


def test_paroquant_cpu_dispatch_still_selects_the_reference():
    """Without a CUDA operand the fused op must dispatch to the reference."""

    x, pairs, theta, scales = _rotation_case(rows=32, hidden=512)
    assert not _rotation_kernel_ready(x, pairs, theta, scales, 128)

    expected = apply_paroquant_rotation_reference(x, pairs, theta, scales=scales, group_size=128)
    actual = apply_paroquant_rotation(x, pairs, theta, scales=scales, group_size=128)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused ParoQuant rotation requires CUDA or ROCm")
@pytest.mark.cuda
def test_paroquant_fused_bf16_rotation_rounds_once():
    """The fused op must match the reference, not the old per-stage rounding."""

    if not _load_rotation_extension():
        pytest.skip("fused ParoQuant rotation extension unavailable")

    x, pairs, theta, scales = _rotation_case(device="cuda")
    # Guard against a silent fallback that would make the comparison vacuous.
    assert _rotation_kernel_ready(x, pairs, theta, scales, 128)
    oracle = apply_paroquant_rotation_reference(
        x.double(), pairs, theta.double(), scales=scales.double(), group_size=128
    )
    reference = apply_paroquant_rotation_reference(x, pairs, theta, scales=scales, group_size=128)
    fused = apply_paroquant_rotation(x, pairs, theta, scales=scales, group_size=128)

    assert fused.dtype is torch.bfloat16
    reference_error = _relative_error(reference, oracle)
    fused_error = _relative_error(fused, oracle)

    # The FP16-workspace kernel was ~8% above the single-rounding floor here
    # and ~1.7e-3 away from the reference; leave 5% for FMA/sincos ordering.
    assert fused_error <= reference_error * 1.05
    assert _relative_error(fused, reference) < 5.0e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ParoQuantTritonLinear requires CUDA or ROCm")
@pytest.mark.cuda
def test_paroquant_triton_linear_bf16_rotation_rounds_once():
    """ParoQuantTritonLinear inherits the fused rotation; pin the same contract."""

    pytest.importorskip("triton")
    if not _load_rotation_extension():
        pytest.skip("fused ParoQuant rotation extension unavailable")

    from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear

    x, pairs, theta, scales = _rotation_case(rows=64, hidden=512)
    module = ParoQuantTritonLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=512,
        out_features=8,
        bias=False,
        krot=8,
    )
    module = module.cuda().eval()
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(scales)
    module._rotation_identity = False

    x = x.cuda()
    runtime_theta, runtime_scales = module._ensure_rotation_runtime_dtype(x.device, x.dtype)
    # ParoQuantTritonLinear must reach the fused rotation, not the reference.
    assert _rotation_kernel_ready(x, module.pairs, runtime_theta, runtime_scales, 128)
    actual = module._rotate_inputs(x)
    reference = apply_paroquant_rotation_reference(
        x, module.pairs, runtime_theta, scales=runtime_scales, group_size=128
    )
    oracle = apply_paroquant_rotation_reference(
        x.double(),
        module.pairs,
        runtime_theta.double(),
        scales=runtime_scales.double(),
        group_size=128,
    )

    assert actual.dtype is torch.bfloat16
    assert _relative_error(actual, reference) < 5.0e-4
    assert _relative_error(actual, oracle) <= _relative_error(reference, oracle) * 1.05
