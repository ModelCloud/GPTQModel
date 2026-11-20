# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AVX512 GPTQ dense fallback: FP16 cold start and post-warm-up reuse (#5759).

The dense fallback of the AVX512-BF16 GPTQ kernel decodes the live packed
layout. Before this regression it called the inherited ATen hooks with the
pre-#5596 signatures, so an FP16 checkpoint (which never reaches the BF16-only
fused kernel) raised ``TypeError`` and a grad-mode forward after the fused
warm-up raised on the ATen-only block-size probe.

These nodes pin both entry points against an FP64 Torch oracle built from the
checkpoint codes, integer zero points, and stored scales.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from tests.test_cpu_int4_snapshot_5013 import _permuted_g_idx
from tests.test_torch_cpu_avx512 import _fill_gptq_buffers, _make_layer


pytestmark = [pytest.mark.cpu]

IN_FEATURES = 128
OUT_FEATURES = 64
GROUP_SIZE = 32
TOKENS = 7


def _skip_unless_avx512():
    if not has_avx512_bf16():
        pytest.skip(avx512_bf16_error())
    ok, err = TorchCpuAvx512Linear.cached_validate_once()
    if not ok:
        pytest.skip(str(err))


def _layer_and_oracle(seed: int = 5759):
    """A GPTQ layer with a noncanonical ``g_idx`` and its FP64 weight oracle."""

    torch.manual_seed(seed)
    groups = IN_FEATURES // GROUP_SIZE
    layer = _make_layer(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        group_size=GROUP_SIZE,
        desc_act=True,
    )
    codes = torch.randint(0, 16, (IN_FEATURES, OUT_FEATURES), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, OUT_FEATURES), dtype=torch.int32)
    scales = (
        torch.rand((groups, OUT_FEATURES), dtype=torch.float32) * 0.04 + 0.01
    ).to(torch.float16)
    g_idx = _permuted_g_idx(IN_FEATURES, GROUP_SIZE, seed=seed)
    _fill_gptq_buffers(layer, codes=codes, zeros=zeros, scales=scales, g_idx=g_idx)
    assert not layer._ret_idx_is_identity()

    group_of_row = g_idx.to(torch.long)
    weight = (
        codes.double() - zeros.index_select(0, group_of_row).double()
    ) * scales.index_select(0, group_of_row).double()
    return layer, weight


def _visible_ulp(reference: torch.Tensor) -> float:
    """One representable step at each reference value, as the test tolerance."""

    return (
        torch.nextafter(reference, torch.full_like(reference, float("inf"))).float()
        - reference.float()
    ).abs().max().item()


def test_fp16_dense_fallback_matches_fp64_oracle():
    """A cold FP16 layer never reaches the BF16 fused op; decode must work."""

    _skip_unless_avx512()
    layer, weight = _layer_and_oracle()
    assert layer._cpu_transformed is False

    # The FP64 decode is bit-exact against the checkpoint identity, so a wrong
    # packed layout or zero-point mapping fails here rather than in the matmul.
    assert torch.equal(layer.dequantize_weight(dtype=torch.float64), weight)

    x = torch.randn(TOKENS, IN_FEATURES, dtype=torch.float16)
    layer.eval()
    output = layer(x)

    assert output.dtype == torch.float16
    assert layer._cpu_transformed is False
    # The fallback decodes at the checkpoint width (FP16), the widest exact
    # representation of ``(code - zero) * scale`` for this fixture.
    reference = (x.double() @ weight.to(torch.float16).double()).to(torch.float16)
    torch.testing.assert_close(
        output.float(),
        reference.float(),
        rtol=0,
        atol=max(_visible_ulp(reference), torch.finfo(torch.float16).tiny),
    )


def test_dense_fallback_after_fused_warm_up_matches_fp64_oracle():
    """A grad call after the fused warm-up decodes the live AVX512 layout."""

    _skip_unless_avx512()
    layer, weight = _layer_and_oracle()
    layer.eval()

    warm_up = torch.randn(TOKENS, IN_FEATURES, dtype=torch.bfloat16)
    with torch.no_grad():
        fused = layer(warm_up)
    assert layer._cpu_transformed is True
    assert layer.linear_mode == "inference"

    # The live layout is no longer the checkpoint layout: a decode that
    # assumed the original packed words would produce wrong numbers here.
    live = layer._runtime_dense_weight(torch.bfloat16)
    torch.testing.assert_close(live.double(), weight, rtol=0, atol=1e-6)

    x = torch.randn(TOKENS, IN_FEATURES, dtype=torch.bfloat16, requires_grad=True)
    output = layer(x)
    assert output.dtype == torch.bfloat16
    assert output.requires_grad and output.grad_fn is not None
    assert layer.linear_mode == "inference"

    reference = (x.detach().double() @ weight).to(torch.bfloat16)
    torch.testing.assert_close(
        output.float(),
        reference.float(),
        rtol=0,
        atol=max(_visible_ulp(reference), torch.finfo(torch.bfloat16).tiny),
    )

    output.float().sum().backward()
    expected_grad = (
        torch.ones(TOKENS, OUT_FEATURES, dtype=torch.float64) @ weight.t()
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        x.grad.float(),
        expected_grad.float(),
        rtol=0,
        atol=max(_visible_ulp(expected_grad), torch.finfo(torch.bfloat16).tiny),
    )

    # The dense call must not have restored or replaced the fused layout.
    with torch.no_grad():
        again = layer(warm_up)
    assert layer._cpu_transformed is True
    assert layer.linear_mode == "inference"
    torch.testing.assert_close(again.float(), fused.float(), rtol=0, atol=0)
