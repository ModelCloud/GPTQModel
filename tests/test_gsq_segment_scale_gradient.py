# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""The contiguous segment-sum scale gradient must match the scatter reference."""

import pytest
import torch

from gptqmodel.quantization.gsq_training import (
    GSQScalarTrainingModule,
    relaxed_scalar_weights,
)


def _module(rows, columns, group, *, bits, device):
    choices = 4 if bits == 2 else 5
    torch.manual_seed(0)
    weight = torch.randn(rows, columns, device=device, dtype=torch.bfloat16)*.02
    scales = weight.abs().view(rows, -1, group).amax(2).clamp_min(1e-3)/2.5
    noise = torch.randn(choices, rows, columns, device=device, dtype=torch.bfloat16)
    return GSQScalarTrainingModule(weight, scales, group, bits=bits, noise=noise)


@pytest.mark.parametrize("group", [64, 128])
@pytest.mark.parametrize("bits", [2, 3])
def test_segment_scale_gradient_matches_scatter(group, bits):
    """Grouping is a reduction-order change, not a semantic one."""
    module = _module(64, 512, group, bits=bits, device="cpu")
    uniform = torch.rand(module.candidates.shape, dtype=torch.bfloat16)
    upstream = torch.randn(64, 512, dtype=torch.bfloat16)

    # Default path: the module supplies the contiguous group size.
    weights = module(uniform=uniform, temperature=1., multiplier=100.)
    weights.backward(upstream)
    segment_scales = module.scales.grad.clone()
    module.zero_grad(set_to_none=True)

    # Reference path: withhold the contiguity hint to force scatter_add_.
    candidates = module.candidates
    if module.initial is not None:
        candidates = torch.arange(
            -2, 3, device=module.candidates.device, dtype=module.candidates.dtype,
        )[:, None, None].expand_as(module.candidates)
    reference = relaxed_scalar_weights(
        module.logits.masked_fill(~module.valid, -torch.inf), module.scales,
        candidates, module.group_index, uniform=uniform,
        temperature=1., multiplier=100., initial=module.initial,
    )
    reference.backward(upstream)
    scatter_scales = module.scales.grad.clone()

    assert torch.equal(weights, reference)
    assert segment_scales.shape == scatter_scales.shape
    torch.testing.assert_close(
        segment_scales, scatter_scales, rtol=0., atol=scatter_scales.abs().max()*1e-5,
    )


def test_contiguous_hint_rejects_a_permuted_group_map():
    """A non-arange map must not silently take the segment-sum reduction."""
    module = _module(8, 512, 128, bits=2, device="cpu")
    permuted = module.group_index.flip(0).contiguous()
    uniform = torch.rand(module.candidates.shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="arange group map"):
        relaxed_scalar_weights(
            module.logits.masked_fill(~module.valid, -torch.inf), module.scales,
            module.candidates, permuted, uniform=uniform,
            temperature=1., multiplier=100., contiguous_group_size=128,
        )
