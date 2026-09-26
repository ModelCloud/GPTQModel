# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Independent gradient checks for the staged GSQ scalar relaxation."""

import pytest
import torch

from gptqmodel.quantization.gsq_training import relaxed_scalar_weights


@pytest.mark.parametrize("with_initial", [False, True])
def test_scalar_relaxation_matches_autograd(with_initial):
    torch.manual_seed(19)
    logits = torch.randn(5, 3, 8, dtype=torch.float64, requires_grad=True)
    scales = torch.rand(3, 2, dtype=torch.float64, requires_grad=True) + 0.3
    candidates = torch.arange(-2, 3, dtype=torch.float64)[:, None, None].expand(5, 3, 8)
    groups = torch.arange(8) // 4
    uniform = torch.rand(5, 3, 8, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
    initial = torch.randn(3, 8, dtype=torch.float64) if with_initial else None
    gradient = torch.randn(3, 8, dtype=torch.float64)
    actual = relaxed_scalar_weights(logits, scales, candidates, groups,
                                    uniform=uniform, temperature=0.7,
                                    multiplier=1.8, initial=initial)
    actual_grads = torch.autograd.grad((actual * gradient).sum(), (logits, scales))

    reference_logits = logits.detach().clone().requires_grad_()
    reference_scales = scales.detach().clone().requires_grad_()
    noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
    probabilities = ((reference_logits * 1.8 + noise) / 0.7).softmax(0)
    expected = (probabilities * candidates).sum(0)
    if initial is not None:
        expected = expected + initial
    reference = expected * reference_scales[:, groups]
    reference_grads = torch.autograd.grad((reference * gradient).sum(),
                                          (reference_logits, reference_scales))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-12)
