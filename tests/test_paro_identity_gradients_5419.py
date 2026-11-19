# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Identity Paro rotations must keep their parameter gradients (issue #5419).

``_rotate_inputs`` short-circuits an identity rotation to a no-op. That is
correct in value, but the identity still has nonzero derivatives with respect
to ``theta`` and ``channel_scales``: at ``theta == 0`` and unit scales the
angle's derivative is the paired activation and the scale's derivative is the
rotated value. A layer whose rotation parameters are trainable therefore lost
those gradients, and the first backward raised because the output had no grad
fn at all.

These tests pin the fix: the shortcut runs only when no enabled gradient path
needs the rotation operands; the identity gradient matches the PyTorch
reference, including the analytic 2 / 1 values from the issue, while the
inference shortcut stays bit-identical and call-free.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import paro as paro_module
from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.paro import (
    apply_paro_rotation_reference,
    build_identity_rotation_buffers,
)


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unpacked integer weights into the AWQ bit layout the kernels read."""
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


IN_FEATURES = 16
OUT_FEATURES = 16
GROUP_SIZE = 16
KROT = 1


def _make_identity_module(requires_grad=False):
    """A 16x16 int4 Paro layer whose only nonzero weight is W[0, 0] == 1."""
    module = ParoLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=True,
        krot=KROT,
    )
    int_weight = torch.full((IN_FEATURES, OUT_FEATURES), 2 ** (4 - 1), dtype=torch.int32)
    int_weight[0, 0] = 2 ** (4 - 1) + 1
    zero_points = torch.full((1, OUT_FEATURES), 2 ** (4 - 1), dtype=torch.int32)
    scales = torch.zeros((1, OUT_FEATURES), dtype=torch.float16)
    scales[0, 0] = 1.0
    module.qweight.copy_(_pack_awq_tensor(int_weight, 4))
    module.qzeros.copy_(_pack_awq_tensor(zero_points, 4))
    module.scales.copy_(scales)
    module.bias.zero_()
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=IN_FEATURES,
        group_size=GROUP_SIZE,
        krot=KROT,
        dtype=torch.float32,
    )
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    assert module._rotation_identity is True
    if requires_grad:
        module.theta.requires_grad_(True)
        module.channel_scales.requires_grad_(True)
    return module


def test_identity_rotation_keeps_the_analytic_parameter_gradients():
    """``scale0 * (x0 * cos + x1 * sin)`` has angle grad 2 and scale grad 1."""

    module = _make_identity_module(requires_grad=True)
    module.train()

    x = torch.zeros(1, IN_FEATURES)
    x[0, 0] = 1.0
    x[0, 1] = 2.0

    y = module(x)
    assert y.requires_grad, "the identity output must stay in the autograd graph"
    assert y[0, 0].item() == pytest.approx(1.0)

    y[0, 0].backward()

    assert module.theta.grad is not None
    assert module.channel_scales.grad is not None
    assert module.theta.grad[0, 0].item() == pytest.approx(2.0, abs=1e-6)
    scale_grad = module.channel_scales.grad.reshape(-1)
    assert scale_grad[0].item() == pytest.approx(1.0, abs=1e-6)
    # Only the first pair and the first scale feed the only nonzero weight.
    assert module.theta.grad[0, 1:].abs().sum().item() == 0.0
    assert scale_grad[1:].abs().sum().item() == 0.0


def test_identity_rotation_gradients_match_the_reference():
    """The full identity gradient must match the PyTorch reference forward."""

    module = _make_identity_module(requires_grad=True)
    module.train()

    torch.manual_seed(0)
    x = torch.randn(3, 5, IN_FEATURES, requires_grad=True)
    y = module(x)
    grad_out = torch.randn(y.shape, dtype=torch.float32)
    (y.float() * grad_out).sum().backward()

    assert x.grad is not None
    assert module.theta.grad is not None
    assert module.channel_scales.grad is not None

    x_ref = x.detach().clone().requires_grad_(True)
    theta_ref = module.theta.detach().clone().requires_grad_(True)
    scales_ref = module.channel_scales.detach().clone().requires_grad_(True)
    rotated = apply_paro_rotation_reference(
        x_ref.reshape(-1, IN_FEATURES),
        module.pairs,
        theta_ref,
        scales=scales_ref,
        group_size=GROUP_SIZE,
    )
    weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales.float(),
        bits=4,
        group_size=GROUP_SIZE,
    )
    y_ref = (rotated.float() @ weight.float()).reshape(3, 5, OUT_FEATURES)
    (y_ref * grad_out).sum().backward()

    for name, actual, expected in (
        ("x", x.grad, x_ref.grad),
        ("theta", module.theta.grad, theta_ref.grad),
        ("channel_scales", module.channel_scales.grad, scales_ref.grad),
    ):
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6, msg=name)


def test_inference_keeps_the_identity_shortcut(monkeypatch):
    """The no-op shortcut stays bit-identical when no gradient needs it."""

    module = _make_identity_module(requires_grad=False)
    calls = []
    for name in ("apply_paro_rotation", "apply_paro_rotation_autograd",
                 "apply_paro_rotation_vectorized"):
        original = getattr(paro_module, name)

        def recorder(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(paro_module, name, recorder)

    x = torch.randn(3, 5, IN_FEATURES)

    # Inference: no rotation call at all, and the input object passes through.
    module.eval()
    with torch.no_grad():
        out = module(x)
    assert calls == []
    assert out.shape == (3, 5, OUT_FEATURES)

    # Training with frozen rotation parameters and an input gradient: still no
    # rotation call, because only the activations need a gradient.
    module.train()
    calls.clear()
    module(x.clone().requires_grad_(True)).sum().backward()
    assert calls == []


def test_trainable_identity_parameters_leave_the_shortcut(monkeypatch):
    """Trainable identity parameters must reach the autograd rotation."""

    module = _make_identity_module(requires_grad=True)
    module.train()
    calls = []
    for name in ("apply_paro_rotation", "apply_paro_rotation_autograd"):
        original = getattr(paro_module, name)

        def recorder(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(paro_module, name, recorder)

    module(torch.randn(3, 5, IN_FEATURES)).sum().backward()
    assert calls == ["apply_paro_rotation_autograd"]
