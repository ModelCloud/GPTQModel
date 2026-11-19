# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Training gradients for the Paro kernels (issue #4958).

``ParoLinear`` (and therefore ``ParoTritonLinear``) declares
``SUPPORTS_TRAINING = True`` and is what ``AUTO``/``AUTO_TRAINABLE`` load, but
its forward cut the autograd graph twice:

* ``_rotate_inputs`` always called ``apply_paro_rotation``, which dispatches to
  the fused ``gptqmodel_paro::rotate`` op; that op has no autograd kernel, so
  the input and rotation-parameter gradients were silently dropped;
* the native AWQ GEMM (``gptqmodel_awq::gemm_forward*``) and the Triton plans
  are raw kernel launches with no autograd either, so even a differentiable
  rotation would have lost ``dL/dx`` at the matmul.

These tests pin the contract: a train-mode forward produces a finite
``x.grad``, ``theta.grad`` and ``channel_scales.grad`` that match a pure
PyTorch reference, a parameter upstream of the layer receives its share of the
gradient, and inference keeps the fast paths unchanged.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import paro as paro_module
from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.nn_modules.qlinear.paro_triton import ParoTritonLinear
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.paro import (
    _load_rotation_extension,
    apply_paro_rotation,
    apply_paro_rotation_reference,
    build_identity_rotation_buffers,
)


BITS = 4
IN_FEATURES = 128
OUT_FEATURES = 128
GROUP_SIZE = 128
KROT = 8
ROWS = 3
SEQ = 7

# The layer rounds the FP32 product once to the activation dtype, so the
# comparison is bounded by that single rounding plus the rotation arithmetic.
TOLERANCES = {
    torch.float32: 1e-6,
    torch.float16: 5e-3,
    torch.bfloat16: 2e-2,
}


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


def _make_paro_module(cls, device: str, dtype: torch.dtype, seed: int = 0):
    """Build a Paro layer with a non-identity rotation on ``device``."""
    torch.manual_seed(seed)
    groups = IN_FEATURES // GROUP_SIZE
    int_weight = torch.randint(
        0, 2**BITS, size=(IN_FEATURES, OUT_FEATURES), dtype=torch.int32
    )
    zero_points = torch.full((groups, OUT_FEATURES), 2 ** (BITS - 1), dtype=torch.int32)
    scales = (torch.rand(groups, OUT_FEATURES, dtype=torch.float16) * 0.75) + 0.25
    bias = torch.randn(OUT_FEATURES, dtype=torch.float16)

    module = cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=True,
        krot=KROT,
    ).to(device=device)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=IN_FEATURES,
        group_size=GROUP_SIZE,
        krot=KROT,
        device=device,
        dtype=dtype,
    )
    # A non-identity rotation is what makes the fused kernel (and its missing
    # backward) reachable; identity short-circuits `_rotate_inputs`.
    theta.uniform_(-0.25, 0.25)
    channel_scales.uniform_(0.7, 1.3)

    module.qweight.copy_(_pack_awq_tensor(int_weight, BITS).to(device))
    module.qzeros.copy_(_pack_awq_tensor(zero_points, BITS).to(device))
    module.scales.copy_(scales.to(device))
    module.bias.copy_(bias.to(device))
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    assert module._rotation_identity is False
    return module


def _reference_forward(x, *, pairs, theta, channel_scales, qweight, qzeros, scales, bias):
    """Dense FP32 reference for ``y = rotate(x) @ dequant(qweight) + bias``."""
    rotated = apply_paro_rotation_reference(
        x.reshape(-1, x.shape[-1]),
        pairs,
        theta,
        scales=channel_scales,
        group_size=GROUP_SIZE,
    )
    weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales.float(),
        bits=BITS,
        group_size=GROUP_SIZE,
    )
    out = torch.matmul(rotated.float(), weight.float())
    if bias is not None:
        out = out + bias.float()
    return out.reshape(*x.shape[:-1], OUT_FEATURES).to(x.dtype)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.float() - expected.float()).abs().max() / expected.float().abs().max()
    ).item()


def _assert_training_gradients_match_reference(cls, device: str, dtype: torch.dtype):
    module = _make_paro_module(cls, device, dtype)
    module.train()
    module.theta.requires_grad_(True)
    module.channel_scales.requires_grad_(True)

    x = torch.randn(
        ROWS, SEQ, IN_FEATURES, device=device, dtype=dtype, requires_grad=True
    )
    y = module(x)
    grad_out = torch.randn(y.shape, device=device, dtype=torch.float32)
    (y.float() * grad_out).sum().backward()

    # Before the fix the forward returned a tensor whose gradient was silently
    # dropped, so every one of these was None.
    assert x.grad is not None, "Paro training lost the input gradient"
    assert torch.isfinite(x.grad.float()).all()
    assert module.theta.grad is not None, "Paro training lost the rotation gradient"
    assert module.channel_scales.grad is not None

    x_ref = x.detach().clone().requires_grad_(True)
    theta_ref = module.theta.detach().clone().requires_grad_(True)
    channel_scales_ref = module.channel_scales.detach().clone().requires_grad_(True)
    y_ref = _reference_forward(
        x_ref,
        pairs=module.pairs,
        theta=theta_ref,
        channel_scales=channel_scales_ref,
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bias=module.bias,
    )
    (y_ref.float() * grad_out).sum().backward()

    tolerance = TOLERANCES[dtype]
    assert _relative_error(y, y_ref) < tolerance
    assert _relative_error(x.grad, x_ref.grad) < tolerance
    assert _relative_error(module.theta.grad, theta_ref.grad) < tolerance
    assert _relative_error(module.channel_scales.grad, channel_scales_ref.grad) < tolerance


def test_paro_linear_cpu_training_gradients_match_reference():
    _assert_training_gradients_match_reference(ParoLinear, "cpu", torch.float32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paro_linear_cuda_training_gradients_match_reference(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the fused Paro rotation")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("this GPU does not support BF16")
    _assert_training_gradients_match_reference(ParoLinear, "cuda", dtype)


def test_paro_triton_linear_cuda_training_gradients_match_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Paro Triton plans")
    ok, _ = ParoTritonLinear.validate_once()
    if not ok:
        pytest.skip("the Paro Triton kernel is not available on this device")
    _assert_training_gradients_match_reference(ParoTritonLinear, "cuda", torch.float16)


@pytest.mark.parametrize(
    "device,dtype", [("cpu", torch.float32), ("cuda", torch.float16)]
)
def test_upstream_parameter_receives_paro_gradient(device, dtype):
    """A parameter *before* a Paro layer must get its share of the gradient.

    This is the user-visible symptom in the issue: the loss still
    backpropagates, but everything upstream of a Paro projection trains on a
    gradient that misses the residual-stream contribution of the projection.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the fused Paro rotation")
    module = _make_paro_module(ParoLinear, device, dtype)
    module.train()
    projection = torch.nn.Linear(IN_FEATURES, IN_FEATURES, bias=False, dtype=dtype).to(
        device=device
    )
    with torch.no_grad():
        projection.weight.copy_(torch.eye(IN_FEATURES, dtype=dtype))

    x = torch.randn(ROWS, SEQ, IN_FEATURES, device=device, dtype=dtype, requires_grad=True)
    y = module(projection(x))
    # A unit gradient keeps the FP16 case away from the dtype's range: the
    # point here is that the upstream parameter is in the graph at all.
    y.sum().backward()

    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert projection.weight.grad.abs().sum().item() > 0


def test_rotation_dispatch_keeps_the_inference_path(monkeypatch):
    """The autograd wrapper is used only when a gradient is actually needed."""
    module = _make_paro_module(ParoLinear, "cpu", torch.float32)
    calls = []
    for name in ("apply_paro_rotation", "apply_paro_rotation_autograd"):
        original = getattr(paro_module, name)

        def recorder(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(paro_module, name, recorder)

    x = torch.randn(ROWS, SEQ, IN_FEATURES, dtype=torch.float32)

    # Eval, no grad: the vectorized CPU reference (unchanged).
    module.eval()
    with torch.no_grad():
        module(x)
    assert calls == []

    # Train, no grad: the plain rotation (unchanged).
    calls.clear()
    module.train()
    with torch.no_grad():
        module(x)
    assert calls == ["apply_paro_rotation"]

    # Train with an input gradient: the autograd-capable wrapper.
    calls.clear()
    module(x.clone().requires_grad_(True)).sum().backward()
    assert calls == ["apply_paro_rotation_autograd"]

    # Grad enabled but nothing requires a gradient: still the plain rotation,
    # so inference numerics cannot change.
    calls.clear()
    module(x)
    assert calls == ["apply_paro_rotation"]


def test_autograd_rotation_forward_is_bitwise_identical_to_the_fused_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the fused Paro rotation")
    if not _load_rotation_extension():
        pytest.skip("the fused Paro rotation extension is not available")

    module = _make_paro_module(ParoLinear, "cuda", torch.float16)
    x = torch.randn(ROWS, SEQ, IN_FEATURES, device="cuda", dtype=torch.float16)
    x_flat = x.reshape(-1, IN_FEATURES).contiguous()

    module.eval()
    with torch.no_grad():
        fused = apply_paro_rotation(
            x_flat,
            module.pairs,
            module.theta,
            scales=module.channel_scales,
            group_size=GROUP_SIZE,
        )

    module.train()
    module.theta.requires_grad_(True)
    training = module._rotate_inputs(x_flat)

    assert training.requires_grad
    assert torch.equal(training.detach(), fused)


def test_cuda_awq_gemm_is_reserved_for_inference():
    """The native AWQ GEMM has no autograd; inference must keep it."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the native AWQ GEMM")

    module = _make_paro_module(ParoLinear, "cuda", torch.float16)
    x_flat = torch.randn(ROWS * SEQ, IN_FEATURES, device="cuda", dtype=torch.float16)

    module.eval()
    with torch.no_grad():
        rotated = module._rotate_inputs(x_flat)
        assert module._forward_cuda_awq_kernel(rotated) is not None

    module.train()
    with torch.no_grad():
        rotated = module._rotate_inputs(x_flat)
    assert module._forward_cuda_awq_kernel(rotated) is not None

    needs_grad = rotated.detach().clone().requires_grad_(True)
    assert module._forward_cuda_awq_kernel(needs_grad) is None
