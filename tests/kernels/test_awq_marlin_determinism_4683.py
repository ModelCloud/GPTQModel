# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ Marlin determinism regression (#4683).

``AwqMarlinLinear`` is the AUTO kernel for AWQ checkpoints on CUDA. The issue
reports that it occasionally returns a different output for an identical input,
sometimes with one 64x64 tile off by ~0.25, while GPTQ ``MarlinLinear`` (the
same extension, no zero points) stays bitwise stable; as a result the wikitext
PPL of one AWQ checkpoint varies from process to process.

Nothing covered that. ``test_awq_machete_marlin`` compares two back-to-back
calls against the Torch AWQ oracle with an ``atol``/``rtol`` tolerance, which
both hides a single bad tile inside the tolerance and never repeats a launch
enough times to hit a 1-in-600 race.

These tests repeat the *launch*: several hundred forwards run back to back with
no host synchronisation between them, the same window the reporter used. The
same input through the same kernel must produce the same bits, so repeated
launches are asserted bitwise. The Torch AWQ comparison stays a tolerance check
because it is a different quantizer, not a determinism oracle.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear, marlin_import_exception
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error


# One launch per forward. The reporter measured 1 mismatch in 600 to 3000
# launches on sm_89, so a few hundred launches per shape is the smallest window
# that is still meaningful without making the suite slow.
REPEATS = 300


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    assert unpacked.shape[1] % pack_factor == 0
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for lane, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (lane * bits)
    return packed


def _mock_awq_tensors(
    *,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    seed: int = 13,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    groups = in_features // group_size
    maxq = (1 << bits) - 1

    float_weight = torch.randn(in_features, out_features, generator=generator, dtype=torch.float32) * 0.2
    weight_groups = float_weight.view(groups, group_size, out_features)

    w_min = weight_groups.amin(dim=1)
    w_max = weight_groups.amax(dim=1)
    scales = ((w_max - w_min).clamp_min(1e-6) / maxq).to(torch.float16)
    zero_points = torch.round((-w_min / scales.to(torch.float32))).clamp_(0, maxq).to(torch.int32)
    quantized = torch.round(
        weight_groups / scales.to(torch.float32).unsqueeze(1) + zero_points.unsqueeze(1)
    ).clamp_(0, maxq).to(torch.int32)

    return (
        _pack_awq_tensor(quantized.view(in_features, out_features), bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
    )


def _build_module(module_cls, *, device, bits, group_size, in_features, out_features,
                  qweight, qzeros, scales, dtype):
    module = module_cls(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
        dtype=dtype,
    ).to(device)

    with torch.no_grad():
        module.qweight.copy_(qweight.to(device))
        module.qzeros.copy_(qzeros.to(device))
        module.scales.copy_(scales.to(dtype=dtype, device=device))

    module.post_init()
    module.eval()
    return module


def _skip_reason(dtype: torch.dtype) -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required for the AWQ Marlin determinism test"
    if marlin_import_exception is not None:
        return f"AWQ Marlin kernel unavailable: {marlin_import_exception}"
    if not marlin_runtime_available(dtype):
        return marlin_runtime_error(dtype)
    return None


def _marlin_and_torch(device, *, in_features, out_features, dtype, seed=13):
    qweight, qzeros, scales = _mock_awq_tensors(
        bits=4,
        group_size=128,
        in_features=in_features,
        out_features=out_features,
        seed=seed,
    )
    common = {
        "device": device,
        "bits": 4,
        "group_size": 128,
        "in_features": in_features,
        "out_features": out_features,
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "dtype": dtype,
    }
    return _build_module(AwqMarlinLinear, **common), _build_module(AwqTorchLinear, **common)


# The Llama-3.2-1B-Instruct AWQ shapes the issue measured: q/k/v/o (K=2048),
# gate/up (N=8192) and down (K=8192), at prefill batch sizes.
_SHAPES = [
    (512, 2048, 2048),
    (512, 512, 2048),
    (512, 8192, 2048),
    (512, 2048, 8192),
    (2048, 2048, 2048),
]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size_m,size_n,size_k", _SHAPES)
def test_repeated_launches_are_bitwise_identical(dtype, size_m, size_n, size_k):
    """Same input, same kernel, hundreds of launches: the bits must not move."""

    reason = _skip_reason(dtype)
    if reason is not None:
        pytest.skip(reason)

    device = torch.device("cuda:0")
    marlin, _torch_module = _marlin_and_torch(
        device, in_features=size_k, out_features=size_n, dtype=dtype
    )
    x = torch.randn((size_m, size_k), device=device, dtype=dtype)

    with torch.inference_mode():
        # No host synchronisation between launches: the launches queue back to
        # back exactly like the reporter's 40-forward loop. Both the mismatch
        # flag and the magnitude accumulate on the device, so one bad launch
        # anywhere in the loop is caught, not just the last one.
        expected = marlin(x).clone()
        mismatched = torch.zeros((), dtype=torch.bool, device=device)
        worst = torch.zeros((), device=device)
        for _ in range(REPEATS):
            delta = (marlin(x) - expected).abs()
            mismatched = mismatched | (delta > 0).any()
            worst = torch.maximum(worst, delta.amax())
    torch.cuda.synchronize()

    assert not mismatched.item(), (
        f"AWQ Marlin is not deterministic for M={size_m} N={size_n} K={size_k} "
        f"{dtype}: max|delta|={worst.item():.6f}"
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_alternating_batch_shapes_keep_their_own_result():
    """The module reuses one lock/reduce workspace across launches (#4683).

    A forward with a different M selects a different split-K slicing, so the
    lock counters a launch starts from must not depend on the previous launch's
    shape. Alternate the shapes on one module and compare each shape against a
    fresh module's first result.
    """

    reason = _skip_reason(torch.float16)
    if reason is not None:
        pytest.skip(reason)

    device = torch.device("cuda:0")
    marlin, _torch_module = _marlin_and_torch(
        device, in_features=2048, out_features=2048, dtype=torch.float16
    )
    shapes = (64, 512)
    inputs = {
        size_m: torch.randn((size_m, 2048), device=device, dtype=torch.float16)
        for size_m in shapes
    }

    with torch.inference_mode():
        warm = {size_m: marlin(inputs[size_m]).clone() for size_m in shapes}
        mismatched = {size_m: torch.zeros((), dtype=torch.bool, device=device) for size_m in shapes}
        for _ in range(REPEATS):
            for size_m in shapes:
                delta = (marlin(inputs[size_m]) - warm[size_m]).abs()
                mismatched[size_m] = mismatched[size_m] | (delta > 0).any()
    torch.cuda.synchronize()

    for size_m in shapes:
        assert not mismatched[size_m].item(), (
            f"M={size_m} changed after a forward with the other shape"
        )

    for size_m in shapes:
        fresh, _ = _marlin_and_torch(
            device, in_features=2048, out_features=2048, dtype=torch.float16
        )
        with torch.inference_mode():
            first = fresh(inputs[size_m]).clone()
        torch.cuda.synchronize()
        assert torch.equal(first, warm[size_m]), (
            f"M={size_m} depends on the launch history of the module's workspace"
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_marlin_matches_the_torch_awq_oracle():
    """The determinism assertions above must not pin a wrong-but-stable value."""

    reason = _skip_reason(torch.float16)
    if reason is not None:
        pytest.skip(reason)

    device = torch.device("cuda:0")
    marlin, torch_awq = _marlin_and_torch(
        device, in_features=2048, out_features=2048, dtype=torch.float16
    )
    x = torch.randn((512, 2048), device=device, dtype=torch.float16)

    with torch.inference_mode():
        actual = marlin(x)
        expected = torch_awq(x)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.02)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_each_module_owns_its_lock_workspace():
    """Two modules must not share the split-K lock array (the #4665 failure mode)."""

    reason = _skip_reason(torch.float16)
    if reason is not None:
        pytest.skip(reason)

    device = torch.device("cuda:0")
    first, _ = _marlin_and_torch(device, in_features=2048, out_features=2048, dtype=torch.float16)
    second, _ = _marlin_and_torch(
        device, in_features=2048, out_features=2048, dtype=torch.float16, seed=29
    )

    assert first.workspace.numel() >= torch.cuda.get_device_properties(device).multi_processor_count
    assert first.workspace.data_ptr() != second.workspace.data_ptr()
    assert not first.workspace.any()
