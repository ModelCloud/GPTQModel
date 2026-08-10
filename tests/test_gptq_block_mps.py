# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import gptqmodel.utils.gptq_block_mps as block_module

MPS_AVAILABLE = block_module.gptq_block_mps_supported()
mps_only = pytest.mark.skipif(
    not MPS_AVAILABLE, reason="requires torch.mps.compile_shader on macOS"
)


def _serial_block(weights, hessian_inverse, scale, zero, maxq, group_size, groupwise):
    quantized = torch.empty_like(weights)
    errors = torch.empty_like(weights)
    for column in range(weights.shape[1]):
        group = column // group_size
        column_scale = scale[:, group]
        column_zero = zero[:, group]
        weight = weights[:, column]
        if groupwise:
            q = column_scale * torch.clamp(
                torch.round(weight / column_scale), -maxq, maxq
            )
        else:
            q = column_scale * (
                torch.clamp(torch.round(weight / column_scale) + column_zero, 0, maxq)
                - column_zero
            )
        quantized[:, column] = q
        error = (weight - q) / hessian_inverse[column, column]
        errors[:, column] = error
        weights[:, column:] = torch.addr(
            weights[:, column:],
            error,
            hessian_inverse[column, column:],
            alpha=-1,
        )
    return quantized, errors


def _case(seed, rows, columns, group_size, groupwise):
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn(rows, columns, generator=generator)
    hessian_inverse = torch.triu(
        torch.randn(columns, columns, generator=generator) * 0.03
    )
    hessian_inverse.diagonal().copy_(torch.rand(columns, generator=generator) + 0.5)
    scale = torch.rand(rows, columns // group_size, generator=generator) * 0.2 + 0.01
    if groupwise:
        zero = torch.zeros_like(scale)
        maxq = 7
    else:
        zero = torch.randint(0, 16, scale.shape, generator=generator).float()
        maxq = 15
    return tuple(
        tensor.to("mps") for tensor in (weights, hessian_inverse, scale, zero)
    ), maxq


def _reference_params(weights, group_size, maxq, *, symmetric, groupwise):
    grouped = weights.reshape(
        weights.shape[0], weights.shape[1] // group_size, group_size
    )
    zero_base = torch.zeros(grouped.shape[:2], device=weights.device)
    xmin_raw, xmax_raw = torch.aminmax(grouped, dim=-1)
    xmin = torch.minimum(xmin_raw, zero_base)
    xmax = torch.maximum(xmax_raw, zero_base)
    if symmetric:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        xmin = torch.where(xmin < 0, -xmax, xmin)
    zero_range = (xmin == 0) & (xmax == 0)
    xmin = torch.where(zero_range, -torch.ones_like(xmin), xmin)
    xmax = torch.where(zero_range, torch.ones_like(xmax), xmax)
    if groupwise:
        return xmax / maxq, torch.zeros_like(xmax)
    scale = (xmax - xmin) / maxq
    if symmetric:
        return scale, torch.full_like(scale, (maxq + 1) / 2)
    return scale, torch.round(-xmin / scale)


def _reference_scale_search(
    weights,
    group_size,
    maxq,
    *,
    symmetric,
    groupwise,
    mode,
    importance=None,
    candidate_count=80,
    grid=100,
    mse=2.0,
):
    grouped = weights.reshape(
        weights.shape[0], weights.shape[1] // group_size, group_size
    )
    zero_base = torch.zeros(grouped.shape[:2], device=weights.device)
    xmin_raw, xmax_raw = torch.aminmax(grouped, dim=-1)
    xmin = torch.minimum(xmin_raw, zero_base)
    xmax = torch.maximum(xmax_raw, zero_base)
    if symmetric:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        xmin = torch.where(xmin < 0, -xmax, xmin)
    zero_range = (xmin == 0) & (xmax == 0)
    xmin = torch.where(zero_range, -torch.ones_like(xmin), xmin)
    xmax = torch.where(zero_range, torch.ones_like(xmax), xmax)
    shrink = (
        1
        - torch.arange(candidate_count, device=weights.device, dtype=torch.float32)
        / grid
    )
    xmin_all = shrink[:, None, None] * xmin[None]
    xmax_all = shrink[:, None, None] * xmax[None]
    if groupwise:
        scale = xmax_all / maxq
        zero = torch.zeros_like(scale)
    else:
        scale = (xmax_all - xmin_all) / maxq
        if symmetric:
            zero = torch.full_like(scale, (maxq + 1) / 2)
        else:
            zero = torch.round(-xmin_all / scale)
    if groupwise:
        reconstructed = scale[..., None] * torch.clamp(
            torch.round(grouped[None] / scale[..., None]), -maxq, maxq
        )
    else:
        reconstructed = scale[..., None] * (
            torch.clamp(
                torch.round(grouped[None] / scale[..., None]) + zero[..., None],
                0,
                maxq,
            )
            - zero[..., None]
        )
    error = torch.abs(reconstructed - grouped[None])
    if mode == "activation":
        normalized = torch.nan_to_num(
            importance, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0)
        mean = normalized.mean(dim=-1, keepdim=True)
        normalized = torch.where(
            torch.isfinite(mean) & (mean > 0),
            normalized / mean,
            torch.ones_like(normalized),
        )
        loss = (error.square() * normalized[None, None]).sum(dim=-1)
    else:
        loss = error.pow(mse).sum(dim=-1)
    best = loss.argmin(dim=0)
    rows = torch.arange(weights.shape[0], device=weights.device)[:, None]
    groups = torch.arange(grouped.shape[1], device=weights.device)[None, :]
    return scale[best, rows, groups], zero[best, rows, groups]


@mps_only
@pytest.mark.mps
@pytest.mark.parametrize(
    ("rows", "columns", "group_size", "groupwise"),
    [
        (1, 32, 32, False),
        (13, 64, 16, False),
        (65, 128, 32, True),
        (257, 128, 128, False),
        (4096, 128, 32, False),
    ],
)
@pytest.mark.parametrize("seed", [7, 811, 2029])
def test_mps_block_matches_sequential_mps_reference(
    rows, columns, group_size, groupwise, seed
):
    operands, maxq = _case(seed, rows, columns, group_size, groupwise)
    reference_weights = operands[0].clone()
    reference_quantized, reference_errors = _serial_block(
        reference_weights,
        operands[1],
        operands[2],
        operands[3],
        maxq,
        group_size,
        groupwise,
    )
    actual_weights = operands[0].clone()
    actual_quantized, actual_errors = block_module.gptq_block_mps(
        actual_weights,
        *operands[1:],
        maxq,
        group_size,
        groupwise=groupwise,
    )
    torch.mps.synchronize()

    assert torch.isfinite(actual_quantized).all()
    assert torch.isfinite(actual_errors).all()
    torch.testing.assert_close(actual_quantized, reference_quantized, atol=0, rtol=0)
    torch.testing.assert_close(actual_errors, reference_errors, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_weights, reference_weights, atol=2e-5, rtol=2e-5)


@mps_only
@pytest.mark.mps
def test_mps_block_matches_rounding_ties_saturation_zeros_and_extreme_scales():
    columns = 32
    group_size = 8
    boundaries = torch.tensor(
        [-100.0, -8.5, -7.5, -1.5, -0.5, 0.0, 0.5, 1.5, 7.5, 8.5, 100.0]
    )
    weights = boundaries.repeat((3 * columns // boundaries.numel()) + 1)[
        : 3 * columns
    ].reshape(3, columns)
    weights[1].zero_()
    weights[2].mul_(1e-6)
    hessian_inverse = torch.eye(columns)
    scale = torch.tensor(
        [[1.0, 1e-3, 1e3, 0.25], [1.0, 1.0, 1.0, 1.0], [1e-6, 1e-6, 1e-6, 1e-6]]
    )
    zero = torch.tensor(
        [[8.0, 8.0, 8.0, 8.0], [0.0, 4.0, 8.0, 15.0], [8.0, 8.0, 8.0, 8.0]]
    )
    operands = tuple(
        tensor.to("mps") for tensor in (weights, hessian_inverse, scale, zero)
    )

    reference_weights = operands[0].clone()
    reference = _serial_block(reference_weights, *operands[1:], 15, group_size, False)
    actual_weights = operands[0].clone()
    actual = block_module.gptq_block_mps(actual_weights, *operands[1:], 15, group_size)
    torch.mps.synchronize()

    torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], reference[1], atol=0, rtol=0)
    torch.testing.assert_close(actual_weights, reference_weights, atol=0, rtol=0)
    assert torch.count_nonzero(actual[0][1]) == 0


@mps_only
@pytest.mark.mps
@pytest.mark.parametrize(
    ("rows", "columns", "group_size", "symmetric", "groupwise"),
    [
        (1, 32, 8, False, False),
        (17, 64, 32, True, False),
        (257, 128, 128, False, False),
        (65, 128, 32, True, True),
    ],
)
@pytest.mark.parametrize("seed", [17, 503, 4001])
def test_mps_fused_params_match_eager_scale_and_correction(
    rows, columns, group_size, symmetric, groupwise, seed
):
    operands, _ = _case(seed, rows, columns, group_size, groupwise)
    maxq = 7 if groupwise else 15
    expected_scale, expected_zero = _reference_params(
        operands[0],
        group_size,
        maxq,
        symmetric=symmetric,
        groupwise=groupwise,
    )
    reference_weights = operands[0].clone()
    reference = _serial_block(
        reference_weights,
        operands[1],
        expected_scale,
        expected_zero,
        maxq,
        group_size,
        groupwise,
    )
    actual_weights = operands[0].clone()
    actual_scale = torch.empty_like(expected_scale)
    actual_zero = torch.empty_like(expected_zero)
    actual = block_module.gptq_block_mps(
        actual_weights,
        operands[1],
        actual_scale,
        actual_zero,
        maxq,
        group_size,
        groupwise=groupwise,
        find_params=True,
        symmetric=symmetric,
    )
    torch.mps.synchronize()

    torch.testing.assert_close(actual_scale, expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(actual_zero, expected_zero, atol=0, rtol=0)
    torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], reference[1], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_weights, reference_weights, atol=2e-5, rtol=2e-5)


@mps_only
@pytest.mark.mps
@pytest.mark.parametrize("mode", ["mse", "activation"])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("groupwise", [False, True])
@pytest.mark.parametrize("seed", [19, 887, 4099])
def test_mps_fused_scale_search_matches_eager_candidates(
    mode, symmetric, groupwise, seed
):
    operands, _ = _case(seed, 37, 128, 32, groupwise)
    maxq = 7 if groupwise else 15
    importance = torch.rand(4, 32, device="mps")
    if seed == 19:
        importance[0].zero_()
        importance[1, 0] = float("nan")
        importance[1, 1] = float("inf")
        importance[1, 2] = -1.0
    expected_scale, expected_zero = _reference_scale_search(
        operands[0],
        32,
        maxq,
        symmetric=symmetric,
        groupwise=groupwise,
        mode=mode,
        importance=importance,
    )
    reference_weights = operands[0].clone()
    reference = _serial_block(
        reference_weights,
        operands[1],
        expected_scale,
        expected_zero,
        maxq,
        32,
        groupwise,
    )
    actual_weights = operands[0].clone()
    actual_scale = torch.empty_like(expected_scale)
    actual_zero = torch.empty_like(expected_zero)
    actual = block_module.gptq_block_mps(
        actual_weights,
        operands[1],
        actual_scale,
        actual_zero,
        maxq,
        32,
        groupwise=groupwise,
        find_params=True,
        symmetric=symmetric,
        scale_search=mode,
        importance=importance if mode == "activation" else None,
        candidate_count=80,
    )
    torch.mps.synchronize()

    torch.testing.assert_close(actual_scale, expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(actual_zero, expected_zero, atol=0, rtol=0)
    torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], reference[1], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_weights, reference_weights, atol=2e-5, rtol=2e-5)


@mps_only
@pytest.mark.mps
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_mps_fused_scale_search_preserves_eager_nonfinite_params(nonfinite):
    operands, _ = _case(711, 2, 32, 16, False)
    weights = operands[0].clone()
    weights[0, 3] = nonfinite
    expected_scale, expected_zero = _reference_scale_search(
        weights,
        16,
        15,
        symmetric=False,
        groupwise=False,
        mode="mse",
        candidate_count=80,
    )
    actual_scale = torch.empty_like(expected_scale)
    actual_zero = torch.empty_like(expected_zero)
    block_module.gptq_block_mps(
        weights,
        operands[1],
        actual_scale,
        actual_zero,
        15,
        16,
        find_params=True,
        scale_search="mse",
        candidate_count=80,
    )
    torch.mps.synchronize()

    torch.testing.assert_close(
        actual_scale, expected_scale, atol=0, rtol=0, equal_nan=True
    )
    torch.testing.assert_close(
        actual_zero, expected_zero, atol=0, rtol=0, equal_nan=True
    )


@mps_only
@pytest.mark.mps
def test_mps_block_is_deterministic_and_reuses_output_buffers():
    operands, maxq = _case(91, 33, 64, 16, False)
    expected = None
    for _ in range(10):
        weights = operands[0].clone()
        out = (torch.empty_like(weights), torch.empty_like(weights))
        result = block_module.gptq_block_mps(weights, *operands[1:], maxq, 16, out=out)
        assert result[0] is out[0]
        assert result[1] is out[1]
        torch.mps.synchronize()
        snapshot = tuple(tensor.cpu() for tensor in (*result, weights))
        if expected is None:
            expected = snapshot
        else:
            for actual, first in zip(snapshot, expected):
                torch.testing.assert_close(actual, first, atol=0, rtol=0)


def test_mps_shader_cache_initialization_is_bounded_and_thread_safe(monkeypatch):
    sentinel = object()
    calls = 0
    calls_lock = threading.Lock()
    start = threading.Barrier(8)

    def compile_shader(_source):
        nonlocal calls
        with calls_lock:
            calls += 1
        return sentinel

    monkeypatch.setattr(block_module, "_MPS_SHADER_LIBRARY", None)
    monkeypatch.setattr(block_module, "_MPS_SHADER_ERROR", None)
    monkeypatch.setattr(block_module, "gptq_block_mps_supported", lambda: True)
    monkeypatch.setattr(torch.mps, "compile_shader", compile_shader)

    def resolve_library():
        start.wait()
        return block_module._shader_library()

    with ThreadPoolExecutor(max_workers=8) as executor:
        libraries = list(executor.map(lambda _: resolve_library(), range(8)))

    assert calls == 1
    assert all(library is sentinel for library in libraries)


def test_mps_shader_cache_fails_closed_when_runtime_is_unsupported(monkeypatch):
    monkeypatch.setattr(block_module, "_MPS_SHADER_LIBRARY", None)
    monkeypatch.setattr(block_module, "_MPS_SHADER_ERROR", None)
    monkeypatch.setattr(block_module, "gptq_block_mps_supported", lambda: False)
    with pytest.raises(RuntimeError, match="requires macOS"):
        block_module._shader_library()


def test_mps_shader_compile_failure_is_cached_without_traceback_retention(monkeypatch):
    calls = 0

    def compile_shader(_source):
        nonlocal calls
        calls += 1
        raise RuntimeError("invalid shader")

    monkeypatch.setattr(block_module, "_MPS_SHADER_LIBRARY", None)
    monkeypatch.setattr(block_module, "_MPS_SHADER_ERROR", None)
    monkeypatch.setattr(block_module, "gptq_block_mps_supported", lambda: True)
    monkeypatch.setattr(torch.mps, "compile_shader", compile_shader)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="Metal GPTQ shader compilation failed"):
            block_module._shader_library()
    assert calls == 1
    assert isinstance(block_module._MPS_SHADER_ERROR, str)


@pytest.mark.parametrize(
    ("argument", "value", "exception", "message"),
    [
        ("maxq", True, TypeError, "maxq must be an integer"),
        ("group_size", 2.5, TypeError, "group_size must be an integer"),
        ("groupwise", 1, TypeError, "groupwise must be a bool"),
        ("find_params", 1, TypeError, "find_params must be a bool"),
        ("symmetric", 1, TypeError, "symmetric must be a bool"),
        ("maxq", 0, ValueError, "maxq must be positive"),
        ("maxq", 256, ValueError, "maxq <= 255"),
        ("group_size", 0, ValueError, "group_size must be positive"),
        ("group_size", 3, ValueError, "must divide count"),
    ],
)
def test_mps_block_rejects_invalid_scalar_arguments(
    argument, value, exception, message
):
    weights = torch.empty(2, 4)
    hessian_inverse = torch.eye(4)
    scale = torch.ones(2, 2)
    zero = torch.zeros(2, 2)
    kwargs = {
        "maxq": 15,
        "group_size": 2,
        "groupwise": False,
        "find_params": False,
        "symmetric": False,
        argument: value,
    }
    with pytest.raises(exception, match=message):
        block_module.gptq_block_mps(weights, hessian_inverse, scale, zero, **kwargs)


def test_mps_block_rejects_invalid_scale_search_arguments():
    operands = (torch.empty(2, 4), torch.eye(4), torch.ones(2, 2), torch.zeros(2, 2))
    invalid = [
        ({"scale_search": "other"}, ValueError, "scale_search must be one of"),
        ({"scale_search": "mse"}, ValueError, "requires find_params=True"),
        (
            {"find_params": True, "scale_search": "mse"},
            ValueError,
            "candidate_count must be positive",
        ),
        (
            {
                "find_params": True,
                "scale_search": "mse",
                "candidate_count": 1,
                "grid": 0,
            },
            ValueError,
            "grid must be positive",
        ),
        (
            {
                "find_params": True,
                "scale_search": "mse",
                "candidate_count": 1,
                "mse": 0,
            },
            ValueError,
            "mse must be finite and positive",
        ),
        ({"candidate_count": True}, TypeError, "candidate_count must be an integer"),
        ({"grid": 1.5}, TypeError, "grid must be an integer"),
        ({"mse": object()}, TypeError, "mse must be a number"),
    ]
    for kwargs, exception, message in invalid:
        with pytest.raises(exception, match=message):
            block_module.gptq_block_mps(*operands, 15, 2, **kwargs)


@mps_only
@pytest.mark.mps
def test_mps_block_rejects_invalid_activation_importance():
    operands, maxq = _case(73, 8, 32, 16, False)
    base = {"find_params": True, "scale_search": "activation", "candidate_count": 1}
    invalid = [
        (None, ValueError, "requires importance"),
        (torch.ones(2, 8, device="mps"), ValueError, "must have shape"),
        (
            torch.ones(2, 16, device="mps", dtype=torch.float16),
            TypeError,
            "dtype float32",
        ),
        (torch.ones(2, 16), ValueError, "share the weights device"),
        (torch.ones(16, 2, device="mps").t(), ValueError, "must be contiguous"),
    ]
    for importance, exception, message in invalid:
        with pytest.raises(exception, match=message):
            block_module.gptq_block_mps(
                *operands, maxq, 16, importance=importance, **base
            )


def test_mps_block_rejects_non_mps_and_invalid_shapes():
    with pytest.raises(ValueError, match="two-dimensional"):
        block_module.gptq_block_mps(
            torch.empty(4), torch.eye(4), torch.ones(1, 1), torch.zeros(1, 1), 15, 4
        )

    weights = torch.empty(2, 4)
    with pytest.raises(ValueError, match="must be MPS tensors"):
        block_module.gptq_block_mps(
            weights, torch.eye(4), torch.ones(2, 2), torch.zeros(2, 2), 15, 2
        )

    overflowing = torch.empty((2**26, 128), device="meta")
    with pytest.raises(ValueError, match=r"rows \* count <="):
        block_module.gptq_block_mps(
            overflowing, torch.empty(0), torch.empty(0), torch.empty(0), 15, 128
        )


@mps_only
@pytest.mark.mps
def test_mps_block_rejects_aliasing_and_noncontiguous_operands():
    operands, maxq = _case(18, 8, 32, 16, False)
    noncontiguous_weights = torch.empty(8, 64, device="mps")[:, ::2]
    with pytest.raises(ValueError, match="must be contiguous"):
        block_module.gptq_block_mps(noncontiguous_weights, *operands[1:], maxq, 16)

    weights = operands[0].clone()
    with pytest.raises(ValueError, match="must not alias"):
        block_module.gptq_block_mps(
            weights,
            *operands[1:],
            maxq,
            16,
            out=(weights, torch.empty_like(weights)),
        )

    output = torch.empty_like(weights)
    with pytest.raises(ValueError, match="must not alias"):
        block_module.gptq_block_mps(
            weights, *operands[1:], maxq, 16, out=(output, output)
        )

    shared_params = torch.empty_like(operands[2])
    with pytest.raises(ValueError, match="scale/zero tensors must not alias"):
        block_module.gptq_block_mps(
            weights,
            operands[1],
            shared_params,
            shared_params,
            maxq,
            16,
            find_params=True,
        )

    hessian_scale = (
        operands[1].reshape(-1)[: operands[2].numel()].reshape_as(operands[2])
    )
    with pytest.raises(ValueError, match="must not alias weights/Hinv/importance"):
        block_module.gptq_block_mps(
            weights,
            operands[1],
            hessian_scale,
            operands[3],
            maxq,
            16,
            find_params=True,
        )

    aliased_importance = weights.reshape(-1)[:32].reshape(2, 16)
    with pytest.raises(ValueError, match="importance must not alias mutable weights"):
        block_module.gptq_block_mps(
            weights,
            operands[1],
            operands[2],
            operands[3],
            maxq,
            16,
            find_params=True,
            scale_search="activation",
            importance=aliased_importance,
            candidate_count=1,
        )


@mps_only
@pytest.mark.mps
def test_mps_block_rejects_invalid_tensor_contracts():
    operands, maxq = _case(31, 8, 32, 16, False)
    weights, hessian_inverse, scale, zero = operands

    invalid_calls = [
        (
            (torch.empty(0, 32, device="mps"), hessian_inverse, scale, zero),
            {},
            "dimensions must be positive",
        ),
        (
            (
                torch.empty(1, 129, device="mps"),
                torch.eye(129, device="mps"),
                torch.ones(1, 1, device="mps"),
                torch.zeros(1, 1, device="mps"),
            ),
            {"group_size": 129},
            "count <= 128",
        ),
        ((weights.half(), hessian_inverse, scale, zero), {}, "expects float32"),
        ((weights, hessian_inverse.cpu(), scale, zero), {}, "must share one device"),
        ((weights, torch.eye(31, device="mps"), scale, zero), {}, "must have shape"),
        (
            (weights, hessian_inverse, scale[:, :1].contiguous(), zero),
            {},
            "scale/zero must have shape",
        ),
    ]
    for call_operands, kwargs, message in invalid_calls:
        with pytest.raises((TypeError, ValueError), match=message):
            block_module.gptq_block_mps(
                *call_operands, maxq, kwargs.pop("group_size", 16), **kwargs
            )

    with pytest.raises(TypeError, match="out must be"):
        block_module.gptq_block_mps(*operands, maxq, 16, out=[weights, weights])
    with pytest.raises(ValueError, match="must match weights shape"):
        block_module.gptq_block_mps(
            *operands,
            maxq,
            16,
            out=(torch.empty(1, device="mps"), torch.empty_like(weights)),
        )
    with pytest.raises(TypeError, match="must have dtype float32"):
        block_module.gptq_block_mps(
            *operands,
            maxq,
            16,
            out=(torch.empty_like(weights).half(), torch.empty_like(weights)),
        )
    with pytest.raises(ValueError, match="must share the weights device"):
        block_module.gptq_block_mps(
            *operands,
            maxq,
            16,
            out=(torch.empty_like(weights), torch.empty_like(weights).cpu()),
        )
    noncontiguous_output = torch.empty(8, 64, device="mps")[:, ::2]
    with pytest.raises(ValueError, match="out tensors must be contiguous"):
        block_module.gptq_block_mps(
            *operands, maxq, 16, out=(noncontiguous_output, torch.empty_like(weights))
        )


@mps_only
@pytest.mark.mps
def test_mps_shader_does_not_retain_per_call_tensors():
    operands, maxq = _case(71, 8, 32, 16, False)
    outputs = block_module.gptq_block_mps(*operands, maxq, 16)
    references = [weakref.ref(tensor) for tensor in (*operands, *outputs)]
    del operands, outputs

    assert all(reference() is None for reference in references)
    # Releasing Python tensors before completion remains safe because the MPS
    # command buffer owns the encoded Metal resources until execution ends.
    torch.mps.synchronize()


@mps_only
@pytest.mark.mps
def test_mps_block_concurrent_launches_have_no_cross_call_state():
    cases = [_case(1000 + seed, 64, 64, 16, seed % 2 == 0) for seed in range(8)]
    references = []
    launch_inputs = []
    for operands, maxq in cases:
        reference_weights = operands[0].clone()
        reference = _serial_block(reference_weights, *operands[1:], maxq, 16, maxq == 7)
        references.append((*reference, reference_weights))
        launch_inputs.append((operands[0].clone(), *operands[1:], maxq))

    # Compile first so this stresses concurrent launches through the immutable
    # cached library rather than serial first-use initialization.
    block_module._shader_library()
    start = threading.Barrier(len(launch_inputs))

    def launch(call_inputs):
        weights, hessian_inverse, scale, zero, maxq = call_inputs
        start.wait()
        quantized, errors = block_module.gptq_block_mps(
            weights,
            hessian_inverse,
            scale,
            zero,
            maxq,
            16,
            groupwise=maxq == 7,
        )
        return quantized, errors, weights

    with ThreadPoolExecutor(max_workers=len(launch_inputs)) as executor:
        actual = list(executor.map(launch, launch_inputs))
    torch.mps.synchronize()

    for result, reference in zip(actual, references):
        torch.testing.assert_close(result[0], reference[0], atol=0, rtol=0)
        torch.testing.assert_close(result[1], reference[1], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(result[2], reference[2], atol=2e-5, rtol=2e-5)
