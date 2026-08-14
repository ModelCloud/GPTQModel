# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import torch

from gptqmodel.utils import diagnostic_metrics
from scripts.analyze_gptq_low_bit_grid import tensor_metrics


def _flatten_metrics(value: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for name, item in value.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            flattened.update(_flatten_metrics(item, key))
        elif isinstance(item, (bool, float, int)):
            flattened[key] = float(item)
    return flattened


def _assert_metric_parity(reference: dict[str, Any], actual: dict[str, Any]) -> None:
    reference_values = _flatten_metrics(reference)
    actual_values = _flatten_metrics(actual)
    assert actual_values.keys() == reference_values.keys()
    for name, expected in reference_values.items():
        measured = actual_values[name]
        if math.isnan(expected):
            assert math.isnan(measured), name
        else:
            assert measured == pytest.approx(expected, abs=1e-6, rel=1e-6), name


@pytest.fixture(scope="module", autouse=True)
def _require_native_extension():
    if not diagnostic_metrics._DIAGNOSTIC_METRICS_CPU_EXTENSION.load():
        pytest.skip(
            diagnostic_metrics._DIAGNOSTIC_METRICS_CPU_EXTENSION.last_error_message()
        )


@pytest.mark.parametrize("normalize_distribution", [False, True])
@pytest.mark.parametrize(
    ("shape", "seed"),
    [
        ((17, 257), 1),
        ((2, 3, 19), 2),
        ((9, 1), 3),
        ((4, 5), 4),
    ],
)
def test_native_metrics_match_reference_for_normal_and_edge_widths(
    monkeypatch,
    normalize_distribution,
    shape,
    seed,
):
    generator = torch.Generator().manual_seed(seed)
    dense = torch.randn(shape, generator=generator)
    quantized = dense + 0.25 * torch.randn(shape, generator=generator)

    monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "0")
    reference = tensor_metrics(
        dense, quantized, normalize_distribution=normalize_distribution
    )
    monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "1")
    actual = tensor_metrics(
        dense, quantized, normalize_distribution=normalize_distribution
    )

    _assert_metric_parity(reference, actual)


@pytest.mark.parametrize("normalize_distribution", [False, True])
def test_native_metrics_preserve_exact_identical_and_zero_signal_results(
    monkeypatch, normalize_distribution
):
    for dense in (torch.ones((4, 9)), torch.zeros((3, 6))):
        monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "0")
        reference = tensor_metrics(
            dense, dense.clone(), normalize_distribution=normalize_distribution
        )
        monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "1")
        actual = tensor_metrics(
            dense, dense.clone(), normalize_distribution=normalize_distribution
        )

        _assert_metric_parity(reference, actual)
        assert actual["rmse"] == 0.0
        assert actual["relative_l2"] == 0.0
        assert actual["kl_forward"]["max"] == pytest.approx(0.0, abs=1e-7)
        assert actual["top1_agreement"] == 1.0


def test_native_metrics_match_reference_for_ties_noncontiguous_and_float64(monkeypatch):
    dense_storage = torch.tensor(
        [[2.0, 0.0, 2.0, 1.0, 2.0, -1.0], [1.0, 3.0, 3.0, 0.0, 3.0, 2.0]],
        dtype=torch.float64,
    )
    quantized_storage = torch.tensor(
        [[2.0, 0.0, 1.0, 2.0, 2.0, -1.0], [3.0, 1.0, 3.0, 0.0, 2.0, 3.0]],
        dtype=torch.float64,
    )
    dense = dense_storage[:, ::2]
    quantized = quantized_storage[:, ::2]
    assert not dense.is_contiguous()

    monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "0")
    reference = tensor_metrics(dense, quantized, normalize_distribution=False)
    monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "1")
    actual = tensor_metrics(dense, quantized, normalize_distribution=False)

    _assert_metric_parity(reference, actual)


def _fp64_row_metric_oracle(
    dense: torch.Tensor,
    quantized: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Exact fp64 row metrics for the same formulas as the harness reference."""

    dense64 = dense.double()
    quantized64 = quantized.double()
    dense_log_probability = torch.log_softmax(dense64, -1)
    quantized_log_probability = torch.log_softmax(quantized64, -1)
    dense_probability = dense_log_probability.exp()
    quantized_probability = quantized_log_probability.exp()
    midpoint = (dense_probability + quantized_probability) * 0.5
    midpoint_log = midpoint.clamp_min(1e-30).log()
    dense_square = dense64.square().sum(-1)
    quantized_square = quantized64.square().sum(-1)
    dense_norm = dense_square.sqrt()
    quantized_norm = quantized_square.sqrt()
    denominator = dense_norm * quantized_norm
    cosine = torch.where(
        denominator == 0.0,
        torch.zeros_like(denominator),
        (dense64 * quantized64).sum(-1) / denominator,
    )
    return {
        "row_cosine": cosine,
        "kl_forward": (dense_probability * (dense_log_probability - quantized_log_probability)).sum(-1),
        "kl_reverse": (quantized_probability * (quantized_log_probability - dense_log_probability)).sum(-1),
        "jensen_shannon": 0.5
        * (
            (dense_probability * (dense_log_probability - midpoint_log)).sum(-1)
            + (quantized_probability * (quantized_log_probability - midpoint_log)).sum(-1)
        ),
        "total_variation": 0.5 * (dense_probability - quantized_probability).abs().sum(-1),
        "hellinger": ((dense_probability.sqrt() - quantized_probability.sqrt()).square().sum(-1) * 0.5).sqrt(),
        "dense_entropy": -(dense_probability * dense_log_probability).sum(-1),
        "dense_to_quantized_cross_entropy": -(dense_probability * quantized_log_probability).sum(-1),
    }


@pytest.mark.parametrize(
    ("dense", "quantized", "expected_rmse"),
    (
        (torch.full((2, 7), 1e20), torch.full((2, 7), -1e20), 2e20),
        (torch.full((2, 7), 1e-30), torch.full((2, 7), 2e-30), 1e-30),
    ),
)
def test_native_metrics_match_fp64_reference_without_fp32_reduction_overflow_or_underflow(
    monkeypatch,
    dense,
    quantized,
    expected_rmse,
):
    # The native kernel must compute the exact fp64 metric values without fp32
    # reduction overflow/underflow. The fp32 harness reference saturates (e.g.
    # row_cosine 0.0 for 1e20/-1e20 and 1e-30/2e-30 inputs), so this gate uses
    # an independent fp64 oracle and allows only fp32 output-storage rounding.
    monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "1")
    actual = tensor_metrics(dense, quantized, normalize_distribution=False)

    oracle = _fp64_row_metric_oracle(dense, quantized)
    for metric, expected in oracle.items():
        measured = torch.tensor(actual[metric]["mean"])
        expected_fp32 = expected.float().mean()
        assert torch.allclose(measured, expected_fp32, atol=1e-8, rtol=1e-8), metric
    assert math.isfinite(actual["rmse"])
    assert actual["rmse"] == pytest.approx(expected_rmse, rel=1e-6)
    # Exact values for the crafted cases.
    if expected_rmse == 2e20:
        assert actual["row_cosine"]["mean"] == pytest.approx(-1.0, abs=1e-8)
    else:
        assert actual["row_cosine"]["mean"] == pytest.approx(1.0, abs=1e-8)


def test_native_metrics_reject_shape_mismatch_before_extension_load(monkeypatch):
    monkeypatch.setattr(
        diagnostic_metrics._DIAGNOSTIC_METRICS_CPU_EXTENSION,
        "load",
        lambda: pytest.fail("shape validation must run before extension loading"),
    )
    with pytest.raises(ValueError, match="metric shape mismatch"):
        diagnostic_metrics.native_tensor_metrics(
            torch.zeros((2, 3)),
            torch.zeros((2, 4)),
            normalize_distribution=False,
        )


def test_native_metrics_returns_none_when_disabled_or_extension_is_unavailable(
    monkeypatch,
):
    dense = torch.zeros((2, 3))
    monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "off")
    assert (
        diagnostic_metrics.native_tensor_metrics(
            dense, dense, normalize_distribution=False
        )
        is None
    )

    monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "1")
    monkeypatch.setattr(
        diagnostic_metrics._DIAGNOSTIC_METRICS_CPU_EXTENSION, "load", lambda: False
    )
    assert (
        diagnostic_metrics.native_tensor_metrics(
            dense, dense, normalize_distribution=False
        )
        is None
    )


def test_native_operation_validates_dtype_contiguity_and_nonempty_inputs():
    operation = diagnostic_metrics._DIAGNOSTIC_METRICS_CPU_EXTENSION.op(
        "tensor_metrics_cpu"
    )
    with pytest.raises(RuntimeError, match="must be float32"):
        operation(
            torch.zeros((2, 3), dtype=torch.float64),
            torch.zeros((2, 3), dtype=torch.float64),
            False,
        )
    with pytest.raises(RuntimeError, match="must be contiguous"):
        operation(torch.zeros((2, 6))[:, ::2], torch.zeros((2, 6))[:, ::2], False)
    with pytest.raises(RuntimeError, match="must be nonempty"):
        operation(torch.zeros((2, 0)), torch.zeros((2, 0)), False)


def test_native_metrics_are_thread_safe_under_free_threaded_parallel_calls(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "1")
    generator = torch.Generator().manual_seed(19)
    dense = torch.randn((32, 127), generator=generator)
    quantized = dense + 0.1 * torch.randn((32, 127), generator=generator)
    expected = diagnostic_metrics.native_tensor_metrics(
        dense, quantized, normalize_distribution=True
    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: diagnostic_metrics.native_tensor_metrics(
                    dense,
                    quantized,
                    normalize_distribution=True,
                ),
                range(12),
            )
        )

    assert expected is not None
    for result in results:
        assert result is not None
        _assert_metric_parity(expected, result)
