# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Bitwise regression: vectorized MSE shrink grid vs the sequential oracle.

The selection must stay identical: same objective values, first-minimum tie
order, and NaN rejection. The sequential loop and quantization arithmetic below
are independent of the batched implementation.
"""
import os
import unittest
from unittest.mock import patch

import pytest
import torch
from parameterized import parameterized, parameterized_class
from torch.utils._python_dispatch import TorchDispatchMode

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.quantizer import QQQQuantizer, Quantizer, _mse_shrink_grid


def _oracle_quantize(x, scale, zero, maxq, groupwise):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    zero_mask = scale == 0
    zero = torch.where(zero_mask & ~torch.isfinite(zero), torch.zeros_like(zero), zero)
    scale = torch.where(zero_mask, torch.full_like(scale, torch.finfo(scale.dtype).tiny), scale)
    if groupwise:
        return scale * torch.clamp(torch.round(x / scale), -maxq, maxq)
    return scale * (torch.clamp(torch.round(x / scale) + zero, 0, maxq) - zero)


def _legacy_find_params(self, x, weight=False):
    """Sequential ``Quantizer.find_params`` with independent quantization math."""
    dev = x.device
    if self.maxq.device != dev:
        self.maxq = self.maxq.to(dev)

    shape = x.shape
    if self.perchannel:
        if weight:
            x = x.flatten(1)
        else:
            if len(shape) == 4:
                x = x.permute([1, 0, 2, 3])
                x = x.flatten(1)
            if len(shape) == 3:
                x = x.reshape((-1, shape[-1])).t()
            if len(shape) == 2:
                x = x.t()
    else:
        x = x.flatten().unsqueeze(0)

    tmp = torch.zeros(x.shape[0], device=dev)
    xmin_raw, xmax_raw = torch.aminmax(x, dim=1)
    xmin = torch.minimum(xmin_raw, tmp)
    xmax = torch.maximum(xmax_raw, tmp)

    if self.qcfg.sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        xmin = torch.where(xmin < 0, -xmax, xmin)
    both_zero = xmax == 0 if self.qcfg.sym else (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, xmin - 1, xmin)
    xmax = torch.where(both_zero, xmax + 1, xmax)

    if self._maxq_value < 0:
        self.scale = xmax
        self.zero = xmin
    else:
        if self.requires_groupwise_processing():
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.qcfg.sym:
                self.zero = torch.full_like(self.scale, (self._maxq_value + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

    mse = float(getattr(self.qcfg, "mse", 0.0) or 0.0)
    if mse > 0.0:
        best = torch.full([x.shape[0]], float("inf"), device=dev)
        search_adjacent_zero_points = (
            self._maxq_value == 3 and not self.qcfg.sym and not self.requires_groupwise_processing()
        )
        maxq_value = self._maxq_value if search_adjacent_zero_points else 0
        for i in range(int(self.maxshrink * self.grid)):
            p = 1 - i / self.grid
            xmin1 = p * xmin
            xmax1 = p * xmax
            scale1 = (
                xmax1 / self.maxq
                if self.requires_groupwise_processing()
                else (xmax1 - xmin1) / self.maxq
            )
            zero1 = torch.round(-xmin1 / scale1) if not self.qcfg.sym else self.zero
            zero_candidates = (zero1,)
            if search_adjacent_zero_points:
                ideal = -xmin1 / scale1
                lower = torch.floor(ideal).clamp(0, maxq_value)
                upper = torch.ceil(ideal).clamp(0, maxq_value)
                alternate_zero = torch.where(zero1 == lower, upper, lower)
                zero_candidates = (zero1, alternate_zero)

            for zero_candidate in zero_candidates:
                q = _oracle_quantize(
                    x,
                    scale1.unsqueeze(1),
                    zero_candidate.unsqueeze(1),
                    self._maxq_value,
                    self.requires_groupwise_processing(),
                )
                q -= x
                q.abs_()
                q.pow_(mse)
                err = torch.sum(q, 1)
                better = err < best
                best = torch.where(better, err, best)
                self.scale = torch.where(better, scale1, self.scale)
                self.zero = torch.where(better, zero_candidate, self.zero)
    if not self.perchannel:
        if weight:
            tmp = shape[0]
        else:
            tmp = shape[1] if len(shape) != 3 else shape[2]
        self.scale = self.scale.repeat(tmp)
        self.zero = self.zero.repeat(tmp)

    if weight:
        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return
    if len(shape) == 4:
        self.scale = self.scale.reshape((1, -1, 1, 1))
        self.zero = self.zero.reshape((1, -1, 1, 1))
    if len(shape) == 3:
        self.scale = self.scale.reshape((1, 1, -1))
        self.zero = self.zero.reshape((1, 1, -1))
    if len(shape) == 2:
        self.scale = self.scale.unsqueeze(0)
        self.zero = self.zero.unsqueeze(0)


def _case(kind, shape, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g).to(dtype)
    if kind == "randn":
        return x
    if kind == "dead":
        x[::2] = 0.0
        if x.dim() > 1:
            x[:, ::3] = 0.0
        return x
    if kind == "const":
        return torch.zeros(shape, dtype=dtype)
    if kind == "outlier":
        x[::3] *= 1e4
        return x
    if kind == "naninf":
        flat = x.flatten()
        flat[::11] = float("nan")
        flat[5::13] = float("inf")
        return x
    if kind == "tiny":
        return x * 1e-8
    raise ValueError(kind)


_DEVICES = ["cpu", "cuda"]


@parameterized_class("device", [(device,) for device in _DEVICES])
class TestMseShrinkGridVectorized(unittest.TestCase):
    """``scale``/``zero`` must be bitwise-equal to the sequential oracle."""

    def setUp(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

    def _run_pair(self, qcfg, x, weight, perchannel, trits=False, quantizer_cls=Quantizer):
        x = x.to(self.device)
        legacy = quantizer_cls(qcfg)
        new = quantizer_cls(qcfg)
        legacy.configure(perchannel=perchannel, trits=trits)
        new.configure(perchannel=perchannel, trits=trits)
        _legacy_find_params(legacy, x, weight=weight)
        new.find_params(x, weight=weight)
        return legacy, new

    @parameterized.expand([
        (4, False, "randn"),
        (4, True, "randn"),
        (3, False, "dead"),
        (2, False, "randn"),   # adjacent-zero-point search path
        (2, False, "const"),   # all-equal candidates -> tie-order check
        (8, False, "outlier"),
        (4, False, "naninf"),  # NaN objectives must stay unselectable
        (4, False, "tiny"),    # zero/near-zero scales hit the zero-scale guard
    ])
    def test_weight_grid_matches_oracle(self, bits, sym, kind):
        qcfg = QuantizeConfig(bits=bits, group_size=-1, sym=sym, mse=2.4)
        x = _case(kind, (96, 512))
        legacy, new = self._run_pair(qcfg, x, weight=True, perchannel=True)
        self.assertTrue(
            torch.allclose(legacy.scale, new.scale, rtol=0, atol=0, equal_nan=True),
            "scale mismatch",
        )
        self.assertTrue(
            torch.allclose(legacy.zero, new.zero, rtol=0, atol=0, equal_nan=True),
            "zero mismatch",
        )

    def test_trits_grid_matches_oracle(self):
        qcfg = QuantizeConfig(bits=4, group_size=-1, sym=False, mse=2.4)
        x = _case("randn", (64, 256))
        legacy, new = self._run_pair(qcfg, x, weight=True, perchannel=True, trits=True)
        self.assertTrue(
            torch.allclose(legacy.scale, new.scale, rtol=0, atol=0, equal_nan=True))
        self.assertTrue(
            torch.allclose(legacy.zero, new.zero, rtol=0, atol=0, equal_nan=True))

    def test_activation_inputs_match_oracle(self):
        qcfg = QuantizeConfig(bits=4, group_size=-1, sym=False, mse=2.4)
        for x in (
            _case("randn", (32, 768)),
            _case("outlier", (2, 16, 384)),
            _case("randn", (2, 8, 6, 128)),
        ):
            legacy, new = self._run_pair(qcfg, x, weight=False, perchannel=True)
            self.assertTrue(
                torch.allclose(legacy.scale, new.scale, rtol=0, atol=0, equal_nan=True))
            self.assertTrue(
                torch.allclose(legacy.zero, new.zero, rtol=0, atol=0, equal_nan=True))

    def test_tiny_budget_chunking_matches_oracle(self):
        """A 0-MB budget forces one candidate per chunk; exercise the boundary."""
        with patch.dict(os.environ, {"GPTQMODEL_MSE_GRID_BUDGET_MB": "0"}):
            qcfg = QuantizeConfig(bits=4, group_size=-1, sym=False, mse=2.4)
            x = _case("randn", (96, 512))
            legacy, new = self._run_pair(qcfg, x, weight=True, perchannel=True)
            self.assertTrue(
                torch.allclose(legacy.scale, new.scale, rtol=0, atol=0, equal_nan=True))
            self.assertTrue(
                torch.allclose(legacy.zero, new.zero, rtol=0, atol=0, equal_nan=True))

    @parameterized.expand([
        (torch.float16, Quantizer, False),
        (torch.bfloat16, Quantizer, False),
        (torch.float64, Quantizer, False),
        (torch.float32, QQQQuantizer, True),
        (torch.float16, QQQQuantizer, True),
    ])
    def test_dtypes_and_groupwise(self, dtype, quantizer_cls, sym):
        qcfg = QuantizeConfig(bits=4, group_size=-1, sym=sym, mse=2.4)
        for perchannel in (False, True):
            legacy, new = self._run_pair(
                qcfg, _case("randn", (16, 64), dtype=dtype), weight=True,
                perchannel=perchannel, quantizer_cls=quantizer_cls,
            )
            torch.testing.assert_close(new.scale, legacy.scale, rtol=0, atol=0)
            torch.testing.assert_close(new.zero, legacy.zero, rtol=0, atol=0)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
@pytest.mark.parametrize("budget", ["0", "1"])
def test_mse_thresholds_and_candidate_ties(device, budget, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_MSE_GRID_BUDGET_MB", budget)
    centers = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], device=device)
    x = torch.cat([
        centers, torch.nextafter(centers, torch.full_like(centers, -torch.inf)),
        torch.nextafter(centers, torch.full_like(centers, torch.inf)),
        torch.tensor([-100, -0.0, 0.0, 100], device=device),
    ]).repeat(4, 1)
    scales = torch.tensor([0.0, 1.0, 1.0, 0.5, 2.0], device=device)[:, None].expand(-1, 4)
    zeros = torch.zeros_like(scales)
    initial = torch.ones(4, device=device)
    best = torch.full_like(initial, torch.inf)
    expected_scale, expected_zero = initial, initial
    for scale, zero in zip(scales, zeros):
        q = _oracle_quantize(x, scale[:, None], zero[:, None], 3, True)
        error = (q - x).abs().pow(2.4).sum(1)
        better = error < best
        expected_scale = torch.where(better, scale, expected_scale)
        expected_zero = torch.where(better, zero, expected_zero)
        best = torch.where(better, error, best)
    actual_scale, actual_zero = _mse_shrink_grid(
        x, scales, zeros, 2.4, maxq_value=3, groupwise=True,
        guard_zero_scale=True, scale=initial, zero=initial,
    )
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual_zero, expected_zero, rtol=0, atol=0)


@pytest.mark.parametrize("budget", ["0", "1"])
def test_mse_strided_transpose_objectives(budget, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_MSE_GRID_BUDGET_MB", budget)
    x = torch.randn(64, 128, generator=torch.Generator().manual_seed(0))[::2, ::2].t()
    scales = torch.tensor([0.125, 0.25, 0.5])[:, None].expand(-1, x.shape[0])
    zeros = torch.full_like(scales, 8.0)
    expected = []
    for scale, zero in zip(scales, zeros):
        q = _oracle_quantize(x, scale[:, None], zero[:, None], 15, False)
        expected.append((q - x).abs().pow(2.4).sum(1))
    errors = []

    class TrackErrors(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.sum.dim_IntList:
                errors.append(output.reshape(-1, x.shape[0]).clone())
            return output

    with TrackErrors():
        _mse_shrink_grid(
            x, scales, zeros, 2.4, maxq_value=15, groupwise=False,
            guard_zero_scale=True, scale=scales[0], zero=zeros[0],
        )
    torch.testing.assert_close(torch.cat(errors), torch.stack(expected), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
