# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Bitwise regression: vectorized MSE shrink grid vs the sequential oracle.

``Quantizer.find_params`` and the AWQ ``_compute_best_clip`` loops batch their
grid candidates into chunked kernels instead of launching a dozen small ops per
candidate.  The selection must stay bitwise-identical: same objective values,
same first-minimum tie order, same NaN rejection.  The sequential loops are
reproduced here verbatim as independent oracles.
"""
import os
import unittest

import torch
from parameterized import parameterized

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.quantizer import Quantizer, quantize


def _legacy_find_params(self, x, weight=False):
    """Pre-vectorization ``Quantizer.find_params`` verbatim oracle."""
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
        search_adjacent_zero_points = self._search_adjacent_zero_points()
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
                alternate_zero = self._adjacent_zero_point(
                    xmin1, scale1, zero1, maxq=maxq_value,
                )
                zero_candidates = (zero1, alternate_zero)

            for zero_candidate in zero_candidates:
                q = quantize(
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


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestMseShrinkGridVectorized(unittest.TestCase):
    """``scale``/``zero`` must be bitwise-equal to the sequential oracle."""

    def _run_pair(self, qcfg, x, weight, perchannel, trits=False):
        legacy = Quantizer(qcfg)
        new = Quantizer(qcfg)
        legacy.configure(perchannel=perchannel, trits=trits)
        new.configure(perchannel=perchannel, trits=trits)
        _legacy_find_params(legacy, x.clone(), weight=weight)
        new.find_params(x.clone(), weight=weight)
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
        os.environ["GPTQMODEL_MSE_GRID_BUDGET_MB"] = "0"
        try:
            qcfg = QuantizeConfig(bits=4, group_size=-1, sym=False, mse=2.4)
            x = _case("randn", (96, 512))
            legacy, new = self._run_pair(qcfg, x, weight=True, perchannel=True)
            self.assertTrue(
                torch.allclose(legacy.scale, new.scale, rtol=0, atol=0, equal_nan=True))
            self.assertTrue(
                torch.allclose(legacy.zero, new.zero, rtol=0, atol=0, equal_nan=True))
        finally:
            os.environ.pop("GPTQMODEL_MSE_GRID_BUDGET_MB", None)


if __name__ == "__main__":
    unittest.main()
