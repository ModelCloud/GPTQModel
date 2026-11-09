# SPDX-License-Identifier: Apache-2.0
"""Accurate bounded FP32 XTX products with an FP64 running sum.

The quantizer still receives ordinary FP32 raw covariance. Native activations,
row order, sample counts, damping, Hadamard transforms and factorization are
unchanged. Keeping the outer sum in FP64 avoids losing small contributions
when a full calibration stream contains thousands of bounded products.
"""
from __future__ import annotations

import os

import torch


HESSIAN_ACCUMULATION_CONTRACT = "fp32-products-k128-fp64-sum-fp32-storage-v1"


class BoundedHessianAccumulator:
    def __init__(self, width: int, *, device="cuda:0", max_rows: int = 128):
        if type(width) is not int or width <= 0:
            raise ValueError("Hessian width must be a positive integer")
        if type(max_rows) is not int or not 1 <= max_rows <= 128:
            raise ValueError("Hessian products require 1..128 actual rows")
        self.width, self.max_rows = width, max_rows
        self._sum = torch.zeros((width, width), device=device, dtype=torch.float64)
        self.count = 0
        self._failed = False

    @property
    def device(self):
        return self._sum.device

    @torch.inference_mode()
    def add(self, inputs: torch.Tensor) -> int:
        if self._failed:
            raise RuntimeError("cannot reuse a Hessian after a failed partial update")
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim < 2
                or inputs.shape[-1] != self.width
                or inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError("Hessian inputs require native FP16/BF16/FP32 rows of the declared width")
        rows = inputs.reshape(-1, self.width)
        if not len(rows) or not bool(torch.isfinite(rows).all()):
            raise ValueError("Hessian observations must be nonempty and finite")
        if self.device.type == "cuda" and (torch.backends.cuda.matmul.allow_tf32
                or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1"):
            raise ValueError("Hessian products require IEEE FP32 matmul with TF32 disabled")
        try:
            for start in range(0, len(rows), self.max_rows):
                value = rows[start:start+self.max_rows].to(device=self.device, dtype=torch.float32)
                product = torch.mm(value.T, value)
                if not bool(torch.isfinite(product).all()):
                    raise ValueError("nonfinite FP32 covariance product")
                self._sum.add_(product)
            if not bool(torch.isfinite(self._sum).all()):
                raise ValueError("nonfinite FP64 covariance sum")
        except BaseException:
            self._failed = True
            raise
        self.count += len(rows)
        return len(rows)

    update = add

    @torch.inference_mode()
    def snapshot(self) -> torch.Tensor:
        """Return owned FP32 storage without mutating the running sum."""
        if self._failed:
            raise RuntimeError("cannot export a Hessian after a failed partial update")
        result = self._sum.to(dtype=torch.float32)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("covariance cannot be represented as finite FP32 storage")
        return result
