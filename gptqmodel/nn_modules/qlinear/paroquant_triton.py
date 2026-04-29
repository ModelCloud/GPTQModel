# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant Triton runtime implementation adapted from the ParoQuant paper and
# public project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""ParoQuant Triton-backed quantized linear layer."""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import torch

from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT
from ...quantization.paroquant.modules.triton.gemm import (
    paroquant_dequantize_triton,
    paroquant_gemm_triton_decode,
    paroquant_gemm_triton_prefill,
)
from ...utils import has_gil_disabled
from ...utils.backend import BACKEND
from .paroquant import ParoLinear


class ParoQuantTritonLinear(ParoLinear):
    """Use Triton fused kernels for ParoQuant prefill/decode execution."""

    SUPPORTS_BACKENDS = [BACKEND.PAROQUANT_TRITON]
    SUPPORTS_METHODS = ParoLinear.SUPPORTS_METHODS
    SUPPORTS_FORMATS = {FORMAT.PAROQUANT: 0}
    SUPPORTS_BITS = ParoLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = ParoLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = ParoLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = ParoLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = ParoLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = ParoLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = ParoLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = ParoLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = ParoLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = ParoLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = ParoLinear.SUPPORTS_ADAPTERS
    SUPPORTS_DTYPES = ParoLinear.SUPPORTS_DTYPES
    QUANT_TYPE = "awq_paroquant_triton"
    PAROQUANT_TRITON_AUTOTUNE = os.environ.get("GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE", "1") != "0"
    PAROQUANT_TRITON_AUTOTUNE_WARMUP = max(0, int(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE_WARMUP", "2")))
    PAROQUANT_TRITON_AUTOTUNE_ITERS = max(1, int(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE_ITERS", "4")))
    PAROQUANT_TRITON_AUTOTUNE_MARGIN = max(0.0, float(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE_MARGIN", "0.05")))
    PAROQUANT_TRITON_DECODE_MAX_ROWS = max(1, int(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_DECODE_MAX_ROWS", "8")))

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        """Validate the Triton and CUDA runtime prerequisites once per process."""
        from packaging import version
        from triton import __version__ as triton_version

        triton_v = version.parse(triton_version)

        if triton_v < version.parse("2.0.0"):
            raise ImportError(f"triton version must be >= 2.0.0: actual = {triton_version}")

        if has_gil_disabled() and triton_v < version.parse("3.4.0"):
            raise Exception("GIL is disabled and not compatible with current Triton. Please upgrade to Triton >= 3.4.0")

        if not torch.cuda.is_available():
            raise RuntimeError("ParoQuant Triton requires CUDA.")

        return True, None

    def __init__(self, *args, **kwargs):
        """Initialize Triton autotune settings and the per-shape plan cache."""
        kwargs.setdefault("backend", BACKEND.PAROQUANT_TRITON)
        super().__init__(*args, **kwargs)
        self.paroquant_triton_autotune_enabled = self.PAROQUANT_TRITON_AUTOTUNE
        self.paroquant_triton_autotune_warmup = self.PAROQUANT_TRITON_AUTOTUNE_WARMUP
        self.paroquant_triton_autotune_iters = self.PAROQUANT_TRITON_AUTOTUNE_ITERS
        self.paroquant_triton_autotune_margin = self.PAROQUANT_TRITON_AUTOTUNE_MARGIN
        self.paroquant_triton_decode_max_rows = self.PAROQUANT_TRITON_DECODE_MAX_ROWS
        self._plan_cache: dict[str, str] = {}

    def post_init(self):
        """Defer to the shared ParoQuant setup without forcing compute dtype."""
        super().post_init()

    def clear_autotune(self):
        """Drop cached plan decisions after major module state changes."""
        super().clear_autotune()
        self._plan_cache = {}

    @staticmethod
    def _sync_benchmark_device(device: torch.device) -> None:
        """Synchronize CUDA timing measurements used by autotune."""
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    def _classify_forward_kind(self, x: torch.Tensor, x_flat: torch.Tensor) -> str:
        """Classify the workload as decode or prefill for plan selection."""
        if x.dim() >= 3 and x.shape[-2] == 1 and x_flat.shape[0] <= self.paroquant_triton_decode_max_rows:
            return "decode"
        if x_flat.shape[0] <= self.paroquant_triton_decode_max_rows:
            return "decode"
        return "prefill"

    def _forward_triton_dense(self, rotated: torch.Tensor) -> torch.Tensor:
        """Dense fallback used when fused Triton plans are unavailable or slower."""
        weight = paroquant_dequantize_triton(self.qweight, self.scales, self.qzeros)
        if weight.dtype != rotated.dtype or weight.device != rotated.device:
            weight = weight.to(device=rotated.device, dtype=rotated.dtype)

        out = torch.matmul(rotated, weight)
        if self.bias is not None:
            out = out + self.bias.to(device=rotated.device, dtype=rotated.dtype)
        return out

    def _forward_triton_decode(self, rotated: torch.Tensor) -> torch.Tensor:
        """Run the fused Triton kernel optimized for small-row decode workloads."""
        out = paroquant_gemm_triton_decode(rotated, self.qweight, self.scales, self.qzeros)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_triton_prefill(self, rotated: torch.Tensor) -> torch.Tensor:
        """Run the fused Triton kernel optimized for larger prefill batches."""
        out = paroquant_gemm_triton_prefill(rotated, self.qweight, self.scales, self.qzeros)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _run_plan(self, plan: str, rotated: torch.Tensor) -> torch.Tensor:
        """Dispatch one named execution plan."""
        if plan == "dense":
            return self._forward_triton_dense(rotated)
        if plan == "decode_fused":
            return self._forward_triton_decode(rotated)
        if plan == "prefill_fused":
            return self._forward_triton_prefill(rotated)
        raise ValueError(f"Unknown ParoQuant Triton plan: {plan}")

    def _candidate_plans(self, kind: str) -> list[str]:
        """Return the execution plans worth benchmarking for a workload class."""
        if kind == "decode":
            return ["decode_fused", "dense", "prefill_fused"]
        return ["prefill_fused", "dense", "decode_fused"]

    def _benchmark_plan(self, plan: str, rotated: torch.Tensor) -> float:
        """Measure mean latency for one candidate plan."""
        with torch.inference_mode():
            for _ in range(self.paroquant_triton_autotune_warmup):
                self._run_plan(plan, rotated)
            self._sync_benchmark_device(rotated.device)

            start = time.perf_counter()
            for _ in range(self.paroquant_triton_autotune_iters):
                self._run_plan(plan, rotated)
            self._sync_benchmark_device(rotated.device)

        return (time.perf_counter() - start) / self.paroquant_triton_autotune_iters

    def _select_plan(self, kind: str, rotated: torch.Tensor) -> str:
        """Pick the best plan, bounded by a bias toward the default fused path."""
        default_plan = "decode_fused" if kind == "decode" else "prefill_fused"
        if self.training or not self.paroquant_triton_autotune_enabled:
            return default_plan

        cached = self._plan_cache.get(kind)
        if cached is not None:
            return cached

        try:
            timings = {plan: self._benchmark_plan(plan, rotated) for plan in self._candidate_plans(kind)}
            best_plan = min(timings, key=timings.get)
            best_time = timings[best_plan]
            default_time = timings[default_plan]

            if best_plan != default_plan and best_time > default_time * (1.0 - self.paroquant_triton_autotune_margin):
                best_plan = default_plan
        except Exception:
            best_plan = default_plan

        self._plan_cache[kind] = best_plan
        return best_plan

    def forward(self, x: torch.Tensor):
        """Rotate inputs, pick a Triton plan, and preserve adapter semantics."""
        original_shape = x.shape[:-1] + (self.out_features,)
        adapter_input = x.reshape(-1, x.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        rotated = self._rotate_inputs(x_flat)

        plan = self._select_plan(self._classify_forward_kind(x, x_flat), rotated)
        out = self._run_plan(plan, rotated)

        if self.adapter:
            out = self.adapter.apply(x=adapter_input, out=out)

        return out.reshape(original_shape)


__all__ = ["ParoQuantTritonLinear"]
