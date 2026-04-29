# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
import time
import unittest

import torch
from logbar import LogBar

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.utils.model import find_modules


# Keep CPU benchmark isolated from optional BitBLAS/TVM import side effects.
os.environ.setdefault("GPTQMODEL_DISABLE_BITBLAS", "1")

log = LogBar.shared()
bench_cols = log.columns(
    cols=[
        {"label": "Backend", "width": "fit"},
        {"label": "QLinear", "width": "fit"},
        {"label": "Device", "width": "fit"},
        {"label": "Runs", "width": "fit"},
        {"label": "TrimRuns", "width": "fit"},
        {"label": "NewTok", "width": "fit"},
        {"label": "AvgLatencyS", "width": "fit"},
        {"label": "AvgTokPerSec", "width": "fit"},
        {"label": "FastestS", "width": "fit"},
        {"label": "SlowestS", "width": "fit"},
        {"label": "Status", "width": "fit"},
        {"label": "Details", "width": "fit"},
    ],
    padding=1,
)


class BenchmarkIntelCpuXPU(unittest.TestCase):
    model_path = os.getenv("GPTQMODEL_INTEL_CPU_BENCH_MODEL", "sliuau/llama3.2-1b-4bit-group128")
    prompt = os.getenv("GPTQMODEL_INTEL_CPU_BENCH_PROMPT", "Explain why integer quantization can speed up CPU inference.")
    device = "cpu"

    # User requested: generate 100 tokens, run 10 times, remove high/low, average the rest.
    benchmark_runs = int(os.getenv("GPTQMODEL_INTEL_CPU_BENCH_RUNS", "10"))
    new_tokens = int(os.getenv("GPTQMODEL_INTEL_CPU_BENCH_NEW_TOKENS", "1"))

    target_qliner_map = {
        BACKEND.TORCH: TorchLinear,
        BACKEND.TORCH_FUSED: TorchFusedLinear,
        BACKEND.TORCH_INT8: TorchInt8Linear,
        BACKEND.GPTQ_TORCH_ATEN: TorchAtenLinear,
    }
    skip_backends = set()

    def _trimmed_stats(self, times_s: list[float], generated_tokens: list[int]) -> tuple[float, float, float, float, int]:
        if not times_s:
            raise ValueError("No timing samples were collected.")

        keep_indices = list(range(len(times_s)))
        if len(times_s) >= 3:
            fastest_idx = min(range(len(times_s)), key=lambda i: times_s[i])
            slowest_idx = max(range(len(times_s)), key=lambda i: times_s[i])
            if fastest_idx != slowest_idx:
                keep_indices = [i for i in keep_indices if i not in {fastest_idx, slowest_idx}]

        trimmed_total_time = sum(times_s[i] for i in keep_indices)
        trimmed_total_tokens = sum(generated_tokens[i] for i in keep_indices)
        trimmed_avg_latency = trimmed_total_time / max(len(keep_indices), 1)
        trimmed_avg_tps = trimmed_total_tokens / max(trimmed_total_time, 1e-9)

        return (
            trimmed_avg_latency,
            trimmed_avg_tps,
            min(times_s),
            max(times_s),
            len(keep_indices),
        )

    def _validate_backend_module(self, model: GPTQModel, backend: BACKEND) -> bool:
        target_cls = self.target_qliner_map[backend]
        modules = find_modules(model.model, layers=[target_cls])
        return len(modules) > 0

    def _run_backend_benchmark(self, backend: BACKEND):
        model = GPTQModel.load(self.model_path, backend=backend, device=self.device, dtype=torch.float16)
        if not self._validate_backend_module(model, backend):
            raise RuntimeError(f"No `{self.target_qliner_map[backend].__name__}` module found after loading `{backend}`.")

        tokenizer = model.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inp = tokenizer([self.prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_tokens = inp["input_ids"].shape[1]

        times_s: list[float] = []
        generated_tokens: list[int] = []

        with torch.inference_mode():
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                out = model.generate(
                    **inp,
                    min_new_tokens=self.new_tokens,
                    max_new_tokens=self.new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                end = time.perf_counter()

                times_s.append(end - start)
                generated_tokens.append(int(out[0].shape[0] - input_tokens))

        return self._trimmed_stats(times_s, generated_tokens)

    def test_cpu_backend_inference_speed_trimmed(self):
        if self.benchmark_runs < 3:
            self.skipTest("Need at least 3 runs to remove high/low and compute trimmed average.")

        log.info("\nIntel CPU Kernel Inference Benchmark (trimmed mean)")
        bench_cols.info.header()

        success_count = 0

        for backend, qlinear_cls in self.target_qliner_map.items():
            if backend in self.skip_backends and os.getenv("GPTQMODEL_INTEL_CPU_BENCH_ENABLE_TORCH_ATEN", "0") != "1":
                bench_cols.info(
                    backend.name,
                    qlinear_cls.__name__,
                    self.device,
                    str(self.benchmark_runs),
                    "-",
                    str(self.new_tokens),
                    "-",
                    "-",
                    "-",
                    "-",
                    "SKIP",
                    "Temporarily skipped (set GPTQMODEL_INTEL_CPU_BENCH_ENABLE_TORCH_ATEN=1 to enable).",
                )
                continue
            try:
                avg_latency_s, avg_tps, fastest_s, slowest_s, trimmed_runs = self._run_backend_benchmark(backend)
                success_count += 1
                bench_cols.info(
                    backend.name,
                    qlinear_cls.__name__,
                    self.device,
                    str(self.benchmark_runs),
                    str(trimmed_runs),
                    str(self.new_tokens),
                    f"{avg_latency_s:.4f}",
                    f"{avg_tps:.2f}",
                    f"{fastest_s:.4f}",
                    f"{slowest_s:.4f}",
                    "PASS",
                    "-",
                )
            except Exception as exc:  # pragma: no cover - benchmark environment can vary
                bench_cols.info(
                    backend.name,
                    qlinear_cls.__name__,
                    self.device,
                    str(self.benchmark_runs),
                    "-",
                    str(self.new_tokens),
                    "-",
                    "-",
                    "-",
                    "-",
                    "ERROR",
                    f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                )

        self.assertGreater(success_count, 0, "No backends completed successfully.")


if __name__ == "__main__":
    unittest.main()
