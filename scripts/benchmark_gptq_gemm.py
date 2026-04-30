# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from gptqmodel import GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import METHOD
from gptqmodel.utils.backend import BACKEND, normalize_backend
from gptqmodel.utils.model import find_modules


@dataclass
class BackendResult:
    backend: str
    status: str
    target: str | None = None
    samples: int = 0
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    load_peak_allocated_mb: float | None = None
    load_peak_reserved_mb: float | None = None
    forward_peak_allocated_mb: float | None = None
    forward_peak_reserved_mb: float | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    error: str | None = None


def parse_shapes(raw: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        rows, samples = part.split(":", maxsplit=1)
        shapes.append((int(rows), int(samples)))
    if not shapes:
        raise ValueError("At least one shape must be provided.")
    return shapes


def parse_backends(raw: str) -> list[BACKEND]:
    backends = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        backends.append(normalize_backend(part, quant_method=METHOD.GPTQ))
    return backends


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def module_device(module: torch.nn.Module) -> torch.device | None:
    for tensor in module.parameters(recurse=False):
        if tensor is not None and not tensor.is_meta:
            return tensor.device
    for tensor in module.buffers(recurse=False):
        if tensor is not None and not tensor.is_meta:
            return tensor.device
    return None


def find_target_module(model: GPTQModel, target: str | None) -> tuple[str, BaseQuantLinear]:
    modules = find_modules(model.model, layers=[model.qlinear_kernel])
    if target is not None:
        module = modules.get(target)
        if module is None:
            available = ", ".join(list(modules)[:10])
            raise ValueError(f"Target `{target}` not found. First available modules: {available}")
        return target, module
    if not modules:
        raise ValueError("No quant linear modules found in loaded model.")
    name = next(iter(modules))
    return name, modules[name]


def make_samples(shapes: Iterable[tuple[int, int]], in_features: int, device: torch.device) -> list[torch.Tensor]:
    samples = []
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    for rows, count in shapes:
        for _ in range(count):
            samples.append(torch.rand((rows, in_features), device=device, dtype=torch.float16, generator=generator))
    return samples


def run_backend(
    *,
    model_path: str,
    backend: BACKEND,
    target: str | None,
    shapes: list[tuple[int, int]],
    reference_outputs: list[torch.Tensor] | None,
    samples: list[torch.Tensor] | None,
    warmup: int,
) -> tuple[BackendResult, list[torch.Tensor] | None, list[torch.Tensor] | None]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = BackendResult(backend=backend.value, status="ok")
    try:
        load_started = time.perf_counter()
        model = GPTQModel.load(model_path, backend=backend, dtype=torch.float16, device="cuda:0")
        torch.cuda.synchronize()
        _ = load_started
        result.load_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        result.load_peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        target_name, module = find_target_module(model, target)
        result.target = target_name
        device = module_device(module) or torch.device("cuda:0")

        if samples is None:
            samples = make_samples(shapes, module.in_features, device)
        else:
            samples = [sample.to(device=device, dtype=torch.float16) for sample in samples]

        for sample in samples[:warmup]:
            module(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        outputs: list[torch.Tensor] = []
        timings: list[float] = []
        for sample in samples:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = module(sample)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
            outputs.append(out.detach().to(device="cpu", dtype=torch.float32))

        result.samples = len(outputs)
        result.mean_ms = statistics.fmean(timings) if timings else 0.0
        result.p50_ms = percentile(timings, 50.0)
        result.p95_ms = percentile(timings, 95.0)
        result.forward_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        result.forward_peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        if reference_outputs is not None:
            max_abs_diff = 0.0
            mean_abs_diff = 0.0
            for ref, actual in zip(reference_outputs, outputs):
                diff = (actual - ref).abs()
                max_abs_diff = max(max_abs_diff, float(diff.max().item()))
                mean_abs_diff += float(diff.mean().item())
            result.max_abs_diff = max_abs_diff
            result.mean_abs_diff = mean_abs_diff / len(outputs) if outputs else 0.0

        del model
        torch.cuda.empty_cache()
        return result, outputs, samples
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        torch.cuda.empty_cache()
        return result, None, samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GPTQ-GEMM against existing GPTQ kernels.")
    parser.add_argument("--model", required=True, help="Quantized GPTQ model path or HF repo id.")
    parser.add_argument("--target", default=None, help="Optional quant linear module name to benchmark.")
    parser.add_argument(
        "--backends",
        default="gptq-gemm,gptq_marlin,gptq_exllama_v2,gptq_triton,gptq_machete,gptq_bitblas",
        help="Comma-separated candidate backends. Torch reference is always run first.",
    )
    parser.add_argument("--shapes", default="1:64,8:32,16:16,32:8", help="rows:samples comma list.")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for GPTQ-GEMM benchmarking.")

    shapes = parse_shapes(args.shapes)
    candidates = parse_backends(args.backends)
    results: list[BackendResult] = []

    reference_result, reference_outputs, samples = run_backend(
        model_path=args.model,
        backend=BACKEND.GPTQ_TORCH,
        target=args.target,
        shapes=shapes,
        reference_outputs=None,
        samples=None,
        warmup=args.warmup,
    )
    results.append(reference_result)
    if reference_outputs is None:
        raise SystemExit(f"Torch reference failed: {reference_result.error}")

    for backend in candidates:
        result, _, samples = run_backend(
            model_path=args.model,
            backend=backend,
            target=reference_result.target,
            shapes=shapes,
            reference_outputs=reference_outputs,
            samples=samples,
            warmup=args.warmup,
        )
        results.append(result)

    rows = [asdict(row) for row in results]
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
