#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel  # noqa: E402
from gptqmodel.utils import internal_gguf  # noqa: E402


DEFAULT_MODEL = Path(
    "/monster/data/model/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf"
)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"

    lines = [separator, format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def _event_samples(fn: Callable[[], object], *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _summary_row(stage: str, samples: list[float], tokens_per_iteration: int) -> list[str]:
    median_ms = statistics.median(samples)
    return [
        stage,
        str(len(samples)),
        f"{statistics.mean(samples):.4f}",
        f"{median_ms:.4f}",
        f"{_percentile(samples, 0.95):.4f}",
        f"{min(samples):.4f}",
        f"{max(samples):.4f}",
        f"{tokens_per_iteration * 1000.0 / median_ms:.2f}",
    ]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _triton_cache_bytes(modules: list[GGUFTritonKernel]) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for module in modules:
        cache_stores = (module._gguf_triton_cache, module._gguf_q2_native_scale_cache)
        for cache_store in cache_stores:
            for cache in cache_store.values():
                for value in cache.values():
                    if not torch.is_tensor(value):
                        continue
                    key = (value.data_ptr(), _tensor_bytes(value))
                    if key in seen:
                        continue
                    seen.add(key)
                    total += key[1]
    return total


def _memory_row(stage: str, modules: list[GGUFTritonKernel]) -> list[str]:
    return [
        stage,
        f"{torch.cuda.memory_allocated() / (1024**2):.2f}",
        f"{torch.cuda.memory_reserved() / (1024**2):.2f}",
        f"{torch.cuda.max_memory_allocated() / (1024**2):.2f}",
        f"{sum(_tensor_bytes(module.qweight) for module in modules) / (1024**2):.2f}",
        f"{_triton_cache_bytes(modules) / (1024**2):.2f}",
    ]


def _profile_range(name: str, fn: Callable[[], object], iterations: int) -> None:
    torch.cuda.nvtx.range_push(name)
    try:
        for _ in range(iterations):
            fn()
    finally:
        torch.cuda.nvtx.range_pop()


def _run_model(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    wrapper = GPTQModel.load(
        str(args.model),
        backend=BACKEND.GGUF_TRITON,
        profile="low_memory",
        device=f"cuda:{args.device}",
        dtype=torch.float16,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    model = wrapper.model.eval()
    modules = [module for module in model.modules() if isinstance(module, GGUFTritonKernel)]
    fused_rms_norms = sum(
        1 for module in model.modules() if getattr(module, "_gptqmodel_prism_q2_rms_norm", False)
    )
    fused_swiglu = sum(
        1 for module in model.modules() if getattr(module, "_gptqmodel_prism_q2_swiglu", False)
    )
    fused_qkv = sum(
        1 for module in model.modules() if getattr(module, "_gptqmodel_prism_q2_qkv", False)
    )
    memory_rows = [_memory_row("loaded", modules)]

    vocab_size = int(model.config.vocab_size)
    input_ids = (torch.arange(args.prompt_tokens, device="cuda", dtype=torch.long) % vocab_size).unsqueeze(0)

    def prefill():
        return model(input_ids=input_ids, use_cache=True)

    with torch.inference_mode():
        prefill_samples = _event_samples(prefill, warmup=args.warmup, iterations=args.iterations)
        prefill_output = prefill()
        cache = prefill_output.past_key_values
        next_token = prefill_output.logits[:, -1:, :].argmax(dim=-1)
        if args.release_prefill_cache:
            for module in modules:
                module.release_q2_prefill_cache()
            torch.cuda.empty_cache()
            memory_rows.append(_memory_row("cache-released", modules))

        def decode():
            nonlocal cache, next_token
            output = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            next_token = output.logits[:, -1:, :].argmax(dim=-1)
            return output

        decode_samples = _event_samples(decode, warmup=args.warmup, iterations=args.iterations)
        memory_rows.append(_memory_row("warmed", modules))

        if args.capture:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            _profile_range("prism_q2_0_prefill", prefill, args.profile_prefill)
            if args.release_prefill_cache and args.profile_prefill:
                for module in modules:
                    module.release_q2_prefill_cache()
            _profile_range("prism_q2_0_decode", decode, args.profile_decode)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

    properties = torch.cuda.get_device_properties(args.device)
    print(
        f"model={args.model} backend=GGUF_TRITON dtype=fp16 device={args.device} "
        f"gpu={properties.name!r} capability={properties.major}.{properties.minor} "
        f"sms={properties.multi_processor_count} prompt_tokens={args.prompt_tokens} "
        f"warmup={args.warmup} iterations={args.iterations} load_s={load_seconds:.3f} "
        f"qlinear_modules={len(modules)} fused_rms_norms={fused_rms_norms} "
        f"fused_swiglu={fused_swiglu} fused_qkv={fused_qkv}"
    )
    print(
        _ascii_table(
            ["stage", "iters", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "tokens/s@p50"],
            [
                _summary_row("prefill", prefill_samples, args.prompt_tokens),
                _summary_row("decode", decode_samples, 1),
            ],
        )
    )
    print(
        _ascii_table(
            ["stage", "allocated_MiB", "reserved_MiB", "peak_MiB", "qweight_MiB", "cache_MiB"],
            memory_rows,
        )
    )


def _find_tensor(path: Path, tensor_name: str) -> internal_gguf.ReaderTensor:
    reader = internal_gguf.GGUFReader(path)
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            return tensor
    raise ValueError(f"Tensor {tensor_name!r} was not found in {path}.")


def _run_kernel(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.device)
    tensor = _find_tensor(args.model, args.tensor_name)
    logical_shape = tuple(reversed(tensor.shape.tolist()))
    if len(logical_shape) != 2:
        raise ValueError(f"Expected a matrix tensor, got {logical_shape} for {tensor.name}.")
    out_features, in_features = logical_shape

    reference = GGUFTorchLinear(
        bits="q2_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    ).to(args.device_string).eval()
    packed = torch.from_numpy(np.array(tensor.data, dtype=np.uint8, copy=True, order="C"))
    reference.qweight.copy_(packed.to(args.device_string))

    kernel = GGUFTritonKernel(
        bits="q2_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    ).to(args.device_string).eval()
    kernel.load_state_dict(reference.state_dict(), strict=True)

    torch.manual_seed(args.seed)
    x = torch.randn((args.rows, in_features), device=args.device_string, dtype=torch.float16)
    with torch.inference_mode():
        if args.prime_prefill_cache:
            kernel(torch.zeros((8, in_features), device=args.device_string, dtype=torch.float16))
        expected = reference._forward_dequant_matmul(x)
        actual = kernel(x)
        difference = (expected.float() - actual.float()).abs()
        samples = _event_samples(kernel_forward := lambda: kernel(x), warmup=args.warmup, iterations=args.iterations)

        if args.capture:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            _profile_range("prism_q2_0_kernel", kernel_forward, args.profile_kernel)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

    cache_bytes = _triton_cache_bytes([kernel])
    properties = torch.cuda.get_device_properties(args.device)
    print(
        f"model={args.model} tensor={tensor.name} qtype={tensor.tensor_type.name} rows={args.rows} "
        f"shape={out_features}x{in_features} dtype=fp16 device={args.device} gpu={properties.name!r} "
        f"capability={properties.major}.{properties.minor} sms={properties.multi_processor_count} "
        f"warmup={args.warmup} iterations={args.iterations}"
    )
    print(
        _ascii_table(
            ["path", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "calls/s@p50"],
            [[
                "triton",
                f"{statistics.mean(samples):.4f}",
                f"{statistics.median(samples):.4f}",
                f"{_percentile(samples, 0.95):.4f}",
                f"{min(samples):.4f}",
                f"{max(samples):.4f}",
                f"{1000.0 / statistics.median(samples):.2f}",
            ]],
        )
    )
    print(
        f"correctness_mae={difference.mean().item():.8f} correctness_max_abs={difference.max().item():.8f} "
        f"qweight_MiB={_tensor_bytes(kernel.qweight) / (1024**2):.3f} cache_MiB={cache_bytes / (1024**2):.3f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark and profile Prism Q2_0 model or kernel inference.")
    parser.add_argument("--mode", choices=("model", "kernel"), default="model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--profile-prefill", type=int, default=2)
    parser.add_argument("--profile-decode", type=int, default=8)
    parser.add_argument("--release-prefill-cache", action="store_true")
    parser.add_argument("--tensor-name", default="blk.0.attn_q.weight")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--profile-kernel", type=int, default=3)
    parser.add_argument("--prime-prefill-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.device_string = f"cuda:{args.device}"
    return args


if __name__ == "__main__":
    _run_kernel(args) if (args := _parse_args()).mode == "kernel" else _run_model(args)
