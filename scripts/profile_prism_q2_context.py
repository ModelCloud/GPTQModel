#!/usr/bin/env python3

"""Measure supported Prism Q2 eager prefill/decode scaling and a separate KV-cache memory ceiling.

The static-cache search allocates a full FP16 KV cache and executes one decode token, but it does not prefill real
tokens or extend the model's trained/configured context. Treat that result only as a single-GPU memory-capacity bound.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers
import triton
from transformers import StaticCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


DEFAULT_MODEL = Path("/monster/data/model/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf")
DEFAULT_LENGTHS = (128, 512, 2048, 8192, 32768)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


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


def _cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _mib(value: int) -> float:
    return value / (1024**2)


def _event_time(operation, /, *args, **kwargs) -> tuple[float, float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    output = operation(*args, **kwargs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end), (time.perf_counter() - host_start) * 1000.0, output


def _prefill_decode_case(
    model: torch.nn.Module,
    *,
    context_length: int,
    device: torch.device,
    prefill_iterations: int,
    decode_warmup: int,
    decode_iterations: int,
    capture: bool,
) -> tuple[list[str], int]:
    vocab_size = int(model.config.vocab_size)
    tokens = (torch.arange(context_length, device=device, dtype=torch.long) % vocab_size).unsqueeze(0)

    warm_output = model(input_ids=tokens, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(device)
    del warm_output
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    cuda_samples = []
    host_samples = []
    output = None
    for iteration in range(prefill_iterations):
        cuda_ms, host_ms, current_output = _event_time(
            model,
            input_ids=tokens,
            use_cache=True,
            logits_to_keep=1,
        )
        cuda_samples.append(cuda_ms)
        host_samples.append(host_ms)
        if iteration + 1 == prefill_iterations:
            output = current_output
        else:
            del current_output
            gc.collect()
    assert output is not None

    cache = output.past_key_values
    next_token = output.logits[:, -1:, :].argmax(dim=-1)
    cache_length = int(cache.get_seq_length())
    finite = bool(torch.isfinite(output.logits).all().item())
    allocated_mib = _mib(torch.cuda.memory_allocated(device))
    reserved_mib = _mib(torch.cuda.memory_reserved(device))
    prefill_peak_mib = _mib(torch.cuda.max_memory_allocated(device))

    def decode():
        nonlocal cache, next_token
        decode_output = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = decode_output.past_key_values
        next_token = decode_output.logits[:, -1:, :].argmax(dim=-1)
        return decode_output

    for _ in range(decode_warmup):
        output = decode()
    torch.cuda.synchronize(device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(decode_iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(decode_iterations)]
    if capture:
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"prism_q2_dynamic_decode_context_{context_length}")
    for index in range(decode_iterations):
        if capture:
            torch.cuda.nvtx.range_push(f"decode_token_{index}")
        starts[index].record()
        output = decode()
        ends[index].record()
        if capture:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    if capture:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    decode_samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    final_token = int(next_token.item())
    final_cache_length = int(cache.get_seq_length())
    peak_mib = _mib(torch.cuda.max_memory_allocated(device))

    prefill_p50 = statistics.median(cuda_samples)
    decode_p50 = statistics.median(decode_samples)
    row = [
        f"{context_length:,}",
        f"{prefill_p50:.3f}",
        f"{context_length * 1000.0 / prefill_p50:.2f}",
        f"{statistics.median(host_samples):.3f}",
        f"{decode_p50:.3f}",
        f"{1000.0 / decode_p50:.2f}",
        f"{_percentile(decode_samples, 0.95):.3f}",
        f"{allocated_mib:.2f}",
        f"{reserved_mib:.2f}",
        f"{max(prefill_peak_mib, peak_mib):.2f}",
        "yes" if finite and cache_length == context_length else "no",
    ]
    print(
        f"CASE context={context_length} prefill_p50_ms={prefill_p50:.3f} "
        f"eager_decode_p50_ms={decode_p50:.3f} cache={cache_length}->{final_cache_length} "
        f"final_token={final_token} finite={finite}"
    )

    del cache, tokens, next_token, output
    _cleanup_cuda()
    return row, final_token


def _set_cache_length(cache: StaticCache, context_length: int) -> None:
    for layer in cache.layers:
        layer.cumulative_length.fill_(context_length - 1)


def _cache_decode_trial(
    model: torch.nn.Module,
    *,
    context_length: int,
    device: torch.device,
    num_key_value_heads: int,
    head_dim: int,
) -> tuple[bool, list[str]]:
    cache = None
    output = None
    _cleanup_cuda()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    token = torch.zeros((1, 1), device=device, dtype=torch.long)
    position = torch.full((1, 1), context_length - 1, device=device, dtype=torch.long)
    try:
        cache = StaticCache(config=model.config, max_cache_len=context_length)
        cache.early_initialization(
            batch_size=1,
            num_heads=num_key_value_heads,
            head_dim=head_dim,
            dtype=torch.float16,
            device=device,
        )
        _set_cache_length(cache, context_length)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = model(
            input_ids=token,
            position_ids=position,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        end.record()
        end.synchronize()
        decode_ms = start.elapsed_time(end)
        finite = bool(torch.isfinite(output.logits).all().item())
        allocated = torch.cuda.memory_allocated(device)
        peak = torch.cuda.max_memory_allocated(device)
        free, _ = torch.cuda.mem_get_info(device)
        row = [
            f"{context_length:,}",
            "pass" if finite else "nonfinite",
            f"{decode_ms:.3f}",
            f"{_mib(allocated - baseline_allocated):.2f}",
            f"{_mib(peak):.2f}",
            f"{_mib(free):.2f}",
        ]
        success = finite
    except torch.OutOfMemoryError:
        row = [f"{context_length:,}", "OOM", "-", "-", "-", "-"]
        success = False
    finally:
        del cache, output, position, token
        _cleanup_cuda()
    print(f"CACHE_TRIAL context={context_length} status={row[1]} decode_ms={row[2]}")
    return success, row


def _search_cache_limit(
    model: torch.nn.Module,
    *,
    device: torch.device,
    configured_context: int,
    max_probe_tokens: int,
    granularity: int,
) -> tuple[list[list[str]], int, int | None]:
    num_key_value_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", 0) or model.config.hidden_size // model.config.num_attention_heads)
    rows = []

    def trial(context_length: int) -> bool:
        success, row = _cache_decode_trial(
            model,
            context_length=context_length,
            device=device,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        rows.append(row)
        return success

    low = configured_context
    if not trial(low):
        return rows, 0, low

    high = low * 2
    while high <= max_probe_tokens and trial(high):
        low = high
        high *= 2

    if high > max_probe_tokens:
        high = max_probe_tokens
        if low == high or trial(high):
            return rows, high, None

    first_oom = high
    while high - low > granularity:
        midpoint = ((low + high) // (2 * granularity)) * granularity
        midpoint = max(low + granularity, min(midpoint, high - granularity))
        if trial(midpoint):
            low = midpoint
        else:
            high = midpoint
            first_oom = min(first_oom, midpoint)
    return rows, low, high


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Prism Q2 context scaling and single-GPU cache capacity.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--prefill-iterations", type=int, default=2)
    parser.add_argument("--decode-warmup", type=int, default=2)
    parser.add_argument("--decode-iterations", type=int, default=20)
    parser.add_argument("--skip-cache-search", action="store_true")
    parser.add_argument("--capture", action="store_true", help="Capture the timed dynamic-decode iterations only.")
    parser.add_argument("--max-probe-tokens", type=int, default=1_048_576)
    parser.add_argument("--cache-search-granularity", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.prefill_iterations < 1 or args.decode_iterations < 1 or args.decode_warmup < 0:
        raise ValueError("prefill/decode iteration counts must be positive and decode warmup cannot be negative.")
    if args.cache_search_granularity < 1:
        raise ValueError("cache search granularity must be positive.")

    torch.manual_seed(20260722)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    _cleanup_cuda()
    properties = torch.cuda.get_device_properties(device)
    load_start = time.perf_counter()
    wrapper = GPTQModel.load(
        str(args.model),
        backend=BACKEND.GGUF_TRITON,
        profile="low_memory",
        device=device,
        dtype=torch.float16,
    )
    model = wrapper.model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    configured_context = int(model.config.max_position_embeddings)
    rope_parameters = dict(getattr(model.config, "rope_parameters", {}) or {})
    lengths = sorted(set(args.lengths))
    if any(length < 1 or length > configured_context for length in lengths):
        raise ValueError(f"full-prefill lengths must be in [1, {configured_context}]; actual values: {lengths}.")
    if args.capture and len(lengths) != 1:
        raise ValueError("--capture requires exactly one context length so the profiler receives one bounded range.")

    print(
        f"ENV model={args.model} backend=GGUF_TRITON format=Q2_0 bpw=2.125 dtype=fp16 batch=1 "
        f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__} "
        f"transformers={transformers.__version__} gpu={properties.name!r} "
        f"capability={properties.major}.{properties.minor} sms={properties.multi_processor_count} "
        f"memory_bytes={properties.total_memory} load_s={load_seconds:.3f}"
    )
    print(
        f"CONFIG max_position_embeddings={configured_context} rope_type={rope_parameters.get('rope_type')} "
        f"rope_factor={rope_parameters.get('factor')} "
        f"original_context={rope_parameters.get('original_max_position_embeddings')}"
    )

    speed_rows = []
    validated_context = 0
    final_tokens = {}
    with torch.inference_mode():
        for context_length in lengths:
            try:
                row, final_token = _prefill_decode_case(
                    model,
                    context_length=context_length,
                    device=device,
                    prefill_iterations=args.prefill_iterations,
                    decode_warmup=args.decode_warmup,
                    decode_iterations=args.decode_iterations,
                    capture=args.capture,
                )
            except torch.OutOfMemoryError:
                speed_rows.append([f"{context_length:,}", "OOM", "-", "-", "-", "-", "-", "-", "-", "-", "no"])
                _cleanup_cuda()
                break
            speed_rows.append(row)
            validated_context = context_length
            final_tokens[context_length] = final_token

        print(
            _ascii_table(
                [
                    "context",
                    "prefill p50 ms",
                    "prefill tok/s",
                    "host p50 ms",
                    "eager decode p50 ms",
                    "eager decode tok/s",
                    "eager decode p95 ms",
                    "alloc MiB",
                    "reserved MiB",
                    "peak MiB",
                    "valid",
                ],
                speed_rows,
            )
        )

        cache_rows = []
        cache_limit = 0
        first_oom = None
        if not args.skip_cache_search:
            cache_rows, cache_limit, first_oom = _search_cache_limit(
                model,
                device=device,
                configured_context=configured_context,
                max_probe_tokens=args.max_probe_tokens,
                granularity=args.cache_search_granularity,
            )
            print(
                _ascii_table(
                    ["cache capacity", "status", "decode ms", "cache alloc MiB", "peak MiB", "free MiB"],
                    cache_rows,
                )
            )

    head_dim = int(getattr(model.config, "head_dim", 0) or model.config.hidden_size // model.config.num_attention_heads)
    kv_bytes_per_token = (
        2
        * int(model.config.num_hidden_layers)
        * int(model.config.num_key_value_heads)
        * head_dim
        * torch.tensor([], dtype=torch.float16).element_size()
    )
    print(
        f"SUMMARY configured_context={configured_context} validated_full_prefill={validated_context} "
        f"cache_decode_capacity={cache_limit} first_oom={first_oom} "
        f"search_granularity={args.cache_search_granularity} kv_bytes_per_token={kv_bytes_per_token} "
        f"final_tokens={final_tokens}"
    )


if __name__ == "__main__":
    main()
