#!/usr/bin/env python3

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


def _extract_requested_gpu(argv: list[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg == "--gpu" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
    return None


_requested_gpu = _extract_requested_gpu(sys.argv[1:])
if _requested_gpu is not None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_requested_gpu))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from tabulate import tabulate
from transformers import AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.utils.torch import torch_empty_cache


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_MODELS_DIR = REPO_ROOT / "tests" / "models"
if str(TESTS_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_MODELS_DIR))

from test_llama3_2 import TestLlama3_2  # noqa: E402


DEFAULT_PROMPT = (
    "Write a detailed but compact explanation of how attention works in a transformer model, "
    "including self-attention, key/query/value projections, and why KV caching helps autoregressive decoding."
)

DEFAULT_BATCH_PROMPTS = [
    (
        "Explain why grouped-query attention reduces KV-cache memory pressure during autoregressive decoding, "
        "and describe the main tradeoff compared with full multi-head attention."
    ),
    (
        "Summarize how rotary position embeddings are applied inside a decoder-only transformer and why they can "
        "generalize better to longer contexts than absolute learned positions."
    ),
    (
        "Describe the main differences between post-training quantization and quantization-aware training for large "
        "language models, including the practical impact on deployment."
    ),
    (
        "Write a compact explanation of how FlashAttention reduces memory traffic, and clarify when it can still be "
        "slower than an SDPA-based path in real inference workloads."
    ),
]

DEFAULT_STREAM_PROMPT_TARGETS = [256, 1024, 2048, 4096]
DEFAULT_STREAM_TOPICS = [
    "attention kernel dispatch for quantized decoder-only models",
    "kv-cache memory pressure under long-context generation",
    "continuous batching for mixed prompt-length inference traffic",
    "prefix-sharing opportunities in repeated system prompts",
    "prefill versus decode scheduling tradeoffs in online serving",
    "how prompt padding hurts small-batch throughput",
    "batch admission control when request arrivals are bursty",
    "memory fragmentation and page allocation for kv caches",
]
STREAM_SHARED_PREFIX_SENTENCE = (
    "You are producing a technical analysis of transformer inference serving, with emphasis on quantized decoding, "
    "attention kernels, kv-cache reuse, request scheduling, and latency-throughput tradeoffs. "
)
STREAM_FILLER_SENTENCE = (
    "Discuss practical implications for prefill, decode, memory bandwidth, batching policy, and cache reuse in detail. "
)


@dataclass
class BenchmarkResult:
    mode: str
    batch_size: int
    requested_attn_impl: str
    resolved_attn_impl: str
    new_tokens_per_request: int
    total_new_tokens: int
    latency_s: float
    total_toks_per_s: float
    baseline_reserved_gib: float
    peak_reserved_gib: float
    delta_peak_reserved_gib: float
    baseline_allocated_gib: float
    peak_allocated_gib: float
    delta_peak_allocated_gib: float


@dataclass
class StreamRequest:
    request_id: str
    prompt: str
    arrival_s: float
    prompt_tokens: int


@dataclass
class StreamBenchmarkResult:
    mode: str
    requested_attn_impl: str
    resolved_attn_impl: str
    request_count: int
    max_new_tokens_per_request: int
    arrival_gap_ms: float
    prompt_targets: str
    makespan_s: float
    reqs_per_s: float
    total_toks_per_s: float
    latency_p50_s: float
    latency_p95_s: float
    queue_p50_s: float
    queue_p95_s: float
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    peak_reserved_gib: float
    peak_allocated_gib: float


def bytes_to_gib(value: int) -> float:
    return value / (1024 ** 3)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def prompt_targets_str(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_default_artifact_dir() -> Path:
    return REPO_ROOT / "benchmark_artifacts" / f"llama3_2_1b_gptq_full_{now_stamp()}"


def ensure_empty_or_new_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def cuda_sync(device_index: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_index)


def cleanup_cuda() -> None:
    gc.collect()
    torch_empty_cache()


def quantize_once(artifact_dir: Path) -> None:
    test = TestLlama3_2(methodName="test_llama3_2")
    test.PIN_CUDA_DEVICE = 0
    test.SAVE_PATH = str(artifact_dir)
    test.DELETE_QUANTIZED_MODEL = False
    test.LOAD_BACKEND = BACKEND.MARLIN
    test.USE_FLASH_ATTN = True

    model, tokenizer, _ = test.quantModel(
        test.NATIVE_MODEL_ID,
        batch_size=test.QUANT_BATCH_SIZE,
        trust_remote_code=test.TRUST_REMOTE_CODE,
        dtype=test.TORCH_DTYPE,
        need_eval=False,
        call_perform_post_quant_validation=False,
    )

    del tokenizer
    del model
    cleanup_cuda()


def load_quantized_model(artifact_dir: Path, attn_implementation: str):
    model = GPTQModel.load(
        str(artifact_dir),
        trust_remote_code=False,
        backend=BACKEND.MARLIN,
        device_map={"": "cuda:0"},
        attn_implementation=attn_implementation,
    )
    return model, model.tokenizer


def build_shared_prefix(decoder, target_tokens: int) -> str:
    text = ""
    while len(decoder(text, add_special_tokens=True)["input_ids"]) < target_tokens:
        text += STREAM_SHARED_PREFIX_SENTENCE
    return text


def build_prompt_to_target_tokens(decoder, shared_prefix: str, unique_suffix: str, target_tokens: int) -> str:
    text = shared_prefix + unique_suffix
    while len(decoder(text, add_special_tokens=True)["input_ids"]) < target_tokens:
        text += STREAM_FILLER_SENTENCE
    return text


def build_stream_workload(
    artifact_dir: Path,
    request_count: int,
    arrival_gap_ms: float,
    prompt_targets: list[int],
    shared_prefix_tokens: int,
) -> list[StreamRequest]:
    decoder = AutoTokenizer.from_pretrained(str(artifact_dir), trust_remote_code=False)
    shared_prefix = build_shared_prefix(decoder, shared_prefix_tokens)
    requests = []
    for index in range(request_count):
        target_tokens = prompt_targets[index % len(prompt_targets)]
        topic = DEFAULT_STREAM_TOPICS[index % len(DEFAULT_STREAM_TOPICS)]
        unique_suffix = (
            f"Request {index}: provide a compact but precise explanation of {topic}. "
            f"Include a comparison against request id {index} traffic behavior, and keep the answer deterministic. "
        )
        prompt = build_prompt_to_target_tokens(
            decoder,
            shared_prefix,
            unique_suffix,
            max(target_tokens, shared_prefix_tokens + 32),
        )
        prompt_tokens = len(decoder(prompt, add_special_tokens=True)["input_ids"])
        requests.append(
            StreamRequest(
                request_id=f"req_{index}",
                prompt=prompt,
                arrival_s=(arrival_gap_ms / 1000.0) * index,
                prompt_tokens=prompt_tokens,
            )
        )
    return requests


def base_tokenizer(tokenizer):
    return getattr(tokenizer, "tokenizer", tokenizer)


def prepare_inputs(tokenizer, prompts: list[str], device) -> tuple[dict, int, int, int]:
    decoder = base_tokenizer(tokenizer)
    original_padding_side = getattr(decoder, "padding_side", "right")
    decoder.padding_side = "left"
    inputs = decoder(prompts, return_tensors="pt", padding=True).to(device)
    decoder.padding_side = original_padding_side
    padded_prompt_len = int(inputs["input_ids"].shape[-1])
    pad_token_id = decoder.pad_token_id if decoder.pad_token_id is not None else decoder.eos_token_id
    eos_token_id = -1
    return inputs, padded_prompt_len, pad_token_id, eos_token_id


def prepare_cb_inputs(tokenizer, prompts: list[str]) -> tuple[list[list[int]], int, int]:
    decoder = base_tokenizer(tokenizer)
    encoded = decoder(prompts, add_special_tokens=True)
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("Continuous batching inputs must be a non-empty list of token-id lists.")
    pad_token_id = decoder.pad_token_id if decoder.pad_token_id is not None else decoder.eos_token_id
    eos_token_id = -1
    return [list(ids) for ids in input_ids], pad_token_id, eos_token_id


def run_generate(
    model,
    inputs: dict,
    *,
    min_new_tokens: int,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: int | None,
    cache_implementation: str | None = None,
):
    kwargs = dict(inputs)
    kwargs.update(
        {
            "do_sample": False,
            "num_beams": 1,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
    )
    if cache_implementation is not None:
        kwargs["cache_implementation"] = cache_implementation
    return model.generate(**kwargs)


def sequence_batch_from_generate_output(output: torch.Tensor) -> torch.Tensor:
    if output.dim() == 3:
        return output[0]
    if output.dim() == 2:
        return output
    raise ValueError(f"Unexpected generate output shape: {tuple(output.shape)}")


def collect_cb_results(manager, request_count: int) -> list:
    results = []
    while len(results) < request_count:
        result = manager.get_result(timeout=1)
        if result is not None:
            results.append(result)
    return results


def evict_cb_results(manager, results: list) -> None:
    for item in results:
        manager.evict_request_from_cache(item.request_id)


def benchmark_paged_mode(
    artifact_dir: Path,
    prompts: list[str],
    *,
    mode_name: str,
    attn_implementation: str,
    warmup_tokens: int,
    max_new_tokens: int,
) -> BenchmarkResult:
    model, tokenizer = load_quantized_model(artifact_dir, attn_implementation=attn_implementation)
    model.eval()
    device_index = 0

    prompt_token_lists, pad_token_id, eos_token_id = prepare_cb_inputs(tokenizer, prompts)
    generation_config = copy.deepcopy(model.model.generation_config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.pad_token_id = pad_token_id
    generation_config.eos_token_id = eos_token_id

    with model.model.continuous_batching_context_manager(
        generation_config=generation_config,
        manual_eviction=True,
        block=True,
        timeout=30,
        use_async_batching=False,
    ) as manager:
        warmup_results = []
        if warmup_tokens > 0:
            manager.add_requests(prompt_token_lists, max_new_tokens=warmup_tokens)
            warmup_results = collect_cb_results(manager, len(prompt_token_lists))
            evict_cb_results(manager, warmup_results)
            cuda_sync(device_index)

        baseline_reserved = torch.cuda.memory_reserved(device_index)
        baseline_allocated = torch.cuda.memory_allocated(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)

        started = time.perf_counter()
        manager.add_requests(prompt_token_lists, max_new_tokens=max_new_tokens)
        measured_results = collect_cb_results(manager, len(prompt_token_lists))
        cuda_sync(device_index)
        elapsed = time.perf_counter() - started

        peak_reserved = torch.cuda.max_memory_reserved(device_index)
        peak_allocated = torch.cuda.max_memory_allocated(device_index)

    batch_size = len(measured_results)
    if batch_size == 0:
        raise ValueError("Continuous batching returned no results.")
    new_tokens_per_request = len(measured_results[0].generated_tokens)
    total_new_tokens = sum(len(item.generated_tokens) for item in measured_results)
    resolved_attn_impl = str(getattr(model.config, "_attn_implementation", attn_implementation))

    del measured_results
    del warmup_results
    del tokenizer
    del model
    cleanup_cuda()

    total_toks_per_s = float(total_new_tokens) / elapsed if elapsed > 0 else 0.0
    return BenchmarkResult(
        mode=mode_name,
        batch_size=batch_size,
        requested_attn_impl=attn_implementation,
        resolved_attn_impl=resolved_attn_impl,
        new_tokens_per_request=new_tokens_per_request,
        total_new_tokens=total_new_tokens,
        latency_s=elapsed,
        total_toks_per_s=total_toks_per_s,
        baseline_reserved_gib=bytes_to_gib(baseline_reserved),
        peak_reserved_gib=bytes_to_gib(peak_reserved),
        delta_peak_reserved_gib=bytes_to_gib(max(peak_reserved - baseline_reserved, 0)),
        baseline_allocated_gib=bytes_to_gib(baseline_allocated),
        peak_allocated_gib=bytes_to_gib(peak_allocated),
        delta_peak_allocated_gib=bytes_to_gib(max(peak_allocated - baseline_allocated, 0)),
    )


def benchmark_paged_mode_subprocess(
    artifact_dir: Path,
    prompts: list[str],
    *,
    mode_name: str,
    attn_implementation: str,
    warmup_tokens: int,
    max_new_tokens: int,
    gpu: int,
) -> BenchmarkResult:
    with tempfile.NamedTemporaryFile(prefix="paged_benchmark_", suffix=".json", delete=False) as handle:
        result_path = Path(handle.name)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--gpu",
        str(gpu),
        "--internal-paged-result-path",
        str(result_path),
        "--artifact-dir",
        str(artifact_dir),
        "--internal-prompts-json",
        json.dumps(prompts),
        "--internal-mode-name",
        mode_name,
        "--internal-attn-implementation",
        attn_implementation,
        "--internal-warmup-tokens",
        str(warmup_tokens),
        "--internal-max-new-tokens",
        str(max_new_tokens),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Paged benchmark subprocess failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return BenchmarkResult(**payload)
    finally:
        result_path.unlink(missing_ok=True)


def benchmark_mode(
    artifact_dir: Path,
    prompts: list[str],
    *,
    mode_name: str,
    attn_implementation: str,
    warmup_tokens: int,
    max_new_tokens: int,
    use_paged_cache: bool,
    gpu: int,
) -> BenchmarkResult:
    if use_paged_cache:
        return benchmark_paged_mode_subprocess(
            artifact_dir,
            prompts,
            mode_name=mode_name,
            attn_implementation=attn_implementation,
            warmup_tokens=warmup_tokens,
            max_new_tokens=max_new_tokens,
            gpu=gpu,
        )

    model, tokenizer = load_quantized_model(artifact_dir, attn_implementation=attn_implementation)
    model.eval()
    device_index = 0

    inputs, padded_prompt_len, pad_token_id, eos_token_id = prepare_inputs(tokenizer, prompts, model.device)
    cache_impl = "paged" if use_paged_cache else None

    warmup_output = run_generate(
        model,
        inputs,
        min_new_tokens=warmup_tokens,
        max_new_tokens=warmup_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        cache_implementation=cache_impl,
    )
    del warmup_output
    cuda_sync(device_index)

    baseline_reserved = torch.cuda.memory_reserved(device_index)
    baseline_allocated = torch.cuda.memory_allocated(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)

    started = time.perf_counter()
    measured_output = run_generate(
        model,
        inputs,
        min_new_tokens=max_new_tokens,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        cache_implementation=cache_impl,
    )
    cuda_sync(device_index)
    elapsed = time.perf_counter() - started

    peak_reserved = torch.cuda.max_memory_reserved(device_index)
    peak_allocated = torch.cuda.max_memory_allocated(device_index)
    sequence_batch = sequence_batch_from_generate_output(measured_output)
    batch_size = int(sequence_batch.shape[0])
    new_tokens_per_request = int(sequence_batch.shape[-1] - padded_prompt_len)
    total_new_tokens = batch_size * new_tokens_per_request

    resolved_attn_impl = str(getattr(model.config, "_attn_implementation", attn_implementation))

    del sequence_batch
    del measured_output
    del inputs
    del tokenizer
    del model
    cleanup_cuda()

    total_toks_per_s = float(total_new_tokens) / elapsed if elapsed > 0 else 0.0
    return BenchmarkResult(
        mode=mode_name,
        batch_size=batch_size,
        requested_attn_impl=attn_implementation,
        resolved_attn_impl=resolved_attn_impl,
        new_tokens_per_request=new_tokens_per_request,
        total_new_tokens=total_new_tokens,
        latency_s=elapsed,
        total_toks_per_s=total_toks_per_s,
        baseline_reserved_gib=bytes_to_gib(baseline_reserved),
        peak_reserved_gib=bytes_to_gib(peak_reserved),
        delta_peak_reserved_gib=bytes_to_gib(max(peak_reserved - baseline_reserved, 0)),
        baseline_allocated_gib=bytes_to_gib(baseline_allocated),
        peak_allocated_gib=bytes_to_gib(peak_allocated),
        delta_peak_allocated_gib=bytes_to_gib(max(peak_allocated - baseline_allocated, 0)),
    )


def warmup_static_generate(model, tokenizer, prompt: str, warmup_tokens: int) -> None:
    if warmup_tokens <= 0:
        return
    inputs, _, pad_token_id, eos_token_id = prepare_inputs(tokenizer, [prompt], model.device)
    output = run_generate(
        model,
        inputs,
        min_new_tokens=warmup_tokens,
        max_new_tokens=warmup_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    del output
    del inputs
    cuda_sync(0)


def make_stream_result(
    *,
    mode_name: str,
    requested_attn_impl: str,
    resolved_attn_impl: str,
    request_count: int,
    max_new_tokens_per_request: int,
    arrival_gap_ms: float,
    prompt_targets: list[int],
    makespan_s: float,
    latencies: list[float],
    queue_delays: list[float],
    ttfts: list[float] | None,
    peak_reserved: int,
    peak_allocated: int,
) -> StreamBenchmarkResult:
    total_new_tokens = request_count * max_new_tokens_per_request
    return StreamBenchmarkResult(
        mode=mode_name,
        requested_attn_impl=requested_attn_impl,
        resolved_attn_impl=resolved_attn_impl,
        request_count=request_count,
        max_new_tokens_per_request=max_new_tokens_per_request,
        arrival_gap_ms=arrival_gap_ms,
        prompt_targets=prompt_targets_str(prompt_targets),
        makespan_s=makespan_s,
        reqs_per_s=(request_count / makespan_s) if makespan_s > 0 else 0.0,
        total_toks_per_s=(total_new_tokens / makespan_s) if makespan_s > 0 else 0.0,
        latency_p50_s=percentile(latencies, 0.50),
        latency_p95_s=percentile(latencies, 0.95),
        queue_p50_s=percentile(queue_delays, 0.50),
        queue_p95_s=percentile(queue_delays, 0.95),
        ttft_p50_s=None if not ttfts else percentile(ttfts, 0.50),
        ttft_p95_s=None if not ttfts else percentile(ttfts, 0.95),
        peak_reserved_gib=bytes_to_gib(peak_reserved),
        peak_allocated_gib=bytes_to_gib(peak_allocated),
    )


def benchmark_stream_static_mode(
    artifact_dir: Path,
    workload: list[StreamRequest],
    *,
    mode_name: str,
    attn_implementation: str,
    warmup_tokens: int,
    max_new_tokens: int,
    arrival_gap_ms: float,
    prompt_targets: list[int],
) -> StreamBenchmarkResult:
    model, tokenizer = load_quantized_model(artifact_dir, attn_implementation=attn_implementation)
    model.eval()
    device_index = 0

    warmup_static_generate(model, tokenizer, workload[0].prompt, warmup_tokens)
    torch.cuda.reset_peak_memory_stats(device_index)

    pending: list[StreamRequest] = []
    next_index = 0
    latencies: list[float] = []
    queue_delays: list[float] = []
    completion_times: list[float] = []

    started_at = time.perf_counter()
    while len(completion_times) < len(workload):
        now_rel = time.perf_counter() - started_at
        while next_index < len(workload) and workload[next_index].arrival_s <= now_rel:
            pending.append(workload[next_index])
            next_index += 1

        if not pending:
            if next_index >= len(workload):
                break
            sleep_s = max(workload[next_index].arrival_s - now_rel, 0.0)
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.01))
            continue

        batch_requests = pending
        pending = []
        batch_start_rel = time.perf_counter() - started_at
        prompts = [request.prompt for request in batch_requests]
        inputs, _, pad_token_id, eos_token_id = prepare_inputs(tokenizer, prompts, model.device)
        output = run_generate(
            model,
            inputs,
            min_new_tokens=max_new_tokens,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        del output
        del inputs
        cuda_sync(device_index)
        batch_end_rel = time.perf_counter() - started_at

        for request in batch_requests:
            latencies.append(batch_end_rel - request.arrival_s)
            queue_delays.append(max(batch_start_rel - request.arrival_s, 0.0))
            completion_times.append(batch_end_rel)

    peak_reserved = torch.cuda.max_memory_reserved(device_index)
    peak_allocated = torch.cuda.max_memory_allocated(device_index)
    resolved_attn_impl = str(getattr(model.config, "_attn_implementation", attn_implementation))

    del tokenizer
    del model
    cleanup_cuda()

    makespan_s = max(completion_times) if completion_times else 0.0
    return make_stream_result(
        mode_name=mode_name,
        requested_attn_impl=attn_implementation,
        resolved_attn_impl=resolved_attn_impl,
        request_count=len(workload),
        max_new_tokens_per_request=max_new_tokens,
        arrival_gap_ms=arrival_gap_ms,
        prompt_targets=prompt_targets,
        makespan_s=makespan_s,
        latencies=latencies,
        queue_delays=queue_delays,
        ttfts=None,
        peak_reserved=peak_reserved,
        peak_allocated=peak_allocated,
    )


def benchmark_stream_paged_mode(
    artifact_dir: Path,
    workload: list[StreamRequest],
    *,
    mode_name: str,
    attn_implementation: str,
    warmup_tokens: int,
    max_new_tokens: int,
    arrival_gap_ms: float,
    prompt_targets: list[int],
    scheduler_name: str,
    use_async_batching: bool,
) -> StreamBenchmarkResult:
    model, tokenizer = load_quantized_model(artifact_dir, attn_implementation=attn_implementation)
    model.eval()
    device_index = 0

    prompt_token_lists, pad_token_id, eos_token_id = prepare_cb_inputs(tokenizer, [request.prompt for request in workload])
    generation_config = copy.deepcopy(model.model.generation_config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.pad_token_id = pad_token_id
    generation_config.eos_token_id = eos_token_id
    generation_config.scheduler = scheduler_name

    with model.model.continuous_batching_context_manager(
        generation_config=generation_config,
        manual_eviction=True,
        block=True,
        timeout=30,
        use_async_batching=use_async_batching,
        allow_block_sharing=True,
    ) as manager:
        if warmup_tokens > 0:
            manager.add_request(
                prompt_token_lists[0],
                request_id="warmup",
                max_new_tokens=warmup_tokens,
                record_timestamps=True,
            )
            warmup_result = collect_cb_results(manager, 1)[0]
            manager.evict_request_from_cache(warmup_result.request_id)
            del warmup_result
            cuda_sync(device_index)

        torch.cuda.reset_peak_memory_stats(device_index)

        next_index = 0
        latencies: list[float] = []
        queue_delays: list[float] = []
        ttfts: list[float] = []
        completion_times: list[float] = []
        started_at_abs = time.perf_counter()

        while len(completion_times) < len(workload):
            now_rel = time.perf_counter() - started_at_abs
            while next_index < len(workload) and workload[next_index].arrival_s <= now_rel:
                request = workload[next_index]
                manager.add_request(
                    prompt_token_lists[next_index],
                    request_id=request.request_id,
                    max_new_tokens=max_new_tokens,
                    record_timestamps=True,
                )
                next_index += 1

            result = manager.get_result(timeout=0.01)
            if result is None:
                if next_index < len(workload):
                    sleep_s = max(workload[next_index].arrival_s - (time.perf_counter() - started_at_abs), 0.0)
                    if sleep_s > 0:
                        time.sleep(min(sleep_s, 0.01))
                continue

            if result.error is not None:
                raise RuntimeError(f"Continuous batching request failed: {result.request_id}: {result.error}")

            completion_times.append(result.lifespan[1] - started_at_abs)
            latencies.append(result.lifespan[1] - result.created_time)
            queue_delays.append(max(result.lifespan[0] - result.created_time, 0.0))
            if result.timestamps:
                ttfts.append(result.timestamps[0] - result.created_time)
            manager.evict_request_from_cache(result.request_id)

        peak_reserved = torch.cuda.max_memory_reserved(device_index)
        peak_allocated = torch.cuda.max_memory_allocated(device_index)

    resolved_attn_impl = str(getattr(model.config, "_attn_implementation", attn_implementation))

    del tokenizer
    del model
    cleanup_cuda()

    makespan_s = max(completion_times) if completion_times else 0.0
    return make_stream_result(
        mode_name=mode_name,
        requested_attn_impl=attn_implementation,
        resolved_attn_impl=resolved_attn_impl,
        request_count=len(workload),
        max_new_tokens_per_request=max_new_tokens,
        arrival_gap_ms=arrival_gap_ms,
        prompt_targets=prompt_targets,
        makespan_s=makespan_s,
        latencies=latencies,
        queue_delays=queue_delays,
        ttfts=ttfts,
        peak_reserved=peak_reserved,
        peak_allocated=peak_allocated,
    )


def benchmark_stream_paged_mode_subprocess(
    artifact_dir: Path,
    workload: list[StreamRequest],
    *,
    mode_name: str,
    attn_implementation: str,
    warmup_tokens: int,
    max_new_tokens: int,
    arrival_gap_ms: float,
    prompt_targets: list[int],
    scheduler_name: str,
    use_async_batching: bool,
    gpu: int,
) -> StreamBenchmarkResult:
    with tempfile.NamedTemporaryFile(prefix="stream_workload_", suffix=".json", delete=False) as workload_handle:
        workload_path = Path(workload_handle.name)
    with tempfile.NamedTemporaryFile(prefix="stream_result_", suffix=".json", delete=False) as result_handle:
        result_path = Path(result_handle.name)

    workload_payload = {
        "arrival_gap_ms": arrival_gap_ms,
        "prompt_targets": prompt_targets,
        "scheduler_name": scheduler_name,
        "use_async_batching": use_async_batching,
        "requests": [asdict(item) for item in workload],
    }
    workload_path.write_text(json.dumps(workload_payload), encoding="utf-8")

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--gpu",
        str(gpu),
        "--artifact-dir",
        str(artifact_dir),
        "--internal-stream-result-path",
        str(result_path),
        "--internal-stream-workload-path",
        str(workload_path),
        "--internal-mode-name",
        mode_name,
        "--internal-attn-implementation",
        attn_implementation,
        "--internal-warmup-tokens",
        str(warmup_tokens),
        "--internal-max-new-tokens",
        str(max_new_tokens),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    try:
        if completed.returncode != 0:
            raise RuntimeError(
                "Stream paged benchmark subprocess failed.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return StreamBenchmarkResult(**payload)
    finally:
        workload_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def benchmark_stream_mode(
    artifact_dir: Path,
    workload: list[StreamRequest],
    *,
    mode_name: str,
    attn_implementation: str,
    warmup_tokens: int,
    max_new_tokens: int,
    use_paged_cache: bool,
    arrival_gap_ms: float,
    prompt_targets: list[int],
    scheduler_name: str,
    use_async_batching: bool,
    gpu: int,
) -> StreamBenchmarkResult:
    if use_paged_cache:
        return benchmark_stream_paged_mode_subprocess(
            artifact_dir,
            workload,
            mode_name=mode_name,
            attn_implementation=attn_implementation,
            warmup_tokens=warmup_tokens,
            max_new_tokens=max_new_tokens,
            arrival_gap_ms=arrival_gap_ms,
            prompt_targets=prompt_targets,
            scheduler_name=scheduler_name,
            use_async_batching=use_async_batching,
            gpu=gpu,
        )

    return benchmark_stream_static_mode(
        artifact_dir,
        workload,
        mode_name=mode_name,
        attn_implementation=attn_implementation,
        warmup_tokens=warmup_tokens,
        max_new_tokens=max_new_tokens,
        arrival_gap_ms=arrival_gap_ms,
        prompt_targets=prompt_targets,
    )


def render_ascii_table(results: list[BenchmarkResult]) -> str:
    rows = []
    for item in results:
        rows.append(
            [
                item.mode,
                item.batch_size,
                item.requested_attn_impl,
                item.resolved_attn_impl,
                item.new_tokens_per_request,
                item.total_new_tokens,
                f"{item.latency_s:.3f}",
                f"{item.total_toks_per_s:.2f}",
                f"{item.peak_reserved_gib:.2f}",
                f"{item.delta_peak_reserved_gib:.2f}",
                f"{item.peak_allocated_gib:.2f}",
                f"{item.delta_peak_allocated_gib:.2f}",
            ]
        )
    headers = [
        "mode",
        "batch",
        "requested_attn",
        "resolved_attn",
        "new_tokens_each",
        "total_new_tokens",
        "latency_s",
        "total_tok_s",
        "peak_reserved_gib",
        "delta_reserved_gib",
        "peak_alloc_gib",
        "delta_alloc_gib",
    ]
    return tabulate(rows, headers=headers, tablefmt="grid")


def render_stream_ascii_table(results: list[StreamBenchmarkResult]) -> str:
    rows = []
    for item in results:
        rows.append(
            [
                item.mode,
                item.request_count,
                item.max_new_tokens_per_request,
                f"{item.arrival_gap_ms:.0f}",
                item.prompt_targets,
                f"{item.makespan_s:.3f}",
                f"{item.reqs_per_s:.2f}",
                f"{item.total_toks_per_s:.2f}",
                f"{item.latency_p50_s:.3f}",
                f"{item.latency_p95_s:.3f}",
                f"{item.queue_p50_s:.3f}",
                f"{item.queue_p95_s:.3f}",
                "n/a" if item.ttft_p50_s is None else f"{item.ttft_p50_s:.3f}",
                "n/a" if item.ttft_p95_s is None else f"{item.ttft_p95_s:.3f}",
                f"{item.peak_allocated_gib:.2f}",
                f"{item.peak_reserved_gib:.2f}",
            ]
        )
    headers = [
        "mode",
        "requests",
        "new_tok_each",
        "arrival_ms",
        "prompt_targets",
        "makespan_s",
        "req_s",
        "tok_s",
        "lat_p50_s",
        "lat_p95_s",
        "queue_p50_s",
        "queue_p95_s",
        "ttft_p50_s",
        "ttft_p95_s",
        "peak_alloc_gib",
        "peak_reserved_gib",
    ]
    return tabulate(rows, headers=headers, tablefmt="grid")


def parse_batch_sizes(raw_value: str) -> list[int]:
    values = []
    for piece in raw_value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError(f"Batch sizes must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one batch size must be provided.")
    return values


def parse_int_list(raw_value: str) -> list[int]:
    values = []
    for piece in raw_value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError(f"Values must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one positive integer must be provided.")
    return values


def prompts_for_batch(batch_size: int, single_prompt: str | None = None) -> list[str]:
    if batch_size == 1 and single_prompt:
        return [single_prompt]
    if batch_size > len(DEFAULT_BATCH_PROMPTS):
        raise ValueError(
            f"Batch size {batch_size} requested, but only {len(DEFAULT_BATCH_PROMPTS)} distinct default prompts are available."
        )
    return DEFAULT_BATCH_PROMPTS[:batch_size]


def results_json_path(artifact_dir: Path, batch_sizes: list[int]) -> Path:
    if batch_sizes == [1]:
        return artifact_dir / "attention_benchmark_results.json"
    return artifact_dir / "attention_benchmark_batch_results.json"


def stream_results_json_path(artifact_dir: Path) -> Path:
    return artifact_dir / "attention_benchmark_stream_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize Llama-3.2-1B-Instruct and benchmark attention modes.")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index to pin via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Directory to store quantized artifacts.")
    parser.add_argument(
        "--scenario",
        choices=["closed_batch", "serve_stream"],
        default="closed_batch",
        help="Benchmark scenario to run.",
    )
    parser.add_argument(
        "--reuse-artifact",
        action="store_true",
        help="Reuse an existing artifact directory instead of requiring a new quantization output directory.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1",
        help="Comma-separated batch sizes to benchmark, for example `1,2,4`.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt used for the 512-token benchmark.")
    parser.add_argument("--warmup-tokens", type=int, default=32, help="Warmup generation length before measuring.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Measured generation length.")
    parser.add_argument(
        "--stream-request-count",
        type=int,
        default=12,
        help="Number of requests in serve_stream mode.",
    )
    parser.add_argument(
        "--stream-arrival-ms",
        type=float,
        default=75.0,
        help="Inter-arrival gap in milliseconds for serve_stream mode.",
    )
    parser.add_argument(
        "--stream-prompt-targets",
        type=str,
        default=prompt_targets_str(DEFAULT_STREAM_PROMPT_TARGETS),
        help="Comma-separated prompt token targets for serve_stream mode.",
    )
    parser.add_argument(
        "--stream-shared-prefix-tokens",
        type=int,
        default=192,
        help="Approximate shared-prefix token count for serve_stream mode.",
    )
    parser.add_argument(
        "--stream-scheduler",
        type=str,
        default="prefill_first",
        help="Continuous batching scheduler for paged serve_stream mode.",
    )
    parser.add_argument(
        "--stream-use-async-batching",
        action="store_true",
        help="Enable async batching for paged serve_stream mode.",
    )
    parser.add_argument("--internal-paged-result-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-stream-result-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-stream-workload-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-prompts-json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-mode-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-attn-implementation", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-warmup-tokens", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-max-new-tokens", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.internal_paged_result_path is not None:
        prompts = json.loads(args.internal_prompts_json)
        result = benchmark_paged_mode(
            args.artifact_dir.resolve(),
            prompts,
            mode_name=args.internal_mode_name,
            attn_implementation=args.internal_attn_implementation,
            warmup_tokens=args.internal_warmup_tokens,
            max_new_tokens=args.internal_max_new_tokens,
        )
        args.internal_paged_result_path.write_text(json.dumps(asdict(result)), encoding="utf-8")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if args.internal_stream_result_path is not None:
        payload = json.loads(args.internal_stream_workload_path.read_text(encoding="utf-8"))
        workload = [StreamRequest(**item) for item in payload["requests"]]
        result = benchmark_stream_paged_mode(
            args.artifact_dir.resolve(),
            workload,
            mode_name=args.internal_mode_name,
            attn_implementation=args.internal_attn_implementation,
            warmup_tokens=args.internal_warmup_tokens,
            max_new_tokens=args.internal_max_new_tokens,
            arrival_gap_ms=float(payload["arrival_gap_ms"]),
            prompt_targets=[int(value) for value in payload["prompt_targets"]],
            scheduler_name=str(payload["scheduler_name"]),
            use_async_batching=bool(payload["use_async_batching"]),
        )
        args.internal_stream_result_path.write_text(json.dumps(asdict(result)), encoding="utf-8")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    stream_prompt_targets = parse_int_list(args.stream_prompt_targets)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    artifact_dir = args.artifact_dir or make_default_artifact_dir()
    artifact_dir = artifact_dir.resolve()
    if artifact_dir.exists():
        if not args.reuse_artifact:
            raise FileExistsError(
                f"Artifact directory already exists: {artifact_dir}. Pass --reuse-artifact to skip requantization."
            )
    else:
        ensure_empty_or_new_dir(artifact_dir)

    physical_gpu_index = args.gpu
    logical_gpu_index = 0
    gpu_name = torch.cuda.get_device_name(logical_gpu_index)

    print(f"Using physical GPU {physical_gpu_index} as logical cuda:{logical_gpu_index}: {gpu_name}")
    print(f"Quantized artifact directory: {artifact_dir}")
    print(f"Scenario: {args.scenario}")
    if args.scenario == "closed_batch":
        print(f"Batch sizes: {batch_sizes}")
    else:
        print(
            "Serve stream config: "
            f"requests={args.stream_request_count}, arrival_ms={args.stream_arrival_ms}, "
            f"prompt_targets={stream_prompt_targets}, shared_prefix_tokens={args.stream_shared_prefix_tokens}, "
            f"scheduler={args.stream_scheduler}, async_batching={args.stream_use_async_batching}"
        )

    if args.reuse_artifact:
        print("Reusing existing quantized artifact directory.")
    else:
        quantize_once(artifact_dir)

    modes = [
        {
            "mode_name": "sdpa",
            "attn_implementation": "sdpa",
            "use_paged_cache": False,
        },
        {
            "mode_name": "flash_attention_2",
            "attn_implementation": "flash_attention_2",
            "use_paged_cache": False,
        },
        {
            "mode_name": "paged(sdpa)",
            "attn_implementation": "sdpa",
            "use_paged_cache": True,
        },
    ]
    if args.scenario == "serve_stream":
        modes.append(
            {
                "mode_name": "paged(fa2)",
                "attn_implementation": "flash_attention_2",
                "use_paged_cache": True,
            }
        )

    if args.scenario == "closed_batch":
        results: list[BenchmarkResult] = []
        prompts_by_batch = {}
        for batch_size in batch_sizes:
            prompts = prompts_for_batch(batch_size, single_prompt=args.prompt if batch_size == 1 and batch_sizes == [1] else None)
            prompts_by_batch[str(batch_size)] = prompts
            for mode in modes:
                print(f"\nBenchmarking {mode['mode_name']} with batch={batch_size}...")
                result = benchmark_mode(
                    artifact_dir,
                    prompts,
                    mode_name=mode["mode_name"],
                    attn_implementation=mode["attn_implementation"],
                    warmup_tokens=args.warmup_tokens,
                    max_new_tokens=args.max_new_tokens,
                    use_paged_cache=mode["use_paged_cache"],
                    gpu=physical_gpu_index,
                )
                results.append(result)

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_dir": str(artifact_dir),
            "physical_gpu_index": physical_gpu_index,
            "logical_gpu_index": logical_gpu_index,
            "gpu_name": gpu_name,
            "model_id": TestLlama3_2.NATIVE_MODEL_ID,
            "scenario": args.scenario,
            "batch_sizes": batch_sizes,
            "prompts_by_batch": prompts_by_batch,
            "warmup_tokens": args.warmup_tokens,
            "max_new_tokens": args.max_new_tokens,
            "results": [asdict(item) for item in results],
            "ascii_table": render_ascii_table(results),
        }

        json_path = results_json_path(artifact_dir, batch_sizes)
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("\n" + metadata["ascii_table"])
        print(f"\nResults JSON: {json_path}")
        return 0

    workload = build_stream_workload(
        artifact_dir,
        request_count=args.stream_request_count,
        arrival_gap_ms=args.stream_arrival_ms,
        prompt_targets=stream_prompt_targets,
        shared_prefix_tokens=args.stream_shared_prefix_tokens,
    )
    stream_results: list[StreamBenchmarkResult] = []
    for mode in modes:
        print(f"\nBenchmarking {mode['mode_name']} with serve_stream...")
        result = benchmark_stream_mode(
            artifact_dir,
            workload,
            mode_name=mode["mode_name"],
            attn_implementation=mode["attn_implementation"],
            warmup_tokens=args.warmup_tokens,
            max_new_tokens=args.max_new_tokens,
            use_paged_cache=mode["use_paged_cache"],
            arrival_gap_ms=args.stream_arrival_ms,
            prompt_targets=stream_prompt_targets,
            scheduler_name=args.stream_scheduler,
            use_async_batching=args.stream_use_async_batching,
            gpu=physical_gpu_index,
        )
        stream_results.append(result)

    stream_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "physical_gpu_index": physical_gpu_index,
        "logical_gpu_index": logical_gpu_index,
        "gpu_name": gpu_name,
        "model_id": TestLlama3_2.NATIVE_MODEL_ID,
        "scenario": args.scenario,
        "stream_request_count": args.stream_request_count,
        "stream_arrival_ms": args.stream_arrival_ms,
        "stream_prompt_targets": stream_prompt_targets,
        "stream_shared_prefix_tokens": args.stream_shared_prefix_tokens,
        "stream_scheduler": args.stream_scheduler,
        "stream_use_async_batching": args.stream_use_async_batching,
        "warmup_tokens": args.warmup_tokens,
        "max_new_tokens": args.max_new_tokens,
        "workload": [asdict(item) for item in workload],
        "results": [asdict(item) for item in stream_results],
        "ascii_table": render_stream_ascii_table(stream_results),
    }

    stream_json_path = stream_results_json_path(artifact_dir)
    stream_json_path.write_text(json.dumps(stream_metadata, indent=2), encoding="utf-8")

    print("\n" + stream_metadata["ascii_table"])
    print(f"\nResults JSON: {stream_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
