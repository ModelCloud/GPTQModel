#!/usr/bin/env python3
"""Laguna single-GPU prefill/decode phase throughput benchmark."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import laguna_benchmark_common as common


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

RUN_LABEL = os.environ.get("LAGUNA_PHASE_RUN_LABEL", "phase")
VLLM_GPU_MEMORY_UTILIZATION = float(
    os.environ.get("LAGUNA_VLLM_GPU_MEMORY_UTILIZATION", "0.90")
)
SGLANG_MEM_FRACTION_STATIC = float(
    os.environ.get("LAGUNA_SGLANG_MEM_FRACTION_STATIC", "0.90")
)

if not RUN_LABEL or any(
    not (character.isalnum() or character in "-_") for character in RUN_LABEL
):
    raise ValueError(
        "LAGUNA_PHASE_RUN_LABEL must contain only letters, digits, '-' or '_'."
    )
if not 0 < VLLM_GPU_MEMORY_UTILIZATION <= 1:
    raise ValueError("LAGUNA_VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1].")
if not 0 < SGLANG_MEM_FRACTION_STATIC < 1:
    raise ValueError("LAGUNA_SGLANG_MEM_FRACTION_STATIC must be in (0, 1).")


def phase_result_path(framework: str) -> Path:
    return common.OUTPUT_DIR / f"laguna_prefill_decode_{framework}_{RUN_LABEL}.json"


def is_oom_error(error: BaseException) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in message
        for marker in (
            "cuda out of memory",
            "outofmemoryerror",
            "out of memory",
            "cannot allocate memory",
            "no available memory",
        )
    )


def _as_sglang_list(outputs: Any, batch_size: int) -> list[dict[str, Any]]:
    if isinstance(outputs, dict):
        outputs = [outputs]
    if not isinstance(outputs, list) or len(outputs) != batch_size:
        raise RuntimeError(
            "SGLang returned unexpected batch shape: "
            f"type={type(outputs)!r}, "
            f"length={len(outputs) if isinstance(outputs, list) else 'n/a'}"
        )
    return outputs


def extract_sglang_phase(outputs: Any, batch_size: int) -> dict[str, Any]:
    items = _as_sglang_list(outputs, batch_size)
    output_lengths = [len(item["output_ids"]) for item in items]
    if any(length != common.OUTPUT_LEN for length in output_lengths):
        raise RuntimeError(
            f"SGLang output lengths were not all {common.OUTPUT_LEN}: "
            f"{output_lengths}"
        )

    meta = [item["meta_info"] for item in items]
    required = (
        "e2e_latency",
        "request_finished_ts",
        "decode_throughput",
    )
    missing = [
        (index, key)
        for index, item_meta in enumerate(meta)
        for key in required
        if key not in item_meta
    ]
    if missing:
        raise RuntimeError(f"SGLang phase metadata is missing fields: {missing}")

    request_finishes = [float(item["request_finished_ts"]) for item in meta]
    request_decode_seconds = [
        (common.OUTPUT_LEN - 1) / float(item["decode_throughput"]) for item in meta
    ]
    request_prefill_seconds = [
        float(item["e2e_latency"]) - decode_seconds
        for item, decode_seconds in zip(meta, request_decode_seconds, strict=True)
    ]
    frontend_first_tokens = [
        finish - decode_seconds
        for finish, decode_seconds in zip(
            request_finishes, request_decode_seconds, strict=True
        )
    ]

    prefill_seconds = max(request_prefill_seconds)
    decode_seconds = max(request_decode_seconds)
    first_token_spread_seconds = max(frontend_first_tokens) - min(frontend_first_tokens)
    request_received = [
        float(item["request_received_ts"])
        for item in meta
        if "request_received_ts" in item
    ]
    scheduler_start_spread_seconds = (
        max(request_received) - min(request_received)
        if len(request_received) == batch_size
        else 0.0
    )
    frontend_first_token_spread_seconds = max(frontend_first_tokens) - min(
        frontend_first_tokens
    )
    if prefill_seconds <= 0 or decode_seconds <= 0:
        raise RuntimeError(
            "SGLang returned non-positive phase duration: "
            f"prefill={prefill_seconds}, decode={decode_seconds}"
        )

    return {
        "output_lengths": output_lengths,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "first_token_spread_seconds": first_token_spread_seconds,
        "scheduler_start_spread_seconds": scheduler_start_spread_seconds,
        "frontend_first_token_spread_seconds": frontend_first_token_spread_seconds,
        "request_prefill_seconds_min": min(request_prefill_seconds),
        "request_prefill_seconds_max": max(request_prefill_seconds),
        "request_decode_seconds_min": min(request_decode_seconds),
        "request_decode_seconds_max": max(request_decode_seconds),
    }


def extract_vllm_phase(outputs: Any, batch_size: int) -> dict[str, Any]:
    if not isinstance(outputs, list) or len(outputs) != batch_size:
        raise RuntimeError(
            "vLLM returned unexpected batch shape: "
            f"type={type(outputs)!r}, "
            f"length={len(outputs) if isinstance(outputs, list) else 'n/a'}"
        )

    output_lengths = []
    metrics = []
    for item in outputs:
        if len(item.outputs) != 1:
            raise RuntimeError(
                f"vLLM returned {len(item.outputs)} candidates; expected one."
            )
        output_lengths.append(len(item.outputs[0].token_ids))
        if item.metrics is None:
            raise RuntimeError(
                "vLLM RequestOutput.metrics is unavailable; "
                "disable_log_stats must be False."
            )
        metrics.append(item.metrics)
    if any(length != common.OUTPUT_LEN for length in output_lengths):
        raise RuntimeError(
            f"vLLM output lengths were not all {common.OUTPUT_LEN}: "
            f"{output_lengths}"
        )

    scheduled = [float(item.scheduled_ts) for item in metrics]
    first_tokens = [float(item.first_token_ts) for item in metrics]
    last_tokens = [float(item.last_token_ts) for item in metrics]
    if any(timestamp <= 0 for timestamp in scheduled + first_tokens + last_tokens):
        raise RuntimeError(
            "vLLM returned unset phase timestamps: "
            f"scheduled={scheduled}, first={first_tokens}, last={last_tokens}"
        )

    request_prefill_seconds = [float(item.first_token_latency) for item in metrics]
    request_decode_seconds = [
        last - first for first, last in zip(first_tokens, last_tokens, strict=True)
    ]
    prefill_seconds = max(request_prefill_seconds)
    decode_seconds = max(request_decode_seconds)
    first_token_spread_seconds = max(first_tokens) - min(first_tokens)
    scheduler_start_spread_seconds = max(scheduled) - min(scheduled)
    if prefill_seconds <= 0 or decode_seconds <= 0:
        raise RuntimeError(
            "vLLM returned non-positive phase duration: "
            f"prefill={prefill_seconds}, decode={decode_seconds}"
        )

    return {
        "output_lengths": output_lengths,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "first_token_spread_seconds": first_token_spread_seconds,
        "scheduler_start_spread_seconds": scheduler_start_spread_seconds,
        "request_prefill_seconds_min": min(request_prefill_seconds),
        "request_prefill_seconds_max": max(request_prefill_seconds),
        "request_decode_seconds_min": min(request_decode_seconds),
        "request_decode_seconds_max": max(request_decode_seconds),
    }


def make_result(
    framework: str,
    target: str,
    preflight: list[dict[str, Any]],
    torch_module: Any,
) -> dict[str, Any]:
    result = common.base_result(
        framework,
        target,
        preflight,
        torch_module,
        common.package_version(framework),
    )
    result["run_label"] = RUN_LABEL
    result["benchmark"] = {
        "mode": "offline engine phase timing",
        "single_physical_gpu": True,
        "batch_semantics": "submitted request count; framework scheduler semantics are recorded below",
        "requested_batch_sizes": common.BATCH_SIZES,
        "max_resident_requests_engine_setting": None,
        "input_tokens_per_request": common.INPUT_LEN,
        "output_tokens_per_request": common.OUTPUT_LEN,
        "decode_tokens_per_request": common.OUTPUT_LEN - 1,
        "context_margin_tokens": common.CONTEXT_MARGIN,
        "max_model_length": common.MAX_MODEL_LEN,
        "temperature": 0.0,
        "ignore_eos": True,
        "prefix_cache": False,
        "warmups_per_shape": 1,
        "timed_repeats": common.REPEATS,
        "prefill_metric": (
            "aggregate input tokens / maximum per-request time to first token; "
            "first generated token belongs to prefill"
        ),
        "decode_metric": (
            "aggregate generated tokens after the first token / batch decode makespan"
        ),
        "output_metric": ("aggregate output tokens / end-to-end wall time"),
        "total_metric": ("aggregate input and output tokens / end-to-end wall time"),
        "capacity_boundary": ("CUDA OOM; no benchmark-side resident-capacity cutoff"),
        "timing_source": "engine request TTFT and decode timestamps",
        "prompt_seed": common.SEED,
        "prompt_sha256": common.prompt_sha256(common.make_prompts()),
    }
    result["rows"] = []
    result["capacity_boundary"] = None
    return result


def run_phase_rows(
    framework: str,
    generate: Callable[[int], Any],
    extract_phase: Callable[[Any, int], dict[str, Any]],
    target: str,
    result: dict[str, Any],
    output_path: Path,
) -> None:
    peak_memory_mib = int(common.query_gpu(target)["memory.used"])
    for batch_size in common.BATCH_SIZES:
        print(f"WARMUP_PHASE framework={framework} batch={batch_size}", flush=True)
        warmup_wall_start = time.perf_counter()
        try:
            warmup_outputs = generate(batch_size)
            warmup_wall_seconds = time.perf_counter() - warmup_wall_start
            warmup_phase = extract_phase(warmup_outputs, batch_size)
        except BaseException as error:
            if not is_oom_error(error):
                raise
            row = {
                "batch_size": batch_size,
                "status": "oom",
                "error": f"{type(error).__name__}: {error}",
            }
            result["rows"].append(row)
            result["capacity_boundary"] = row
            common.atomic_write_json(output_path, result)
            print(
                f"CAPACITY framework={framework} batch={batch_size} status=oom",
                flush=True,
            )
            break

        common.verify_exclusive_after_warmup(target)
        peak_memory_mib = max(
            peak_memory_mib, int(common.query_gpu(target)["memory.used"])
        )

        samples: list[dict[str, Any]] = []
        for repeat in range(common.REPEATS):
            wall_start = time.perf_counter()
            outputs = generate(batch_size)
            wall_seconds = time.perf_counter() - wall_start
            phase = extract_phase(outputs, batch_size)
            prefill_tokens = batch_size * common.INPUT_LEN
            decode_tokens = batch_size * (common.OUTPUT_LEN - 1)
            phase["wall_seconds"] = wall_seconds
            phase["prefill_tps"] = prefill_tokens / phase["prefill_seconds"]
            phase["decode_tps"] = decode_tokens / phase["decode_seconds"]
            phase["output_tps"] = batch_size * common.OUTPUT_LEN / wall_seconds
            phase["total_tps"] = (
                batch_size * (common.INPUT_LEN + common.OUTPUT_LEN) / wall_seconds
            )
            samples.append(phase)
            peak_memory_mib = max(
                peak_memory_mib, int(common.query_gpu(target)["memory.used"])
            )
            print(
                f"SAMPLE_PHASE framework={framework} batch={batch_size} "
                f"repeat={repeat + 1}/{common.REPEATS} "
                f"prefill_s={phase['prefill_seconds']:.6f} "
                f"prefill_tps={phase['prefill_tps']:.3f} "
                f"decode_s={phase['decode_seconds']:.6f} "
                f"decode_tps={phase['decode_tps']:.3f} "
                f"output_tps={phase['output_tps']:.3f} "
                f"total_tps={phase['total_tps']:.3f} "
                f"wall_s={wall_seconds:.6f}",
                flush=True,
            )

        prefill_seconds_total = sum(
            float(sample["prefill_seconds"]) for sample in samples
        )
        decode_seconds_total = sum(
            float(sample["decode_seconds"]) for sample in samples
        )
        prefill_tps_samples = [float(sample["prefill_tps"]) for sample in samples]
        decode_tps_samples = [float(sample["decode_tps"]) for sample in samples]
        row = {
            "batch_size": batch_size,
            "status": "ok",
            "input_tokens_per_request": common.INPUT_LEN,
            "output_tokens_per_request": common.OUTPUT_LEN,
            "decode_tokens_per_request": common.OUTPUT_LEN - 1,
            "warmup_wall_seconds": warmup_wall_seconds,
            "warmup_phase": warmup_phase,
            "timed_repeats": common.REPEATS,
            "samples": samples,
            "prefill_latency_seconds": prefill_seconds_total / common.REPEATS,
            "prefill_tps": (
                batch_size * common.INPUT_LEN * common.REPEATS / prefill_seconds_total
            ),
            "prefill_tps_stddev": statistics.stdev(prefill_tps_samples),
            "decode_latency_seconds": decode_seconds_total / common.REPEATS,
            "decode_tps": (
                batch_size
                * (common.OUTPUT_LEN - 1)
                * common.REPEATS
                / decode_seconds_total
            ),
            "decode_tps_stddev": statistics.stdev(decode_tps_samples),
            "wall_seconds_mean": statistics.mean(
                float(sample["wall_seconds"]) for sample in samples
            ),
            "output_tps": statistics.mean(
                float(sample["output_tps"]) for sample in samples
            ),
            "output_tps_stddev": statistics.stdev(
                float(sample["output_tps"]) for sample in samples
            ),
            "total_tps": statistics.mean(
                float(sample["total_tps"]) for sample in samples
            ),
            "total_tps_stddev": statistics.stdev(
                float(sample["total_tps"]) for sample in samples
            ),
            "first_token_spread_seconds_max": max(
                float(sample["first_token_spread_seconds"]) for sample in samples
            ),
            "scheduler_start_spread_seconds_max": max(
                float(sample["scheduler_start_spread_seconds"]) for sample in samples
            ),
        }
        result["rows"].append(row)
        result["gpu"]["peak_sampled_memory_used_mib"] = peak_memory_mib
        common.atomic_write_json(output_path, result)
        print(
            f"RESULT_PHASE framework={framework} batch={batch_size} "
            f"prefill_tps={row['prefill_tps']:.3f} "
            f"prefill_stddev={row['prefill_tps_stddev']:.3f} "
            f"decode_tps={row['decode_tps']:.3f} "
            f"decode_stddev={row['decode_tps_stddev']:.3f} "
            f"output_tps={row['output_tps']:.3f} "
            f"total_tps={row['total_tps']:.3f} "
            f"gpu_mem={peak_memory_mib}MiB",
            flush=True,
        )


def run_sglang() -> None:
    target = common.visible_gpu_target()
    preflight = common.strict_idle_gate(target)

    os.environ.setdefault(
        "SGLANG_EXTERNAL_MODEL_PACKAGE",
        "laguna_sglang_runtime_models",
    )
    os.environ.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
    sglang_python = str(common.SGLANG_REPO / "python")
    if sglang_python not in sys.path:
        sys.path.insert(0, sglang_python)

    import torch
    import sglang as sgl

    result = make_result("sglang", target, preflight, torch)
    result["runtime"]["sglang_kernel"] = common.package_version("sglang-kernel")
    output_path = phase_result_path("sglang")
    engine = None
    try:
        override = common.load_quantization_override()
        max_requested_batch_size = max(common.BATCH_SIZES)
        max_requested_prefill_tokens = max_requested_batch_size * common.INPUT_LEN
        engine = sgl.Engine(
            model_path=str(common.MODEL),
            tokenizer_path=str(common.MODEL),
            trust_remote_code=False,
            dtype="float16",
            quantization="gptq_marlin",
            context_length=common.MAX_MODEL_LEN,
            tp_size=1,
            mem_fraction_static=SGLANG_MEM_FRACTION_STATIC,
            max_running_requests=max_requested_batch_size,
            max_prefill_tokens=max_requested_prefill_tokens,
            prefill_max_requests=max_requested_batch_size,
            chunked_prefill_size=max_requested_prefill_tokens,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            disable_decode_cuda_graph=True,
            disable_prefill_cuda_graph=True,
            enable_metrics=True,
            json_model_override_args=json.dumps(override),
            random_seed=common.SEED,
            log_level="info",
        )
        result["runtime"]["phase_engine_config"] = {
            "max_running_requests": max_requested_batch_size,
            "max_running_requests_source": "largest_requested_batch_size",
            "max_prefill_tokens": max_requested_prefill_tokens,
            "max_prefill_tokens_source": "largest_requested_batch_size_times_input_length",
            "prefill_max_requests": max_requested_batch_size,
            "prefill_max_requests_source": "largest_requested_batch_size",
            "chunked_prefill_size": max_requested_prefill_tokens,
            "chunked_prefill_size_source": "largest_requested_batch_size_times_input_length",
            "disable_overlap_schedule": True,
            "enable_metrics": True,
            "mem_fraction_static": SGLANG_MEM_FRACTION_STATIC,
        }
        result["benchmark"].update(
            {
                "batch_semantics": (
                    "one submitted SGLang batch; request and prefill limits are "
                    "derived from the largest requested batch size"
                ),
                "max_resident_requests_engine_setting": max_requested_batch_size,
                "capacity_boundary": (
                    "CUDA OOM; SGLang request and prefill limits are derived "
                    "from the requested workload rather than a separate "
                    "benchmark capacity parameter"
                ),
            }
        )
        prompts = common.make_prompts()
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": common.OUTPUT_LEN,
            "min_new_tokens": common.OUTPUT_LEN,
            "ignore_eos": True,
        }

        def generate(batch_size: int) -> Any:
            return engine.generate(
                input_ids=prompts[:batch_size],
                sampling_params=sampling_params,
            )

        run_phase_rows(
            "sglang",
            generate,
            extract_sglang_phase,
            target,
            result,
            output_path,
        )
        result["success"] = True
        result["benchmark_completed_at_utc"] = common.utc_now()
        common.atomic_write_json(output_path, result)
    except BaseException:
        result["benchmark_completed_at_utc"] = common.utc_now()
        result["error"] = traceback.format_exc()
        common.atomic_write_json(output_path, result)
        raise
    finally:
        if engine is not None:
            engine.shutdown()


def run_vllm() -> None:
    target = common.visible_gpu_target()
    preflight = common.strict_idle_gate(target)

    vllm_repo = str(common.VLLM_REPO)
    if vllm_repo not in sys.path:
        sys.path.insert(0, vllm_repo)

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    result = make_result("vllm", target, preflight, torch)
    result["runtime"]["vllm_import_version"] = vllm.__version__
    output_path = phase_result_path("vllm")
    try:
        override = common.load_quantization_override()
        llm = LLM(
            model=str(common.MODEL),
            tokenizer=str(common.MODEL),
            trust_remote_code=False,
            dtype="float16",
            quantization="auto_gptq",
            tensor_parallel_size=1,
            max_model_len=common.MAX_MODEL_LEN,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=False,
            enforce_eager=True,
            seed=common.SEED,
            hf_overrides=override,
            model_class_overrides={
                "LagunaForCausalLM": "laguna_vllm_runtime_model:LagunaForCausalLM"
            },
            disable_log_stats=False,
        )
        scheduler_config = llm.llm_engine.vllm_config.scheduler_config
        result["runtime"]["phase_engine_config"] = {
            "max_num_seqs": int(scheduler_config.max_num_seqs),
            "max_num_seqs_source": "vllm_default",
            "max_num_batched_tokens": int(scheduler_config.max_num_batched_tokens),
            "max_num_batched_tokens_source": "vllm_default",
            "disable_log_stats": False,
            "enforce_eager": True,
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        }
        result["benchmark"].update(
            {
                "batch_semantics": (
                    "submitted request count; vLLM may schedule it in multiple " "waves"
                ),
                "max_resident_requests_engine_setting": None,
                "capacity_boundary": (
                    "CUDA OOM only; no benchmark-side resident-capacity cutoff"
                ),
            }
        )
        cache_config = llm.llm_engine.vllm_config.cache_config
        kv_cache_size_tokens = int(cache_config.kv_cache_size_tokens or 0)
        if kv_cache_size_tokens <= 0:
            raise RuntimeError(
                "vLLM did not report a positive kv_cache_size_tokens capacity."
            )
        estimated_full_length_resident_requests = (
            kv_cache_size_tokens // common.MAX_MODEL_LEN
        )
        result["runtime"]["phase_engine_config"].update(
            {
                "kv_cache_size_tokens": kv_cache_size_tokens,
                "estimated_full_length_resident_requests": (
                    estimated_full_length_resident_requests
                ),
            }
        )
        prompts = [
            TokensPrompt(prompt_token_ids=prompt) for prompt in common.make_prompts()
        ]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=common.OUTPUT_LEN,
            min_tokens=common.OUTPUT_LEN,
            ignore_eos=True,
            detokenize=False,
        )

        def generate(batch_size: int) -> Any:
            return llm.generate(
                prompts[:batch_size],
                sampling_params,
                use_tqdm=False,
            )

        run_phase_rows(
            "vllm",
            generate,
            extract_vllm_phase,
            target,
            result,
            output_path,
        )
        result["success"] = True
        result["benchmark_completed_at_utc"] = common.utc_now()
        common.atomic_write_json(output_path, result)
    except BaseException:
        result["benchmark_completed_at_utc"] = common.utc_now()
        result["error"] = traceback.format_exc()
        common.atomic_write_json(output_path, result)
        raise


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("sglang", "vllm"):
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} sglang|vllm")
    if sys.argv[1] == "sglang":
        run_sglang()
    else:
        run_vllm()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
