#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Score ARC/GSM8K/MMLU suites through Evalution.

Each invocation requires exactly one of ARC-Challenge, GSM8K-Platinum CoT,
MMLU-STEM, or the four MMLU-History subjects used by the Ultra reports. Keeping
one suite per fresh engine process preserves independent startup/runtime timing
and lets every suite use its historically efficient tensor-parallel and batching
configuration. The wrapper supports either vLLM or SGLang without embedding
model paths, repository paths, GPU indices, or conda installation paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

if __package__:
    from . import engine_common as common
else:
    import engine_common as common


MMLU_HISTORY_SUBSETS = (
    "humanities.high_school_european_history",
    "humanities.high_school_us_history",
    "humanities.high_school_world_history",
    "humanities.prehistory",
)

FULL_TASK_PROTOCOL = {
    "arc_challenge": {
        "expected_samples": 1172,
        "metric": "acc,exam",
        "setup": "multiple-choice log-likelihood",
    },
    "gsm8k_platinum_cot": {
        "expected_samples": 1209,
        "metric": "acc,num",
        "setup": "CoT generation with the tokenizer chat template",
    },
    "mmlu_stem": {
        "expected_samples": 3153,
        "expected_subjects": 19,
        "metric": "acc,ll",
        "setup": "5-shot multiple-choice log-likelihood",
    },
    "mmlu_history": {
        "expected_samples": 930,
        "expected_subjects": list(MMLU_HISTORY_SUBSETS),
        "metric": "acc,ll",
        "setup": "5-shot multiple-choice log-likelihood",
    },
}

TASK_DISPLAY_NAMES = {
    "arc_challenge": "ARC-Challenge",
    "gsm8k_platinum_cot": "GSM8K-Platinum CoT",
    "mmlu_stem": "MMLU-STEM",
    "mmlu_history": "MMLU-History",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("vllm", "sglang"))
    parser.add_argument(
        "--model", required=True, help="Local checkpoint path or Hugging Face model ID."
    )
    parser.add_argument("--tokenizer", help="Tokenizer path/ID. Defaults to --model.")
    parser.add_argument("--revision")
    parser.add_argument(
        "--output", type=Path, required=True, help="Raw Evalution JSON output."
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        help="Generated YAML path. Defaults to <output-stem>.evalution.yaml.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Validated score summary. Defaults to <output-stem>.scores.json.",
    )
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        choices=tuple(FULL_TASK_PROTOCOL),
        help=(
            "The single suite to run in this fresh engine process. Specify exactly once."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quantization",
        help="Optional runtime quantization override; checkpoint metadata is used by default.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--model-kwargs-json",
        help="Extra model/runtime kwargs as a JSON object or @path. Explicit entries override defaults.",
    )
    parser.add_argument(
        "--tokenizer-kwargs-json",
        help="Extra tokenizer kwargs as a JSON object or @path.",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--sglang-max-running-requests", type=int, default=64)
    parser.add_argument(
        "--sglang-attention-backend",
        default="flashinfer",
        help="Use 'auto' to leave backend selection to SGLang.",
    )
    parser.add_argument(
        "--sglang-prefill-attention-backend",
        default="triton",
        help="Use 'auto' to leave prefill backend selection to SGLang.",
    )
    parser.add_argument(
        "--sglang-sampling-backend",
        default="auto",
        help="Use 'auto' to leave sampling backend selection to SGLang.",
    )
    parser.add_argument("--sglang-nondeterministic", action="store_true")
    parser.add_argument("--sglang-enable-cuda-graph", action="store_true")
    parser.add_argument(
        "--sglang-disable-fused-wqa-wkv",
        action="store_true",
        help=(
            "Set SGLANG_OPT_FUSE_WQA_WKV=0 in the Evalution subprocess. This is useful on runtimes or "
            "architectures where the fused DeepSeek WQA/WKV path does not return valid choice logprobs."
        ),
    )
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=256)
    parser.add_argument("--mmlu-fewshot", type=int, default=5)
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional smoke-test bound for the selected suite. Omit for its complete dataset.",
    )
    parser.add_argument(
        "--evalution-executable",
        default="evalution",
        help="Evalution executable name/path from the active environment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and schema-validate the YAML, print the command, and do not load a model.",
    )
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval-seconds", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=16)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.tensor_parallel_size < 1 or args.context_length < 1:
        raise ValueError(
            "Batch size, tensor parallel size, and context length must be positive."
        )
    if args.gsm8k_max_new_tokens < 1 or args.mmlu_fewshot < 0:
        raise ValueError(
            "GSM8K max-new-tokens must be positive and MMLU few-shot must be non-negative."
        )
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive when provided.")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        raise ValueError("--vllm-gpu-memory-utilization must be in (0, 1].")
    if not 0 < args.sglang_mem_fraction_static < 1:
        raise ValueError("--sglang-mem-fraction-static must be in (0, 1).")
    if args.sglang_max_running_requests < 1:
        raise ValueError("--sglang-max-running-requests must be positive.")
    if args.idle_samples < 1 or args.idle_interval_seconds < 0:
        raise ValueError(
            "Idle samples must be positive and the interval must be non-negative."
        )
    if args.idle_memory_tolerance_mib < 0:
        raise ValueError("Idle memory tolerance must be non-negative.")
    if args.sglang_disable_fused_wqa_wkv and args.engine != "sglang":
        raise ValueError("--sglang-disable-fused-wqa-wkv requires --engine sglang.")
    if len(args.task) != 1:
        raise ValueError(
            "Specify exactly one --task per invocation so every suite gets an independent engine startup."
        )


def _selected_tasks(args: argparse.Namespace) -> tuple[str, ...]:
    return (args.task[0],)


def _optional_backend(value: str) -> str | None:
    normalized = value.strip()
    return None if normalized.lower() in {"", "auto", "none"} else normalized


def _validate_model_kwargs(engine: str, model_kwargs: Mapping[str, Any]) -> None:
    reserved = {
        "vllm": {
            "model",
            "tokenizer",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "max_model_len",
            "dtype",
            "seed",
            "revision",
            "quantization",
            "enforce_eager",
        },
        "sglang": {
            "model_path",
            "tokenizer_path",
            "tp_size",
            "mem_fraction_static",
            "context_length",
            "dtype",
            "random_seed",
            "revision",
            "quantization",
        },
    }[engine]
    conflicts = sorted(reserved.intersection(model_kwargs))
    if conflicts:
        raise ValueError(
            f"--model-kwargs-json may not override CLI-owned {engine} keys: {conflicts}"
        )


def _task_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    common_options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "stream": True,
    }
    if args.max_rows is not None:
        common_options["max_rows"] = args.max_rows
    specs = {
        "arc_challenge": {"type": "arc_challenge", **common_options},
        "gsm8k_platinum_cot": {
            "type": "gsm8k_platinum",
            "variant": "cot",
            "apply_chat_template": True,
            "max_new_tokens": args.gsm8k_max_new_tokens,
            **common_options,
        },
        "mmlu_stem": {
            "type": "mmlu",
            "subsets": "stem",
            "num_fewshot": args.mmlu_fewshot,
            **common_options,
        },
        "mmlu_history": {
            "type": "mmlu",
            "subsets": list(MMLU_HISTORY_SUBSETS),
            "num_fewshot": args.mmlu_fewshot,
            **common_options,
        },
    }
    return [specs[task] for task in _selected_tasks(args)]


def build_evalution_spec(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    model_kwargs = common.load_json_object(args.model_kwargs_json)
    tokenizer_kwargs = common.load_json_object(args.tokenizer_kwargs_json)
    _validate_model_kwargs(args.engine, model_kwargs)

    model: dict[str, Any] = {
        "path": args.model,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.tokenizer:
        model["tokenizer_path"] = args.tokenizer
    if args.revision:
        model["revision"] = args.revision
    if tokenizer_kwargs:
        model["tokenizer_kwargs"] = tokenizer_kwargs

    if args.engine == "vllm":
        engine: dict[str, Any] = {
            "type": "VLLM",
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_model_len": args.context_length,
            "enforce_eager": args.vllm_enforce_eager,
            "seed": args.seed,
        }
    else:
        attention_backend = _optional_backend(args.sglang_attention_backend)
        sampling_backend = _optional_backend(args.sglang_sampling_backend)
        prefill_backend = _optional_backend(args.sglang_prefill_attention_backend)
        engine = {
            "type": "SGLang",
            "device": "cuda:0",
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "mem_fraction_static": args.sglang_mem_fraction_static,
            "max_running_requests": args.sglang_max_running_requests,
            "tp_size": args.tensor_parallel_size,
            "seed": args.seed,
        }
        if attention_backend:
            engine["attention_backend"] = attention_backend
        if sampling_backend:
            engine["sampling_backend"] = sampling_backend
        model_kwargs.setdefault(
            "enable_deterministic_inference", not args.sglang_nondeterministic
        )
        model_kwargs.setdefault("disable_cuda_graph", not args.sglang_enable_cuda_graph)
        if prefill_backend:
            model_kwargs.setdefault("prefill_attention_backend", prefill_backend)

    if args.quantization:
        engine["quantization"] = args.quantization
    if model_kwargs:
        model["model_kwargs"] = model_kwargs
    return {"engine": engine, "model": model, "tests": _task_specs(args)}


def _derived_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    config_path = args.config_output or args.output.with_name(
        f"{args.output.stem}.evalution.yaml"
    )
    summary_path = args.summary_output or args.output.with_name(
        f"{args.output.stem}.scores.json"
    )
    resolved = {args.output.resolve(), config_path.resolve(), summary_path.resolve()}
    if len(resolved) != 3:
        raise ValueError(
            "Raw output, generated config, and score summary paths must be distinct."
        )
    return config_path, summary_path


def write_evalution_yaml(path: Path, spec: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required by Evalution but is not installed."
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(spec), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve_evalution_executable(value: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find Evalution executable {value!r} in the active environment."
        )
    return resolved


def _validate_generated_config(executable: str, config_path: Path) -> None:
    result = subprocess.run(
        [executable, "emit-python", str(config_path.resolve())],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Evalution rejected the generated YAML: {detail}")


def _logical_test_name(test: Mapping[str, Any]) -> str | None:
    name = str(test.get("name", ""))
    if name == "arc_challenge":
        return "arc_challenge"
    if name == "gsm8k_platinum_cot":
        return "gsm8k_platinum_cot"
    if name == "mmlu_stem":
        return "mmlu_stem"
    history_fragments = tuple(
        subset.replace(".", "_") for subset in MMLU_HISTORY_SUBSETS
    )
    if name.startswith("mmlu_") and all(
        fragment in name for fragment in history_fragments
    ):
        return "mmlu_history"
    return None


def summarize_gpu_usage(
    preflight: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gpus = []
    seen_uuids: set[str] = set()
    for sample in preflight:
        uuid = str(sample["uuid"])
        if uuid in seen_uuids:
            continue
        seen_uuids.add(uuid)
        gpus.append(
            {
                "physical_id": sample["physical_id"],
                "pci_bus_id": sample["pci_bus_id"],
                "uuid": uuid,
                "model": sample["name"],
                "driver_version": sample["driver_version"],
                "compute_capability": sample["compute_capability"],
                "memory_total_mib": sample["memory_total_mib"],
            }
        )
    return gpus


class EvalutionTimingTracker:
    def __init__(
        self,
        *,
        process_started_monotonic: float,
        process_started_at_utc: str,
        task_order: Sequence[str] | None = None,
    ) -> None:
        self.process_started_monotonic = process_started_monotonic
        self.process_started_at_utc = process_started_at_utc
        self._task_order = tuple(task_order or FULL_TASK_PROTOCOL)
        self._next_task_index = 0
        self._active_task: str | None = None
        self._active_started_monotonic: float | None = None
        self.engine_startup_seconds: float | None = None
        self.tasks: dict[str, dict[str, Any]] = {
            task: {"state": "pending"} for task in self._task_order
        }

    @staticmethod
    def _seconds(value: float) -> float:
        return round(value, 6)

    def observe_line(
        self,
        line: str,
        *,
        now_monotonic: float | None = None,
        now_utc: str | None = None,
    ) -> None:
        current_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        current_utc = common.utc_now() if now_utc is None else now_utc
        start_marker = "running test suite "
        if start_marker in line and self._next_task_index < len(self._task_order):
            logical_name = self._task_order[self._next_task_index]
            suite_type = line.split(start_marker, 1)[1].strip().split()[0]
            self._next_task_index += 1
            self._active_task = logical_name
            self._active_started_monotonic = current_monotonic
            self.tasks[logical_name] = {
                "state": "running",
                "evalution_suite_type": suite_type,
                "started_at_utc": current_utc,
            }
            if self.engine_startup_seconds is None:
                self.engine_startup_seconds = self._seconds(
                    current_monotonic - self.process_started_monotonic
                )

        completed_marker = "completed test "
        if completed_marker not in line:
            return
        evalution_name = line.split(completed_marker, 1)[1].strip().split()[0]
        logical_name = _logical_test_name({"name": evalution_name})
        if logical_name is None or logical_name != self._active_task:
            return
        if self._active_started_monotonic is None:
            return
        self.tasks[logical_name].update(
            {
                "state": "completed",
                "evalution_name": evalution_name,
                "completed_at_utc": current_utc,
                "duration_seconds": self._seconds(
                    current_monotonic - self._active_started_monotonic
                ),
            }
        )
        self._active_task = None
        self._active_started_monotonic = None

    def finish(
        self,
        *,
        return_code: int,
        now_monotonic: float | None = None,
        now_utc: str | None = None,
    ) -> dict[str, Any]:
        current_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        current_utc = common.utc_now() if now_utc is None else now_utc
        if self._active_task is not None and self._active_started_monotonic is not None:
            self.tasks[self._active_task].update(
                {
                    "state": "incomplete",
                    "completed_at_utc": current_utc,
                    "duration_seconds": self._seconds(
                        current_monotonic - self._active_started_monotonic
                    ),
                }
            )
        complete = all(record["state"] == "completed" for record in self.tasks.values())
        return {
            "evaluation_started_at_utc": self.process_started_at_utc,
            "evaluation_completed_at_utc": current_utc,
            "evaluation_wall_time_seconds": self._seconds(
                current_monotonic - self.process_started_monotonic
            ),
            "engine_startup_seconds": self.engine_startup_seconds,
            "return_code": return_code,
            "complete": complete,
            "tasks": self.tasks,
        }


def _validate_arc_exam_sample(
    sample: Mapping[str, Any],
    sample_value: float,
    *,
    choice_count: int,
) -> None:
    extracted = sample.get("extracted")
    metadata = sample.get("metadata")
    if not isinstance(extracted, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("arc_challenge sample is missing tie-aware exam metadata.")

    selected_count = metadata.get("selected_count")
    if (
        isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or selected_count < 1
    ):
        raise ValueError(
            f"arc_challenge sample has invalid selected_count {selected_count!r}."
        )
    try:
        gold_index = int(extracted["gold_index"])
        selected_indices = tuple(
            int(index.strip())
            for index in str(extracted["selected_indices"]).split(",")
            if index.strip()
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "arc_challenge sample has malformed selected/gold indices."
        ) from error
    if (
        len(selected_indices) != selected_count
        or len(set(selected_indices)) != selected_count
        or not 0 <= gold_index < choice_count
        or any(not 0 <= index < choice_count for index in selected_indices)
    ):
        raise ValueError(
            "arc_challenge sample selected/gold indices do not match its choice and selected counts."
        )

    expected = 1.0 / selected_count if gold_index in selected_indices else 0.0
    if not math.isclose(sample_value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"arc_challenge sample has invalid tie-aware {sample_value!r}; expected {expected!r}."
        )


def extract_score_summary(
    payload: Mapping[str, Any],
    *,
    require_full_coverage: bool,
    required_tasks: Sequence[str] | None = None,
) -> dict[str, Any]:
    tests = payload.get("tests")
    if not isinstance(tests, list):
        raise ValueError("Evalution result does not contain a tests list.")
    matched: dict[str, Mapping[str, Any]] = {}
    for test in tests:
        if not isinstance(test, Mapping):
            continue
        logical_name = _logical_test_name(test)
        if logical_name is None:
            continue
        if logical_name in matched:
            raise ValueError(
                f"Evalution result contains duplicate {logical_name} suites."
            )
        matched[logical_name] = test
    selected_tasks = tuple(required_tasks or FULL_TASK_PROTOCOL)
    missing = sorted(set(selected_tasks).difference(matched))
    if missing:
        raise ValueError(f"Evalution result is missing required suites: {missing}")

    scores: dict[str, Any] = {}
    for logical_name in selected_tasks:
        protocol = FULL_TASK_PROTOCOL[logical_name]
        test = matched[logical_name]
        metric = str(protocol["metric"])
        metrics = test.get("metrics")
        samples = test.get("samples")
        if not isinstance(metrics, Mapping) or metric not in metrics:
            raise ValueError(f"{logical_name} result is missing metric {metric!r}.")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{logical_name} result has no samples.")
        value = float(metrics[metric])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{logical_name} reported invalid score {value!r}.")
        if require_full_coverage and len(samples) != int(protocol["expected_samples"]):
            raise ValueError(
                f"{logical_name} has {len(samples)} samples; expected {protocol['expected_samples']}."
            )

        sample_values = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError(f"{logical_name} contains a malformed sample.")
            sample_scores = sample.get("scores")
            if not isinstance(sample_scores, Mapping) or metric not in sample_scores:
                raise ValueError(f"{logical_name} sample is missing score {metric!r}.")
            sample_value = float(sample_scores[metric])
            if not math.isfinite(sample_value) or not 0 <= sample_value <= 1:
                raise ValueError(
                    f"{logical_name} sample has invalid {metric!r} score {sample_value!r}."
                )
            choice_logprobs = None
            if logical_name != "gsm8k_platinum_cot":
                metadata = sample.get("metadata")
                choice_logprobs = (
                    metadata.get("choice_logprobs")
                    if isinstance(metadata, Mapping)
                    else None
                )
                if not isinstance(choice_logprobs, list) or not choice_logprobs:
                    raise ValueError(
                        f"{logical_name} sample is missing choice_logprobs."
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in choice_logprobs
                ):
                    raise ValueError(
                        f"{logical_name} sample contains non-finite choice_logprobs."
                    )
            if logical_name == "arc_challenge":
                _validate_arc_exam_sample(
                    sample,
                    sample_value,
                    choice_count=len(choice_logprobs),
                )
            elif sample_value not in (0.0, 1.0):
                raise ValueError(
                    f"{logical_name} sample has non-binary {metric!r} score {sample_value!r}."
                )
            sample_values.append(sample_value)
        score_sum = sum(sample_values)
        recomputed = score_sum / len(sample_values)
        if abs(recomputed - value) > 1e-12:
            raise ValueError(
                f"{logical_name} reported {metric}={value}, but samples recompute to {recomputed}."
            )

        record: dict[str, Any] = {
            "evalution_name": test.get("name"),
            "metric": metric,
            "score": value,
            "score_sum": score_sum,
            "correct": sum(sample_value == 1.0 for sample_value in sample_values),
            "partial_credit_samples": sum(
                0.0 < sample_value < 1.0 for sample_value in sample_values
            ),
            "samples": len(samples),
            "expected_full_samples": protocol["expected_samples"],
        }
        if logical_name.startswith("mmlu_"):
            subsets = sorted(
                {
                    str(sample.get("metadata", {}).get("subset"))
                    for sample in samples
                    if isinstance(sample.get("metadata"), Mapping)
                    and sample.get("metadata", {}).get("subset") is not None
                }
            )
            record["subsets"] = subsets
            if (
                require_full_coverage
                and logical_name == "mmlu_stem"
                and len(subsets) != 19
            ):
                raise ValueError(
                    f"MMLU-STEM contains {len(subsets)} subjects; expected 19."
                )
            if (
                require_full_coverage
                and logical_name == "mmlu_history"
                and subsets != sorted(MMLU_HISTORY_SUBSETS)
            ):
                raise ValueError(
                    f"MMLU-History subjects differ from the required four: {subsets}"
                )
        scores[logical_name] = record
    return scores


def _print_scores(
    scores: Mapping[str, Mapping[str, Any]],
    task_timings: Mapping[str, Mapping[str, Any]],
) -> None:
    rows = [
        (
            logical_name,
            record["metric"],
            f"{float(record['score']):.10f}",
            f"{record['correct']}/{record['samples']}",
            f"{float(task_timings[logical_name]['duration_seconds']):.3f}",
        )
        for logical_name, record in scores.items()
    ]
    print(
        common.render_ascii_table(
            ("Suite", "Metric", "Score", "Correct", "Time (s)"), rows
        )
    )


def _print_live_status(
    args: argparse.Namespace,
    preflight: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> None:
    physical_ids = sorted({str(sample["physical_id"]) for sample in preflight})
    decoder_bits = args.quantization or "checkpoint metadata"
    requested_metrics = " | ".join(
        TASK_DISPLAY_NAMES[task] for task in _selected_tasks(args)
    )
    rows = [
        (
            ",".join(physical_ids),
            f"{args.engine}:{Path(args.model).name or args.model}",
            decoder_bits,
            "n/a (scoring)",
            "n/a",
            f"{requested_metrics} pending",
            f"running ({elapsed_seconds / 60:.1f} min)",
        )
    ]
    print(
        common.render_ascii_table(
            (
                "GPU ID(s)",
                "Candidate",
                "Model / Decoder Bits",
                "Endpoint Bits",
                "Size (MB)",
                "Metrics",
                "State",
            ),
            rows,
        ),
        flush=True,
    )


def _stream_process_output(
    stream: Any,
    output_queue: queue.Queue[str | None],
) -> None:
    try:
        for line in iter(stream.readline, ""):
            output_queue.put(line)
    finally:
        stream.close()
        output_queue.put(None)


def _evalution_process_environment(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, str]]:
    environment = os.environ.copy()
    overrides: dict[str, str] = {}
    if args.engine == "sglang" and args.sglang_disable_fused_wqa_wkv:
        overrides["SGLANG_OPT_FUSE_WQA_WKV"] = "0"
    environment.update(overrides)
    return environment, overrides


def _run_evalution_with_live_status(
    command: Sequence[str],
    args: argparse.Namespace,
    preflight: Sequence[Mapping[str, Any]],
    process_environment: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    process_started_monotonic = time.monotonic()
    process_started_at_utc = common.utc_now()
    tracker = EvalutionTimingTracker(
        process_started_monotonic=process_started_monotonic,
        process_started_at_utc=process_started_at_utc,
        task_order=_selected_tasks(args),
    )
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=None if process_environment is None else dict(process_environment),
    )
    if process.stdout is None:
        raise RuntimeError("Evalution subprocess stdout pipe was not created.")
    output_queue: queue.Queue[str | None] = queue.Queue()
    output_thread = threading.Thread(
        target=_stream_process_output,
        args=(process.stdout, output_queue),
        daemon=True,
    )
    output_thread.start()
    next_status_at = process_started_monotonic + 60
    stream_closed = False
    try:
        while not stream_closed or process.poll() is None:
            current_monotonic = time.monotonic()
            timeout = max(0.0, min(1.0, next_status_at - current_monotonic))
            try:
                line = output_queue.get(timeout=timeout)
            except queue.Empty:
                line = ""
            if line is None:
                stream_closed = True
            elif line:
                print(line, end="", flush=True)
                tracker.observe_line(line)

            current_monotonic = time.monotonic()
            if current_monotonic >= next_status_at:
                _print_live_status(
                    args,
                    preflight,
                    current_monotonic - process_started_monotonic,
                )
                next_status_at += 60

        return_code = process.wait()
        output_thread.join(timeout=1)
        return return_code, tracker.finish(return_code=return_code)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def run_main(args: argparse.Namespace) -> int:
    run_started_at_utc = common.utc_now()
    spec = build_evalution_spec(args)
    config_path, summary_path = _derived_paths(args)
    write_evalution_yaml(config_path, spec)
    executable = _resolve_evalution_executable(args.evalution_executable)
    _validate_generated_config(executable, config_path)
    command = [
        executable,
        "run",
        str(config_path.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    print(f"CONFIG {config_path.resolve()}")
    print(f"COMMAND {shlex.join(command)}")
    if args.dry_run:
        return 0

    expected_environment = f"{args.engine}_test"
    active_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if active_environment != expected_environment:
        print(
            f"WARNING expected conda environment {expected_environment!r}; active={active_environment!r}",
            file=sys.stderr,
            flush=True,
        )
    targets = common.visible_gpu_targets(args.tensor_parallel_size)
    preflight = common.strict_idle_gate(
        targets,
        sample_count=args.idle_samples,
        interval_seconds=args.idle_interval_seconds,
        memory_tolerance_mib=args.idle_memory_tolerance_mib,
    )
    torch_runtime = common.collect_torch_runtime_metadata(len(targets))
    gpus = summarize_gpu_usage(preflight)
    selected_tasks = _selected_tasks(args)
    selected_protocol = {
        task: FULL_TASK_PROTOCOL[task] for task in selected_tasks
    }
    process_environment, environment_overrides = _evalution_process_environment(args)
    summary: dict[str, Any] = {
        "schema_version": 3,
        "success": False,
        "started_at_utc": run_started_at_utc,
        "gpu_models": sorted({gpu["model"] for gpu in gpus}),
        "gpus": gpus,
        "wrapper_command": common.exact_command(),
        "evalution_command": shlex.join(command),
        "evalution_environment_overrides": environment_overrides,
        "engine": args.engine,
        "model": args.model,
        "config": str(config_path.resolve()),
        "raw_result": str(args.output.resolve()),
        "full_coverage_requested": args.max_rows is None,
        "max_rows": args.max_rows,
        "tasks_requested": list(selected_tasks),
        "protocol": selected_protocol,
        "strict_idle_preflight": preflight,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "conda_environment": active_environment,
            "evalution_executable": executable,
            "torch": torch_runtime,
            "packages": {
                package: common.package_version(package)
                for package in (
                    "datasets",
                    "evalution",
                    "sglang",
                    "torch",
                    "transformers",
                    "triton",
                    "vllm",
                )
            },
            "requested_dtype": args.dtype,
            "batch_size": args.batch_size,
            "tasks": list(selected_tasks),
            "context_length": args.context_length,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
    }
    common.atomic_write_json(summary_path, summary)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        return_code, timing = _run_evalution_with_live_status(
            command,
            args,
            preflight,
            process_environment,
        )
        summary["evalution_return_code"] = return_code
        summary["evaluation_started_at_utc"] = timing["evaluation_started_at_utc"]
        summary["evaluation_completed_at_utc"] = timing["evaluation_completed_at_utc"]
        summary["evaluation_wall_time_seconds"] = timing["evaluation_wall_time_seconds"]
        summary["engine_startup_seconds"] = timing["engine_startup_seconds"]
        summary["task_timings"] = timing["tasks"]
        if return_code:
            summary["error"] = f"Evalution exited with status {return_code}."
            return return_code
        if not timing["complete"]:
            raise RuntimeError(
                "Evalution completed successfully but per-task timing events were incomplete: "
                f"{timing['tasks']}"
            )
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        scores = extract_score_summary(
            payload,
            require_full_coverage=args.max_rows is None,
            required_tasks=selected_tasks,
        )
        for logical_name, record in scores.items():
            task_timing = timing["tasks"][logical_name]
            record["started_at_utc"] = task_timing["started_at_utc"]
            record["completed_at_utc"] = task_timing["completed_at_utc"]
            record["duration_seconds"] = task_timing["duration_seconds"]
        summary["scores"] = scores
        summary["success"] = True
        _print_scores(scores, timing["tasks"])
        return 0
    except BaseException:
        summary["error"] = traceback.format_exc()
        raise
    finally:
        summary["completed_at_utc"] = common.utc_now()
        common.atomic_write_json(summary_path, summary)


def main(argv: Sequence[str] | None = None) -> int:
    return run_main(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
