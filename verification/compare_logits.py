#!/usr/bin/env python3
"""Compare native and post-quant next-token distributions with vLLM or SGLang.

Each engine runs in a fresh subprocess on identical raw token IDs. The script
requests the complete vocabulary distribution at bounded prompt positions and
reports KL(P_native || P_quant), top-token agreement, top-k overlap, probability
distance, and centered-logit error. Centering removes the additive constant that
cannot be recovered from engine log-probability APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import engine_common as common
else:
    import engine_common as common


DEFAULT_PROMPTS = (
    "Explain why the sky appears blue in two concise sentences.",
    "Compute 37 * 19 and show the essential arithmetic.",
    "Write a Python function that returns the median of a non-empty list.",
    "Summarize the difference between prefill and decode in language-model inference.",
    "Translate 'reliable numerical verification' into Chinese.",
    "A train travels 120 km in 90 minutes. What is its average speed in km/h?",
    "Name three properties that make an experiment reproducible.",
    "If all roses are flowers and some flowers fade quickly, what follows logically?",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("vllm", "sglang"))
    parser.add_argument("--native-model", required=True)
    parser.add_argument("--quantized-model", required=True)
    parser.add_argument(
        "--tokenizer", help="Shared tokenizer path/ID. Defaults to the native model."
    )
    parser.add_argument("--native-revision")
    parser.add_argument("--quantized-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument(
        "--native-quantization",
        help="Optional engine quantization override for the native model.",
    )
    parser.add_argument(
        "--quantized-quantization",
        help="Optional engine quantization override for the post-quant model.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="JSON/JSONL/TXT held-out prompts; defaults to built-in smoke prompts.",
    )
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--positions-per-prompt", type=int, default=4)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Wrap string prompts as one user message. JSON records with messages always use the chat template.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Raw inputs/log-probabilities. Defaults beside --output under <output-stem>_artifacts/.",
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--native-tensor-parallel-size",
        type=int,
        help="Native-only TP size. Defaults to --tensor-parallel-size.",
    )
    parser.add_argument(
        "--quantized-tensor-parallel-size",
        type=int,
        help="Quantized-only TP size. Defaults to --tensor-parallel-size.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--context-margin", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--engine-args-json", help="Common engine kwargs as JSON or @path."
    )
    parser.add_argument(
        "--native-engine-args-json", help="Native-only engine kwargs as JSON or @path."
    )
    parser.add_argument(
        "--quantized-engine-args-json",
        help="Quantized-only engine kwargs as JSON or @path.",
    )
    parser.add_argument(
        "--reuse-native-artifact",
        type=Path,
        help="Validated native .npz from an interrupted run; only the quantized worker is relaunched.",
    )
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval-seconds", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=16)
    parser.add_argument("--engine-transition-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--engine-transition-poll-seconds", type=float, default=1.0)
    parser.add_argument("--max-artifact-gib", type=float, default=8.0)
    parser.add_argument(
        "--max-kld-mean", type=float, help="Optional failing quality gate."
    )
    parser.add_argument(
        "--min-top1-agreement",
        type=float,
        help="Optional failing quality gate in [0, 1].",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    native_tp, quantized_tp = _role_tensor_parallel_sizes(args)
    if (
        args.max_prompts < 1
        or args.max_prompt_tokens < 1
        or args.positions_per_prompt < 1
    ):
        raise ValueError(
            "Prompt count, prompt length, and positions per prompt must be positive."
        )
    if (
        args.batch_size < 1
        or args.top_k < 1
        or args.tensor_parallel_size < 1
        or native_tp < 1
        or quantized_tp < 1
    ):
        raise ValueError(
            "Batch size, top-k, and tensor parallel size must be positive."
        )
    if args.context_margin < 0 or args.max_artifact_gib <= 0:
        raise ValueError(
            "Context margin must be non-negative and max artifact GiB must be positive."
        )
    if args.engine_transition_timeout_seconds < 0 or args.engine_transition_poll_seconds < 0:
        raise ValueError("Engine-transition timeout and polling interval must be non-negative.")
    if not 0 < args.gpu_memory_utilization <= 1 or not 0 < args.mem_fraction_static < 1:
        raise ValueError("Engine memory fractions must be in their valid ranges.")
    if args.max_kld_mean is not None and args.max_kld_mean < 0:
        raise ValueError("--max-kld-mean must be non-negative.")
    if args.min_top1_agreement is not None and not 0 <= args.min_top1_agreement <= 1:
        raise ValueError("--min-top1-agreement must be in [0, 1].")


def _role_tensor_parallel_sizes(args: argparse.Namespace) -> tuple[int, int]:
    common_tp = int(args.tensor_parallel_size)
    native_tp = getattr(args, "native_tensor_parallel_size", None)
    quantized_tp = getattr(args, "quantized_tensor_parallel_size", None)
    return (
        common_tp if native_tp is None else int(native_tp),
        common_tp if quantized_tp is None else int(quantized_tp),
    )


def _load_prompt_records(path: Path | None, max_prompts: int) -> list[Any]:
    if path is None:
        return list(DEFAULT_PROMPTS[:max_prompts])
    suffix = path.suffix.lower()
    if suffix == ".json":
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            raise ValueError("Prompt JSON must contain a list.")
    elif suffix == ".jsonl":
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        records = [
            line.strip() for line in path.read_text().splitlines() if line.strip()
        ]
    if not records:
        raise ValueError(f"No prompts were loaded from {path}.")
    return records[:max_prompts]


def _record_content(record: Any) -> tuple[str | None, list[dict[str, Any]] | None]:
    if isinstance(record, str):
        return record, None
    if not isinstance(record, dict):
        raise ValueError(
            f"Prompt records must be strings or objects, received {type(record)!r}."
        )
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "A prompt record's messages field must be a non-empty list."
            )
        return None, messages
    text = record.get("text", record.get("prompt"))
    if not isinstance(text, str):
        raise ValueError(
            "Prompt objects require a string 'text'/'prompt' or a 'messages' list."
        )
    return text, None


def _even_prefix_lengths(token_count: int, requested: int) -> list[int]:
    if token_count < 1:
        raise ValueError("Cannot sample positions from an empty token sequence.")
    count = min(token_count, requested)
    if count == 1:
        return [token_count]
    lengths = {
        1 + round(index * (token_count - 1) / (count - 1)) for index in range(count)
    }
    return sorted(lengths)


def _config_vocab_size(config: Any) -> int:
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None and getattr(config, "text_config", None) is not None:
        vocab_size = getattr(config.text_config, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError("Model config does not expose vocab_size.")
    return int(vocab_size)


def prepare_inputs(
    args: argparse.Namespace, artifacts_dir: Path
) -> tuple[dict[str, Any], int]:
    from transformers import AutoConfig, AutoTokenizer

    tokenizer_path = args.tokenizer or args.native_model
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.tokenizer_revision:
        tokenizer_kwargs["revision"] = args.tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    native_config_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    quantized_config_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code
    }
    if args.native_revision:
        native_config_kwargs["revision"] = args.native_revision
    if args.quantized_revision:
        quantized_config_kwargs["revision"] = args.quantized_revision
    native_config = AutoConfig.from_pretrained(
        args.native_model, **native_config_kwargs
    )
    quantized_config = AutoConfig.from_pretrained(
        args.quantized_model, **quantized_config_kwargs
    )
    native_vocab_size = _config_vocab_size(native_config)
    quantized_vocab_size = _config_vocab_size(quantized_config)
    if native_vocab_size != quantized_vocab_size:
        raise ValueError(
            f"Native and quantized vocabularies differ: {native_vocab_size} != {quantized_vocab_size}."
        )
    if args.top_k > native_vocab_size:
        raise ValueError(
            f"--top-k={args.top_k} exceeds vocab_size={native_vocab_size}."
        )

    prompt_records = _load_prompt_records(args.prompts, args.max_prompts)
    rendered_prompts = []
    sequences = []
    positions = []
    for prompt_index, record in enumerate(prompt_records):
        text, messages = _record_content(record)
        uses_chat_template = messages is not None or args.apply_chat_template
        if messages is None and args.apply_chat_template:
            messages = [{"role": "user", "content": text}]
        if messages is not None:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        else:
            rendered = str(text)
            token_ids = tokenizer.encode(rendered, add_special_tokens=True)
        if not token_ids:
            raise ValueError(f"Prompt {prompt_index} tokenized to an empty sequence.")
        if args.truncation_side == "right":
            token_ids = token_ids[: args.max_prompt_tokens]
        else:
            token_ids = token_ids[-args.max_prompt_tokens :]
        if max(token_ids) >= native_vocab_size or min(token_ids) < 0:
            raise ValueError(
                f"Prompt {prompt_index} contains a token outside vocab_size={native_vocab_size}."
            )
        rendered_prompts.append(
            {
                "prompt_index": prompt_index,
                "rendered_text": rendered,
                "token_ids": token_ids,
                "uses_chat_template": uses_chat_template,
            }
        )
        for prefix_length in _even_prefix_lengths(
            len(token_ids), args.positions_per_prompt
        ):
            sequences.append(token_ids[:prefix_length])
            positions.append(
                {
                    "sample_index": len(sequences) - 1,
                    "prompt_index": prompt_index,
                    "prefix_length": prefix_length,
                    "target_token_id": token_ids[prefix_length]
                    if prefix_length < len(token_ids)
                    else None,
                    "is_final_position": prefix_length == len(token_ids),
                }
            )

    payload = {
        "schema_version": 1,
        "tokenizer": tokenizer_path,
        "tokenizer_revision": args.tokenizer_revision,
        "vocab_size": native_vocab_size,
        "prompt_source": str(args.prompts.resolve())
        if args.prompts
        else "built-in smoke prompts",
        "max_prompt_tokens": args.max_prompt_tokens,
        "truncation_side": args.truncation_side,
        "positions_per_prompt_requested": args.positions_per_prompt,
        "rendered_prompts": rendered_prompts,
        "sequences": sequences,
        "positions": positions,
    }
    payload["sequence_sha256"] = common.sha256_json(sequences)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(artifacts_dir / "inputs.json", payload)
    return payload, native_vocab_size


def _dense_from_vllm_position(position: Any, vocab_size: int, numpy_module: Any) -> Any:
    dense = numpy_module.full(vocab_size, -numpy_module.inf, dtype=numpy_module.float32)
    seen = numpy_module.zeros(vocab_size, dtype=numpy_module.bool_)
    if isinstance(position, Mapping):
        items = position.items()
    else:
        raise TypeError(
            f"Unexpected vLLM log-probability position type: {type(position)!r}."
        )
    for token_id, value in items:
        token_id = int(token_id)
        if not 0 <= token_id < vocab_size:
            continue
        logprob = value.logprob if hasattr(value, "logprob") else value
        dense[token_id] = float(logprob)
        seen[token_id] = True
    missing = int((~seen).sum())
    if missing:
        raise RuntimeError(
            f"vLLM full-vocabulary response omitted {missing}/{vocab_size} token IDs."
        )
    return dense


def _dense_from_vllm_output(output: Any, vocab_size: int, numpy_module: Any) -> Any:
    if len(output.outputs) != 1:
        raise RuntimeError(
            f"vLLM returned {len(output.outputs)} candidates; expected one."
        )
    logprobs = output.outputs[0].logprobs
    if logprobs is None or len(logprobs) != 1:
        raise RuntimeError(
            "vLLM did not return exactly one generated-token log-probability distribution."
        )
    if all(
        hasattr(logprobs, field)
        for field in ("start_indices", "end_indices", "token_ids", "logprobs")
    ):
        start = int(logprobs.start_indices[0])
        end = int(logprobs.end_indices[0])
        token_ids = logprobs.token_ids[start:end]
        values = logprobs.logprobs[start:end]
        dense = numpy_module.full(
            vocab_size, -numpy_module.inf, dtype=numpy_module.float32
        )
        seen = numpy_module.zeros(vocab_size, dtype=numpy_module.bool_)
        for token_id, value in zip(token_ids, values, strict=True):
            token_id = int(token_id)
            if 0 <= token_id < vocab_size:
                dense[token_id] = float(value)
                seen[token_id] = True
        missing = int((~seen).sum())
        if missing:
            raise RuntimeError(
                f"vLLM flat response omitted {missing}/{vocab_size} token IDs."
            )
        return dense
    return _dense_from_vllm_position(logprobs[0], vocab_size, numpy_module)


def _dense_from_sglang_top_logprobs(
    top_logprobs: Any, vocab_size: int, numpy_module: Any
) -> Any:
    if not isinstance(top_logprobs, list):
        raise TypeError(f"Unexpected SGLang top-logprobs type: {type(top_logprobs)!r}.")
    dense = numpy_module.full(vocab_size, -numpy_module.inf, dtype=numpy_module.float32)
    seen = numpy_module.zeros(vocab_size, dtype=numpy_module.bool_)
    for item in top_logprobs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise TypeError(f"Unexpected SGLang top-logprob entry: {item!r}.")
        value, token_id = item[:2]
        token_id = int(token_id)
        if 0 <= token_id < vocab_size:
            logprob = float(value)
            # SGLang represents impossible/masked vocabulary entries with the
            # lowest finite float32 value rather than IEEE -inf.  Preserve the
            # probability-zero meaning so these sentinels do not dominate
            # centered-logit statistics.
            if logprob <= float(numpy_module.finfo(numpy_module.float32).min):
                logprob = -numpy_module.inf
            dense[token_id] = logprob
            seen[token_id] = True
    missing = int((~seen).sum())
    if missing:
        raise RuntimeError(
            f"SGLang full-vocabulary response omitted {missing}/{vocab_size} token IDs."
        )
    return dense


def _as_sglang_batch(outputs: Any, expected: int) -> list[dict[str, Any]]:
    if isinstance(outputs, dict):
        outputs = [outputs]
    if not isinstance(outputs, list) or len(outputs) != expected:
        length = len(outputs) if isinstance(outputs, list) else "n/a"
        raise RuntimeError(
            f"SGLang returned unexpected batch shape: type={type(outputs)!r}, length={length}"
        )
    return outputs


def _vllm_inputs(sequences: Sequence[Sequence[int]]) -> list[Any]:
    try:
        from vllm import TokensPrompt
    except ImportError:
        try:
            from vllm.inputs import TokensPrompt
        except ImportError:
            TokensPrompt = None
    if TokensPrompt is None:
        return [{"prompt_token_ids": list(sequence)} for sequence in sequences]
    return [TokensPrompt(prompt_token_ids=list(sequence)) for sequence in sequences]


def _check_reserved_engine_args(engine: str, engine_args: Mapping[str, Any]) -> None:
    reserved = {
        "vllm": {"model", "tokenizer", "tensor_parallel_size", "max_logprobs"},
        "sglang": {"model_path", "tokenizer_path", "tp_size"},
    }[engine]
    conflicts = sorted(reserved.intersection(engine_args))
    if conflicts:
        raise ValueError(
            f"Engine args may not override verification-owned keys: {conflicts}"
        )


def _run_vllm_worker(
    config: dict[str, Any], sequences: Sequence[Sequence[int]], targets: Sequence[str]
) -> Any:
    import numpy as np
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    extra_args = dict(config["engine_args"])
    _check_reserved_engine_args("vllm", extra_args)
    engine_kwargs: dict[str, Any] = {
        "model": config["model"],
        "tokenizer": config["tokenizer"],
        "trust_remote_code": config["trust_remote_code"],
        "dtype": config["dtype"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "max_model_len": config["max_model_len"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "seed": config["seed"],
        "disable_log_stats": True,
        "max_logprobs": config["vocab_size"],
    }
    if config.get("revision"):
        engine_kwargs["revision"] = config["revision"]
        engine_kwargs["tokenizer_revision"] = (
            config.get("tokenizer_revision") or config["revision"]
        )
    elif config.get("tokenizer_revision"):
        engine_kwargs["tokenizer_revision"] = config["tokenizer_revision"]
    if config.get("quantization"):
        engine_kwargs["quantization"] = config["quantization"]
    engine_kwargs.update(extra_args)

    llm = LLM(**engine_kwargs)
    try:
        gpu_metadata = common.collect_gpu_metadata(torch, targets)
        sampling_kwargs = {
            "temperature": 0.0,
            "max_tokens": 1,
            "min_tokens": 1,
            "ignore_eos": True,
            "detokenize": False,
            "logprobs": -1,
        }
        supports_flat_logprobs = (
            "flat_logprobs" in inspect.signature(SamplingParams).parameters
        )
        if supports_flat_logprobs:
            sampling_kwargs["flat_logprobs"] = True
        sampling_params = SamplingParams(
            **sampling_kwargs,
        )
        engine_inputs = _vllm_inputs(sequences)
        rows = []
        start = time.perf_counter()
        exclusivity = None
        for offset in range(0, len(engine_inputs), config["batch_size"]):
            batch = engine_inputs[offset : offset + config["batch_size"]]
            outputs = llm.generate(batch, sampling_params, use_tqdm=False)
            if exclusivity is None:
                exclusivity = common.verify_runtime_exclusive(targets)
            rows.extend(
                _dense_from_vllm_output(output, config["vocab_size"], np)
                for output in outputs
            )
            print(
                f"COLLECT role={config['role']} engine=vllm samples={len(rows)}/{len(engine_inputs)}",
                flush=True,
            )
        matrix = np.stack(rows, axis=0)
        return matrix, {
            "engine_arguments": engine_kwargs,
            "runtime": common.runtime_metadata(torch, "vllm"),
            "engine_import_version": getattr(vllm, "__version__", None),
            "gpus": gpu_metadata,
            "runtime_exclusivity": exclusivity,
            "collection_seconds": time.perf_counter() - start,
            "full_vocabulary_transport": "flat_logprobs"
            if supports_flat_logprobs
            else "per-position mapping",
        }
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _run_sglang_worker(
    config: dict[str, Any], sequences: Sequence[Sequence[int]], targets: Sequence[str]
) -> Any:
    os.environ.setdefault("SGLANG_FORCE_STREAM_INTERVAL", "1")

    import numpy as np
    import sglang as sgl
    import torch

    extra_args = dict(config["engine_args"])
    _check_reserved_engine_args("sglang", extra_args)
    engine_kwargs: dict[str, Any] = {
        "model_path": config["model"],
        "tokenizer_path": config["tokenizer"],
        "trust_remote_code": config["trust_remote_code"],
        "dtype": config["dtype"],
        "context_length": config["max_model_len"],
        "tp_size": config["tensor_parallel_size"],
        "mem_fraction_static": config["mem_fraction_static"],
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "disable_decode_cuda_graph": True,
        "disable_prefill_cuda_graph": True,
        "enable_deterministic_inference": True,
        "random_seed": config["seed"],
        "log_level": "info",
    }
    if config.get("revision"):
        engine_kwargs["revision"] = config["revision"]
    if config.get("quantization"):
        engine_kwargs["quantization"] = config["quantization"]
    engine_kwargs.update(extra_args)

    engine = sgl.Engine(**engine_kwargs)
    try:
        gpu_metadata = common.collect_gpu_metadata(torch, targets)
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 1,
            "min_new_tokens": 1,
            "ignore_eos": True,
        }
        rows = []
        vocabulary_token_ids = list(range(config["vocab_size"]))
        start = time.perf_counter()
        exclusivity = None
        for offset in range(0, len(sequences), config["batch_size"]):
            batch = list(sequences[offset : offset + config["batch_size"]])
            outputs = engine.generate(
                input_ids=batch,
                sampling_params=sampling_params,
                return_logprob=True,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=vocabulary_token_ids,
            )
            if exclusivity is None:
                exclusivity = common.verify_runtime_exclusive(targets)
            for output in _as_sglang_batch(outputs, len(batch)):
                token_logprobs = output.get("meta_info", {}).get(
                    "output_token_ids_logprobs"
                )
                if not isinstance(token_logprobs, list) or len(token_logprobs) != 1:
                    raise RuntimeError(
                        "SGLang did not return one output token-ID logprob distribution."
                    )
                rows.append(
                    _dense_from_sglang_top_logprobs(
                        token_logprobs[0], config["vocab_size"], np
                    )
                )
            print(
                f"COLLECT role={config['role']} engine=sglang samples={len(rows)}/{len(sequences)}",
                flush=True,
            )
        matrix = np.stack(rows, axis=0)
        return matrix, {
            "engine_arguments": engine_kwargs,
            "runtime": common.runtime_metadata(torch, "sglang"),
            "sglang_kernel": common.package_version("sglang-kernel"),
            "gpus": gpu_metadata,
            "runtime_exclusivity": exclusivity,
            "collection_seconds": time.perf_counter() - start,
        }
    finally:
        engine.shutdown()


def _atomic_write_npz(path: Path, matrix: Any, metadata: dict[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            log_probs=matrix,
            metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
        )
    os.replace(temporary, path)


def worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Internal full-vocabulary engine worker."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    inputs = json.loads(Path(config["inputs_path"]).read_text())
    sequences = inputs["sequences"]
    if common.sha256_json(sequences) != inputs["sequence_sha256"]:
        raise RuntimeError(
            "Prepared input hash does not match its serialized token IDs."
        )

    targets = common.visible_gpu_targets(int(config["tensor_parallel_size"]))
    preflight_wait = common.wait_for_strict_idle_gate(
        targets,
        timeout_seconds=float(config["idle_wait_timeout_seconds"]),
        poll_interval_seconds=float(config["idle_wait_poll_seconds"]),
        sample_count=int(config["idle_samples"]),
        interval_seconds=float(config["idle_interval_seconds"]),
        memory_tolerance_mib=int(config["idle_memory_tolerance_mib"]),
    )
    preflight = preflight_wait["strict_idle_preflight"]
    started_at = common.utc_now()
    if config["engine"] == "vllm":
        matrix, engine_metadata = _run_vllm_worker(config, sequences, targets)
    else:
        matrix, engine_metadata = _run_sglang_worker(config, sequences, targets)
    if list(matrix.shape) != [len(sequences), int(config["vocab_size"])]:
        raise RuntimeError(f"Unexpected collected matrix shape: {matrix.shape}")
    metadata = {
        "schema_version": 1,
        "role": config["role"],
        "engine": config["engine"],
        "model": config["model"],
        "revision": config.get("revision"),
        "tokenizer": config["tokenizer"],
        "tokenizer_revision": config.get("tokenizer_revision"),
        "quantization": config.get("quantization"),
        "dtype": config["dtype"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "input_sequence_sha256": inputs["sequence_sha256"],
        "sample_count": len(sequences),
        "vocab_size": config["vocab_size"],
        "matrix_dtype": str(matrix.dtype),
        "strict_idle_preflight": preflight,
        "strict_idle_preflight_wait": preflight_wait,
        "started_at_utc": started_at,
        "completed_at_utc": common.utc_now(),
        "exact_worker_command": common.exact_command(),
        **engine_metadata,
    }
    _atomic_write_npz(Path(config["artifact_path"]), matrix, metadata)
    return 0


def _normalized_log_probs(row: Any, numpy_module: Any) -> Any:
    row = numpy_module.asarray(row)
    # Canonicalize artifacts produced by SGLang versions that serialize
    # probability-zero entries as float32-min.  This also makes recomputation
    # from older raw artifacts mathematically equivalent to using -inf.
    if numpy_module.issubdtype(row.dtype, numpy_module.floating):
        sentinel = float(numpy_module.finfo(numpy_module.float32).min)
        if (row <= sentinel).any():
            row = row.astype(numpy_module.float64, copy=True)
            row[row <= sentinel] = -numpy_module.inf
    if numpy_module.isnan(row).any() or numpy_module.isposinf(row).any():
        raise ValueError(
            "Log-probability rows may not contain NaN or positive infinity."
        )
    finite = numpy_module.isfinite(row)
    if not finite.any():
        raise ValueError("A log-probability row has no finite vocabulary entries.")
    maximum = float(row[finite].max())
    log_normalizer = maximum + float(
        numpy_module.log(numpy_module.exp(row[finite] - maximum).sum(dtype="float64"))
    )
    return row.astype(numpy_module.float64, copy=False) - log_normalizer


def _top_ids(row: Any, top_k: int, numpy_module: Any) -> Any:
    if top_k == row.size:
        candidates = numpy_module.arange(row.size)
    else:
        candidates = numpy_module.argpartition(row, -top_k)[-top_k:]
    order = numpy_module.lexsort((candidates, -row[candidates]))
    return candidates[order]


def _top_margin(row: Any, numpy_module: Any) -> float:
    if row.size < 2:
        return 0.0
    top_two = numpy_module.partition(row, -2)[-2:]
    return float(top_two.max() - top_two.min())


def _cosine(left: Any, right: Any, numpy_module: Any) -> float:
    denominator = float(
        numpy_module.linalg.norm(left) * numpy_module.linalg.norm(right)
    )
    if denominator == 0:
        return 1.0 if numpy_module.array_equal(left, right) else 0.0
    return float(numpy_module.dot(left, right) / denominator)


def _distribution(values: Sequence[float], numpy_module: Any) -> dict[str, float]:
    array = numpy_module.asarray(values, dtype=numpy_module.float64)
    if array.size == 0 or not numpy_module.isfinite(array).all():
        raise ValueError("Metric distributions must be non-empty and finite.")
    return {
        "mean": float(array.mean()),
        "median": float(numpy_module.quantile(array, 0.50)),
        "p95": float(numpy_module.quantile(array, 0.95)),
        "p99": float(numpy_module.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def compare_log_prob_matrices(
    native: Any,
    quantized: Any,
    positions: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    import numpy as np

    native = np.asarray(native)
    quantized = np.asarray(quantized)
    if native.ndim != 2 or native.shape != quantized.shape:
        raise ValueError(
            f"Expected equal two-dimensional matrices, received {native.shape} and {quantized.shape}."
        )
    if native.shape[0] != len(positions):
        raise ValueError(
            f"Position metadata has {len(positions)} rows for {native.shape[0]} matrix rows."
        )
    if not 1 <= top_k <= native.shape[1]:
        raise ValueError(f"top_k must be in [1, {native.shape[1]}].")

    kld_values = []
    total_variation_values = []
    centered_mae_values = []
    centered_rmse_values = []
    centered_cosine_values = []
    logprob_mae_values = []
    logprob_rmse_values = []
    margin_delta_values = []
    top_k_overlaps = []
    top1_agreements = []
    target_native_nll = []
    target_quantized_nll = []
    per_position = []

    for row_index, position in enumerate(positions):
        native_row = _normalized_log_probs(native[row_index], np)
        quantized_row = _normalized_log_probs(quantized[row_index], np)
        native_probability = np.exp(native_row)
        quantized_probability = np.exp(quantized_row)
        positive_native = native_probability > 0
        if np.isneginf(quantized_row[positive_native]).any():
            raise ValueError(
                f"Quantized model assigns zero probability where native is positive at row {row_index}."
            )
        kld = float(
            np.dot(
                native_probability[positive_native],
                native_row[positive_native] - quantized_row[positive_native],
            )
        )
        if kld < -1e-7:
            raise ValueError(
                f"Numerically invalid negative KLD at row {row_index}: {kld}"
            )
        kld = max(kld, 0.0)

        common_finite = np.isfinite(native_row) & np.isfinite(quantized_row)
        if not common_finite.any():
            raise ValueError(
                f"Native and quantized rows have no common finite entries at row {row_index}."
            )
        native_finite = native_row[common_finite]
        quantized_finite = quantized_row[common_finite]
        native_centered = native_finite - native_finite.mean()
        quantized_centered = quantized_finite - quantized_finite.mean()
        centered_delta = quantized_centered - native_centered
        logprob_delta = quantized_finite - native_finite

        native_top_ids = _top_ids(native_row, top_k, np)
        quantized_top_ids = _top_ids(quantized_row, top_k, np)
        native_top1 = int(native_top_ids[0])
        quantized_top1 = int(quantized_top_ids[0])
        top1_agreement = native_top1 == quantized_top1
        top_k_overlap = (
            len(set(native_top_ids.tolist()).intersection(quantized_top_ids.tolist()))
            / top_k
        )
        native_margin = _top_margin(native_row, np)
        quantized_margin = _top_margin(quantized_row, np)

        total_variation = float(
            0.5
            * np.abs(native_probability - quantized_probability).sum(dtype=np.float64)
        )
        centered_mae = float(np.abs(centered_delta).mean())
        centered_rmse = float(np.sqrt(np.square(centered_delta).mean()))
        logprob_mae = float(np.abs(logprob_delta).mean())
        logprob_rmse = float(np.sqrt(np.square(logprob_delta).mean()))
        margin_delta = quantized_margin - native_margin
        target_token_id = position.get("target_token_id")
        target_metrics = None
        if target_token_id is not None:
            target_token_id = int(target_token_id)
            native_nll = float(-native_row[target_token_id])
            quantized_nll = float(-quantized_row[target_token_id])
            if not np.isfinite([native_nll, quantized_nll]).all():
                raise ValueError(
                    f"Observed target token has non-finite NLL at row {row_index}."
                )
            target_native_nll.append(native_nll)
            target_quantized_nll.append(quantized_nll)
            target_metrics = {
                "token_id": target_token_id,
                "native_nll": native_nll,
                "quantized_nll": quantized_nll,
                "nll_delta": quantized_nll - native_nll,
            }

        record = {
            **dict(position),
            "kld_native_to_quantized": kld,
            "total_variation": total_variation,
            "native_top1_token_id": native_top1,
            "quantized_top1_token_id": quantized_top1,
            "top1_agreement": top1_agreement,
            "top_k": top_k,
            "top_k_overlap": top_k_overlap,
            "native_top_k_token_ids": [int(token_id) for token_id in native_top_ids],
            "quantized_top_k_token_ids": [
                int(token_id) for token_id in quantized_top_ids
            ],
            "native_top1_probability": float(native_probability[native_top1]),
            "quantized_top1_probability": float(quantized_probability[quantized_top1]),
            "centered_logit_mae": centered_mae,
            "centered_logit_rmse": centered_rmse,
            "centered_logit_cosine": _cosine(native_centered, quantized_centered, np),
            "log_probability_mae": logprob_mae,
            "log_probability_rmse": logprob_rmse,
            "native_top1_margin": native_margin,
            "quantized_top1_margin": quantized_margin,
            "top1_margin_delta": margin_delta,
            "common_finite_vocab_entries": int(common_finite.sum()),
            "target_token": target_metrics,
        }
        per_position.append(record)
        kld_values.append(kld)
        total_variation_values.append(total_variation)
        centered_mae_values.append(centered_mae)
        centered_rmse_values.append(centered_rmse)
        centered_cosine_values.append(record["centered_logit_cosine"])
        logprob_mae_values.append(logprob_mae)
        logprob_rmse_values.append(logprob_rmse)
        margin_delta_values.append(abs(margin_delta))
        top_k_overlaps.append(top_k_overlap)
        top1_agreements.append(top1_agreement)

    summary = {
        "sample_count": native.shape[0],
        "vocab_size": native.shape[1],
        "kld_direction": "KL(P_native || P_quantized)",
        "kld": _distribution(kld_values, np),
        "top1_agreement": float(np.mean(top1_agreements)),
        "top1_flip_count": int(len(top1_agreements) - sum(top1_agreements)),
        "top_k": top_k,
        "top_k_overlap": _distribution(top_k_overlaps, np),
        "total_variation": _distribution(total_variation_values, np),
        "centered_logit_mae": _distribution(centered_mae_values, np),
        "centered_logit_rmse": _distribution(centered_rmse_values, np),
        "centered_logit_cosine": _distribution(centered_cosine_values, np),
        "log_probability_mae": _distribution(logprob_mae_values, np),
        "log_probability_rmse": _distribution(logprob_rmse_values, np),
        "absolute_top1_margin_delta": _distribution(margin_delta_values, np),
    }
    if target_native_nll:
        native_nll = np.asarray(target_native_nll, dtype=np.float64)
        quantized_nll = np.asarray(target_quantized_nll, dtype=np.float64)
        summary["observed_next_token_nll"] = {
            "sample_count": int(native_nll.size),
            "native_mean": float(native_nll.mean()),
            "quantized_mean": float(quantized_nll.mean()),
            "mean_delta": float((quantized_nll - native_nll).mean()),
        }
    worst = sorted(
        per_position, key=lambda item: item["kld_native_to_quantized"], reverse=True
    )[:20]
    return {
        "summary": summary,
        "per_position": per_position,
        "worst_positions_by_kld": worst,
    }


def _load_artifact(path: Path) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        matrix = payload["log_probs"]
        metadata = json.loads(str(payload["metadata_json"].item()))
    return matrix, metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_reused_native_artifact(
    args: argparse.Namespace,
    path: Path,
    inputs: Mapping[str, Any],
    vocab_size: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import numpy as np

    resolved = path.expanduser().resolve()
    matrix, metadata = _load_artifact(resolved)
    native_tp, _ = _role_tensor_parallel_sizes(args)
    expected = {
        "role": "native",
        "engine": args.engine,
        "model": args.native_model,
        "revision": args.native_revision,
        "tokenizer": args.tokenizer or args.native_model,
        "tokenizer_revision": args.tokenizer_revision,
        "quantization": args.native_quantization,
        "dtype": args.dtype,
        "tensor_parallel_size": native_tp,
        "input_sequence_sha256": inputs["sequence_sha256"],
        "sample_count": len(inputs["sequences"]),
        "vocab_size": vocab_size,
        "matrix_dtype": "float32",
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    expected_shape = (len(inputs["sequences"]), vocab_size)
    if tuple(matrix.shape) != expected_shape:
        mismatches["matrix_shape"] = {
            "expected": list(expected_shape),
            "observed": list(matrix.shape),
        }
    if matrix.dtype != np.float32:
        mismatches["numpy_dtype"] = {"expected": "float32", "observed": str(matrix.dtype)}
    if np.isnan(matrix).any() or np.isposinf(matrix).any():
        mismatches["matrix_values"] = {
            "expected": "no NaN or positive infinity",
            "observed": "invalid values present",
        }
    if not np.isfinite(matrix).any(axis=1).all():
        mismatches["finite_rows"] = {
            "expected": "at least one finite value in every row",
            "observed": "one or more all-nonfinite rows",
        }
    foreign = metadata.get("runtime_exclusivity", {}).get("foreign_compute_processes")
    if foreign != []:
        mismatches["runtime_exclusivity.foreign_compute_processes"] = {
            "expected": [],
            "observed": foreign,
        }
    if mismatches:
        raise ValueError(f"Reusable native artifact failed validation: {mismatches}")

    provenance = {
        "source_path": str(resolved),
        "source_sha256": _sha256_file(resolved),
        "validated_fields": expected,
        "matrix_shape": list(matrix.shape),
        "runtime_engine_import_version": metadata.get("engine_import_version"),
        "runtime_packages": metadata.get("runtime", {}).get("packages"),
        "original_started_at_utc": metadata.get("started_at_utc"),
        "original_completed_at_utc": metadata.get("completed_at_utc"),
    }
    return matrix, metadata, provenance


def _worker_config(
    args: argparse.Namespace,
    *,
    role: str,
    model: str,
    revision: str | None,
    quantization: str | None,
    tensor_parallel_size: int,
    engine_args: dict[str, Any],
    inputs: dict[str, Any],
    vocab_size: int,
    artifacts_dir: Path,
) -> tuple[dict[str, Any], Path]:
    artifact_path = artifacts_dir / f"{role}.log_probs.npz"
    config = {
        "role": role,
        "engine": args.engine,
        "model": model,
        "revision": revision,
        "quantization": quantization,
        "tokenizer": args.tokenizer or args.native_model,
        "tokenizer_revision": args.tokenizer_revision,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": tensor_parallel_size,
        # vLLM may run a short tokenizer/chat-template warmup internally.  Keep
        # enough headroom for that warmup even when a smoke test samples only a
        # tiny prefix; real comparisons still use the exact captured token IDs.
        "max_model_len": max(
            64,
            max(len(sequence) for sequence in inputs["sequences"])
            + 1
            + args.context_margin,
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "mem_fraction_static": args.mem_fraction_static,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "vocab_size": vocab_size,
        "engine_args": engine_args,
        "inputs_path": str((artifacts_dir / "inputs.json").resolve()),
        "artifact_path": str(artifact_path.resolve()),
        "idle_samples": args.idle_samples,
        "idle_interval_seconds": args.idle_interval_seconds,
        "idle_memory_tolerance_mib": args.idle_memory_tolerance_mib,
        "idle_wait_timeout_seconds": args.engine_transition_timeout_seconds,
        "idle_wait_poll_seconds": args.engine_transition_poll_seconds,
    }
    config_path = artifacts_dir / f"{role}.worker.json"
    common.atomic_write_json(config_path, config)
    return config, config_path


def _live_rows(
    args: argparse.Namespace,
    gpu_ids_by_role: Mapping[str, str],
    states: Mapping[str, str],
    sample_count: int,
) -> list[tuple[str, ...]]:
    return [
        (
            gpu_ids_by_role[role],
            "native" if role == "native" else "post-quant",
            "native"
            if role == "native"
            else (args.quantized_quantization or "checkpoint metadata"),
            "unknown",
            "unavailable",
            f"full-vocab rows {sample_count}; KLD pending",
            states[role],
        )
        for role in ("native", "quantized")
    ]


def _print_live_table(
    args: argparse.Namespace,
    gpu_ids_by_role: Mapping[str, str],
    states: Mapping[str, str],
    sample_count: int,
) -> None:
    print(
        common.render_ascii_table(
            (
                "GPU ID(s)",
                "Candidate",
                "Model/decoder bits",
                "Endpoint bits",
                "Size MB",
                "Metrics",
                "State",
            ),
            _live_rows(args, gpu_ids_by_role, states, sample_count),
        ),
        flush=True,
    )


def _run_worker_process(
    args: argparse.Namespace,
    config_path: Path,
    role: str,
    states: dict[str, str],
    gpu_ids_by_role: Mapping[str, str],
    sample_count: int,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "__worker__",
        "--config",
        str(config_path),
    ]
    states[role] = "running"
    process = subprocess.Popen(command)
    while True:
        try:
            return_code = process.wait(timeout=60)
            break
        except subprocess.TimeoutExpired:
            _print_live_table(args, gpu_ids_by_role, states, sample_count)
    if return_code != 0:
        states[role] = f"failed ({return_code})"
        raise subprocess.CalledProcessError(return_code, command)
    states[role] = "collected"
    _print_live_table(args, gpu_ids_by_role, states, sample_count)


def _quality_gates(
    args: argparse.Namespace, summary: Mapping[str, Any]
) -> dict[str, Any]:
    gates = []
    if args.max_kld_mean is not None:
        observed = float(summary["kld"]["mean"])
        gates.append(
            {
                "metric": "kld.mean",
                "operator": "<=",
                "threshold": args.max_kld_mean,
                "observed": observed,
                "passed": observed <= args.max_kld_mean,
            }
        )
    if args.min_top1_agreement is not None:
        observed = float(summary["top1_agreement"])
        gates.append(
            {
                "metric": "top1_agreement",
                "operator": ">=",
                "threshold": args.min_top1_agreement,
                "observed": observed,
                "passed": observed >= args.min_top1_agreement,
            }
        )
    return {"passed": all(gate["passed"] for gate in gates), "gates": gates}


def _print_final_summary(report: Mapping[str, Any]) -> None:
    summary = report["comparison"]["summary"]
    print(
        common.render_ascii_table(
            (
                "Samples",
                "Vocabulary",
                "Mean KLD",
                "P95 KLD",
                "P99 KLD",
                "Top-1",
                "Top-k overlap",
                "Logit RMSE",
            ),
            [
                (
                    summary["sample_count"],
                    summary["vocab_size"],
                    f"{summary['kld']['mean']:.8g}",
                    f"{summary['kld']['p95']:.8g}",
                    f"{summary['kld']['p99']:.8g}",
                    f"{summary['top1_agreement']:.4%}",
                    f"{summary['top_k_overlap']['mean']:.4%}",
                    f"{summary['centered_logit_rmse']['mean']:.8g}",
                )
            ],
        )
    )


def run_main(args: argparse.Namespace) -> int:
    _validate_args(args)
    native_tp, quantized_tp = _role_tensor_parallel_sizes(args)
    active_gpu_count = quantized_tp if args.reuse_native_artifact is not None else max(native_tp, quantized_tp)
    targets = common.visible_gpu_targets(active_gpu_count)
    parent_preflight = common.strict_idle_gate(
        targets,
        sample_count=args.idle_samples,
        interval_seconds=args.idle_interval_seconds,
        memory_tolerance_mib=args.idle_memory_tolerance_mib,
    )
    artifacts_dir = args.artifacts_dir or args.output.with_name(
        f"{args.output.stem}_artifacts"
    )
    inputs, vocab_size = prepare_inputs(args, artifacts_dir)
    sample_count = len(inputs["sequences"])
    estimated_bytes = 2 * sample_count * vocab_size * 4
    estimated_gib = estimated_bytes / (1024**3)
    if estimated_gib > args.max_artifact_gib:
        raise ValueError(
            f"Two float32 log-probability artifacts need about {estimated_gib:.3f} GiB, above "
            f"--max-artifact-gib={args.max_artifact_gib}. Reduce prompts/positions or raise the explicit limit."
        )

    common_engine_args = common.load_json_object(args.engine_args_json)
    native_engine_args = {
        **common_engine_args,
        **common.load_json_object(args.native_engine_args_json),
    }
    quantized_engine_args = {
        **common_engine_args,
        **common.load_json_object(args.quantized_engine_args_json),
    }
    native_config, native_config_path = _worker_config(
        args,
        role="native",
        model=args.native_model,
        revision=args.native_revision,
        quantization=args.native_quantization,
        tensor_parallel_size=native_tp,
        engine_args=native_engine_args,
        inputs=inputs,
        vocab_size=vocab_size,
        artifacts_dir=artifacts_dir,
    )
    quantized_config, quantized_config_path = _worker_config(
        args,
        role="quantized",
        model=args.quantized_model,
        revision=args.quantized_revision,
        quantization=args.quantized_quantization,
        tensor_parallel_size=quantized_tp,
        engine_args=quantized_engine_args,
        inputs=inputs,
        vocab_size=vocab_size,
        artifacts_dir=artifacts_dir,
    )

    reused_native = None
    if args.reuse_native_artifact is not None:
        reused_native = _validate_reused_native_artifact(
            args, args.reuse_native_artifact, inputs, vocab_size
        )
        native_config["artifact_path"] = str(args.reuse_native_artifact.expanduser().resolve())
        common.atomic_write_json(native_config_path, native_config)

    active_gpu_ids = [str(sample["physical_id"]) for sample in parent_preflight[: len(targets)]]
    gpu_ids_by_role = {
        "native": (
            f"artifact TP{native_tp}"
            if reused_native is not None
            else ",".join(active_gpu_ids[:native_tp])
        ),
        "quantized": ",".join(active_gpu_ids[:quantized_tp]),
    }
    states = {"native": "pending", "quantized": "pending"}
    report: dict[str, Any] = {
        "schema_version": 2,
        "success": False,
        "started_at_utc": common.utc_now(),
        "exact_command": common.exact_command(),
        "engine": args.engine,
        "native_model": args.native_model,
        "quantized_model": args.quantized_model,
        "tokenizer": args.tokenizer or args.native_model,
        "dtype": args.dtype,
        "tensor_parallel_size": native_tp if native_tp == quantized_tp else None,
        "native_tensor_parallel_size": native_tp,
        "quantized_tensor_parallel_size": quantized_tp,
        "active_gpu_count": active_gpu_count,
        "parent_strict_idle_preflight": parent_preflight,
        "inputs_path": str((artifacts_dir / "inputs.json").resolve()),
        "input_sequence_sha256": inputs["sequence_sha256"],
        "sample_count": sample_count,
        "vocab_size": vocab_size,
        "estimated_raw_artifact_gib": estimated_gib,
        "artifacts": {
            "native": native_config["artifact_path"],
            "quantized": quantized_config["artifact_path"],
            "native_worker_config": str(native_config_path.resolve()),
            "quantized_worker_config": str(quantized_config_path.resolve()),
        },
    }
    if reused_native is not None:
        report["native_artifact_reuse"] = reused_native[2]
    common.atomic_write_json(args.output, report)
    try:
        if reused_native is None:
            _run_worker_process(
                args, native_config_path, "native", states, gpu_ids_by_role, sample_count
            )
            report["engine_transition_idle_gate"] = common.wait_for_strict_idle_gate(
                targets,
                timeout_seconds=args.engine_transition_timeout_seconds,
                poll_interval_seconds=args.engine_transition_poll_seconds,
                sample_count=args.idle_samples,
                interval_seconds=args.idle_interval_seconds,
                memory_tolerance_mib=args.idle_memory_tolerance_mib,
            )
        else:
            states["native"] = "reused (validated)"
            report["engine_transition_idle_gate"] = {
                "state": "not_applicable",
                "reason": "The native engine was not relaunched; a validated native artifact was reused.",
            }
            _print_live_table(args, gpu_ids_by_role, states, sample_count)
        common.atomic_write_json(args.output, report)
        _run_worker_process(
            args, quantized_config_path, "quantized", states, gpu_ids_by_role, sample_count
        )
        native_matrix, native_metadata = _load_artifact(
            Path(native_config["artifact_path"])
        )
        quantized_matrix, quantized_metadata = _load_artifact(
            Path(quantized_config["artifact_path"])
        )
        for metadata in (native_metadata, quantized_metadata):
            if metadata["input_sequence_sha256"] != inputs["sequence_sha256"]:
                raise RuntimeError(
                    "An engine artifact was collected from different input token IDs."
                )
            if metadata["engine"] != args.engine:
                raise RuntimeError(
                    "An engine artifact does not match the requested backend."
                )
        comparison = compare_log_prob_matrices(
            native_matrix,
            quantized_matrix,
            inputs["positions"],
            top_k=args.top_k,
        )
        report.update(
            {
                "native_artifact_metadata": native_metadata,
                "quantized_artifact_metadata": quantized_metadata,
                "comparison": comparison,
            }
        )
        report["quality_gates"] = _quality_gates(args, comparison["summary"])
        report["success"] = bool(report["quality_gates"]["passed"])
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["completed_at_utc"] = common.utc_now()
        common.atomic_write_json(args.output, report)
        if "comparison" in report:
            _print_final_summary(report)
    return 0 if report["success"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "__worker__":
        return worker_main(arguments[1:])
    return run_main(build_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
