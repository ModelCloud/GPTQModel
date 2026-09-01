#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run packed full-model CUDA rotation-arm quality and decode benchmarks.

Quantization follows the progressive dense-reconstruction protocol used by
``validate_qvq_rotation_full_model_mlx.py``.  Pack-ready P32 tensors are kept
in memory, then installed into real QVQLinear modules for CUDA quality and
cached-decode measurements.  Research-only folded-axis checkpoints are never
serialized.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import DownloadMode, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear  # noqa: E402
from gptqmodel.quantization.qvq import (  # noqa: E402
    pack_qvq_binary_bank_ids,
    quantize_qvq_linear,
)
from gptqmodel.quantization.qvq_transform_llama import (  # noqa: E402
    LlamaQVQTransformImplementor,
)
from gptqmodel.quantization.qvq_transform_planner import (  # noqa: E402
    QVQTransformPlan,
    QVQTransformPlanner,
)
from gptqmodel.quantization.qvq_transform_runtime import (  # noqa: E402
    install_qvq_grouped_p32_input_transforms,
    install_qvq_shared_input_transforms,
)
from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda  # noqa: E402
from scripts.validate_qvq_rotation_full_model_mlx import (  # noqa: E402
    MODEL_ID,
    ROLE_GROUPS,
    WIKITEXT_ID,
    _capture,
    _csv,
    _device_inputs,
    _empty_cache,
    _encode,
    _evaluate,
    _sample_metadata,
    _select_wikitext,
    _synchronize,
    _write,
)


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _tensor_sha256(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _validation_streams(*, stream_count, samples_per_stream, seed):
    dataset = load_dataset(
        WIKITEXT_ID,
        "wikitext-2-raw-v1",
        split="validation",
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    candidates = []
    for row_index, row in enumerate(dataset):
        text = " ".join(row["text"].split())
        if len(text) >= 128:
            candidates.append((row_index, text))
    used_rows = set()
    streams = []
    for stream_index in range(stream_count):
        shuffled = list(candidates)
        random.Random(seed + 1 + stream_index).shuffle(shuffled)
        selected = []
        for row_index, text in shuffled:
            if row_index in used_rows:
                continue
            selected.append(
                {
                    "row": row_index,
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text,
                }
            )
            used_rows.add(row_index)
            if len(selected) == samples_per_stream:
                break
        if len(selected) != samples_per_stream:
            raise RuntimeError(
                f"validation stream {stream_index} has only {len(selected)} disjoint rows"
            )
        streams.append(selected)
    return streams


@torch.inference_mode()
def _dense_logits(model, encoded, device):
    rows = []
    for item in encoded:
        output = model(**_device_inputs(item, device), use_cache=False)
        rows.append(
            output.logits.detach().float().cpu().reshape(-1, output.logits.shape[-1])
        )
    return rows


def _runtime_tensors(result):
    packed_bank_ids = pack_qvq_binary_bank_ids(result.bank_ids)
    tensors = {
        "trellis": result.trellis.detach().cpu().contiguous(),
        "SU": result.SU.detach().cpu().contiguous(),
        "SV": result.SV.detach().cpu().contiguous(),
        "bank_ids": packed_bank_ids.detach().cpu().contiguous(),
        "bank_alt_id": result.bank_alt_id.detach().cpu().contiguous(),
    }
    if result.bias is not None:
        tensors["bias"] = result.bias.detach().cpu().contiguous()
    return tensors


def _quantize_module(module, descriptor, inputs, bits, seed, device):
    inputs = inputs.to(device)
    weight = module.weight.detach().float()
    hessian = inputs.T @ inputs / inputs.shape[0]
    dense_output = inputs @ weight.T
    started = time.perf_counter()
    result = quantize_qvq_linear(
        weight,
        hessian,
        bits=bits,
        seed=seed,
        input_hadamard=descriptor.input_transform.is_online_full_hadamard,
        output_hadamard=descriptor.output_transform.is_online_full_hadamard,
        rounding="block_ldlq",
        vector_size=2,
        trellis_window=16,
        v2b2_p32=True,
        bank_count=2,
    )
    _synchronize(device)
    fit_seconds = time.perf_counter() - started
    quantized_output = inputs @ result.weight.T
    runtime_tensors = _runtime_tensors(result)
    storage_bits = sum(
        tensor.numel() * tensor.element_size() * 8
        for tensor in runtime_tensors.values()
    )
    record = {
        "module": descriptor.module_name,
        "role": descriptor.role.value,
        "shape": list(weight.shape),
        "input_hadamard": result.input_hadamard,
        "output_hadamard": result.output_hadamard,
        "weight_relative_l2": float((result.weight - weight).norm() / weight.norm()),
        "output_relative_l2": float(
            (quantized_output - dense_output).norm() / dense_output.norm()
        ),
        "effective_bpw": storage_bits / weight.numel(),
        "storage_bits": storage_bits,
        "weight_elements": weight.numel(),
        "fit_seconds": fit_seconds,
        "tensor_sha256": {
            name: _tensor_sha256(tensor) for name, tensor in runtime_tensors.items()
        },
    }
    with torch.inference_mode():
        module.weight.copy_(result.weight)
    del result, hessian, inputs, dense_output, quantized_output
    _empty_cache(device)
    return record, runtime_tensors


def _install_packed_modules(model, module_payloads, descriptors, device):
    model.half()
    installed = {}
    for module_name, tensors in module_payloads.items():
        current = model.get_submodule(module_name)
        descriptor = descriptors[module_name]
        if not isinstance(current, torch.nn.Linear):
            raise TypeError(f"packed target {module_name!r} is not a dense Linear")
        replacement = QVQLinear(
            bits=2,
            in_features=current.in_features,
            out_features=current.out_features,
            bias=current.bias is not None,
            name=module_name,
            tensors={name: tensor.to(device) for name, tensor in tensors.items()},
            dtype=torch.float16,
            out_dtype=torch.float16,
            vector_size=2,
            trellis_window=16,
            bank_count=2,
            v2b2_p32=True,
            input_hadamard=descriptor.input_transform.is_online_full_hadamard,
            output_hadamard=descriptor.output_transform.is_online_full_hadamard,
        ).eval()
        replacement.post_init()
        parent_name, _, child_name = module_name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, replacement)
        installed[module_name] = replacement
    return installed


def _aggregate_evaluation(model, streams, dense_streams, device):
    per_stream = []
    all_encoded = []
    all_dense = []
    for stream_index, (encoded, dense_logits) in enumerate(
        zip(streams, dense_streams, strict=True)
    ):
        metrics = _evaluate(model, encoded, dense_logits, {}, device)
        metrics["stream"] = stream_index
        per_stream.append(metrics)
        all_encoded.extend(encoded)
        all_dense.extend(dense_logits)
    aggregate = _evaluate(model, all_encoded, all_dense, {}, device)
    return {"aggregate": aggregate, "streams": per_stream}


@torch.inference_mode()
def _prefill(model, input_ids, attention_mask):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )


def _timing_summary(event_ms, wall_ms, batch_size):
    return {
        "event_median_ms": statistics.median(event_ms),
        "event_p95_ms": _percentile(event_ms, 0.95),
        "event_min_ms": min(event_ms),
        "event_max_ms": max(event_ms),
        "event_stdev_ms": statistics.stdev(event_ms) if len(event_ms) > 1 else 0.0,
        "wall_median_ms": statistics.median(wall_ms),
        "wall_p95_ms": _percentile(wall_ms, 0.95),
        "wall_min_ms": min(wall_ms),
        "wall_max_ms": max(wall_ms),
        "wall_stdev_ms": statistics.stdev(wall_ms) if len(wall_ms) > 1 else 0.0,
        "tokens_per_second": batch_size * 1000 / statistics.median(wall_ms),
        "event_samples_ms": event_ms,
        "wall_samples_ms": wall_ms,
    }


@torch.inference_mode()
def _benchmark_decode(
    model,
    prompt,
    device,
    *,
    batch_sizes,
    prefill_warmup,
    prefill_iterations,
    decode_warmup,
    decode_iterations,
):
    rows = []
    base_ids = prompt["input_ids"].to(device)
    base_mask = prompt["attention_mask"].to(device)
    for batch_size in batch_sizes:
        input_ids = base_ids.repeat(batch_size, 1)
        attention_mask = base_mask.repeat(batch_size, 1)
        for _ in range(prefill_warmup):
            warm = _prefill(model, input_ids, attention_mask)
            del warm
        _synchronize(device)
        prefill_event_ms = []
        prefill_wall_ms = []
        for _ in range(prefill_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            _synchronize(device)
            wall_started = time.perf_counter()
            start.record()
            output = _prefill(model, input_ids, attention_mask)
            end.record()
            _synchronize(device)
            prefill_wall_ms.append((time.perf_counter() - wall_started) * 1000)
            prefill_event_ms.append(start.elapsed_time(end))
            del output

        output = _prefill(model, input_ids, attention_mask)
        past_key_values = output.past_key_values
        next_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        running_mask = attention_mask
        for _ in range(decode_warmup):
            running_mask = torch.cat(
                (
                    running_mask,
                    torch.ones(
                        (batch_size, 1), dtype=running_mask.dtype, device=device
                    ),
                ),
                dim=1,
            )
            output = model(
                input_ids=next_ids,
                attention_mask=running_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = output.past_key_values
            next_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        _synchronize(device)
        decode_event_ms = []
        decode_wall_ms = []
        for _ in range(decode_iterations):
            running_mask = torch.cat(
                (
                    running_mask,
                    torch.ones(
                        (batch_size, 1), dtype=running_mask.dtype, device=device
                    ),
                ),
                dim=1,
            )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            _synchronize(device)
            wall_started = time.perf_counter()
            start.record()
            output = model(
                input_ids=next_ids,
                attention_mask=running_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            end.record()
            _synchronize(device)
            decode_wall_ms.append((time.perf_counter() - wall_started) * 1000)
            decode_event_ms.append(start.elapsed_time(end))
            past_key_values = output.past_key_values
            next_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        rows.append(
            {
                "batch_size": batch_size,
                "prompt_tokens": int(input_ids.shape[1]),
                "prefill": _timing_summary(
                    prefill_event_ms, prefill_wall_ms, batch_size * input_ids.shape[1]
                ),
                "decode": _timing_summary(decode_event_ms, decode_wall_ms, batch_size),
            }
        )
        del output, past_key_values, next_ids, running_mask
        _empty_cache(device)
    return rows


def _combine_benchmark_cycles(cycles):
    by_batch = defaultdict(
        lambda: {
            "prefill_event": [],
            "prefill_wall": [],
            "decode_event": [],
            "decode_wall": [],
        }
    )
    prompt_tokens = {}
    for cycle in cycles:
        for row in cycle:
            batch_size = row["batch_size"]
            prompt_tokens[batch_size] = row["prompt_tokens"]
            by_batch[batch_size]["prefill_event"].extend(
                row["prefill"]["event_samples_ms"]
            )
            by_batch[batch_size]["prefill_wall"].extend(
                row["prefill"]["wall_samples_ms"]
            )
            by_batch[batch_size]["decode_event"].extend(
                row["decode"]["event_samples_ms"]
            )
            by_batch[batch_size]["decode_wall"].extend(row["decode"]["wall_samples_ms"])
    return [
        {
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens[batch_size],
            "prefill": _timing_summary(
                values["prefill_event"],
                values["prefill_wall"],
                batch_size * prompt_tokens[batch_size],
            ),
            "decode": _timing_summary(
                values["decode_event"], values["decode_wall"], batch_size
            ),
        }
        for batch_size, values in sorted(by_batch.items())
    ]


@torch.inference_mode()
def _benchmark_quant_linear_suite(
    model,
    descriptors_by_layer,
    device,
    *,
    batch_sizes,
    warmup,
    iterations,
):
    """Time the seven packed projections with graph-valid shared inputs."""

    rows = []
    grouped_calls = []
    for layer_index in sorted(descriptors_by_layer):
        layer_descriptors = descriptors_by_layer[layer_index]
        for roles in ROLE_GROUPS:
            descriptors = [item for item in layer_descriptors if item.role in roles]
            if not descriptors:
                continue
            grouped_calls.append(
                tuple(model.get_submodule(item.module_name) for item in descriptors)
            )

    for batch_size in batch_sizes:
        inputs = []
        for modules in grouped_calls:
            inputs.append(
                torch.randn(
                    (batch_size, modules[0].in_features),
                    device=device,
                    dtype=torch.float16,
                )
            )

        def invoke_suite():
            outputs = []
            for modules, x in zip(grouped_calls, inputs, strict=True):
                outputs.extend(module(x) for module in modules)
            return outputs

        for _ in range(warmup):
            outputs = invoke_suite()
            del outputs
        _synchronize(device)
        event_ms = []
        wall_ms = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            _synchronize(device)
            wall_started = time.perf_counter()
            start.record()
            outputs = invoke_suite()
            end.record()
            _synchronize(device)
            wall_ms.append((time.perf_counter() - wall_started) * 1000)
            event_ms.append(start.elapsed_time(end))
            del outputs
        rows.append(
            {
                "batch_size": batch_size,
                "quantized_layers": len(descriptors_by_layer),
                "packed_projections": sum(len(group) for group in grouped_calls),
                "timing": _timing_summary(event_ms, wall_ms, batch_size),
            }
        )
    return rows


def _combine_suite_cycles(cycles):
    by_batch = defaultdict(lambda: {"event": [], "wall": []})
    metadata = {}
    for cycle in cycles:
        for row in cycle:
            batch_size = row["batch_size"]
            metadata[batch_size] = {
                "quantized_layers": row["quantized_layers"],
                "packed_projections": row["packed_projections"],
            }
            by_batch[batch_size]["event"].extend(row["timing"]["event_samples_ms"])
            by_batch[batch_size]["wall"].extend(row["timing"]["wall_samples_ms"])
    return [
        {
            "batch_size": batch_size,
            **metadata[batch_size],
            "timing": _timing_summary(values["event"], values["wall"], batch_size),
        }
        for batch_size, values in sorted(by_batch.items())
    ]


def _paired_bootstrap(reference_metrics, candidate_metrics, *, samples=2000, seed=20260901):
    reference_rows = reference_metrics["aggregate"]["per_text"]
    candidate_rows = candidate_metrics["aggregate"]["per_text"]
    if len(reference_rows) != len(candidate_rows) or not reference_rows:
        raise ValueError("paired bootstrap requires aligned non-empty per-text metrics")
    generator = random.Random(seed)
    kl_deltas = []
    top1_deltas = []
    for _ in range(samples):
        indices = [generator.randrange(len(reference_rows)) for _ in reference_rows]
        tokens = sum(reference_rows[index]["tokens"] for index in indices)
        reference_kl = (
            sum(
                reference_rows[index]["final_kl"] * reference_rows[index]["tokens"]
                for index in indices
            )
            / tokens
        )
        candidate_kl = (
            sum(
                candidate_rows[index]["final_kl"] * candidate_rows[index]["tokens"]
                for index in indices
            )
            / tokens
        )
        reference_top1 = (
            sum(
                reference_rows[index]["top1"] * reference_rows[index]["tokens"]
                for index in indices
            )
            / tokens
        )
        candidate_top1 = (
            sum(
                candidate_rows[index]["top1"] * candidate_rows[index]["tokens"]
                for index in indices
            )
            / tokens
        )
        kl_deltas.append(candidate_kl - reference_kl)
        top1_deltas.append(candidate_top1 - reference_top1)
    return {
        "resamples": samples,
        "unit": "held-out text",
        "final_kl_delta_candidate_minus_reference": {
            "median": statistics.median(kl_deltas),
            "ci95": [_percentile(kl_deltas, 0.025), _percentile(kl_deltas, 0.975)],
        },
        "top1_delta_candidate_minus_reference": {
            "median": statistics.median(top1_deltas),
            "ci95": [
                _percentile(top1_deltas, 0.025),
                _percentile(top1_deltas, 0.975),
            ],
        },
    }


def _hardware_metadata():
    query = "index,name,uuid,pci.bus_id,compute_cap,memory.total,memory.free"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    values = [value.strip() for value in output.split(",")]
    return dict(zip(query.split(","), values, strict=True))


def _git_revision(ref):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", ref], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cuda_compute_processes():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, process_name, used_memory = [item.strip() for item in line.split(",", 2)]
        rows.append(
            {
                "pid": int(pid),
                "process_name": process_name,
                "used_memory_mib": None if used_memory == "[N/A]" else int(used_memory),
            }
        )
    return rows


def _assert_exclusive_cuda_process():
    processes = _cuda_compute_processes()
    foreign = [item for item in processes if item["pid"] != os.getpid()]
    if foreign:
        raise RuntimeError(f"CUDA benchmark requires an idle GPU; found {foreign}")
    return processes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=_csv, default=("A0", "A25"))
    parser.add_argument("--bits", type=float, default=2.0, choices=(2.0,))
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--validation-streams", type=int, default=3)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--decode-batches", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--prefill-warmup", type=int, default=2)
    parser.add_argument("--prefill-iterations", type=int, default=5)
    parser.add_argument("--decode-warmup", type=int, default=5)
    parser.add_argument("--decode-iterations", type=int, default=30)
    parser.add_argument("--module-suite-warmup", type=int, default=10)
    parser.add_argument("--module-suite-iterations", type=int, default=30)
    parser.add_argument("--timing-cycles", type=int, default=3)
    parser.add_argument(
        "--reuse-identical-a31-payloads",
        action="store_true",
        help="reuse A31's fitted P32 payloads for a later A41 runtime-only arm",
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT
        / "artifacts/qvq_rotation_full16_a0_a25_w2_packed_cuda_sm80.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 0):
        parser.error(
            f"this benchmark requires SM80, got SM{capability[0]}{capability[1]}"
        )
    if not 1 <= args.layers <= 16:
        parser.error("--layers must be in [1, 16]")
    positive = (
        args.calibration_samples,
        args.validation_streams,
        args.validation_samples,
        args.sequence_length,
        args.prefill_warmup,
        args.prefill_iterations,
        args.decode_warmup,
        args.decode_iterations,
        args.module_suite_warmup,
        args.module_suite_iterations,
        args.timing_cycles,
        *args.decode_batches,
    )
    if min(positive) <= 0:
        parser.error("sample, timing, sequence, and batch values must be positive")
    if not prewarm_qvq_cuda():
        parser.error("QVQ CUDA extension failed to load")
    startup_compute_processes = _assert_exclusive_cuda_process()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    calibration_samples = _select_wikitext("train", args.calibration_samples, args.seed)
    validation_sample_streams = _validation_streams(
        stream_count=args.validation_streams,
        samples_per_stream=args.validation_samples,
        seed=args.seed,
    )
    calibration = _encode(tokenizer, calibration_samples, args.sequence_length)
    validation_streams = [
        _encode(tokenizer, samples, args.sequence_length)
        for samples in validation_sample_streams
    ]
    validation_all = [item for stream in validation_streams for item in stream]

    reference_model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to(device)
    )
    if len(reference_model.model.layers) != 16:
        raise RuntimeError("expected a 16-layer Llama model")
    model_revision = getattr(reference_model.config, "_commit_hash", None)
    dense_logits_all = _dense_logits(reference_model, validation_all, device)
    dense_logits_streams = []
    offset = 0
    for stream in validation_streams:
        dense_logits_streams.append(dense_logits_all[offset : offset + len(stream)])
        offset += len(stream)
    del reference_model
    gc.collect()
    _empty_cache(device)

    payload = {
        "schema": "qvq.rotation-folding.packed-cuda-full-model.v1",
        "status": "running",
        "model": args.model,
        "model_revision": model_revision,
        "hardware": _hardware_metadata(),
        "device_capability": list(capability),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "repository_revision": _git_revision("HEAD"),
        "origin_main_revision": _git_revision("origin/main"),
        "runtime_path": "QVQLinear CUDA qvq_cuda_gemv V2B2-P32 planar with planned shared inputs",
        "startup_compute_processes": startup_compute_processes,
        "bits": args.bits,
        "quantized_layers": args.layers,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "calibration_source": f"{WIKITEXT_ID}/wikitext-2-raw-v1:train",
        "validation_source": f"{WIKITEXT_ID}/wikitext-2-raw-v1:validation",
        "final_evaluation_source": "reserved wikitext test split (not read)",
        "calibration": _sample_metadata(calibration_samples),
        "validation_streams": [
            _sample_metadata(samples) for samples in validation_sample_streams
        ],
        "arms": {},
    }
    _write(args.json, payload)

    packed_models = {}
    shared_states_by_arm = {}
    descriptors_by_arm = {}
    reusable_a31 = None
    for arm in args.arms:
        arm_started = time.perf_counter()
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                local_files_only=args.local_files_only,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(device)
        )
        planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())
        plan = planner.build_transform_plan(arm)
        rewrite = planner.rewrite_dense_weights(plan, seed=args.seed)
        dense_parity = _evaluate(model, validation_all, dense_logits_all, {}, device)
        descriptors_by_layer = defaultdict(list)
        descriptors = {}
        for descriptor in plan.modules:
            layer_index = int(descriptor.module_name.split(".")[2])
            if layer_index < args.layers:
                descriptors_by_layer[layer_index].append(descriptor)
                descriptors[descriptor.module_name] = descriptor
        arm_payload = {
            "description": plan.description,
            "online_hadamards_per_block": plan.online_hadamards_per_block,
            "rewrite": rewrite,
            "dense_parity": dense_parity,
            "modules": [],
            "completed_layers": 0,
            "status": "fitting",
        }
        payload["arms"][arm] = arm_payload
        _write(args.json, payload)
        reuse_a31 = (
            args.reuse_identical_a31_payloads
            and arm == "A41"
            and reusable_a31 is not None
        )
        if reuse_a31:
            source_descriptors = reusable_a31["descriptors"]
            candidate_signature = tuple(
                (
                    item.module_name,
                    item.input_transform,
                    item.output_transform,
                    item.local_input_scale,
                    item.local_output_scale,
                )
                for item in descriptors.values()
            )
            source_signature = tuple(
                (
                    item.module_name,
                    item.input_transform,
                    item.output_transform,
                    item.local_input_scale,
                    item.local_output_scale,
                )
                for item in source_descriptors.values()
            )
            if candidate_signature != source_signature:
                raise RuntimeError("A41 cannot reuse A31 payloads with a different transform plan")
            runtime_payloads = reusable_a31["runtime_payloads"]
            arm_payload["modules"] = copy.deepcopy(reusable_a31["modules"])
            arm_payload["completed_layers"] = args.layers
            arm_payload["fit_reused_from"] = "A31"
            arm_payload["reconstructed_quality"] = copy.deepcopy(
                reusable_a31["reconstructed_quality"]
            )
            print("A41: reusing byte-identical fitted A31 P32 payloads", flush=True)
        else:
            runtime_payloads = {}
            for layer_index in range(args.layers):
                layer_descriptors = descriptors_by_layer[layer_index]
                for roles in ROLE_GROUPS:
                    grouped = [item for item in layer_descriptors if item.role in roles]
                    names = [item.module_name for item in grouped]
                    captured = _capture(model, calibration, names, device)
                    for descriptor in grouped:
                        current = model.get_submodule(descriptor.module_name)
                        record, tensors = _quantize_module(
                            current,
                            descriptor,
                            captured[descriptor.module_name],
                            args.bits,
                            args.seed,
                            device,
                        )
                        arm_payload["modules"].append(record)
                        runtime_payloads[descriptor.module_name] = tensors
                        print(
                            f"{arm} L{layer_index:02d} {descriptor.role.value}: "
                            f"{record['fit_seconds']:.1f}s, "
                            f"wrel={record['weight_relative_l2']:.4f}, "
                            f"orel={record['output_relative_l2']:.4f}",
                            flush=True,
                        )
                    del captured
                    gc.collect()
                    _empty_cache(device)
                arm_payload["completed_layers"] = layer_index + 1
                _write(args.json, payload)

            arm_payload["reconstructed_quality"] = _aggregate_evaluation(
                model, validation_streams, dense_logits_streams, device
            )
            if arm == "A31" and args.reuse_identical_a31_payloads:
                reusable_a31 = {
                    "runtime_payloads": runtime_payloads,
                    "modules": copy.deepcopy(arm_payload["modules"]),
                    "descriptors": descriptors,
                    "reconstructed_quality": copy.deepcopy(
                        arm_payload["reconstructed_quality"]
                    ),
                }
        installed = _install_packed_modules(
            model, runtime_payloads, descriptors, device
        )
        if len(installed) != args.layers * 7:
            raise RuntimeError(
                f"expected {args.layers * 7} packed modules, installed {len(installed)}"
            )
        runtime_plan = QVQTransformPlan(
            arm=plan.arm,
            description=plan.description,
            modules=tuple(
                descriptor
                for descriptor in plan.modules
                if descriptor.module_name in descriptors
            ),
            metadata=plan.metadata,
        )
        if arm == "A41":
            shared_states = install_qvq_grouped_p32_input_transforms(
                model, runtime_plan
            )
        else:
            shared_states = install_qvq_shared_input_transforms(model, runtime_plan)
        grouped_metadata_bytes = sum(
            getattr(state, "metadata_overhead_bytes", 0)
            for state in shared_states.values()
        )
        arm_payload["shared_input_runtime"] = {
            "mode": "grouped_p32" if arm == "A41" else "shared_transform",
            "group_count": len(shared_states),
            "grouped_metadata_bytes": grouped_metadata_bytes,
            "groups": {
                basis_id: {
                    "module_names": list(state.group.module_names),
                    "su_sha256": _tensor_sha256(
                        installed[state.group.module_names[0]].SU
                    ),
                    "grouped_gemv": hasattr(state, "grouped_gemv_invocations"),
                    "metadata_overhead_bytes": getattr(
                        state, "metadata_overhead_bytes", 0
                    ),
                    "bank_alt_ids": (
                        state.bank_alt_ids.tolist()
                        if hasattr(state, "bank_alt_ids")
                        else None
                    ),
                    "bank_alt_boundaries": list(
                        getattr(state, "bank_alt_boundaries", ())
                    ),
                }
                for basis_id, state in shared_states.items()
            },
        }
        arm_payload["packed_quality"] = _aggregate_evaluation(
            model, validation_streams, dense_logits_streams, device
        )
        qvq_storage_bits = sum(
            item["storage_bits"] for item in arm_payload["modules"]
        )
        storage_bits = qvq_storage_bits + grouped_metadata_bytes * 8
        weight_elements = sum(
            item["weight_elements"] for item in arm_payload["modules"]
        )
        arm_payload.update(
            {
                "packed_modules": len(installed),
                "qvq_payload_bpw": qvq_storage_bits / weight_elements,
                "transform_metadata_bits": grouped_metadata_bytes * 8,
                "effective_bpw": storage_bits / weight_elements,
                "elapsed_seconds": time.perf_counter() - arm_started,
                "status": "complete",
            }
        )
        _write(args.json, payload)
        packed_models[arm] = model
        shared_states_by_arm[arm] = shared_states
        descriptors_by_arm[arm] = descriptors_by_layer
        del installed, runtime_payloads
        gc.collect()
        _empty_cache(device)

    payload["timing_compute_processes"] = _assert_exclusive_cuda_process()
    timing_cycles = {arm: [] for arm in args.arms}
    for cycle_index in range(args.timing_cycles):
        cycle_order = args.arms if cycle_index % 2 == 0 else tuple(reversed(args.arms))
        for arm in cycle_order:
            print(f"CUDA timing cycle {cycle_index + 1}: {arm}", flush=True)
            timing_cycles[arm].append(
                _benchmark_decode(
                    packed_models[arm],
                    calibration[0],
                    device,
                    batch_sizes=args.decode_batches,
                    prefill_warmup=args.prefill_warmup,
                    prefill_iterations=args.prefill_iterations,
                    decode_warmup=args.decode_warmup,
                    decode_iterations=args.decode_iterations,
                )
            )
            payload["arms"][arm]["decode_benchmark"] = {
                "cycles": timing_cycles[arm],
                "aggregate": _combine_benchmark_cycles(timing_cycles[arm]),
            }
            payload["arms"][arm]["shared_input_runtime"]["counters"] = {
                basis_id: {
                    "transform_invocations": state.transform_invocations,
                    "completed_cycles": state.completed_cycles,
                    "grouped_gemv_invocations": getattr(
                        state, "grouped_gemv_invocations", 0
                    ),
                    "pending_consumers": list(state.pending_consumers),
                }
                for basis_id, state in shared_states_by_arm[arm].items()
            }
            _write(args.json, payload)

    suite_cycles = {arm: [] for arm in args.arms}
    for cycle_index in range(args.timing_cycles):
        cycle_order = args.arms if cycle_index % 2 == 0 else tuple(reversed(args.arms))
        for arm in cycle_order:
            print(f"CUDA QuantLinear suite cycle {cycle_index + 1}: {arm}", flush=True)
            suite_cycles[arm].append(
                _benchmark_quant_linear_suite(
                    packed_models[arm],
                    descriptors_by_arm[arm],
                    device,
                    batch_sizes=args.decode_batches,
                    warmup=args.module_suite_warmup,
                    iterations=args.module_suite_iterations,
                )
            )
            payload["arms"][arm]["quant_linear_suite_benchmark"] = {
                "cycles": suite_cycles[arm],
                "aggregate": _combine_suite_cycles(suite_cycles[arm]),
            }
            _write(args.json, payload)

    reference_arm = "A0" if "A0" in payload["arms"] else args.arms[0]
    payload["comparisons"] = {}
    for candidate_arm in args.arms:
        if candidate_arm == reference_arm:
            continue
        payload["comparisons"][f"{candidate_arm}_minus_{reference_arm}"] = {
            "reference_arm": reference_arm,
            "candidate_arm": candidate_arm,
            "reconstructed": _paired_bootstrap(
                payload["arms"][reference_arm]["reconstructed_quality"],
                payload["arms"][candidate_arm]["reconstructed_quality"],
            ),
            "packed": _paired_bootstrap(
                payload["arms"][reference_arm]["packed_quality"],
                payload["arms"][candidate_arm]["packed_quality"],
            ),
        }
    payload["status"] = "complete"
    _write(args.json, payload)
    del packed_models
    gc.collect()
    _empty_cache(device)


if __name__ == "__main__":
    main()
