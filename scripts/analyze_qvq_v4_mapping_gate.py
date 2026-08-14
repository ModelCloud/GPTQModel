#!/usr/bin/env python3
"""Held-out two-layer gate for offline V4 mapping candidates.

Candidate trellises are evaluated through their dense reconstructed weights.
They are never installed as QVQLinear modules or serialized because the
current checkpoint format has no mapping metadata.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if __package__:
    from scripts.analyze_gptq_low_bit_grid import (
        TARGET_SUFFIXES,
        capture_calibration_hessians,
        capture_forward,
        decoder_layers,
        load_nm_calibration_batches,
        load_nm_evaluation_batch,
        target_modules,
        tensor_metrics,
    )
else:
    from analyze_gptq_low_bit_grid import (
        TARGET_SUFFIXES,
        capture_calibration_hessians,
        capture_forward,
        decoder_layers,
        load_nm_calibration_batches,
        load_nm_evaluation_batch,
        target_modules,
        tensor_metrics,
    )

from gptqmodel.quantization.qvq import (
    QVQQuantizationTelemetry,
    default_qvq_trellis_batch_size,
    quantize_qvq_linear,
)
from gptqmodel.quantization.qvq_v4_candidates import V4Candidate, v4_candidate_codebook


CANDIDATES = (
    V4Candidate("canonical", 0xA5A5),
    V4Candidate("mask-5a5a", 0x5A5A),
    V4Candidate("mask-3c3c", 0x3C3C),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bits", type=float, default=2.0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--calibration-rows", type=int, default=128)
    parser.add_argument("--evaluation-rows", type=int, default=128)
    parser.add_argument("--evaluation-row-offset", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.bits != 2.0:
        raise ValueError("the first V4 mapping gate is intentionally fixed to W2")
    if args.layers != 2:
        raise ValueError("the first V4 mapping gate requires exactly two decoder layers")
    if args.evaluation_row_offset < args.calibration_rows:
        raise ValueError("held-out evaluation rows must not overlap calibration rows")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    source_layers = int(config.num_hidden_layers)
    config.num_hidden_layers = args.layers
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration, calibration_stats = load_nm_calibration_batches(
        tokenizer,
        config,
        dataset_path=args.dataset,
        rows=args.calibration_rows,
        concat_size=None,
        batch_size=1,
    )
    evaluation, evaluation_stats = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=args.dataset,
        row_offset=args.evaluation_row_offset,
        rows=args.evaluation_rows,
        max_length=args.max_length,
    )
    evaluation = {name: value.to(device) for name, value in evaluation.items()}
    modules = target_modules(model, layer_count=args.layers)
    expected_modules = args.layers * len(TARGET_SUFFIXES)
    if len(modules) != expected_modules or len(decoder_layers(model)) != args.layers:
        raise ValueError(f"expected {expected_modules} target modules across exactly {args.layers} layers")

    print("Capturing calibration Hessians", flush=True)
    hessians, sample_counts = capture_calibration_hessians(model, calibration, modules, device=device)
    if set(sample_counts.values()) != {calibration_stats["valid_tokens"]}:
        raise AssertionError("Hessian sample counts do not match valid calibration tokens")
    print("Capturing dense held-out outputs", flush=True)
    dense_logits, evaluation_inputs, dense_outputs = capture_forward(
        model,
        evaluation,
        modules,
        capture_inputs=True,
        layer_count=args.layers,
    )
    original_weights = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}
    report = {
        "settings": {
            "model": str(args.model),
            "source_layers": source_layers,
            "layers": args.layers,
            "bits": args.bits,
            "seed": args.seed,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "calibration": calibration_stats,
            "evaluation": evaluation_stats,
            "serialization": "disabled; dense reconstruction gate only",
        },
        "arms": {},
    }
    trellis_batch_size = default_qvq_trellis_batch_size(args.bits, device)
    for candidate in CANDIDATES:
        started = time.perf_counter()
        print(f"Quantizing {candidate.name}", flush=True)
        codebook = v4_candidate_codebook(candidate, device=device)
        reconstructions = {}
        local = {}
        for index, (name, module) in enumerate(modules.items(), start=1):
            result = quantize_qvq_linear(
                original_weights[name].to(device),
                hessians[name].to(device),
                bits=args.bits,
                seed=args.seed,
                vector_size=4,
                trellis_batch_size=trellis_batch_size,
                experimental_codebook=codebook,
                telemetry=QVQQuantizationTelemetry(),
            )
            reconstruction = result.weight.detach().cpu()
            reconstructions[name] = reconstruction
            bias = None if module.bias is None else module.bias.detach().cpu()
            local_output = torch.nn.functional.linear(evaluation_inputs[name], reconstruction, bias)
            local[name] = tensor_metrics(dense_outputs[name], local_output, normalize_distribution=True)
            local[name]["quantization_telemetry"] = result.telemetry
            print(f"{candidate.name}: module {index}/{len(modules)} {name}", flush=True)

        with torch.no_grad():
            for name, module in modules.items():
                module.weight.copy_(reconstructions[name].to(device))
        quantized_logits, _, live_outputs = capture_forward(
            model,
            evaluation,
            modules,
            capture_inputs=False,
            layer_count=args.layers,
        )
        report["arms"][candidate.name] = {
            "xor_mask": f"0x{candidate.xor_mask:04x}",
            "seconds": time.perf_counter() - started,
            "local": local,
            "layers": {
                f"layer.{index}": tensor_metrics(
                    dense_outputs[f"layer.{index}.hidden"],
                    live_outputs[f"layer.{index}.hidden"],
                    normalize_distribution=True,
                )
                for index in range(args.layers)
            },
            "logits": tensor_metrics(dense_logits, quantized_logits, normalize_distribution=False),
        }
        logits = report["arms"][candidate.name]["logits"]
        print(
            f"{candidate.name}: KLD={logits['kl_forward']['mean']:.8f} "
            f"JSD={logits['jensen_shannon']['mean']:.8f} "
            f"top1={logits['top1_agreement']:.6f} top5={logits['top5_overlap']['mean']:.6f}",
            flush=True,
        )
        with torch.no_grad():
            for name, module in modules.items():
                module.weight.copy_(original_weights[name].to(device))
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        del codebook, reconstructions, quantized_logits, live_outputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
