# SPDX-License-Identifier: Apache-2.0

"""Stage-1 QVQ transform-folding search on real Llama 3.2 1B weights.

This intentionally uses disjoint calibration, validation, and dense-parity
streams.  It screens aligned submatrices from every decoder projection before
any expensive propagated or full-model quantization promotion.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import (
    pack_qvq_bank_ids,
    pack_qvq_binary_bank_ids,
    quantize_qvq_linear,
)
from gptqmodel.quantization.qvq_transform_llama import LlamaQVQTransformImplementor
from gptqmodel.quantization.qvq_transform_planner import QVQTransformPlanner

MODEL_ID = "ModelCloud/Llama3.2-1B-Instruct"
DEFAULT_ARMS = ("A0", "A1", "A3", "A4", "A6")
DEFAULT_RATES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
CALIBRATION_TEXTS = (
    "A small compiler lowers typed expressions into a compact instruction stream.",
    "Explain why ocean tides depend on both the Moon and the Sun.",
    "The customer ordered tea, bread, fruit, and a blue ceramic bowl.",
    "Write a careful invariant for a ring buffer with one producer and one consumer.",
)
VALIDATION_TEXTS = (
    "A telescope collects light and focuses it onto a detector.",
    "Compare a balanced binary tree with a hash table for ordered queries.",
    "The train crossed the valley just before the evening storm arrived.",
)
DENSE_EVALUATION_TEXT = (
    "Give two concise reasons to test numerical software on held-out data."
)


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _csv_rates(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(","))
    if not values or any(item not in DEFAULT_RATES for item in values):
        raise argparse.ArgumentTypeError("rates must be a subset of 1,1.5,2,2.5,3,3.5")
    return values


def _model() -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()


def _capture_inputs(model, tokenizer, names, texts, device):
    captured = defaultdict(list)
    modules = dict(model.named_modules())
    handles = []
    for name in names:

        def hook(_module, inputs, _name=name):
            captured[_name].append(
                inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
            )

        handles.append(modules[name].register_forward_pre_hook(hook))
    try:
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(rows, dim=0) for name, rows in captured.items()}


def _dense_logits(model, tokenizer, device):
    encoded = tokenizer(DENSE_EVALUATION_TEXT, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        return model(**encoded, use_cache=False).logits.detach().float().cpu()


def _relative_l2(actual, reference) -> float:
    return float(
        torch.linalg.vector_norm(actual - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    )


def _dense_metrics(actual, reference):
    delta = actual - reference
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": _relative_l2(actual, reference),
        "top1_identity": float(
            (actual.argmax(-1) == reference.argmax(-1)).float().mean()
        ),
    }


def _local_metrics(weight, reconstructed, validation_inputs):
    reference = validation_inputs @ weight.T
    actual = validation_inputs @ reconstructed.T
    reference_probability = F.softmax(reference, dim=-1)
    local_kl = F.kl_div(
        F.log_softmax(actual, dim=-1),
        reference_probability,
        reduction="batchmean",
    )
    target = reference.argmax(dim=-1, keepdim=True)
    result = {
        "weight_relative_l2": _relative_l2(reconstructed, weight),
        "local_output_relative_l2": _relative_l2(actual, reference),
        "local_kl": float(local_kl),
    }
    for k in (1, 5, 10):
        result[f"top{k}"] = float(
            (actual.topk(k, dim=-1).indices == target).any(dim=-1).float().mean()
        )
    return result


def _storage(result, weight_elements):
    packed_bank_ids = None
    if result.bank_ids is not None:
        packed_bank_ids = (
            pack_qvq_binary_bank_ids(result.bank_ids)
            if result.bank_selector_bits == 1
            else pack_qvq_bank_ids(result.bank_ids)
        )
    tensors = (
        result.trellis,
        result.SU,
        result.SV,
        packed_bank_ids,
        result.bank_alt_id,
    )
    bits = sum(
        tensor.numel() * tensor.element_size() * 8
        for tensor in tensors
        if tensor is not None
    )
    return {"serialized_bytes": bits // 8, "effective_bpw": bits / weight_elements}


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=_csv_strings, default=DEFAULT_ARMS)
    parser.add_argument("--rates", type=_csv_rates, default=DEFAULT_RATES)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "artifacts/qvq_rotation_stage1_m4max.json",
    )
    args = parser.parse_args()
    if args.block < 16 or args.block % 16:
        parser.error("--block must be positive and divisible by 16")
    if not torch.backends.mps.is_available():
        parser.error("this Apple search requires an available PyTorch MPS device")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    payload = {
        "schema": "qvq.rotation-folding.stage1.v1",
        "model": MODEL_ID,
        "device": "mps",
        "block": args.block,
        "layer": args.layer,
        "seed": args.seed,
        "streams": {
            "calibration": list(CALIBRATION_TEXTS),
            "validation": list(VALIDATION_TEXTS),
            "dense_evaluation": DENSE_EVALUATION_TEXT,
        },
        "arms": {},
        "results": [],
    }
    baseline_logits = None
    for arm in args.arms:
        started = time.perf_counter()
        model = _model()
        planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())
        plan = planner.build_transform_plan(arm)
        rewrite = planner.rewrite_dense_weights(plan, seed=args.seed)
        descriptors = [
            descriptor
            for descriptor in plan.modules
            if descriptor.module_name.startswith(f"model.layers.{args.layer}.")
        ]
        if len(descriptors) != 7:
            raise RuntimeError(
                f"expected seven layer projections, got {len(descriptors)}"
            )
        model.to("mps")
        logits = _dense_logits(model, tokenizer, "mps")
        if baseline_logits is None:
            baseline_logits = logits
        dense = _dense_metrics(logits, baseline_logits)
        names = [descriptor.module_name for descriptor in descriptors]
        calibration = _capture_inputs(model, tokenizer, names, CALIBRATION_TEXTS, "mps")
        validation = _capture_inputs(model, tokenizer, names, VALIDATION_TEXTS, "mps")
        modules = dict(model.named_modules())
        arm_record = {
            "description": plan.description,
            "online_hadamards_per_block": plan.online_hadamards_per_block,
            "other_online_transforms_per_block": plan.other_online_transforms_per_block,
            "folded_transforms_per_block": plan.folded_transforms_per_block,
            "rewrite": rewrite,
            "dense_parity": dense,
        }
        payload["arms"][arm] = arm_record
        print(
            f"{arm}: H={plan.online_hadamards_per_block}, dense rel={dense['relative_l2']:.3e}, "
            f"top1={dense['top1_identity']:.3f}",
            flush=True,
        )
        for descriptor in descriptors:
            module = modules[descriptor.module_name]
            block = min(args.block, module.in_features, module.out_features)
            block -= block % 16
            weight = (
                module.weight.detach()[:block, :block]
                .to(device="mps", dtype=torch.float32)
                .contiguous()
            )
            cal = calibration[descriptor.module_name][:, :block].to(
                "mps", dtype=torch.float32
            )
            val = validation[descriptor.module_name][:, :block].to(
                "mps", dtype=torch.float32
            )
            hessian = cal.T @ cal / max(1, cal.shape[0])
            input_hadamard = descriptor.input_transform.is_online_full_hadamard
            output_hadamard = descriptor.output_transform.is_online_full_hadamard
            for bits in args.rates:
                fit_started = time.perf_counter()
                result = quantize_qvq_linear(
                    weight,
                    hessian,
                    bits=bits,
                    seed=args.seed,
                    input_hadamard=input_hadamard,
                    output_hadamard=output_hadamard,
                    rounding="block_ldlq",
                    vector_size=2,
                    trellis_window=16,
                    v2b2_p32=True,
                    bank_count=2,
                )
                metrics = _local_metrics(weight, result.weight.float(), val)
                row = {
                    "arm": arm,
                    "rate": bits,
                    "module": descriptor.module_name,
                    "role": descriptor.role.value,
                    "input_transform": descriptor.input_transform.kind.value,
                    "input_placement": descriptor.input_transform.placement.value,
                    "output_transform": descriptor.output_transform.kind.value,
                    "output_placement": descriptor.output_transform.placement.value,
                    "input_hadamard": input_hadamard,
                    "output_hadamard": output_hadamard,
                    "shape": list(weight.shape),
                    "calibration_tokens": int(cal.shape[0]),
                    "validation_tokens": int(val.shape[0]),
                    "fit_seconds": time.perf_counter() - fit_started,
                    **_storage(result, weight.numel()),
                    **metrics,
                }
                payload["results"].append(row)
                print(
                    f"  W{bits:g} {descriptor.role.value:12s} "
                    f"wrel={metrics['weight_relative_l2']:.4f} "
                    f"orel={metrics['local_output_relative_l2']:.4f} "
                    f"KL={metrics['local_kl']:.4g}",
                    flush=True,
                )
                _write(args.json, payload)
                del result
        arm_record["elapsed_seconds"] = time.perf_counter() - started
        _write(args.json, payload)
        del modules, model, calibration, validation, logits
        gc.collect()
        torch.mps.empty_cache()
    print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
