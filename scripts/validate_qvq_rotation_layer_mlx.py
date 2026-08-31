# SPDX-License-Identifier: Apache-2.0

"""Promoted one-layer propagation test for QVQ folding arms on MPS."""

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


def _csv(value):
    result = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one arm")
    return result


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _forward(model, tokenizer, texts, device, *, layer=None):
    logits = []
    layer_outputs = []
    handle = None
    if layer is not None:
        handle = layer.register_forward_hook(
            lambda _module, _inputs, output: layer_outputs.append(
                (output[0] if isinstance(output, tuple) else output)
                .detach()
                .float()
                .cpu()
                .reshape(
                    -1,
                    output[0].shape[-1]
                    if isinstance(output, tuple)
                    else output.shape[-1],
                )
            )
        )
    try:
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model(**encoded, use_cache=False)
            logits.append(
                output.logits.detach()
                .float()
                .cpu()
                .reshape(-1, output.logits.shape[-1])
            )
    finally:
        if handle is not None:
            handle.remove()
    return torch.cat(logits), None if layer is None else torch.cat(layer_outputs)


def _capture(model, tokenizer, names, texts, device):
    rows = defaultdict(list)
    modules = dict(model.named_modules())
    handles = []
    for name in names:

        def hook(_module, inputs, _name=name):
            rows[_name].append(
                inputs[0].detach().float().cpu().reshape(-1, inputs[0].shape[-1])
            )

        handles.append(modules[name].register_forward_pre_hook(hook))
    try:
        _forward(model, tokenizer, texts, device)
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(values) for name, values in rows.items()}


def _metrics(actual_logits, dense_logits, actual_layer, dense_layer):
    target = dense_logits.argmax(dim=-1, keepdim=True)
    result = {
        "final_kl": float(
            F.kl_div(
                F.log_softmax(actual_logits, dim=-1),
                F.softmax(dense_logits, dim=-1),
                reduction="batchmean",
            )
        ),
        "logits_relative_l2": float(
            (actual_logits - dense_logits).norm() / dense_logits.norm()
        ),
        "layer_output_relative_l2": float(
            (actual_layer - dense_layer).norm() / dense_layer.norm()
        ),
    }
    for k in (1, 5, 10):
        result[f"top{k}"] = float(
            (actual_logits.topk(k, dim=-1).indices == target).any(dim=-1).float().mean()
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=_csv, default=("A0", "A22"))
    parser.add_argument(
        "--bits", type=float, default=2.0, choices=(1, 1.5, 2, 2.5, 3, 3.5)
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "artifacts/qvq_rotation_layer0_w2_m4max.json",
    )
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    payload = {
        "schema": "qvq.rotation-folding.layer-propagation.v1",
        "model": MODEL_ID,
        "device": "mps",
        "bits": args.bits,
        "layer": args.layer,
        "calibration": list(CALIBRATION_TEXTS),
        "validation": list(VALIDATION_TEXTS),
        "arms": {},
    }
    for arm in args.arms:
        arm_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
        planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())
        plan = planner.build_transform_plan(arm)
        rewrite = planner.rewrite_dense_weights(plan, seed=args.seed)
        descriptors = [
            item
            for item in plan.modules
            if item.module_name.startswith(f"model.layers.{args.layer}.")
        ]
        model.to("mps")
        layer = model.model.layers[args.layer]
        dense_logits, dense_layer = _forward(
            model, tokenizer, VALIDATION_TEXTS, "mps", layer=layer
        )
        calibration = _capture(
            model,
            tokenizer,
            [item.module_name for item in descriptors],
            CALIBRATION_TEXTS,
            "mps",
        )
        modules = dict(model.named_modules())
        module_records = []
        for descriptor in descriptors:
            module = modules[descriptor.module_name]
            inputs = calibration[descriptor.module_name].to("mps")
            hessian = inputs.T @ inputs / inputs.shape[0]
            weight = module.weight.detach().float()
            started = time.perf_counter()
            result = quantize_qvq_linear(
                weight,
                hessian,
                bits=args.bits,
                seed=args.seed,
                input_hadamard=descriptor.input_transform.is_online_full_hadamard,
                output_hadamard=descriptor.output_transform.is_online_full_hadamard,
                rounding="block_ldlq",
                vector_size=2,
                trellis_window=16,
                v2b2_p32=True,
                bank_count=2,
            )
            torch.mps.synchronize()
            fit_seconds = time.perf_counter() - started
            relative_l2 = float((result.weight - weight).norm() / weight.norm())
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
            storage_bits = sum(
                tensor.numel() * tensor.element_size() * 8
                for tensor in tensors
                if tensor is not None
            )
            with torch.inference_mode():
                module.weight.copy_(result.weight)
            module_record = {
                "module": descriptor.module_name,
                "role": descriptor.role.value,
                "shape": list(weight.shape),
                "input_hadamard": result.input_hadamard,
                "output_hadamard": result.output_hadamard,
                "weight_relative_l2": relative_l2,
                "effective_bpw": storage_bits / weight.numel(),
                "fit_seconds": fit_seconds,
            }
            module_records.append(module_record)
            print(
                f"{arm} {descriptor.role.value}: {fit_seconds:.1f}s, wrel={relative_l2:.4f}",
                flush=True,
            )
            del result, hessian, inputs
            torch.mps.empty_cache()
        actual_logits, actual_layer = _forward(
            model, tokenizer, VALIDATION_TEXTS, "mps", layer=layer
        )
        metrics = _metrics(actual_logits, dense_logits, actual_layer, dense_layer)
        payload["arms"][arm] = {
            "description": plan.description,
            "online_hadamards_per_block": plan.online_hadamards_per_block,
            "rewrite": rewrite,
            "modules": module_records,
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - arm_started,
        }
        _write(args.json, payload)
        print(f"{arm} propagated: {metrics}", flush=True)
        del model, calibration, dense_logits, dense_layer, actual_logits, actual_layer
        gc.collect()
        torch.mps.empty_cache()
    _write(args.json, payload)


if __name__ == "__main__":
    main()
