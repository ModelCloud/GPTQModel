# SPDX-License-Identifier: Apache-2.0

"""Progressive full-model W2 validation for promoted QVQ folding arms.

The runner deliberately recaptures activations after every dependency group:
Q/K/V, attention O, gate/up, then down.  Earlier quantized layers therefore
participate in calibration of every later layer, and down_proj is fitted from
the actually quantized SwiGLU product rather than stale dense activations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import DownloadMode, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import (  # noqa: E402
    pack_qvq_bank_ids,
    pack_qvq_binary_bank_ids,
    quantize_qvq_linear,
)
from gptqmodel.quantization.qvq_transform_llama import LlamaQVQTransformImplementor  # noqa: E402
from gptqmodel.quantization.qvq_transform_planner import (  # noqa: E402
    ProjectionRole,
    QVQTransformPlanner,
)

MODEL_ID = "ModelCloud/Llama3.2-1B-Instruct"
WIKITEXT_ID = "Salesforce/wikitext"
ROLE_GROUPS = (
    (
        ProjectionRole.ATTENTION_Q,
        ProjectionRole.ATTENTION_K,
        ProjectionRole.ATTENTION_V,
    ),
    (ProjectionRole.ATTENTION_O,),
    (ProjectionRole.MLP_GATE, ProjectionRole.MLP_UP),
    (ProjectionRole.MLP_DOWN,),
)


def _csv(value):
    result = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one arm")
    return result


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _select_wikitext(split, count, seed):
    dataset = load_dataset(
        WIKITEXT_ID,
        "wikitext-2-raw-v1",
        split=split,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    candidates = []
    for row_index, row in enumerate(dataset):
        text = " ".join(row["text"].split())
        if len(text) >= 128:
            candidates.append((row_index, text))
    generator = random.Random(seed)
    generator.shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(
            f"WikiText {split} has only {len(candidates)} usable rows; requested {count}"
        )
    return [
        {
            "row": row_index,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        }
        for row_index, text in candidates[:count]
    ]


def _encode(tokenizer, samples, max_length):
    encoded = []
    for sample in samples:
        item = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded.append({key: value.cpu() for key, value in item.items()})
        sample["tokens"] = int(item["input_ids"].numel())
    return encoded


def _sample_metadata(samples):
    return [
        {key: value for key, value in sample.items() if key != "text"}
        for sample in samples
    ]


def _device_inputs(encoded, device):
    return {key: value.to(device) for key, value in encoded.items()}


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@torch.inference_mode()
def _dense_reference(model, encoded, layers, device):
    logits = []
    layer_outputs = {index: [] for index in range(len(layers))}
    handles = []
    for index, layer in enumerate(layers):

        def hook(_module, _inputs, output, _index=index):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_outputs[_index].append(
                hidden.detach().float().cpu().reshape(-1, hidden.shape[-1])
            )

        handles.append(layer.register_forward_hook(hook))
    try:
        for item in encoded:
            output = model(**_device_inputs(item, device), use_cache=False)
            logits.append(
                output.logits.detach()
                .float()
                .cpu()
                .reshape(-1, output.logits.shape[-1])
            )
    finally:
        for handle in handles:
            handle.remove()
    return logits, layer_outputs


@torch.inference_mode()
def _evaluate(model, encoded, dense_logits, dense_layers, device, *, layer_indices=()):
    token_count = 0
    kl_sum = 0.0
    logits_delta_sq = 0.0
    logits_reference_sq = 0.0
    max_abs = 0.0
    exact_logit_values = 0
    logit_values = 0
    top_hits = {1: 0, 5: 0, 10: 0}
    layer_delta_sq = defaultdict(float)
    layer_reference_sq = defaultdict(float)
    per_text = []
    current_layers = {}
    handles = []
    for layer_index in layer_indices:
        layer = model.model.layers[layer_index]

        def hook(_module, _inputs, output, _index=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            current_layers[_index] = (
                hidden.detach().float().reshape(-1, hidden.shape[-1])
            )

        handles.append(layer.register_forward_hook(hook))
    try:
        for sample_index, item in enumerate(encoded):
            current_layers.clear()
            output = model(**_device_inputs(item, device), use_cache=False)
            actual = output.logits.detach().float().reshape(-1, output.logits.shape[-1])
            target_logits = dense_logits[sample_index].to(device)
            tokens = int(actual.shape[0])
            text_kl_sum = F.kl_div(
                F.log_softmax(actual, dim=-1),
                F.softmax(target_logits, dim=-1),
                reduction="sum",
            )
            delta = actual - target_logits
            text_exact_logit_values = int((actual == target_logits).sum())
            text_logit_values = actual.numel()
            text_delta_sq = float(delta.square().sum())
            text_reference_sq = float(target_logits.square().sum())
            target = target_logits.argmax(dim=-1, keepdim=True)
            text_hits = {}
            for k in top_hits:
                hits = int((actual.topk(k, dim=-1).indices == target).any(dim=-1).sum())
                top_hits[k] += hits
                text_hits[f"top{k}"] = hits / tokens
            per_text.append(
                {
                    "tokens": tokens,
                    "final_kl": float(text_kl_sum) / tokens,
                    "logits_relative_l2": math.sqrt(text_delta_sq / text_reference_sq),
                    "exact_logits_fraction": text_exact_logit_values
                    / text_logit_values,
                    **text_hits,
                }
            )
            token_count += tokens
            kl_sum += float(text_kl_sum)
            logits_delta_sq += text_delta_sq
            logits_reference_sq += text_reference_sq
            exact_logit_values += text_exact_logit_values
            logit_values += text_logit_values
            max_abs = max(max_abs, float(delta.abs().max()))
            for layer_index in layer_indices:
                actual_layer = current_layers[layer_index]
                target_layer = dense_layers[layer_index][sample_index].to(device)
                layer_delta_sq[layer_index] += float(
                    (actual_layer - target_layer).square().sum()
                )
                layer_reference_sq[layer_index] += float(target_layer.square().sum())
            del output, actual, target_logits, delta
    finally:
        for handle in handles:
            handle.remove()
    result = {
        "tokens": token_count,
        "final_kl": kl_sum / token_count,
        "logits_relative_l2": math.sqrt(logits_delta_sq / logits_reference_sq),
        "max_abs_logits_delta": max_abs,
        "exact_logits_fraction": exact_logit_values / logit_values,
        "per_text": per_text,
    }
    for k, hits in top_hits.items():
        result[f"top{k}"] = hits / token_count
    if layer_indices:
        result["layer_output_relative_l2"] = {
            str(index): math.sqrt(layer_delta_sq[index] / layer_reference_sq[index])
            for index in layer_indices
        }
    return result


@torch.inference_mode()
def _capture(model, encoded, names, device):
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
        for item in encoded:
            model(**_device_inputs(item, device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(values) for name, values in rows.items()}


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
    output_delta = quantized_output - dense_output
    output_relative_l2 = float(output_delta.norm() / dense_output.norm())
    local_kl = float(
        F.kl_div(
            F.log_softmax(quantized_output, dim=-1),
            F.softmax(dense_output, dim=-1),
            reduction="batchmean",
        )
    )
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
    record = {
        "module": descriptor.module_name,
        "role": descriptor.role.value,
        "shape": list(weight.shape),
        "input_hadamard": result.input_hadamard,
        "output_hadamard": result.output_hadamard,
        "weight_relative_l2": float((result.weight - weight).norm() / weight.norm()),
        "output_relative_l2": output_relative_l2,
        "local_kl": local_kl,
        "effective_bpw": storage_bits / weight.numel(),
        "storage_bits": storage_bits,
        "weight_elements": weight.numel(),
        "fit_seconds": fit_seconds,
    }
    with torch.inference_mode():
        module.weight.copy_(result.weight)
    del result, hessian, inputs, dense_output, quantized_output, output_delta
    _empty_cache(device)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=_csv, default=("A0", "A25"))
    parser.add_argument(
        "--bits", type=float, default=2.0, choices=(1, 1.5, 2, 2.5, 3, 3.5)
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="preserve completed arms in an existing compatible JSON artifact",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if device_name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    device = torch.device(device_name)
    if args.json is None:
        device_suffix = "cuda" if device.type == "cuda" else "m4max"
        args.json = (
            REPO_ROOT / f"artifacts/qvq_rotation_full16_a0_a25_w2_{device_suffix}.json"
        )
    if args.calibration_samples <= 0 or args.validation_samples <= 0:
        parser.error("sample counts must be positive")
    if args.sequence_length <= 0:
        parser.error("sequence length must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    calibration_samples = _select_wikitext("train", args.calibration_samples, args.seed)
    validation_samples = _select_wikitext(
        "validation", args.validation_samples, args.seed + 1
    )
    calibration = _encode(tokenizer, calibration_samples, args.sequence_length)
    validation = _encode(tokenizer, validation_samples, args.sequence_length)

    reference_started = time.perf_counter()
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
    layer_count = len(reference_model.model.layers)
    model_revision = getattr(reference_model.config, "_commit_hash", None)
    if layer_count != 16:
        raise RuntimeError(
            f"expected Llama 3.2 1B to have 16 layers, found {layer_count}"
        )
    dense_logits, dense_layers = _dense_reference(
        reference_model,
        validation,
        reference_model.model.layers,
        device,
    )
    del reference_model
    gc.collect()
    _empty_cache(device)

    new_payload = {
        "schema": "qvq.rotation-folding.full-model-propagation.v1",
        "status": "running",
        "model": args.model,
        "model_revision": model_revision,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "Apple MPS"
        ),
        "bits": args.bits,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        "layer_count": layer_count,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "calibration_source": f"{WIKITEXT_ID}/wikitext-2-raw-v1:train",
        "validation_source": f"{WIKITEXT_ID}/wikitext-2-raw-v1:validation",
        "final_evaluation_source": "reserved wikitext test split (not read)",
        "calibration": _sample_metadata(calibration_samples),
        "validation": _sample_metadata(validation_samples),
        "reference_seconds": time.perf_counter() - reference_started,
        "arms": {},
    }
    if args.resume and args.json.exists():
        payload = json.loads(args.json.read_text(encoding="utf-8"))
        expected = {
            "schema": new_payload["schema"],
            "model": new_payload["model"],
            "device": new_payload["device"],
            "bits": new_payload["bits"],
            "layer_count": new_payload["layer_count"],
            "seed": new_payload["seed"],
            "sequence_length": new_payload["sequence_length"],
            "calibration": new_payload["calibration"],
            "validation": new_payload["validation"],
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"resume artifact is incompatible: {mismatches}")
        prior_revision = payload.get("model_revision")
        if prior_revision is not None and prior_revision != model_revision:
            raise RuntimeError(
                "resume artifact has a different model revision: "
                f"{prior_revision!r} != {model_revision!r}"
            )
        for key in (
            "model_revision",
            "device_name",
            "torch_version",
            "cuda_version",
            "device_capability",
        ):
            payload[key] = new_payload[key]
        payload["status"] = "running"
        payload["resume_reference_seconds"] = time.perf_counter() - reference_started
    else:
        payload = new_payload
    _write(args.json, payload)

    for arm in args.arms:
        if payload["arms"].get(arm, {}).get("status") == "complete":
            print(
                f"{arm}: already complete; preserving checkpointed result", flush=True
            )
            continue
        arm_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
        planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())
        plan = planner.build_transform_plan(arm)
        rewrite = planner.rewrite_dense_weights(plan, seed=args.seed)
        model.to(device)
        dense_parity = _evaluate(
            model,
            validation,
            dense_logits,
            dense_layers,
            device,
            layer_indices=range(layer_count),
        )
        arm_payload = {
            "description": plan.description,
            "online_hadamards_per_block": plan.online_hadamards_per_block,
            "rewrite": rewrite,
            "dense_parity": dense_parity,
            "modules": [],
            "trajectory": [],
            "completed_layers": 0,
            "status": "running",
        }
        payload["arms"][arm] = arm_payload
        _write(args.json, payload)
        descriptors_by_layer = defaultdict(list)
        for descriptor in plan.modules:
            layer_index = int(descriptor.module_name.split(".")[2])
            descriptors_by_layer[layer_index].append(descriptor)
        modules = dict(model.named_modules())

        for layer_index in range(layer_count):
            layer_descriptors = descriptors_by_layer[layer_index]
            for roles in ROLE_GROUPS:
                descriptors = [item for item in layer_descriptors if item.role in roles]
                captured = _capture(
                    model,
                    calibration,
                    [item.module_name for item in descriptors],
                    device,
                )
                for descriptor in descriptors:
                    record = _quantize_module(
                        modules[descriptor.module_name],
                        descriptor,
                        captured[descriptor.module_name],
                        args.bits,
                        args.seed,
                        device,
                    )
                    arm_payload["modules"].append(record)
                    print(
                        f"{arm} L{layer_index:02d} {descriptor.role.value}: "
                        f"{record['fit_seconds']:.1f}s, wrel={record['weight_relative_l2']:.4f}, "
                        f"orel={record['output_relative_l2']:.4f}",
                        flush=True,
                    )
                del captured
                gc.collect()
                _empty_cache(device)
            trajectory = _evaluate(
                model,
                validation,
                dense_logits,
                dense_layers,
                device,
                layer_indices=(layer_index,),
            )
            trajectory["through_layer"] = layer_index
            arm_payload["trajectory"].append(trajectory)
            arm_payload["completed_layers"] = layer_index + 1
            _write(args.json, payload)
            print(
                f"{arm} through L{layer_index:02d}: KL={trajectory['final_kl']:.6f}, "
                f"logits_rel={trajectory['logits_relative_l2']:.5f}, "
                f"top1={trajectory['top1']:.5f}",
                flush=True,
            )

        full_metrics = _evaluate(
            model,
            validation,
            dense_logits,
            dense_layers,
            device,
            layer_indices=range(layer_count),
        )
        storage_bits = sum(item["storage_bits"] for item in arm_payload["modules"])
        weight_elements = sum(
            item["weight_elements"] for item in arm_payload["modules"]
        )
        arm_payload.update(
            {
                "metrics": full_metrics,
                "effective_bpw": storage_bits / weight_elements,
                "elapsed_seconds": time.perf_counter() - arm_started,
                "status": "complete",
            }
        )
        _write(args.json, payload)
        del model
        gc.collect()
        _empty_cache(device)

    payload["status"] = "complete"
    _write(args.json, payload)


if __name__ == "__main__":
    main()
