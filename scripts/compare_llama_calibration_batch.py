#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare exact Llama GPTQ artifacts across calibration batch sizes.

Each invocation runs in a fresh process.  Produce the batch-1 reference first,
then pass it through ``--reference`` for batch 2 or 4.  Generation equality is
deliberately not used as the accuracy contract: this tool fingerprints the
Hessian, inverse Hessian, reconstructed weights, scales, zero points, group
indices, and logical integer codes returned by GPTQ before packing.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/monster/data/model/Llama-3.2-1B-Instruct")
DEFAULT_CALIBRATION = Path("/monster/data/model/dataset/nm-calibration/llm.parquet")
DEFAULT_PHYSICAL_GPUS = (6, 7)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--geometry", choices=("natural", "fixed", "variable"), default="natural")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--group-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--auto-forward-data-parallel", action="store_true")
    parser.add_argument("--physical-gpus", type=int, nargs="+", default=list(DEFAULT_PHYSICAL_GPUS))
    args = parser.parse_args()
    for name in ("batch_size", "rows", "seq_len", "layers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _nvidia_inventory() -> list[dict[str, str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    fields = ("index", "pci_bus_id", "uuid", "name", "memory_total_mib", "memory_used_mib", "utilization_pct")
    return [dict(zip(fields, (item.strip() for item in row.split(",")))) for row in output.splitlines() if row.strip()]


def _idle_gate(physical_gpus: list[int], samples: int = 3) -> list[dict[str, str]]:
    accepted: list[dict[str, str]] = []
    for sample_index in range(samples):
        inventory = _nvidia_inventory()
        by_index = {int(item["index"]): item for item in inventory}
        current = []
        for physical_gpu in physical_gpus:
            if physical_gpu not in by_index:
                raise RuntimeError(f"Physical GPU {physical_gpu} is absent from nvidia-smi inventory.")
            item = by_index[physical_gpu]
            if int(item["utilization_pct"]) != 0 or int(item["memory_used_mib"]) != 0:
                raise RuntimeError(
                    f"Physical GPU {physical_gpu} is not idle: util={item['utilization_pct']}%, "
                    f"memory={item['memory_used_mib']} MiB."
                )
            current.append(item)
        accepted = current
        print(f"idle-gate sample {sample_index + 1}/{samples}: {json.dumps(current, sort_keys=True)}", flush=True)
        if sample_index + 1 < samples:
            time.sleep(1.0)
    return accepted


def _tensor_sha256(tensor) -> str:
    import torch

    value = tensor.detach().contiguous().to(device="cpu")
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(metadata + b"\0" + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_sample_sha256(tensor, maximum_axis_values: int = 128) -> str:
    import torch

    value = tensor.detach()
    for axis, axis_size in enumerate(value.shape):
        if axis_size <= maximum_axis_values:
            continue
        indexes = torch.linspace(0, axis_size - 1, maximum_axis_values, dtype=torch.float64, device=value.device)
        value = value.index_select(axis, indexes.round().to(dtype=torch.long))
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    sampled = value.contiguous().to(device="cpu")
    return hashlib.sha256(metadata + b"\0" + sampled.view(torch.uint8).numpy().tobytes()).hexdigest()


def _hessian_sha256(tensor) -> tuple[str, str]:
    # Hash common Llama projection Hessians in full.  The 8192x8192 down-proj
    # Hessian is 256 MiB, so retain a deterministic bounded diagnostic sample.
    if tensor.numel() <= 4096 * 4096:
        return _tensor_sha256(tensor), "full"
    return _tensor_sample_sha256(tensor), "sampled"


def _gptq_module_name(task) -> str:
    """Return the layer-qualified name retained by the NamedModule wrapper."""

    named_module = getattr(task, "_named_module", None)
    return str(getattr(named_module, "full_name", task.name))


def _build_calibration(frame, tokenizer, rows: int, seq_len: int, geometry: str) -> list[dict[str, Any]]:
    import torch

    calibration = []
    minimum = max(16, seq_len // 4)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if geometry == "fixed" and not isinstance(pad_token_id, int):
        raise ValueError("Fixed calibration geometry requires an integer tokenizer pad_token_id or eos_token_id.")
    padding_side = getattr(tokenizer, "padding_side", "right")
    if padding_side not in ("left", "right"):
        raise ValueError(f"Unsupported tokenizer padding_side: {padding_side!r}")

    for row_index, text in enumerate(frame["text"].iloc[:rows].tolist()):
        if geometry == "natural":
            tokenized = tokenizer(str(text), add_special_tokens=True, truncation=False)
            input_ids = list(tokenized["input_ids"])
            calibration.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                }
            )
            continue

        target = seq_len
        if geometry == "variable":
            target = minimum + ((row_index * 53) % (seq_len - minimum + 1))
        tokenized = tokenizer(str(text), add_special_tokens=True, truncation=True, max_length=target)
        natural_input_ids = list(tokenized["input_ids"])
        input_ids = natural_input_ids
        attention_mask = [1] * len(input_ids)
        if geometry == "fixed" and len(input_ids) < target:
            # Keep physical tensor geometry fixed while preserving the original
            # valid-token distribution used by length-aware Hessian buckets.
            padding = [pad_token_id] * (target - len(input_ids))
            padding_mask = [0] * len(padding)
            if padding_side == "left":
                input_ids = padding + input_ids
                attention_mask = padding_mask + attention_mask
            else:
                input_ids = input_ids + padding
                attention_mask = attention_mask + padding_mask
        elif geometry == "variable" and len(input_ids) < target:
            # Give variable geometry deterministic target lengths without
            # introducing masked padding into its forward path.
            content = input_ids[1:] or input_ids
            while len(input_ids) < target:
                input_ids.extend(content[: target - len(input_ids)])
            attention_mask = [1] * len(input_ids)
        input_ids = input_ids[:target]
        attention_mask = attention_mask[:target]
        calibration.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )
    return calibration


def compare_payloads(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    tensor_fields = (
        "hessian_sha256",
        "hessian_inverse_sha256",
        "weight_sha256",
        "scale_sha256",
        "zero_sha256",
        "g_idx_sha256",
        "code_sha256",
    )
    reference_modules = reference.get("modules", {})
    candidate_modules = candidate.get("modules", {})
    shared = sorted(reference_modules.keys() & candidate_modules.keys())
    mismatch_counts = {field: 0 for field in tensor_fields}
    changed = []
    for name in shared:
        mismatches = []
        for field in tensor_fields:
            if reference_modules[name].get(field) != candidate_modules[name].get(field):
                mismatch_counts[field] += 1
                mismatches.append(field)
        if mismatches:
            changed.append({"module": name, "tensor_mismatches": mismatches})
    exact = (
        not changed
        and reference_modules.keys() == candidate_modules.keys()
        and reference.get("calibration_sha256") == candidate.get("calibration_sha256")
    )
    return {
        "exact": exact,
        "reference_module_count": len(reference_modules),
        "candidate_module_count": len(candidate_modules),
        "shared_module_count": len(shared),
        "missing_from_candidate": sorted(reference_modules.keys() - candidate_modules.keys()),
        "missing_from_reference": sorted(candidate_modules.keys() - reference_modules.keys()),
        "tensor_mismatch_counts": mismatch_counts,
        "changed_module_count": len(changed),
        "changed_modules": changed,
    }


def main() -> None:
    args = _parse_args()
    inventory = _idle_gate(args.physical_gpus)

    import pandas as pd
    import torch
    from transformers import AutoTokenizer

    import gptqmodel
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import FORMAT, METHOD, ScaleSearchConfig
    from gptqmodel.quantization.config import (
        HessianConfig,
        LengthAwareConfig,
        LengthAwareMode,
        QuantizeConfig,
        VramStrategy,
    )
    from gptqmodel.quantization.diagnostics import (
        analyze_output_error,
        analyze_reconstruction_error,
        sample_reconstructed_quant_codes,
    )
    from gptqmodel.quantization.gptq import GPTQ

    if REPO_ROOT != Path(gptqmodel.__file__).resolve().parents[1]:
        raise RuntimeError(f"Imported gptqmodel from unexpected path: {gptqmodel.__file__}")
    if getattr(sys, "_is_gil_enabled", lambda: True)():
        raise RuntimeError("Run with Python 3.14 free-threading enabled (PYTHON_GIL=0).")
    if torch.cuda.device_count() != len(args.physical_gpus):
        raise RuntimeError(
            f"Expected {len(args.physical_gpus)} visible CUDA devices, found {torch.cuda.device_count()}."
        )

    torch.manual_seed(898)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=False)
    frame = pd.read_parquet(args.calibration)
    calibration = _build_calibration(frame, tokenizer, args.rows, args.seq_len, args.geometry)
    calibration_digest = hashlib.sha256()
    for item in calibration:
        calibration_digest.update(item["input_ids"].numpy().tobytes())
        calibration_digest.update(item["attention_mask"].numpy().tobytes())
    calibration_sha256 = calibration_digest.hexdigest()

    records: dict[str, dict[str, Any]] = {}
    original_quantize = GPTQ.quantize
    original_hessian_inverse = GPTQ.hessian_inverse

    def capture_hessian_inverse(self, hessian, *inverse_args, **inverse_kwargs):
        hessian_digest = hessian_kind = None
        if hessian is not None:
            hessian_digest, hessian_kind = _hessian_sha256(hessian)
        result = original_hessian_inverse(self, hessian, *inverse_args, **inverse_kwargs)
        inverse = result[0] if isinstance(result, tuple) else result
        inverse_digest, inverse_kind = _hessian_sha256(inverse)
        records.setdefault(_gptq_module_name(self), {}).update(
            {
                "hessian_sha256": hessian_digest,
                "hessian_hash_kind": hessian_kind,
                "hessian_inverse_sha256": inverse_digest,
                "hessian_inverse_hash_kind": inverse_kind,
            }
        )
        return result

    def capture_quantize(self, *quantize_args, **quantize_kwargs):
        result = original_quantize(self, *quantize_args, **quantize_kwargs)
        wq, scales, zeros, g_idx = result[:4]
        dense_weight = self.module.weight.data
        module_name = _gptq_module_name(self)
        seed = int(hashlib.sha256(module_name.encode()).hexdigest()[:16], 16)
        held_out = torch.randn(
            128,
            int(dense_weight.shape[1]),
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        held_out[0].mul_(8.0)
        codes = sample_reconstructed_quant_codes(
            wq,
            scales,
            zeros,
            g_idx,
            bits=int(getattr(self.qcfg, "runtime_bits", self.qcfg.bits)),
        )
        records.setdefault(module_name, {}).update(
            {
                "weight_sha256": _tensor_sha256(wq),
                "scale_sha256": _tensor_sha256(scales),
                "zero_sha256": _tensor_sha256(zeros),
                "g_idx_sha256": _tensor_sha256(g_idx),
                "code_sha256": _tensor_sha256(codes["codes"]) if codes is not None else None,
                "reconstruction": analyze_reconstruction_error(dense_weight, wq),
                "output_error": analyze_output_error(held_out, dense_weight, wq, bias=getattr(self.module, "bias", None)),
                "samples": int(result[7]),
                "loss": float(result[5]),
            }
        )
        return result

    GPTQ.hessian_inverse = capture_hessian_inverse
    GPTQ.quantize = capture_quantize
    config = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=4,
        group_size=args.group_size,
        sym=True,
        desc_act=False,
        act_group_aware=True,
        scale_search=ScaleSearchConfig.ACTIVATION,
        auto_forward_data_parallel=args.auto_forward_data_parallel,
        dense_vram_strategy=VramStrategy.EXCLUSIVE,
        moe_vram_strategy=VramStrategy.EXCLUSIVE,
        offload_to_disk=False,
        quantization_diagnostics="auto",
        hessian=HessianConfig(
            length_aware=LengthAwareConfig(
                mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
                target_bucket_count=6,
                bucket_weight_exponent=0.2,
            )
        ),
    )
    model = GPTQModel.load(
        str(args.model),
        quantize_config=config,
        trust_remote_code=False,
        dtype="auto",
        device_map="auto",
    )
    total_layers = int(model.model.config.num_hidden_layers)
    layer_scope = None if args.layers >= total_layers else slice(0, args.layers)
    started = time.perf_counter()
    model.quantize(
        calibration,
        calibration_sort=None,
        batch_size=args.batch_size,
        backend="auto",
        tokenizer=tokenizer,
        calibration_data_min_length=1,
        layer_scope=layer_scope,
    )
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - started

    resolved_configuration = config.to_dict()
    length_aware = resolved_configuration.get("meta", {}).get("hessian", {}).get("length_aware")
    if not isinstance(length_aware, dict):
        raise RuntimeError("Length-aware Hessian validation failed: serialized configuration is missing.")
    boundaries = length_aware.get("bucket_boundaries")
    scales = length_aware.get("bucket_scales")
    weights = length_aware.get("bucket_weights")
    if (
        length_aware.get("mode") != LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT.value
        or length_aware.get("target_bucket_count") != 6
        or not isinstance(boundaries, list)
        or len(boundaries) != 7
        or not isinstance(scales, list)
        or len(scales) != 6
        or not isinstance(weights, list)
        or len(weights) != 6
    ):
        raise RuntimeError(f"Length-aware Hessian did not resolve the requested 6/6 buckets: {length_aware!r}")

    calibration_lengths = [int(item["attention_mask"].sum().item()) for item in calibration]
    runtime_boundaries = [float("inf") if boundary is None else float(boundary) for boundary in boundaries]
    bucket_counts = [0] * 6
    for length in calibration_lengths:
        bucket_counts[bisect.bisect_right(runtime_boundaries, length) - 1] += 1
    if any(count == 0 for count in bucket_counts):
        raise RuntimeError(f"Length-aware Hessian resolved an empty runtime bucket: counts={bucket_counts}")
    print(
        "length-aware validation: "
        f"resolved=6/6 counts={bucket_counts} boundaries={boundaries} weights={weights}",
        flush=True,
    )
    if args.save_model is not None:
        args.save_model.mkdir(parents=True, exist_ok=True)
        model.save(str(args.save_model))

    payload = {
        "schema": "gptqmodel.llama-calibration-batch-accuracy.v1",
        "batch_size": args.batch_size,
        "rows": args.rows,
        # Natural geometry preserves each complete tokenizer output, so no
        # artificial sequence-length cap applies to this run.
        "seq_len": None if args.geometry == "natural" else args.seq_len,
        "geometry": args.geometry,
        "layers": min(args.layers, total_layers),
        "wall_s": wall_s,
        "calibration_sha256": calibration_sha256,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gil_enabled": sys._is_gil_enabled(),
            "physical_gpus": inventory,
            "visible_devices": [
                {
                    "ordinal": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "uuid": torch.cuda.get_device_properties(index).uuid,
                }
                for index in range(torch.cuda.device_count())
            ],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "configuration": resolved_configuration,
        "length_aware_validation": {
            "requested_bucket_count": 6,
            "resolved_bucket_count": 6,
            "bucket_counts": bucket_counts,
            "bucket_boundaries": boundaries,
            "bucket_scales": scales,
            "bucket_weights": weights,
            "calibration_length_min": min(calibration_lengths),
            "calibration_length_max": max(calibration_lengths),
            "calibration_unique_lengths": len(set(calibration_lengths)),
            "calibration_valid_tokens": sum(calibration_lengths),
        },
        "modules": dict(sorted(records.items())),
    }
    if args.reference is not None:
        payload["comparison"] = compare_payloads(
            json.loads(args.reference.read_text(encoding="utf-8")),
            payload,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "batch_size": args.batch_size,
                "wall_s": wall_s,
                "modules": len(records),
                "comparison": payload.get("comparison"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
