#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare serial and two-GPU batch-1 DeepSeek V4 GPTQ artifacts.

This is an investigation harness, not a performance benchmark. It fingerprints
the tensors returned directly by GPTQ before packing and records the existing
dense-reference reconstruction/output metrics. Run serial and parallel modes
in fresh processes, then pass the serial JSON to ``--reference`` on the
parallel run to produce a module-by-module comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("/monster/data/model/DeepSeek-V4-Flash-0731-BF16-Defused")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CALIBRATION = REPO_ROOT / "dataset/calibration_mix_128k_deepseek_v4_flash_0731/calibration.parquet"


def tensor_sha256(tensor) -> str:
    """Hash tensor metadata and exact bytes without dtype conversion."""

    import torch

    value = tensor.detach().contiguous().to(device="cpu")
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(metadata + b"\0" + raw).hexdigest()


def tensor_sample_sha256(tensor, *, maximum_axis_values: int = 64) -> str:
    """Hash a deterministic bounded sample while retaining full tensor metadata."""

    import torch

    value = tensor.detach()
    for axis, axis_size in enumerate(value.shape):
        if axis_size <= maximum_axis_values:
            continue
        indexes = torch.linspace(
            0,
            axis_size - 1,
            maximum_axis_values,
            dtype=torch.float64,
            device=value.device,
        ).round().to(dtype=torch.long)
        value = value.index_select(axis, indexes)
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    sampled = value.contiguous().to(device="cpu")
    return hashlib.sha256(metadata + b"\0" + sampled.view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_artifacts(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare exact GPTQ artifacts and numerical dense-reference metrics."""

    reference_modules = reference.get("modules", {})
    candidate_modules = candidate.get("modules", {})
    shared_names = sorted(reference_modules.keys() & candidate_modules.keys())
    tensor_fields = (
        "hessian_sha256",
        "hessian_inverse_sha256",
        "weight_sha256",
        "scale_sha256",
        "zero_sha256",
        "g_idx_sha256",
        "code_sha256",
    )
    mismatch_counts = {field: 0 for field in tensor_fields}
    changed_modules: list[dict[str, Any]] = []
    maximum_metric_delta: dict[str, float] = {}

    for name in shared_names:
        reference_record = reference_modules[name]
        candidate_record = candidate_modules[name]
        mismatched = []
        for field in tensor_fields:
            if reference_record.get(field) != candidate_record.get(field):
                mismatch_counts[field] += 1
                mismatched.append(field)

        metric_deltas = {}
        reference_metrics = reference_record.get("output_error") or {}
        candidate_metrics = candidate_record.get("output_error") or {}
        for metric in (
            "mean_absolute_error",
            "rmse",
            "relative_l2_error",
            "softmax_kld_mean",
            "top1_agreement",
        ):
            if metric not in reference_metrics or metric not in candidate_metrics:
                continue
            delta = float(candidate_metrics[metric]) - float(reference_metrics[metric])
            metric_deltas[metric] = delta
            maximum_metric_delta[metric] = max(maximum_metric_delta.get(metric, 0.0), abs(delta))

        if mismatched or any(delta != 0.0 for delta in metric_deltas.values()):
            changed_modules.append(
                {
                    "module": name,
                    "tensor_mismatches": mismatched,
                    "metric_deltas": metric_deltas,
                }
            )

    return {
        "reference_mode": reference.get("mode"),
        "candidate_mode": candidate.get("mode"),
        "reference_module_count": len(reference_modules),
        "candidate_module_count": len(candidate_modules),
        "shared_module_count": len(shared_names),
        "missing_from_candidate": sorted(reference_modules.keys() - candidate_modules.keys()),
        "missing_from_reference": sorted(candidate_modules.keys() - reference_modules.keys()),
        "tensor_mismatch_counts": mismatch_counts,
        "maximum_absolute_metric_delta": maximum_metric_delta,
        "changed_module_count": len(changed_modules),
        "changed_modules": changed_modules,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serial", "parallel"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--diagnostic-input-rows", type=int, default=256)
    args = parser.parse_args()
    if args.layers <= 0:
        parser.error("--layers must be positive")
    return args


def _calibration_rows(frame) -> list[dict[str, str]]:
    rows = []
    for messages in frame["messages"]:
        rows.append({"text": "\n".join(message["content"] for message in list(messages))})
    return rows


def main() -> None:
    args = _parse_args()
    if getattr(sys, "_is_gil_enabled", lambda: True)():
        raise RuntimeError("Run with Python free-threading enabled (PYTHON_GIL=0 and -X gil=0).")

    import pandas as pd
    import torch
    from transformers import AutoTokenizer

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import FORMAT, METHOD, ScaleSearchConfig
    from gptqmodel.quantization.config import (
        ExpertsRoutingBypass,
        HessianConfig,
        LengthAwareConfig,
        LengthAwareMode,
        MoEConfig,
        MoEExecutionConfig,
        QuantizeConfig,
        VramStrategy,
    )
    from gptqmodel.quantization.diagnostics import (
        analyze_output_error,
        analyze_reconstruction_error,
        sample_reconstructed_quant_codes,
    )
    from gptqmodel.quantization.gptq import GPTQ

    # Fail closed if path precedence ever changes and selects an installed wheel
    # instead of the branch being investigated.
    import gptqmodel

    if REPO_ROOT != Path(gptqmodel.__file__).resolve().parents[1]:
        raise RuntimeError(f"Imported gptqmodel from unexpected path: {gptqmodel.__file__}")

    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected exactly two visible CUDA devices, found {torch.cuda.device_count()}.")

    records: dict[str, dict[str, Any]] = {}
    records_lock = threading.Lock()
    original_quantize = GPTQ.quantize
    original_hessian_inverse = GPTQ.hessian_inverse
    hessian_module_names = {
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.o_b_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.0.down_proj",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
    }

    def capture_hessian_inverse(self, hessian, *inverse_args, **inverse_kwargs):
        module_name = str(self.name)
        capture = module_name in hessian_module_names and hessian is not None
        hessian_sha256 = tensor_sample_sha256(hessian) if capture else None
        result = original_hessian_inverse(self, hessian, *inverse_args, **inverse_kwargs)
        if capture:
            inverse = result[0] if isinstance(result, tuple) else result
            with records_lock:
                records.setdefault(module_name, {}).update(
                    {
                        "hessian_sha256": hessian_sha256,
                        "hessian_inverse_sha256": tensor_sample_sha256(inverse),
                    }
                )
        return result

    def capture_quantize(self, *quantize_args, **quantize_kwargs):
        result = original_quantize(self, *quantize_args, **quantize_kwargs)
        wq, scales, zeros, g_idx = result[:4]
        dense_weight = self.module.weight.data
        reconstruction = analyze_reconstruction_error(dense_weight, wq)
        # Use an independent, name-seeded FP32 stream so serial and parallel
        # runs replay exactly the same held-out inputs without enabling channel
        # diagnostics (channel mode deliberately disables shared-Hessian reuse).
        module_name = str(self.name)
        seed = int(hashlib.sha256(module_name.encode()).hexdigest()[:16], 16)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        held_out = torch.randn(
            args.diagnostic_input_rows,
            int(dense_weight.shape[1]),
            dtype=torch.float32,
            generator=generator,
        )
        if held_out.numel():
            held_out[0].mul_(8.0)
        output_error = analyze_output_error(
            held_out,
            dense_weight,
            wq,
            bias=getattr(self.module, "bias", None),
        )
        code_sample = sample_reconstructed_quant_codes(
            wq,
            scales,
            zeros,
            g_idx,
            bits=int(getattr(self.qcfg, "runtime_bits", self.qcfg.bits)),
        )
        record = {
            "weight_sha256": tensor_sha256(wq),
            "scale_sha256": tensor_sha256(scales),
            "zero_sha256": tensor_sha256(zeros),
            "g_idx_sha256": tensor_sha256(g_idx),
            "code_sha256": tensor_sha256(code_sample["codes"]) if code_sample is not None else None,
            "reconstruction": reconstruction,
            "output_error": output_error,
            "samples": int(result[7]),
            "loss": float(result[5]),
        }
        with records_lock:
            records.setdefault(module_name, {}).update(record)
        del held_out
        return result

    GPTQ.quantize = capture_quantize
    GPTQ.hessian_inverse = capture_hessian_inverse
    frame = pd.read_parquet(args.calibration)
    calibration = _calibration_rows(frame)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    length_aware = LengthAwareConfig(
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        target_bucket_count=6,
        bucket_weight_exponent=0.2,
    )
    config = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=4,
        group_size=64,
        sym=True,
        desc_act=False,
        act_group_aware=True,
        scale_search=ScaleSearchConfig.ACTIVATION,
        moe=MoEConfig(
            routing=ExpertsRoutingBypass(),
            execution=MoEExecutionConfig(batch_size=None, parallel_input_capture=True),
        ),
        auto_forward_data_parallel=args.mode == "parallel",
        calibration_data_device="balanced",
        dense_vram_strategy=VramStrategy.EXCLUSIVE,
        moe_vram_strategy=VramStrategy.EXCLUSIVE,
        hessian=HessianConfig(length_aware=length_aware),
        quantization_diagnostics="auto",
    )
    model = GPTQModel.load(
        str(args.model),
        quantize_config=config,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    )
    started = time.perf_counter()
    model.quantize(
        calibration,
        batch_size=1,
        backend="auto",
        tokenizer=tokenizer,
        layer_scope=slice(0, args.layers),
    )
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - started
    payload = {
        "schema": "gptqmodel.parallel-forward-accuracy.v1",
        "mode": args.mode,
        "wall_s": wall_s,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gil_enabled": sys._is_gil_enabled(),
            "visible_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        },
        "configuration": config.to_dict(),
        "modules": dict(sorted(records.items())),
    }
    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        payload["comparison"] = compare_artifacts(reference, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mode": args.mode, "wall_s": wall_s, "modules": len(records)}))


if __name__ == "__main__":
    main()
