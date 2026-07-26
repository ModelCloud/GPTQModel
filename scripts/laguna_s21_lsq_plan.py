#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Generate an LSQ-style pre-quantization sensitivity report and planned
QuantizeConfig for Laguna-S-2.1.

Target: GPTQ 4bit / group_size 128 / desc_act=False / GAR (act_group_aware=True)
/ activation scale_search. Modules flagged by the weight-proxy analyzer as
needing a higher effective bit rate are promoted to 4bit / group_size 32.

The script loads the dense checkpoint, runs the GPT-QModel
QuantizationAnalyzer on the (already defused) module tree, writes the JSON
report and plan, and emits a ready-to-use QuantizeConfig with dynamic
overrides merged in. It does NOT run the actual GPTQ calibration/quantization.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


# Keep GPU allocator skill defaults early.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gptqmodel import GPTQModel
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.analysis import (
    QuantizationAnalyzer,
    _module_role,
    _weight_matrix,
    apply_analysis_plan,
    report_to_json,
)
from gptqmodel.quantization.config import AnalysisConfig, QuantizeConfig


def _install_local_compute_patch() -> None:
    """Patch QuantizationAnalyzer so each weight is analyzed on the device it
    already occupies, allowing an accelerate-dispatched model to use all GPUs.
    """

    _orig = QuantizationAnalyzer._analyze_weight

    def _analyze_weight_local(self, weight, *, bit_width, group_size, sym):
        original_device = self.compute_device
        self.compute_device = weight.device
        try:
            return _orig(self, weight, bit_width=bit_width, group_size=group_size, sym=sym)
        finally:
            self.compute_device = original_device

    QuantizationAnalyzer._analyze_weight = _analyze_weight_local


class TelemetryAnalyzer(QuantizationAnalyzer):
    """QuantizationAnalyzer with per-step progress and device/time telemetry."""

    def __init__(self, *args, log_interval: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_interval = log_interval
        self._telemetry = {"n": 0, "t0": time.perf_counter(), "total": 0}

    def analyze_model(self, model, **kwargs):
        orig_named = model.named_modules
        total = sum(1 for _ in orig_named())
        self._telemetry["total"] = total

        def _named_modules(*args, **kw):
            yield from tqdm(
                orig_named(*args, **kw),
                total=total,
                desc="[lsq] scanning modules",
                unit="mod",
                mininterval=2.0,
            )

        model.named_modules = _named_modules
        try:
            return super().analyze_model(model, **kwargs)
        finally:
            model.named_modules = orig_named

    def analyze_module(self, module, *, module_name, **kwargs):
        self._telemetry["n"] += 1
        n = self._telemetry["n"]
        weight = _weight_matrix(module, kwargs.get("weight"))
        if n % self.log_interval == 0 or n == 1:
            elapsed = time.perf_counter() - self._telemetry["t0"]
            total = self._telemetry["total"]
            print(
                f"[lsq telemetry] {n}/{total} {module_name} "
                f"device={weight.device} shape={tuple(weight.shape)} "
                f"elapsed={elapsed:.2f}s avg={elapsed/n:.4f}s",
                flush=True,
            )
        return super().analyze_module(module, module_name=module_name, **kwargs)


def _dispatch_model_to_gpus(model: torch.nn.Module, gpu_count: int) -> torch.nn.Module:
    """Spread a CPU-loaded dense model across all visible GPUs for the scan.

    accelerate's infer_auto_device_map is too slow for MoE checkpoints with
    tens of thousands of expert modules, so we do a simple per-layer
    round-robin move (≈5 GB of BF16 parameters per Laguna-S-2.1 layer).
    """

    from gptqmodel.utils.model import get_module

    layers = get_module(model, "model.layers")
    if layers is None or not isinstance(layers, (list, torch.nn.ModuleList)):
        raise RuntimeError("Could not locate model.layers for per-layer GPU dispatch.")

    for i, layer in enumerate(layers):
        device = f"cuda:{i % gpu_count}"
        layer.to(device)

    # Keep the embedding and LM-head on cuda:0 for the endpoint scan.
    for endpoint in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(model, endpoint, None)
        if callable(getter):
            try:
                getter().to("cuda:0")
            except Exception:
                pass

    print(f"[dispatch] Moved {len(layers)} decoder layers round-robin across {gpu_count} GPU(s)")
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/monster/data/model/Laguna-S-2.1")
    parser.add_argument("--output", default="/root/GPT-QModel-Ultra-2/laguna_s21_lsq_plan")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--promotion-group-size", type=int, default=32)
    parser.add_argument("--recommendation-percentile", type=float, default=95.0)
    parser.add_argument("--min-recommendation-risk", type=float, default=20.0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--compute-device", default="local", help="'local' means compute on each weight's current device (needed for multi-GPU dispatch). Use 'cuda' or 'cpu' to force one device.")
    parser.add_argument("--dispatch-gpus", type=int, default=-1, help="Number of GPUs to spread the dense model onto before the scan; -1 uses all visible CUDA devices, 0 keeps the model on CPU.")
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100, help="Telemetry log interval in analyzed modules.")
    return parser.parse_args()


def _analyze_model_parallel(
    gptq_model: GPTQModel,
    model: torch.nn.Module,
    base_analyzer: TelemetryAnalyzer,
    module_groups: list[list[str]],
    group_source: str,
    num_gpus: int,
) -> dict[str, Any]:
    """Run the LSQ weight scan across decoder layers in parallel, one GPU per
    layer, to keep all GPUs busy and avoid the single-device CPU/GPU sync
    bottleneck seen in the sequential analyzer path.
    """

    from concurrent.futures import ThreadPoolExecutor

    from gptqmodel.utils.model import get_layers_with_prefixes

    qcfg = base_analyzer.qcfg
    analysis_cfg = base_analyzer.config
    log_interval = base_analyzer.log_interval

    analyzers: dict[int, TelemetryAnalyzer] = {}
    for i in range(num_gpus):
        a = TelemetryAnalyzer(
            qcfg,
            analysis_cfg,
            compute_device=f"cuda:{i}",
            log_interval=log_interval,
        )
        a.set_module_groups(module_groups, source=group_source)
        analyzers[i] = a

    layers, layer_names = get_layers_with_prefixes(model, gptq_model.extract_layers_node())
    if not layers:
        raise RuntimeError("No decoder layers found for parallel LSQ scan.")

    # Ensure every layer is resident on one of the GPUs.
    for i, layer in enumerate(layers):
        layer.to(f"cuda:{i % num_gpus}")

    # Keep endpoints together on cuda:0.
    for getter_name in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(model, getter_name, None)
        if callable(getter):
            try:
                getter().to("cuda:0")
            except Exception:
                pass

    input_ids, output_ids = QuantizationAnalyzer._endpoint_ids(model)

    # Rough module count for telemetry.
    total = sum(1 for _ in model.named_modules())
    for a in analyzers.values():
        a._telemetry["total"] = total

    def _process_layer(layer_index: int) -> list[dict[str, Any]]:
        gpu = layer_index % num_gpus
        analyzer = analyzers[gpu]
        layer = layers[layer_index]
        layer_prefix = layer_names[layer_index]
        records: list[dict[str, Any]] = []

        for name, mod in layer.named_modules():
            if not name:
                continue
            if not analyzer._is_analyzable(mod):
                continue
            full_name = f"{layer_prefix}.{name}"
            if analyzer.qcfg.dynamic_get(layer_name=full_name) is False:
                continue
            role = _module_role(full_name, mod, input_ids, output_ids)
            if not analyzer.config.include_endpoints and role in {"input_embedding", "lm_head", "embedding"}:
                continue
            records.append(analyzer.analyze_module(mod, module_name=full_name, role=role))
        return records

    all_records: list[dict[str, Any]] = []
    print(f"[lsq] Parallel scan of {len(layers)} layer(s) on {num_gpus} GPU(s) ...")
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        for result in executor.map(_process_layer, range(len(layers))):
            all_records.extend(result)

    # Analyze endpoints with the cuda:0 analyzer.
    endpoint_analyzer = analyzers[0]
    for full_name, mod, role in (
        (gptq_model.get_input_embeddings_name(), model.get_input_embeddings(), "input_embedding"),
        (gptq_model.get_output_embeddings_name(), model.get_output_embeddings(), "lm_head"),
    ):
        if mod is None or full_name is None:
            continue
        if endpoint_analyzer.qcfg.dynamic_get(layer_name=full_name) is not False:
            all_records.append(endpoint_analyzer.analyze_module(mod, module_name=full_name, role=role))

    return endpoint_analyzer.build_report(all_records, skipped=[])


def _reorganize_expert_module_groups(simple_groups: list[list[str]]) -> list[list[str]]:
    """Convert the flat expert alias groups from simple_layer_modules into
    per-expert companion groups so gate/up/down overrides stay linked by
    expert index.
    """

    expert_re = re.compile(r"mlp\.experts\.(\d+)\.")
    new_groups: list[list[str]] = []

    for group in simple_groups:
        if not any(expert_re.search(name) for name in group):
            new_groups.append(group)
            continue

        by_index: dict[int, list[str]] = {}
        for name in group:
            m = expert_re.search(name)
            if not m:
                continue
            idx = int(m.group(1))
            by_index.setdefault(idx, []).append(name)

        for idx in sorted(by_index):
            new_groups.append(by_index[idx])

    return new_groups


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    plan = report.get("plan", {})
    dynamic = plan.get("dynamic", {})
    recommendations = plan.get("recommendations", [])
    records = report.get("records", [])

    by_role: dict[str, int] = {}
    flagged_by_role: dict[str, int] = {}
    for rec in records:
        by_role[rec.get("role", "linear")] = by_role.get(rec.get("role", "linear"), 0) + 1

    for rec in recommendations:
        role = rec.get("role", "linear")
        if rec.get("action") == "promote_fusion_companion":
            continue
        flagged_by_role[role] = flagged_by_role.get(role, 0) + 1

    return {
        "analyzed_modules": report.get("summary", {}).get("analyzed_modules", len(records)),
        "flagged_modules": len([r for r in recommendations if r.get("action") != "promote_fusion_companion"]),
        "fusion_companions": len([r for r in recommendations if r.get("action") == "promote_fusion_companion"]),
        "dynamic_rules": len(dynamic),
        "roles": by_role,
        "flagged_by_role": flagged_by_role,
    }


def main() -> None:
    args = _parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    qcfg = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=False,
        act_group_aware=True,
        scale_search="activation",
        sym=True,
        preprocessors=[
            AnalysisConfig(
                promotion_bits=args.bits,
                promotion_group_size=args.promotion_group_size,
                recommendation_percentile=args.recommendation_percentile,
                min_recommendation_risk=args.min_recommendation_risk,
                emit_markdown=True,
                emit_json=True,
                include_endpoints=True,
                fusion_profile="model_definition",
            )
        ],
        offload_to_disk=False,
    )

    print(f"[load] Loading dense Laguna-S-2.1 from {args.model_path} ...")
    print(f"[cfg]  {qcfg}")

    model = GPTQModel.load(
        args.model_path,
        quantize_config=qcfg,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        dtype=args.dtype,
    )

    print(f"[model] {type(model).__name__}, device_map={args.device_map}")

    if args.dispatch_gpus != 0 and torch.cuda.is_available():
        gpu_count = args.dispatch_gpus if args.dispatch_gpus > 0 else torch.cuda.device_count()
        print(f"[dispatch] Spreading dense model across {gpu_count} GPU(s) ...")
        model.model = _dispatch_model_to_gpus(model.model, gpu_count)

    if args.compute_device == "local":
        _install_local_compute_patch()
        compute_device = "cpu"
    else:
        compute_device = args.compute_device

    # Build module groups from the model definition and reorganize the MoE
    # expert groups so each expert's gate/up/down companions are kept together.
    simple_groups = model.simple_layer_modules(
        model_config=model.model.config,
        quantize_config=model.quantize_config,
        is_awq_quantize=False,
        include_capture_only=False,
    )
    module_groups = _reorganize_expert_module_groups(simple_groups)
    group_source = f"{type(model).__module__}.{type(model).__name__}.simple_layer_modules"

    analysis_cfg = model.quantize_config.preprocessors[0]
    base_analyzer = TelemetryAnalyzer(
        model.quantize_config,
        analysis_cfg,
        compute_device=compute_device,
        log_interval=args.log_interval,
    )
    base_analyzer.set_module_groups(module_groups, source=group_source)

    selection = None
    if args.max_modules is not None:
        from gptqmodel.quantization.analysis import AnalysisSelection
        selection = AnalysisSelection(max_modules=args.max_modules)

    print("[lsq] Running weight-only sensitivity scan ...")
    t0 = time.perf_counter()
    if args.dispatch_gpus != 0 and torch.cuda.is_available():
        gpu_count = args.dispatch_gpus if args.dispatch_gpus > 0 else torch.cuda.device_count()
        report = _analyze_model_parallel(
            gptq_model=model,
            model=model.model,
            base_analyzer=base_analyzer,
            module_groups=module_groups,
            group_source=group_source,
            num_gpus=gpu_count,
        )
    else:
        report = base_analyzer.analyze_model(model.model, selection=selection)
    scan_seconds = time.perf_counter() - t0
    print(f"[lsq] Scan complete in {scan_seconds:.1f}s")

    plan = report["plan"]

    # Merge the plan into a clean QuantizeConfig for the real quantization run.
    final_qcfg, apply_info = apply_analysis_plan(model.quantize_config, plan, inplace=False)
    # Remove the AnalysisConfig preprocessor from the final config; the
    # dynamic overrides are the lasting artifact of the LSQ planning step.
    final_qcfg.preprocessors = []

    # Persist artifacts.
    with open(output / "lsq_report.json", "w") as f:
        f.write(report_to_json(report))

    with open(output / "lsq_plan.json", "w") as f:
        json.dump(plan, f, indent=2, sort_keys=True, default=str)

    with open(output / "lsq_markdown.md", "w") as f:
        f.write(report.get("markdown", ""))

    with open(output / "quant_config_base.json", "w") as f:
        json.dump(model.quantize_config.to_dict(), f, indent=2, sort_keys=True, default=str)

    with open(output / "quant_config_planned.json", "w") as f:
        json.dump(final_qcfg.to_dict(), f, indent=2, sort_keys=True, default=str)

    summary = {
        "model_path": str(args.model_path),
        "base_config": plan.get("base_config"),
        "scan_seconds": scan_seconds,
        "summary": _summarize_report(report),
        "dynamic_rules_preview": dict(list(plan.get("dynamic", {}).items())[:20]),
        "apply_info": apply_info,
        "post_quant_notes": {
            "embed_lm_head_target": {
                "bits": args.bits,
                "group_size": args.group_size,
                "desc_act": False,
                "act_group_aware": True,
                "scale_search": "activation",
                "sym": True,
                "method": "post_quantize_embeddings / requantize with embed_only",
            },
            "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        },
    }

    with open(output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)

    print("\n[summary]")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] Artifacts written to {output}")


if __name__ == "__main__":
    main()
