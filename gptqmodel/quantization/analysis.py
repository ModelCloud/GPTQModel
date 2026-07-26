# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import heapq
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

import torch
import transformers

from .config import (
    AnalysisConfig,
    BaseQuantizeConfig,
    quant_bits_width,
    serialize_quant_bits,
)


ANALYSIS_SCHEMA_VERSION = "1.0"
PLAN_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class AnalysisSelection:
    """Optional limits for a standalone model scan."""

    module_pattern: Optional[str] = None
    max_modules: Optional[int] = None

    def compile_pattern(self) -> Optional[Pattern[str]]:
        if not self.module_pattern:
            return None
        return re.compile(self.module_pattern)


def _finite_number(value: float) -> Optional[float]:
    """Return a JSON-safe finite float."""

    value = float(value)
    return value if math.isfinite(value) else None


def _module_role(name: str, module: torch.nn.Module, input_ids: set[int], output_ids: set[int]) -> str:
    """Classify common transformer module roles without requiring a model adapter."""

    module_id = id(module)
    lowered = name.lower()
    if module_id in input_ids:
        return "input_embedding"
    if module_id in output_ids or lowered.endswith(("lm_head", "output_layer", "embed_out")):
        return "lm_head"
    if isinstance(module, torch.nn.Embedding):
        return "embedding"
    if "shared_expert" in lowered:
        return "shared_expert"
    if "expert" in lowered:
        return "expert"
    if any(token in lowered for token in ("router", "route_proj", "switch")):
        return "router"
    for suffix, role in (
        ("q_proj", "attention_q"),
        ("k_proj", "attention_k"),
        ("v_proj", "attention_v"),
        ("o_proj", "attention_o"),
        ("out_proj", "attention_o"),
        ("gate_proj", "mlp_gate"),
        ("up_proj", "mlp_up"),
        ("down_proj", "mlp_down"),
    ):
        if lowered.endswith(suffix):
            return role
    return "linear"


def _layer_index(name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


WeightResolver = Callable[[str, torch.nn.Module], Optional[torch.Tensor]]


def _weight_matrix(module: torch.nn.Module, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return a 2-D view matching GPTQ's output-row/input-feature convention."""

    weight = module.weight.detach() if weight is None else weight.detach()
    if isinstance(module, transformers.Conv1D):
        weight = weight.T
    elif weight.dim() > 2:
        weight = weight.reshape(weight.shape[0], -1)
    elif weight.dim() == 1:
        weight = weight.reshape(1, -1)
    return weight


def _sample_abs(weight: torch.Tensor, max_values: int) -> torch.Tensor:
    """Take a deterministic bounded absolute-value sample on CPU."""

    flat = weight.reshape(-1)
    if not flat.numel():
        return torch.empty(0, dtype=torch.float32)
    stride = max(1, math.ceil(flat.numel() / max(max_values, 1)))
    sample = flat[::stride][:max_values].to(device="cpu", dtype=torch.float32)
    sample = torch.abs(sample[torch.isfinite(sample)])
    return sample


def score_quantizability(stats: Dict[str, Any]) -> float:
    """Convert measured risks into a 0-100 score where 100 is easiest."""

    if stats.get("nonfinite_count", 0):
        return 0.0
    rel_rmse_penalty = min(55.0, float(stats["rel_rmse"]) * 300.0)
    small_value_penalty = min(25.0, float(stats["small_value_fraction"]) * 30.0)
    bad_block_penalty = min(30.0, float(stats["bad_block_fraction"]) * 45.0)
    outlier_log = math.log10(max(float(stats["max_to_median_abs"]), 1.0))
    outlier_penalty = min(35.0, max(0.0, outlier_log - 1.0) * 10.0)
    score = 100.0 - rel_rmse_penalty - small_value_penalty - bad_block_penalty - outlier_penalty
    return round(max(0.0, min(100.0, score)), 2)


class QuantizationAnalyzer:
    """Chunked, weight-only quantization-risk analyzer."""

    def __init__(
        self,
        quantize_config: BaseQuantizeConfig,
        analysis_config: Optional[AnalysisConfig] = None,
        *,
        compute_device: str | torch.device = "cpu",
        module_groups: Optional[Sequence[Sequence[str]]] = None,
        module_group_source: Optional[str] = None,
    ):
        self.qcfg = quantize_config
        self.config = analysis_config or AnalysisConfig()
        self.compute_device = torch.device(compute_device)
        self.set_module_groups(module_groups, source=module_group_source)

    def set_module_groups(
        self,
        module_groups: Optional[Sequence[Sequence[str]]],
        *,
        source: Optional[str] = None,
    ) -> None:
        """Set the model-definition groups that govern compatibility closure."""

        normalized = []
        for group in module_groups or []:
            members = []
            for module_name in group:
                name = str(module_name).split(":", 1)[0]
                if name and name not in members:
                    members.append(name)
            if members:
                normalized.append(members)
        self.module_groups = normalized
        self.module_group_source = source

    def analyze_model(
        self,
        model: torch.nn.Module,
        *,
        selection: Optional[AnalysisSelection] = None,
        weight_resolver: Optional[WeightResolver] = None,
    ) -> Dict[str, Any]:
        """Analyze all materialized quantizable weights in a model."""

        selection = selection or AnalysisSelection()
        pattern = selection.compile_pattern()
        input_ids, output_ids = self._endpoint_ids(model)
        records: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []
        aliases: Dict[Tuple[int, str, int, bool], str] = {}

        for name, module in model.named_modules():
            if not name or pattern is not None and pattern.search(name) is None:
                continue
            if not self._is_analyzable(module):
                continue
            if selection.max_modules is not None and len(records) >= selection.max_modules:
                break
            if self.qcfg.dynamic_get(layer_name=name) is False:
                skipped.append({"module": name, "reason": "disabled by quantize_config.dynamic"})
                continue

            role = _module_role(name, module, input_ids, output_ids)
            if not self.config.include_endpoints and role in {"input_embedding", "lm_head", "embedding"}:
                skipped.append({"module": name, "reason": "endpoint analysis disabled"})
                continue
            try:
                resolved_weight = weight_resolver(name, module) if weight_resolver is not None else None
                weight = _weight_matrix(module, resolved_weight)
                if weight.device.type == "meta":
                    raise ValueError("weight is on the meta device; run analysis while the module is materialized")
                bits = self.qcfg.dynamic_get(name, "bits", self.qcfg.bits)
                group_size = int(self.qcfg.dynamic_get(name, "group_size", self.qcfg.group_size))
                sym = bool(self.qcfg.dynamic_get(name, "sym", self.qcfg.sym))
                cache_key = (id(module.weight), str(serialize_quant_bits(bits)), group_size, sym)
                tied_to = aliases.get(cache_key)
                record = self.analyze_module(
                    module,
                    module_name=name,
                    role=role,
                    bits=bits,
                    group_size=group_size,
                    sym=sym,
                    weight=resolved_weight,
                )
                if tied_to is None:
                    aliases[cache_key] = name
                else:
                    record["tied_to"] = tied_to
                records.append(record)
            except Exception as exc:
                skipped.append({"module": name, "reason": str(exc)})

        return self.build_report(records, skipped=skipped)

    def analyze_module(
        self,
        module: torch.nn.Module,
        *,
        module_name: str,
        role: Optional[str] = None,
        bits: Any = None,
        group_size: Optional[int] = None,
        sym: Optional[bool] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Analyze one module and retain its worst rows, features, and groups."""

        bits = self.qcfg.bits if bits is None else bits
        group_size = self.qcfg.group_size if group_size is None else int(group_size)
        sym = self.qcfg.sym if sym is None else bool(sym)
        weight = _weight_matrix(module, weight)
        if weight.device.type == "meta":
            raise ValueError("weight is on the meta device")
        if weight.dim() != 2:
            raise ValueError(f"expected a 2-D weight matrix, got shape {tuple(weight.shape)}")

        stats, regions = self._analyze_weight(
            weight,
            bit_width=quant_bits_width(bits),
            group_size=group_size,
            sym=sym,
        )
        resolved_role = role or "linear"
        if resolved_role in {"input_embedding", "embedding"}:
            self._rename_row_regions(regions, "embedding_row")
        elif resolved_role == "lm_head":
            self._rename_row_regions(regions, "vocab_row")
        quant_score = score_quantizability(stats)
        return {
            "module": module_name,
            "layer": _layer_index(module_name),
            "role": resolved_role,
            "module_type": module.__class__.__name__,
            "shape": list(stats["shape"]),
            "numel": int(weight.numel()),
            "source_dtype": str(weight.dtype).removeprefix("torch."),
            "source_device": str(weight.device),
            "method": str(self.qcfg.method).split(".")[-1],
            "format": str(self.qcfg.format).split(".")[-1],
            "bits": serialize_quant_bits(bits),
            "group_size": group_size,
            "sym": sym,
            "quant_score": quant_score,
            "risk_score": round(100.0 - quant_score, 2),
            **stats,
            "regions": regions,
        }

    @staticmethod
    def _rename_row_regions(regions: List[Dict[str, Any]], row_kind: str) -> None:
        """Use endpoint-specific row labels while retaining group coordinates."""

        for region in regions:
            if region["kind"] == "output_channel":
                region["kind"] = row_kind
            elif region["kind"] == "group":
                region["row_kind"] = row_kind
                region["row_index"] = region["output_channel"]

    def _analyze_weight(
        self,
        weight: torch.Tensor,
        *,
        bit_width: int,
        group_size: int,
        sym: bool,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        rows, cols = (int(weight.shape[0]), int(weight.shape[1]))
        effective_group_size = cols if group_size <= 0 else min(group_size, cols)
        qmax = (2 ** (bit_width - 1) - 1) if sym else (2**bit_width - 1)
        qmin = -qmax if sym else 0
        eps = torch.finfo(torch.float32).eps
        max_chunk_values = self.config.max_chunk_values
        row_chunk = max(1, min(rows, max_chunk_values // max(cols, 1)))

        # Accumulate row and feature error/signal tensors on the compute
        # device and defer CPU transfers until the final top-k selection.
        row_sse = torch.zeros(rows, dtype=torch.float64, device=self.compute_device)
        row_signal = torch.zeros(rows, dtype=torch.float64, device=self.compute_device)
        feature_sse = torch.zeros(cols, dtype=torch.float64, device=self.compute_device)
        feature_signal = torch.zeros(cols, dtype=torch.float64, device=self.compute_device)
        group_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        heap_serial = 0

        # Keep running totals on the compute device and sync to Python scalars
        # only once per module. This avoids the host/device round-trips that
        # dominate the sequential analyzer path on MoE checkpoints.
        total_sse = torch.zeros((), dtype=torch.float64, device=self.compute_device)
        total_signal = torch.zeros((), dtype=torch.float64, device=self.compute_device)
        total_dot = torch.zeros((), dtype=torch.float64, device=self.compute_device)
        total_quant_signal = torch.zeros((), dtype=torch.float64, device=self.compute_device)
        total_small = torch.zeros((), dtype=torch.long, device=self.compute_device)
        total_bad_blocks = torch.zeros((), dtype=torch.long, device=self.compute_device)
        total_saturation = torch.zeros((), dtype=torch.long, device=self.compute_device)
        total_nonfinite = torch.zeros((), dtype=torch.long, device=self.compute_device)
        observed_max_abs = torch.zeros((), dtype=torch.float32, device=self.compute_device)

        total_blocks = 0
        scales: List[torch.Tensor] = []

        for row_start in range(0, rows, row_chunk):
            chunk = weight[row_start : row_start + row_chunk].to(
                device=self.compute_device,
                dtype=torch.float32,
            )
            finite = torch.isfinite(chunk)
            total_nonfinite += (~finite).sum()
            chunk = torch.where(finite, chunk, torch.zeros_like(chunk))
            observed_max_abs = torch.maximum(observed_max_abs, torch.abs(chunk).max())
            chunk_row_sse = torch.zeros(chunk.shape[0], device=self.compute_device)
            chunk_row_signal = torch.zeros(chunk.shape[0], device=self.compute_device)

            for col_start in range(0, cols, effective_group_size):
                col_end = min(col_start + effective_group_size, cols)
                block = chunk[:, col_start:col_end]
                if sym:
                    scale = torch.amax(torch.abs(block), dim=1, keepdim=True).clamp_min(eps) / max(qmax, 1)
                    zero = None
                    projected = block / scale
                    quantized = torch.round(projected).clamp(qmin, qmax)
                    dequantized = quantized * scale
                else:
                    block_min = torch.amin(block, dim=1, keepdim=True)
                    block_max = torch.amax(block, dim=1, keepdim=True)
                    scale = (block_max - block_min).clamp_min(eps) / max(qmax, 1)
                    zero = torch.round(-block_min / scale).clamp(qmin, qmax)
                    projected = block / scale + zero
                    quantized = torch.round(projected).clamp(qmin, qmax)
                    dequantized = (quantized - zero) * scale

                diff = block - dequantized
                squared_error = diff * diff
                squared_signal = block * block
                block_sse = torch.sum(squared_error, dim=1)
                block_signal = torch.sum(squared_signal, dim=1)
                block_rel = torch.sqrt(block_sse / block_signal.clamp_min(eps))

                total_sse += block_sse.sum().double()
                total_signal += block_signal.sum().double()
                total_dot += (block * dequantized).sum().double()
                total_quant_signal += (dequantized * dequantized).sum().double()
                total_small += (torch.abs(block) <= (0.5 * scale)).sum()
                total_bad_blocks += (block_rel > self.config.bad_block_rel_rmse_threshold).sum()
                total_blocks += block_rel.numel()
                total_saturation += ((projected < qmin) | (projected > qmax)).sum()

                chunk_row_sse += block_sse
                chunk_row_signal += block_signal
                feature_sse[col_start:col_end] += squared_error.sum(dim=0).double()
                feature_signal[col_start:col_end] += squared_signal.sum(dim=0).double()

                scales.append(scale)
                heap_serial = self._retain_group_regions(
                    group_heap,
                    block_rel,
                    scale,
                    zero,
                    row_start=row_start,
                    col_start=col_start,
                    col_end=col_end,
                    serial=heap_serial,
                )

            row_end = row_start + chunk.shape[0]
            row_sse[row_start:row_end] = chunk_row_sse.double()
            row_signal[row_start:row_end] = chunk_row_signal.double()

        if scales:
            all_scales = torch.cat([s.view(-1) for s in scales])
            scale_count = all_scales.numel()
            scale_min = float(all_scales.min().item())
            scale_max = float(all_scales.max().item())
            scale_mean = float(all_scales.sum().item()) / scale_count
        else:
            scale_count = 0
            scale_min = 0.0
            scale_max = 0.0
            scale_mean = 0.0

        total_sse_float = float(total_sse.item())
        total_signal_float = float(total_signal.item())
        total_dot_float = float(total_dot.item())
        total_quant_signal_float = float(total_quant_signal.item())
        total_small_int = int(total_small.item())
        total_bad_blocks_int = int(total_bad_blocks.item())
        total_saturation_int = int(total_saturation.item())
        total_nonfinite_int = int(total_nonfinite.item())
        observed_max_abs_float = float(observed_max_abs.item())

        sample = _sample_abs(weight, self.config.max_sample_values)
        nonzero_sample = sample[sample > 0]
        median_abs = float(torch.median(nonzero_sample).item()) if nonzero_sample.numel() else 0.0
        max_abs = observed_max_abs_float
        quantiles = self._quantiles(sample)
        rel_rmse = math.sqrt(total_sse_float / max(total_signal_float, eps))
        cosine = total_dot_float / math.sqrt(max(total_signal_float * total_quant_signal_float, eps))
        sqnr_db = 10.0 * math.log10(max(total_signal_float, eps) / max(total_sse_float, eps))
        max_to_median = max_abs / max(median_abs, eps)

        output_regions = self._axis_regions(
            row_sse,
            row_signal,
            kind="output_channel",
            limit=self.config.regions_per_module,
        )
        input_regions = self._axis_regions(
            feature_sse,
            feature_signal,
            kind="input_feature",
            limit=self.config.regions_per_module,
        )
        group_regions = [item[2] for item in sorted(group_heap, reverse=True)]
        regions = sorted(
            [*output_regions, *input_regions, *group_regions],
            key=lambda item: (-item["rel_rmse"], item["kind"], item.get("index", -1)),
        )

        total_values = rows * cols
        stats = {
            "shape": (rows, cols),
            "rel_rmse": float(rel_rmse),
            "sqnr_db": _finite_number(sqnr_db),
            "cosine_similarity": _finite_number(cosine),
            "small_value_fraction": total_small_int / max(total_values, 1),
            "bad_block_fraction": total_bad_blocks_int / max(total_blocks, 1),
            "saturation_fraction": total_saturation_int / max(total_values, 1),
            "nonfinite_count": total_nonfinite_int,
            "max_abs": max_abs,
            "median_abs": median_abs,
            "max_to_median_abs": float(max_to_median),
            "abs_quantiles": quantiles,
            "scale_min": 0.0 if scale_count == 0 else scale_min,
            "scale_mean": scale_mean,
            "scale_max": scale_max,
            "num_blocks": total_blocks,
            "effective_group_size": effective_group_size,
            "sampled_values": int(sample.numel()),
        }
        return stats, regions

    def _retain_group_regions(
        self,
        heap: List[Tuple[float, int, Dict[str, Any]]],
        rel_rmse: torch.Tensor,
        scale: torch.Tensor,
        zero: Optional[torch.Tensor],
        *,
        row_start: int,
        col_start: int,
        col_end: int,
        serial: int,
    ) -> int:
        limit = self.config.regions_per_module
        count = min(limit, int(rel_rmse.numel()))
        if count <= 0:
            return serial
        values, indices = torch.topk(rel_rmse, k=count)
        # Gather scales and zeros for the retained indices in one vectorized
        # operation instead of one `.item()` call per top-k region.
        scale_vals = scale[indices, 0]
        zero_vals = zero[indices, 0] if zero is not None else None
        values_list = values.tolist()
        indices_list = indices.tolist()
        scale_list = scale_vals.tolist()
        zero_list = zero_vals.tolist() if zero_vals is not None else None
        for i in range(count):
            record = {
                "kind": "group",
                "output_channel": row_start + int(indices_list[i]),
                "input_start": col_start,
                "input_end": col_end,
                "rel_rmse": float(values_list[i]),
                "scale": float(scale_list[i]),
            }
            if zero_list is not None:
                record["zero"] = float(zero_list[i])
            item = (float(values_list[i]), serial, record)
            serial += 1
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        return serial

    @staticmethod
    def _axis_regions(
        error: torch.Tensor,
        signal: torch.Tensor,
        *,
        kind: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        rel = torch.sqrt(error / signal.clamp_min(torch.finfo(torch.float64).eps))
        count = min(limit, int(rel.numel()))
        if count <= 0:
            return []
        values, indices = torch.topk(rel, k=count)
        # Gather the signal values for the selected indices in one shot to
        # avoid one `.item()`/index transfer per region.
        signal_top = signal[indices]
        values_list = values.tolist()
        indices_list = indices.tolist()
        signal_list = signal_top.tolist()
        return [
            {
                "kind": kind,
                "index": int(indices_list[i]),
                "rel_rmse": float(values_list[i]),
                "signal_l2": math.sqrt(max(float(signal_list[i]), 0.0)),
            }
            for i in range(count)
        ]

    @staticmethod
    def _quantiles(sample: torch.Tensor) -> Dict[str, float]:
        if not sample.numel():
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "p99_9": 0.0}
        values = torch.quantile(sample, torch.tensor([0.5, 0.9, 0.99, 0.999]))
        return {
            "p50": float(values[0]),
            "p90": float(values[1]),
            "p99": float(values[2]),
            "p99_9": float(values[3]),
        }

    def build_report(
        self,
        records: Iterable[Dict[str, Any]],
        *,
        skipped: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        records = list(records)
        self._add_anomaly_percentiles(records)
        records.sort(key=lambda item: (item["quant_score"], -item["risk_score"], item["module"]))
        regions = []
        for record in records:
            for region in record.get("regions", []):
                regions.append({"module": record["module"], "role": record["role"], **region})
        regions.sort(key=lambda item: (-item["rel_rmse"], item["module"], item["kind"]))
        regions = regions[: self.config.top_k_regions]
        plan = build_analysis_plan(
            records,
            self.qcfg,
            self.config,
            module_groups=self.module_groups,
            module_group_source=self.module_group_source,
        )
        direct_recommendations = [
            item
            for item in plan["recommendations"]
            if item["action"] in {"promote_module", "review_endpoint"}
        ]
        fusion_companions = [
            item for item in plan["recommendations"] if item["action"] == "promote_fusion_companion"
        ]
        report = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "stage": "pre_quantization_weight_proxy",
            "score_semantics": "quant_score: 100 is easiest to quantize; risk_score: 100 is highest risk",
            "config": {
                "method": str(self.qcfg.method).split(".")[-1],
                "format": str(self.qcfg.format).split(".")[-1],
                "bits": serialize_quant_bits(self.qcfg.bits),
                "group_size": self.qcfg.group_size,
                "sym": self.qcfg.sym,
                "desc_act": self.qcfg.desc_act,
            },
            "summary": {
                "analyzed_modules": len(records),
                "skipped_modules": len(skipped or []),
                "flagged_modules": len(direct_recommendations),
                "fusion_companions": len(fusion_companions),
                "proposed_dynamic_rules": len(plan["dynamic"]),
                "retained_regions": len(regions),
            },
            "records": records,
            "regions": regions,
            "skipped": skipped or [],
            "plan": plan,
            "limitations": [
                "Weight RTN error is a pre-quantization proxy, not proof of end-to-end quality impact.",
                "Activation, Hessian, packing, backend, tokenizer, and evaluator evidence require later-stage checks.",
                "The generated plan is module-granular; arbitrary mixed bits inside one packed module are not applied.",
            ],
            "indexing": {
                "module_names": "emitted verbatim from model.named_modules; no human-facing renumbering",
                "layer_indices": "preserved from qualified module names",
                "region_indices": "zero-based tensor indices",
                "group_input_end": "exclusive in JSON",
            },
        }
        report["markdown"] = render_analysis_markdown(report, top_k=self.config.top_k)
        return report

    @staticmethod
    def _add_anomaly_percentiles(records: Sequence[Dict[str, Any]]) -> None:
        by_role: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            by_role.setdefault(record["role"], []).append(record)
        for role_records in by_role.values():
            ordered = sorted(role_records, key=lambda item: (item["risk_score"], item["module"]))
            denominator = max(len(ordered), 1)
            for rank, record in enumerate(ordered, start=1):
                record["role_risk_percentile"] = round(100.0 * rank / denominator, 2)

    @staticmethod
    def _endpoint_ids(model: torch.nn.Module) -> Tuple[set[int], set[int]]:
        input_ids: set[int] = set()
        output_ids: set[int] = set()
        for getter_name, destination in (
            ("get_input_embeddings", input_ids),
            ("get_output_embeddings", output_ids),
        ):
            getter = getattr(model, getter_name, None)
            if callable(getter):
                try:
                    endpoint = getter()
                except Exception:
                    endpoint = None
                if endpoint is not None:
                    destination.add(id(endpoint))
        return input_ids, output_ids

    @staticmethod
    def _is_analyzable(module: torch.nn.Module) -> bool:
        weight = getattr(module, "weight", None)
        return (
            isinstance(weight, torch.Tensor)
            and weight.dim() >= 1
            and isinstance(
                module,
                (
                    torch.nn.Embedding,
                    torch.nn.Linear,
                    torch.nn.Conv1d,
                    torch.nn.Conv2d,
                    transformers.Conv1D,
                ),
            )
        )


def build_analysis_plan(
    records: Sequence[Dict[str, Any]],
    qcfg: BaseQuantizeConfig,
    config: AnalysisConfig,
    *,
    module_groups: Optional[Sequence[Sequence[str]]] = None,
    module_group_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a reviewable module-level mixed-precision proposal."""

    recommendations: List[Dict[str, Any]] = []
    dynamic: Dict[str, Dict[str, Any]] = {}
    current_bits = quant_bits_width(qcfg.bits)
    promotion_bits = max(current_bits, config.promotion_bits)

    for record in records:
        percentile = float(record.get("role_risk_percentile", 0.0))
        if percentile < config.recommendation_percentile:
            continue
        if float(record["risk_score"]) < config.min_recommendation_risk:
            continue
        reasons = _recommendation_reasons(record)
        endpoint = record["role"] in {"input_embedding", "embedding", "lm_head"}
        action = "review_endpoint" if endpoint else "promote_module"
        overrides: Dict[str, Any] = {"bits": promotion_bits}
        if record["group_size"] <= 0 or record["group_size"] > config.promotion_group_size:
            overrides["group_size"] = config.promotion_group_size
        recommendation = {
            "module": record["module"],
            "role": record["role"],
            "action": action,
            "confidence": "heuristic",
            "risk_score": record["risk_score"],
            "role_risk_percentile": percentile,
            "reasons": reasons,
            "proposed_overrides": overrides,
        }
        recommendations.append(recommendation)
        if not endpoint:
            dynamic[f"+:^{re.escape(record['module'])}$"] = overrides

    fusion_groups = []
    if config.fusion_profile == "model_definition":
        fusion_groups = _add_definition_group_companions(
            recommendations=recommendations,
            dynamic=dynamic,
            records=records,
            module_groups=module_groups or [],
            module_group_source=module_group_source,
        )

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "base_config": {
            "method": str(qcfg.method).split(".")[-1],
            "format": str(qcfg.format).split(".")[-1],
            "bits": serialize_quant_bits(qcfg.bits),
            "group_size": qcfg.group_size,
            "sym": qcfg.sym,
        },
        "recommendations": recommendations,
        "dynamic": dynamic,
        "fusion_profile": config.fusion_profile,
        "fusion_groups": fusion_groups,
        "application": "review_then_apply",
    }


def _add_definition_group_companions(
    *,
    recommendations: List[Dict[str, Any]],
    dynamic: Dict[str, Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    module_groups: Sequence[Sequence[str]],
    module_group_source: Optional[str],
) -> List[Dict[str, Any]]:
    records_by_name = {record["module"]: record for record in records}
    direct = {
        recommendation["module"]: recommendation
        for recommendation in recommendations
        if recommendation["action"] == "promote_module"
    }
    fusion_groups: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for module_name, recommendation in direct.items():
        match = _match_definition_group(module_name, records_by_name, module_groups)
        if match is None:
            continue
        group_index, suffixes, prefix, members = match
        kind = _definition_group_kind(suffixes)
        key = (group_index, prefix)
        group = fusion_groups.setdefault(
            key,
            {
                "kind": kind,
                "definition_group_index": group_index,
                "definition_group": list(suffixes),
                "group_source": module_group_source or "GPTQModel model definition (source unavailable)",
                "fusibility_basis": "gptqmodel_model_definition_module_group",
                "members": members,
                "triggers": [],
                "companions": [],
                "proposed_overrides": copy.deepcopy(recommendation["proposed_overrides"]),
                "reason": (
                    "GPTQModel declares these modules in the same supported model-definition group, so the analyzer "
                    "keeps their proposed quantization policy compatible. Serving-engine implementations are "
                    "context only and do not determine this closure."
                ),
            },
        )
        group["triggers"].append(module_name)

    existing_recommendations = {recommendation["module"] for recommendation in recommendations}
    for group in fusion_groups.values():
        overrides = group["proposed_overrides"]
        for member in group["members"]:
            if member in direct:
                continue
            group["companions"].append(member)
            if member in existing_recommendations:
                continue
            record = records_by_name[member]
            recommendation = {
                "module": member,
                "role": record["role"],
                "action": "promote_fusion_companion",
                "confidence": "compatibility",
                "risk_score": record["risk_score"],
                "role_risk_percentile": record["role_risk_percentile"],
                "reasons": [
                    (
                        f"model-definition group companion of {', '.join(group['triggers'])}; "
                        "GPTQModel groups these modules in one supported quantization block"
                    ),
                    group["reason"],
                ],
                "triggered_by": list(group["triggers"]),
                "fusion_group": group["kind"],
                "definition_group_index": group["definition_group_index"],
                "group_source": group["group_source"],
                "fusibility_basis": group["fusibility_basis"],
                "proposed_overrides": copy.deepcopy(overrides),
            }
            recommendations.append(recommendation)
            existing_recommendations.add(member)
            dynamic[f"+:^{re.escape(member)}$"] = copy.deepcopy(overrides)

    return sorted(
        fusion_groups.values(),
        key=lambda item: (item["definition_group_index"], item["members"][0]),
    )


def _match_definition_group(
    module_name: str,
    records_by_name: Dict[str, Dict[str, Any]],
    module_groups: Sequence[Sequence[str]],
) -> Optional[Tuple[int, List[str], str, List[str]]]:
    for group_index, group in enumerate(module_groups):
        suffixes = list(group)
        if len(suffixes) <= 1:
            continue
        for suffix in suffixes:
            if not module_name.endswith(suffix):
                continue
            prefix = module_name[: -len(suffix)]
            if prefix and not prefix.endswith("."):
                continue
            members = [f"{prefix}{candidate}" for candidate in suffixes]
            if all(member in records_by_name for member in members):
                return group_index, suffixes, prefix, members
    return None


def _definition_group_kind(suffixes: Sequence[str]) -> str:
    leaf_names = {suffix.rsplit(".", 1)[-1] for suffix in suffixes}
    if leaf_names == {"q_proj", "k_proj", "v_proj"}:
        return "qkv_projection_group"
    if leaf_names == {"gate_proj", "up_proj"}:
        return "gate_up_projection_group"
    return "model_definition_group"


def _recommendation_reasons(record: Dict[str, Any]) -> List[str]:
    reasons = []
    if record["rel_rmse"] >= 0.05:
        reasons.append(f"module relative RMSE is {record['rel_rmse']:.4f}")
    if record["bad_block_fraction"] >= 0.05:
        reasons.append(f"{record['bad_block_fraction'] * 100:.2f}% of groups exceed the error threshold")
    if record["max_to_median_abs"] >= 100:
        reasons.append(f"max/median absolute weight ratio is {record['max_to_median_abs']:.1f}")
    regions = record.get("regions", [])
    if regions:
        worst = max(regions, key=lambda item: item["rel_rmse"])
        reasons.append(f"worst {worst['kind']} relative RMSE is {worst['rel_rmse']:.4f}")
    return reasons or ["risk score is anomalous relative to modules with the same role"]


def apply_analysis_plan(
    qcfg: BaseQuantizeConfig,
    plan: Dict[str, Any],
    *,
    inplace: bool = False,
) -> Tuple[BaseQuantizeConfig, Dict[str, Any]]:
    """Explicitly merge a reviewed plan while preserving existing dynamic rules."""

    if str(plan.get("schema_version")) != PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported analysis plan schema `{plan.get('schema_version')}`.")
    destination = qcfg if inplace else copy.deepcopy(qcfg)
    if destination.dynamic is None:
        destination.dynamic = {}

    applied = []
    conflicts = []
    for pattern, overrides in plan.get("dynamic", {}).items():
        module = _exact_module_from_pattern(pattern)
        existing = destination.dynamic_get(module) if module is not None else None
        if existing is not None:
            conflicts.append(
                {
                    "pattern": pattern,
                    "module": module,
                    "reason": "an existing dynamic rule already matches; existing user rule retained",
                }
            )
            continue
        destination.dynamic[pattern] = copy.deepcopy(overrides)
        applied.append({"pattern": pattern, "overrides": copy.deepcopy(overrides)})
    return destination, {"applied": applied, "conflicts": conflicts}


def _exact_module_from_pattern(pattern: str) -> Optional[str]:
    raw = pattern.removeprefix("+:")
    if not raw.startswith("^") or not raw.endswith("$"):
        return None
    escaped = raw[1:-1]
    try:
        return re.sub(r"\\(.)", r"\1", escaped)
    except re.error:
        return None


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return lines


def _format_region(region: Optional[Dict[str, Any]]) -> str:
    if region is None:
        return "not retained"
    error = f"{region['rel_rmse'] * 100:.2f}%"
    kind = region["kind"]
    if kind == "group":
        row = region.get("row_index", region.get("output_channel"))
        start = int(region["input_start"])
        end = int(region["input_end"]) - 1
        return f"row {row}, inputs {start}–{end}: {error} (scale {region['scale']:.6g})"
    index = region["index"]
    signal = region.get("signal_l2")
    signal_text = f", signal L2 {signal:.4g}" if signal is not None else ""
    label = {
        "output_channel": "output row",
        "embedding_row": "token row",
        "vocab_row": "vocabulary row",
        "input_feature": "input feature",
    }.get(kind, kind)
    return f"{label} {index}: {error}{signal_text}"


def _worst_regions(record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], ...]:
    row_kinds = {"output_channel", "embedding_row", "vocab_row"}
    row = None
    feature = None
    group = None
    for region in record.get("regions", []):
        target = None
        if region["kind"] in row_kinds:
            target = row
        elif region["kind"] == "input_feature":
            target = feature
        elif region["kind"] == "group":
            target = group
        else:
            continue
        if target is None or region["rel_rmse"] > target["rel_rmse"]:
            if region["kind"] in row_kinds:
                row = region
            elif region["kind"] == "input_feature":
                feature = region
            else:
                group = region
    return row, feature, group


def _estimate_plan_delta_gib(
    recommendations: Sequence[Dict[str, Any]],
    records_by_name: Dict[str, Dict[str, Any]],
    base_config: Dict[str, Any],
) -> Optional[float]:
    """Estimate weight plus FP16-scale growth; backend metadata is intentionally excluded."""

    base_bits = base_config.get("bits")
    if not isinstance(base_bits, int):
        return None
    delta_bytes = 0.0
    for recommendation in recommendations:
        if recommendation.get("action") not in {"promote_module", "promote_fusion_companion"}:
            continue
        record = records_by_name.get(recommendation["module"])
        overrides = recommendation.get("proposed_overrides", {})
        promoted_bits = overrides.get("bits", base_bits)
        if record is None or not isinstance(promoted_bits, int):
            return None
        delta_bytes += int(record["numel"]) * (promoted_bits - base_bits) / 8
        rows, cols = record["shape"]
        old_group = int(record["group_size"])
        old_group = cols if old_group <= 0 else min(old_group, cols)
        new_group = int(overrides.get("group_size", old_group))
        new_group = cols if new_group <= 0 else min(new_group, cols)
        delta_bytes += rows * (math.ceil(cols / new_group) - math.ceil(cols / old_group)) * 2
    return delta_bytes / (1024**3)


def render_analysis_markdown(report: Dict[str, Any], *, top_k: int = 32) -> str:
    """Render the complete analysis as a decision-oriented human report."""

    summary = report.get("summary", {})
    config = report.get("config", {})
    records = report.get("records", [])
    records_by_name = {record["module"]: record for record in records}
    plan = report.get("plan", {})
    recommendations = plan.get("recommendations", [])
    promoted = [
        item
        for item in recommendations
        if item.get("action") in {"promote_module", "promote_fusion_companion"}
    ]
    direct_promotions = [item for item in recommendations if item.get("action") == "promote_module"]
    fusion_companions = [
        item for item in recommendations if item.get("action") == "promote_fusion_companion"
    ]
    endpoint_reviews = [item for item in recommendations if item.get("action") == "review_endpoint"]
    fusion_groups = plan.get("fusion_groups", [])
    model = report.get("model", "not recorded")

    lines = [
        "# Quantization error analysis",
        "",
        "> Evidence status: **Observed pre-quantization proxy**. These measurements localize candidates; they do "
        "not prove downstream quality impact.",
        "",
        "## Executive summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Item", "Value"],
            [
                ["Model", f"`{model}`"],
                ["Stage", f"`{report.get('stage', 'unknown')}`"],
                ["Weight source", f"`{report.get('weight_source', 'materialized_model')}`"],
                ["Analyzed modules", summary.get("analyzed_modules", len(records))],
                ["Skipped modules", summary.get("skipped_modules", len(report.get("skipped", [])))],
                ["Retained abnormal regions", summary.get("retained_regions", len(report.get("regions", [])))],
                [
                    "Directly flagged modules/endpoints",
                    summary.get("flagged_modules", len(direct_promotions) + len(endpoint_reviews)),
                ],
                [
                    "Definition-group compatibility companions",
                    summary.get("fusion_companions", len(fusion_companions)),
                ],
                ["Proposed module rules", len(promoted)],
                ["Endpoint review items", len(endpoint_reviews)],
            ],
        )
    )

    lines.extend(["", "## Quantization configuration", ""])
    lines.extend(
        _markdown_table(
            ["Setting", "Value"],
            [
                ["Method / format", f"{config.get('method', 'unknown')} / {config.get('format', 'unknown')}"],
                ["Weight precision", f"{config.get('bits', 'unknown')}-bit"],
                ["Group size", config.get("group_size", "unknown")],
                ["Symmetric", config.get("sym", "unknown")],
                ["Activation order requested", config.get("desc_act", "unknown")],
            ],
        )
    )

    runtime = report.get("runtime")
    if runtime:
        physical = runtime.get("physical_gpu", {})
        lines.extend(["", "## Runtime and hardware", ""])
        lines.extend(
            _markdown_table(
                ["Item", "Value"],
                [
                    ["Physical GPU", physical.get("physical_index", "not recorded")],
                    ["PCI bus", f"`{physical.get('pci_bus_id', 'not recorded')}`"],
                    ["GPU UUID", f"`{physical.get('uuid', 'not recorded')}`"],
                    ["GPU", runtime.get("visible_gpu_name", physical.get("name", "not recorded"))],
                    ["Compute capability / SMs", f"{runtime.get('compute_capability', '?')} / {runtime.get('sm_count', '?')}"],
                    ["Visible memory", f"{physical.get('memory_total_mib', 0) / 1024:.1f} GiB"],
                    ["PyTorch / CUDA", f"{runtime.get('torch', '?')} / {runtime.get('torch_cuda', '?')}"],
                ],
            )
        )

    indexing = report.get("indexing", {})
    checkpoint_index = report.get("checkpoint_index")
    lines.extend(["", "## Index semantics", ""])
    index_rows = [
        ["Module names", indexing.get("module_names", "not recorded")],
        ["Layer indices", indexing.get("layer_indices", "preserved without renumbering")],
        ["Row / feature / token / vocabulary indices", indexing.get("region_indices", "zero-based")],
        ["JSON group `input_end`", indexing.get("group_input_end", "exclusive")],
        ["Displayed group ranges", "zero-based and inclusive at both ends"],
    ]
    if checkpoint_index:
        index_rows.extend(
            [
                ["Safetensors index", f"`{checkpoint_index['index_file']}`"],
                [
                    "Observed checkpoint layers",
                    (
                        f"{checkpoint_index['layer_min']}–{checkpoint_index['layer_max']} "
                        f"({checkpoint_index['layer_count']} unique indices)"
                    ),
                ],
                ["Checkpoint tensor keys", checkpoint_index["tensor_key_count"]],
            ]
        )
    lines.extend(_markdown_table(["Index item", "Contract"], index_rows))
    lines.extend(
        [
            "",
            "No `+1` conversion is performed for readability. For example, `model.layers.0...` means checkpoint "
            "layer 0, not human layer 1.",
        ]
    )

    lines.extend(
        [
            "",
            "## How to read the scores",
            "",
            "- `quant_score`: inverse of risk; 100 is easiest to quantize under this proxy.",
            "- `risk_score`: composite 0–100 prioritization score; higher is more suspicious.",
            "- `role percentile`: comparison with modules of the same role. `100%` is the worst member of that role.",
            "- `relative RMSE`: `||W - Wq||₂ / ||W||₂`; lower is better.",
            "- `bad groups`: share of row groups above the configured relative-RMSE threshold.",
            "- `max/median`: absolute-weight outlier pressure. Large ratios can amplify grouped scale error.",
            "- A 100% low-signal feature error can mean that a tiny feature was rounded to zero; it is not automatically "
            "important to model quality.",
            "",
            "## Direct recommendations and definition-group companions",
            "",
        ]
    )
    recommendation_rows = []
    for recommendation in recommendations:
        record = records_by_name[recommendation["module"]]
        overrides = recommendation["proposed_overrides"]
        if recommendation["action"] == "promote_module":
            action = (
                f"direct: {overrides.get('bits', record['bits'])}-bit, "
                f"g{overrides.get('group_size', record['group_size'])}"
            )
        elif recommendation["action"] == "promote_fusion_companion":
            triggers = ", ".join(recommendation.get("triggered_by", []))
            action = (
                f"group companion: {overrides.get('bits', record['bits'])}-bit, "
                f"g{overrides.get('group_size', record['group_size'])}; trigger {triggers}"
            )
        else:
            action = "manual endpoint review"
        recommendation_rows.append(
            [
                f"`{record['module']}`",
                record["role"],
                f"{record['risk_score']:.2f}",
                f"{record['role_risk_percentile']:.2f}%",
                f"{record['rel_rmse'] * 100:.2f}%",
                f"{record['bad_block_fraction'] * 100:.2f}%",
                f"{record['max_to_median_abs']:.1f}×",
                action,
            ]
        )
    if recommendation_rows:
        lines.extend(
            _markdown_table(
                ["Module", "Role", "Risk", "Role pct", "Rel. RMSE", "Bad groups", "Max/median", "Action"],
                recommendation_rows,
            )
        )
    else:
        lines.append("No modules crossed the configured recommendation thresholds.")

    lines.extend(["", "## Exact localized regions", ""])
    localization_rows = []
    for recommendation in recommendations:
        record = records_by_name[recommendation["module"]]
        row, feature, group = _worst_regions(record)
        localization_rows.append(
            [
                f"`{record['module']}`",
                _format_region(row),
                _format_region(feature),
                _format_region(group),
            ]
        )
    if localization_rows:
        lines.extend(
            _markdown_table(
                ["Module", "Worst row", "Worst input feature", "Worst group"],
                localization_rows,
            )
        )
        lines.extend(
            [
                "",
                "Group input ranges are inclusive in this table. The JSON stores `input_end` as an exclusive bound.",
            ]
        )
    else:
        lines.append("No flagged regions were retained.")

    estimated_delta = _estimate_plan_delta_gib(recommendations, records_by_name, config)
    lines.extend(["", "## Proposed quantizer mapping", ""])
    if promoted:
        lines.append(
            f"The plan contains **{len(promoted)} exact module-level dynamic rules**: "
            f"{len(direct_promotions)} direct risk promotions and "
            f"{len(fusion_companions)} definition-group companions. "
            "Existing user-authored `dynamic` rules retain precedence when the plan is explicitly applied."
        )
        lines.append("")
        plan_rows = []
        for recommendation in promoted:
            overrides = recommendation["proposed_overrides"]
            plan_rows.append(
                [
                    f"`{recommendation['module']}`",
                    recommendation["role"],
                    overrides.get("bits", "unchanged"),
                    overrides.get("group_size", "unchanged"),
                ]
            )
        lines.extend(_markdown_table(["Module", "Role", "Bits", "Group size"], plan_rows))
        if estimated_delta is not None:
            lines.extend(
                [
                    "",
                    f"Estimated storage increase for these module rules: **approximately {estimated_delta:.2f} GiB**. "
                    "This includes weight bits and an FP16-scale approximation, but excludes backend-specific padding "
                    "and auxiliary metadata.",
                ]
            )
    else:
        lines.append("No module-level dynamic rules were proposed.")
    if endpoint_reviews:
        names = ", ".join(f"`{item['module']}`" for item in endpoint_reviews)
        lines.extend(
            [
                "",
                f"Endpoint review only: {names}. These are not inserted into the automatic dynamic mapping.",
            ]
        )

    lines.extend(["", "## GPTQModel definition-group closure", ""])
    if fusion_groups:
        lines.extend(
            [
                "Group companions are compatibility recommendations, not claims that their local error is anomalous. "
                "They inherit the triggering module's proposed policy because GPTQModel's model definition places "
                "them in the same supported module group. This definition metadata—not a vLLM/SGLang heuristic—is "
                "the analyzer's fusibility authority.",
                "",
            ]
        )
        fusion_rows = []
        for group in fusion_groups:
            overrides = group["proposed_overrides"]
            fusion_rows.append(
                [
                    group["kind"],
                    group["definition_group_index"],
                    f"`{group['group_source']}`",
                    ", ".join(f"`{member}`" for member in group["members"]),
                    ", ".join(f"`{member}`" for member in group["triggers"]),
                    ", ".join(f"`{member}`" for member in group["companions"]) or "none",
                    f"{overrides.get('bits', 'unchanged')}-bit, g{overrides.get('group_size', 'unchanged')}",
                    group["reason"],
                ]
            )
        lines.extend(
            _markdown_table(
                ["Group", "Index", "Definition source", "Members", "Direct triggers", "Companions", "Policy", "Reason"],
                fusion_rows,
            )
        )
        lines.extend(
            [
                "",
                "For Qwen3, the definition groups `q_proj`/`k_proj`/`v_proj` together and "
                "`gate_proj`/`up_proj` together. `o_proj` and `down_proj` are later, separate groups, so they are not "
                "companions of those groups. Runtime engine fusion can explain why a compatible policy is useful, "
                "but it does not add or remove companions.",
            ]
        )
    else:
        lines.append(
            "No definition groups were activated. The profile may be disabled, model-definition grouping metadata "
            "may be unavailable, or no direct recommendation matched a complete supported group."
        )

    lines.extend(["", f"## Ranked module overview (top {min(top_k, len(records))})", ""])
    ranked_rows = []
    for record in records[:top_k]:
        ranked_rows.append(
            [
                f"`{record['module']}`",
                record["role"],
                f"{record['quant_score']:.2f}",
                f"{record['risk_score']:.2f}",
                f"{record['rel_rmse'] * 100:.3f}%",
                f"{record['bad_block_fraction'] * 100:.2f}%",
                f"{record['max_to_median_abs']:.1f}×",
                "×".join(str(value) for value in record["shape"]),
            ]
        )
    lines.extend(
        _markdown_table(
            ["Module", "Role", "Quant score", "Risk", "Rel. RMSE", "Bad groups", "Max/median", "Shape"],
            ranked_rows,
        )
    )

    lines.extend(
        [
            "",
            "## Generated artifacts",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Artifact", "Purpose"],
            [
                ["`quantization_analysis.md`", "This human-readable decision report"],
                ["`quantization_analysis.json`", "Complete module metrics, evidence, runtime, and limitations"],
                ["`quantization_regions.json`", "Bounded row, feature, and group mappings"],
                ["`quantization_plan.json`", "Reviewable recommendations and proposed dynamic overrides"],
                ["`quantize_config.planned.json`", "Emitted only with `--apply-plan`; ready for a controlled run"],
            ],
        )
    )

    lines.extend(["", "## Interpretation and limitations", ""])
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "- `desc_act`, Hessian feedback, calibration coverage, and live activation importance are not modeled by "
            "this weight-only RTN projection.",
            "- Treat the plan as a controlled experiment. Require held-out activation/logit or evaluation rescue before "
            "calling any flagged module causal.",
            "",
            "**Conclusion:** the artifact identifies where to investigate and what module-level precision controls to "
            "test. It does not independently establish that applying every recommendation improves model quality.",
        ]
    )
    return "\n".join(lines)


def report_to_json(report: Dict[str, Any]) -> str:
    """Serialize a report with stable formatting."""

    return json.dumps(report, indent=2, sort_keys=True)


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "AnalysisSelection",
    "QuantizationAnalyzer",
    "WeightResolver",
    "apply_analysis_plan",
    "build_analysis_plan",
    "render_analysis_markdown",
    "report_to_json",
    "score_quantizability",
]
