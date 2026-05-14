# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import torch
import transformers

from ..looper.loop_processor import ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..models._const import SUPPORTS_MODULE_TYPES
from ..models.base import CAPTURE_ONLY_FLAG
from ..quantization.config import AnalysisConfig, quant_bits_width, serialize_quant_bits
from ..utils.logger import render_table, setup_logger
from ..utils.model import get_module
from ..utils.torch import CPU

log = setup_logger()


class AnalysisProcessor(LoopProcessor):
    """Scan pre-quant weights and rank modules by config-specific quantizability."""

    def __init__(self, *args, **kwargs):
        """Initialize a no-forward processor used only for early weight analysis."""

        kwargs = dict(kwargs)
        kwargs.pop("calculate_w_wq_diff", None)
        qcfg = kwargs.pop("qcfg")
        tokenizer = kwargs.pop("tokenizer", None)
        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=None,
            calibration_concat_size=None,
            calibration_sort=None,
            calibration_concat_separator=None,
            batch_size=1,
            execution_config=ExecutionConfig(
                require_fwd=False,
                fwd_replay_after_process=False,
            ),
        )
        self.qcfg = qcfg
        self.config = self._resolve_config()
        self.records: List[Dict[str, Any]] = []
        self.markdown = ""
        self.json_payload = ""
        self._analyzed = False

    def _resolve_config(self) -> AnalysisConfig:
        """Find the configured analysis payload from normalized preprocessors."""

        for preprocessor in getattr(self.qcfg, "preprocessors", []) or []:
            if isinstance(preprocessor, AnalysisConfig):
                return preprocessor
        return AnalysisConfig()

    def analyze_model(
        self,
        *,
        layers,
        layer_modules: List[List[str]],
        layers_prefix: Optional[str],
        **kwargs,
    ) -> None:
        """Run the whole-model weight scan before layer quantization starts."""

        del kwargs
        if self._analyzed:
            return

        records: List[Dict[str, Any]] = []
        unique_module_names = self._unique_module_names(layer_modules)
        for layer_index, layer in enumerate(layers):
            for module_name in unique_module_names:
                full_name = (
                    f"{layers_prefix}.{layer_index}.{module_name}"
                    if layers_prefix
                    else f"{layer_index}.{module_name}"
                )
                if self.qcfg.dynamic_get(layer_name=full_name) is False:
                    continue
                module = get_module(layer, module_name)
                if module is None or not isinstance(module, tuple(SUPPORTS_MODULE_TYPES)):
                    continue
                try:
                    records.append(
                        self._analyze_module(
                            module=module,
                            layer_index=layer_index,
                            module_name=module_name,
                            full_name=full_name,
                        )
                    )
                except Exception as exc:
                    log.warn(f"AnalysisProcessor: skipped `{full_name}` due to analysis error: {exc}")

        self.records = sorted(records, key=lambda item: (item["quant_score"], -item["risk_score"], item["module"]))
        self._build_reports()
        self.log = self.records[: self.config.top_k]
        self._emit_reports()
        self._analyzed = True

    def _unique_module_names(self, layer_modules: List[List[str]]) -> List[str]:
        """Return unique quantized module names in forward order."""

        seen = set()
        names: List[str] = []
        for group in layer_modules:
            for name in group:
                normalized = name.split(CAPTURE_ONLY_FLAG, 1)[0]
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                names.append(normalized)
        return names

    def _analyze_module(
        self,
        *,
        module: torch.nn.Module,
        layer_index: int,
        module_name: str,
        full_name: str,
    ) -> Dict[str, Any]:
        """Analyze one module under the module-specific dynamic quant config."""

        weight = self._weight_matrix(module)
        bits = self.qcfg.dynamic_get(full_name, "bits", self.qcfg.bits)
        group_size = int(self.qcfg.dynamic_get(full_name, "group_size", self.qcfg.group_size))
        sym = bool(self.qcfg.dynamic_get(full_name, "sym", self.qcfg.sym))
        bit_width = quant_bits_width(bits)

        stats = self._group_quant_stats(
            weight=weight,
            bit_width=bit_width,
            group_size=group_size,
            sym=sym,
        )
        del weight

        quant_score = self._score_quantizability(stats)
        risk_score = round(100.0 - quant_score, 2)
        rows, cols = stats["shape"]
        return {
            "module": full_name,
            "layer": layer_index,
            "name": module_name,
            "shape": [rows, cols],
            "method": str(self.qcfg.method).split(".")[-1],
            "format": str(self.qcfg.format).split(".")[-1],
            "bits": serialize_quant_bits(bits),
            "group_size": group_size,
            "sym": sym,
            "quant_score": quant_score,
            "risk_score": risk_score,
            "rel_rmse": stats["rel_rmse"],
            "small_value_fraction": stats["small_value_fraction"],
            "bad_block_fraction": stats["bad_block_fraction"],
            "max_abs": stats["max_abs"],
            "median_abs": stats["median_abs"],
            "max_to_median_abs": stats["max_to_median_abs"],
            "num_blocks": stats["num_blocks"],
        }

    def _weight_matrix(self, module: torch.nn.Module) -> torch.Tensor:
        """Return a CPU float32 2-D view matching quantization row grouping."""

        weight = module.weight.detach()
        if isinstance(module, transformers.Conv1D):
            weight = weight.T
        elif weight.dim() > 2:
            weight = weight.reshape(weight.shape[0], -1)
        elif weight.dim() == 1:
            weight = weight.reshape(1, -1)
        return weight.to(device=CPU, dtype=torch.float32)

    def _group_quant_stats(
        self,
        *,
        weight: torch.Tensor,
        bit_width: int,
        group_size: int,
        sym: bool,
    ) -> Dict[str, Any]:
        """Estimate RTN-style grouped quantization error and outlier pressure."""

        rows, cols = weight.shape
        effective_group_size = cols if group_size <= 0 else min(group_size, cols)
        qmax = (2 ** (bit_width - 1) - 1) if sym else (2 ** bit_width - 1)
        qmin = -qmax if sym else 0
        eps = torch.finfo(torch.float32).eps

        total_sse = 0.0
        total_signal = float(torch.sum(weight * weight).item())
        total_values = int(weight.numel())
        total_small = 0
        total_bad_blocks = 0
        total_blocks = 0
        max_abs = float(torch.max(torch.abs(weight)).item()) if total_values else 0.0
        abs_flat = torch.abs(weight).reshape(-1)
        nonzero_abs = abs_flat[abs_flat > 0]
        median_abs = float(torch.median(nonzero_abs).item()) if nonzero_abs.numel() else 0.0

        for start in range(0, cols, effective_group_size):
            block = weight[:, start : start + effective_group_size]
            if sym:
                scale = torch.amax(torch.abs(block), dim=1, keepdim=True).clamp_min(eps) / max(qmax, 1)
                quantized = torch.round(block / scale).clamp(qmin, qmax)
                dequantized = quantized * scale
            else:
                block_min = torch.amin(block, dim=1, keepdim=True)
                block_max = torch.amax(block, dim=1, keepdim=True)
                scale = (block_max - block_min).clamp_min(eps) / max(qmax, 1)
                zero = torch.round(-block_min / scale).clamp(qmin, qmax)
                quantized = torch.round(block / scale + zero).clamp(qmin, qmax)
                dequantized = (quantized - zero) * scale

            diff = block - dequantized
            total_sse += float(torch.sum(diff * diff).item())
            total_small += int((torch.abs(block) <= (0.5 * scale)).sum().item())

            block_sse = torch.sum(diff * diff, dim=1)
            block_signal = torch.sum(block * block, dim=1).clamp_min(eps)
            block_rel_rmse = torch.sqrt(block_sse / block_signal)
            total_bad_blocks += int((block_rel_rmse > self.config.bad_block_rel_rmse_threshold).sum().item())
            total_blocks += int(block_rel_rmse.numel())

        rel_rmse = math.sqrt(total_sse / max(total_signal, eps))
        small_value_fraction = total_small / max(total_values, 1)
        bad_block_fraction = total_bad_blocks / max(total_blocks, 1)
        max_to_median_abs = max_abs / max(median_abs, eps)

        return {
            "shape": (int(rows), int(cols)),
            "rel_rmse": float(rel_rmse),
            "small_value_fraction": float(small_value_fraction),
            "bad_block_fraction": float(bad_block_fraction),
            "max_abs": float(max_abs),
            "median_abs": float(median_abs),
            "max_to_median_abs": float(max_to_median_abs),
            "num_blocks": int(total_blocks),
        }

    def _score_quantizability(self, stats: Dict[str, Any]) -> float:
        """Convert measured risks into a 0-100 score where 100 is best."""

        rel_rmse_penalty = min(55.0, stats["rel_rmse"] * 300.0)
        small_value_penalty = min(25.0, stats["small_value_fraction"] * 30.0)
        bad_block_penalty = min(30.0, stats["bad_block_fraction"] * 45.0)
        outlier_log = math.log10(max(stats["max_to_median_abs"], 1.0))
        outlier_penalty = min(35.0, max(0.0, outlier_log - 1.0) * 10.0)
        score = 100.0 - rel_rmse_penalty - small_value_penalty - bad_block_penalty - outlier_penalty
        return round(max(0.0, min(100.0, score)), 2)

    def _build_reports(self) -> None:
        """Build markdown and JSON reports from ranked records."""

        top_records = self.records[: self.config.top_k]
        rows = [
            [
                item["module"],
                item["quant_score"],
                item["risk_score"],
                item["bits"],
                item["group_size"],
                f"{item['rel_rmse'] * 100:.3f}",
                f"{item['small_value_fraction'] * 100:.2f}",
                f"{item['bad_block_fraction'] * 100:.2f}",
                f"{item['max_to_median_abs']:.1f}",
                f"{item['shape'][0]}x{item['shape'][1]}",
            ]
            for item in top_records
        ]
        self.markdown = render_table(
            rows,
            headers=[
                "module",
                "quant_score",
                "risk_score",
                "bits",
                "group",
                "rel_rmse_%",
                "small_%",
                "bad_blocks_%",
                "max/median",
                "shape",
            ],
            tablefmt="github",
        )
        self.json_payload = json.dumps(
            {
                "score_semantics": "quant_score: 100 is easiest to quantize; low scores are EoRA candidates",
                "top_k": self.config.top_k,
                "records": top_records,
            },
            indent=2,
        )

    def _emit_reports(self) -> None:
        """Emit the configured report formats immediately after the scan."""

        if not self.records:
            log.info("AnalysisProcessor: no quantizable modules found for analysis.")
            return
        if self.config.emit_markdown:
            log.info(
                "AnalysisProcessor: early quantizability report "
                f"(100=easiest, low score=problematic/EoRA candidate):\n{self.markdown}"
            )
        if self.config.emit_json:
            log.info(f"AnalysisProcessor: early quantizability JSON:\n{self.json_payload}")

    def preprocess(self, module: NamedModule, **kwargs):
        """No-op because the model-wide scan already happened before layers run."""

        del module, kwargs

    def is_skipped(self, module: NamedModule) -> bool:
        """Keep modules in the no-forward lifecycle so processor ordering is stable."""

        del module
        return False

    def pre_process_fwd_hook(self, name: str):
        """Return a no-op hook; analysis does not inspect activations."""

        del name

        def _noop(module, inputs, output):
            del module, inputs, output
            return None

        return _noop

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """No-op layer process hook for the no-forward lifecycle."""

        del module, device, subset, previous_subset, subset_index, subset_total

    def finalize(self, model, **kwargs):
        """Expose the analysis payload on the model after quantization."""

        del kwargs
        model.quantize_analysis = {
            "records": self.records,
            "markdown": self.markdown,
            "json": self.json_payload,
        }

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Report that analysis is weight-only and does not require calibration inputs."""

        del processor_index
        return False

    def name(self) -> str:
        """Return the processor label used in logs."""

        return "analysis"


__all__ = ["AnalysisProcessor"]
