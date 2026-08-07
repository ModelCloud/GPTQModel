# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import transformers

from ..looper.loop_processor import ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..models._const import SUPPORTS_MODULE_TYPES
from ..models.base import (
    CAPTURE_ONLY_FLAG,
    MODULE_TREE_FLAG_DOWN,
    MODULE_TREE_FLAG_GATE,
    MODULE_TREE_FLAG_K,
    MODULE_TREE_FLAG_Q,
    MODULE_TREE_FLAG_UP,
    MODULE_TREE_FLAG_V,
)
from ..quantization.analysis import QuantizationAnalyzer, report_to_json, score_quantizability
from ..quantization.config import AnalysisConfig
from ..utils.logger import setup_logger
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
        self.report: Dict[str, Any] = {}
        self.plan: Dict[str, Any] = {}
        self.analyzer = QuantizationAnalyzer(qcfg, self.config, compute_device=CPU)
        self._checkpoint_weight_resolver = None
        self._module_tree_flags_resolver = lambda _name: frozenset()
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

        if self._analyzed:
            return

        gptq_model = kwargs.get("gptq_model")
        if gptq_model is not None:
            group_source = (
                f"{type(gptq_model).__module__}.{type(gptq_model).__name__}.module_tree"
            )
            self.analyzer.set_module_groups(layer_modules, source=group_source)
            resolver = getattr(gptq_model, "get_module_tree_flags", None)
            if callable(resolver):
                self._module_tree_flags_resolver = resolver
        turtle = getattr(gptq_model, "turtle_model", None)
        model = kwargs.get("model")
        if model is not None and callable(getattr(turtle, "checkpoint_tensors_for_submodule", None)):
            def _resolve_checkpoint_weight(module):
                tensors = turtle.checkpoint_tensors_for_submodule(
                    target_model=model,
                    target_submodule=module,
                    recurse=False,
                )
                return tensors.get("weight")

            self._checkpoint_weight_resolver = _resolve_checkpoint_weight

        records: List[Dict[str, Any]] = []
        analyzed_names = set()
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
                    analyzed_names.add(full_name)
                except Exception as exc:
                    log.warn(f"AnalysisProcessor: skipped `{full_name}` due to analysis error: {exc}")

        if self.config.include_endpoints:
            self._analyze_endpoints(
                model=model,
                records=records,
                analyzed_names=analyzed_names,
            )

        self.records = records
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

        bits = self.qcfg.dynamic_get(full_name, "bits", self.qcfg.bits)
        group_size = int(self.qcfg.dynamic_get(full_name, "group_size", self.qcfg.group_size))
        sym = bool(self.qcfg.dynamic_get(full_name, "sym", self.qcfg.sym))
        record = self.analyzer.analyze_module(
            module,
            module_name=full_name,
            role=self._role_from_flags(self._module_tree_flags_resolver(module_name)),
            bits=bits,
            group_size=group_size,
            sym=sym,
            weight=self._resolve_checkpoint_weight(module),
        )
        record["layer"] = layer_index
        record["name"] = module_name
        return record

    def _analyze_endpoints(
        self,
        *,
        model: Optional[torch.nn.Module],
        records: List[Dict[str, Any]],
        analyzed_names: set[str],
    ) -> None:
        """Include input embeddings and LM head even when they are outside decoder layers."""

        if model is None:
            return
        names_by_id = {id(module): name for name, module in model.named_modules()}
        for getter_name, role in (
            ("get_input_embeddings", "input_embedding"),
            ("get_output_embeddings", "lm_head"),
        ):
            getter = getattr(model, getter_name, None)
            if not callable(getter):
                continue
            try:
                module = getter()
            except Exception as exc:
                log.warn(f"AnalysisProcessor: failed to resolve `{getter_name}`: {exc}")
                continue
            if module is None or not isinstance(module, tuple(SUPPORTS_MODULE_TYPES)):
                continue
            full_name = names_by_id.get(id(module), getter_name)
            if full_name in analyzed_names or self.qcfg.dynamic_get(layer_name=full_name) is False:
                continue
            try:
                bits = self.qcfg.dynamic_get(full_name, "bits", self.qcfg.bits)
                group_size = int(self.qcfg.dynamic_get(full_name, "group_size", self.qcfg.group_size))
                sym = bool(self.qcfg.dynamic_get(full_name, "sym", self.qcfg.sym))
                record = self.analyzer.analyze_module(
                    module,
                    module_name=full_name,
                    role=role,
                    bits=bits,
                    group_size=group_size,
                    sym=sym,
                    weight=self._resolve_checkpoint_weight(module),
                )
                record["layer"] = None
                record["name"] = full_name.rsplit(".", 1)[-1]
                records.append(record)
                analyzed_names.add(full_name)
            except Exception as exc:
                log.warn(f"AnalysisProcessor: skipped endpoint `{full_name}` due to analysis error: {exc}")

    def _resolve_checkpoint_weight(self, module: torch.nn.Module) -> Optional[torch.Tensor]:
        """Read a shell module's weight from LazyTurtle without mutating the shell."""

        if self._checkpoint_weight_resolver is None:
            return None
        return self._checkpoint_weight_resolver(module)

    @staticmethod
    def _role_from_flags(flags: frozenset[str]) -> str:
        for flag, role in (
            (MODULE_TREE_FLAG_Q, "attention_q"),
            (MODULE_TREE_FLAG_K, "attention_k"),
            (MODULE_TREE_FLAG_V, "attention_v"),
            (MODULE_TREE_FLAG_GATE, "mlp_gate"),
            (MODULE_TREE_FLAG_UP, "mlp_up"),
            (MODULE_TREE_FLAG_DOWN, "mlp_down"),
        ):
            if flag in flags:
                return role
        return "linear"

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
        if cols % effective_group_size == 0:
            return self._group_quant_stats_divisible(
                weight=weight,
                bit_width=bit_width,
                group_size=effective_group_size,
                sym=sym,
            )

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

    def _group_quant_stats_divisible(
        self,
        *,
        weight: torch.Tensor,
        bit_width: int,
        group_size: int,
        sym: bool,
    ) -> Dict[str, Any]:
        """Vectorized grouped stats for the common no-tail group layout."""

        rows, cols = weight.shape
        qmax = (2 ** (bit_width - 1) - 1) if sym else (2 ** bit_width - 1)
        qmin = -qmax if sym else 0
        eps = torch.finfo(torch.float32).eps

        total_values = int(weight.numel())
        total_signal = float(torch.sum(weight * weight).item())
        max_abs = float(torch.max(torch.abs(weight)).item()) if total_values else 0.0
        abs_flat = torch.abs(weight).reshape(-1)
        nonzero_abs = abs_flat[abs_flat > 0]
        median_abs = float(torch.median(nonzero_abs).item()) if nonzero_abs.numel() else 0.0

        if not total_values:
            return {
                "shape": (int(rows), int(cols)),
                "rel_rmse": 0.0,
                "small_value_fraction": 0.0,
                "bad_block_fraction": 0.0,
                "max_abs": 0.0,
                "median_abs": 0.0,
                "max_to_median_abs": 0.0,
                "num_blocks": 0,
            }

        total_sse = 0.0
        total_small = 0
        total_bad_blocks = 0
        total_blocks = 0
        num_groups = cols // group_size
        max_chunk_values = 8 * 1024 * 1024
        row_chunk = max(1, min(rows, max_chunk_values // max(cols, 1)))

        for row_start in range(0, rows, row_chunk):
            chunk = weight[row_start : row_start + row_chunk]
            grouped = chunk.reshape(chunk.shape[0], num_groups, group_size)
            if sym:
                scale = torch.amax(torch.abs(grouped), dim=2, keepdim=True).clamp_min(eps) / max(qmax, 1)
                quantized = torch.round(grouped / scale).clamp(qmin, qmax)
                dequantized = quantized * scale
            else:
                block_min = torch.amin(grouped, dim=2, keepdim=True)
                block_max = torch.amax(grouped, dim=2, keepdim=True)
                scale = (block_max - block_min).clamp_min(eps) / max(qmax, 1)
                zero = torch.round(-block_min / scale).clamp(qmin, qmax)
                quantized = torch.round(grouped / scale + zero).clamp(qmin, qmax)
                dequantized = (quantized - zero) * scale

            diff = grouped - dequantized
            total_sse += float(torch.sum(diff * diff).item())
            total_small += int((torch.abs(grouped) <= (0.5 * scale)).sum().item())

            block_sse = torch.sum(diff * diff, dim=2)
            block_signal = torch.sum(grouped * grouped, dim=2).clamp_min(eps)
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

        return score_quantizability(stats)

    def _build_reports(self) -> None:
        """Build markdown and JSON reports from ranked records."""

        self.report = self.analyzer.build_report(self.records)
        self.records = self.report["records"]
        self.plan = self.report["plan"]
        self.markdown = self.report["markdown"]
        serializable_report = {key: value for key, value in self.report.items() if key != "markdown"}
        self.json_payload = report_to_json(serializable_report)

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
            "regions": self.report.get("regions", []),
            "plan": self.plan,
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
