# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, List, Tuple
from unittest import mock

import torch
from model_test import ModelTest

from gptqmodel import GPTQModel
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.models.writer import (
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.torch import torch_empty_cache


@dataclass(frozen=True)
class LayerMetrics:
    loss: Decimal
    samples: int



def _is_free_threaded() -> bool:
    gil_check = getattr(sys, "_is_gil_enabled", None)
    if callable(gil_check):
        return not gil_check()
    env_value = os.environ.get("PYTHON_GIL", "1").lower()
    return env_value in {"0", "false", "off"}


class TestMultiVsSingleGPU(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3311
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3549
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.05
    APPLY_CHAT_TEMPLATE = True
    V2 = False
    DEBUG = True
    ACT_GROUP_AWARE = False
    DESC_ACT = True
    DATASET_SIZE = 1024
    DATASET_SORT = "desc"
    QUANT_BATCH_SIZE = 4
    USE_FLASH_ATTN = True

    def test_quantization_first_layer_metrics_match_between_single_and_dual_gpu(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for multi-GPU regression test")

        visible_devices = torch.cuda.device_count()
        if visible_devices < 2:
            self.skipTest("Requires at least two CUDA devices")

        if sys.version_info < (3, 13):
            self.skipTest("Requires Python 3.13 free-threaded runtime")

        if not _is_free_threaded():
            self.skipTest("PYTHON_GIL must be disabled (set PYTHON_GIL=0) for multi-threaded quantization")

        single_layer_metrics, single_batch_stats = self._quantize_first_layer(device_indices=[0])
        multi_layer_metrics, multi_batch_stats = self._quantize_first_layer(device_indices=[0, 1])

        self.assertTrue(single_layer_metrics, "Single-GPU quantization produced no layer-0 metrics")
        self.assertTrue(multi_layer_metrics, "Multi-GPU quantization produced no layer-0 metrics")
        self.assertEqual(
            set(single_layer_metrics.keys()),
            set(multi_layer_metrics.keys()),
            "Layer-0 module set differs between single-GPU and multi-GPU quantization",
        )

        mismatches: Dict[str, Dict[str, str]] = {}
        for module_name in single_layer_metrics:
            single = single_layer_metrics[module_name]
            multi = multi_layer_metrics[module_name]
            if single.samples != multi.samples or single.loss != multi.loss:
                mismatches[module_name] = {
                    "single_samples": str(single.samples),
                    "multi_samples": str(multi.samples),
                    "single_loss": str(single.loss),
                    "multi_loss": str(multi.loss),
                }

        if mismatches:
            debug_details = self._format_batch_debug(single_batch_stats, multi_batch_stats)
            details = "; ".join(
                f"{module}: loss {info['single_loss']} vs {info['multi_loss']}, "
                f"samples {info['single_samples']} vs {info['multi_samples']}"
                for module, info in mismatches.items()
            )
            self.fail(
                "Layer-0 quantization metrics diverged between device configurations: "
                f"{details}; batch-debug: {debug_details}"
            )

    def _quantize_first_layer(
        self, device_indices: Iterable[int]
    ) -> Tuple[Dict[str, LayerMetrics], Dict[str, List[Dict[str, float]]]]:
        target_devices = [torch.device(f"cuda:{idx}") for idx in device_indices]
        def selection(_base_device):
            return target_devices

        class _StopAfterLayer:
            def __init__(self, layer_idx: int):
                self._layer_idx = layer_idx
                self._triggered = False

            def layer_complete(self, *, layer_idx: int, submodule_finalized: bool):
                if self._triggered:
                    return None
                if submodule_finalized and layer_idx >= self._layer_idx:
                    self._triggered = True
                    raise StopMainLoop

        quant_config = QuantizeConfig(
            quant_method=self.METHOD,
            format=self.FORMAT,
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            desc_act=self.DESC_ACT if not self.ACT_GROUP_AWARE else False,
            act_group_aware=self.ACT_GROUP_AWARE,
            fail_safe=self.FAIL_SAFE,
            sym=self.SYM,
            v2=self.V2,
            adapter=self.EORA,
            device=target_devices[0],
        )

        load_kwargs = {}
        if self.USE_FLASH_ATTN:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=quant_config,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
            debug=self.DEBUG,
            **load_kwargs,
        )

        dataset = self.load_dataset(model.tokenizer, self.DATASET_SIZE)
        model.layer_callback = _StopAfterLayer(layer_idx=0)

        batch_debug: Dict[str, List[Dict[str, float]]] = {}
        primary_handles: Dict[str, str] = {}

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch("gptqmodel.looper.module_looper.select_forward_devices", new=selection)
            )
            stack.enter_context(
                mock.patch("gptqmodel.utils.looper_helpers.select_forward_devices", new=selection)
            )
            stack.enter_context(self._capture_primary_handles(primary_handles))
            stack.enter_context(self._capture_batches(batch_debug, primary_handles))
            model.quantize(
                dataset,
                calibration_sort=self.DATASET_SORT,
                backend=self.QUANT_BACKEND,
                batch_size=self.QUANT_BATCH_SIZE,
            )

        first_layer_stats = self._extract_first_layer_metrics(model.quant_log)

        # Clear GPU memory before the next run
        del dataset
        del model
        torch_empty_cache()

        return first_layer_stats, batch_debug

    def _extract_first_layer_metrics(self, quant_log: List[Dict[str, str]]) -> Dict[str, LayerMetrics]:
        layer_metrics: Dict[str, LayerMetrics] = {}
        for entry in quant_log:
            try:
                layer_index = int(entry.get(PROCESS_LOG_LAYER))
            except (TypeError, ValueError):
                continue

            if layer_index != 0:
                continue

            module_name = entry.get(PROCESS_LOG_MODULE)
            if not module_name:
                continue

            loss_value = entry.get(QUANT_LOG_LOSS)
            sample_value = entry.get(QUANT_LOG_NSAMPLES)
            if loss_value is None or sample_value is None:
                continue

            layer_metrics[module_name] = LayerMetrics(
                loss=Decimal(loss_value),
                samples=int(sample_value),
            )
        return layer_metrics

    @staticmethod
    def _format_batch_debug(
        single_batch_stats: Dict[str, List[Dict[str, float]]],
        multi_batch_stats: Dict[str, List[Dict[str, float]]],
    ) -> str:
        def _summarize(stats: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
            summary: Dict[str, Dict[str, float]] = {}
            for name, entries in stats.items():
                per_handle: Dict[str, Dict[str, float]] = {}
                for item in entries:
                    handle = item["handle"]
                    info = per_handle.setdefault(
                        handle,
                        {
                            "batches": 0.0,
                            "samples": 0.0,
                            "sum_hash": 0.0,
                            "device": item.get("device", "?"),
                            "primary": False,
                        },
                    )
                    info["batches"] += 1.0
                    info["samples"] += item.get("tokens", 0.0)
                    info["sum_hash"] += item["sum"]
                    info["primary"] = info["primary"] or bool(item.get("is_primary", False))
                summary[name] = dict(sorted(per_handle.items(), key=lambda kv: kv[0]))
            return summary

        single_summary = _summarize(single_batch_stats)
        multi_summary = _summarize(multi_batch_stats)

        parts = []
        module_names = sorted(set(single_summary) | set(multi_summary))
        for module in module_names:
            single_info = single_summary.get(module, {})
            multi_info = multi_summary.get(module, {})
            parts.append(
                f"{module}:single={single_info},multi={multi_info}"
            )
        return " | ".join(parts)

    @staticmethod
    def _capture_batches(storage: Dict[str, List[Dict[str, float]]], primary_handles: Dict[str, str]):
        from gptqmodel.quantization.gptq import GPTQ  # local import to avoid circular refs

        original_add_batch = GPTQ.add_batch

        def wrapped_add_batch(self, inp, out, batch_index=None):  # type: ignore[override]
            module_name = getattr(self, "name", "<unknown>")
            # Summaries calculated before running original implementation
            try:
                sum_value = inp.detach().to(dtype=torch.float64).sum().item()
            except Exception:  # pragma: no cover - defensive logging
                sum_value = float("nan")
            device = str(getattr(inp, "device", "unknown"))

            token_count = float(inp.numel() // max(inp.shape[-1], 1))

            original_add_batch(self, inp, out, batch_index=batch_index)

            storage.setdefault(module_name, []).append(
                {
                    "tokens": token_count,
                    "sum": float(sum_value),
                    "handle": hex(id(self)),
                    "device": device,
                    "is_primary": hex(id(self)) == primary_handles.get(module_name),
                    "batch_index": None if batch_index is None else int(batch_index),
                }
            )

        return mock.patch.object(GPTQ, "add_batch", new=wrapped_add_batch)

    @staticmethod
    def _capture_primary_handles(primary_handles: Dict[str, str]):
        from gptqmodel.looper.gptq_processor import GPTQProcessor  # local import to avoid cycles

        original_preprocess = GPTQProcessor.preprocess

        def wrapped_preprocess(self, module, fail_safe=False):  # type: ignore[override]
            result = original_preprocess(self, module, fail_safe)
            task = self.tasks.get(module.name)
            if task is not None:
                primary_handles[module.name] = hex(id(task))
            return result

        return mock.patch.object(GPTQProcessor, "preprocess", new=wrapped_preprocess)
