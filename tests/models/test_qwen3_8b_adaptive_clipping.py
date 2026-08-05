# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Adaptive Clipping v1 comparison for Qwen3-8B (base model, no chat template)."""

import copy
import hashlib
import json
import logging
import os
import tempfile
import time

import pytest
from model_test import ModelTest

from gptqmodel import BACKEND

log = logging.getLogger(__name__)


class TestQwen3_8BAdaptiveClipping(ModelTest):
    """Compare Adaptive Clipping v1 on Qwen3-8B vs. the same config with clipping disabled."""

    NATIVE_MODEL_ID = os.environ.get(
        "QWEN3_8B_MODEL_PATH",
        "/monster/data/model/Qwen3-8B",
    )
    SAVE_PATH = None
    DELETE_QUANTIZED_MODEL = True
    LOAD_BACKEND = BACKEND.AUTO
    QUANT_BACKEND = BACKEND.AUTO
    APPLY_CHAT_TEMPLATE = False

    EVAL_BATCH_SIZE = "auto"
    DATASET_SIZE = 128
    DATASET_SIZE_FAST = 128
    DATASET_CONCAT_SIZE = 512
    DATASET_CONCAT_SIZE_FAST = 512

    # Qwen3-8B is a base model; do not wrap prompts with a chat template.
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": False,
            "acc,num": {"value": 0.92, "floor_pct": 0.05, "ceil_pct": 0.20},
        },
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.54, "floor_pct": 0.05, "ceil_pct": 0.20},
            "acc_norm": {"value": 0.55, "floor_pct": 0.05, "ceil_pct": 0.20},
            "evalution_suite_kwargs": {"stream": False},
            "evalution_batch_size": 64,
        },
    }
    EVAL_TASKS_FAST = copy.deepcopy(EVAL_TASKS_SLOW)

    # Use the v4.2 adaptive-damping defaults in both branches so the comparison
    # isolates only the effect of adaptive clipping.
    V42_ADAPTIVE_DAMPING = {
        "enabled": True,
        "base_percdamp": 0.05,
        "min": 0.02,
        "max": 0.08,
        "method": "power_iteration",
        "eigen_iterations": 10,
        "spectral_alpha": 0.25,
        "group_error_enabled": True,
        "group_size_prior_enabled": True,
        "group_error_use_hessian_weighting": True,
        "group_error_measure_raw_residual": True,
    }

    ADAPTIVE_DAMPING = V42_ADAPTIVE_DAMPING

    BASELINE_ADAPTIVE_CLIPPING = None
    ENABLED_ADAPTIVE_CLIPPING = {
        "enabled": True,
        "metric": "hessian_diag",
        "per_group": True,
    }

    ALLOWED_REGRESSION_PCT = 0.5

    # Cache the no-clipping baseline so repeated iterations only quantize once.
    BASELINE_CACHE_DIR = os.environ.get(
        "GPTQMODEL_QWEN3_8B_ADAPTIVE_CLIPPING_BASELINE_CACHE",
        "/tmp/qwen3_8b_adaptive_clipping_baseline",
    )
    BASELINE_RESULTS_JSON = os.path.join(BASELINE_CACHE_DIR, "eval_records.json")

    @classmethod
    def _baseline_config_hash(cls):
        payload = {
            "adaptive_damping": cls.V42_ADAPTIVE_DAMPING,
            "adaptive_clipping": cls.BASELINE_ADAPTIVE_CLIPPING,
            "dataset_size": cls.DATASET_SIZE,
            "dataset_concat_size": cls.DATASET_CONCAT_SIZE,
            "eval_tasks": cls.EVAL_TASKS_SLOW,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _load_baseline_cache(cls):
        if not os.path.exists(cls.BASELINE_RESULTS_JSON):
            return None
        try:
            with open(cls.BASELINE_RESULTS_JSON, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if cached.get("config_hash") != cls._baseline_config_hash():
            return None
        return cached.get("results"), cached.get("elapsed", 0.0)

    @classmethod
    def _save_baseline_cache(cls, results, elapsed):
        os.makedirs(cls.BASELINE_CACHE_DIR, exist_ok=True)
        payload = {
            "config_hash": cls._baseline_config_hash(),
            "elapsed": elapsed,
            "results": results,
        }
        with open(cls.BASELINE_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _is_baseline_config(self, clipping_cfg):
        return clipping_cfg is self.BASELINE_ADAPTIVE_CLIPPING or clipping_cfg == self.BASELINE_ADAPTIVE_CLIPPING

    def _run_config(self, clipping_cfg, save_path):
        is_baseline = self._is_baseline_config(clipping_cfg)
        if is_baseline:
            cached = self._load_baseline_cache()
            if cached is not None:
                results, elapsed = cached
                log.info("Reusing cached adaptive-clipping baseline from %s", self.BASELINE_CACHE_DIR)
                return results, elapsed

        self.ADAPTIVE_CLIPPING = clipping_cfg
        model_save_path = save_path
        self.SAVE_PATH = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        self.clear_directory(model_save_path)

        start = time.perf_counter()
        q_model, _, _ = self.quantModel(
            self.NATIVE_MODEL_ID,
            batch_size=self.QUANT_BATCH_SIZE,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
        )
        elapsed = time.perf_counter() - start

        if self._loaded_model_was_prequantized:
            self.skipTest(
                "The dense model path unexpectedly resolved to a pre-quantized checkpoint; "
                "skipping because a fresh quantization is required for this comparison."
            )

        backend = self._current_load_backend()
        results = self._post_quant_eval_records.get(backend, {})

        if is_baseline:
            self._save_baseline_cache(results, elapsed)

        self._cleanup_quantized_model(q_model, enabled=True)
        return results, elapsed

    @staticmethod
    def _get_metric(task_results, key):
        if key in task_results:
            return float(task_results[key])
        for k, v in task_results.items():
            if k.startswith(key):
                return float(v)
        if key == "acc":
            for k, v in task_results.items():
                if k.startswith("accuracy,loglikelihood") and "norm" not in k:
                    return float(v)
        if key == "acc_norm":
            for k, v in task_results.items():
                if k.startswith("accuracy,loglikelihood") and "norm" in k:
                    return float(v)
        raise KeyError(f"Metric `{key}` not found in {task_results}")

    @staticmethod
    def _get_gsm8k(task_results):
        for metric in ("acc,num", "acc,none", "acc"):
            if metric in task_results:
                return float(task_results[metric])
        raise KeyError(f"No gsm8k accuracy metric in {task_results}")

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.cuda
    @pytest.mark.model
    def test_qwen3_8b_adaptive_clipping(self):
        """Adaptive clipping v1 must not regress Qwen3-8B vs. clipping disabled."""
        baseline_path = os.path.join(tempfile.gettempdir(), "qwen3_8b_adaptive_clipping_baseline")
        clipping_path = os.path.join(tempfile.gettempdir(), "qwen3_8b_adaptive_clipping_enabled")

        baseline_results, baseline_elapsed = self._run_config(self.BASELINE_ADAPTIVE_CLIPPING, baseline_path)
        clipping_results, clipping_elapsed = self._run_config(self.ENABLED_ADAPTIVE_CLIPPING, clipping_path)

        print("\nAdaptive Clipping v1 Qwen3-8B comparison:")
        print(f"  baseline: {baseline_results} (elapsed {baseline_elapsed:.1f}s)")
        print(f"  clipping: {clipping_results} (elapsed {clipping_elapsed:.1f}s)")
        print(f"  runtime delta: {((clipping_elapsed - baseline_elapsed) / max(baseline_elapsed, 1e-9) * 100.0):+.1f}%")

        for task_name in ("gsm8k_platinum_cot", "arc_challenge"):
            baseline_task = baseline_results.get(task_name, {})
            clipping_task = clipping_results.get(task_name, {})

            if task_name == "gsm8k_platinum_cot":
                metrics = {"acc,num": self._get_gsm8k}
            else:
                metrics = {
                    "acc": lambda r: self._get_metric(r, "acc"),
                    "acc_norm": lambda r: self._get_metric(r, "acc_norm"),
                }

            for metric_name, getter in metrics.items():
                try:
                    baseline_value = getter(baseline_task)
                    clipping_value = getter(clipping_task)
                except KeyError:
                    continue

                diff_pct = (clipping_value - baseline_value) / max(baseline_value, 1e-12) * 100.0
                print(
                    f"  {task_name}:{metric_name}: baseline={baseline_value:.4f}, "
                    f"clipping={clipping_value:.4f}, diff={diff_pct:+.2f}%"
                )
                self.assertGreaterEqual(
                    diff_pct,
                    -self.ALLOWED_REGRESSION_PCT,
                    f"Adaptive clipping regressed `{task_name}:{metric_name}` by "
                    f"{abs(diff_pct):.2f}% (baseline={baseline_value:.4f}, clipping={clipping_value:.4f}).",
                )
