# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""End-to-end adaptive-damping plus calibration-aware clipping comparison for Qwen3-8B."""

import copy
import hashlib
import json
import logging
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

import pytest
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig


log = logging.getLogger(__name__)


class TestQwen3_8BAdaptiveClipping(ModelTest):
    """Measure exact GPTQ-error clipping on top of adaptive damping v4.2."""

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
    # isolates which exclusive range selector is better: the automatic
    # activation scale search or exact GPTQ-error adaptive clipping.
    V42_ADAPTIVE_DAMPING = {
        "enabled": True,
        "base_percdamp": 0.05,
        "min": 0.02,
        "max": 0.08,
        "method": "power_iteration",
        "eigen_iterations": 10,
        "spectral_alpha": 0.25,
        "module_prior_enabled": True,
        "group_error_enabled": True,
        "online_feedback_enabled": True,
        "group_size_prior_enabled": True,
        "group_error_use_hessian_weighting": True,
        "group_error_measure_raw_residual": True,
    }

    ADAPTIVE_DAMPING = V42_ADAPTIVE_DAMPING

    BASELINE_ADAPTIVE_CLIPPING = None
    BASELINE_SCALE_SEARCH = ScaleSearchConfig.ACTIVATION
    ENABLED_ADAPTIVE_CLIPPING = {
        "enabled": True,
        "metric": "gptq_error",
        "per_group": True,
    }
    ENABLED_SCALE_SEARCH = None
    # ModelTest explicitly forwards this value into QuantizeConfig. Keep the
    # clipping-disabled control aligned with the product's automatic default.
    SCALE_SEARCH = BASELINE_SCALE_SEARCH

    ALLOWED_REGRESSION_PCT = 0.5
    MAX_ADAPTIVE_RANGE_REGRESSION_PCT = 2.0
    EXPECTED_METRIC_COUNT = 3
    BASELINE_CACHE_SCHEMA = 2

    # Cache the no-clipping baseline so repeated iterations only quantize once.
    BASELINE_CACHE_DIR = os.environ.get(
        "GPTQMODEL_QWEN3_8B_ADAPTIVE_CLIPPING_BASELINE_CACHE",
        "/tmp/qwen3_8b_adaptive_clipping_eval_cache",
    )
    BASELINE_RESULTS_JSON = os.path.join(BASELINE_CACHE_DIR, "eval_records.json")

    @classmethod
    def _baseline_config_hash(cls):
        payload = {
            "schema": cls.BASELINE_CACHE_SCHEMA,
            "model": cls._model_fingerprint(),
            "quantization": {
                key: str(getattr(cls, key, None))
                for key in (
                    "FORMAT",
                    "METHOD",
                    "BITS",
                    "GROUP_SIZE",
                    "DESC_ACT",
                    "SYM",
                    "ACT_GROUP_AWARE",
                    "MSE",
                    "SCALE_SEARCH",
                    "QUANT_BATCH_SIZE",
                    "TORCH_DTYPE",
                    "QUANT_BACKEND",
                    "LOAD_BACKEND",
                )
            },
            "adaptive_damping": cls.V42_ADAPTIVE_DAMPING,
            "adaptive_clipping": cls.BASELINE_ADAPTIVE_CLIPPING,
            "dataset_size": cls.DATASET_SIZE,
            "dataset_concat_size": cls.DATASET_CONCAT_SIZE,
            "eval_tasks": cls.EVAL_TASKS_SLOW,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _model_fingerprint(cls):
        model_path = Path(cls.NATIVE_MODEL_ID)
        if not model_path.is_dir():
            return {"model_id": cls.NATIVE_MODEL_ID}
        files = []
        for pattern in ("config.json", "*.index.json", "*.safetensors"):
            for path in sorted(model_path.glob(pattern)):
                stat = path.stat()
                files.append((path.name, stat.st_size, stat.st_mtime_ns))
        return {"model_path": str(model_path.resolve()), "files": files}

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
        results = cached.get("results")
        cls._validated_metrics(results, source="cached clipping baseline")
        return results, float(cached.get("elapsed", 0.0))

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

    def _run_config(
        self,
        clipping_cfg,
        scale_search,
        save_path,
        *,
        adaptive_damping=V42_ADAPTIVE_DAMPING,
        use_baseline_cache=True,
    ):
        is_baseline = (
            use_baseline_cache
            and self._is_baseline_config(clipping_cfg)
            and scale_search == self.BASELINE_SCALE_SEARCH
            and adaptive_damping == self.V42_ADAPTIVE_DAMPING
        )
        if is_baseline:
            cached = self._load_baseline_cache()
            if cached is not None:
                results, elapsed = cached
                log.info("Reusing cached adaptive-clipping baseline from %s", self.BASELINE_CACHE_DIR)
                return results, elapsed

        self.ADAPTIVE_CLIPPING = clipping_cfg
        self.ADAPTIVE_DAMPING = adaptive_damping
        self.SCALE_SEARCH = scale_search
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

        try:
            if self._loaded_model_was_prequantized:
                self.skipTest(
                    "The dense model path unexpectedly resolved to a pre-quantized checkpoint; "
                    "skipping because a fresh quantization is required for this comparison."
                )

            backend = self._current_load_backend()
            results = self._post_quant_eval_records.get(backend, {})
            self._validated_metrics(
                results,
                source="fresh clipping baseline" if is_baseline else "fresh clipping candidate",
            )
            if is_baseline:
                self._save_baseline_cache(results, elapsed)
            return results, elapsed
        finally:
            self._cleanup_quantized_model(q_model, enabled=True)

    @staticmethod
    def _get_metric(task_results, key):
        if key in task_results:
            return float(task_results[key])
        for k, v in task_results.items():
            if k.startswith(f"{key},"):
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

    @classmethod
    def _validated_metrics(cls, results, *, source):
        if not isinstance(results, dict):
            raise AssertionError(f"{source} did not return a task-result mapping: {results!r}")
        task_metrics = {
            "gsm8k_platinum_cot": {"acc,num": cls._get_gsm8k},
            "arc_challenge": {
                "acc": lambda r: cls._get_metric(r, "acc"),
                "acc_norm": lambda r: cls._get_metric(r, "acc_norm"),
            },
        }
        validated = {}
        for task_name, metrics in task_metrics.items():
            if task_name not in results or not isinstance(results[task_name], dict):
                raise AssertionError(f"{source} is missing required task `{task_name}`: {results!r}")
            for metric_name, getter in metrics.items():
                try:
                    value = getter(results[task_name])
                except (KeyError, TypeError, ValueError) as exc:
                    raise AssertionError(
                        f"{source} is missing required metric `{task_name}:{metric_name}`"
                    ) from exc
                if not math.isfinite(value):
                    raise AssertionError(f"{source} returned non-finite `{task_name}:{metric_name}`={value}")
                validated[(task_name, metric_name)] = value
        if len(validated) != cls.EXPECTED_METRIC_COUNT:
            raise AssertionError(f"{source} validated {len(validated)} metrics, expected {cls.EXPECTED_METRIC_COUNT}")
        return validated

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.cuda
    @pytest.mark.model
    def test_qwen3_8b_adaptive_damping_plus_adaptive_clipping(self):
        """Activation range search must remain the best aggregate range default with adaptive damping."""
        baseline_path = os.path.join(tempfile.gettempdir(), "qwen3_8b_adaptive_clipping_baseline")
        clipping_path = os.path.join(tempfile.gettempdir(), "qwen3_8b_adaptive_clipping_enabled")

        baseline_results, baseline_elapsed = self._run_config(
            self.BASELINE_ADAPTIVE_CLIPPING,
            self.BASELINE_SCALE_SEARCH,
            baseline_path,
        )
        clipping_results, clipping_elapsed = self._run_config(
            self.ENABLED_ADAPTIVE_CLIPPING,
            self.ENABLED_SCALE_SEARCH,
            clipping_path,
        )
        baseline_metrics = self._validated_metrics(
            baseline_results,
            source="adaptive-damping-v4.2 plus activation-scale-search control",
        )
        clipping_metrics = self._validated_metrics(
            clipping_results,
            source="adaptive-damping-v4.2 plus adaptive-clipping candidate",
        )

        print("\nAdaptive Damping v4.2 + Adaptive Clipping Qwen3-8B comparison:")
        print(f"  damping + activation scale search: {baseline_results} (elapsed {baseline_elapsed:.1f}s)")
        print(f"  damping + clipping enabled: {clipping_results} (elapsed {clipping_elapsed:.1f}s)")
        print(f"  runtime delta: {((clipping_elapsed - baseline_elapsed) / max(baseline_elapsed, 1e-9) * 100.0):+.1f}%")

        activation_advantages = []
        for task_metric, baseline_value in baseline_metrics.items():
            task_name, metric_name = task_metric
            clipping_value = clipping_metrics[task_metric]
            clipping_diff_pct = (clipping_value - baseline_value) / max(abs(baseline_value), 1e-12) * 100.0
            activation_advantage_pct = (
                (baseline_value - clipping_value) / max(abs(clipping_value), 1e-12) * 100.0
            )
            activation_advantages.append(activation_advantage_pct)
            print(
                f"  {task_name}:{metric_name}: baseline={baseline_value:.4f}, "
                f"clipping={clipping_value:.4f}, clipping_diff={clipping_diff_pct:+.2f}%"
            )
            self.assertGreaterEqual(
                activation_advantage_pct,
                -self.MAX_ADAPTIVE_RANGE_REGRESSION_PCT,
                f"Activation scale search regressed `{task_name}:{metric_name}` by "
                f"{abs(activation_advantage_pct):.2f}% against adaptive clipping.",
            )
        self.assertGreaterEqual(
            statistics.fmean(activation_advantages),
            0.0,
            "Activation scale search lost the aggregate adaptive-damping task comparison; reconsider the default.",
        )

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.cuda
    @pytest.mark.model
    def test_qwen3_8b_static_damping_adaptive_clipping_vs_activation_scale_search(self):
        """The fixed-damping activation default must not regress against GPTQ-error clipping."""
        activation_path = os.path.join(tempfile.gettempdir(), "qwen3_8b_static_damping_activation_search")
        clipping_path = os.path.join(tempfile.gettempdir(), "qwen3_8b_static_damping_adaptive_clipping")

        activation_results, activation_elapsed = self._run_config(
            self.BASELINE_ADAPTIVE_CLIPPING,
            self.BASELINE_SCALE_SEARCH,
            activation_path,
            adaptive_damping=None,
            use_baseline_cache=False,
        )
        clipping_results, clipping_elapsed = self._run_config(
            self.ENABLED_ADAPTIVE_CLIPPING,
            self.ENABLED_SCALE_SEARCH,
            clipping_path,
            adaptive_damping=None,
            use_baseline_cache=False,
        )
        activation_metrics = self._validated_metrics(
            activation_results,
            source="static-damping plus activation-scale-search control",
        )
        clipping_metrics = self._validated_metrics(
            clipping_results,
            source="static-damping plus adaptive-clipping candidate",
        )

        print("\nStatic Damping + Adaptive Clipping Qwen3-8B comparison:")
        print(f"  static damping + activation scale search: {activation_results} (elapsed {activation_elapsed:.1f}s)")
        print(f"  static damping + clipping enabled: {clipping_results} (elapsed {clipping_elapsed:.1f}s)")
        print(f"  runtime delta: {((clipping_elapsed - activation_elapsed) / max(activation_elapsed, 1e-9) * 100.0):+.1f}%")

        for task_metric, activation_value in activation_metrics.items():
            task_name, metric_name = task_metric
            clipping_value = clipping_metrics[task_metric]
            clipping_diff_pct = (clipping_value - activation_value) / max(abs(activation_value), 1e-12) * 100.0
            default_advantage_pct = (
                (activation_value - clipping_value) / max(abs(clipping_value), 1e-12) * 100.0
            )
            print(
                f"  {task_name}:{metric_name}: activation={activation_value:.4f}, "
                f"clipping={clipping_value:.4f}, clipping_diff={clipping_diff_pct:+.2f}%"
            )
            self.assertGreaterEqual(
                default_advantage_pct,
                -self.ALLOWED_REGRESSION_PCT,
                f"The fixed-damping activation default regressed `{task_name}:{metric_name}` by "
                f"{abs(default_advantage_pct):.2f}% against GPTQ-error adaptive clipping.",
            )


def test_adaptive_damping_plus_clipping_model_gate_configuration():
    damping = TestQwen3_8BAdaptiveClipping.ADAPTIVE_DAMPING
    clipping = TestQwen3_8BAdaptiveClipping.ENABLED_ADAPTIVE_CLIPPING
    activation_control = QuantizeConfig(
        bits=4,
        group_size=128,
        adaptive_damping=damping,
        adaptive_clipping=None,
    )
    combined_candidate = QuantizeConfig(
        bits=4,
        group_size=128,
        adaptive_damping=damping,
        adaptive_clipping=clipping,
        scale_search=None,
    )

    assert damping["enabled"] is True
    assert damping["online_feedback_enabled"] is True
    assert TestQwen3_8BAdaptiveClipping.BASELINE_SCALE_SEARCH == ScaleSearchConfig.ACTIVATION
    assert TestQwen3_8BAdaptiveClipping.ENABLED_SCALE_SEARCH is None
    assert activation_control.scale_search == ScaleSearchConfig.ACTIVATION
    assert combined_candidate.scale_search is None
    assert clipping == {"enabled": True, "metric": "gptq_error", "per_group": True}


def test_adaptive_clipping_model_gate_rejects_missing_metrics():
    incomplete = {
        "gsm8k_platinum_cot": {"acc,num": 0.9},
        "arc_challenge": {"acc": 0.5},
    }

    with pytest.raises(AssertionError, match="arc_challenge:acc_norm"):
        TestQwen3_8BAdaptiveClipping._validated_metrics(incomplete, source="synthetic result")


def test_adaptive_clipping_model_gate_keeps_acc_distinct_from_acc_norm():
    results = {
        "gsm8k_platinum_cot": {"acc,num": 0.9},
        "arc_challenge": {"acc_norm,none": 0.6, "acc,none": 0.5},
    }

    metrics = TestQwen3_8BAdaptiveClipping._validated_metrics(results, source="synthetic result")

    assert metrics[("arc_challenge", "acc")] == 0.5
    assert metrics[("arc_challenge", "acc_norm")] == 0.6
