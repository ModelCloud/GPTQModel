# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""End-to-end v4.2 online group-feedback adaptive-damping comparison for Qwen3-8B."""

import copy
import hashlib
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path

import pytest
from model_test import ModelTest

from gptqmodel import BACKEND


log = logging.getLogger(__name__)


class TestQwen3_8BAdaptiveDampingV42(ModelTest):
    """Compare adaptive damping v3 vs. v4.2 on Qwen3-8B (base model, no chat template)."""

    NATIVE_MODEL_ID = os.environ.get(
        "QWEN3_8B_MODEL_PATH",
        "/monster/data/model/Qwen3-8B",
    )
    SAVE_PATH = None
    DELETE_QUANTIZED_MODEL = True
    EVAL_BATCH_SIZE = "auto"
    DATASET_SIZE = 128
    DATASET_SIZE_FAST = 128
    DATASET_CONCAT_SIZE = 512
    DATASET_CONCAT_SIZE_FAST = 512
    LOAD_BACKEND = BACKEND.AUTO
    QUANT_BACKEND = BACKEND.AUTO
    APPLY_CHAT_TEMPLATE = False

    # Full slow eval: no row cap.
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": False,
            "acc,num": {"value": 0.92, "floor_pct": 0.05, "ceil_pct": 0.20},
        },
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.54, "floor_pct": 0.05, "ceil_pct": 0.20},
            "acc_norm": {"value": 0.55, "floor_pct": 0.05, "ceil_pct": 0.20},
            # Use the cached local dataset instead of streaming from the Hub.
            "evalution_suite_kwargs": {"stream": False},
            # Cap the loglikelihood batch size to avoid OOM/auto retries that re-run
            # the whole task list on GPU 7.
            "evalution_batch_size": 64,
        },
    }

    # Fast eval uses the full benchmark rows so results are comparable to slow mode.
    EVAL_TASKS_FAST = copy.deepcopy(EVAL_TASKS_SLOW)

    V3_ADAPTIVE_DAMPING = {
        "enabled": True,
        "base_percdamp": 0.05,
        "min": 0.02,
        "max": 0.08,
        "method": "power_iteration",
        "eigen_iterations": 10,
        "spectral_alpha": 0.25,
        "module_prior_enabled": True,
    }

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

    ALLOWED_REGRESSION_PCT = 2.0
    EXPECTED_METRIC_COUNT = 3
    V3_CACHE_SCHEMA = 2

    # Persistent cache for the v3 baseline so comparison tests do not re-quantize
    # and re-evaluate the same baseline across iterations.
    V3_CACHE_DIR = os.environ.get(
        "GPTQMODEL_QWEN3_8B_ADAPTIVE_DAMPING_V3_CACHE",
        "/monster/data/model/Qwen3-8B-adaptive-damping-v3-baseline",
    )
    V3_RESULTS_JSON = os.path.join(V3_CACHE_DIR, "eval_records.json")

    @classmethod
    def _v3_config_hash(cls):
        """Stable hash of the v3 adaptive damping config to invalidate stale caches."""
        payload = {
            "schema": cls.V3_CACHE_SCHEMA,
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
            "adaptive_damping": cls.V3_ADAPTIVE_DAMPING,
            "dataset_size": cls.DATASET_SIZE,
            "dataset_concat_size": cls.DATASET_CONCAT_SIZE,
            "eval_tasks": cls.EVAL_TASKS_SLOW,
        }
        payload = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

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
    def _load_v3_baseline(cls):
        """Return cached (results, elapsed) for v3, or None if missing/stale."""
        if not os.path.exists(cls.V3_RESULTS_JSON):
            return None
        try:
            with open(cls.V3_RESULTS_JSON, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if cached.get("config_hash") != cls._v3_config_hash():
            return None
        results = cached.get("results")
        cls._validated_metrics(results, source="cached adaptive-damping v3 baseline")
        return results, float(cached.get("elapsed", 0.0))

    @classmethod
    def _save_v3_baseline(cls, results, elapsed):
        """Persist v3 baseline results and elapsed time."""
        os.makedirs(cls.V3_CACHE_DIR, exist_ok=True)
        payload = {
            "config_hash": cls._v3_config_hash(),
            "elapsed": elapsed,
            "results": results,
        }
        with open(cls.V3_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _is_v3_config(self, adaptive_cfg):
        return adaptive_cfg is self.V3_ADAPTIVE_DAMPING or adaptive_cfg == self.V3_ADAPTIVE_DAMPING

    def _run_config(self, adaptive_cfg, save_path):
        """Quantize and evaluate once with the given adaptive damping config.

        The v3 baseline is persisted under ``V3_CACHE_DIR`` (only the lightweight
        ``eval_records.json``) and reused on subsequent runs unless
        ``ADAPTIVE_DAMPING_V3_FORCE_RECREATE=1`` is set. The quantized model itself
        is kept in a local temp directory so the eval pipeline does not pay the
        cost of round-tripping multi-gigabyte safetensors through the shared model
        storage.
        """
        is_v3 = self._is_v3_config(adaptive_cfg)
        force_recreate = os.environ.get("ADAPTIVE_DAMPING_V3_FORCE_RECREATE", "0") == "1"
        if is_v3 and not force_recreate:
            cached = self._load_v3_baseline()
            if cached is not None:
                results, elapsed = cached
                log.info(f"Reusing cached v3 baseline from {self.V3_CACHE_DIR}")
                return results, elapsed

        self.ADAPTIVE_DAMPING = adaptive_cfg
        model_save_path = tempfile.mkdtemp(prefix="qwen3_8b_adaptive_v3_") if is_v3 else save_path
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
                source="fresh adaptive-damping v3 baseline" if is_v3 else "fresh adaptive-damping v4.2 candidate",
            )
            if is_v3:
                self._save_v3_baseline(results, elapsed)
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
        """gsm8k_platinum_cot reports accuracy under `acc,num` or `acc,none`."""
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
    def test_adaptive_damping_v4_2_online_feedback_qwen3_8b(self):
        """v4.2 online group feedback must not regress Qwen3-8B vs. v3."""
        v3_path = self.V3_CACHE_DIR
        v42_path = os.path.join(tempfile.gettempdir(), "qwen3_8b_adaptive_v42")

        v3_results, v3_elapsed = self._run_config(self.V3_ADAPTIVE_DAMPING, v3_path)
        v42_results, v42_elapsed = self._run_config(self.V42_ADAPTIVE_DAMPING, v42_path)
        v3_metrics = self._validated_metrics(v3_results, source="adaptive-damping v3 baseline")
        v42_metrics = self._validated_metrics(v42_results, source="adaptive-damping v4.2 candidate")

        print("\nAdaptive damping v4.2 Qwen3-8B comparison:")
        print(f"  v3:   {v3_results} (elapsed {v3_elapsed:.1f}s)")
        print(f"  v4.2: {v42_results} (elapsed {v42_elapsed:.1f}s)")
        print(f"  runtime delta: {((v42_elapsed - v3_elapsed) / v3_elapsed * 100.0):+.1f}%")

        for task_metric, v3_value in v3_metrics.items():
            task_name, metric_name = task_metric
            v42_value = v42_metrics[task_metric]
            diff_pct = (v42_value - v3_value) / max(abs(v3_value), 1e-12) * 100.0
            print(
                f"  {task_name}:{metric_name}: v3={v3_value:.4f}, "
                f"v4.2={v42_value:.4f}, diff={diff_pct:+.2f}%"
            )
            self.assertGreaterEqual(
                diff_pct,
                -self.ALLOWED_REGRESSION_PCT,
                f"v4.2 online feedback regressed `{task_name}:{metric_name}` by "
                f"{abs(diff_pct):.2f}% (v3={v3_value:.4f}, v4.2={v42_value:.4f}).",
            )


def test_adaptive_damping_model_gate_rejects_empty_results():
    with pytest.raises(AssertionError, match="missing required task"):
        TestQwen3_8BAdaptiveDampingV42._validated_metrics({}, source="synthetic result")
