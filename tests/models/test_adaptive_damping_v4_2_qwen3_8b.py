# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""End-to-end v4.2 online group-feedback adaptive-damping comparison for Qwen3-8B."""

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
    }

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

    ALLOWED_REGRESSION_PCT = 2.0

    # Persistent cache for the v3 baseline so comparison tests do not re-quantize
    # and re-evaluate the same baseline across iterations.
    V3_CACHE_DIR = "/monster/data/model/Qwen3-8B-adaptive-damping-v3-baseline"
    V3_RESULTS_JSON = os.path.join(V3_CACHE_DIR, "eval_records.json")

    @classmethod
    def _v3_config_hash(cls):
        """Stable hash of the v3 adaptive damping config to invalidate stale caches."""
        payload = json.dumps(cls.V3_ADAPTIVE_DAMPING, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

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
        return cached.get("results"), cached.get("elapsed", 0.0)

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

        if self._loaded_model_was_prequantized:
            self.skipTest(
                "The dense model path unexpectedly resolved to a pre-quantized checkpoint; "
                "skipping because a fresh quantization is required for this comparison."
            )

        backend = self._current_load_backend()
        results = self._post_quant_eval_records.get(backend, {})

        if is_v3:
            self._save_v3_baseline(results, elapsed)

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
        """gsm8k_platinum_cot reports accuracy under `acc,num` or `acc,none`."""
        for metric in ("acc,num", "acc,none", "acc"):
            if metric in task_results:
                return float(task_results[metric])
        raise KeyError(f"No gsm8k accuracy metric in {task_results}")

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

        print("\nAdaptive damping v4.2 Qwen3-8B comparison:")
        print(f"  v3:   {v3_results} (elapsed {v3_elapsed:.1f}s)")
        print(f"  v4.2: {v42_results} (elapsed {v42_elapsed:.1f}s)")
        print(f"  runtime delta: {((v42_elapsed - v3_elapsed) / v3_elapsed * 100.0):+.1f}%")

        for task_name in ("gsm8k_platinum_cot", "arc_challenge"):
            v3_task = v3_results.get(task_name, {})
            v42_task = v42_results.get(task_name, {})

            if task_name == "gsm8k_platinum_cot":
                metrics = {"acc,num": self._get_gsm8k}
            else:
                metrics = {"acc": lambda r: self._get_metric(r, "acc"), "acc_norm": lambda r: self._get_metric(r, "acc_norm")}

            for metric_name, getter in metrics.items():
                try:
                    v3_value = getter(v3_task)
                    v42_value = getter(v42_task)
                except KeyError:
                    continue
                diff_pct = (v42_value - v3_value) / max(v3_value, 1e-12) * 100.0
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
