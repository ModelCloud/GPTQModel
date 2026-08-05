# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""End-to-end adaptive-damping regression test for Llama-3.2-1B-Instruct.

Based on ``tests/models/test_llama3_2.py``. Quantizes the model twice with
adaptive damping enabled and disabled, then compares post-quant ARC-Challenge
scores to verify the feature does not regress quality.
"""

import os
import tempfile

import pytest
from model_test import ModelTest

from gptqmodel import BACKEND


# fmt: off
class TestLlama32AdaptiveDamping(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    SAVE_PATH = None
    DELETE_QUANTIZED_MODEL = True
    EVAL_BATCH_SIZE = "auto"
    DATASET_SIZE = 128
    DATASET_SIZE_FAST = 128
    DATASET_CONCAT_SIZE = 512
    DATASET_CONCAT_SIZE_FAST = 512
    LOAD_BACKEND = BACKEND.TORCH
    QUANT_BACKEND = BACKEND.AUTO
    APPLY_CHAT_TEMPLATE = True

    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": True,
            "evalution_suite_kwargs": {"max_rows": 256},
            "acc": {
                "value": 0.3242,
                "floor_pct": 0.02,
                "ceil_pct": 0.02,
            },
            "acc_norm": {
                "value": 0.3515,
                "floor_pct": 0.02,
                "ceil_pct": 0.02,
            },
        },
    }
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST

    ADAPTIVE_DAMPING = {
        "enabled": True,
        "base_percdamp": 0.05,
        "min": 0.02,
        "max": 0.08,
        "method": "power_iteration",
        "eigen_iterations": 10,
        "spectral_alpha": 0.25,
    }

    ALLOWED_REGRESSION_PCT = 2.0

    def _run_adaptive_config(self, adaptive_cfg, save_path):
        """Quantize and evaluate once with the given adaptive damping config."""
        self.ADAPTIVE_DAMPING = adaptive_cfg
        self.SAVE_PATH = save_path
        os.makedirs(save_path, exist_ok=True)
        self.clear_directory(save_path)

        q_model, _, _ = self.quantModel(
            self.NATIVE_MODEL_ID,
            batch_size=self.QUANT_BATCH_SIZE,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
        )

        if self._loaded_model_was_prequantized:
            self.skipTest(
                "The dense model path unexpectedly resolved to a pre-quantized checkpoint; "
                "skipping because a fresh quantization is required for this comparison."
            )

        backend = self._current_load_backend()
        results = self._post_quant_eval_records.get(backend, {})
        self._cleanup_quantized_model(q_model, enabled=True)
        return results

    @staticmethod
    def _get_metric(task_results, key):
        if key in task_results:
            return float(task_results[key])
        for k, v in task_results.items():
            if k.startswith(key):
                return float(v)
        # Evalution may report ARC metrics as `accuracy,loglikelihood` and
        # `accuracy,loglikelihood_norm` rather than `acc` / `acc_norm`.
        if key == "acc":
            for k, v in task_results.items():
                if k.startswith("accuracy,loglikelihood") and "norm" not in k:
                    return float(v)
        if key == "acc_norm":
            for k, v in task_results.items():
                if k.startswith("accuracy,loglikelihood") and "norm" in k:
                    return float(v)
        raise KeyError(f"Metric `{key}` not found in {task_results}")

    @pytest.mark.skip(reason="Llama-3.2-1B is too small for this comparison; use Qwen3-8B instead.")
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.cuda
    @pytest.mark.model
    def test_adaptive_damping_no_regression(self):
        """Adaptive damping must not regress ARC-Challenge vs. fixed damping."""
        disabled_path = os.path.join(tempfile.gettempdir(), "llama32_adaptive_disabled")
        enabled_path = os.path.join(tempfile.gettempdir(), "llama32_adaptive_enabled")

        disabled_results = self._run_adaptive_config(
            {"enabled": False},
            disabled_path,
        )
        enabled_results = self._run_adaptive_config(
            self.ADAPTIVE_DAMPING,
            enabled_path,
        )

        disabled_task = disabled_results.get("arc_challenge", {})
        enabled_task = enabled_results.get("arc_challenge", {})

        print("\nAdaptive damping ARC-Challenge comparison:")
        print(f"  disabled: {disabled_task}")
        print(f"  enabled:  {enabled_task}")

        for metric_name in ("acc", "acc_norm"):
            disabled_value = self._get_metric(disabled_task, metric_name)
            enabled_value = self._get_metric(enabled_task, metric_name)
            diff_pct = (enabled_value - disabled_value) / max(disabled_value, 1e-12) * 100.0
            print(
                f"  {metric_name}: disabled={disabled_value:.4f}, "
                f"enabled={enabled_value:.4f}, diff={diff_pct:+.2f}%"
            )
            self.assertGreaterEqual(
                diff_pct,
                -self.ALLOWED_REGRESSION_PCT,
                f"Adaptive damping regressed `{metric_name}` by {abs(diff_pct):.2f}% "
                f"(disabled={disabled_value:.4f}, enabled={enabled_value:.4f}).",
            )
