# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import tempfile

import test_llama3_2 as llama3_2

from gptqmodel import ScaleSearchConfig
from gptqmodel.utils.logger import render_table
from gptqmodel.utils.torch import torch_empty_cache


class TestLlama3_2ScaleSearchAB(llama3_2.TestLlama3_2):
    """Compare post-quant quality with only the scale-search objective changed."""

    # The A/B is intentionally full-model even when the general model-test mode
    # defaults to fast; partially quantized scores cannot validate scale search.
    EVAL_TASKS_FAST = copy.deepcopy(llama3_2.TestLlama3_2.EVAL_TASKS_FAST)
    EVAL_TASKS_FAST["gsm8k_platinum_cot"]["evalution_model_args"]["attn_implementation"] = "flash_attention_2"
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST
    DELETE_QUANTIZED_MODEL = True
    USE_FLASH_ATTN = True
    test_llama3_2 = None

    def _quantize_and_score_arm(self, *, label, scale_search, save_path):
        """Quantize one isolated arm and return its post-quant Evalution metrics."""

        self.SCALE_SEARCH = scale_search
        self.SAVE_PATH = save_path
        model = None
        try:
            model, _, _ = self.quantModel(
                self.NATIVE_MODEL_ID,
                batch_size=self.QUANT_BATCH_SIZE,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
            )
            backend = self._current_load_backend()
            results = copy.deepcopy(self._post_quant_eval_records.get(backend))
            if not results:
                raise AssertionError(f"No post-quant results were captured for A/B arm `{label}`.")
            return results
        finally:
            if model is not None:
                self._cleanup_quantized_model(model, enabled=True)
                del model
            torch_empty_cache()

    @staticmethod
    def _quality_metrics(results):
        """Extract the stable ARC and GSM8K Platinum metrics from the base test."""

        arc_metrics = results.get("arc_challenge", {})
        gsm8k_metrics = results.get("gsm8k_platinum_cot", {})
        return {
            "arc_challenge :: acc": float(arc_metrics["accuracy,loglikelihood"]),
            "arc_challenge :: acc_norm": float(arc_metrics["accuracy,loglikelihood_norm"]),
            "gsm8k_platinum_cot :: acc,num": float(gsm8k_metrics["acc,num"]),
        }

    def test_activation_scale_search_post_quant_ab(self):
        """Require activation-aware search to improve mean ARC/GSM8K quality without regressions."""

        with tempfile.TemporaryDirectory(prefix="llama3_2_scale_search_ab_") as root:
            default_results = self._quantize_and_score_arm(
                label="default",
                scale_search=None,
                save_path=os.path.join(root, "default"),
            )
            activation_results = self._quantize_and_score_arm(
                label="activation",
                scale_search=ScaleSearchConfig.ACTIVATION,
                save_path=os.path.join(root, "activation"),
            )

        default = self._quality_metrics(default_results)
        activation = self._quality_metrics(activation_results)
        rows = []
        for metric in default:
            rows.append(
                [
                    metric,
                    f"{default[metric]:.6f}",
                    f"{activation[metric]:.6f}",
                    f"{activation[metric] - default[metric]:+.6f}",
                ]
            )
        default_mean = sum(default.values()) / len(default)
        activation_mean = sum(activation.values()) / len(activation)
        rows.append(["mean", f"{default_mean:.6f}", f"{activation_mean:.6f}", f"{activation_mean - default_mean:+.6f}"])
        print(
            "\nScale-search post-quant A/B:\n"
            + render_table(rows, headers=["Metric", "Default", "Activation", "Delta"], tablefmt="grid")
        )

        for metric in default:
            self.assertGreaterEqual(
                activation[metric],
                default[metric],
                f"Activation scale search regressed {metric}.",
            )
        self.assertGreater(
            activation_mean,
            default_mean,
            "Activation scale search did not increase mean ARC/GSM8K score.",
        )
