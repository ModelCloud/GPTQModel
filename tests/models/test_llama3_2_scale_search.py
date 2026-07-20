# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import math
import os
import tempfile

import test_llama3_2 as llama3_2

from gptqmodel import ScaleSearchConfig
from gptqmodel.utils.logger import render_table
from gptqmodel.utils.torch import torch_empty_cache


class TestLlama3_2ScaleSearchAB(llama3_2.TestLlama3_2):
    """Compare post-quant quality with only the scale-search objective changed."""

    QKV_SCALE_SEARCH_PATTERN = r"+:^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj)$"
    O_PROJ_SCALE_SEARCH_PATTERN = r"+:^model\.layers\.\d+\.self_attn\.o_proj$"
    QKVO_SCALE_SEARCH_PATTERN = r"+:^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
    MLP_SCALE_SEARCH_PATTERN = r"+:^model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)$"

    # Bound fixed-batch generation for the 131K-context Llama config. Batch 32
    # can consume roughly 75 GiB and stop making practical progress, while the
    # Transformers 5.14 paged manager currently stalls during initialization.
    GSM8K_FIXED_BATCH_SIZE = 4
    # The A/B is intentionally full-model even when the general model-test mode
    # defaults to fast; partially quantized scores cannot validate scale search.
    EVAL_TASKS_FAST = copy.deepcopy(llama3_2.TestLlama3_2.EVAL_TASKS_FAST)
    EVAL_TASKS_FAST["gsm8k_platinum_cot"]["evalution_model_args"]["attn_implementation"] = "flash_attention_2"
    EVAL_TASKS_FAST["gsm8k_platinum_cot"]["evalution_batch_size"] = GSM8K_FIXED_BATCH_SIZE
    EVAL_TASKS_FAST["gsm8k_platinum_cot"]["evalution_suite_kwargs"]["batch_size"] = GSM8K_FIXED_BATCH_SIZE
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST
    DELETE_QUANTIZED_MODEL = True
    USE_FLASH_ATTN = True
    test_llama3_2 = None

    def _quantize_and_score_arm(self, *, label, scale_search, save_path, dynamic=None, need_eval=True):
        """Quantize one isolated arm and return its post-quant Evalution metrics."""

        original_scale_search = self.SCALE_SEARCH
        original_dynamic = self.DYNAMIC
        original_save_path = self.SAVE_PATH
        self.SCALE_SEARCH = scale_search
        self.DYNAMIC = dynamic
        self.SAVE_PATH = save_path
        if hasattr(self, "_post_quant_eval_records"):
            self._post_quant_eval_records.clear()
        model = None
        try:
            model, _, _ = self.quantModel(
                self.NATIVE_MODEL_ID,
                batch_size=self.QUANT_BATCH_SIZE,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                need_eval=need_eval,
                call_perform_post_quant_validation=need_eval,
            )
            if not need_eval:
                return None
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
            self.SCALE_SEARCH = original_scale_search
            self.DYNAMIC = original_dynamic
            self.SAVE_PATH = original_save_path

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

    def test_gsm8k_evaluation_uses_bounded_fixed_batches(self):
        """Prevent the A/B from restoring either known-stalling generation path."""

        task = self.EVAL_TASKS_FAST["gsm8k_platinum_cot"]
        self.assertEqual(task["evalution_model_args"]["attn_implementation"], "flash_attention_2")
        self.assertEqual(task["evalution_batch_size"], self.GSM8K_FIXED_BATCH_SIZE)
        self.assertEqual(task["evalution_suite_kwargs"]["batch_size"], self.GSM8K_FIXED_BATCH_SIZE)

    def test_scale_search_objective_post_quant_ab(self):
        """Compare disabled, diagonal, full-Hessian, and shrinkage objectives."""

        arms = {
            "Disabled": None,
            "Activation": ScaleSearchConfig.ACTIVATION,
            "Hessian": ScaleSearchConfig.HESSIAN,
            "Hybrid": ScaleSearchConfig.HYBRID,
        }
        results_by_arm = {}
        with tempfile.TemporaryDirectory(prefix="llama3_2_scale_search_ab_") as root:
            for label, scale_search in arms.items():
                results_by_arm[label] = self._quantize_and_score_arm(
                    label=label.lower(),
                    scale_search=scale_search,
                    save_path=os.path.join(root, label.lower()),
                )

        metrics_by_arm = {
            label: self._quality_metrics(results)
            for label, results in results_by_arm.items()
        }
        disabled = metrics_by_arm["Disabled"]
        rows = []
        for metric in disabled:
            scores = [metrics_by_arm[label][metric] for label in arms]
            rows.append(
                [
                    metric,
                    *(f"{score:.6f}" for score in scores),
                    *(f"{score - scores[0]:+.6f}" for score in scores[1:]),
                ]
            )
        means = {
            label: sum(metrics.values()) / len(metrics)
            for label, metrics in metrics_by_arm.items()
        }
        rows.append(
            [
                "mean",
                *(f"{means[label]:.6f}" for label in arms),
                *(f"{means[label] - means['Disabled']:+.6f}" for label in list(arms)[1:]),
            ]
        )
        print(
            "\nScale-search post-quant A/B:\n"
            + render_table(
                rows,
                headers=[
                    "Metric",
                    *arms,
                    "Activation Δ",
                    "Hessian Δ",
                    "Hybrid Δ",
                ],
                tablefmt="grid",
            )
        )

        for label, metrics in metrics_by_arm.items():
            for metric, score in metrics.items():
                self.assertTrue(math.isfinite(score), f"{label} produced a non-finite {metric} score.")

        activation = metrics_by_arm["Activation"]
        for metric in disabled:
            self.assertGreaterEqual(
                activation[metric],
                disabled[metric],
                f"Activation scale search regressed {metric}.",
            )
        self.assertGreater(
            means["Activation"],
            means["Disabled"],
            "Activation scale search did not increase mean ARC/GSM8K score.",
        )
        self.assertGreaterEqual(
            means["Hybrid"],
            means["Disabled"],
            "Hybrid scale search regressed mean ARC/GSM8K quality versus disabled search.",
        )

    def test_scale_search_objective_by_projection_scope_post_quant_ab(self):
        """Compare scale-search objectives independently on QKV, O, and MLP projections."""

        methods = {
            "Activation": ScaleSearchConfig.ACTIVATION,
            "Hessian": ScaleSearchConfig.HESSIAN,
            "Hybrid": ScaleSearchConfig.HYBRID,
        }
        scopes = {
            "QKV only": self.QKV_SCALE_SEARCH_PATTERN,
            "O only": self.O_PROJ_SCALE_SEARCH_PATTERN,
            "MLP only": self.MLP_SCALE_SEARCH_PATTERN,
        }
        single_arm = os.environ.get("GPTQMODEL_SCALE_SEARCH_SCOPE_ARM")
        if single_arm:
            quant_only = os.environ.get("GPTQMODEL_SCALE_SEARCH_SCOPE_QUANT_ONLY") == "1"
            arm_specs = {
                "disabled": (None, None),
                **{
                    f"{scope_label.lower()}_{method_label.lower()}": (
                        None,
                        {pattern: {"scale_search": scale_search.value}},
                    )
                    for scope_label, pattern in {
                        "qkv": scopes["QKV only"],
                        "o": scopes["O only"],
                        "mlp": scopes["MLP only"],
                    }.items()
                    for method_label, scale_search in methods.items()
                },
                # Full-coverage policy suggested by the independently measured
                # QKVO activation and MLP Hessian projection preferences.
                "qkvo_activation_else_hessian": (
                    ScaleSearchConfig.HESSIAN,
                    {self.QKVO_SCALE_SEARCH_PATTERN: {"scale_search": ScaleSearchConfig.ACTIVATION.value}},
                ),
                # Full-model controls complete the Qwen policy sweep without
                # changing projection-specific behavior in the scoped arms.
                "all_activation": (ScaleSearchConfig.ACTIVATION, None),
                "all_hessian": (ScaleSearchConfig.HESSIAN, None),
                "all_hybrid": (ScaleSearchConfig.HYBRID, None),
            }
            if single_arm not in arm_specs:
                raise ValueError(
                    f"Unknown GPTQMODEL_SCALE_SEARCH_SCOPE_ARM={single_arm!r}; expected one of {list(arm_specs)}."
                )
            scale_search, dynamic = arm_specs[single_arm]
            with tempfile.TemporaryDirectory(prefix=f"llama3_2_scale_search_{single_arm}_") as root:
                results = self._quantize_and_score_arm(
                    label=single_arm,
                    scale_search=scale_search,
                    dynamic=dynamic,
                    save_path=os.path.join(root, single_arm),
                    need_eval=not quant_only,
                )
            if quant_only:
                print(f"\nSCALE_SEARCH_SCOPE_QUANT_RESULT {single_arm} completed")
                return
            metrics = self._quality_metrics(results)
            print(f"\nSCALE_SEARCH_SCOPE_RESULT {single_arm} {json.dumps(metrics, sort_keys=True)}")
            for metric, score in metrics.items():
                self.assertTrue(math.isfinite(score), f"{single_arm} produced a non-finite {metric} score.")
            return

        with tempfile.TemporaryDirectory(prefix="llama3_2_scale_search_scope_ab_") as root:
            disabled_results = self._quantize_and_score_arm(
                label="disabled",
                scale_search=None,
                save_path=os.path.join(root, "disabled"),
            )
            results_by_scope = {}
            for scope_label, pattern in scopes.items():
                scope_results = {"Disabled": disabled_results}
                for method_label, scale_search in methods.items():
                    arm_label = f"{scope_label.lower().replace(' ', '_')}_{method_label.lower()}"
                    scope_results[method_label] = self._quantize_and_score_arm(
                        label=arm_label,
                        scale_search=None,
                        dynamic={pattern: {"scale_search": scale_search.value}},
                        save_path=os.path.join(root, arm_label),
                    )
                results_by_scope[scope_label] = scope_results

        metrics_by_scope = {
            scope: {
                label: self._quality_metrics(results)
                for label, results in scope_results.items()
            }
            for scope, scope_results in results_by_scope.items()
        }
        arm_labels = ["Disabled", *methods]
        for scope, metrics_by_arm in metrics_by_scope.items():
            disabled = metrics_by_arm["Disabled"]
            rows = []
            for metric in disabled:
                scores = [metrics_by_arm[label][metric] for label in arm_labels]
                rows.append(
                    [
                        metric,
                        *(f"{score:.6f}" for score in scores),
                        *(f"{score - scores[0]:+.6f}" for score in scores[1:]),
                    ]
                )
            means = {
                label: sum(metrics.values()) / len(metrics)
                for label, metrics in metrics_by_arm.items()
            }
            rows.append(
                [
                    "mean",
                    *(f"{means[label]:.6f}" for label in arm_labels),
                    *(f"{means[label] - means['Disabled']:+.6f}" for label in methods),
                ]
            )
            print(
                f"\nScale-search post-quant A/B ({scope}):\n"
                + render_table(
                    rows,
                    headers=[
                        "Metric",
                        *arm_labels,
                        "Activation Δ",
                        "Hessian Δ",
                        "Hybrid Δ",
                    ],
                    tablefmt="grid",
                )
            )

        for scope, metrics_by_arm in metrics_by_scope.items():
            for label, metrics in metrics_by_arm.items():
                for metric, score in metrics.items():
                    self.assertTrue(
                        math.isfinite(score),
                        f"{scope} {label} produced a non-finite {metric} score.",
                    )
