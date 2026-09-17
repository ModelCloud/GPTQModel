# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math

from model_test import ModelTest


class TestZDTaichu5(ModelTest):
    NATIVE_MODEL_ID = "TaichuAI/ZDTaichu5.0-9B"
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 1
    QUANT_BATCH_SIZE = 1
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    # The evaluator metadata is declared here first so the native baseline uses
    # the same ARC task and chat-template settings as the quantized run.
    EVAL_TASKS_SLOW = {"arc_challenge": {"chat_template": True}}
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_zdtaichu5(self):
        # Populate task discovery metadata before ModelTest evaluates the dense
        # model. ARC values are measured from this runtime's native checkpoint.
        self.get_eval_tasks()
        with self.model_compat_test_context():
            native_results = self._get_current_native_eval_results()

        native_metrics = (native_results or {}).get("arc_challenge", {})
        self.assertIsInstance(native_metrics, dict)
        acc_key = self._resolve_metric_key("acc", native_metrics)
        acc_norm_key = self._resolve_metric_key("acc_norm", native_metrics)
        self.assertIsNotNone(acc_key, "Native ARC result did not contain acc")
        self.assertIsNotNone(acc_norm_key, "Native ARC result did not contain acc_norm")

        acc = float(native_metrics[acc_key])
        acc_norm = float(native_metrics[acc_norm_key])
        self.assertTrue(math.isfinite(acc) and acc > 0, f"Invalid native ARC acc: {acc}")
        self.assertTrue(
            math.isfinite(acc_norm) and acc_norm > 0,
            f"Invalid native ARC acc_norm: {acc_norm}",
        )

        # Runtime native ARC baseline: these values intentionally remain dynamic
        # because the official model has no checked-in measured ARC scores.
        measured_tasks = {
            "arc_challenge": {
                "chat_template": True,
                "acc": {"value": acc, "floor_pct": 0.04},
                "acc_norm": {"value": acc_norm, "floor_pct": 0.04},
            }
        }
        self.EVAL_TASKS_SLOW = measured_tasks
        self.EVAL_TASKS_FAST = self.derive_fast_eval_tasks(measured_tasks)
        self.quantize_and_evaluate()
