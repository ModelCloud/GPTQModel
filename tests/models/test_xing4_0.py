# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

from model_test import ModelTest


class TestXing4_0(ModelTest):
    NATIVE_MODEL_ID = "XingChen-AGI/Xing4.0-29B-A4B"
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "last"
    EVAL_BATCH_SIZE = "auto"

    def test_xing4_0(self):
        # Bootstrap the expected values from the current native checkpoint so
        # this public-model test never hard-codes an unmeasured score.
        native_results = self._get_current_native_eval_results()
        self.assertIsInstance(native_results, dict)
        native_metrics = native_results.get("arc_challenge")
        self.assertIsInstance(native_metrics, dict)

        baselines = {"chat_template": False}
        for metric_name in ("acc", "acc_norm"):
            metric_key = self._resolve_metric_key(metric_name, native_metrics)
            self.assertIsNotNone(
                metric_key,
                f"Native Xing4.0 baseline is missing `{metric_name}`",
            )
            value = float(native_metrics[metric_key])
            self.assertTrue(
                math.isfinite(value) and value > 0.0,
                f"Native Xing4.0 `{metric_key}` must be finite and positive, got {value!r}",
            )
            baselines[metric_name] = {"value": value, "floor_pct": 0.04}

        # Configure the instance only after the native baseline has been
        # checked.  Fast mode receives the same values with its normal derived
        # tolerance, while the class metadata above keeps native eval usable.
        self.EVAL_TASKS_SLOW = {"arc_challenge": baselines}
        self.EVAL_TASKS_FAST = self.derive_fast_eval_tasks(self.EVAL_TASKS_SLOW)
        self.quantize_and_evaluate()
