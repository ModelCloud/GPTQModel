# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGlm5Next(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/GLM-5.3-Flash-REAP50-BF16"
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.49146757679180886, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5273037542662116, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    # The REAP50 BF16 checkpoint is 331 GB. Even after 4-bit quantization, the
    # model plus KDA workspaces should retain multi-GPU loading headroom.
    EVAL_SINGLE_GPU = False

    def test_glm5_next(self):
        self.quantize_and_evaluate()
