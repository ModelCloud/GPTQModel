# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGlmOCR(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/GLM-OCR" # zai-org/GLM-OCR
    NATIVE_ARC_CHALLENGE_ACC = 0.3302
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3455
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = NATIVE_ARC_CHALLENGE_ACC_SLOW
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": NATIVE_ARC_CHALLENGE_ACC},
            "acc_norm": {"value": NATIVE_ARC_CHALLENGE_ACC_NORM},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_glm_ocr(self):
        self.quantize_and_evaluate()
