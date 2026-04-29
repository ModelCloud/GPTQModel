# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest

import transformers
from model_test import ModelTest
from packaging.version import Version


class TestInternlm2_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/internlm2_5-1_8b-chat" # "internlm/internlm2_5-1_8b-chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.3217
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3575
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = NATIVE_ARC_CHALLENGE_ACC_SLOW
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": NATIVE_ARC_CHALLENGE_ACC},
            "acc_norm": {"value": NATIVE_ARC_CHALLENGE_ACC_NORM},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if Version(transformers.__version__) > Version("4.44.2"):
            raise unittest.SkipTest(
                "InternLM2.5 requires transformers<=4.44.2 in this test environment"
            )

    def test_internlm2_5(self):
        # transformers<=4.44.2 run normal
        self.quantize_and_evaluate()

