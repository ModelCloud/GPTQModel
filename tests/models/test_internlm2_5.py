# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest

from model_test import ModelTest

from gptqmodel.models.definitions.internlm2 import InternLM2QModel
from gptqmodel.models.loader import check_versions


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
        # InternLM2.5 ships remote modeling code that the definition pins to
        # transformers<=4.44.2. Gate the test through the same requirement the
        # loader enforces so the skip reason always names the installed and
        # required versions, and the test runs automatically once the pin is
        # lifted.
        try:
            check_versions(InternLM2QModel, InternLM2QModel.require_pkgs)
        except ValueError as exc:
            raise unittest.SkipTest(f"InternLM2.5 is gated by its model definition: {exc}") from None

    def test_internlm2_5(self):
        self.quantize_and_evaluate()

