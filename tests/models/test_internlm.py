# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest

import transformers
from model_test import ModelTest
from packaging.version import Version


class TestInternlm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/internlm-7b" # "internlm/internlm-7b"
    NATIVE_ARC_CHALLENGE_ACC = 0.4164
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4309
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = NATIVE_ARC_CHALLENGE_ACC_SLOW
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW
    TRUST_REMOTE_CODE = True
    USE_VLLM = False
    USE_FLASH_ATTN = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if Version(transformers.__version__) > Version("4.44.2"):
            raise unittest.SkipTest(
                "InternLM requires transformers<=4.44.2 in this test environment"
            )

    def test_internlm(self):
        # transformers<=4.44.2 run normal
        self.quantize_and_evaluate()
