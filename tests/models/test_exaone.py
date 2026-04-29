# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest

from model_test import ModelTest
from transformers.cache_utils import DynamicCache


class TestExaone(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/EXAONE-3.0-7.8B-Instruct" # "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4232
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4164
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = NATIVE_ARC_CHALLENGE_ACC_SLOW
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not hasattr(DynamicCache, "from_legacy_cache"):
            raise unittest.SkipTest(
                "Exaone remote code requires transformers.cache_utils.DynamicCache.from_legacy_cache"
            )

    def test_exaone(self):
        self.quantize_and_evaluate()
