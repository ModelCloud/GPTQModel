# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import importlib.util
import unittest

from model_test import ModelTest


class TestCodeGen(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/codegen2-1B_P" # "Salesforce/codegen2-1B_P"
    NATIVE_ARC_CHALLENGE_ACC = 0.1749
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2005
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
        if importlib.util.find_spec("transformers.onnx") is None:
            raise unittest.SkipTest(
                "CodeGen remote config requires transformers.onnx, which is unavailable in this environment"
            )

    def test_codegen(self):
        self.quant_lm_eval()
