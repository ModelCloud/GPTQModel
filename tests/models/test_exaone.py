# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestExaone(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/EXAONE-3.0-7.8B-Instruct" # "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4232
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4164
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6

    def test_exaone(self):
        self.quant_lm_eval()


