# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/stablelm-base-alpha-3b"
    NATIVE_ARC_CHALLENGE_ACC = 0.2363
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2577
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6

    def test_stablelm(self):
        self.quant_lm_eval()
