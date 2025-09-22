# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestInstella(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Instella-3B-Instruct/"
    NATIVE_ARC_CHALLENGE_ACC = 0.4377
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4804
    TRUST_REMOTE_CODE = True

    def test_instella(self):
        self.quant_lm_eval()
