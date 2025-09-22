# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGranite(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/granite-3.0-2b-instruct" # "ibm-granite/granite-3.0-2b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4505
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4770
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2

    def test_granite(self):
        self.quant_lm_eval()
