# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestDeepseekV2Lite(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/DeepSeek-Coder-V2-Lite-Instruct" # "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4753
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4855
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_deepseekv2lite(self):
        self.quant_lm_eval()


