# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestLongLlama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/long_llama_3b_instruct" # "syzymon/long_llama_3b_instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3515
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3652
    TRUST_REMOTE_CODE = True
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.5
    USE_VLLM = False

    def test_longllama(self):
        self.quant_lm_eval()
