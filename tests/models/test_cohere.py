# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestCohere(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/aya-expanse-8b" # "CohereForAI/aya-expanse-8b"
    NATIVE_ARC_CHALLENGE_ACC = 0.5401
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5640
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.20
    EVAL_BATCH_SIZE = 4

    def test_cohere(self):
        self.quant_lm_eval()
