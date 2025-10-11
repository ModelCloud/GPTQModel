# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestLing(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ling-mini-2.0/"
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    NATIVE_ARC_CHALLENGE_ACC = 0.5009
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5137
    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = True
    # EVAL_BATCH_SIZE = 6
    V2 = False
    DEBUG = True
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DATASET_SIZE = 2048
    DATASET_SORT = "desc"
    QUANT_BATCH_SIZE = 8
    CALIB_NOISE_MODE = "unseen"
    CALIB_NOISE_PERCENT = 0.025

    def test_mimo(self):
        self.quant_lm_eval()
