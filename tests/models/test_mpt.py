# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestMpt(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/mpt-7b-instruct" # "mosaicml/mpt-7b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.4275
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4454
    APPLY_CHAT_TEMPLATE = False
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6
    DATASET_SIZE = 96
    MAX_QUANT_LAYERS = None
    MOCK_QUANTIZATION = True
    OFFLOAD_TO_DISK = False

    def test_mpt(self):
        self.quant_lm_eval()
