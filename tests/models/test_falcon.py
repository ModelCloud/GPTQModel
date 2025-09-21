# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch  # noqa: E402from tests.model_test import ModelTest
from model_test import ModelTest


class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/falcon-7b-instruct" # "tiiuae/falcon-7b-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3993
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4292
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = False
    TORCH_DTYPE = torch.float16
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.52
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False

    def test_falcon(self):
        self.quant_lm_eval()
