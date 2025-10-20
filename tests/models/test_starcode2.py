# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch  # noqa: E402
from model_test import ModelTest


class TestStarCode2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/starcoder2-3b"
    NATIVE_ARC_CHALLENGE_ACC = 0.2329
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2824
    TORCH_DTYPE = torch.float16
    def test_starcode2(self):
        self.quant_lm_eval()


