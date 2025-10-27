# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch  # noqa: E402
from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/falcon-7b-instruct" # "tiiuae/falcon-7b-instruct"
    TRUST_REMOTE_CODE = False
    TORCH_DTYPE = torch.float16
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": 0.3993, "floor_pct": 0.52},
            "acc_norm": {"value": 0.4292, "floor_pct": 0.52},
        },
    }
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False

    def test_falcon(self):
        self.quant_lm_eval()
