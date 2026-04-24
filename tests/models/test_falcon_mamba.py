# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch  # noqa: E402

from gptqmodel import BACKEND
from model_test import ModelTest


class TestFalcon(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/falcon-mamba-7b-instruct" # "tiiuae/falcon-mamba-7b-instruct"
    TRUST_REMOTE_CODE = False
    TORCH_DTYPE = torch.float16
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.5179, "floor_pct": 0.52},
            "acc_norm": {"value": 0.5042, "floor_pct": 0.52},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False
    ACT_GROUP_AWARE = False
    USE_FLASH_ATTN = False
    LOAD_BACKEND = BACKEND.TORCH
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_falcon(self):
        self.quantize_and_evaluate()
