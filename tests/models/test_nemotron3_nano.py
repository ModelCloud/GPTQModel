# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import transformers.utils.import_utils as import_utils

import_utils._torchao_available = False
import_utils._torchao_version = "0.0.0"
os.environ.setdefault("cuda_pci_bus_order", "9")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["cuda_pci_bus_order"]

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestNemotron3Nano(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    TRUST_REMOTE_CODE = True
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.35,
                "floor_pct": 1.0,
                "ceil_pct": 100.0,
            },
        },
    }

    def test_nemotron3_nano(self):
        self.quant_lm_eval()
