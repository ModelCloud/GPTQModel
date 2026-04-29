# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import sys

import torch


TESTS_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TESTS_MODELS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_MODELS_ROOT)

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.quantization import FORMAT, METHOD


# | Metric                                             | EXLLAMA_V3 |
# |----------------------------------------------------|------------|
# | arc_challenge :: acc,none                          |     0.3174 |
# | arc_challenge :: acc_norm,none                     |     0.3456 |
# | gsm8k_platinum_cot :: acc,num |     0.4715 |
# | gsm8k_platinum_cot :: exact_match,strict-match     |     0.4218 |
# | mmlu_stem :: acc,none                              |     0.3977 |
class TestLlama3_2_ExllamaV3(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.4715,
                "floor_pct": 0.04,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.3977,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3174,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3456,
                "floor_pct": 0.04,
            },
        },
    }

    FORMAT = FORMAT.EXL3
    METHOD = METHOD.EXL3
    BITS = 4.0
    GROUP_SIZE = -1
    ACT_GROUP_AWARE = False
    TORCH_DTYPE = torch.float16
    QUANT_BACKEND = BACKEND.EXLLAMA_V3
    LOAD_BACKEND = BACKEND.EXLLAMA_V3

    def test_llama3_2_exllamav3(self):
        self.quantize_and_evaluate()

        module = self.model.model.model.layers[0].self_attn.q_proj
        assert isinstance(module, ExllamaV3Linear)
        assert module.trellis.dtype == torch.int16
        assert module.suh.dtype == torch.float16
        assert module.svh.dtype == torch.float16
        assert module.mcg.dtype == torch.int32

        storage = module.tensor_storage_entry()
        assert storage["quant_format"] == "exl3"
        assert storage["bits_per_weight"] == 4
        assert "model.layers.0.self_attn.q_proj.trellis" in storage["stored_tensors"]
