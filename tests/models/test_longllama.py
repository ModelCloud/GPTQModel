# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.backend import BACKEND


class TestLongLlama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/long_llama_3b_instruct" # "syzymon/long_llama_3b_instruct"
    TRUST_REMOTE_CODE = True
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.3515, "floor_pct": 0.5},
            "acc_norm": {"value": 0.3652, "floor_pct": 0.5},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    USE_VLLM = False
    USE_FLASH_ATTN = False
    LOAD_BACKEND = BACKEND.TORCH
    OFFLOAD_TO_DISK = False  # Local checkpoint is a monolithic .bin, so LazyTurtle offload is unavailable.

    def test_longllama(self):
        self.quantize_and_evaluate()
