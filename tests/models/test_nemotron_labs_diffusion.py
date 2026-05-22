# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestNemotronUltra(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Nemotron-Labs-Diffusion-3B" # nvidia/Nemotron-Labs-Diffusion-3B
    # FIXME Evalution appears to be incompatible with NemoStation/Marlin-2B support; the original model's scores are also quite low.
    # original model score: {'arc_challenge': {'accuracy,loglikelihood': 0.19795221843003413, 'accuracy,loglikelihood_norm': 0.20819112627986347}}
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.197098976109215, "floor_pct": 0.36},
            "acc_norm": {"value": 0.2235494880546075, "floor_pct": 0.36},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    TRUST_REMOTE_CODE = True
    SAVE_PATH = "./temp/Nemotron-Labs-Diffusion"

    def test_nemotron_ultra(self):
        self.quantize_and_evaluate()
