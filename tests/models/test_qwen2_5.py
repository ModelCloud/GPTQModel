# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.2901 |
# | arc_challenge :: acc_norm,none |   0.3140 |
# | mmlu_stem :: acc,none          |   0.3993 |
# | gsm8k_plat :: exact,flexible   |   0.3581 |
#
# Slow-mode baselines re-measured on 2026-11-16 (NVIDIA GB10, CUDA 13.2,
# transformers 5.22.1) with:
#   CUDA_HOME=<complete CUDA toolkit> \
#   GPTQMODEL_MODEL_TEST_MODE=slow \
#   PYTHONPATH=/root/work-4398b \
#   /root/venv/bin/python -m pytest tests/models/test_qwen2_5.py -q -s
# (on the measuring host CUDA_HOME was
# /root/gptqmodel/.venv/lib/python3.12/site-packages/nvidia/cu13, because the
# primary venv's CUDA wheel omits the crt/ headers needed by the Marlin JIT).
# Model: Qwen/Qwen2.5-0.5B-Instruct revision
# 7ae557604adf67be50417f59c2c2f167def9a775, materialized at
# /monster/data/model/Qwen2.5-0.5B-Instruct. Evaluation is deterministic
# (do_sample=false, seed 898); a second evaluation of the saved checkpoint
# reproduced the gsm8k value exactly (0.35814722911497104). The acceptance
# thresholds are unchanged; only the stored expectations move to the measured
# values.
class TestQwen2_5(ModelTest):
    GROUP_SIZE = 32
    HESSIAN_CHUNK_SIZE = 256 * 1024 * 1024
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.35814722911497104,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "acc": {"value": 0.2901023890784983, "floor_pct": 0.04},
            "acc_norm": {"value": 0.31399317406143346, "floor_pct": 0.04},
        },
        "mmlu_stem": {
            "acc": {"value": 0.39930225182366, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.38626964433416044,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "arc_challenge": {
            "acc": {
                "value": 0.2977815699658703,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.34044368600682595,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "mmlu_stem": {
            "acc": {
                "value": 0.3967649857278782,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }

    def test_qwen2_5(self):
        self.quantize_and_evaluate()
