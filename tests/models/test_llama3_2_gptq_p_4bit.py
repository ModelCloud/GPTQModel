# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptq_p_accuracy_base import GptqPAccuracyBase


class TestLlama3_2GptqP4Bit(GptqPAccuracyBase):
    BITS = 4
    SAVE_PATH = "/tmp/llama3_2_gptq_p_4bit_saved_ckpt"
    EVAL_TASKS_FAST = GptqPAccuracyBase.arc_challenge_tasks(acc=0.3140, acc_norm=0.3498, gsm8k=0.42)
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST

    def test_llama3_2_gptq_p_4bit(self):
        self.quantize_and_evaluate()
