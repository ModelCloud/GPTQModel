# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from gptqmodel import BACKEND
from scripts.eora_svd_algorithm_eval import task_backends


def test_task_backends_keep_unverified_two_bit_kernel_out_of_quality_sweep():
    assert task_backends(2) == {
        "arc_challenge": BACKEND.GPTQ_TORCH,
        "gsm8k_platinum_cot": BACKEND.GPTQ_TORCH,
    }
    assert task_backends(3)["gsm8k_platinum_cot"] == BACKEND.GPTQ_TRITON
    assert task_backends(4)["gsm8k_platinum_cot"] == BACKEND.MARLIN
