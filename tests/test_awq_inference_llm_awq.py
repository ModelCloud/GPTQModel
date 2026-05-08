# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
from awq_test_utils import run_inference_only_generation_test

from gptqmodel import BACKEND


pytestmark = [pytest.mark.model, pytest.mark.slow]


def test_inference_quantized_by_llm_awq():
    run_inference_only_generation_test(
        "ModelCloud/opt-125m-llm-awq",  # this quantized by llm-awq
        backend=BACKEND.AUTO,
        max_new_tokens=512,
        extra_terms=("food", "market", "country"),
    )
