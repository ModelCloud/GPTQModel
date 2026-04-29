# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
from models.model_test import ModelTest

from gptqmodel import GPTQModel


pytestmark = [pytest.mark.model, pytest.mark.slow]

def test_qqq_inference():
    model = GPTQModel.load("HandH1998/QQQ-Llama-3-8b-g128")
    str_output = ModelTest.generate_stable_with_limit(
        model,
        model.tokenizer,
        "The capital city of France is named",
    )
    assert "paris" in str_output.lower() or "city" in str_output.lower()
