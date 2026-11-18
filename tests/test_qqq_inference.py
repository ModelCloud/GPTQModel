# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from pathlib import Path

import pytest
from models.model_test import ModelTest

from gptqmodel import GPTQModel


pytestmark = [pytest.mark.model, pytest.mark.slow]

# Local copy of HandH1998/QQQ-Llama-3-8b-g128.
MODEL_PATH = "/monster/data/model/QQQ-Llama-3-8b-g128"


@pytest.mark.skipif(not Path(MODEL_PATH).is_dir(), reason=f"local checkpoint missing: {MODEL_PATH}")
def test_qqq_inference():
    model = GPTQModel.load(MODEL_PATH)
    str_output = ModelTest.generate_stable_with_limit(
        model,
        model.tokenizer,
        "The capital city of France is named",
        # The answer lands in the first few tokens; a long tail only multiplies
        # the per-token cost of the uncached CPU INT4 expansion.
        max_new_tokens=16,
    )
    assert "paris" in str_output.lower() or "city" in str_output.lower()
