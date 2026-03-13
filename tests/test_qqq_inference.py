# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest

from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

pytestmark = [pytest.mark.model, pytest.mark.slow]


from gptqmodel import GPTQModel

def test_qqq_inference():
    model = GPTQModel.load("HandH1998/QQQ-Llama-3-8b-g128")
    result = model.generate("The capital city of France is named")[0]
    str_output = model.tokenizer.decode(result)
    assert "paris" in str_output.lower() or "city" in str_output.lower()
