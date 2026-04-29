# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

import pytest


os.environ["GPTQMODEL_USE_MODELSCOPE"] = "True"
from models.model_test import ModelTest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402


pytestmark = [pytest.mark.model, pytest.mark.slow]


class TestLoadModelscope(ModelTest):

    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"

    def test_load_modelscope(self):
        model = GPTQModel.load(self.MODEL_ID)

        str_output = self.generate_stable_with_limit(
            model,
            model.tokenizer,
            "The capital city of France is named",
        )
        assert "paris" in str_output.lower() or "city" in str_output.lower()

        del model
