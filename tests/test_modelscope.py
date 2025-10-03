# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


os.environ["GPTQMODEL_USE_MODELSCOPE"] = "True"
from models.model_test import ModelTest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402


class TestLoadModelscope(ModelTest):

    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"

    def test_load_modelscope(self):
        model = GPTQModel.load(self.MODEL_ID)

        result = model.generate("The capital of mainland China is")[0]
        str_output = model.tokenizer.decode(result)
        assert "beijing" in str_output.lower() or "bei-jing" in str_output.lower()

        del model
