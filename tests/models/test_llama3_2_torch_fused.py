# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import tempfile

from parameterized import parameterized

from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
from gptqmodel.models._const import DEVICE
from model_test import ModelTest

class TestLlama_TorchFused(ModelTest):
    NATIVE_MODEL_ID = "ModelCloud/Llama-3.2-1B-gptqmodel-ci-4bit"

    @parameterized.expand([
        BACKEND.TORCH,
        BACKEND.TORCH_FUSED
    ])
    def test_with_torch_fused_cpu(self, backend):
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            backend=BACKEND.TORCH_FUSED,
            device=DEVICE.CPU,
        )
        tokenizer = model.tokenizer
        generate_str = tokenizer.decode(
            model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(model.device),
                           max_new_tokens=512)[0])

        print(f"generate_str: {generate_str}")

        self.assertIn("paris", generate_str.lower())
