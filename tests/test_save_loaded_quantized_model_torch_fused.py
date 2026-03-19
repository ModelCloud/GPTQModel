# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import tempfile  # noqa: E402
import unittest  # noqa: E402

from models.model_test import ModelTest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

class TestSaveTorchFused(unittest.TestCase):
    def test_save(self):
        prompt = "I am in Paris and"
        backend = BACKEND.TORCH_FUSED
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # origin model produce correct output
        origin_model = GPTQModel.load(MODEL_ID, backend=backend)
        origin_model_predicted_text = ModelTest.generate_stable_with_limit(
            origin_model,
            tokenizer,
            prompt,
            min_new_tokens=60,
            max_new_tokens=60,
            skip_special_tokens=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            origin_model.save(tmpdir)

            # saved model produce wrong output
            new_model = GPTQModel.load(tmpdir, backend=backend)

            new_model_predicted_text = ModelTest.generate_stable_with_limit(
                new_model,
                tokenizer,
                prompt,
                min_new_tokens=60,
                max_new_tokens=60,
                skip_special_tokens=False,
            )

            print("origin_model_predicted_text", origin_model_predicted_text)
            print("new_model_predicted_text", new_model_predicted_text)

            self.assertEqual(origin_model_predicted_text[:20], new_model_predicted_text[:20])
