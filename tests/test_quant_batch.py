# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402

import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization import QuantizeConfig  # noqa: E402


class TestQuantBatch(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"

    def _generate(self, model, tokenizer, prompt: str = "Paris is known as"):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return outputs

    @classmethod
    def setUpClass(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.NATIVE_MODEL_ID, use_fast=True)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.calibration_dataset = self.load_dataset(self.tokenizer, self.DATASET_SIZE)

    def test_diff_batch(self):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
        )

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )

        model.quantize(self.calibration_dataset, batch_size=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(
                tmp_dir,
            )

            del model

            model = GPTQModel.load(
                tmp_dir,
            )

            batch_1_ids = self._generate(model, self.tokenizer)

            model = GPTQModel.load(
                self.NATIVE_MODEL_ID,
                quantize_config=quantize_config,
            )

        model.quantize(self.calibration_dataset, batch_size=256)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(
                tmp_dir,
            )

            del model

            model = GPTQModel.load(
                tmp_dir,
            )

            batch_n_ids = self._generate(model, self.tokenizer)

            del model

        self.assertTrue((batch_1_ids == batch_n_ids).all().item())
