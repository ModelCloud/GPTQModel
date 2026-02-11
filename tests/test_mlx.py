# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import sys


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import tempfile  # noqa: E402

from mlx_lm import generate, load  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct/gptq_4bits_01-07_14-18-11_maxlen1024_ns1024_descFalse_damp0.1/"

class TestExport(ModelTest):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.NATIVE_MODEL_ID, use_fast=True)
        cls.calibration_dataset = cls.load_dataset(cls.tokenizer, cls.DATASET_SIZE)

    def test_export_mlx(self):
        with tempfile.TemporaryDirectory() as export_dir:
            GPTQModel.export(
                model_id_or_path=self.NATIVE_MODEL_ID,
                target_path=export_dir,
                format="mlx"
            )
            mlx_model, tokenizer = load(export_dir)

            messages = [{"role": "user", "content": self.INFERENCE_PROMPT}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            text = generate(mlx_model, tokenizer, prompt=prompt, verbose=True)

            self.assertIn("paris", text.lower())

######### test_mlx_generate.py ##########

class TestMlxGenerate(ModelTest):
    def test_mlx_generate(self):
        mlx_model = GPTQModel.load(
            NATIVE_MODEL_ID,
            backend=BACKEND.MLX
        )

        messages = [{"role": "user", "content": self.INFERENCE_PROMPT}]
        prompt = mlx_model.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        text = mlx_model.generate(prompt=prompt)
        assert "paris" in text.lower()



