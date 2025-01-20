import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import tempfile  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from mlx_lm import generate, load  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestExport(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct/gptq_4bits_01-07_14-18-11_maxlen1024_ns1024_descFalse_damp0.1/"

    @classmethod
    def setUpClass(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.NATIVE_MODEL_ID, use_fast=True)
        self.calibration_dataset = self.load_dataset(self.tokenizer)

    def test_export_mlx(self):
        with tempfile.TemporaryDirectory() as export_dir:
            GPTQModel.export(
                model_id_or_path=self.NATIVE_MODEL_ID,
                target_path=export_dir,
                format="mlx"
            )
            mlx_model, tokenizer = load(export_dir)

            prompt = "Tell me the city name. Which city is the capital of France?"

            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            text = generate(mlx_model, tokenizer, prompt=prompt, verbose=True)

            self.assertIn("paris", text.lower())
