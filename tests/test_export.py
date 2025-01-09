import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import tempfile
import unittest

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import FORMAT, QuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer


class TestExport(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_id, use_fast=True)
        traindata = load_dataset("allenai/c4",data_files="en/c4-train.00001-of-01024.json.gz",split="train").filter(lambda x: len(x['text']) <= 512)
        self.calibration_dataset = [self.tokenizer(example["text"]) for example in traindata.select(range(1))]

    def test_export_mlx(self):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=FORMAT.GPTQ,
            desc_act=False,
        )

        model = GPTQModel.load(
            self.pretrained_model_id,
            quantize_config=quantize_config,
        )

        model.quantize(self.calibration_dataset)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            model.tokenizer.save_pretrained(tmp_dir)

            with tempfile.TemporaryDirectory() as export_dir:
                GPTQModel.export(
                    model_id_or_path=tmp_dir,
                    target_path=export_dir,
                    format="mlx"
                )

                from mlx_lm import load, generate
                mlx_model, tokenizer = load(export_dir)

                prompt = "Write a story about Einstein"

                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )

                text = generate(mlx_model, tokenizer, prompt=prompt, verbose=True)
                
                assert text == "Write a story about Einstein"


            
