# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
import unittest  # noqa: E402

from datasets import load_dataset  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT, QuantizeConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestQuantWithTrustRemoteTrue(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/MiniCPM-2B-dpo-bf16"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, use_fast=True, trust_remote_code=True)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        self.calibration_dataset = [self.tokenizer(example["text"]) for example in traindata.select(range(1024))]

    def test_diff_batch(self):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=FORMAT.GPTQ,
        )

        model = GPTQModel.load(
            self.MODEL_ID,
            quantize_config=quantize_config,
            trust_remote_code=True,
        )

        model.quantize(self.calibration_dataset, batch_size=64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(
                tmp_dir,
            )

            del model
            py_files = [f for f in os.listdir(tmp_dir) if f.endswith('.py')]
            expected_files = ["modeling_minicpm.py", "configuration_minicpm.py"]
            for file in expected_files:
                self.assertIn(file, py_files, f"File {file} is missing in the actual files list")



