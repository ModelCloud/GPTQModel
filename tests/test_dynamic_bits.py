# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
import unittest  # noqa: E402

from datasets import load_dataset  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT, QuantizeConfig  # noqa: E402
from gptqmodel.utils import Perplexity  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestDynamic(unittest.TestCase):
    NATIVE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def calculate_avg_ppl(self, model, tokenizer):
        ppl = Perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path="wikitext",
            dataset_name="wikitext-2-raw-v1",
            split="test",
            text_column="text",
        )

        all = ppl.calculate(n_ctx=512, n_batch=512)

        # average ppl
        avg = sum(all) / len(all)

        return avg

    @classmethod
    def setUpClass(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.NATIVE_MODEL_ID, use_fast=True)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        self.calibration_dataset = [self.tokenizer(example["text"]) for example in traindata.select(range(1024))]

    def test_dynamic_bits(self):
        dynamic_bits = {
            # `.*\.` matches the layers_node prefix
            r".*\.18\..*gate.*": 8, # match layer 18 (index start at 0) gate module
            r".*\.19\..*gate.*": 8, # match layer 19 (index start at 0) gate module
            r".*\.20\..*gate.*": 8, # match layer 21 (index start at 0) gate module
            r".*\.21\..*gate.*": 8, # match layer 22 (index start at 0) gate module
        }
        quantize_config = QuantizeConfig(
            bits=4,
            dynamic_bits=dynamic_bits,
            group_size=128,
        )
        model = GPTQModel.from_pretrained(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )
        model.quantize(self.calibration_dataset, batch_size=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
            )

            del model

            model = GPTQModel.from_quantized(
                tmp_dir,
                backend=BACKEND.TRITON,
            )

            dynamic_bits_ppl = self.calculate_avg_ppl(model, self.tokenizer)

            del model
            assert dynamic_bits_ppl < 10
