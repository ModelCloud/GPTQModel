import torch
import unittest
import tempfile
import math
from transformers import AutoTokenizer
from gptqmodel.utils import Perplexity
from gptqmodel import GPTQModel
from gptqmodel.quantization import FORMAT, QuantizeConfig
from parameterized import parameterized

class TestPerplexity(unittest.TestCase):
    NATIVE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    DATASET_PATH = "wikitext"
    DATASET_NAME = "wikitext-2-raw-v1"
    DATASET_SPLIT = "test"
    DATASET_COLUMN = "text"

    N_CTX = 512
    N_BATCH = 512

    def calculate_avg_ppl(self, model, tokenizer):
        ppl = Perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path=self.DATASET_PATH,
            dataset_name=self.DATASET_NAME,
            split=self.DATASET_SPLIT,
            text_column=self.DATASET_COLUMN,
        )

        all = ppl.calculate(n_ctx=self.N_CTX, n_batch=self.N_BATCH)

        # average ppl 
        avg = sum(all) / len(all)

        return avg

    def setUp(self):
        from transformers import AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.NATIVE_MODEL_ID, use_fast=True)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.NATIVE_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self.native_ppl = self.calculate_avg_ppl(model, self.tokenizer)

        print(f"Native PPL: {self.native_ppl}")

        #  4090: [wikitext-2-raw-v1, test, text, 512, 512] data split, tinyllama ppl == 8.4790
        assert self.native_ppl < 8.5

        return self.native_ppl

    def get_wikitext2_data(self, n_samples=1024):
        from datasets import load_dataset
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        traindata = traindata.filter(lambda x: len(x['text']) >= 512)

        ds = traindata

        traindataset = []
        for example in ds:
            if len(traindataset) == n_samples:
                break

            traindataset.append(self.tokenizer(example["text"]))

        return traindataset

    @parameterized.expand(
        [
            FORMAT.GPTQ_V2,
            FORMAT.GPTQ,
            FORMAT.MARLIN,
        ]
    )
    def test_quantized_perplexity(self, format: FORMAT):
        cal_data = self.get_wikitext2_data()

        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=format,
        )

        model = GPTQModel.from_pretrained(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )

        model.quantize(cal_data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
            )

            model = GPTQModel.from_quantized(
                tmp_dir,
                device_map="auto",
            )

            quantized_ppl = self.calculate_avg_ppl(model, self.tokenizer)

            print(f"Format {format}, Quantized PPL: {quantized_ppl}")

            # 4090: [wikitext-2-raw-v1, test, text, 512, 512] data split
            # FORMAT.GTPQ and FORMAT.GTPQ_V2 ppl == 8.7954, FORMAT.MARLIN ppl == 8.9865
            assert abs(quantized_ppl - self.native_ppl) < 0.6
