import torch
import unittest

from transformers import AutoTokenizer
from auto_gptq_next.utils import Perplexity
from auto_gptq_next import AutoGPTQNext
from auto_gptq_next.quantization import FORMAT, QuantizeConfig
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

        avg = sum(all) / len(all)

        # use 4090, wikitext-2-raw-v1, test, text, 512, 512 as reference
        assert avg < 8.5

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

    @parameterized.expand(
        [
            FORMAT.GPTQ_V2,
            FORMAT.GPTQ,
            FORMAT.MARLIN,
        ]
    )
    def test_quantized_perplexity(self, format: FORMAT):
        cal_data = [
            self.tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            ),
            self.tokenizer("Today I am in Paris and it is a wonderful day."),
        ]

        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=format,
        )

        model = AutoGPTQNext.from_pretrained(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )

        model.quantize(cal_data)

        quantized_ppl = self.calculate_avg_ppl(model, self.tokenizer)

        print(f"Quantized PPL: {quantized_ppl}")

        assert quantized_ppl - self.native_ppl < 1.0
