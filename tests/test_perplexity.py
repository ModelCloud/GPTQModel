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
            model,
            tokenizer,
            self.DATASET_PATH,
            self.DATASET_NAME,
            self.DATASET_SPLIT,
            self.DATASET_COLUMN,
        )

        all_perplexity = ppl.calculate_perplexity(self.N_CTX, self.N_BATCH)

        avg_perplexity = sum(all_perplexity) / len(all_perplexity)

        # use 4090, wikitext-2-raw-v1, test, text, 512, 512 as reference
        assert avg_perplexity < 8.5

        return avg_perplexity

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

    @parameterized.expand(
        [
            (False, True, FORMAT.GPTQ_V2),
            (False, False, FORMAT.GPTQ),
            (True, True, FORMAT.MARLIN),
        ]
    )
    def test_quantized_perplexity(self, use_marlin: bool, sym: bool, format: FORMAT):
        calibration_dataset = [
            self.tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            ),
            self.tokenizer("Today I am in Paris and it is a wonderful day."),
        ]

        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=True,
            sym=sym,
            format=format,
        )

        model = AutoGPTQNext.from_pretrained(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )

        model.quantize(calibration_dataset)

        quantized_ppl = self.calculate_avg_ppl(model, self.tokenizer)

        assert quantized_ppl - self.native_ppl < 1.0
