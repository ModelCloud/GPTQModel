import torch
import unittest

from transformers import AutoTokenizer
from auto_gptq_next.utils import Perplexity

class TestPerplexity(unittest.TestCase):
    NATIVE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    QUANTIZED_MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    DATASET_PATH = "wikitext"
    DATASET_NAME = "wikitext-2-raw-v1"
    DATASET_SPLIT = "test"
    DATASET_COLUMN = "text"

    N_CTX = 512
    N_BATCH = 512

    def test_native_model_perplexity(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, use_fast=True)

        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
        )

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

        assert avg_perplexity < 9

    def test_quantized_gptq_format_model_perplexity(self):
        from auto_gptq_next import AutoGPTQNext
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, use_fast=True)

        quantized_model = AutoGPTQNext.from_quantized(
            self.QUANTIZED_MODEL_ID,
            device_map="auto",
            use_safetensors=True,
            disable_exllama=False,
        )

        ppl = Perplexity(
            quantized_model,
            tokenizer,
            self.DATASET_PATH,
            self.DATASET_NAME,
            self.DATASET_SPLIT,
            self.DATASET_COLUMN,
        )

        all_perplexity = ppl.calculate_perplexity(self.N_CTX, self.N_BATCH)

        avg_perplexity = sum(all_perplexity) / len(all_perplexity)

        assert avg_perplexity < 9
