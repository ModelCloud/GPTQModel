import torch
import unittest
import tempfile

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

        return avg

    def calculate_native_ppl(self):
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

        # use 4090, wikitext-2-raw-v1, test, text, 512, 512 as reference
        assert self.native_ppl < 8.5

    def get_wikitext2_data(self, n_samples=1024):
        from datasets import load_dataset
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # avoid using very short rows for calibration, min 128 chars
        traindata = traindata.filter(lambda x: len(x['text']) >= n_samples)

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
        if not hasattr(self, "native_ppl"):
            self.native_ppl = self.calculate_native_ppl()

        cal_data = self.get_wikitext2_data()

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

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(
                tmp_dir,
            )

            model = AutoGPTQNext.from_quantized(
                tmp_dir,
                device_map="auto",
                use_marlin=format == FORMAT.MARLIN,
            )

            quantized_ppl = self.calculate_avg_ppl(model, self.tokenizer)

            print(f"Format {format}, Quantized PPL: {quantized_ppl}")

            assert quantized_ppl - self.native_ppl < 1.0
