# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
import unittest  # noqa: E402

from datasets import load_dataset  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT, QuantizeConfig  # noqa: E402
from gptqmodel.utils import Perplexity  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


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

    @classmethod
    def setUpClass(self):
        from transformers import AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.NATIVE_MODEL_ID, use_fast=True)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.NATIVE_MODEL_ID,
            device_map="auto",
        )

        self.native_ppl = self.calculate_avg_ppl(self, model, self.tokenizer)

        print(f"Native PPL: {self.native_ppl}")

        #  4090: [wikitext-2-raw-v1, test, text, 512, 512] data split, tinyllama ppl == 8.4790
        assert self.native_ppl < 8.5

        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        self.calibration_dataset = [self.tokenizer(example["text"]) for example in traindata.select(range(1024))]

    @parameterized.expand(
        [
            FORMAT.GPTQ_V2,
            FORMAT.GPTQ,
            FORMAT.MARLIN,
            FORMAT.BITBLAS,
        ]
    )
    def test_quantized_perplexity(self, format: FORMAT):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=format,
            desc_act=False if format == FORMAT.MARLIN else True
        )

        if format == FORMAT.MARLIN or format == FORMAT.BITBLAS:
            # MARLIN and BITBLAS Only supported when desc_act is False.
            quantize_config.desc_act = False

        model = GPTQModel.from_pretrained(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )

        model.quantize(self.calibration_dataset, batch_size=256)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
            )

            del model

            model = GPTQModel.from_quantized(
                tmp_dir,
                device_map="auto",
            )

            quantized_ppl = self.calculate_avg_ppl(model, self.tokenizer)

            print(f"Format {format}, Quantized PPL: {quantized_ppl}")

            # 4090: [wikitext-2-raw-v1, test, text, 512, 512] data split
            # FORMAT.GTPQ and FORMAT.GTPQ_V2 ppl == 8.7863, FORMAT.MARLIN ppl == 9.0036
            assert abs(quantized_ppl - self.native_ppl) < 0.6
