# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
import unittest  # noqa: E402

from datasets import load_dataset  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization.config import FORMAT, QUANT_METHOD, AutoRoundQuantizeConfig, QuantizeConfig  # noqa: E402
from gptqmodel.utils import Perplexity  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


class TestPerplexity(unittest.TestCase):
    TINYLLAMA_MODEL_ID = "ModelCloud/tinyllama-15M-stories"
    OPT_MODEL_ID = "facebook/opt-125m"

    OPT_DATASET_PATH = "wikitext"
    OPT_DATASET_NAME = "wikitext-2-raw-v1"
    OPT_DATASET_SPLIT = "test"
    OPT_DATASET_COLUMN = "text"
    TINYLLAMA_DATASET_PATH = "skeskinen/TinyStories-hf"
    TINYLLAMA_DATASET_NAME = "default"
    TINYLLAMA_DATASET_SPLIT = "train"
    TINYLLAMA_DATASET_COLUMN = "text"

    N_CTX = 512
    N_BATCH = 512

    def calculate_avg_ppl(self, model, tokenizer, format: FORMAT):
        ppl = Perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path=self.OPT_DATASET_PATH if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_DATASET_PATH,
            dataset_name=self.OPT_DATASET_NAME if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_DATASET_NAME,
            split=self.OPT_DATASET_SPLIT if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_DATASET_SPLIT,
            text_column=self.OPT_DATASET_COLUMN if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_DATASET_COLUMN,
        )

        all = ppl.calculate(n_ctx=self.N_CTX, n_batch=self.N_BATCH)

        # average ppl
        avg = sum(all) / len(all)

        return avg

    @classmethod
    def setUpClass(self):
        self.tinyllama_tokenizer = AutoTokenizer.from_pretrained(self.TINYLLAMA_MODEL_ID, use_fast=True)

        if not self.tinyllama_tokenizer.pad_token_id:
            self.tinyllama_tokenizer.pad_token_id = self.tinyllama_tokenizer.eos_token_id

        self.opt_tokenizer = AutoTokenizer.from_pretrained(self.OPT_MODEL_ID, use_fast=True)

        if not self.opt_tokenizer.pad_token_id:
            self.opt_tokenizer.pad_token_id = self.opt_tokenizer.eos_token_id

        self.tinyllama_calibration_dataset, self.tinyllama_native_ppl = self.calculate_native_ppl(self, self.tinyllama_tokenizer, FORMAT.GPTQ)
        self.opt_calibration_dataset, self.opt_native_ppl = self.calculate_native_ppl(self, self.opt_tokenizer, FORMAT.MARLIN)


    def calculate_native_ppl(self, tokenizer, format):
        model_id = self.OPT_MODEL_ID if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_MODEL_ID
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )

        native_ppl = self.calculate_avg_ppl(self, model, tokenizer, format)

        print(f"{model_id} Native PPL: {native_ppl}")

        #  4090: [wikitext-2-raw-v1, test, text, 512, 512] data split, tinyllama ppl == 8.4790, opt ppl == 30.02
        # assert self.native_ppl < 30.5

        dataset_path = self.OPT_DATASET_PATH if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_DATASET_PATH
        dataset_name = self.OPT_DATASET_NAME if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_DATASET_NAME
        dataset_split = self.OPT_DATASET_SPLIT if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_DATASET_SPLIT
        dataset_column = self.OPT_DATASET_COLUMN if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_DATASET_COLUMN

        length = 512 if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else 2048
        traindata = load_dataset(dataset_path, dataset_name, split=dataset_split).filter(lambda x: len(x[dataset_column]) >= length)
        calibration_dataset = [tokenizer(example[dataset_column]) for example in traindata.select(range(1024))]
        return calibration_dataset, native_ppl

    @parameterized.expand(
        [
            (QUANT_METHOD.GPTQ, FORMAT.GPTQ_V2),
            (QUANT_METHOD.GPTQ, FORMAT.GPTQ),
            (QUANT_METHOD.GPTQ, FORMAT.MARLIN),
            (QUANT_METHOD.GPTQ, FORMAT.BITBLAS),
            (QUANT_METHOD.AUTO_ROUND, FORMAT.GPTQ),
        ]
    )
    def test_quantized_perplexity(self, method: QUANT_METHOD, format: FORMAT):
        if method == QUANT_METHOD.GPTQ:
            quantize_config = QuantizeConfig(
                bits=4,
                group_size=128,
                format=format,
                desc_act=False if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else True
            )
        elif method == QUANT_METHOD.AUTO_ROUND:
            quantize_config = AutoRoundQuantizeConfig(
                bits=4,
                group_size=128,
                format=format,
            )
        else:
            raise ValueError(f"Invalid quantization method: {method}")

        model = GPTQModel.from_pretrained(
            self.OPT_MODEL_ID if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.TINYLLAMA_MODEL_ID,
            quantize_config=quantize_config,
        )

        dataset = self.opt_calibration_dataset if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.tinyllama_calibration_dataset
        model.quantize(dataset, batch_size=256)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
            )

            del model

            model = GPTQModel.from_quantized(
                tmp_dir,
                device_map="auto",
            )

            quantized_ppl = self.calculate_avg_ppl(model, self.opt_tokenizer if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.tinyllama_tokenizer, format)

            print(f"Format {format}, Quantized PPL: {quantized_ppl}")

            # 4090: [wikitext-2-raw-v1, test, text, 512, 512] data split
            # FORMAT.GTPQ and FORMAT.GTPQ_V2 Tinyllama ppl == 8.7863, FORMAT.MARLIN Tinyllama ppl == 9.0036
            # FORMAT.MARLIN opt ppl == 33.43, FORMAT.BITBLAS opt ppl == 32.61, native opt ppl == 30.39
            # FORMAT.GTPQ and FORMAT.GTPQ_V2 Tinyllama-15M ppl == 111.32, native Tinyllama-15M ppl == 54.61
            if format == FORMAT.MARLIN or format == FORMAT.BITBLAS:
                assert abs(quantized_ppl - self.opt_native_ppl) < 3.5
            else:
                assert abs(quantized_ppl - self.tinyllama_native_ppl) < 56.8
