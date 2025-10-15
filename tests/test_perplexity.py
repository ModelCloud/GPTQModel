# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os
import time


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
import unittest  # noqa: E402

from datasets import load_dataset  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig  # noqa: E402
from gptqmodel.utils.perplexity import Perplexity  # noqa: E402
from gptqmodel.utils.rocm import IS_ROCM  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402


class TestPerplexity(unittest.TestCase):
    TINYLLAMA_MODEL_ID = "/monster/data/model/tinyllama-15M-stories" # "ModelCloud/tinyllama-15M-stories"
    OPT_MODEL_ID = "/monster/data/model/opt-125m" # "facebook/opt-125m"

    tinyllama_native_ppl = 54.616613778642396
    opt_125m_native_ppl = 30.3942897844937

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

    @classmethod
    def get_config_with_format(self, format: FORMAT):
        if format == FORMAT.MARLIN or format == FORMAT.BITBLAS:
            return self.OPT_DATASET_PATH, self.OPT_DATASET_NAME, self.OPT_DATASET_SPLIT, self.OPT_DATASET_COLUMN, self.OPT_MODEL_ID, self.opt_tokenizer
        else:
            return self.TINYLLAMA_DATASET_PATH, self.TINYLLAMA_DATASET_NAME, self.TINYLLAMA_DATASET_SPLIT, self.TINYLLAMA_DATASET_COLUMN, self.TINYLLAMA_MODEL_ID, self.tinyllama_tokenizer

    @classmethod
    def calculate_avg_ppl(self, path, name, split, column, model, tokenizer):
        ppl = Perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path=path,
            dataset_name=name,
            split=split,
            text_column=column,
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

        self.tinyllama_calibration_dataset, self.tinyllama_native_ppl = self.calculate_native_ppl(FORMAT.GPTQ)
        self.opt_calibration_dataset, self.opt_native_ppl = self.calculate_native_ppl( FORMAT.MARLIN)


    @classmethod
    def calculate_native_ppl(self, format):
        dataset_path, dataset_name, dataset_split, dataset_column, model_id, tokenizer = self.get_config_with_format(format)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )

        if model_id == self.TINYLLAMA_MODEL_ID:
            native_ppl = self.tinyllama_native_ppl
        elif model_id == self.OPT_MODEL_ID:
            native_ppl = self.opt_125m_native_ppl
        else:
            native_ppl = self.calculate_avg_ppl(
                dataset_path,
                dataset_name,
                dataset_split,
                dataset_column,
                model,
                tokenizer,
            )

        print(f"{model_id} Native PPL: {native_ppl}")

        #  4090: [wikitext-2-raw-v1, test, text, 512, 512] data split, tinyllama ppl == 8.4790, opt ppl == 30.02
        # assert self.native_ppl < 30.5

        length = 512 if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else 2048
        traindata = load_dataset(dataset_path, dataset_name, split=dataset_split).filter(lambda x: len(x[dataset_column]) >= length)
        calibration_dataset = traindata.select(range(1024))
        return calibration_dataset, native_ppl

    @parameterized.expand(
        [
            # (QUANT_METHOD.GPTQ, FORMAT.GPTQ, 8, 32), # A100, 4889 max ram
            (METHOD.GPTQ, FORMAT.GPTQ, 8, 32), # A100, 6571 max ram
            # (QUANT_METHOD.GPTQ, FORMAT.GPTQ_V2, 8, 32),
            # (QUANT_METHOD.GPTQ, FORMAT.GPTQ_V2, 4, 32),
            # (QUANT_METHOD.GPTQ, FORMAT.GPTQ, 4, 32),
            # (QUANT_METHOD.GPTQ, FORMAT.BITBLAS, 4, 32),
            # (QUANT_METHOD.AUTO_ROUND, FORMAT.GPTQ, 4, 32),
        ]
    )
    def test_quantized_perplexity(self, method: METHOD, format: FORMAT, bits: int, group_size: int):
        if method == METHOD.GPTQ:
            quantize_config = QuantizeConfig(
                bits=bits,
                group_size=group_size,
                format=format,
                desc_act=False if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else True,
                # inject this rule so dynamic logic is checked even if zero matches happen
                dynamic={
                    "-:.*mlp\.NEVER_NEGATIVE_MATCH_proj.*": {"bits": 8 if format != FORMAT.BITBLAS else 4, "group_size": 32},
                    "+:.*mlp\.NEVER_POSITIVE_MATCH_proj.*": {"bits": 8 if format != FORMAT.BITBLAS else 4, "group_size": 32},
                    ":.*mlp\.NEVER_POSITIVE_MATCH2_proj.*": {"bits": 8 if format != FORMAT.BITBLAS else 4, "group_size": 32},
                }
            )
        else:
            raise ValueError(f"Invalid quantization method: {method}")

        dataset_path, dataset_name, dataset_split, dataset_column, model_id, tokenizer = self.get_config_with_format(format)

        model = GPTQModel.load(
            model_id,
            quantize_config=quantize_config,
        )

        dataset = self.opt_calibration_dataset if format == FORMAT.MARLIN or format == FORMAT.BITBLAS else self.tinyllama_calibration_dataset
        start = time.time()
        model.quantize(
            dataset,
            batch_size=128 if IS_ROCM else 256,
        )
        quant_time = time.time() - start

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(
                tmp_dir,
            )

            # TODO: move to a new test
            # test upload
            # model.push_to_hub(
            #     repo_id="ModelCloud/CiUploadTest",
            #     quantized_path=tmp_dir,
            #     private=True,
            #     exists_ok=True,
            # )

            del model
            torch_empty_cache()

            # GPTQModel.push_to_hub(
            #     repo_id="ModelCloud/CiUploadTest",
            #     quantized_path=tmp_dir,
            #     private=True,
            #     exists_ok=True,
            # )

            model = GPTQModel.load(
                tmp_dir,
                backend=BACKEND.TORCH,
                device_map="auto",
            )

            start = time.time()
            quantized_ppl = self.calculate_avg_ppl(
                dataset_path,
                dataset_name,
                dataset_split,
                dataset_column,
                model,
                tokenizer,
            )
            ppl_time = time.time() - start

            print(f"Format {format}, Quantized PPL: {quantized_ppl}, Quant Time: {quant_time:.2f}, PPL Time: {ppl_time:.2f}")

            # 4090: [wikitext-2-raw-v1, test, text, 512, 512] data split
            # FORMAT.GTPQ and FORMAT.GTPQ_V2 Tinyllama ppl == 8.7863, FORMAT.MARLIN Tinyllama ppl == 9.0036
            # FORMAT.MARLIN opt ppl == 33.43, FORMAT.BITBLAS opt ppl == 32.61, native opt ppl == 30.39
            # FORMAT.GTPQ and FORMAT.GTPQ_V2 Tinyllama-15M ppl == 111.32, native Tinyllama-15M ppl == 54.61
            if format == FORMAT.MARLIN or format == FORMAT.BITBLAS:
                assert abs(quantized_ppl - self.opt_native_ppl) < 4.5
            else:
                assert abs(quantized_ppl - self.tinyllama_native_ppl) < 60
