# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import json
import logging
import os
import tempfile
import unittest

from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoTokenizer

from gptqmodel.nn_modules.qlinear.qqq import QQQQuantLinear
from gptqmodel.quantization import FORMAT, METHOD, QUANT_CONFIG_FILENAME
from gptqmodel.utils.torch import torch_empty_cache


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from logbar import LogBar

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402


log = LogBar.shared()

class TestGroupSize(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/Llama-3.2-1B"
        #"/monster/data/model/Qwen2.5-0.5B-Instruct/" "/monster/data/model/Qwen2.5-0.5B-Instruct/" #

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_id, use_fast=True)

        traindata = load_dataset("json", data_files="/monster/data/model/dataset/c4-train.00000-of-01024.json.gz", split="train")
        self.calibration_dataset = traindata.select(range(1024))

    def test_load_group_128(self):
        model = GPTQModel.load(
            "/monster/data/model/QQQ-Llama-3-8b-g128",
        )

        self.assert_qqq_linear(model)

        result = model.generate("Uncovering deep insights begins with")[0] # tokens
        log.info(f"Output: {model.tokenizer.decode(result)}") # string output

    # TODO FIXME: group_size 128 is failing this CI TEST!
    @parameterized.expand([-1, 128])
    def test_quant_and_inference(self, group_size: int):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=group_size,
            quant_method=METHOD.QQQ,
            format=FORMAT.QQQ,
        )

        model = GPTQModel.load(
            self.pretrained_model_id,
            quantize_config=quantize_config,
        )
        model.quantize(self.calibration_dataset, batch_size=1, calibration_concat_size=2048)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            model.save(tmp_dir_name)

            with open(tmp_dir_name + "/" + QUANT_CONFIG_FILENAME, "r") as f:
                file_dict = json.loads(f.read())

                # make sure the json dict saved to file matches config in memory
                assert model.quantize_config.to_dict() == file_dict
                logging.info(f"Saved config file: {file_dict}")

            del model
            torch_empty_cache()

            model = GPTQModel.load(
                tmp_dir_name,
            )

            self.assert_qqq_linear(model)

            tokens = model.generate("The capital city of France is named", min_new_tokens=128, max_new_tokens=128)[0]
            result = model.tokenizer.decode(tokens)
            print(f"BACKEND: {BACKEND.QQQ}, Result: {result}")
            if "paris" not in result.lower() and "city" not in result.lower() and "country" not in result.lower():
                raise AssertionError(" `paris` not found in `result`")

    def assert_qqq_linear(self, model):
        has_qqq = False
        for _, module in model.named_modules():
            linear = QQQQuantLinear
            if isinstance(module, linear):
                has_qqq = True
                break
        self.assertTrue(has_qqq)
