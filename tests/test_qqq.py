# -- do not touch
import json
import logging
import os
import tempfile
import unittest

from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel.quantization import QUANT_CONFIG_FILENAME

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from gptqmodel import GPTQModel, QuantizeConfig, BACKEND  # noqa: E402
# -- end do not touch
from logbar import LogBar

log = LogBar.shared()

class TestGroupSize(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/Llama-3.2-1B"
        #"/monster/data/model/Qwen2.5-0.5B-Instruct/" "/monster/data/model/Qwen2.5-0.5B-Instruct/" #

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_id, use_fast=True)

        traindata = load_dataset("json", data_files="/monster/data/model/dataset/c4-train.00000-of-01024.json.gz", split="train")
        self.calibration_dataset = [self.tokenizer(example["text"]) for example in traindata.select(range(1))]


    def test_load_group_128(self):
        model = GPTQModel.load(
            "/monster/data/model/QQQ-Llama-3-8b-g128",
        )

        result = model.generate("Uncovering deep insights begins with")[0] # tokens
        log.info(f"Output: {model.tokenizer.decode(result)}") # string output

    def test_quant_and_inference(self):
        quantize_config = QuantizeConfig(
            bits=4
        )

        model = GPTQModel.load(
            self.pretrained_model_id,
            quantize_config=quantize_config,
        )
        model.qqq_quantize(self.calibration_dataset, batch_size=1, calibration_dataset_concat_size=2048)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir_name = "test_qqq_quant_model"

            model.save(tmp_dir_name)

            with open(tmp_dir_name + "/" + QUANT_CONFIG_FILENAME, "r") as f:
                file_dict = json.loads(f.read())

                # make sure the json dict saved to file matches config in memory
                assert model.quantize_config.to_dict() == file_dict
                logging.info(f"Saved config file: {file_dict}")

            tokens = model.generate("Capital of France is")[0]
            result = model.tokenizer.decode(tokens)
            print(f"BACKEND: {BACKEND.QQQ}, Result: {result}")
            if "paris" not in result.lower():
                raise AssertionError(" `paris` not found in `result`")
