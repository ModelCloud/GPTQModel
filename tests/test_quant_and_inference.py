import time
import unittest
import tempfile

import torch
from parameterized import parameterized
from gptqmodel import GPTQModel
from gptqmodel.quantization import QuantizeConfig, FORMAT
from transformers import AutoTokenizer


class TestQuantInference(unittest.TestCase):
    NATIVE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    DATASET_PATH = "wikitext"
    DATASET_NAME = "wikitext-2-raw-v1"
    DATASET_SPLIT = "test"
    DATASET_COLUMN = "text"

    N_CTX = 512
    N_BATCH = 512

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

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.NATIVE_MODEL_ID, use_fast=True)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @parameterized.expand(
        [
            FORMAT.GPTQ,
            # FORMAT.GPTQ,
            # FORMAT.GPTQ,
        ]
    )
    def test_quantize(self, format):
        # start_time = time.perf_counter()
        #
        # cal_data = self.get_wikitext2_data()
        #
        # quantize_config = QuantizeConfig(
        #     bits=4,
        #     group_size=128,
        #     format=format,
        # )
        #
        # model = GPTQModel.from_pretrained(
        #     self.NATIVE_MODEL_ID,
        #     quantize_config=quantize_config,
        # )
        #
        # model.quantize(cal_data)
        #
        # print(f"Quant take {time.perf_counter() - start_time}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # model.save_quantized(
            #    tmp_dir,
            # )
            #
            # del model

            infer_start_time = time.perf_counter()

            model = GPTQModel.from_quantized(
                # tmp_dir,
                "./test_quant_model/",
                device_map="auto",
            )

            for data in self.get_wikitext2_data()[0:25]:
                generate_kwargs = dict(
                    input_ids=torch.tensor([data["input_ids"]]).to("cuda"),
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=512,
                )

                outputs = model.generate(**generate_kwargs)

            print(f"Inference 25 times, take {time.perf_counter() - infer_start_time}")