import os
import tempfile
import unittest

from gptqmodel import GPTQModel
from gptqmodel.quantization import FORMAT, QuantizeConfig

from transformers import AutoTokenizer


class TestSharded(unittest.TestCase):

    def get_wikitext2_data(self, tokenizer, n_samples=1):
        from datasets import load_dataset
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        traindata = traindata.filter(lambda x: len(x['text']) >= 512)

        ds = traindata

        traindataset = []
        for example in ds:
            if len(traindataset) == n_samples:
                break

            traindataset.append(tokenizer(example["text"]))

        return traindataset

    def test_save_and_load(self):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        model = GPTQModel.from_pretrained(
            model_name,
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                format=FORMAT.GPTQ_V2,
            ))

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        cal_data = self.get_wikitext2_data(tokenizer)

        model.quantize(cal_data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
                max_shard_size="10MB"
            )

            files_and_dirs = os.listdir(tmp_dir)

            self.assertTrue(len(files_and_dirs) == 72)

            model = GPTQModel.from_quantized(
                tmp_dir,
                device="cuda:0",
            )

            tokens = model.generate(**tokenizer("1337", return_tensors="pt").to(model.device), max_new_tokens=20)[0]
            result = tokenizer.decode(tokens)

            self.assertTrue(result == "<s> 1337 \n- 1437 \n- 1537 \n- ")
