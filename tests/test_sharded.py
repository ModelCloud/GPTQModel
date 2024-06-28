# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import os  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

from gptqmodel import Backend, GPTQModel  # noqa: E402
from gptqmodel.quantization import QuantizeConfig  # noqa: E402
from gptqmodel.quantization.config import FORMAT  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


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
            ))

        tokenizer = AutoTokenizer.from_pretrained(model_name)

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

            print(result)
            self.assertTrue(len(result) > 0)

    def test_save_and_load_no_shard(self):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        model = GPTQModel.from_pretrained(
            model_name,
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
            ))

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        cal_data = self.get_wikitext2_data(tokenizer)

        model.quantize(cal_data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
                max_shard_size=None
            )

            files_and_dirs = os.listdir(tmp_dir)

            self.assertTrue(len(files_and_dirs) > 0)

            model = GPTQModel.from_quantized(
                tmp_dir,
                device="cuda:0",
            )

            tokens = model.generate(**tokenizer("1337", return_tensors="pt").to(model.device), max_new_tokens=20)[0]
            result = tokenizer.decode(tokens)

            print(result)
            self.assertTrue(len(result) > 0)

    def test_save_and_load_unsupports_shard(self):
        model_name = "facebook/opt-125m"

        model = GPTQModel.from_pretrained(
            model_name,
            quantize_config=QuantizeConfig(
                bits=4,
                group_size=128,
                format=FORMAT.BITBLAS,
                desc_act=False,
            ))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        cal_data = self.get_wikitext2_data(tokenizer)
        model.quantize(cal_data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
                max_shard_size="10MB",
            )

            files_and_dirs = os.listdir(tmp_dir)

            self.assertTrue(len(files_and_dirs) > 0)

            model = GPTQModel.from_quantized(
                tmp_dir,
                device="cuda:0",
                backend=Backend.BITBLAS,
            )

            tokens = model.generate(**tokenizer("1337", return_tensors="pt").to(model.device), max_new_tokens=20)[0]
            result = tokenizer.decode(tokens)
            print(result)
            self.assertTrue(len(result) > 0)
