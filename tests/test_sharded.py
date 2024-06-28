# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import os  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

from gptqmodel import Backend, GPTQModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestSharded(unittest.TestCase):
    MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def test_save_and_load(self):
        model = GPTQModel.from_quantized(
            self.MODEL_ID,
            device_map="auto",
            backend=Backend.MARLIN,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
                max_shard_size="100MB"
            )

            del model

            index_file_path = os.path.join(tmp_dir, "gptq_model-4bit-128g.safetensors.index.json")
            self.assertTrue(os.path.exists(index_file_path))

            model = GPTQModel.from_quantized(
                tmp_dir,
                device_map="auto",
            )

            tokens = model.generate(**tokenizer("1337", return_tensors="pt").to(model.device), max_new_tokens=20)[0]
            result = tokenizer.decode(tokens)

            print(result)
            self.assertTrue(len(result) > 0)

    def test_save_and_load_no_shard(self):
        model = GPTQModel.from_quantized(
            self.MODEL_ID,
            device_map="auto",
            backend=Backend.MARLIN,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
                max_shard_size=None,
            )

            del model

            safetensors_file_path = os.path.join(tmp_dir, "gptq_model-4bit-128g.safetensors")
            self.assertTrue(os.path.exists(safetensors_file_path))

            model = GPTQModel.from_quantized(
                tmp_dir,
                device_map="auto",
            )

            tokens = model.generate(**tokenizer("1337", return_tensors="pt").to(model.device), max_new_tokens=20)[0]
            result = tokenizer.decode(tokens)

            print(result)
            self.assertTrue(len(result) > 0)

    def test_save_and_load_unsupports_shard(self):
        model = GPTQModel.from_quantized(
            self.MODEL_ID,
            device_map="auto",
            backend=Backend.BITBLAS,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_quantized(
                tmp_dir,
                max_shard_size="100MB",
            )

            del model

            index_file_path = os.path.join(tmp_dir, "gptq_model-4bit-128g.safetensors.index.json")
            self.assertTrue(os.path.isfile(index_file_path))

            model = GPTQModel.from_quantized(
                tmp_dir,
                device_map="auto",
            )

            tokens = model.generate(**tokenizer("1337", return_tensors="pt").to(model.device), max_new_tokens=20)[0]
            result = tokenizer.decode(tokens)

            print(result)
            self.assertTrue(len(result) > 0)
