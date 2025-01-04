# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import os  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestSharded(unittest.TestCase):
    MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    @classmethod
    def setUpClass(cls):
        cls.prompt = "I am in Paris and"
        cls.device = torch.device("cuda:0")
        cls.reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\n(The stage is now dark, but the audience can see the characters walking around the stage.)\n\n(The stage is now lit up, but the audience can only see the characters' silhouettes.)\n\n("

    def test_save_and_load(self):
        model = GPTQModel.load(
            self.MODEL_ID,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(
                tmp_dir,
                max_shard_size="100MB"
            )

            del model

            index_file_path = os.path.join(tmp_dir, "model.safetensors.index.json")
            self.assertTrue(os.path.exists(index_file_path))

            model = GPTQModel.load(
                tmp_dir,
                device_map="auto",
            )

            inp = tokenizer(self.prompt, return_tensors="pt").to(self.device)

            tokens = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=60, max_new_tokens=60)
            result = tokenizer.decode(tokens[0])

            self.assertEqual(result[:100], self.reference_output[:100])

    def test_save_and_load_no_shard(self):
        model = GPTQModel.load(
            self.MODEL_ID,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(
                tmp_dir,
                max_shard_size=None,
            )

            del model

            safetensors_file_path = os.path.join(tmp_dir, "model.safetensors")
            self.assertTrue(os.path.exists(safetensors_file_path))

            model = GPTQModel.load(
                tmp_dir,
                device_map="auto",
            )

            inp = tokenizer(self.prompt, return_tensors="pt").to(self.device)

            tokens = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=60, max_new_tokens=60)
            result = tokenizer.decode(tokens[0])

            self.assertEqual(result[:100], self.reference_output[:100])
