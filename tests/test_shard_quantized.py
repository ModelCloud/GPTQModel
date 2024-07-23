import tempfile
import unittest

import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel


class TestShardQuantized(unittest.TestCase):
    def test_shard(self):
        origin_model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(origin_model_id)
        inp = tokenizer(prompt, return_tensors="pt").to(device)

        # Load origin model
        origin_model = GPTQModel.from_quantized(origin_model_id, device="cuda:0", torch_dtype=torch.float16)
        origin_model_res = origin_model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)
        origin_model_predicted_text = tokenizer.decode(origin_model_res[0])

        # Re-shard Model
        with tempfile.TemporaryDirectory() as tmp_dir:
            GPTQModel.shard_quantized(origin_model_id, save_dir=tmp_dir, max_shard_size="300MB")

            del origin_model

            # Load re-sharded model
            shard_model = GPTQModel.from_quantized(tmp_dir, device="cuda:0", torch_dtype=torch.float16)
            shard_model_res = shard_model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)
            shard_model_predicted_text = tokenizer.decode(shard_model_res[0])

        print("origin_model_predicted_text", origin_model_predicted_text)
        print("shard_model_predicted_text", shard_model_predicted_text)

        self.assertEqual(origin_model_predicted_text[:20], shard_model_predicted_text[:20])

    def test_again_save_quantized_model(self):
        origin_model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        model = GPTQModel.from_quantized(origin_model_id, device="cuda:0", torch_dtype=torch.float16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(Exception) as raise_exception:
                model.save_quantized(tmp_dir)

            print("catch exception:", raise_exception.exception)

            self.assertTrue('Saving a quantized model again is not supported' in str(raise_exception.exception))
