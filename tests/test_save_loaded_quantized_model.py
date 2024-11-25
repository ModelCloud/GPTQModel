# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import tempfile  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

class TestSave(unittest.TestCase):
    @parameterized.expand(
        [
            (BACKEND.AUTO),
            (BACKEND.EXLLAMA_V2),
            (BACKEND.TRITON),
            (BACKEND.BITBLAS),
            (BACKEND.MARLIN),
            (BACKEND.IPEX),
        ]
    )
    def test_save(self, backend):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0") if backend != BACKEND.IPEX else torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        inp = tokenizer(prompt, return_tensors="pt").to(device)

        # origin model produce correct output
        origin_model = GPTQModel.load(MODEL_ID, backend=backend)
        origin_model_res = origin_model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)
        origin_model_predicted_text = tokenizer.decode(origin_model_res[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            origin_model.save(tmpdir)

            # saved model produce wrong output
            new_model = GPTQModel.load(tmpdir, backend=backend)

            new_model_res = new_model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)
            new_model_predicted_text = tokenizer.decode(new_model_res[0])

            print("origin_model_predicted_text",origin_model_predicted_text)
            print("new_model_predicted_text",new_model_predicted_text)

            self.assertEqual(origin_model_predicted_text[:20], new_model_predicted_text[:20])
