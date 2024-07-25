import unittest
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from parameterized import parameterized
from gptqmodel import GPTQModel,BACKEND

MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

class TestSave(unittest.TestCase):
    @parameterized.expand(
        [
            (BACKEND.AUTO),
            (BACKEND.EXLLAMA_V2),
            (BACKEND.EXLLAMA),
            (BACKEND.TRITON),
            (BACKEND.BITBLAS),
            (BACKEND.MARLIN),
            (BACKEND.QBITS),
        ]
    )
    def test_save(self, backend):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0") if backend != BACKEND.QBITS else torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        inp = tokenizer(prompt, return_tensors="pt").to(device)

        # origin model produce correct output
        origin_model = GPTQModel.from_quantized(MODEL_ID, backend=backend)
        origin_model_res = origin_model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)
        origin_model_predicted_text = tokenizer.decode(origin_model_res[0])

        origin_model.save_quantized("./test_reshard")

        # saved model produce wrong output
        new_model = GPTQModel.from_quantized("./test_reshard", backend=backend)

        new_model_res = new_model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)
        new_model_predicted_text = tokenizer.decode(new_model_res[0])

        print("origin_model_predicted_text",origin_model_predicted_text)
        print("new_model_predicted_text",new_model_predicted_text)

        self.assertEqual(origin_model_predicted_text[:20], new_model_predicted_text[:20])
