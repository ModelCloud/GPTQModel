# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_marlin import MarlinQuantLinear  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestQ4Marlin(unittest.TestCase):
    def test_generation(self):
        reference_output = "<s> I am in Paris and I am feeling very sad and lonely. everybody I know is busy and I don't have any friends here. I am staying in a small apartment in the 11th arrondissement and I am feeling very isolated. I miss my friends and family back home and I don'"

        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"

        try:
            model_q = GPTQModel.from_quantized(model_id, device="cuda:0", backend=BACKEND.MARLIN)
        except ValueError as e:
            raise e

        has_marlin = False
        for _, module in model_q.named_modules():
            if isinstance(module, MarlinQuantLinear):
                has_marlin = True
                break
        self.assertTrue(has_marlin)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text, reference_output)

    def test_bias(self):
        # TheBloke/Llama-2-7B-Chat-GPTQ has bias, but they are all zeros, use a checkpoint which really uses bias.
        model_id = "s3nh/starcoderbase-1b-GPTQ"
        try:
            model_q = GPTQModel.from_quantized(model_id, device="cuda:0", backend=BACKEND.MARLIN)
        except ValueError as e:
            raise e

        for _, param in model_q.named_parameters():
            self.assertNotEqual(param.device, torch.device("meta"))

        for _, param in model_q.named_buffers():
            self.assertNotEqual(param.device, torch.device("meta"))

        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_proj.bias) > 0)
        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_attn.bias) > 0)

        tokenizer = AutoTokenizer.from_pretrained("Xenova/starcoderbase-1b")

        prompt = "Today I am in Paris and"
        inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertIn("Today I am in Paris and I am a student of", predicted_text)
