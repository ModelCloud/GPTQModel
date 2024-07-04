# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402

from gptqmodel.nn_modules.qlinear.qlinear_bitblas import BitBLASQuantLinear  # noqa: E402

try:
    from gptqmodel_exllama_kernels import prepare_buffers, set_tuning_params  # noqa: F401
except ImportError as e:
    print(f"[WARNING] Could not load gptqmodel_exllama_kernels: {e}")

from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import Backend, GPTQModel  # noqa: E402


class TestQ4BitBLAS(unittest.TestCase):
    def test_generation(self):
        reference_output = "</s>I am in Paris and I am going to be there for a week. I am going to be in the middle of the city and I am going to be in the middle of the city. I am going to be in the middle of the city and I am going to be in the middle of the city. I am"

        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        model_id = "LnL-AI/opt-125M-autoround-lm_head-false-symTrue"

        try:
            model_q = GPTQModel.from_quantized(model_id, device="cuda:0", backend=Backend.BITBLAS)
        except ValueError as e:
            raise e

        has_bitblas = False
        for _, module in model_q.named_modules():
            if isinstance(module, BitBLASQuantLinear):
                has_bitblas = True
                break
        self.assertTrue(has_bitblas)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text, reference_output)

    def test_bias(self):
        # TheBloke/Llama-2-7B-Chat-GPTQ has bias, but they are all zeros, use a checkpoint which really uses bias.
        model_id = "s3nh/starcoderbase-1b-GPTQ"
        try:
            model_q = GPTQModel.from_quantized(model_id, device="cuda:0", backend=Backend.BITBLAS)
        except ValueError as e:
            raise e

        for _, param in model_q.named_parameters():
            self.assertTrue(param.device != torch.device("meta"))

        for _, param in model_q.named_buffers():
            self.assertTrue(param.device != torch.device("meta"))

        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_proj.bias) > 0)
        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_attn.bias) > 0)

        tokenizer = AutoTokenizer.from_pretrained("Xenova/starcoderbase-1b")

        prompt = "Today I am in Paris and"
        inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])
        self.assertTrue(predicted_text.startswith("Today I am in Paris and I am a student of the Master's"))
