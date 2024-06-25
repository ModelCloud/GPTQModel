# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_exllamav2 import QuantLinear  # noqa: E402
from gptqmodel.utils.importer import select_quant_linear  # noqa: E402
from test_q4_exallama import CUDA_OLD_REFERENCE

try:
    from exllama_kernels import prepare_buffers, set_tuning_params  # noqa: F401
except ImportError as e:
    print(f"[WARNING] Could not load exllama_kernels: {e}")

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.utils.model import gptqmodel_post_init  # noqa: E402
from test_q4_cuda import get_diff
from transformers import AutoTokenizer  # noqa: E402


class TestsQ4ExllamaV2(unittest.TestCase):
    def test_exllamav2(self):
        group_size = 128

        m = 1
        k = 1024
        n = 1024
        device = torch.device("cuda:0")

        linear_class = select_quant_linear(
            bits=4,
            group_size=group_size,
            desc_act=False,
            sym=True,
            use_triton=False,
        )

        linear = linear_class(
            bits=4,
            group_size=group_size,
            desc_act=False,
            sym=True,
            infeatures=k,
            outfeatures=n,
            bias=False,
        )

        self.assertTrue(isinstance(linear, QuantLinear))

        torch.manual_seed(42)

        linear.qweight = torch.randint(-100, 100, size=linear.qweight.shape, dtype=torch.int32)
        linear.scales = linear.scales + 0.002
        linear.qzeros += 0b00010001000100010001000100010001  # for new weight format

        linear = linear.eval()
        linear = linear.to(device)

        linear = gptqmodel_post_init(linear, use_act_order=False)

        inp = torch.rand(1, m, k, dtype=torch.float16).to(device)

        with torch.no_grad():
            res = linear(inp)[0][0]

        reference = CUDA_OLD_REFERENCE.to(device)

        self.assertTrue(
            torch.allclose(res, reference, rtol=3e-5, atol=2e-2),
            get_diff(res, reference),
        )

    def test_generation_no_act_order(self):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        # Reference generated with the cuda-old kernel
        reference_output = "<s> I am in Paris and I am going to the Louvre Museum. What time does it open and what is the best way to get there?\nThe Louvre Museum in Paris is open from 9:00 AM to 6:00 PM every day except for Tuesdays. The best way to get"

        model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"

        model_q = GPTQModel.from_quantized(model_id, device="cuda:0", use_triton=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text, reference_output)

    def test_generation_with_act_order(self):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        # Reference generated with the cuda-old kernel
        reference_output = "<s> I am in Paris and I am so excited to be here. I am here for the first time in my life and I am so grateful for this opportunity. I am here to learn and to grow and to meet new people and to experience new things. I am here to see the Eiffel Tower and to walk along"

        model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"

        model_q = GPTQModel.from_quantized(
            model_id,
            device="cuda:0",
            use_triton=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text, reference_output)

    def test_exllama_v2_buffer_size(self):
        # prompt = "I'm in Paris and" * 450
        prompt = "I'm in Paris and" * 500
        device = torch.device("cuda:0")

        model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"

        model_q = GPTQModel.from_quantized(
            model_id,
            device="cuda:0",
            use_triton=False,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        self.assertTrue(inp["input_ids"].shape[1] > 2048)  # 2048 is the default max_input_length for LLama

        _ = model_q.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)
