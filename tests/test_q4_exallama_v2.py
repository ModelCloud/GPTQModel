# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import Backend, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_exllamav2 import QuantLinear  # noqa: E402
from gptqmodel.utils.importer import select_quant_linear  # noqa: E402
from gptqmodel.utils.model import gptqmodel_post_init  # noqa: E402
from gptqmodel.quantization import FORMAT
from test_q4_cuda import get_diff  # noqa: E402
from test_q4_exallama import CUDA_OLD_REFERENCE  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

GENERATE_EVAL_SIZE = 100

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
            backend=Backend.EXLLAMA_V2,
            format=FORMAT.GPTQ,
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

    def test_generation_desc_act_false(self):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        # Reference generated with the cuda-old kernel
        reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\n(The stage is now dark, but the audience can see the characters walking around the stage.)\n\n(The stage is now lit up, but the audience can only see the characters' silhouettes.)\n\n("

        model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

        model_q = GPTQModel.from_quantized(model_id, device="cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])

    def test_generation_desc_act_true(self):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        # Reference generated with the cuda-old kernel
        reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\n(The stage is now dark, but the audience can see the characters walking around the stage.)\n\n(The stage is now lit up, but the audience can see the characters walking around the stage.)\n\n(The"

        model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        revision = "desc_act_true"

        model_q = GPTQModel.from_quantized(
            model_id,
            rivision=revision,
            device="cuda:0",
            backend=Backend.EXLLAMA_V2,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])

    def test_exllama_v2_buffer_size(self):
        # prompt = "I'm in Paris and" * 450
        prompt = "I'm in Paris and" * 500
        device = torch.device("cuda:0")

        model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        revision = "desc_act_true"

        model_q = GPTQModel.from_quantized(
            model_id,
            # revision=revision,
            device="cuda:0",
            backend=Backend.EXLLAMA_V2,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        self.assertTrue(inp["input_ids"].shape[1] > 2048)  # 2048 is the default max_input_length for LLama

        _ = model_q.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)
