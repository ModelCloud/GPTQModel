# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear as TritonV2QuantLinear  # noqa: E402

try:
    from gptqmodel_exllama_kernels import prepare_buffers, set_tuning_params  # noqa: F401
except ImportError as e:
    print(f"[WARNING] Could not load gptqmodel_exllama_kernels: {e}")

from gptqmodel import GPTQModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestsQ4Triton(unittest.TestCase):
    def test_generation_no_act_order(self):
        prompt = "I am in Paris and"

        reference_output = "<s> I am in Paris and I am going to the Louvre Museum. What time does it open and what is the best way to get there?\nThe Louvre Museum in Paris is open from 9:00 AM to 6:00 PM every day except for Tuesdays. The best way to get"
        new_tokens = 60

        model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"

        model_q = GPTQModel.from_quantized(
            model_id,
            device="cuda:0",
            use_triton=True,
            disable_exllama=True,
            disable_exllamav2=True,
            torch_dtype=torch.float16,
        )
        for _, submodule in model_q.named_modules():
            if isinstance(submodule, TritonV2QuantLinear):
                break
        else:
            raise ValueError("Did not find a tritonv2 linear layer")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        # This one uses Autocast.
        res = model_q.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
        predicted_text = tokenizer.decode(res[0])
        self.assertEqual(predicted_text, reference_output)

        # This one does not.
        res = model_q.model.generate(**inp, num_beams=1, min_new_tokens=new_tokens, max_new_tokens=new_tokens)
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
            use_triton=True,
            disable_exllama=True,
            disable_exllamav2=True,
        )
        for _, submodule in model_q.named_modules():
            if isinstance(submodule, TritonV2QuantLinear):
                break
        else:
            raise ValueError("Did not find a tritonv2 linear layer")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text, reference_output)
