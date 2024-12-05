# -- do not touch
import os

from gptqmodel.nn_modules.qlinear import BaseQuantLinear

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import random  # noqa: E402
import unittest  # noqa: E402

import numpy  # noqa: E402
import torch  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestLmHead(unittest.TestCase):
    MODEL_ID = "/monster/data/model/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse" # "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"
    DEVICE = "cuda:0"

    @classmethod
    def setUpClass(cls):
        seed = 898
       # stabilize generation
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def test_load(self):
        prompt = "My name is Lewis and I like to"

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        inputs = tokenizer(prompt, return_tensors="pt").to(device=self.DEVICE)

        model = GPTQModel.load(self.MODEL_ID, device=self.DEVICE)

       # validate lm_head is loaded as quantized layer
        assert isinstance(model.model.lm_head, BaseQuantLinear)
