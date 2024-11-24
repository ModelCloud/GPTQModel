# -- do not touch
import os

import torch

from gptqmodel import BACKEND
from tests.model_test import ModelTest

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

GENERATE_EVAL_SIZE = 100


class TestsIPEX(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"  # "bigscience/bloom-560m"
    NATIVE_ARC_CHALLENGE_ACC = 0.28
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.31
    TORCH_DTYPE = torch.float16
    LOAD_BACKEND = BACKEND.IPEX
    DELETE_QUANTIZED_MODEL = False

    def test_ipex_format(self):
        self.quant_lm_eval()
